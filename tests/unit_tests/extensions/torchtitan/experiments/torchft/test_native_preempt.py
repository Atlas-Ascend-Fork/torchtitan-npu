# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This source code is licensed under the BSD-style license found in LICENSE.

import importlib.util
import io
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call
from urllib.error import HTTPError, URLError

import pytest

_ROOT = Path(__file__).resolve().parents[6]
_LAUNCHER = _ROOT / "torchtitan_npu/extensions/torchtitan/experiments/torchft/preempt.py"
_SCHEDULER = _ROOT / "tests/integration_tests/torchft_preemption.py"


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preempt = load_script(_LAUNCHER)
scheduler = load_script(_SCHEDULER)


@pytest.mark.parametrize(
    "missing", [("TT_PREEMPT_LOCAL",), ("TT_PREEMPT_SLOT",), ("TT_PREEMPT_LOCAL", "TT_PREEMPT_SLOT")]
)
def test_preemption_requires_launcher_environment(tmp_path, monkeypatch, missing):
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.ft_manager import FTManagerEx

    monkeypatch.setenv("TT_PREEMPT_LOCAL", str(tmp_path))
    monkeypatch.setenv("TT_PREEMPT_SLOT", "torchtitan_ft_0")
    for name in missing:
        monkeypatch.delenv(name)
    config = FTManagerEx.Config(enable_preemption=True, replica_id=0, group_size=2, min_replica_size=1)
    with pytest.raises(ValueError, match=r"preempt\.py launch"):
        FTManagerEx(config)


def test_preemption_registers_worker_for_launcher(tmp_path, monkeypatch):
    from torchtitan_npu.extensions.torchtitan.experiments.torchft import ft_manager

    monkeypatch.setenv("TT_PREEMPT_LOCAL", str(tmp_path))
    monkeypatch.setenv("TT_PREEMPT_SLOT", "torchtitan_ft_1")
    monkeypatch.setenv("RANK", "3")
    instance = "torchtitan_ft_1:restart"
    manager = Mock()
    manager._store.get.return_value = instance.encode()
    monkeypatch.setattr(ft_manager, "ManagerEx", Mock(return_value=manager))
    monkeypatch.setattr(ft_manager, "ProcessGroupHCCLEx", Mock())
    monkeypatch.setattr(ft_manager.torchft.process_group, "ManagedProcessGroup", Mock())

    config = ft_manager.FTManagerEx.Config(enable_preemption=True, replica_id=1, group_size=2, min_replica_size=1)
    ft_manager.FTManagerEx(config)
    assert json.loads((tmp_path / "rank-3").read_text()) == [instance, os.getpid()]


def test_stop_local_leaves_another_launch_running(tmp_path):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"],
        env=dict(os.environ, TT_PREEMPT_LOCAL=str(tmp_path / "other")),
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"],
        env=dict(os.environ, TT_PREEMPT_LOCAL=str(tmp_path)),
    )
    try:
        preempt.stop_local(child, tmp_path)
        assert child.poll() is not None
        assert other.poll() is None
    finally:
        for process in (child, other):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def test_unplanned_exit_does_not_release_cards(tmp_path, monkeypatch):
    calls = []

    def status(action, **kwargs):
        calls.append(action)
        return {"lease": None}

    monkeypatch.setattr(preempt, "send_request", status)
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    child.wait(timeout=10)
    with pytest.raises(RuntimeError, match="Unplanned torchrun exit 3"):
        preempt.supervise(child, tmp_path, "torchtitan_ft_0", "node-a")
    assert calls == ["status"]


def test_supervise_keeps_training_through_status_disconnect(tmp_path, monkeypatch):
    status = Mock(side_effect=[URLError("connection reset"), {"lease": None}])
    child = Mock(returncode=0)
    # The child stays alive throughout the outage and exits after connectivity returns.
    child.poll.side_effect = lambda: 0 if status.call_count >= 2 else None
    stop = Mock()
    monkeypatch.setattr(preempt, "send_request", status)
    monkeypatch.setattr(preempt, "stop_local", stop)
    monkeypatch.setattr(preempt.time, "sleep", lambda _: None)

    assert preempt.supervise(child, tmp_path, "torchtitan_ft_0", "node-a") is False
    assert status.call_args_list == [call("status"), call("status")]
    stop.assert_not_called()


def test_supervise_retries_release_for_registered_instance(tmp_path, monkeypatch):
    slot = "torchtitan_ft_1"
    instance = slot + ":original"
    identity = {"request": "inference-1", "slot": slot, "instance": instance, "node": "node-b"}
    (tmp_path / "rank-0").write_text(json.dumps([instance, 123]))
    (tmp_path / "rank-0.tmp").write_text("[")
    lease = {
        "request": identity["request"],
        "phase": "draining",
        "targets": {slot: instance},
        "manager_exits": [slot],
        "receipts": [],
    }
    child = Mock()
    child.poll.return_value = None
    events = []
    attempts = []
    polls = Mock(side_effect=[{"lease": lease}] * 4)

    def control(action, **params):
        if action == "status":
            return polls()
        assert action == "released"
        events.append(action)
        attempts.append(params)
        if len(attempts) == 1:
            raise URLError("temporarily unavailable")
        return {"lease": {**lease, "phase": "released", "receipts": [[slot, "node-b"]]}}

    monkeypatch.setattr(preempt, "send_request", control)
    monkeypatch.setattr(preempt, "stop_local", lambda child, directory: events.append("stopped"))
    monkeypatch.setattr(preempt.time, "sleep", lambda _: None)

    assert preempt.supervise(child, tmp_path, slot, "node-b") is True
    assert events[0] == "stopped"
    assert attempts == [identity, identity]


def test_supervise_ignores_drain_for_an_older_instance(tmp_path, monkeypatch):
    slot = "torchtitan_ft_1"
    (tmp_path / "rank-0").write_text(json.dumps([slot + ":new", os.getpid()]))
    draining = {
        "request": "previous-request",
        "phase": "draining",
        "targets": {slot: slot + ":old"},
        "manager_exits": [slot],
        "receipts": [],
    }
    status = Mock(side_effect=[{"lease": draining}, {"lease": None}])
    child = Mock(returncode=0)
    child.poll.side_effect = lambda: 0 if status.call_count >= 2 else None
    stop = Mock()
    monkeypatch.setattr(preempt, "send_request", status)
    monkeypatch.setattr(preempt, "stop_local", stop)
    monkeypatch.setattr(preempt.time, "sleep", lambda _: None)

    assert preempt.supervise(child, tmp_path, slot, "node-b") is False
    assert status.call_args_list == [call("status"), call("status")]
    stop.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="The node launcher uses POSIX locks and /proc on Linux.")
def test_launcher_restart_finishes_drain_before_starting_new_instance(tmp_path, monkeypatch):
    slot = "torchtitan_ft_1"
    node = "node-b"
    instance = slot + ":old"
    prior = tmp_path / f"{slot}-{node}" / "run-old"
    prior.mkdir(parents=True)
    (prior / "rank-0").write_text(json.dumps([instance, 123]))
    lease = {"request": "job", "phase": "drain_timed_out", "targets": {slot: instance}, "manager_exits": [slot]}
    status = {"layout": {slot: {node: [2]}}, "lease": lease}
    child = Mock(returncode=0)
    child.poll.return_value = 0
    cleanup = Mock()
    popen = Mock(return_value=child)
    polls = Mock(side_effect=[status] * 6)
    receipts = []

    def control(action, **params):
        if action == "released":
            cleanup.assert_called_once_with(None, prior)
            popen.assert_not_called()
            receipts.append(params)
            lease["phase"] = "joining"  # Simulate the scheduler returning the devices.
        else:
            assert action == "status"
            return polls()
        return status

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(preempt, "send_request", control)
    monkeypatch.setattr(preempt, "stop_local", cleanup)
    monkeypatch.setattr(preempt.time, "sleep", lambda _: None)
    monkeypatch.setattr(preempt.subprocess, "Popen", popen)

    preempt.launch(Namespace(replica=1, node=node, state_dir=tmp_path, command=["--", "torchrun", "train.py"]))

    assert receipts == [{"request": "job", "slot": slot, "instance": instance, "node": node}]
    popen.assert_called_once()
    assert popen.call_args.args[0] == ["torchrun", "train.py"]
    assert Path(popen.call_args.kwargs["env"]["TT_PREEMPT_LOCAL"]) != prior
    assert not prior.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="The node launcher uses POSIX locks and /proc on Linux.")
def test_launcher_keeps_child_running_across_status_outage(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TORCHFT_LIGHTHOUSE", "http://lighthouse")
    monkeypatch.setenv("TORCHFT_PREEMPT_TOKEN", "test")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    attempts = 0
    resumed = tmp_path / "resumed"
    completed = tmp_path / "completed"

    def urlopen(req, timeout):
        nonlocal attempts
        assert json.loads(req.data) == {"action": "status"}
        attempts += 1
        if attempts == 1:
            return io.BytesIO(b'{"layout":{"torchtitan_ft_0":{"node-a":[0]}}}')
        # The first two reads precede the launch. Interrupt the supervisor polls.
        if attempts in (3, 4):
            raise URLError(ConnectionRefusedError("refused"))
        if attempts == 5:
            resumed.touch()
        return io.BytesIO(b'{"lease":null}')

    monkeypatch.setattr(preempt, "urlopen", urlopen)
    child = """
import sys, time
from pathlib import Path
deadline = time.monotonic() + 10
while not Path(sys.argv[1]).exists():
    if time.monotonic() > deadline:
        sys.exit(1)
    time.sleep(0.05)
Path(sys.argv[2]).touch()
"""
    args = SimpleNamespace(
        replica=0,
        node="node-a",
        state_dir=tmp_path,
        command=["--", sys.executable, "-c", child, str(resumed), str(completed)],
    )
    preempt.launch(args)
    assert completed.exists()
    logs = capsys.readouterr().err
    assert logs.count("Lighthouse status unreachable") == 1
    assert logs.count("Lighthouse status recovered") == 1


def test_status_outage_expires_with_last_error(monkeypatch):
    monkeypatch.setenv("TORCHFT_LIGHTHOUSE", "http://lighthouse")
    monkeypatch.setattr(scheduler, "_STATUS_RETRY_SECONDS", 2.5)
    now = 0.0
    timeouts = []

    def sleep(seconds):
        nonlocal now
        now += seconds

    def refused(action, *, timeout):
        timeouts.append(timeout)
        raise URLError(ConnectionRefusedError("still down"))

    monkeypatch.setattr(
        scheduler, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep, strftime=lambda _: "now")
    )
    monkeypatch.setattr(scheduler, "send_request", refused)
    with pytest.raises(TimeoutError, match=r"unavailable for 2\.5s; last error: .*still down"):
        scheduler.control("status")
    assert timeouts == [2.5, 1.5, 0.5]


@pytest.mark.parametrize("action", ["status", "suspend", "resume", "released", "manager_exited"])
def test_http_errors_and_failed_writes_are_reported_immediately(action, monkeypatch):
    monkeypatch.setenv("TORCHFT_LIGHTHOUSE", "http://lighthouse")
    monkeypatch.setenv("TORCHFT_PREEMPT_TOKEN", "test")
    calls = []

    def fail(req, timeout):
        calls.append(json.loads(req.data)["action"])
        if action == "status":
            raise HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"invalid token"))
        raise URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(preempt, "urlopen", fail)
    monkeypatch.setattr(scheduler, "send_request", preempt.send_request)
    with pytest.raises((RuntimeError, URLError), match=r"401 invalid token|refused"):
        scheduler.control(action)
    assert calls == [action]


@pytest.mark.parametrize(
    ("args", "payload"),
    [
        (["status"], {"action": "status"}),
        (
            ["suspend", "--request", "job", "--cards", "4", "--node", "node-b"],
            {"action": "suspend", "request": "job", "cards": 4, "node": "node-b"},
        ),
        (["resume", "--request", "job"], {"action": "resume", "request": "job"}),
        (["retry", "--request", "job"], {"action": "retry", "request": "job"}),
        (["cancel", "--request", "job"], {"action": "cancel", "request": "job"}),
    ],
)
def test_scheduler_cli_sends_one_lighthouse_command(args, payload, monkeypatch, capsys):
    send = Mock(return_value={"ok": True})
    monkeypatch.setattr(scheduler, "send_request", send)
    monkeypatch.setattr(sys, "argv", [str(_SCHEDULER), *args])

    scheduler.main()

    action = payload["action"]
    send.assert_called_once_with(action, **{k: v for k, v in payload.items() if k != "action"})
    assert json.loads(capsys.readouterr().out) == {"ok": True}
