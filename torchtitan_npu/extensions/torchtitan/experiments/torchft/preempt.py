# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Node launcher for native Lighthouse planned preemption."""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REQUEST_TIMEOUT = 10.0


def send_request(action, *, timeout=_REQUEST_TIMEOUT, **params):
    """Send one authenticated Lighthouse request. The caller owns retries."""
    url = os.environ["TORCHFT_LIGHTHOUSE"].rstrip("/") + "/preemption"
    headers = {"Authorization": "Bearer " + os.environ["TORCHFT_PREEMPT_TOKEN"], "Content-Type": "application/json"}
    req = Request(url, data=json.dumps({"action": action, **params}).encode(), headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code in (408, 429, 500, 502, 503, 504):
            raise ConnectionError(f"Lighthouse {action}: HTTP {error.code}") from error
        raise RuntimeError(f"Lighthouse {action}: {error.code} {error.read().decode()}") from error


def launcher_control(action, *, child=None, **params):
    delay = 0.25
    failures = 0
    while True:
        try:
            result = send_request(action, **params)
            if failures:
                print(f"Lighthouse {action} recovered", file=sys.stderr, flush=True)
            return result
        except (URLError, ConnectionError, TimeoutError, IncompleteRead) as error:
            failures += 1
            if failures == 1 or failures % 12 == 0:
                print(f"Lighthouse {action} unreachable (attempt {failures}): {error}", file=sys.stderr, flush=True)
            # Keep a live training group running during control-plane outages.
            if child is not None and child.poll() is not None and failures >= 6:
                raise RuntimeError("torchrun exited while Lighthouse status was unavailable") from error
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


def worker_directory(replica_id: int) -> Path:
    directory = os.environ.get("TT_PREEMPT_LOCAL")
    slot = os.environ.get("TT_PREEMPT_SLOT")
    if not directory or not slot:
        raise ValueError(
            "fault_tolerance.enable_preemption requires the node launcher. "
            "Start torchrun with 'preempt.py launch --replica ID --state-dir DIR -- torchrun ...' "
            "to set TT_PREEMPT_LOCAL and TT_PREEMPT_SLOT."
        )
    if slot != f"torchtitan_ft_{replica_id}":
        raise ValueError("Launcher and trainer replica IDs differ")
    return Path(directory)


def record_worker(manager, directory: Path):
    """Register the native instance once, before its first quorum request."""
    from torchft.manager import REPLICA_ID_KEY

    instance = manager._store.get(REPLICA_ID_KEY).decode()
    path = directory / f"rank-{int(os.environ['RANK'])}"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps([instance, os.getpid()]))
    tmp.replace(path)


def live_workers(directory):
    import psutil

    # torchrun workers can have their own sessions. Match this invocation's
    # inherited marker, including data-loader children, before signalling.
    for proc in psutil.process_iter():
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            if proc.environ().get("TT_PREEMPT_LOCAL") == str(directory):
                yield proc


def stop_local(child, directory):
    import psutil

    for sig, seconds in [(signal.SIGTERM, 20), (signal.SIGKILL, 10)]:
        deadline = time.monotonic() + seconds
        while True:
            if child is not None:
                child.poll()
            found = False
            for proc in live_workers(directory):
                found = True
                with suppress(psutil.NoSuchProcess):
                    proc.send_signal(sig)
            if not found and (child is None or child.poll() is not None):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
    raise RuntimeError(f"Training processes still alive for {directory}")


def drain_local(lease, child, directory, slot, node):
    import psutil

    target = lease["targets"][slot]
    records = {}
    for path in directory.glob("rank-*"):
        if path.name.removeprefix("rank-").isdecimal():
            records[path.name] = json.loads(path.read_text())
    if not records or any(instance != target for instance, _ in records.values()):
        return None
    identity = dict(request=lease["request"], slot=slot, instance=target)
    if slot in lease["manager_exits"]:
        stop_local(child, directory)
        launcher_control("released", **identity, node=node)
        print(f"Released {slot} on {node}: {target}", flush=True)
        return True
    if "rank-0" in records:
        pid = records["rank-0"][1]
        try:
            exited = psutil.Process(pid).environ().get("TT_PREEMPT_LOCAL") != str(directory)
        except psutil.NoSuchProcess:
            exited = True
        except psutil.AccessDenied:
            exited = False
        if exited:
            launcher_control("manager_exited", **identity)
    return False


def supervise(child, directory, slot, node):
    while True:
        lease = launcher_control("status", child=child)["lease"]
        draining = lease and slot in lease["targets"] and lease["phase"] in ("draining", "drain_timed_out")
        handled = drain_local(lease, child, directory, slot, node) if draining else None
        if handled:
            return True
        if handled is None and child.poll() is not None:
            if child.returncode:
                raise RuntimeError(f"Unplanned torchrun exit {child.returncode} for {slot}")
            return False
        time.sleep(0.25)


def launch(args):
    import fcntl

    import psutil  # noqa: F401 - check before starting torchrun

    slot = f"torchtitan_ft_{args.replica}"
    node = args.node
    state = launcher_control("status")
    expected = state["layout"][slot][node]
    cards = [int(x) for x in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
    if expected != cards:
        raise ValueError(f"Launcher devices {cards} differ from layout {expected}")
    if not args.command or args.command[0] != "--" or len(args.command) < 2:
        raise ValueError("Specify -- followed by the torchrun command")
    root = Path(args.state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{slot}-{node}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = root / f"{slot}-{node}"
        attempts.mkdir(exist_ok=True)
        while True:
            lease = launcher_control("status")["lease"]
            phase = lease["phase"] if lease and slot in lease["targets"] else None
            # A launcher may have died after Kill or before its receipt was delivered.
            # Reuse the original invocation markers under the same slot/node lock.
            for prior in attempts.glob("run-*"):
                if phase in ("draining", "drain_timed_out"):
                    if drain_local(lease, None, prior, slot, node) is not True:
                        continue
                elif any(live_workers(prior)):
                    raise RuntimeError(f"Previous training processes still active for {prior}")
                shutil.rmtree(prior)
            if phase in ("preparing", "draining", "drain_timed_out", "released"):
                time.sleep(0.5)
                continue
            directory = Path(tempfile.mkdtemp(dir=attempts, prefix="run-"))
            env = dict(os.environ, TT_PREEMPT_LOCAL=str(directory), TT_PREEMPT_SLOT=slot)
            child = subprocess.Popen(args.command[1:], env=env, start_new_session=True)
            try:
                released = supervise(child, directory, slot, node)
            finally:
                stop_local(child, directory)
            shutil.rmtree(directory)
            if not released:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    worker = sub.add_parser("launch")
    worker.add_argument("--replica", type=int, required=True)
    worker.add_argument("--node", default=socket.gethostname())
    worker.add_argument("--state-dir", required=True)
    worker.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    launch(args)


if __name__ == "__main__":
    main()
