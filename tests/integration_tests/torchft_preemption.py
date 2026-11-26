# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Simulate a scheduler against a running Lighthouse from the training checkout."""

import argparse
import errno
import json
import os
import runpy
import secrets
import socket
import subprocess
import sys
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import URLError

# Load the standalone transport without importing the NPU training package on the control host.
_LAUNCHER = Path(__file__).resolve().parents[2] / "torchtitan_npu/extensions/torchtitan/experiments/torchft/preempt.py"
send_request = runpy.run_path(str(_LAUNCHER))["send_request"]
_STATUS_RETRY_SECONDS = 30.0


def control(action, *, retry_status=True, **params):
    """Retry read-only polling across a short Lighthouse transport outage."""
    if action != "status" or not retry_status:
        return send_request(action, **params)

    started = time.monotonic()
    deadline = started + _STATUS_RETRY_SECONDS
    last_error = None
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            state = send_request(action, timeout=min(10.0, remaining), **params)
        except (OSError, IncompleteRead) as error:
            cause = error.reason if isinstance(error, URLError) else error
            if not isinstance(cause, (ConnectionError, TimeoutError, IncompleteRead)) and getattr(
                cause, "errno", None
            ) not in (errno.ENETUNREACH, errno.EHOSTUNREACH, socket.EAI_AGAIN):
                raise
            if last_error is None:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} Lighthouse status unreachable "
                    f"at {os.environ['TORCHFT_LIGHTHOUSE']} ({error}); retrying for up to {_STATUS_RETRY_SECONDS}s",
                    file=sys.stderr,
                    flush=True,
                )
            last_error = error
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        else:
            if last_error is not None:
                elapsed = time.monotonic() - started
                print(f"Lighthouse status recovered after {elapsed:.1f}s", file=sys.stderr, flush=True)
            return state
    raise TimeoutError(
        f"Lighthouse status at {os.environ['TORCHFT_LIGHTHOUSE']} unavailable for "
        f"{time.monotonic() - started:.1f}s; last error: {last_error}"
    ) from last_error


def probe_devices(devices, timeout):
    """Use a separate process so every NPU context is closed before resume."""
    program = """
import sys
import torch
import torch_npu

count = int(sys.argv[1])
if torch.npu.device_count() != count:
    raise RuntimeError(f"Expected {count} released NPUs")
for i in range(count):
    if torch.arange(16, device=f"npu:{i}").sum().item() != 120:
        raise RuntimeError(f"Incorrect sum on released device npu:{i}")
"""
    subprocess.run(
        [sys.executable, "-c", program, str(len(devices))],
        env={**os.environ, "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, devices))},
        timeout=timeout,
        check=True,
    )


def verify_cycle(node, cards, after_step, timeout, probe):
    """Exercise two replicas, each placed on a single node; both may share a host."""
    layout = control("status")["layout"]
    if len(layout) != 2 or any(len(nodes) != 1 for nodes in layout.values()):
        raise ValueError(f"Verification needs two single-node replicas, got {layout}")
    placements = [nodes[node] for nodes in layout.values() if len(nodes.get(node, [])) >= cards]
    if not placements:
        raise ValueError(f"No single replica can release {cards} cards on {node}")
    if probe:
        if node != socket.gethostname():
            raise ValueError("Run --probe on the requested node using its hostname")
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if visible and not set().union(*placements) <= set(map(int, visible.split(","))):
            raise ValueError("Probe may need devices outside ASCEND_RT_VISIBLE_DEVICES")

    request = "verify-" + secrets.token_hex(6)

    def wait(label, predicate, *, own_lease=True):
        deadline = time.monotonic() + timeout
        while True:
            state = control("status")
            lease, quorum = state.get("lease") or {}, state.get("quorum") or {}
            if own_lease and (
                lease.get("request") != request or lease["phase"] in ("cancelled", "drain_timed_out", "join_timed_out")
            ):
                raise RuntimeError(f"{label}: unexpected lease {lease}")
            if predicate(lease, quorum):
                print(f"{label}: request={request} phase={lease.get('phase')} quorum={quorum}", flush=True)
                return quorum
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label}: lease={lease}, quorum={quorum}")
            time.sleep(0.5)

    original = wait(
        "both training",
        lambda _, q: len(q) == 2 and {k.rsplit(":", 1)[0] for k in q} == set(layout) and min(q.values()) >= after_step,
        own_lease=False,
    )
    lease = control("suspend", request=request, cards=cards, node=node)["lease"]
    ((slot, old),) = lease["targets"].items()
    if old not in original or old.rsplit(":", 1)[0] != slot or layout[slot].get(node) not in placements:
        raise RuntimeError(f"Unexpected target: {lease}")
    (survivor,) = original.keys() - {old}
    released = wait(
        "released",
        lambda lease, q: (
            lease["phase"] == "released"
            and slot in lease["manager_exits"]
            and [slot, node] in lease["receipts"]
            and set(q) == {survivor}
        ),
    )
    wait("survivor advanced", lambda _, q: set(q) == {survivor} and q[survivor] >= released[survivor] + 2)
    if probe:
        devices = layout[slot][node]
        probe_devices(devices, timeout)
        print(f"Independent process verified {node} devices {devices}", flush=True)
    control("resume", request=request)
    joined = wait(
        "new instance joined",
        lambda lease, q: (
            lease["phase"] == "restored"
            and len(q) == 2
            and survivor in q
            and old not in q
            and any(k.rsplit(":", 1)[0] == slot for k in q)
        ),
    )
    wait("both advanced", lambda _, q: q.keys() == joined.keys() and min(q.values()) > max(joined.values()))
    print(f"PASS: suspend/resume request={request}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    suspend = sub.add_parser("suspend")
    suspend.add_argument("--request", required=True)
    suspend.add_argument("--cards", type=int, required=True)
    suspend.add_argument("--node")
    for action in ("resume", "retry", "cancel"):
        sub.add_parser(action).add_argument("--request", required=True)
    verify = sub.add_parser("verify", help="exercise suspend and resume on a running two-replica job")
    verify.add_argument("--node", default=socket.gethostname())
    verify.add_argument("--cards", type=int, required=True)
    verify.add_argument("--after-step", type=int, default=3)
    verify.add_argument("--timeout", type=int, default=900, help="timeout per phase, in seconds")
    verify.add_argument("--probe", action="store_true", help="compute on each released NPU before resume")
    args = parser.parse_args()
    if args.action == "verify":
        if args.cards < 1 or args.timeout < 1 or args.after_step < 0:
            parser.error("cards and timeout must be positive and after-step nonnegative")
        verify_cycle(args.node, args.cards, args.after_step, args.timeout, args.probe)
    else:
        print(json.dumps(control(**vars(args), retry_status=False), indent=2))


if __name__ == "__main__":
    main()
