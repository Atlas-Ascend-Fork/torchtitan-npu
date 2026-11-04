# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared execution adapter for single-node and multi-node training cases.

TorchTitan's run_tests()/GPUPool remains the single-node test runner.
Cross-node orchestration stays in Lite Actions; only the per-node recipe and
TensorBoard assertions live here, never SSH/resource scheduling.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from tests.integration_tests import OverrideDefinitions


def dispatcher_port_env() -> dict[str, str]:
    """Apply dispatcher-issued rendezvous ports after model recipe defaults."""
    overrides = {}
    for key, target in (("LITE_CI_MASTER_PORT", "MASTER_PORT"), ("LITE_CI_HCCL_IF_BASE_PORT", "HCCL_IF_BASE_PORT")):
        raw = os.environ.get(key)
        if raw is not None:
            if not raw.isascii() or not raw.isdecimal() or not 1024 <= int(raw) <= 65535:
                raise ValueError(f"invalid dispatcher-issued {key}")
            overrides[target] = raw
    for key, target in (
        ("LITE_CI_HCCL_HOST_PORT_RANGE", "HCCL_HOST_SOCKET_PORT_RANGE"),
        ("LITE_CI_HCCL_NPU_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE"),
    ):
        raw = os.environ.get(key)
        if raw is not None:
            match = __import__("re").fullmatch(r"([1-9][0-9]{3,4})-([1-9][0-9]{3,4})", raw)
            if not match or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
                raise ValueError(f"invalid dispatcher-issued {key}")
            overrides[target] = raw
    return overrides


def prepare_launch(test: OverrideDefinitions, output_dir: Path) -> tuple[OverrideDefinitions, Path]:
    """Use local logs and case-owned checkpoints, shared across distributed ranks."""
    env = {**os.environ, **(test.env_vars or {})}
    for name in test.required_env:
        if not env.get(name) or not Path(env[name]).exists():
            raise ValueError(f"{test.test_name}: missing site asset {name}")
    if test.expected_steps is not None and len(test.expected_steps) != len(test.override_args):
        raise ValueError("Expected one step sequence per training phase")
    if test.check_resume and (len(test.override_args) != 2 or test.expected_steps is None):
        raise ValueError("Resume verification requires two phases with expected steps")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty")
    run_root = output_dir.resolve() / test.test_name / "test_run"
    case_env = {
        **(test.env_vars or {}),
        **dispatcher_port_env(),
        "NGPU": str(test.ngpu),
        "LOG_RANK": ",".join(map(str, range(test.ngpu))),
    }
    if test.device is not None:
        if test.nnodes > 1:
            shared = env.get("LITE_CI_SHARED_CHECKPOINT_DIR", "")
            if not shared or not Path(shared).is_absolute() or not Path(shared).is_dir():
                raise ValueError("LITE_CI_SHARED_CHECKPOINT_DIR must be an existing absolute shared directory")
            # The dispatcher allocates one fresh shared directory per case/run.
            # Checking emptiness on every node would race with rank 0 writes.
            case_env["CKPT_SAVE_LOAD_PATH"] = shared
        else:
            case_env["CKPT_SAVE_LOAD_PATH"] = str(run_root / "checkpoint")
    return replace(test, env_vars=case_env), run_root


def record_completion(test: OverrideDefinitions, output_dir: Path) -> None:
    (output_dir / "launch-complete.json").write_text(
        json.dumps(
            {
                "test_id": test.test_name,
                "nnodes": test.nnodes,
                "ngpu": test.ngpu,
                "phases": len(test.override_args),
                "exit_code": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def verify(test: OverrideDefinitions, *, output_dir: Path) -> None:
    """Verify local launch completion and the logging rank's phase metrics.

    The dispatcher must also require successful launch on every remote node.
    Finite metrics are completion checks, not a numerical/performance baseline.
    """
    from tests.integration_tests.loss_compare import (
        compare_checkpoint_metrics,
        extract_scalar_from_tensorboard,
        TB_LOSS_TAG,
    )

    completed = json.loads((output_dir / "launch-complete.json").read_text(encoding="utf-8"))
    expected = {
        "test_id": test.test_name,
        "nnodes": test.nnodes,
        "ngpu": test.ngpu,
        "phases": len(test.override_args),
        "exit_code": 0,
    }
    if completed != expected:
        raise ValueError("Launch completion identity does not match the selected case")
    if not test.expected_steps:
        raise ValueError("CI verification requires explicit expected steps")
    run_root = output_dir.resolve() / test.test_name / "test_run"
    tags = (TB_LOSS_TAG, "grad_norm") if test.check_grad_norm else (TB_LOSS_TAG,)
    for index, steps in enumerate(test.expected_steps):
        for tag in tags:
            values = extract_scalar_from_tensorboard(run_root, f"tb_phase_{index}", tag)
            if set(values) != set(steps):
                raise RuntimeError(
                    f"{test.test_name}: phase {index} {tag} expected {list(steps)}, got {sorted(values)}"
                )
    if test.check_resume:
        compare_checkpoint_metrics(run_root, test.expected_steps)
    print(
        f"[CI_VERIFY] completion PASS test={test.test_name}; external accuracy/performance baselines not checked",
        flush=True,
    )


def run_single(test: OverrideDefinitions, *, output_dir: Path | None = None) -> None:
    from tests.integration_tests.run_tests import _resolve_npu_pool_ids, run_tests

    if output_dir is None:
        parser = argparse.ArgumentParser(description=test.test_descr)
        parser.add_argument("output_dir", type=Path)
        output_dir = parser.parse_args().output_dir
    if len(_resolve_npu_pool_ids(test.ngpu)) < test.ngpu:
        raise ValueError(f"{test.test_name}: insufficient NPU allocation")
    test, _ = prepare_launch(test, output_dir)
    run_tests(
        argparse.Namespace(output_dir=str(output_dir), ngpu=test.ngpu, test_name=test.test_name, exclude=None),
        [test],
        parallel=True,
    )
    record_completion(test, output_dir)
    verify(test, output_dir=output_dir)


def run_distributed(test: OverrideDefinitions, *, phase: str, output_dir: Path) -> None:
    if phase == "verify":
        verify(test, output_dir=output_dir)
        return
    if phase != "launch":
        raise ValueError("phase must be launch or verify")
    ips = [x.strip() for x in os.environ.get("NODE_IPS", "").split(",")]
    if len(ips) != test.nnodes or any(not x for x in ips) or len(set(ips)) != len(ips):
        raise ValueError(f"NODE_IPS requires exactly {test.nnodes} distinct nonempty addresses")
    for key, expected in (("NGPU", test.ngpu), ("NNODES", test.nnodes)):
        if os.environ.get(key, str(expected)) != str(expected):
            raise ValueError(f"{key} must be {expected}")
    test, run_root = prepare_launch(test, output_dir)
    from tests.integration_tests.run_tests import _run_cmd, DETERMINISTIC_ARGS

    env = {key: value for key, value in os.environ.items() if key not in test.unset_env}
    env.update({**(test.env_vars or {}), "NNODES": str(test.nnodes)})
    for index, overrides in enumerate(test.override_args):
        command = ["bash", test.train_script, "--dump_folder", str(run_root), *(test.train_args or ())]
        if test.check_resume or test.check_loss:
            command.extend(DETERMINISTIC_ARGS)
        command.extend(overrides)
        command.append(f"--metrics.save_tb_folder=tb_phase_{index}")
        result = _run_cmd(command, env=env, timeout=test.timeout, prefix=f"{test.test_name}:{index}")
        if result.returncode != 0:
            raise RuntimeError(f"{test.test_name}: phase {index} failed with rc={result.returncode}")
    record_completion(test, output_dir)
