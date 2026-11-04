#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lite Actions adapter: trusted case definitions and fixed SHA selectors."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re

from tests.integration_tests.tools.lite_actions.suites import case_ids
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.training_cases import build_test_list

SAFE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")


def catalog() -> dict[str, OverrideDefinitions]:
    found = {}
    for test in build_test_list():
        if not SAFE.fullmatch(test.test_name) or test.test_name in found or test.disabled:
            raise ValueError("duplicate/invalid/disabled CI case")
        if not isinstance(test.nnodes, int) or test.nnodes < 1 or not isinstance(test.ngpu, int) or test.ngpu < 1:
            raise ValueError("invalid topology")
        found[test.test_name] = test
    if not found:
        raise ValueError("empty test catalog")
    return found


def select(catalogue: dict, *, test_id: str | None = None, suite: str | None = None):
    if (test_id is None) == (suite is None):
        raise ValueError("select exactly one ID or suite")
    if suite is not None:
        return [select(catalogue, test_id=name)[0] for name in case_ids(suite)]
    if test_id not in catalogue:
        raise ValueError("unknown CI test")
    return [catalogue[test_id]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inspect", "launch", "verify"))
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--test-id")
    sel.add_argument("--suite")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    if a.phase != "inspect" and (a.test_id is None or a.output_dir is None):
        p.error("launch/verify require an explicit test ID and output directory")
    try:
        selected = select(catalog(), test_id=a.test_id, suite=a.suite)
    except ValueError as exc:
        p.error(str(exc))
    if a.phase == "inspect":
        print(
            json.dumps(
                [
                    {
                        "test_id": t.test_name,
                        "ngpu": t.ngpu,
                        "nnodes": t.nnodes,
                        "env_vars": dict(t.env_vars or {}),
                        "device": t.device,
                        "required_env": list(t.required_env),
                        "phases": len(t.override_args),
                    }
                    for t in selected
                ],
                separators=(",", ":"),
            )
        )
        return
    test = selected[0]
    # Also support direct, manual invocation of the fixed model-owned entrypoint.
    # Remote SSH has already sourced the case's toolkit before Python starts.
    os.environ.update(test.env_vars or {})
    from tests.integration_tests.tools.lite_actions.runner import run_single, run_distributed, verify

    if test.nnodes == 1:
        if a.phase == "launch":
            run_single(test, output_dir=a.output_dir)
        else:
            verify(test, output_dir=a.output_dir)
    else:
        run_distributed(test, phase=a.phase, output_dir=a.output_dir)


if __name__ == "__main__":
    main()
