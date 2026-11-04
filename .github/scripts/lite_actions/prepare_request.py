#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trusted CPU-only workflow -> Matrix and source-SHA-bound Artifact.

GitHub prepares one immutable, SHA-bound case plan. The dispatcher owns training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}\Z")
# Only the upstream organization may publish CI plans or consume results.
REPO = "torchtitan-npu/torchtitan-npu"
if (source := os.environ.get("GITHUB_REPOSITORY", REPO)) != REPO:
    raise ValueError("untrusted GitHub CI repository")
PIPELINES = {"base", "pr", "nightly"}


def select_suite(event: str, requested: str) -> str:
    """Automatic events fix the suite; manual runs choose a declared layer."""
    if event == "pull_request":
        return "pr"
    if event == "schedule":
        return "nightly"
    if event == "workflow_dispatch" and requested in PIPELINES:
        return requested
    raise ValueError("Unsupported training event or suite")


def expand_cases(suite: str) -> list[dict]:
    """Expand the selected layer using definitions in the current checkout."""
    from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select

    catalogue = catalog()
    expanded, seen = [], set()
    for test in select(catalogue, suite=suite):
        if test.test_name in seen:
            raise ValueError("duplicate expanded Test Case")
        if not isinstance(test.ngpu, int) or not isinstance(test.nnodes, int):
            raise ValueError("invalid resource requirements")
        if test.ngpu < 1 or test.nnodes < 1:
            raise ValueError("invalid resource requirements")
        label = test.test_descr.strip()
        if not label or len(label) > 115 or any(ord(x) < 32 for x in label):
            raise ValueError("invalid Test Case description")
        item = {"test_id": test.test_name, "nnodes": test.nnodes, "ngpu": test.ngpu, "label": label}
        if test.device is not None:
            item.update(device=test.device, required_env=list(test.required_env), phases=len(test.override_args))
        expanded.append(item)
        seen.add(test.test_name)
    return expanded


def digest_plan(plan: dict) -> str:
    canon = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def create_plan(*, cases: list[dict], run_id: int, attempt: int, sha: str, repo: str, pipeline: str) -> dict:
    if repo != REPO or pipeline not in PIPELINES:
        raise ValueError("untrusted repository/pipeline")
    if run_id < 1 or attempt < 1 or not SHA.fullmatch(sha):
        raise ValueError("invalid GitHub identity")
    plan = {
        "schema": 2,
        "repo": repo,
        "run_id": run_id,
        "attempt": attempt,
        "sha": sha,
        "pipeline": pipeline,
        "cases": cases,
    }
    return {**plan, "plan_digest": digest_plan(plan)}


def main() -> None:
    pipeline = select_suite(os.environ["GITHUB_EVENT_NAME"], os.environ.get("CI_SUITE", "base"))
    request_dir = Path(".ci-request")
    request_dir.mkdir(exist_ok=True)
    cases = expand_cases(pipeline)
    # GitHub Actions supports up to 256 Matrix combinations; never truncate.
    if not cases or len(cases) > 256:
        raise ValueError("GitHub Matrix supports 1..256 test cases per workflow run")
    result = create_plan(
        cases=cases,
        run_id=int(os.environ["GITHUB_RUN_ID"]),
        attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        sha=os.environ.get("CI_SOURCE_SHA") or os.environ["GITHUB_SHA"],
        repo=os.environ["GITHUB_REPOSITORY"],
        pipeline=pipeline,
    )
    matrix = {"include": [{"test_id": case["test_id"], "label": case["label"]} for case in cases]}
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        raise ValueError("GITHUB_OUTPUT required for Matrix")
    with open(output, "a", encoding="utf-8") as handle:
        handle.write("matrix=" + json.dumps(matrix, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.write("plan_digest=" + result["plan_digest"] + "\n")
        handle.write("pipeline=" + pipeline + "\n")
    print("Prepared CI Matrix:", [case["test_id"] for case in cases])
    (request_dir / "ci-request.json").write_text(
        json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
