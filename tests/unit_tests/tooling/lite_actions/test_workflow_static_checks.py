#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep the Training workflow's Lite Actions selectors aligned with the case catalog."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.integration_tests.tools.lite_actions import entrypoint

pytestmark = pytest.mark.tooling

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def load_workflow(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name}: workflow root must be a mapping"
    return document


def event_section(workflow: dict):
    """Return the ``on:`` mapping.

    PyYAML parses the bare key ``on`` as the boolean ``True``; accept either
    spelling so the check does not depend on that quoting accident.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key]
    return None


def test_workflows_are_parseable_yaml():
    files = workflow_files()
    assert files, "no workflows discovered under .github/workflows"
    for path in files:
        load_workflow(path)


def test_every_workflow_declares_permissions():
    for path in workflow_files():
        assert "permissions" in load_workflow(path), (
            f"{path.name}: missing top-level 'permissions'")


def test_manual_suite_choices_resolve_to_registered_cases():
    catalogue = entrypoint.catalog()
    for path in workflow_files():
        event = event_section(load_workflow(path)) or {}
        dispatch = event.get("workflow_dispatch") or {}
        suite = dispatch.get("inputs", {}).get("suite")
        if suite is None:
            continue
        assert suite["type"] == "choice"
        assert suite["default"] in suite["options"]
        for name in suite["options"]:
            assert entrypoint.select(catalogue, suite=name)
