# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve the repository-owned CI layers without importing the NPU runtime."""

import json
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[4] / ".github" / "ci" / "suites.json"


def case_ids(suite: str, *, manifest: Path = MANIFEST) -> list[str]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("suites"), dict):
        raise ValueError("Unsupported CI suite manifest")
    definitions = document["suites"]

    def expand(name: str, stack: tuple[str, ...]) -> list[str]:
        if name in stack:
            raise ValueError(f"Cyclic CI suite inheritance: {(*stack, name)}")
        if name not in definitions:
            raise ValueError(f"Unknown CI suite: {name}")
        spec = definitions[name]
        if not isinstance(spec, dict) or set(spec) != {"extends", "cases"}:
            raise ValueError(f"Invalid CI suite: {name}")
        for field in ("extends", "cases"):
            if not isinstance(spec[field], list) or not all(isinstance(value, str) for value in spec[field]):
                raise ValueError(f"Invalid {field} in CI suite: {name}")
        values = []
        for parent in spec["extends"]:
            values.extend(expand(parent, (*stack, name)))
        values.extend(spec["cases"])
        # A case inherited through several parents still runs exactly once.
        return list(dict.fromkeys(values))

    result = expand(suite, ())
    if not result:
        raise ValueError(f"Empty CI suite: {suite}")
    return result
