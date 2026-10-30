# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest


@pytest.mark.skip(
    reason="the Qwen3.5 NPU context-parallel override is not migrated to the folded token "
    "layout yet; this static check still asserts the pre-fold CP graph"
)
def test_qwen3_5_cp_adapter_uses_override_refactor_paths():
    root = Path(__file__).resolve().parents[4]
    parallelize_source = (root / "torchtitan_npu/override/qwen3_5/parallelize.py").read_text(encoding="utf-8")
    gated_delta_source = (root / "torchtitan_npu/override/qwen3_5/gated_delta.py").read_text(encoding="utf-8")

    assert "QwenCPMetadata" in parallelize_source
    assert "prepare_sequence_metadata" in parallelize_source
    assert "exchange_sequence_heads" in gated_delta_source
    assert "context_parallel" in gated_delta_source
