# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the Qwen3.5 CP varlen-attention refusal."""

import pytest
import torch

# The whole module exercises the not-yet-migrated context-parallel path.
pytestmark = pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
)

_CP_REFUSAL = "not migrated to the folded token layout"


def test_cp_varlen_attention_factory_refuses_unmigrated_path():
    from torchtitan_npu.override.qwen3_5 import varlen_attention

    with pytest.raises(NotImplementedError, match=_CP_REFUSAL):
        varlen_attention.asc_cp(None)


def test_cp_varlen_attention_forward_refuses_unmigrated_path():
    from torchtitan_npu.override.qwen3_5 import varlen_attention

    with pytest.raises(NotImplementedError, match=_CP_REFUSAL):
        varlen_attention.AscVarlenAttention.forward(
            object(),
            torch.zeros(4, 2, 2),
            torch.zeros(4, 2, 2),
            torch.zeros(4, 2, 2),
            attention_masks=None,
        )
