# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=huawei-invalid-name

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import override
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata

from torchtitan_npu.override.qwen3_5.parallelize import (
    QWEN_CONTEXT_PARALLEL_UNSUPPORTED,
)


class AscVarlenAttention(VarlenAttention):
    """Refuse the not-yet-migrated context-parallel varlen attention.

    Upstream 0.4.0 folds the batch dimension into the token row: the base
    ``VarlenAttention.forward`` now takes ``[T, N, H]`` and a
    ``VarlenMetadata``.  The removed implementation exchanged rank-local heads
    on pre-fold ``[B, L, N, H]`` tensors and fed CANN TND attention with
    ``actual_seq_qlen`` from the old CP metadata, so running it against the
    folded caller would mis-shard the sequence.  The class and factory remain
    registered to fail with one explicit explanation until the CP NPU
    validation pass migrates them.
    """

    context_parallel_mesh: DeviceMesh

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        out_transform: (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(QWEN_CONTEXT_PARALLEL_UNSUPPORTED)


@override(
    target=VarlenAttention.Config,
    fqns=["model_spec.model.layers.*.attention.inner_attention"],
    exact=True,
    description="Use CANN TND attention with Qwen3.5/3.6 Context Parallel",
)
def asc_cp(cfg: VarlenAttention.Config) -> AscVarlenAttention.Config:
    raise NotImplementedError(QWEN_CONTEXT_PARALLEL_UNSUPPORTED)
