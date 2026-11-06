# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The quantized selector's AI_CPU metadata for the DeepSeek-V4 fused attention.

Only one operator output depends on precision: the selector's forward grid.  The sliding-window
pair and SLIG's backward metadata are the same operators with the same arguments in both paths,
so this module buys exactly one extra operator call (``quant_lightning_indexer_metadata``) on top
of the fused backend's builder, and replaces that one field of the metadata it produced.

The operator lives in ``cann_ops_transformer`` -- the same package as the BF16 metadata operator,
with the same frames and one extra ``quant_mode`` integer -- so the fused backend never needs to
know that a quantized selector exists: this backend states the ABI on its own config and builds
its own metadata.
"""

from __future__ import annotations

__all__ = [
    "CMP_RATIO",
    "LAYOUT",
    "MASK_MODE",
    "QuantV4Metadata",
    "kernel_aicpu_metadata",
]

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.override.deepseek_v4.asc_metadata import AicpuGeometry, AscV4Metadata

if TYPE_CHECKING:
    from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata

CMP_RATIO = 4
"""The ratio V4's indexer runs at."""
LAYOUT = "TND"
MASK_MODE = 3


def kernel_aicpu_metadata(
    metadata: AscV4Metadata,
    *,
    geometry: AicpuGeometry,
    quant_mode: int,
) -> torch.Tensor:
    """The quantized selector's forward grid for these frames."""
    cmp_k = metadata.cmp_k
    if cmp_k is None or cmp_k.residual is None:
        raise ValueError("the quantized selector needs a pooling geometry's compressed frame.")
    return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        geometry.index_topk,
        quant_mode,
        layout_q=LAYOUT,
        layout_k=LAYOUT,
        mask_mode=MASK_MODE,
        cmp_ratio=CMP_RATIO,
        cu_seqlens_q=metadata.q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        seqused_q=metadata.q.seqused,
        seqused_k=cmp_k.seqused,
        cmp_residual_k=cmp_k.residual,
    )


@dataclass(kw_only=True, slots=True)
class QuantV4Metadata(AscV4Metadata):
    """The quantized backend's metadata.

    Same interface as the fused one (``aicpu_metadata.sparse_attn`` /
    ``aicpu_metadata.lightning_indexer``), because the forward it feeds is the fused forward.
    ``quant_mode`` is the selector's ABI, kept so the metadata can rebuild its own AI_CPU
    metadata for another set of frames.
    """

    quant_mode: int

    @classmethod
    def build_aicpu_metadata(  # pyrefly: ignore [bad-override]
        # This backend's builder states one number more than the fused one: the selector
        # operator's ``quant_mode``.
        cls,
        metadata: DeepSeekV4Metadata,
        *,
        geometry: AicpuGeometry,
        quant_mode: int,
    ) -> AscV4Metadata:
        """The fused metadata, with the selector's grid built by this backend's operator.

        The other operator outputs are the fused builder's: they are the same operators either
        path launches.  A geometry that runs no selector keeps the fused metadata untouched,
        since it has no selector grid to replace.
        """
        asc = AscV4Metadata.build_aicpu_metadata(metadata, geometry=geometry)
        if asc.ratio != CMP_RATIO:
            return asc
        fwd = kernel_aicpu_metadata(asc, geometry=geometry, quant_mode=quant_mode)
        torch._dynamo.maybe_mark_dynamic(fwd, 0)
        # GraphTrainer's tracer reads this annotation for input dimensions.
        setattr(fwd, "_dynamo_unbacked_indices", {0})  # noqa: B010
        return cls(
            ratio=asc.ratio,
            position_k=asc.position_k,
            q=asc.q,
            swa_k=asc.swa_k,
            cmp_k=asc.cmp_k,
            xchg=asc.xchg,
            aicpu_metadata=replace(
                asc.aicpu_metadata,
                lightning_indexer=replace(asc.aicpu_metadata.lightning_indexer, fwd=fwd),
            ),
            a3_workaround=asc.a3_workaround,
            quant_mode=quant_mode,
        )

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        attention_metadata: QuantV4Metadata,
        *,
        permutation: torch.Tensor | None,
    ) -> QuantV4Metadata:
        """This rank's frames and the AI_CPU metadata rebuilt for them.

        ``rebuild_cp_metadata`` re-cuts the frames; this backend's builder then rebuilds every
        operator output, the selector's grid with its own operator.
        """
        return attention_metadata.rebuild_cp_metadata(
            permutation,
            build=partial(
                QuantV4Metadata.build_aicpu_metadata,
                geometry=attention_metadata.aicpu_metadata.geometry,
                quant_mode=attention_metadata.quant_mode,
            ),
        )


if QuantV4Metadata not in SUPPORTED_NODES:
    register_dataclass(QuantV4Metadata, serialized_type_name=f"{QuantV4Metadata.__module__}.{QuantV4Metadata.__name__}")
