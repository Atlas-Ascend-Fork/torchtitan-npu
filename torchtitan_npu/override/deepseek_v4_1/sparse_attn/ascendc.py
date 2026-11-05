# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 SMLA attention and SMLAG-backed indexer distillation."""

from dataclasses import dataclass, replace

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata
from torchtitan_npu.override.deepseek_v4.asc_metadata import (
    AicpuGeometry,
    AscLightningIndexerMetadata,
    AscV4Metadata,
)
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscSparseMLAMixin


@dataclass(frozen=True, kw_only=True, slots=True)
class AscV41LightningIndexerMetadata(AscLightningIndexerMetadata):
    """This stack's indexer metadata: the BF16 pair, plus the candidate pair it grows into."""

    fwd_sparse: torch.Tensor | None = None
    sparse_length: torch.Tensor | None = None


class AscV41Metadata(AscV4Metadata, DeepSeekV41Metadata):  # pyrefly: ignore [inconsistent-inheritance]
    """The V4.1 entry: V4's, with this stack's indexer ratios and metadata."""

    indexer_ratios = (1, 2)

    @classmethod
    def build_aicpu_metadata(cls, metadata, *, geometry):
        entry = super().build_aicpu_metadata(metadata, geometry=geometry)
        metadata = entry.aicpu_metadata
        return replace(
            entry,
            aicpu_metadata=replace(
                metadata,
                lightning_indexer=AscV41LightningIndexerMetadata(
                    fwd=metadata.lightning_indexer.fwd, bwd=metadata.lightning_indexer.bwd
                ),
            ),
        )


if AscV41LightningIndexerMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscV41LightningIndexerMetadata,
        serialized_type_name=f"{AscV41LightningIndexerMetadata.__module__}.{AscV41LightningIndexerMetadata.__name__}",
    )

if AscV41Metadata not in SUPPORTED_NODES:
    register_dataclass(AscV41Metadata, serialized_type_name=f"{AscV41Metadata.__module__}.{AscV41Metadata.__name__}")


class AscV41SparseAttention(AscSparseMLAMixin, CompressedSparseInnerAttention2):
    """The A5 TND kernel behind the reference forward contract.

    ``topk_scores`` is not a loss input: it is the channel the two kernels' backwards talk on,
    SMLAG's returning the teacher and the indexer's SLIKG backward consuming it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention2.Config):
        """The fused config: the reference core's fields plus the geometry its AI_CPU metadata need."""

        window_size: int
        compress_ratio: int
        num_heads: int
        head_dim: int
        index_topk: int
        index_num_heads: int
        index_head_dim: int

        @property
        def geometry(self) -> AicpuGeometry:
            """The numbers the AI_CPU operators are called with, read off this layer's config."""
            return AicpuGeometry(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=self.window_size,
                index_topk=self.index_topk,
                index_num_heads=self.index_num_heads,
                index_head_dim=self.index_head_dim,
            )

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the port's class plus its compression ratio."""
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV41Metadata:
            """This geometry's frames and AI_CPU metadata, once per microbatch (upstream #4897)."""
            # ``slots=True`` recreates this dataclass, so a zero-argument ``super()`` would bind
            # to the decorated-away class and raise: the parent builder is named explicitly.
            row = CompressedSparseInnerAttention2.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            return AscV41Metadata.build_aicpu_metadata(row, geometry=self.geometry)
