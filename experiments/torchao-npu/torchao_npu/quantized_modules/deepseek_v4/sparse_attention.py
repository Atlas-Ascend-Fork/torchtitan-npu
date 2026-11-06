# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 fused attention with a quantized selector.

The quantized stack is a backend of its own: it extends the fused config and builds its own
metadata (the selector's grid is a different operator), while the forward is the fused
one, inherited unchanged -- quantizing the selector does not change a single launch of the
attention.  That is the difference from DeepSeek-V4.1's quantized backend, which also
fake-quantizes its sliding-window input and therefore copies the forward.
"""

from __future__ import annotations

__all__ = ["QuantizedV4SparseAttention"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
    AscV4Metadata,
)

from .attention_metadata import QuantV4Metadata


class QuantizedV4SparseAttention(AscCompressedSparseInnerAttention):
    """The fused attention whose selector grid is built by the quantized selector."""

    @dataclass(kw_only=True, slots=True)
    class Config(AscCompressedSparseInnerAttention.Config):
        quant_mode: int
        """The selector's quantization ABI (``quant_mode`` of the CANN operator pair)."""

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV4Metadata:
            """This backend's per-microbatch metadata: the model's, plus this backend's builder.

            The frames are the model's; the builder is this backend's, which builds every operator
            output the fused builder builds and hands the selector's grid to its own operator.
            """
            del padding_mask, max_num_documents, max_context_length
            metadata = DeepSeekV4Metadata.from_positions(positions, self.compress_ratio)
            return QuantV4Metadata.build_aicpu_metadata(metadata, geometry=self.geometry, quant_mode=self.quant_mode)
