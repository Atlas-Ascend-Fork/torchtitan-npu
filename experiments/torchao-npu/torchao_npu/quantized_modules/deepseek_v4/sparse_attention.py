# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 fused attention with a quantized selector.

The quantized stack is a backend of its own: it extends the fused config and builds its own
metadata entry (the selector's record is a different operator), while the forward is the fused
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

from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
    AscV4Metadata,
)

from .attention_metadata import attach_selector_record


class QuantizedV4SparseAttention(AscCompressedSparseInnerAttention):
    """The fused attention whose selector record is built by the quantized selector."""

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
            """The fused entry, with the selector's record rebuilt by this backend.

            The parent builder is named explicitly: ``slots=True`` recreates this dataclass, so a
            zero-argument ``super()`` would bind to the decorated-away class.
            """
            metadata = AscCompressedSparseInnerAttention.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            return attach_selector_record(metadata, geometry=self.geometry, quant_mode=self.quant_mode)
