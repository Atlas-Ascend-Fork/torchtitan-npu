# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from torchtitan.config import derive
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module


def _query_half(tensor, part):
    return None if tensor is None else tensor[:, part]


class _CPKernel(Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        # Gather once around both calls to the already-built backend.
        self._sharding_config, inner._sharding_config = inner._sharding_config, None
        self._attention_metadata_key = inner.attention_metadata_key


class CPAttention(_CPKernel):
    def forward(self, q, swa_k, attn_sink, attention_metadata, *, cmp_k=None, topk_indices=None, topk_scores=None):
        if attention_metadata.head is None:
            return self.inner(
                q,
                swa_k,
                attn_sink,
                attention_metadata=attention_metadata,
                cmp_k=cmp_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
            )
        width = q.size(1) // 2

        def attend(part, metadata):
            return self.inner(
                q[:, part],
                swa_k,
                attn_sink,
                attention_metadata=metadata,
                cmp_k=cmp_k,
                topk_indices=_query_half(topk_indices, part),
                topk_scores=_query_half(topk_scores, part),
            )

        head = attend(slice(None, width), attention_metadata.head)
        tail = attend(slice(width, None), attention_metadata.tail)
        return torch.cat((head, tail), dim=1)


class CPIndexer(_CPKernel):
    def forward(self, idx_q_BLHiDi, idx_k_BNDi, idx_w_BLHi, attention_metadata, **kwargs):
        if attention_metadata.head is None:
            return self.inner(idx_q_BLHiDi, idx_k_BNDi, idx_w_BLHi, attention_metadata=attention_metadata, **kwargs)
        width = idx_q_BLHiDi.size(1) // 2

        def select(part, metadata):
            candidates = dict(kwargs)
            if "candidates_BL1C" in candidates:
                candidates["candidates_BL1C"] = _query_half(candidates["candidates_BL1C"], part)
            return self.inner(
                idx_q_BLHiDi[:, part],
                idx_k_BNDi,
                idx_w_BLHi[:, part],
                attention_metadata=metadata,
                **candidates,
            )

        head = select(slice(None, width), attention_metadata.head)
        tail = select(slice(width, None), attention_metadata.tail)
        # V4.1 Reindex preserves the inherited candidate pool.
        candidates = kwargs.get("candidates_BL1C")
        return tuple(
            None if h is None else candidates if i == 2 and candidates is not None else torch.cat((h, t), dim=1)
            for i, (h, t) in enumerate(zip(head, tail, strict=True))
        )


class ContextParallelAttentionMixin:
    inner_attention: Module
    indexer: Any

    def __init__(self, config):
        super().__init__(config)  # pyrefly: ignore [bad-argument-count]
        self.inner_attention = CPAttention(self.inner_attention)
        if self.indexer is not None:
            self.indexer.inner_indexer = CPIndexer(self.indexer.inner_indexer)


class ContextParallelConverter(ModelConfigConverter):
    """CP conversion for the pinned ModelConfigConverter API."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config):
        pass

    def convert(self, model_config):
        from ..deepseek_v4_1.attention import Attention as Attention41
        from ..deepseek_v4_1.attention import ContextParallelAttention as ContextParallelAttention41
        from .attention import Attention, ContextParallelAttention

        for layer in [*model_config.layers, *(getattr(model_config, "mtp_layers", None) or [])]:
            attention = layer.attention
            if isinstance(attention, (ContextParallelAttention.Config, ContextParallelAttention41.Config)):
                continue
            if isinstance(attention, Attention41.Config):
                layer.attention = derive(attention, ContextParallelAttention41.Config)
            elif isinstance(attention, Attention.Config):
                layer.attention = derive(attention, ContextParallelAttention.Config)
        return model_config
