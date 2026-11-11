# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.config import derive
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module

from torchtitan_npu.models.deepseek_v4.attention import Attention
from torchtitan_npu.models.deepseek_v4.compressor import Compressor
from torchtitan_npu.models.deepseek_v4_1.attention import Attention as Attention41
from torchtitan_npu.models.deepseek_v4_1.compressor import Compressor as Compressor41


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
    def parallelize(self, parallel_dims):
        self.group = parallel_dims.get_optional_mesh("cp")
        super().parallelize(parallel_dims)

    def _exchange_window(self, win_k, metadata):
        local = win_k.squeeze(0)
        received = all_to_all_single(
            local.index_select(0, metadata.send_index),
            output_split_sizes=metadata.recv_split,
            input_split_sizes=metadata.send_split,
            group=self.group,
        )
        # Empty receives must remain in autograd; repeated keys accumulate at their owner.
        available = torch.cat((local, received))
        head = available.index_select(0, metadata.head_index).unsqueeze(0)
        tail = None if metadata.tail_index is None else available.index_select(0, metadata.tail_index).unsqueeze(0)
        return head, tail

    def forward(self, q, swa_k, attn_sink, attention_metadata, *, cmp_k=None, topk_indices=None, topk_scores=None):
        head_k, tail_k = self._exchange_window(swa_k, attention_metadata.xchg.swa)
        if attention_metadata.head is None:
            return self.inner(
                q,
                head_k,
                attn_sink,
                attention_metadata=attention_metadata,
                cmp_k=cmp_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
            )
        width = q.size(1) // 2

        def attend(part, keys, metadata):
            return self.inner(
                q[:, part],
                keys,
                attn_sink,
                attention_metadata=metadata,
                cmp_k=cmp_k,
                topk_indices=_query_half(topk_indices, part),
                topk_scores=_query_half(topk_scores, part),
            )

        head = attend(slice(None, width), head_k, attention_metadata.head)
        tail = attend(slice(width, None), tail_k, attention_metadata.tail)
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


class ContextParallelAttention(ContextParallelAttentionMixin, Attention):
    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        pass


class ContextParallelAttention41(ContextParallelAttentionMixin, Attention41):
    @dataclass(kw_only=True, slots=True)
    class Config(Attention41.Config):
        pass


class ContextParallelCompressorMixin:
    """Shared flat-input restoration for the V4 and V4.1 CP compressors."""

    group = None

    def parallelize(self, parallel_dims):
        self.group = parallel_dims.get_optional_mesh("cp")
        super().parallelize(parallel_dims)  # pyrefly: ignore [missing-attribute]

    def _restore_flat_input(self, x, metadata):
        if metadata is None:
            return x
        if self.group is None:
            raise RuntimeError("restoring a flat compressor input requires a CP process group")
        tokens = x.squeeze(0)
        if metadata.send_index is not None:
            tokens = tokens.index_select(0, metadata.send_index)
        received = all_to_all_single(
            tokens, output_split_sizes=metadata.recv_split, input_split_sizes=metadata.send_split, group=self.group
        )
        return received.index_select(0, metadata.recv_index).unsqueeze(0)


class ContextParallelCompressor(ContextParallelCompressorMixin, Compressor):
    @dataclass(kw_only=True, slots=True)
    class Config(Compressor.Config):
        pass

    def _prepend_carry(self, x, positions):
        """Flat shards need only the predecessor's last compression block."""
        group = self.group
        assert group is not None
        rank, cp_size = group.get_local_rank(), group.size()
        block_size = self.compress_ratio
        send_split = [block_size if peer == rank + 1 else 0 for peer in range(cp_size)]
        recv_split = [block_size if peer == rank - 1 else 0 for peer in range(cp_size)]
        sent = x[:, -block_size:, :].flatten(0, 1).narrow(0, 0, sum(send_split)).contiguous()
        carry = all_to_all_single(
            sent, output_split_sizes=recv_split, input_split_sizes=send_split, group=group
        ).unsqueeze(0)
        # Keep rank 0's empty receive in autograd. Carry outputs are trimmed before use.
        x = torch.cat((carry, x), dim=1)
        # ``positions`` is padded only to keep the stream the same length as ``x``: the carry
        # block is dropped after pooling, so these zeros describe no token.
        positions = torch.cat((positions.new_zeros((positions.size(0), carry.size(1))), positions), dim=1)
        return x, positions, carry.size(1)

    def _prepare_input(self, x, positions, cu_seqlens):
        """The stream this rank pools: its slab, plus the block the ratio-4 pool reaches back for.

        The ratio-4 window needs one block before the slab, and on a rank whose slab starts
        mid-document that block is the previous rank's last one: the ring prepends it, the pool
        runs on the augmented stream, and the extra output block is trimmed afterwards.

        Returns ``(x, positions, cu_seqlens, trim)``, where ``trim`` is how many pooled rows the
        carry leaves to cut off the front.
        """
        group = self.group
        if group is None or group.size() <= 1:
            return x, positions, cu_seqlens, 0
        local_length = x.size(1)
        carry_length = 0
        if self.compress_ratio == 4:
            x, positions, carry_length = self._prepend_carry(x, positions)
        global_start = group.get_local_rank() * local_length - carry_length
        input_length = local_length + carry_length
        inside = cu_seqlens[(cu_seqlens > global_start) & (cu_seqlens < global_start + input_length)] - global_start
        bounds = torch.cat((inside.new_zeros(1), inside, inside.new_full((1,), input_length))).to(torch.int32)
        return x, positions, bounds, carry_length // self.compress_ratio

    def forward(self, x, positions, *, cu_seqlens=None, attention_metadata=None):
        if attention_metadata is None:
            return super().forward(x, positions, cu_seqlens=cu_seqlens)
        x = self._restore_flat_input(x, None if attention_metadata.xchg is None else attention_metadata.xchg.cmp)
        x, positions, cu_seqlens, trim = self._prepare_input(x, positions, cu_seqlens)
        pooled = super().forward(x, positions, cu_seqlens=cu_seqlens)
        return pooled.narrow(1, trim, pooled.size(1) - trim)


class ContextParallelCompressor41(ContextParallelCompressorMixin, Compressor41):
    @dataclass(kw_only=True, slots=True)
    class Config(Compressor41.Config):
        pass

    def forward(self, x_BLD, positions_BL, cmp_k=None, *, attention_metadata=None):
        if self.is_source and attention_metadata is not None:
            x_BLD = self._restore_flat_input(
                x_BLD, None if attention_metadata.xchg is None else attention_metadata.xchg.cmp
            )
        return super().forward(x_BLD, positions_BL, cmp_k, attention_metadata=attention_metadata)


@cache
def _cp_compressor_type(compressor_type, cp_type):
    if issubclass(cp_type, compressor_type):
        return cp_type

    # Keep the selected constructor/forward, including TorchAO's quantization wrapper.
    class CPCompressor(cp_type, compressor_type):
        @dataclass(kw_only=True, slots=True)
        class Config(compressor_type.Config):
            pass

    return CPCompressor


class ContextParallelConverter(ModelConfigConverter):
    """CP conversion for the pinned ModelConfigConverter API."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config):
        pass

    def convert(self, model_config):
        for layer in [*model_config.layers, *(getattr(model_config, "mtp_layers", None) or [])]:
            attention = layer.attention
            if isinstance(attention, Attention41.Config) and not isinstance(
                attention, ContextParallelAttention41.Config
            ):
                layer.attention = derive(
                    attention,
                    ContextParallelAttention41.Config,
                    compressor=derive(
                        attention.compressor,
                        _cp_compressor_type(type(attention.compressor)._owner, ContextParallelCompressor41).Config,
                    ),
                )
            elif isinstance(attention, Attention.Config) and not isinstance(attention, ContextParallelAttention.Config):
                layer.attention = derive(attention, ContextParallelAttention.Config)
                for owner in (layer.attention, layer.attention.indexer):
                    if owner is not None and owner.compressor is not None:
                        compressor = owner.compressor
                        owner.compressor = derive(
                            compressor,
                            _cp_compressor_type(type(compressor)._owner, ContextParallelCompressor).Config,
                        )
            # Only the CP consumers redistribute these keys to the whole row, so the converter
            # owns the override and copies the map: the sharding config shares it with its source
            # placements.
            row = dense_activation_placement(tp=spmd.R, cp=spmd.R)
            attention_sharding = layer.attention.inner_attention.sharding_config
            attention_sharding.in_dst_shardings = {
                **(attention_sharding.in_dst_shardings or {}),
                "cmp_k": row,
            }
            indexer = layer.attention.indexer
            if indexer is not None and indexer.inner_indexer.sharding_config is not None:
                index_sharding = indexer.inner_indexer.sharding_config
                index_sharding.in_dst_shardings = {**(index_sharding.in_dst_shardings or {}), "idx_k_BNDi": row}
        return model_config
