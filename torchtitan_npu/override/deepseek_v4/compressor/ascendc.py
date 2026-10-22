# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN compressor using the operator's native autograd implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import torch

from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.ops.ascendc.compressor import load_compressor


def build_compressor_state_tables(metadata: DeepSeekV4Metadata) -> None:
    """The fused compressor's state tables, tight and read-only, built eagerly.

    A table's rows and width follow the batch's documents, so it cannot be built
    while tracing (the graph would reuse the traced shape): it is filled at the
    eager metadata boundary and read as a graph input.  Two shapes have no table:
    a metadata carrying a CP window plan (the rank-local segments come from the
    exchange, and that path constructs its own inputs) and a ratio-1 geometry (it
    pools nothing).

    Registered by this package's ``asc`` override, so selecting the fused
    compressor is what selects the prebuild.
    """
    if metadata.window is not None:
        return
    cu_seqlens = metadata.varlen.cu_seq_q
    # The table follows *these* boundaries, not a possibly stale host maximum.  The host tuple
    # must have been materialized by the metadata builder (``include_host_offsets=True``): this
    # function is part of the eager build stage and must never sync a device tensor itself.
    host = getattr(metadata.varlen, "cu_seq_q_host", None)
    if host is None:
        raise ValueError(
            "the compressor's state tables need the host boundaries the metadata builder "
            "materializes; build the varlen with include_host_offsets=True."
        )
    boundaries = list(host)
    n_seqs = len(boundaries) - 1
    max_length = max((end - start for start, end in pairwise(boundaries)), default=0)
    ratio, plan = metadata.ratio, metadata.plan
    if ratio != 1 and plan.exchange is None:
        block_size = 8 if ratio == 4 else 16
        width = max((max_length + block_size - 1) // block_size, 1)
        table = torch.zeros((n_seqs, width), dtype=torch.int32, device=cu_seqlens.device)
        for dim in (0, 1):
            torch._dynamo.maybe_mark_dynamic(table, dim)
        # GraphTrainer's tracer reads this annotation for input dimensions.
        setattr(table, "_dynamo_unbacked_indices", {0, 1})  # noqa: B010
        plan.compressor_state_block_table = table


def _state_inputs(
    cu_seqlens: torch.Tensor,
    x: torch.Tensor,
    *,
    max_length: int,
    ratio: int,
    head_dim: int,
    state_block_table: torch.Tensor | None = None,
):
    cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
    # Two slices, not ``torch.diff``: the document count is an unbacked symbol here and
    # ``diff``'s length check would guard on it, which the FX tracer cannot evaluate.
    seqused = cu_seqlens[1:] - cu_seqlens[:-1]
    block_size = 8 if ratio == 4 else 16
    coff = 2 if ratio == 4 else 1
    # CANN tiling requires state_block_table despite its optional schema.
    # In cache_mode=1, block ID 0 skips cache writes for full-document training.
    if state_block_table is None:
        # CP and metadata providers without prebuilt tables retain their shape bound.
        state_block_table = torch.zeros(
            (seqused.numel(), torch.sym_max((max_length + block_size - 1) // block_size, 1)),
            dtype=torch.int32,
            device=x.device,
        )
    else:
        torch._check(state_block_table.shape[0] == seqused.shape[0])
    state_cache = torch.zeros((1, block_size, 2 * coff * head_dim), dtype=torch.float32, device=x.device)
    return state_block_table, state_cache, cu_seqlens, seqused


class AscCompressor(CompressorImplementation):
    @dataclass(kw_only=True, slots=True)
    class Config(CompressorImplementation.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._compressor_fn = load_compressor()

    def forward(self, compressor: Compressor, x: torch.Tensor, attention_metadata: Any) -> torch.Tensor:
        plan = attention_metadata.plan
        if plan.gather_indices is None or plan.block_positions is None:
            raise ValueError("CANN compressor requires complete compression metadata")
        if plan.exchange is not None:
            # Exchange only missing boundary tokens before the fused projections.
            # Plan segments include the C4 predecessor block; restarting each
            # segment masks its first overlap exactly as the reference does.
            max_length = torch.sym_min(plan.gather_indices.numel(), x.shape[1] + 2 * compressor.compress_ratio)
            x = compressor.token_dispatcher.gather(x, plan)
            if plan.is_empty_host:
                # CompressorGrad rejects empty inputs. The collective must still
                # run, including on ranks with no local compressed blocks.
                local_metadata = DeepSeekV4Metadata(
                    varlen=attention_metadata.varlen,
                    ratio=compressor.compress_ratio,
                    plan=replace(plan, exchange=None),
                    window=attention_metadata.window,
                    window_size=attention_metadata.window_size,
                )
                return compressor._forward(x, local_metadata)
            assert plan.first_indices is not None
            cu_seqlens = torch.cat(
                (
                    plan.first_indices * compressor.compress_ratio,
                    plan.first_indices.new_full((1,), plan.gather_indices.numel()),
                )
            )
        else:
            if plan.is_empty_host:
                return compressor._forward(x, attention_metadata)
            cu_seqlens = attention_metadata.varlen.cu_seq_q
            # Only used by the compatibility path without a prebuilt table.
            max_length = attention_metadata.varlen.max_k

        state_block_table, state_cache, cu_seqlens, seqused = _state_inputs(
            cu_seqlens,
            x,
            max_length=max_length,
            ratio=compressor.compress_ratio,
            head_dim=compressor.head_dim,
            state_block_table=plan.compressor_state_block_table,
        )
        pooled = self._compressor_fn(
            x.reshape(-1, x.shape[-1]).contiguous(),
            compressor.wkv.weight,
            compressor.wgate.weight,
            state_cache,
            # CANN also allocates saved softmax/KV outputs with APE's dtype.
            compressor.ape.float(),
            cmp_ratio=compressor.compress_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            coff=1 + int(compressor.overlap),
            cache_mode=1,
        )
        pooled = compressor.norm(pooled[: plan.gather_indices.numel() // compressor.compress_ratio].to(x.dtype))
        return (
            compressor.rope(pooled.unsqueeze(0).unsqueeze(2), positions=plan.block_positions.unsqueeze(0))
            .squeeze(0)
            .squeeze(1)
        )
