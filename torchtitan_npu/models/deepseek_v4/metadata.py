# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document boundaries, kernel frames and batch-local CP routing; no communication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4.cp_layout import DocumentRanges


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """Boundaries and visible prefixes of one kernel operand."""

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


def _xchg_metadata(token_locations, requested_tokens, *, rank):
    """Route each consumer's requested global tokens from their physical owners.

    token_locations[token] = owner_rank * local_length + owner_local_index.
    Requests are unique and ordered; receives concatenate them in source-rank order.
    """
    local_length = token_locations.numel() // len(requested_tokens)
    owners = token_locations // local_length
    outgoing = [tokens[owners[tokens] == rank] for tokens in requested_tokens]
    wanted = requested_tokens[rank]
    incoming = [wanted[owners[wanted] == peer] for peer in range(len(requested_tokens))]
    return (
        token_locations[torch.cat(outgoing)] % local_length,
        [tokens.numel() for tokens in outgoing],
        [tokens.numel() for tokens in incoming],
        torch.cat(incoming),
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class FlattenMetadata:
    """Restore the compressor's contiguous global token interval."""

    send_index: torch.Tensor | None
    send_split: list[int]
    recv_split: list[int]
    recv_index: torch.Tensor

    @classmethod
    def build(cls, permutation, *, cp_size, cp_rank):
        token_locations = torch.argsort(permutation.flatten())
        local_length = permutation.numel() // cp_size
        requested = list(torch.arange(permutation.numel(), device=permutation.device).split(local_length))
        send_index, send_split, recv_split, received_tokens = _xchg_metadata(token_locations, requested, rank=cp_rank)
        if torch.equal(send_index, torch.arange(local_length, device=permutation.device)):
            send_index = None
        return cls(
            send_index=send_index,
            send_split=send_split,
            recv_split=recv_split,
            recv_index=torch.argsort(received_tokens),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class WindowMetadata:
    """Fetch missing window tokens; head/tail index [local keys, received keys]."""

    send_index: torch.Tensor
    send_split: list[int]
    recv_split: list[int]
    head_index: torch.Tensor
    tail_index: torch.Tensor | None

    @classmethod
    def build(cls, permutation, cu_seqlens, *, head: DocumentRanges, tail: DocumentRanges | None, window_size, cp_rank):
        cp_size = head.start.size(0)
        local_length = permutation.numel() // cp_size
        token_locations = torch.argsort(permutation.flatten())
        head_window = head.window(cu_seqlens, window_size)
        head_prefix = DocumentRanges(head_window.start, head.start)
        tail_window, tail_prefix = None, None
        if tail is not None:
            tail_window = tail.window(cu_seqlens, window_size)
            tail_prefix = DocumentRanges(tail_window.start, tail.start)
        requested = []
        for peer in range(cp_size):
            tokens = head_prefix.token_ids(peer)
            if tail_prefix is not None:
                tokens = torch.cat((tokens, tail_prefix.token_ids(peer)))
            tokens = tokens.unique(sorted=True)
            requested.append(tokens[token_locations[tokens] // local_length != peer])
        send_index, send_split, recv_split, received_tokens = _xchg_metadata(token_locations, requested, rank=cp_rank)
        token_to_buffer = token_locations % local_length
        token_to_buffer[received_tokens] = (
            torch.arange(received_tokens.numel(), device=permutation.device) + local_length
        )
        return cls(
            send_index=send_index,
            send_split=send_split,
            recv_split=recv_split,
            head_index=token_to_buffer[head_window.token_ids(cp_rank)],
            tail_index=None if tail_window is None else token_to_buffer[tail_window.token_ids(cp_rank)],
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class CPExchanges:
    cmp: FlattenMetadata
    swa: WindowMetadata


for metadata_type in (KernelFrame, FlattenMetadata, WindowMetadata, CPExchanges):
    if metadata_type not in SUPPORTED_NODES:
        register_dataclass(metadata_type, serialized_type_name=f"{metadata_type.__module__}.{metadata_type.__name__}")


@dataclass(kw_only=True, slots=True)
class DeepSeekV4Metadata:
    """Global document boundaries and per-operand frames; head/tail are fixed ZigZag entries."""

    ratio: int
    cu_seqlens: torch.Tensor  # Global, even when an operand uses a local window.
    q: KernelFrame
    swa_k: KernelFrame
    cmp_k: KernelFrame | None
    position_k: torch.Tensor  # Document-relative positions of the flat compressor input.
    xchg: CPExchanges | None = None
    head: DeepSeekV4Metadata | None = None
    tail: DeepSeekV4Metadata | None = None

    @classmethod
    def from_positions(cls, positions: torch.Tensor, ratio: int) -> Self:
        if ratio < 0:
            raise ValueError(f"a compression ratio must not be negative, got {ratio}.")
        if positions.dim() != 2:
            raise ValueError(f"positions must be [B, S], got shape {tuple(positions.shape)}.")
        flat = positions.reshape(-1)
        rows = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
        cu_seq_q = torch.cat((torch.unique_consecutive(rows - flat), rows.new_tensor([rows.numel()]))).to(torch.int32)
        lengths = torch.diff(cu_seq_q).to(torch.int32)
        cmp_k = None
        if ratio > 0:
            cmp_k = KernelFrame(
                cu_seqlens=(cu_seq_q // ratio).to(torch.int32),
                seqused=(lengths // ratio).to(torch.int32),
                residual=None if ratio == 1 else (lengths % ratio).to(torch.int32),
            )
        frame = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
        return cls(ratio=ratio, cu_seqlens=cu_seq_q, q=frame, swa_k=frame, cmp_k=cmp_k, position_k=positions)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: DeepSeekV4Metadata,
        *,
        permutation: torch.Tensor | None,
    ) -> DeepSeekV4Metadata:
        del attention_metadata, permutation
        raise ValueError(
            "the DeepSeek-V4 reference stack does not support context parallel: its metadata "
            "describes one contiguous document stream, so it cannot describe a rank-local row. "
            "Enable the fused stack (override.deepseek_v4.sparse_attn.asc) to run with "
            "--parallelism.context-parallel-degree."
        )

    def rebuild_cp_metadata(self, permutation: torch.Tensor | None, *, window_size: int, build):
        from torchtitan_npu.models.deepseek_v4.cp_layout import DOC_CHUNK_LAYOUT, DOC_ZIGZAG_LAYOUT
        from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
            cp_shard_context,
        )

        context = cp_shard_context()
        # Only the layouts this repository registers state a rank-local row.  A pinned balancer
        # (upstream's ``headtail``, which the shard falls back to when the name is unset or
        # unknown) permutes the row into halves these chunk frames cannot describe, so it is
        # refused here instead of producing frames for a row the rank does not hold.
        if context.load_balancer not in (DOC_CHUNK_LAYOUT, DOC_ZIGZAG_LAYOUT):
            raise ValueError(
                f"the {context.load_balancer!r} layout has no rank-local metadata: pass "
                "--parallelism.context-parallel-load-balancer doc_chunk or doc_zigzag."
            )
        assert permutation is not None, "per-document CP requires a token permutation"
        local = self.per_doc_cp_metadata(
            permutation=permutation,
            cp_size=context.cp_mesh.size(),
            cp_rank=context.cp_mesh.get_local_rank(),
            zigzag=context.load_balancer == DOC_ZIGZAG_LAYOUT,
            window_size=window_size,
        )
        if local.head is None:
            return build(local)
        return replace(local, head=build(local.head), tail=build(local.tail))

    def per_doc_cp_metadata(
        self, *, permutation: torch.Tensor, cp_size: int, cp_rank: int, window_size: int, zigzag: bool = False
    ) -> Self:
        num_chunks = 2 * cp_size if zigzag else cp_size
        width = permutation.numel() // cp_size
        if zigzag:
            alignment = num_chunks * max(self.ratio, 1)
            if torch.any(self.q.seqused % alignment):
                raise ValueError(
                    f"doc_zigzag ratio={self.ratio} requires document lengths divisible by {alignment}; "
                    "set dataloader.per_doc_alignment to 2 * CP * max(compress_ratio)."
                )
        token_ids = permutation.reshape(cp_size, width)
        query_cu = self.q.cu_seqlens // num_chunks
        head = DocumentRanges.build(token_ids, query_cu)
        tail = DocumentRanges.build(token_ids, query_cu + width // 2) if zigzag else None
        head_window = head.window(self.cu_seqlens, window_size)
        tail_window = None if tail is None else tail.window(self.cu_seqlens, window_size)
        local = replace(self, position_k=self.position_k[:, cp_rank * width : (cp_rank + 1) * width])
        head_entry = local._query_chunk(head.for_rank(cp_rank), head_window.for_rank(cp_rank))
        if tail is None:
            local = head_entry
        else:
            assert tail_window is not None
            local = replace(
                local, head=head_entry, tail=local._query_chunk(tail.for_rank(cp_rank), tail_window.for_rank(cp_rank))
            )
        return replace(
            local,
            xchg=CPExchanges(
                cmp=FlattenMetadata.build(permutation, cp_size=cp_size, cp_rank=cp_rank),
                swa=WindowMetadata.build(
                    permutation, self.cu_seqlens, head=head, tail=tail, window_size=window_size, cp_rank=cp_rank
                ),
            ),
        )

    def mark_dynamic(self) -> None:
        """Mark batch-dependent leading dimensions so Dynamo does not specialize.

        The document boundaries, the frames, ``position_k`` and both exchange tables follow the
        packing; the arch22 workaround and the selector's forward record are marked when this
        stack carries them.
        """
        tensors: list[torch.Tensor] = [
            self.cu_seqlens,
            self.q.cu_seqlens,
            self.q.seqused,
            self.swa_k.cu_seqlens,
            self.swa_k.seqused,
        ]
        if self.cmp_k is not None:
            tensors += [self.cmp_k.cu_seqlens, self.cmp_k.seqused]
            if self.cmp_k.residual is not None:
                tensors.append(self.cmp_k.residual)
        workaround = getattr(self, "a3_workaround", None)
        if workaround is not None:
            tensors += [gather for gather in (workaround.swa_k_gather, workaround.cmp_k_gather) if gather is not None]
        if self.xchg is not None:
            # Both tables follow the rank-local sequence length, and so does the slab whose
            # positions the compressor rotates.
            tensors += [self.position_k, self.xchg.cmp.recv_index]
            if self.xchg.cmp.send_index is not None:
                tensors.append(self.xchg.cmp.send_index)
            tensors += [self.xchg.swa.send_index, self.xchg.swa.head_index]
            if self.xchg.swa.tail_index is not None:
                tensors.append(self.xchg.swa.tail_index)
        aicpu = getattr(self, "aicpu_metadata", None)
        lightning = getattr(aicpu, "lightning_indexer", None)
        forward = getattr(lightning, "fwd", None)
        if forward is not None:
            tensors.append(forward)
        for tensor in tensors:
            torch._dynamo.maybe_mark_dynamic(tensor, 0)
            # GraphTrainer's tracer reads the dim-0 annotation; the underscore name is the
            # dynamo contract (torchtitan experiments/graph_trainer/dynamic_shapes.py).
            setattr(tensor, "_dynamo_unbacked_indices", {0})  # noqa: B010

    def _query_chunk(self, query: DocumentRanges, window: DocumentRanges) -> Self:
        chunk = query.end - query.start
        reach = query.end - self.cu_seqlens[:-1]
        window_length = window.end - window.start
        query_cu = torch.cat((chunk.new_zeros(1), chunk.cumsum(0))).to(torch.int32)
        window_cu = torch.cat((window_length.new_zeros(1), window_length.cumsum(0))).to(torch.int32)
        cmp_k = None
        if self.cmp_k is not None:
            cmp_k = KernelFrame(
                cu_seqlens=self.cmp_k.cu_seqlens,
                seqused=reach // self.ratio,
                residual=None if self.ratio == 1 else reach % self.ratio,
            )
        return replace(
            self,
            q=KernelFrame(cu_seqlens=query_cu, seqused=chunk),
            swa_k=KernelFrame(cu_seqlens=window_cu, seqused=window_length),
            cmp_k=cmp_k,
        )
