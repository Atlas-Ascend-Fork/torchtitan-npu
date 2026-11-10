# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed attention metadata for DeepSeek-V4.

**One** dataclass with **one** set of fields describes a geometry at every context-parallel
degree, and only the values differ.  One metadata object is built once per microbatch per
geometry by the attention backend's config (the map key is that backend's class plus the layer's
compression ratio) and handed to every layer of the geometry, exactly as upstream #4897's seam
does.

The row is described by one :class:`KernelFrame` per addressed tensor -- the queries, the
sliding-window keys and, for a geometry that pools, the compressed keys.
:meth:`DeepSeekV4Metadata.from_positions` derives them from the row's boundaries, so the reference core and a
fused port grid the same frames; a vendor backend adds only the AI_CPU metadata its kernels
produce.  A kernel that cannot read these frames as they are -- arch22's TND kernels ignore
``seqused`` and anchor their mask at the frame's end -- restates them at its own call site
(``override/a3_workaround.py``) rather than here: nothing about the row changes.

Under context parallelism KV remains the whole row. Doc-chunk keeps each document's chunk
``r``; ZigZag pairs it with the mirrored chunk in ``head`` / ``tail``. Their query frames are
local, while the enclosing entry keeps global KV frames. ``xchg`` restores the projected
keys to flat rank slices. There are no
``cu_seq_q`` / ``cu_seq_k`` aliases: read the grid from the frame that addresses it (row grid =
``swa_k.cu_seqlens``, query grid = ``q.cu_seqlens``, compressed grid = ``cmp_k.cu_seqlens``).

Nothing here describes the compressor's *internal* stream: under the ratio-4 overlap the carry
block a rank prepends lives in ``models/deepseek_v4/cp_compressor.CarryExchange``, which builds
the augmented boundaries at the call site and hands them to the compressor as an argument.
"""

from dataclasses import dataclass, replace
from typing import Self

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4.cp_attention import ExchangeMetadata

__all__ = [
    "DeepSeekV4Metadata",
    "KernelFrame",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """One tensor's coordinate frame: the boundaries needed to address it.

    ``cu_seqlens`` are the ragged document boundaries (``int32``, ``n_documents + 1``), the form
    the kernels take as ``cu_seqlens_*``.  ``seqused`` is the per-document prefix this frame's
    consumer addresses -- how an operator is told to read a prefix of its operand -- and
    ``residual`` the trailing partial group of a compressed axis (``None`` where the axis is not
    compressed; a backend whose kernels need a value substitutes zeros for it).
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


if KernelFrame not in SUPPORTED_NODES:
    register_dataclass(KernelFrame, serialized_type_name=f"{KernelFrame.__module__}.{KernelFrame.__name__}")


@dataclass(kw_only=True, slots=True)
class DeepSeekV4Metadata:
    """One geometry's DeepSeek-V4 attention contract (the V4.1 shape).

    Handed to each DSA layer through the decoder's metadata map, **one per geometry**: the map key
    is the attention config's own class plus the layer's compression ratio, so a layer reads the
    metadata built for the axis its kernels grid.  A vendor backend subclasses it and adds the
    operator outputs its kernels produce (``AscV4Metadata`` is the AscendC one).
    """

    ratio: int
    """The compression ratio this metadata describes.  Every consumer grids this ratio: a layer
    reads the metadata built for its own, so no consumer checks another geometry's plan."""

    q: KernelFrame
    """Local queries in kernel entries; global frame in the enclosing ZigZag entry."""

    swa_k: KernelFrame
    """The sliding-window keys' frame: the whole row at every degree, with ``seqused`` how far
    these queries reach into each document.  Its ``cu_seqlens`` are the **row's** grid -- the one
    a consumer reads when it needs the whole row rather than its own frame."""

    cmp_k: KernelFrame | None
    """The compressed pool's frame (``None`` for a geometry that does not pool): the
    ``ratio``-folded row, with ``seqused`` the reach on the compressed grid and ``residual`` its
    trailing partial group."""

    position_k: torch.Tensor
    """The key stream's own document-relative positions (``[B, C]``, ``C`` the row under one degree).

    They belong to the stream the key projections read, so they are a property of the geometry, not
    of any module: at one degree they are the input's ``positions``; under context parallelism they
    are the flatten slice's own, from the row's boundaries -- which is why a consumer never derives
    them and never learns that the stream moved.
    """

    xchg: ExchangeMetadata | None = None
    """The per-document layout's exchange under context parallelism, ``None`` at one degree.

    It carries the movement the key side applies, not the frames and not the positions: those are
    the same fields at every degree.  A consumer reads the stream through
    :func:`~torchtitan_npu.models.deepseek_v4.cp_attention.restore_key_stream`.
    """

    head: "DeepSeekV4Metadata | None" = None
    tail: "DeepSeekV4Metadata | None" = None
    """Fixed ZigZag query entries; both absent for doc-chunk."""

    @classmethod
    def from_positions(cls, positions: torch.Tensor, ratio: int) -> Self:
        """Build one geometry's metadata from the pre-shard positions.

        Called once per microbatch per geometry by the attention backend's config: the row's
        boundaries (read off the positions), this layer's ratio -- the only per-geometry part of
        the grid -- and the frames that address them.  A vendor backend adds its operator outputs
        on top of the same.

        A ratio of zero addresses no compressed axis: the layer runs the sliding window and the
        sink only, so its metadata carries no compressed frame at all.
        """
        if ratio < 0:
            raise ValueError(f"a compression ratio must not be negative, got {ratio}.")
        if positions.dim() != 2:
            raise ValueError(f"positions must be [B, S], got shape {tuple(positions.shape)}.")
        # A document starts where the positions restart at zero, so each row's document start is
        # its own index minus its position; the row's end closes the last document.  Device
        # arithmetic throughout: this runs once per geometry per microbatch.
        flat = positions.reshape(-1)
        rows = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
        cu_seq_q = torch.cat((torch.unique_consecutive(rows - flat), rows.new_tensor([rows.numel()]))).to(torch.int32)
        lengths = torch.diff(cu_seq_q).to(torch.int32)
        cmp_k = None
        if ratio > 0:
            cmp_k = KernelFrame(
                # ``// 1`` is the identity, so ratio one keeps the row's grid.
                cu_seqlens=(cu_seq_q // ratio).to(torch.int32),
                seqused=(lengths // ratio).to(torch.int32),
                # Ratio one addresses the axis without pooling it: there is no trailing group, and
                # that absence is ``None``, not zeros -- arch22's kernels take the zero group from
                # ``A3KeyWorkaround``, arch35's reject a value here.
                residual=None if ratio == 1 else (lengths % ratio).to(torch.int32),
            )
        frame = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
        return cls(ratio=ratio, q=frame, swa_k=frame, cmp_k=cmp_k, position_k=positions)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: "DeepSeekV4Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "DeepSeekV4Metadata":
        """Refuse context parallel: only the fused stack has a rank-local form.

        The row is the whole packed sequence and this metadata hands one grid to both sides of the
        attention, so a rank that holds a chunk has no metadata a reference consumer could read.
        A CP run therefore stops here, before any layer runs, and the fused metadata
        (``AscV4Metadata``) overrides this with the frames and AI_CPU metadata rebuilt from the
        rank's own chunks.
        """
        del attention_metadata, permutation
        raise ValueError(
            "the DeepSeek-V4 reference stack does not support context parallel: its metadata "
            "describes one contiguous document stream, so it cannot describe a rank-local row. "
            "Enable the fused stack (override.deepseek_v4.sparse_attn.asc) to run with "
            "--parallelism.context-parallel-degree."
        )

    def rebuild_cp_metadata(self, permutation: torch.Tensor | None, *, build):
        """This rank's entry, dispatched by the layout the shard applied.

        ``build`` rebuilds a rank-local entry's operator outputs; the caller states it because
        the hook is a staticmethod and nothing here dispatches through ``type(metadata)``.
        """
        # ``cp_shard_context`` refuses a transform that runs outside ``Decoder._cp_shard``, and
        # that call site publishes a context only under context parallel: the mesh is never
        # absent here.
        from torchtitan_npu.models.deepseek_v4.cp_layout import DOC_ZIGZAG_LAYOUT
        from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
            cp_shard_context,
        )

        context = cp_shard_context()
        assert permutation is not None, "per-document CP requires a token permutation"
        local = self.per_doc_cp_metadata(
            permutation=permutation,
            cp_size=context.cp_mesh.size(),
            cp_rank=context.cp_mesh.get_local_rank(),
            zigzag=context.load_balancer == DOC_ZIGZAG_LAYOUT,
        )
        if local.head is None:
            return build(local)
        return replace(local, head=build(local.head), tail=build(local.tail))

    def per_doc_cp_metadata(
        self, *, permutation: torch.Tensor, cp_size: int, cp_rank: int, zigzag: bool = False
    ) -> Self:
        """Per-document query frames and the shared flat-KV positions and exchange."""
        num_chunks = 2 * cp_size if zigzag else cp_size
        width = permutation.numel() // cp_size
        if zigzag:
            alignment = num_chunks * max(self.ratio, 1)
            if torch.any(self.q.seqused % alignment):
                raise ValueError(
                    f"doc_zigzag ratio={self.ratio} requires document lengths divisible by {alignment}; "
                    "set dataloader.per_doc_alignment to 2 * CP * max(compress_ratio)."
                )
        local = replace(self, position_k=self.position_k[:, cp_rank * width : (cp_rank + 1) * width])
        if zigzag:
            local = replace(
                local,
                head=local._query_chunk(num_chunks=num_chunks, chunk_index=cp_rank),
                tail=local._query_chunk(num_chunks=num_chunks, chunk_index=num_chunks - cp_rank - 1),
            )
        else:
            local = local._query_chunk(num_chunks=num_chunks, chunk_index=cp_rank)
        return replace(local, xchg=ExchangeMetadata.from_permutation(permutation, cp_size=cp_size, cp_rank=cp_rank))

    def mark_dynamic(self) -> None:
        """Mark batch-dependent leading dimensions so Dynamo does not specialize.

        The frames, ``position_k`` and the exchange tables follow the packing; the arch22
        workaround and the selector's forward record are marked when this stack carries them.
        """
        tensors: list[torch.Tensor] = [
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
            tensors += [self.position_k, self.xchg.send_index]
            if self.xchg.recv_index is not None:
                tensors.append(self.xchg.recv_index)
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

    def _query_chunk(self, *, num_chunks: int, chunk_index: int) -> Self:
        """One equal piece of every document, addressing the unchanged global KV grids."""
        chunk = (self.q.seqused // num_chunks).to(torch.int32)
        reach = (chunk_index + 1) * chunk
        cmp_k = None
        if self.cmp_k is not None:
            cmp_k = KernelFrame(
                cu_seqlens=self.cmp_k.cu_seqlens,
                seqused=reach // self.ratio,
                residual=None if self.ratio == 1 else reach % self.ratio,
            )
        return replace(
            self,
            q=KernelFrame(cu_seqlens=self.q.cu_seqlens // num_chunks, seqused=chunk),
            swa_k=KernelFrame(cu_seqlens=self.swa_k.cu_seqlens, seqused=reach),
            cmp_k=cmp_k,
        )
