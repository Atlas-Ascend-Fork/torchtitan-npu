# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed attention metadata for DeepSeek-V4.

**One** dataclass with **one** set of fields describes a geometry at every context-parallel
degree, and only the values differ.  An entry is built once per microbatch per geometry by the
attention backend's config (the map key is that backend's class plus the layer's compression
ratio) and handed to every layer of the geometry, exactly as upstream #4897's seam does.

The row is described by one :class:`KernelFrame` per addressed tensor -- the queries, the
sliding-window keys and, for a geometry that pools, the compressed keys.
:meth:`DeepSeekV4Metadata.from_positions` derives them from the row's boundaries, so the reference core and a
fused port grid the same frames; a vendor backend attaches only the AI_CPU records its kernels
launch.  A kernel that cannot read these frames as they are -- arch22's TND kernels ignore
``seqused`` and anchor their mask at the frame's end -- restates them at its own call site
(``override/a3_workaround.py``) rather than here: nothing about the row changes.

Under context parallelism the queries are **chunk ``r`` of every document** while the KV side
stays the **whole row**.  The input layout (``models/deepseek_v4/cp_attention.py``) moves the
batch's tokens, positions and labels onto the chunk frame once per batch, so ``q.cu_seqlens`` is
the rank-local chunk grid and the KV frames address the row; a layer that projects keys restores
its hidden states onto the flatten slice with the entry's :attr:`DeepSeekV4Metadata.xchg`, which is
what removes the per-layer query movement and the output un-permutation.  There are no
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

    Handed to each DSA layer through the decoder's metadata map, **one entry per geometry**: the
    map key is the attention config's own class plus the layer's compression ratio, so a layer
    reads the entry built for the axis its kernels grid.  A vendor backend extends it with its own
    records (the AscendC entry adds the AI_CPU metadata).
    """

    ratio: int
    """The compression ratio this entry describes.  Every consumer grids this ratio: a layer reads
    the entry built for its own, so no consumer checks another geometry's plan."""

    q: KernelFrame
    """The query frame: the row's boundaries at one degree, this rank's chunks under context
    parallelism, with ``seqused`` the per-document query count."""

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
    """The doc-chunk layout under context parallelism, ``None`` at one degree.

    It carries the movement the key side applies, not the frames and not the positions: those are
    the same fields at every degree.  A consumer reads the stream through
    :func:`~torchtitan_npu.models.deepseek_v4.cp_attention.restore_key_stream`.
    """

    @classmethod
    def from_positions(cls, positions: torch.Tensor, ratio: int) -> Self:
        """Build one geometry's entry from the pre-shard row positions.

        Called once per microbatch per geometry by the attention backend's config: the row's
        boundaries (read off the positions), this layer's ratio -- the only per-geometry part of
        the grid -- and the frames that address them.  A vendor backend adds its records on top
        of the same entry.

        A ratio of zero addresses no compressed axis: the layer runs the sliding window and the
        sink only, so its entry carries no compressed frame at all.
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

        The row is the whole packed sequence and this entry hands one grid to both sides of the
        attention, so a rank that holds a chunk has no metadata a reference consumer could read.
        A CP run therefore stops here, before any layer runs, and the fused entry
        (``AscV4Metadata``) overrides this with the frames and records rebuilt from the rank's own
        chunks.
        """
        del attention_metadata, permutation
        raise ValueError(
            "the DeepSeek-V4 reference stack does not support context parallel: its metadata "
            "describes one contiguous document stream, so it cannot describe a rank-local row. "
            "Enable the fused stack (override.deepseek_v4.sparse_attn.asc) to run with "
            "--parallelism.context-parallel-degree."
        )

    def doc_chunk_entry(self, *, permutation: torch.Tensor, cp_size: int, cp_rank: int) -> Self:
        """This rank's entry: the doc-chunk query frame, the row's KV frames and the movement.

        The frames and the movement are derived from the row's boundaries alone, so the entry
        crosses no collective.  Each document is assumed to be a whole multiple of
        ``cp_size * ratio`` -- the packing side pads for it -- so every prefix is a whole number
        of chunks and every compressed prefix a whole number of groups.
        """
        cu_seqlens = self.q.cu_seqlens
        ratio = self.ratio
        chunk = (self.q.seqused // cp_size).to(torch.int32)
        reach = ((cp_rank + 1) * chunk).to(torch.int32)
        # The compressed axis is a property of the row, not of the degree: a geometry that pools
        # keeps its pool grid here, and one that does not has no frame at either degree.
        cmp_k = None
        if self.cmp_k is not None:
            cmp_k = KernelFrame(
                cu_seqlens=self.cmp_k.cu_seqlens,
                seqused=(reach // ratio).to(torch.int32),
                residual=None if ratio == 1 else (reach % ratio).to(torch.int32),
            )
        # The key stream this rank computes is the row's contiguous slice (the queries are the
        # chunk rows), so its positions are that slice of the row's own positions.
        flatten_len = permutation.numel() // cp_size
        start = cp_rank * flatten_len
        return replace(
            self,
            q=KernelFrame(cu_seqlens=(cu_seqlens // cp_size).to(torch.int32), seqused=chunk),
            swa_k=KernelFrame(cu_seqlens=cu_seqlens.to(torch.int32), seqused=reach),
            cmp_k=cmp_k,
            position_k=self.position_k[:, start : start + flatten_len],
            xchg=ExchangeMetadata.from_permutation(permutation, cp_size=cp_size, cp_rank=cp_rank),
        )
