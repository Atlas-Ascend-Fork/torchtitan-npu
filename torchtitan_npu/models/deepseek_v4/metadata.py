# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed attention metadata for DeepSeek-V4.

The contract is the V4.1 one: **one** dataclass with **one** set of fields describes a geometry at
every context-parallel degree, and only the values differ.  An entry is built once per microbatch
per geometry by the attention backend's config (the map key is that backend's class plus the
layer's compression ratio) and handed to every layer of the geometry, exactly as upstream #4897's
seam does.

The container grid is ``[B, S]``: ``batch_size`` is the leading dimension of ``positions_q``,
and ``S`` is the per-sample sequence length. At CP=1 the flattened microbatch has ``B * S``
tokens, described by ``cu_seq_q[-1]``. The fused path supports ``B > 1`` at CP=1; CP>1
currently requires a single batch row.

``cu_seq_q`` is the **flattened microbatch's** document boundaries: under context parallelism
the KV side stays the whole row and is gathered declaratively at the kernel boundary, so its grid never becomes a
rank-local one.  ``cu_seq_k`` is the same boundaries folded at this entry's ratio (they *are* the
row's for a geometry that does not pool), which is the compressed pool the queries address.

The row is described by one :class:`~torchtitan_npu.models.common.metadata.KernelFrame` per
addressed tensor -- the queries, the sliding-window keys and, for a geometry that pools, the
compressed keys.  ``build_v4_metadata`` derives them from the boundaries above, so the reference
core and a fused port grid the same frames; a vendor backend attaches only the AI_CPU records its
kernels launch.  Under context parallelism the fused entry's ``prepare_cp_metadata`` rebuilds the
frames for the rank's chunks (``shard_asc_v4_metadata``); the reference entry refuses CP, because
it hands one grid to both sides of the attention.

Nothing here describes the compressor's *internal* stream: under the ratio-4 overlap the carry
block a rank prepends lives in ``models/common/carry_exchange.CarryExchange``, which builds the
augmented boundaries at the call site and hands them to the compressor as an argument.  A
compressor is therefore CP-agnostic: it sees one local stream, one set of positions and, for the
overlap, one set of ``cu_seqlens``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchtitan.models.common.attention import create_varlen_metadata_for_document

from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.common.packing import require_document_alignment

if TYPE_CHECKING:
    from torchtitan_npu.models.common.query_exchange import ExchangeMetadata

__all__ = [
    "DeepSeekV4Metadata",
    "KernelFrame",
    "build_v4_metadata",
]


@dataclass(kw_only=True, slots=True)
class DeepSeekV4Metadata:
    """One geometry's DeepSeek-V4 attention contract (the V4.1 shape).

    Handed to each DSA layer through the decoder's metadata map, **one entry per geometry**: the
    map key is the attention config's own class plus the layer's compression ratio, so a layer
    reads the entry built for the axis its kernels grid.  A vendor backend extends it with its own
    records (the AscendC entry adds the AI_CPU metadata).
    """

    cu_seq_q: torch.Tensor
    """The row's packed document boundaries (``int32``, ``n_documents + 1``): the offsets
    ``gather_attn`` takes as ``cu_seqlens`` and the kernels as ``cu_seqlens_*``.  They are the
    **row's** at every degree; under context parallelism this rank's queries are chunk ``r`` of
    every document, which the keyword frames describe, not this tensor."""

    cu_seq_k: torch.Tensor
    """The key side's boundaries: the row's own for a geometry that does not pool, its
    ``ratio``-folded grid for one that does.  Folded by the backend that knows the ratio, so no
    consumer divides a boundary set per layer."""

    ratio: int
    """The compression ratio this entry describes.  Every consumer grids this ratio: a layer reads
    the entry built for its own, so no consumer checks another geometry's plan."""

    positions_q: torch.Tensor
    """The **query** frame's own positions.  The positions carried beside the input describe the
    row this rank holds; under context parallelism the queries are a permuted copy of those rows,
    so every consumer that rotates a query reads them here, at every degree, with no fallback."""

    q: KernelFrame
    """The query frame: the row's boundaries at one degree, this rank's chunks under context
    parallelism, with ``seqused`` the per-document query count."""

    swa_k: KernelFrame
    """The sliding-window keys' frame: the whole row at every degree, with ``seqused`` how far
    these queries reach into each document."""

    cmp_k: KernelFrame | None
    """The compressed pool's frame (``None`` for a geometry that does not pool): the
    ``ratio``-folded row, with ``seqused`` the reach on the compressed grid and ``residual`` its
    trailing partial group."""

    slots: torch.Tensor | None = None
    """The reference selection for a geometry that does not run a selector.

    A geometry that does not select (any ratio but four) reaches every causally complete
    compressed group of the query's own document, and that set is a function of the row's boundaries
    and this entry's ratio alone -- no operand, no layer.  It is therefore built with the entry
    (``[1, seq_len, n_cmp]``, document-local numbering with ``-1`` for an unreachable slot), so
    the reference forward reads it instead of deriving it per layer.  A ratio-4 geometry's
    selection is the selector's, and a fused entry never reads this field."""

    exchange: "ExchangeMetadata | None" = None
    """Under context parallelism the query side's movement (``cp_plan``).  ``None`` at one degree,
    where the slab is the frame."""

    selector_k: KernelFrame | None = None
    """The fused path's rank-local compressed pool, or ``None`` when the whole pool is reached.

    Every fused kernel right-aligns its mask at the end of the key frame it is given (the selector
    reads no ``seqused`` at all), so a rank whose queries reach only a document prefix is handed
    that prefix rather than the gathered row.  Built by ``shard_asc_v4_metadata``; every reference
    build leaves it ``None``."""

    selector_k_gather: torch.Tensor | None = None
    """The ``index_select`` that compacts the gathered compressed row into ``selector_k``'s
    layout.  Differentiable, so the teacher's gradient scatters back onto the gathered row."""

    swa_k_prefix: KernelFrame | None = None
    """The same rank-local treatment for the sliding window: the token prefix each fused kernel's
    end-aligned window must be anchored on.  ``None`` whenever the reach is the whole document."""

    swa_k_gather: torch.Tensor | None = None
    """The ``index_select`` that compacts the gathered sliding-window row into
    ``swa_k_prefix``'s layout."""

    @property
    def batch_size(self) -> int:
        """Batch dimension of the query container, separate from its document count."""
        return self.positions_q.size(0)

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


def _hca_slots(cu_seq_q: torch.Tensor, cu_seq_k: torch.Tensor, ratio: int) -> torch.Tensor:
    """The HCA selection: every causally complete compressed group of the query's document.

    The reference counterpart of a selector's picks: a geometry without a selector does not
    search, it reaches its whole visible prefix, so the slots follow from the two grids and
    the ratio.
    Numbering is document-local with ``-1`` for an unreachable slot, the coordinate system
    ``gather_attn`` reads.
    """
    seqlen = int(cu_seq_q[-1])
    n_cmp = int(cu_seq_k[-1])
    pool_start_L1, pool_end_L1, local_L1 = document_intervals(cu_seq_q, cu_seq_k, num_tokens=seqlen)
    ends_L1 = torch.minimum(pool_end_L1, pool_start_L1 + (local_L1 + 1) // ratio)
    slots_1N = torch.arange(n_cmp, device=cu_seq_q.device).unsqueeze(0)
    local_LN = slots_1N.expand(seqlen, -1) - pool_start_L1
    visible_LN = (slots_1N >= pool_start_L1) & (slots_1N < ends_L1)
    return torch.where(visible_LN, local_LN, -1).unsqueeze(0)


def _frames_from_row(
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    ratio: int,
) -> tuple[KernelFrame, KernelFrame, KernelFrame | None]:
    """The frames of a plain row: boundary differences as the prefixes."""
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    q = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
    swa_k = KernelFrame(cu_seqlens=cu_seq_q, seqused=lengths)
    cmp_k = None
    if ratio > 1:
        # ``cu_seq_k`` was folded at this entry's own ratio by the backend that built it.
        cmp_k = KernelFrame(
            cu_seqlens=cu_seq_k.to(torch.int32),
            seqused=torch.diff(cu_seq_k).to(torch.int32),
            residual=(lengths % ratio).to(torch.int32),
        )
    return q, swa_k, cmp_k


def build_v4_metadata(
    positions: torch.Tensor,
    ratio: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    build_slots: bool = True,
) -> DeepSeekV4Metadata:
    """Build one geometry's entry from the pre-shard row positions.

    Called once per microbatch per geometry by the attention backend's config: the row's
    boundaries, this layer's ratio (the only per-geometry part of the grid) and the frames that
    address them.  The vendor records (the AscendC kernel tensors) are added by that backend's own
    builder.

    Args:
        positions: Per-token positions (``[B, S]``, document-relative: each row starts at zero
            and positions reset at every document start), the convention upstream's
            ``create_varlen_metadata_for_document`` reads the boundaries from.
        ratio: The compression ratio this entry describes -- the layer's own.
        cu_seq_q: The row's boundaries, when the caller already has them.  ``None`` derives them
            from ``positions``.
        build_slots: Whether to precompute the HCA selection for a geometry without a selector
            (``ratio`` above four).  The reference core reads it, so its builder keeps the
            default ``True``; the fused core never reads it and passes ``False``, which keeps
            the ``S x (S / ratio)`` int64 matrix off the fused path.
    """
    if ratio < 1:
        raise ValueError(f"a compression ratio must be positive, got {ratio}.")
    if positions.dim() != 2:
        raise ValueError(f"positions must be [B, S], got shape {tuple(positions.shape)}.")
    if cu_seq_q is None:
        cu_seq_q = create_varlen_metadata_for_document(positions).cu_seq_q
    cu_seq_q = cu_seq_q.to(torch.int32)
    require_document_alignment(cu_seq_q, ratio=ratio)
    lengths = torch.diff(cu_seq_q)
    if int(cu_seq_q[0]) != 0 or int(cu_seq_q[-1]) != positions.numel():
        raise ValueError(
            f"the boundaries describe a different row than the positions: cu_seq_q spans "
            f"[{int(cu_seq_q[0])}, {int(cu_seq_q[-1])}) but the positions hold {positions.numel()} tokens."
        )
    cu_seq_k = cu_seq_q if ratio == 1 else (cu_seq_q // ratio).to(torch.int32)
    q, swa_k, cmp_k = _frames_from_row(cu_seq_q, cu_seq_k, ratio)
    # A geometry without a selector (ratio above four) has a fixed selection: the reference
    # builds it here, once per microbatch and geometry, so its forward only reads it.  The
    # fused core never reads ``slots`` (its HCA branch runs full-causal on the compressed pool),
    # so its builder passes ``build_slots=False`` and keeps this matrix off the fused path.
    slots = None
    if build_slots and ratio not in (1, 4):
        slots = _hca_slots(cu_seq_q, cu_seq_k, ratio)
    return DeepSeekV4Metadata(
        cu_seq_q=cu_seq_q,
        cu_seq_k=cu_seq_k,
        ratio=ratio,
        positions_q=positions,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        slots=slots,
    )
