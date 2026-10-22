# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed compression metadata for DeepSeek-V4.

One entry describes **one compression ratio**: the plan its kernels grid, over the row this
builder derives from ``positions``.  The decoder builds the entry once per microbatch for every
distinct ``(backend, ratio)`` key and hands each layer the one its own attention named, exactly as
upstream #4897's seam does.  Nothing here is a stack-wide ratio table, and no entry carries more
than the geometry it was built for.

The container grid is ``[1, S]`` (the packed scenario runs with ``local_batch_size == 1``; raise
``seq_len`` instead), so ``batch_size == 1`` and ``seq_len`` is the total token count
``cu_seq_q[-1]`` -- both derived from the row, neither stored.

The **vendor records** (the AscendC AI_CPU metadata, for one) are the port's own: they extend
this contract in the backend that consumes them (``override/deepseek_v4``), and this module
carries none.

The **context-parallel layer** lives in ``token_dispatcher.py``: a ratio-independent ``WindowPlan``
(``window``) and the geometry's own block plan (``plan``).  :meth:`DeepSeekV4Metadata.prepare_cp_metadata`
(upstream #4617's hook) **refuses** context parallel: this contract describes one contiguous
document stream.  The fused backend's entry overrides the hook and rebuilds itself for this rank
(``override/deepseek_v4``).

The kernel-layout derivation is the **plain-stream** contract: documents are aligned to the
compression grid (the loader pads each document to ``lcm(compress_ratios)``,
``dataloader.per_doc_alignment``), so a document is always a whole number of blocks and a block
never straddles a document edge.  ``_build_plan`` therefore derives the plan as the row divided by
its ratio -- no per-document scan, no partial trailing block, no empty plain pool -- and refuses a
stream that breaks the contract.  Context-parallel-shaped streams are refused too: those plans
come from ``build_cp_plan`` (``token_dispatcher.py``), where a rank's slab may cut a block and the
borrow machinery applies.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torchtitan.models.common.attention import VarlenMetadata

if TYPE_CHECKING:
    from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
        CPVarlenMetadata,
    )

    from .token_dispatcher import ExchangePlan, WindowPlan

__all__ = [
    "CompressedBlockLayout",
    "CompressedKernelContract",
    "DeepSeekV4Metadata",
    "build_compressed_varlen_metadata",
]


@dataclass(kw_only=True, slots=True)
class CompressedBlockLayout:
    """Kernel contract for one compression ratio (one metadata entry's ``plan``).

    All tensors are built once per batch by the plan build.  Their leading
    dimensions are marked dynamic by the fused path so ``torch.compile``
    does not specialize on batch contents.

    The plan has two parts:

    - **part 1 — the unified contract** (CP and non-CP): the kernel tensors and the compressor
      contract (``gather_indices`` / ``block_positions`` / ``first_indices``), consumed
      identically by the Compressor's dispatcher and the kernels regardless of context parallel;
    - **part 2 — the dispatcher fields** (CP only, ``None`` without): the block exchange routing
      (``exchange``), the container packing (``compressed_rows`` / ``out_width``) and the
      compressed-level gather (``cmp_k_global_gather_indices``).
    """

    # ---- part 1: the unified contract (CP and non-CP) ----
    cu_seqlens_cmp_k: torch.Tensor | None = None
    """Cumulative compressed-block lengths over the packed stream (int32),
    ``[n_seqs + 1]``.  Entry ``i`` is the global start index of sequence
    ``i``'s compressed blocks.  ``None`` for ratio-1 plans."""

    is_empty_host: bool = False
    """Host-side empty-plan flag for paths whose fused kernel rejects empty input.
    A Python branch cannot inspect the unbacked ``gather_indices`` length."""

    block_remainder: torch.Tensor | None
    """Per-sequence block remainder (int32), ``[n_seqs]``.  The aligned document
    contract makes it zero throughout; the kernels still take it."""

    gather_indices: torch.Tensor | None
    """The pooled block-row indices (int64) the dispatcher's ``gather``
    applies.  Without context parallel they are the identity over the row;
    under context parallel they are the assembly into
    ``cat([x_local, exchange_recv_rows])`` in pooled order — the remote gather
    + permute.  ``None`` for ratio-1 plans."""

    block_positions: torch.Tensor | None = None
    """Document-relative block starts (int32) ``[n_blocks]`` — the RoPE
    positions of the pooled keys (block ``b`` of a document sits at
    ``b * ratio``).  The compressor contract: precomputed once per batch so
    no per-layer derivation is needed.  ``None`` for ratio-1 plans."""

    first_indices: torch.Tensor | None = None
    """Document-first block ids (int64) ``[n_segs]`` — the positions of the
    first block of every segment in the ``gather_indices`` block order.  The
    overlap validity mask: the borrowed (previous-block) rows of these blocks
    are masked to zero weight.  ``None`` for ratio-1 plans."""

    compressor_state_block_table: torch.Tensor | None = None
    """Optional backend state table, prebuilt outside the graph.  The fused
    attention entry fills it at the eager metadata boundary (the compressor is
    selected with the fused core)."""

    # ---- part 2: the dispatcher fields (CP only) ----
    exchange: "ExchangePlan | None" = None
    """The block exchange routing (alltoallv input/output splits) of the
    projected ``kv``/``score`` rows completing the plan blocks'
    ``[A, B)`` sub-range.  ``None`` without context parallel."""

    compressed_rows: torch.Tensor | None = None
    """Container-packing selection (int64 ``[n_kept]``): the positions of
    the kept blocks in the pooled key stream (the borrow-source blocks are
    dropped).  ``None`` without context parallel (all blocks are kept)."""

    out_width: int | None = None
    """The container grid width: ``seq_len // ratio`` without context
    parallel, the uniform ``max_kept`` shard width under CP (every rank's
    container is a valid ``S(1)`` shard).  ``None`` for ratio-1 plans."""

    cmp_k_global_gather_indices: torch.Tensor | None = None
    """The compressed-K global gather (int64) — the compressed analogue of
    ``k_global_gather_indices``: per-segment full-prefix blocks as offsets into
    the all-gathered ``[cp * out_width, D]`` container.  ``None`` without
    context parallel."""


@dataclass(kw_only=True, slots=True)
class DeepSeekV4Metadata:
    """One geometry's DeepSeek-V4 varlen attention contract.

    Handed to each DSA layer through the decoder's metadata map, **one entry per geometry**: the
    map key is the attention config's own class plus the layer's compression ratio, so a layer
    reads the entry built for the axis its kernels grid.  Built by the attention backend's config
    from a ``VarlenMetadata`` stream: this geometry's kernel contract (``plan``) over its own row
    (``varlen`` / ``window``).  A vendor backend extends it with its own records (the AscendC
    ``asc``, for one); the reference path consumes this object as it is.

    The container grid is ``[1, S]`` with ``S`` equal to the total token count
    (``local_batch_size == 1``): ``batch_size`` is always ``1`` and ``seq_len`` is
    ``cu_seq_q[-1]`` — both derived as properties, not stored.
    """

    varlen: "VarlenMetadata | CPVarlenMetadata"
    """Token-stream boundaries (``cu_seq_q`` / ``cu_seq_k``).  Under context
    parallel the rank-local ``CPVarlenMetadata`` (the shard path's own
    builder output)."""

    ratio: int
    """The compression ratio this entry describes.  Every consumer grids this ratio: a
    layer reads the entry built for its own, so no consumer checks another geometry's plan."""

    plan: CompressedBlockLayout
    """This geometry's kernel contract.  A ratio-1 plan describes no compressed region."""

    window: "WindowPlan | None" = None
    """The sliding-window plan (CP only, ``None`` without): the ratio-
    independent window exchange + ori-stream assembly the Attention's
    ``swa_k`` gather consumes (``token_dispatcher.WindowPlan``)."""

    window_size: int | None = None
    """The sliding window the plans were built for (the layer's own config
    constant), set by ``build_compressed_varlen_metadata``.  The CP transform
    needs it to re-derive the window plan, and the metadata object is all it
    is handed."""

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: "DeepSeekV4Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "DeepSeekV4Metadata":
        """Refuse context parallel: only the fused stack has a rank-local form.

        This contract addresses the pool the whole row built, and its offsets are the row's own,
        so a rank that holds a chunk has no metadata a reference consumer could read.  A CP run
        therefore stops here, before any layer runs, and the fused entry
        (``AscV4Metadata``) overrides this with the plan, window and records rebuilt from the
        rank's own chunks.
        """
        del attention_metadata, permutation
        raise ValueError(
            "the DeepSeek-V4 reference stack does not support context parallel: its metadata "
            "describes one contiguous document stream, so it cannot describe a rank-local row. "
            "Enable the fused stack (override.deepseek_v4.sparse_attn.asc) to run with "
            "--parallelism.context-parallel-degree."
        )

    @property
    def batch_size(self) -> int:
        """Container batch size (``1`` for the current packed scenario)."""
        return 1


@runtime_checkable
class CompressedKernelContract(Protocol):
    """The per-ratio plan shape shared by the model-dir metadata and the NPU type.

    Consumers that read only the plan (e.g. the Compressor) accept this protocol instead of the
    concrete metadata type.
    """

    ratio: int
    plan: CompressedBlockLayout


def _host_boundaries(varlen: VarlenMetadata) -> tuple[int, ...]:
    """The document boundaries on the host: cached on the stream, or read back once.

    One row's boundaries are read once per microbatch -- at the eager build, never inside a
    compiled layer -- and the layout then works off that tuple.
    """
    cached = varlen.cu_seq_q_host
    if cached is not None:
        return tuple(cached)
    return tuple(varlen.cu_seq_q.cpu().tolist())


def _build_plan(
    varlen: VarlenMetadata,
    ratio: int,
    *,
    boundaries: tuple[int, ...],
    require_aligned: bool,
) -> CompressedBlockLayout:
    """One ratio's kernel contract over a plain, aligned stream.

    Called once per batch by ``build_compressed_varlen_metadata``: the plan is derived purely
    from the document boundaries and this layer's ratio.  Context-parallel streams must use
    ``build_cp_plan`` instead (guarded below).
    """
    if not hasattr(varlen, "cu_seq_q"):
        raise TypeError(f"build_compressed_varlen_metadata expects a varlen stream, got {type(varlen)}.")

    cu_seq_q = varlen.cu_seq_q
    cu = boundaries
    if not cu or cu[0] != 0:
        raise ValueError(f"varlen stream must start at token 0, got cu_seq_q[0]={cu[0] if cu else None}.")
    if not torch.equal(cu_seq_q, varlen.cu_seq_k):
        raise ValueError(
            "build_compressed_varlen_metadata requires a plain stream (cu_seq_q == "
            "cu_seq_k); context-parallel plans come from build_cp_plan."
        )
    if ratio == 1:
        # A window-only geometry: no compressed pool, so nothing to derive.
        return CompressedBlockLayout(
            cu_seqlens_cmp_k=None,
            block_remainder=None,
            gather_indices=None,
        )
    if ratio not in (4, 128):
        raise NotImplementedError(f"CompressedBlockLayout does not support ratio={ratio}; expected 1, 4, or 128.")

    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"documents must be non-empty, got lengths {lengths}.")
    seq_len = cu[-1]
    n_seqs = len(lengths)
    device = cu_seq_q.device
    aligned = all(length % ratio == 0 for length in lengths)
    if not aligned and require_aligned:
        raise ValueError(
            "document lengths are not multiples of the compression ratio "
            f"{ratio}: {lengths}. The dataloader must pad each document to "
            "`per_doc_alignment` (the lcm of the stack's compress_ratios); an "
            "unaligned stream is an input bug. Pass require_aligned=False only for "
            "tests and numerics probes that need the partial-block contract."
        )
    if aligned:
        # The aligned contract the loader guarantees: every document is a whole number of blocks,
        # so the compressed pool is the row itself -- the boundaries are the row's divided by the
        # ratio, the gather keeps every token (in order), and the remainder is zero.  The block
        # positions stay document-relative (the compressor's RoPE restarts at every document),
        # which is the one field that still needs one entry per document.
        n_blocks = seq_len // ratio
        return CompressedBlockLayout(
            cu_seqlens_cmp_k=(cu_seq_q // ratio).to(torch.int32),
            is_empty_host=False,
            block_remainder=torch.zeros(n_seqs, dtype=torch.int32, device=device),
            gather_indices=torch.arange(seq_len, dtype=torch.int64, device=device),
            block_positions=torch.cat(
                [torch.arange(0, length, ratio, dtype=torch.int32, device=device) for length in lengths]
            ),
            first_indices=(cu_seq_q[:-1] // ratio).to(torch.int64),
            compressed_rows=None,
            out_width=n_blocks,
        )

    # The unaligned derivation: a document contributes its complete leading blocks only; the
    # ``len % ratio`` tail produces no compressed entry.  The loader never produces such a stream
    # (``dataloader.per_doc_alignment``), but callers that build metadata from arbitrary token
    # streams (the unit tests, numerics probes) still get the partial-block contract.  That is
    # also why a zero-length compressed segment is only possible here, never in production: an
    # aligned stream's every document holds at least one whole block.
    c_lens = [length // ratio for length in lengths]
    cu_seqs = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.tensor(c_lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
        ]
    )
    pieces = [
        torch.arange(k_start, k_start + ratio * cnt, dtype=torch.int64, device=device)
        for k_start, cnt in zip(cu[:-1], c_lens, strict=True)
        if cnt
    ]
    positions = [torch.arange(0, ratio * cnt, ratio, dtype=torch.int32, device=device) for cnt in c_lens if cnt]
    gather = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), dtype=torch.int64, device=device)
    block_positions = torch.cat(positions, dim=0) if positions else torch.empty((0,), dtype=torch.int32, device=device)
    first_indices = cu_seqs[:-1][torch.diff(cu_seqs) > 0].to(torch.int64)
    return CompressedBlockLayout(
        cu_seqlens_cmp_k=cu_seqs,
        is_empty_host=not pieces,
        block_remainder=torch.tensor(
            [length % ratio for length in lengths],
            dtype=torch.int32,
            device=device,
        ),
        gather_indices=gather,
        block_positions=block_positions,
        first_indices=first_indices,
        compressed_rows=None,
        # The container width is the *kept* block count, not the row's floor: a
        # document's tail group contributes no block.
        out_width=sum(c_lens),
    )


def build_compressed_varlen_metadata(
    varlen: VarlenMetadata,
    ratio: int,
    *,
    window_size: int | None = None,
    require_aligned: bool = True,
) -> DeepSeekV4Metadata:
    """Build one geometry's DSV4 varlen contract from a plain varlen stream.

    Called once per microbatch per geometry by the attention backend's config: the kernel
    contract for *this layer's ratio* (``_build_plan`` over one ratio).  The vendor records (the
    AscendC kernel tensors, for one) are added by that backend's own builder.

    Args:
        varlen: The plain (unsharded) stream; ``cu_seq_q`` is authoritative for the document
            boundaries.  Under context parallel the CP transform re-derives this object per
            rank, so a CP-shaped stream never reaches here.
        ratio: The compression ratio this entry describes — the layer's own.
        window_size: The sliding window the plan belongs to; the CP transform re-derives the
            window plan from it.
        require_aligned: Refuse a document stream whose lengths are not multiples of ``ratio``.
            The loader's ``per_doc_alignment`` guarantees this in production; tests and numerics
            probes that deliberately build unaligned streams pass ``False`` to get the
            partial-block contract.
    """
    boundaries = _host_boundaries(varlen)
    return DeepSeekV4Metadata(
        varlen=varlen,
        ratio=ratio,
        plan=_build_plan(varlen, ratio, boundaries=boundaries, require_aligned=require_aligned),
        window_size=window_size,
    )
