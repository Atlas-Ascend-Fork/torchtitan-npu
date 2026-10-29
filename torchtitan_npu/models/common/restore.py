# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The static doc-chunk layout: the movement between a rank's slab and its chunks.

Shape legend: ``R = cp_size``; ``r = rank``; ``N`` the row's tokens, ``C = N // R`` the slab
width; ``L_d`` document ``d``'s length and ``Q_d = L_d // R`` its chunk.

The dataloader packs one row; the context-parallel shard applies this layout as the **permutation**
it shards by (:func:`doc_chunk_permutation`), so every rank receives **chunk ``r`` of every
document** -- the ``r``-th ``Q_d`` rows of each document, documents in order -- before any hidden
state exists.  That is what makes the per-document reach ``K_d = (r+1)·Q_d`` one problem on every
rank.

:func:`chunk_plan` derives the movement between the two frames from the row's boundaries alone:
pure, on host values, zero communication.  :func:`to_slab` runs it with the functional collective,
so the backward of a restore is the reverse exchange; the opposite direction is the shard's
permutation itself (:func:`doc_chunk_permutation`), which is applied before any hidden state exists.

The move happens **once, in the shard, for the inputs** (the permutation, applied to the token ids,
the positions, the labels and the MTP batch) and **once per layer** for the KV side
(:meth:`RestorePlan.apply`, applied to the hidden states the key projections read).  The query
side never moves: it is computed from the doc-chunk rows the input layout produced, which is what
removes the per-layer query exchange and the output un-permutation.

The two directions are exact inverses -- the movement is a permutation -- so a sharded forward
sees the tokens the unsharded one does, in a different order.

Everything here is stated on **python ints** (the degree and the rank are host values of the run),
so no layout arithmetic ever meets a symbolic rank: the only host round-trips are the split lists
:class:`ExchangeMetadata` carries, which the collective requires and which are derived once per
microbatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single

from torchtitan_npu.models.common.packing import require_document_alignment

DOC_CHUNK_LAYOUT = "doc_chunk"
"""The layout's name on ``--parallelism.context-parallel-load-balancer``.

Every rank's queries are chunk ``r`` of every document.  The CP shard builds its load balancer
from the row's boundaries alone (see ``DocChunkLoadBalancer`` in
``patches/torchtitan/distributed/context_parallel.py``), and the rank-local metadata is derived
from the same boundaries -- no exchange, no collective.
"""

__all__ = [
    "DOC_CHUNK_LAYOUT",
    "ExchangeMetadata",
    "RestorePlan",
    "chunk_index",
    "chunk_plan",
    "doc_chunk_permutation",
    "doc_relative_positions",
    "document_starts",
    "slab_positions",
    "to_slab",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class ExchangeMetadata:
    """One batch's routing between the slab frame and the doc-chunk frame."""

    send_index: torch.Tensor
    """Reorder of this rank's rows (int64, ``[C]``): the rows addressed to peer ``i`` form one
    contiguous slice, peers in order and ascending slot order within a peer."""

    send_split: list[int]
    """``input_split_sizes``: rows this rank addresses to peer ``i``, one entry per peer.  A list,
    because that is what the collective takes -- the plan builds it as one."""

    recv_split: list[int]
    """``output_split_sizes``: rows this rank receives from peer ``i``.  The receiver's buffer is
    already in held-row order -- sources ascend and each source's rows ascend with them -- so
    there is no scatter index and no second reorder."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RestorePlan:
    """doc-chunk -> slab: what the key side needs, shared by every projection that reads it."""

    exchange: ExchangeMetadata
    """The movement itself (the chunks' rows back onto this rank's slab)."""

    def apply(self, x: torch.Tensor, *, group) -> torch.Tensor:
        """Restore ``x`` (doc-chunk rows) onto this rank's slab.

        Differentiable: the reverse exchange is the backward, so the key side's gradient goes
        back to the doc-chunk rows that produced it.
        """
        return to_slab(x, self.exchange, group=group)


def document_starts(positions: torch.Tensor) -> torch.Tensor:
    """The row's document boundaries (``n_documents + 1``), read off its own positions.

    The stack's positions restart at every document, so a row's document start is its own index
    minus its position; the distinct starts are the boundaries, and the row's end closes the last
    one.  Device arithmetic throughout -- no ``nonzero``, no ``tolist`` -- because the CP shard
    runs this on every microbatch and a round-trip there would drain the pipeline.

    This is the one place the layout learns where the documents are when only the positions are
    at hand (the shard runs before any metadata consumer); it must agree with the metadata's
    ``cu_seqlens`` -- the packing grid the positions were built from -- which ``tests`` pins.
    """
    flat = positions.reshape(-1)
    rows = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    # Non-decreasing, one distinct value per document, so ``unique_consecutive`` is the boundary
    # set (a full ``unique`` would sort the row again for nothing).
    doc_start = rows - flat.to(torch.int64)
    return torch.cat((torch.unique_consecutive(doc_start), rows.new_tensor([rows.numel()])))


def doc_relative_positions(cu_seqlens: torch.Tensor, absolute: torch.Tensor) -> torch.Tensor:
    """The document-relative position of each ``absolute`` row, from the row's boundaries.

    The stack's positions restart at every document, so a row's position is its absolute index
    minus the start of the document it falls in.  One derivation, because three callers ask the
    same question of different windows: the slab's own rows, the ratio-4 carry block (whose
    window starts *before* the slab) and a whole row's reconstruction.
    """
    doc = torch.bucketize(absolute, cu_seqlens, right=True) - 1
    return absolute - cu_seqlens[doc.clamp_min(0)]


def chunk_index(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    rank: int | None,
    slab_len: int,
) -> torch.Tensor:
    """Which document chunk each row belongs to.

    ``slab_len`` is the slab width ``C``; with ``rank`` the rows are that rank's slab, without it
    the whole row.  Every document length must be a positive multiple of ``cp_size`` so that a
    chunk is a whole number of rows; the shared packing check enforces it here -- the one check
    in the layout that runs per microbatch, so its callers do not repeat it.
    """
    require_document_alignment(cu_seqlens, cp_size=cp_size)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk = lengths // cp_size
    if rank is None:
        pos = torch.arange(slab_len * cp_size, device=cu_seqlens.device)
    else:
        pos = torch.arange(rank * slab_len, (rank + 1) * slab_len, device=cu_seqlens.device)
    doc = (torch.bucketize(pos, cu_seqlens, right=True) - 1).clamp_min(0)
    return (pos - cu_seqlens[doc]) // chunk[doc]


def doc_chunk_permutation(cu_seqlens: torch.Tensor, cp_size: int, slab_len: int) -> torch.Tensor:
    """The row's global token order under the doc-chunk layout (``[1, N]``, ``int64``).

    The layout is one permutation of the row: rank ``r``'s window of it, ``[r·C, (r+1)·C)``, is
    chunk ``r`` of every document in document order.  Stably sorting the rows by the chunk they
    belong to produces exactly that -- within a chunk the rows keep their global order, so the
    documents stay in order -- which is why the CP shard can apply it *before* slicing and move
    nothing, and why it agrees row for row with the movement :func:`to_slab` undoes.
    """
    return torch.argsort(chunk_index(cu_seqlens, cp_size, None, slab_len), stable=True).unsqueeze(0)


def chunk_plan(cu_seqlens: torch.Tensor, cp_size: int, rank: int, slab_len: int) -> ExchangeMetadata:
    """Derive one rank's slab -> doc-chunk movement from the row's document boundaries.

    ``slab_len`` is the slab width ``C``.  The split lists are the plan's only host values: the
    collective takes them as python lists, and the plan is derived once per microbatch (not per
    layer), so they are the layout's whole host traffic.  The alignment check is
    :func:`chunk_index`'s -- it is the same predicate on the same tensor, and running it twice
    costs a second device read per microbatch.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk = lengths // cp_size

    # Send side: where each row of this rank's slab goes.  The chunk is needed for the sort
    # anyway, and its histogram is the split.
    dst = chunk_index(cu_seqlens, cp_size, rank, slab_len)
    send_index = torch.argsort(dst, stable=True)
    send_split = torch.bincount(dst, minlength=cp_size)

    # Receive side, locally: each document's chunk ``rank`` is one global interval and spans at
    # most two slabs (``chunk <= slab_len``), so it contributes a first segment to the slab it
    # starts in and -- when it crosses -- the remainder to the next one.  The clamp is the
    # crossing test, so no mask and no second pass are needed.
    start = cu_seqlens[:-1] + rank * chunk
    edge = (start // slab_len + 1) * slab_len
    recv_split = torch.bincount(
        torch.cat((start // slab_len, (start // slab_len + 1).clamp(max=cp_size - 1))),
        weights=torch.cat((torch.minimum(start + chunk, edge) - start, (start + chunk - edge).clamp(min=0))).float(),
        minlength=cp_size,
    ).long()

    return ExchangeMetadata(
        send_index=send_index,
        send_split=send_split.tolist(),
        recv_split=recv_split.tolist(),
    )


def to_slab(tensor: torch.Tensor, plan: ExchangeMetadata | None, *, group) -> torch.Tensor:
    """The chunks' rows back onto this rank's slab.

    A permutation, so the adjoint *is* the inverse: the reply is scattered through
    ``send_index`` and nothing accumulates.  Without a plan this is the identity, for the same
    reason as at one degree: the slab *is* the frame there, so the caller's forward needs no
    process group and no branch of its own.
    """
    if plan is None:
        return tensor
    if group is None:
        raise RuntimeError("the doc-chunk layout has no process group; it must be wired before a sharded forward")
    if tensor.size(0) != 1:
        raise ValueError(f"the doc-chunk layout addresses one packed row per batch, got {tuple(tensor.shape)}.")
    received = all_to_all_single(
        tensor.squeeze(0),
        output_split_sizes=plan.send_split,
        input_split_sizes=plan.recv_split,
        group=group,
    )
    out = torch.empty_like(received)
    out[plan.send_index] = received
    return out.unsqueeze(0)


def slab_positions(cu_seqlens: torch.Tensor, cp_size: int, rank: int, slab_len: int) -> torch.Tensor:
    """The document-relative positions of this rank's slab (int64, ``[1, C]``).

    No communication: the row's boundaries and the slab's own interval say where every row sits
    inside its document.
    """
    pos = torch.arange(rank * slab_len, (rank + 1) * slab_len, device=cu_seqlens.device)
    return doc_relative_positions(cu_seqlens, pos).reshape(1, -1)
