# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's context-parallel token layout: chunk ``r`` of every document.

The layout is a property of this stack's attention contract: its rank-local frames are addressed
per document chunk, so the shard has to hand rank ``r`` chunk ``r`` of each document.  It is a
:class:`~torchtitan_npu.patches.torchtitan.distributed.context_parallel.ContextParallelLoadBalancer`
-- upstream #4617's abstraction -- registered under the name its CLI flag carries; the shard patch
carries the registry and the sharding, this module carries the permutation.

The movement that undoes the permutation lives next door in ``cp_attention``/``cp_compressor``: a
layout and its inverse are the shard's permutation and the key side's restore, so they stay side
by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan_npu.patches.torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    register_layout,
)

from .packing import require_document_alignment

DOC_CHUNK_LAYOUT = "doc_chunk"
"""The layout's name on ``--parallelism.context-parallel-load-balancer``.

Every rank's queries are chunk ``r`` of every document; the rank-local metadata is derived from
the same boundaries, so the layout needs no exchange and no collective.
"""

__all__ = [
    "DOC_CHUNK_LAYOUT",
    "DocChunkCPLoadBalancer",
    "chunk_index",
    "doc_chunk_permutation",
]


class DocChunkCPLoadBalancer(ContextParallelLoadBalancer):
    """Rank ``r``'s window is chunk ``r`` of every document, in document order.

    The layout *is* the permutation, so the balancer states it and the shard applies it before
    slicing -- zero communication.  It is a function of the row's document boundaries, the degree
    and the row's length alone: device arithmetic, nothing to agree on across ranks.

    It does not balance work: every rank still holds ``seq_len / cp_size`` tokens, and the causal
    imbalance of a contiguous shard is unchanged.  The name is the seam's -- a static token order
    is what a balancer produces here -- not a claim about the schedule.

    The constructor takes the batch the shard is about to slice: ``seq_len``, the row's boundaries
    (upstream hands the balancer the batch's attention metadata; this pin hands it the positions it
    would have read them from) and the degree the shard runs at.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancer.Config):
        """The doc-chunk layout takes no options: the row's boundaries are its whole input."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        cu_seqlens: torch.Tensor,
        cp_size: int,
    ) -> None:
        del config
        if seq_len % cp_size:
            raise ValueError(f"the row ({seq_len} tokens) does not split into {cp_size} slabs.")
        self.cu_seqlens = cu_seqlens
        self.cp_size = cp_size
        self.seq_len = seq_len

    def generate_permutation(self) -> torch.Tensor:
        """The row's global token order under the doc-chunk layout (``[1, seq_len]``, ``int64``)."""
        return doc_chunk_permutation(self.cu_seqlens, self.cp_size, self.seq_len // self.cp_size)


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

    Stably sorting the rows by the chunk they belong to produces exactly that -- within a chunk the
    rows keep their global order, so the documents stay in order -- which is why the CP shard can
    apply it *before* slicing and move nothing, and why it agrees row for row with the movement
    :func:`to_slab` undoes.
    """
    return torch.argsort(chunk_index(cu_seqlens, cp_size, None, slab_len), stable=True).unsqueeze(0)


# The shard finds a layout by the name its CLI flag carries, so the model package that owns the
# permutation registers it -- at import time, exactly like ``@override``.
register_layout(DOC_CHUNK_LAYOUT, DocChunkCPLoadBalancer)
