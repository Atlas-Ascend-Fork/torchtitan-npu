# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's per-document CP layouts: contiguous chunks or balanced head/tail pairs.

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

DOC_ZIGZAG_LAYOUT = "doc_zigzag"
DOC_CHUNK_LAYOUT = "doc_chunk"
"""The layout's name on ``--parallelism.context-parallel-load-balancer``.

Every rank's queries are chunk ``r`` of every document; the rank-local metadata is derived from
the same boundaries, so the layout needs no exchange and no collective.
"""

__all__ = [
    "DOC_CHUNK_LAYOUT",
    "DOC_ZIGZAG_LAYOUT",
    "DocChunkCPLoadBalancer",
    "DocZigZagCPLoadBalancer",
    "PerDocCPLoadBalancer",
]


class PerDocCPLoadBalancer(ContextParallelLoadBalancer):
    """The shared construction of the document layouts.

    A layout states a token order that gives every rank whole pieces of every document, so the
    shard it drives never cuts a document: the row's boundaries come from the batch's attention
    metadata, the degree from the run.  One construction for all of them -- only the piece count
    and the assignment differ -- which is why the subclasses state just their permutation.

    The constructor takes the batch the shard is about to slice -- the row's length, the degree and
    the batch's attention metadata, where the row's boundaries are -- as upstream's balancers do;
    the degree is a parameter because this pin has no ambient CP mesh to read it from.  Nothing is
    checked here: what the split needs of the row is stated where the row is built.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancer.Config):
        """The document layouts take no options: boundaries and degree are their whole input."""

    def __init__(
        self,
        config: Config,
        *,
        seq_len: int,
        cp_size: int,
        attention_metadata,
    ) -> None:
        del config
        self.cu_seqlens = attention_metadata.q.cu_seqlens
        self.cp_size = cp_size
        self.seq_len = seq_len

    def _document_chunks(self, chunks: int) -> list[torch.Tensor]:
        """Each document as a ``(chunks, L_d // chunks)`` block of global token indices.

        The split is not validated here: every document length must be a positive multiple of
        ``chunks``, and the shard's own checks state that contract before a layout is built.
        """
        return [
            torch.arange(start, end, device=self.cu_seqlens.device, dtype=torch.int64).view(chunks, -1)
            for start, end in zip(self.cu_seqlens[:-1].tolist(), self.cu_seqlens[1:].tolist(), strict=True)
        ]

    def generate_permutation(self) -> torch.Tensor:
        """The row's global token order under this layout (``[1, seq_len]``, ``int64``)."""
        raise NotImplementedError


class DocChunkCPLoadBalancer(PerDocCPLoadBalancer):
    """Rank ``r``'s window is chunk ``r`` of every document, in document order.

    The layout *is* the permutation, so the balancer states it and the shard applies it before
    slicing -- zero communication.  Stacking the chunk-``r`` block of every document along the
    document axis and flattening rank-major is that order: chunk ``r`` of document 0, then of
    document 1, and so on.

    It does not balance work: every rank still holds ``seq_len / cp_size`` tokens, and the causal
    imbalance of a contiguous shard is unchanged.  The name is the seam's -- a static token order
    is what a balancer produces here -- not a claim about the schedule.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PerDocCPLoadBalancer.Config):
        """The doc-chunk policy takes no options."""

    def generate_permutation(self) -> torch.Tensor:
        """The row's global token order under the doc-chunk layout (``[1, seq_len]``, ``int64``)."""
        return torch.cat(self._document_chunks(self.cp_size), dim=1).reshape(-1).unsqueeze(0)


class DocZigZagCPLoadBalancer(PerDocCPLoadBalancer):
    """Per-document head/tail balancing, with all heads before all tails on each rank."""

    @dataclass(kw_only=True, slots=True)
    class Config(PerDocCPLoadBalancer.Config):
        pass

    def generate_permutation(self) -> torch.Tensor:
        num_chunks = 2 * self.cp_size
        lengths = torch.diff(self.cu_seqlens)
        if torch.any(lengths % num_chunks):
            raise ValueError(
                "doc_zigzag requires each document length to be divisible by 2 * CP; "
                "set dataloader.per_doc_alignment to 2 * CP * max(compress_ratio)."
            )
        chunks = torch.cat(self._document_chunks(num_chunks), dim=1)
        return torch.stack((chunks[: self.cp_size], chunks[self.cp_size :].flip(0)), dim=1).reshape(1, -1)


# The shard finds a layout by the name the CLI flag carries, so the package that owns the
# permutation registers it here, at import time -- exactly like ``@override``.  Without this call
# ``build_load_balancer`` returns ``None`` and the shard silently falls back to the pinned
# contiguous split, which is only observable as a numerics change at CP>1.
register_layout(DOC_CHUNK_LAYOUT, DocChunkCPLoadBalancer)
register_layout(DOC_ZIGZAG_LAYOUT, DocZigZagCPLoadBalancer)
