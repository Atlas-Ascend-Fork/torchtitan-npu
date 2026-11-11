# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4's per-document CP layouts: contiguous chunks or balanced head/tail pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan_npu.patches.torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    register_layout,
)

DOC_ZIGZAG_LAYOUT = "doc_zigzag"
DOC_CHUNK_LAYOUT = "doc_chunk"
"""The layout's name on ``--parallelism.context-parallel-load-balancer``."""

__all__ = [
    "DOC_CHUNK_LAYOUT",
    "DOC_ZIGZAG_LAYOUT",
    "DocChunkCPLoadBalancer",
    "DocZigZagCPLoadBalancer",
    "PerDocCPLoadBalancer",
]


@dataclass(frozen=True)
class DocumentRanges:
    """Global [start, end) per document, optionally with a leading CP-rank dimension."""

    start: torch.Tensor
    end: torch.Tensor

    @classmethod
    def build(cls, permuted_rows, cu_seqlens):
        """Read every rank's contiguous run per document back out of the sharded row."""
        return cls(
            permuted_rows[:, cu_seqlens[:-1]].to(torch.int32),
            (permuted_rows[:, cu_seqlens[1:] - 1] + 1).to(torch.int32),
        )

    def for_rank(self, rank):
        return DocumentRanges(self.start[rank], self.end[rank])

    def window(self, cu_seqlens, window_size):
        return DocumentRanges(torch.maximum(cu_seqlens[:-1], self.start - window_size + 1), self.end)

    def token_ids(self, rank):
        """Concrete global token ids of this rank's ranges, laid out in document order."""
        start, end = self.start[rank].to(torch.int64), self.end[rank].to(torch.int64)
        lengths = end - start
        # The runs are laid back to back, so a token's id is its position minus its run start
        # plus the document's own start.
        run_start = lengths.cumsum(0) - lengths
        position = torch.arange(int(lengths.sum()), device=self.start.device, dtype=torch.int64)
        return position - torch.repeat_interleave(run_start, lengths) + torch.repeat_interleave(start, lengths)


class PerDocCPLoadBalancer(ContextParallelLoadBalancer):
    """The shared construction of the document layouts."""

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
        """Each document as a ``(chunks, L_d // chunks)`` block of global token indices."""
        return [
            torch.arange(start, end, device=self.cu_seqlens.device, dtype=torch.int64).view(chunks, -1)
            for start, end in zip(self.cu_seqlens[:-1].tolist(), self.cu_seqlens[1:].tolist(), strict=True)
        ]

    def generate_permutation(self) -> torch.Tensor:
        """The row's global token order under this layout (``[1, seq_len]``, ``int64``)."""
        raise NotImplementedError


class DocChunkCPLoadBalancer(PerDocCPLoadBalancer):
    """Rank ``r``'s window is chunk ``r`` of every document, in document order."""

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
