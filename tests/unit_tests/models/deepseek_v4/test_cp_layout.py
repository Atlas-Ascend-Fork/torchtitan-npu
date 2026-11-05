# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The doc-chunk layout: the permutation the CP shard cuts the row by.

The layout is the piece the framework calls: it is handed the row the shard is about to slice
(not a slab, and not a precomputed permutation) and states the global token order.  These tests
pin that contract -- the shape it answers with and the windows it produces -- against an order
spelled out by hand, because getting the row/slab width wrong here is invisible until a real
two-rank forward fails on an ``expand``.
"""

from types import SimpleNamespace

import torch

from torchtitan_npu.models.deepseek_v4.cp_layout import DocChunkCPLoadBalancer

# Two documents of 8 and 16 tokens over two ranks: chunks of 4 and 8 rows.
_CU_SEQLENS = torch.tensor([0, 8, 24], dtype=torch.int32)
_ROW_LEN, _CP_SIZE = 24, 2


def _balancer(seq_len: int = _ROW_LEN, cp_size: int = _CP_SIZE) -> DocChunkCPLoadBalancer:
    """The layout of this row, built the way the shard builds it -- from one metadata entry."""
    entry = SimpleNamespace(q=SimpleNamespace(cu_seqlens=_CU_SEQLENS))
    return DocChunkCPLoadBalancer.Config().build(seq_len=seq_len, cp_size=cp_size, attention_metadata=entry)


def _inverse(balancer: DocChunkCPLoadBalancer) -> list[int]:
    """The unshard path's order: upstream's ``_generate_indices(restore=True)``."""
    permutation = balancer.generate_permutation()[0]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    return inverse.tolist()


def test_the_layout_permutes_the_whole_row_it_is_handed():
    """Length is the row's, and the order is chunk ``r`` of every document, documents in order."""
    permutation = _balancer().generate_permutation()

    assert permutation.shape == (1, _ROW_LEN)  # the row, not one slab of it
    # Chunk 0 of documents [0, 8) and [8, 24), then chunk 1 of both.
    assert permutation[0].tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        4,
        5,
        6,
        7,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_each_rank_window_is_one_chunk_of_every_document():
    """Rank ``r``'s slice of the rearranged row is chunk ``r`` of each document, in order."""
    order = _balancer().generate_permutation()[0]
    slab = _ROW_LEN // _CP_SIZE

    assert order[:slab].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    assert order[slab:].tolist() == [4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
    assert _inverse(_balancer()) == torch.argsort(order, dim=-1).tolist()
