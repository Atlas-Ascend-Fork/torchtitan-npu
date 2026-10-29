# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The doc-chunk layout as one of ``cp_shard``'s load balancers.

The balancer is the piece the framework calls: it is handed the row the shard is about to slice
(not a slab, and not a precomputed permutation) and states the global token order.  These tests
pin that contract -- the shape it answers with, the windows it produces, and the inverse the
unshard path asks for -- because getting the row/slab width wrong here is invisible until a real
two-rank forward fails on an ``expand``.
"""

import pytest
import torch

import torchtitan_npu  # noqa: F401  imported for its patch chain
from torchtitan_npu.models.common.restore import doc_chunk_permutation
from torchtitan_npu.patches.torchtitan.distributed.context_parallel import DocChunkLoadBalancer

# Two documents of 8 and 16 tokens over two ranks: chunks of 4 and 8 rows.
_CU_SEQLENS = torch.tensor([0, 8, 24], dtype=torch.int32)
_ROW_LEN, _CP_SIZE = 24, 2


def test_the_balancer_permutes_the_whole_row_it_is_handed():
    """Length is the row's, and the order is the layout's own permutation."""
    balancer = DocChunkLoadBalancer(_CU_SEQLENS, _ROW_LEN, _CP_SIZE)

    permutation = balancer._generate_indices()

    assert permutation.shape == (1, _ROW_LEN)  # the row, not one slab of it
    assert permutation[0].tolist() == doc_chunk_permutation(_CU_SEQLENS, _CP_SIZE, _ROW_LEN // _CP_SIZE)[0].tolist()
    assert torch.equal(balancer._generate_indices(restore=True), torch.argsort(permutation, dim=-1))


def test_each_rank_window_is_one_chunk_of_every_document():
    """Rank ``r``'s slice of the rearranged row is chunk ``r`` of each document, in order."""
    order = DocChunkLoadBalancer(_CU_SEQLENS, _ROW_LEN, _CP_SIZE)._generate_indices()[0]
    slab = _ROW_LEN // _CP_SIZE

    assert order[:slab].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    assert order[slab:].tolist() == [4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]


def test_a_row_that_does_not_split_evenly_is_refused():
    """A shard slices equal windows, so an uneven row has no layout to speak of."""
    with pytest.raises(ValueError, match="does not split"):
        DocChunkLoadBalancer(torch.tensor([0, 6], dtype=torch.int32), 6, 4)
