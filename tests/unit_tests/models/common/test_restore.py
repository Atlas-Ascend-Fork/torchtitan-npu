# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The static doc-chunk layout: the plan, the movement between frames, and its inverse.

The plan is pure tensor math, so the movement is pinned here the way the collective runs it:
every rank's send side is split per peer, the receive side is the concatenation of what every
peer addressed to it, and the restore swaps the two splits and scatters through the plan's index.
The simulation is the contract -- a real ``all_to_all_single`` is that same operation.
"""

import pytest
import torch

from torchtitan_npu.models.common.restore import (
    chunk_plan,
    slab_positions,
    to_slab,
)

# Every document is a multiple of both degrees, so its chunks are whole rows.
_ROW = (32, 16, 64)


def _row_cu(lengths=_ROW) -> torch.Tensor:
    return torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)


def _plans(lengths=_ROW, cp_size: int = 2):
    cu = _row_cu(lengths)
    slab = int(cu[-1]) // cp_size
    return cu, slab, [chunk_plan(cu, cp_size, rank, slab) for rank in range(cp_size)]


def _chunk_rows(lengths: tuple[int, ...], cp_size: int, rank: int) -> list[int]:
    """The row indices of chunk ``rank``: each document's own chunk, documents in order."""
    rows: list[int] = []
    start = 0
    for length in lengths:
        rows += list(range(start + rank * (length // cp_size), start + (rank + 1) * (length // cp_size)))
        start += length
    return rows


def _exchange(tensors: list[torch.Tensor], plans) -> list[torch.Tensor]:
    """Slab -> doc-chunk, the way ``all_to_all_single`` runs it (split on the send side)."""
    sent = {}
    for rank, plan in enumerate(plans):
        moved = tensors[rank][plan.send_index]
        offset = 0
        for peer, split in enumerate(plan.send_split):
            sent[rank, peer] = moved[offset : offset + split]
            offset += split
    out = []
    for rank, plan in enumerate(plans):
        parts = [sent[source, rank] for source, split in enumerate(plan.recv_split) if split]
        out.append(torch.cat(parts) if parts else tensors[rank][:0])
    return out


def _restore(tensors: list[torch.Tensor], plans) -> list[torch.Tensor]:
    """doc-chunk -> slab: the swapped splits, then the scatter the plan's index names."""
    sent = {}
    for rank, plan in enumerate(plans):
        offset = 0
        for peer, split in enumerate(plan.recv_split):
            sent[rank, peer] = tensors[rank][offset : offset + split]
            offset += split
    out = []
    for rank, plan in enumerate(plans):
        parts = [sent[source, rank] for source, split in enumerate(plan.send_split) if split]
        received = torch.cat(parts) if parts else tensors[rank][:0]
        slab = torch.empty_like(received)
        slab[plan.send_index] = received
        out.append(slab)
    return out


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_plan_splits_the_slab_and_the_peers_agree(cp_size):
    _, slab, plans = _plans(_ROW, cp_size)
    for plan in plans:
        # Every row of this rank's slab is addressed to exactly one peer.
        assert int(plan.send_index.numel()) == slab
        assert sorted(plan.send_index.tolist()) == list(range(slab))
        assert sum(plan.send_split) == slab
        assert sum(plan.recv_split) == slab
    # What one rank sends is what its peer expects: the two sides of the same table.
    for sender in range(cp_size):
        for receiver in range(cp_size):
            assert plans[sender].send_split[receiver] == plans[receiver].recv_split[sender]


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_movement_reproduces_the_chunk_layout(cp_size):
    _, slab, plans = _plans(_ROW, cp_size)
    slabs = [torch.arange(rank * slab, (rank + 1) * slab) for rank in range(cp_size)]
    moved = _exchange(slabs, plans)
    for rank in range(cp_size):
        assert moved[rank].tolist() == _chunk_rows(_ROW, cp_size, rank)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_restore_is_the_exact_inverse(cp_size):
    _, slab, plans = _plans(_ROW, cp_size)
    slabs = [torch.randn(slab) for _ in range(cp_size)]
    restored = _restore(_exchange(slabs, plans), plans)
    for rank in range(cp_size):
        assert torch.equal(restored[rank], slabs[rank])


def test_a_chunk_may_straddle_two_slabs():
    """A chunk is one interval of the row, not of a slab: a document need not be slab-aligned.

    ``(8, 56)`` with four ranks: the second document starts at row 8, so its first chunk
    (``[8, 22)``) begins in rank 0's slab and ends in rank 1's -- the plan's receive side is
    what puts both segments on rank 1, in row order.
    """
    lengths, cp_size = (8, 56), 4
    _, slab, plans = _plans(lengths, cp_size)
    slabs = [torch.arange(rank * slab, (rank + 1) * slab) for rank in range(cp_size)]
    moved = _exchange(slabs, plans)
    for rank in range(cp_size):
        assert moved[rank].tolist() == _chunk_rows(lengths, cp_size, rank)


def _slab_rows(cp_size: int, rank: int, slab: int) -> torch.Tensor:
    return torch.arange(rank * slab, (rank + 1) * slab)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_restored_positions_are_the_slabs_own(cp_size):
    cu, slab, _ = _plans(_ROW, cp_size)
    for rank in range(cp_size):
        positions = slab_positions(cu, cp_size, rank, slab)
        assert positions.shape == (1, slab)
        for row, position in zip(_slab_rows(cp_size, rank, slab), positions.reshape(-1).tolist(), strict=True):
            document = int(torch.searchsorted(cu[1:], row, right=True))
            assert position == row - int(cu[document])


def test_the_plan_refuses_a_document_that_cannot_be_chunked():
    cu = _row_cu((10, 22))
    with pytest.raises(ValueError, match="per_doc_alignment"):
        chunk_plan(cu, 4, 0, 8)


def test_one_degree_is_the_identity():
    """At one degree the entry carries no plan and nothing moves -- not even a collective."""
    tensor = torch.arange(8).reshape(1, -1)
    assert to_slab(tensor, None, group=None) is tensor


def test_the_movement_keeps_the_container_grid(monkeypatch):
    """A ``[1, L, ...]`` tensor comes back ``[1, L, ...]``: the exchange addresses dim 1."""
    from torchtitan_npu.models.common import restore as restore_module

    _, slab, plans = _plans(_ROW, 2)
    seen = []

    def fake_all_to_all(tensor, *, output_split_sizes, input_split_sizes, group):
        del group
        seen.append((tuple(tensor.shape), tuple(input_split_sizes), tuple(output_split_sizes)))
        return tensor

    monkeypatch.setattr(restore_module, "all_to_all_single", fake_all_to_all)
    tensor = torch.randn(1, slab, 3)
    out = to_slab(tensor, plans[0], group=object())
    assert out.shape == tensor.shape
    # The collective splits the row axis, never the batch axis; the restore swaps the splits.
    plan = plans[0]
    assert seen == [((slab, 3), tuple(plan.recv_split), tuple(plan.send_split))]
