# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from torchtitan_npu.models.deepseek_v4.context_parallel import ContextParallelCompressorMixin
from torchtitan_npu.models.deepseek_v4.metadata import FlattenMetadata
from torchtitan_npu.models.deepseek_v4.cp_layout import DocChunkCPLoadBalancer
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata

# Every document is a multiple of both degrees, so its chunks are whole rows.
_ROW = (32, 16, 64)


def _row_cu(lengths=_ROW) -> torch.Tensor:
    return torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)


def _plans(lengths=_ROW, cp_size: int = 2):
    """This layout's movement, derived from the permutation the shard applies."""
    cu = _row_cu(lengths)
    slab = int(cu[-1]) // cp_size
    row = DeepSeekV4Metadata.from_positions(_positions(lengths), 1)
    permutation = (
        DocChunkCPLoadBalancer.Config()
        .build(seq_len=int(cu[-1]), cp_size=cp_size, attention_metadata=row)
        .generate_permutation()
    )
    return (
        cu,
        slab,
        [FlattenMetadata.build(permutation, cp_size=cp_size, cp_rank=rank) for rank in range(cp_size)],
    )


def _positions(lengths=_ROW) -> torch.Tensor:
    """The packed row's document-relative positions, from its lengths."""
    return torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)


def _chunk_rows(lengths: tuple[int, ...], cp_size: int, rank: int) -> list[int]:
    """The row indices of chunk ``rank``: each document's own chunk, documents in order."""
    rows: list[int] = []
    start = 0
    for length in lengths:
        rows += list(range(start + rank * (length // cp_size), start + (rank + 1) * (length // cp_size)))
        start += length
    return rows


def _exchange(tensors: list[torch.Tensor], plans) -> list[torch.Tensor]:
    """Flat shards to query layout, the inverse of the production exchange."""
    sent = {}
    for rank, plan in enumerate(plans):
        ordered = tensors[rank][torch.argsort(plan.recv_index)]
        offset = 0
        for peer, count in enumerate(plan.recv_split):
            sent[rank, peer] = ordered[offset : offset + count]
            offset += count
    outputs = []
    for rank, plan in enumerate(plans):
        received = torch.cat([sent[source, rank] for source in range(len(plans))])
        outputs.append(received if plan.send_index is None else received[torch.argsort(plan.send_index)])
    return outputs


def _restore(tensors: list[torch.Tensor], plans) -> list[torch.Tensor]:
    """Query layout to flat shards, in the production communication direction."""
    sent = {}
    for rank, plan in enumerate(plans):
        ordered = tensors[rank] if plan.send_index is None else tensors[rank][plan.send_index]
        offset = 0
        for peer, count in enumerate(plan.send_split):
            sent[rank, peer] = ordered[offset : offset + count]
            offset += count
    return [torch.cat([sent[source, rank] for source in range(len(plans))])[plan.recv_index]
            for rank, plan in enumerate(plans)]


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_plan_splits_the_slab_and_the_peers_agree(cp_size):
    _, slab, plans = _plans(_ROW, cp_size)
    for plan in plans:
        # Every row of this rank's slab is addressed to exactly one peer.
        assert int(plan.recv_index.numel()) == slab
        assert sorted(plan.recv_index.tolist()) == list(range(slab))
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
    row_positions = _positions(_ROW)
    for rank in range(cp_size):
        positions = row_positions[:, rank * slab : (rank + 1) * slab]
        assert positions.shape == (1, slab)
        for row, position in zip(_slab_rows(cp_size, rank, slab), positions.reshape(-1).tolist(), strict=True):
            document = int(torch.searchsorted(cu[1:], row, right=True))
            assert position == row - int(cu[document])


def test_one_degree_is_the_identity():
    """At one degree the entry carries no plan and nothing moves -- not even a collective."""
    tensor = torch.arange(8).reshape(1, -1)
    assert ContextParallelCompressorMixin()._restore_flat_input(tensor, None) is tensor


def test_the_movement_keeps_the_container_grid(monkeypatch):
    """A ``[1, L, ...]`` tensor comes back ``[1, L, ...]``: the exchange addresses dim 1."""
    from torchtitan_npu.models.deepseek_v4 import context_parallel as restore_module

    _, slab, plans = _plans(_ROW, 2)
    seen = []

    def fake_all_to_all(tensor, *, output_split_sizes, input_split_sizes, group):
        del group
        seen.append((tuple(tensor.shape), tuple(input_split_sizes), tuple(output_split_sizes)))
        return tensor

    monkeypatch.setattr(restore_module, "all_to_all_single", fake_all_to_all)
    tensor = torch.randn(1, slab, 3)
    module = ContextParallelCompressorMixin()
    module.group = object()
    out = module._restore_flat_input(tensor, plans[0])
    assert out.shape == tensor.shape
    # Split vectors describe this forward: input is sent, output is received.
    plan = plans[0]
    assert seen == [((slab, 3), tuple(plan.send_split), tuple(plan.recv_split))]
