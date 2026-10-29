# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The static doc-chunk movement on real collectives: two gloo ranks, one packed row.

``tests/unit_tests/models/common/test_restore.py`` pins the plan by simulating the movement
rank-by-rank; this file runs the movement itself through ``all_to_all_single`` on two CPU
processes, which is the only place the collective's split semantics (dim-0 splitting, per-peer
order, an empty split still entering the call) are exercised end to end: the layout, its exact
inverse, the plan's positions, and the restore's backward, which must be the reverse exchange.
"""

import socket

import pytest
import torch
import torch.distributed as dist

# Two documents, each a multiple of cp_size * ratio (2 * 4), one of them crossing a slab edge.
_LENGTHS = (32, 48)
_RATIO = 4
_CP = 2
_ROW = sum(_LENGTHS)
_SLAB = _ROW // _CP


def _chunk_rows(rank: int) -> torch.Tensor:
    """The row indices of chunk ``rank``: each document's own chunk, documents in order."""
    rows, start = [], 0
    for length in _LENGTHS:
        rows.append(torch.arange(start + rank * (length // _CP), start + (rank + 1) * (length // _CP)))
        start += length
    return torch.cat(rows)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_chunks(tensor: torch.Tensor, plan, *, group) -> torch.Tensor:
    """The slab -> chunk movement, spelled out here: what the shard's permutation stands in for.

    Production only ever runs the restore, so the opposite direction is the test's own definition
    of the layout -- it is not a business function kept alive for tests.
    """
    from torch.distributed._functional_collectives import all_to_all_single

    moved = all_to_all_single(
        tensor.squeeze(0)[plan.send_index],
        output_split_sizes=plan.recv_split,
        input_split_sizes=plan.send_split,
        group=group,
    )
    return moved.unsqueeze(0)


def _worker(rank: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_CP,
    )
    try:
        from torchtitan_npu.models.common.restore import slab_positions, to_slab
        from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata, doc_chunk_entry

        group = dist.group.WORLD
        tokens = torch.arange(_ROW).reshape(1, -1) + 100
        positions = torch.cat([torch.arange(length) for length in _LENGTHS]).reshape(1, -1)
        labels = tokens + 1
        slab = slice(rank * _SLAB, (rank + 1) * _SLAB)

        row = build_v4_metadata(positions, _RATIO)
        entry = doc_chunk_entry(row, cp_size=_CP, cp_rank=rank, slab_len=_SLAB)
        plan = entry.cp.restore.exchange

        # The frames the entry describes: chunk r of every document over the whole-row KV frame.
        chunk = torch.tensor([length // _CP for length in _LENGTHS], dtype=torch.int32)
        assert torch.equal(entry.q.seqused, chunk)
        assert torch.equal(entry.swa_k.seqused, (rank + 1) * chunk)

        # The layout moves this rank's slab onto its chunks, documents in order.
        moved = _to_chunks(tokens[:, slab], plan, group=group)
        assert torch.equal(moved, tokens[:, _chunk_rows(rank)])
        assert torch.equal(_to_chunks(labels[:, slab], plan, group=group), labels[:, _chunk_rows(rank)])
        expected_positions = torch.cat(
            [torch.arange(rank * (length // _CP), (rank + 1) * (length // _CP)) for length in _LENGTHS]
        ).reshape(1, -1)
        assert torch.equal(_to_chunks(positions[:, slab], plan, group=group), expected_positions)

        # The restore is the exact inverse, its positions come from the entry, and its backward is
        # the reverse exchange.
        assert torch.equal(to_slab(moved, plan, group=group), tokens[:, slab])
        assert torch.equal(entry.position_k, slab_positions(row.swa_k.cu_seqlens, _CP, rank, _SLAB))

        x = (torch.randn(1, _SLAB, 3, dtype=torch.float64) + 10.0 * rank).requires_grad_()
        slabbed = to_slab(x, plan, group=group)
        weight = torch.arange(slabbed.numel(), dtype=torch.float64).reshape(slabbed.shape)
        (slabbed * weight).sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, _to_chunks(weight, plan, group=group))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo is unavailable")
def test_the_chunk_layout_and_its_restore_run_on_real_collectives():
    port = _free_port()
    torch.multiprocessing.spawn(_worker, args=(port,), nprocs=_CP, join=True)
