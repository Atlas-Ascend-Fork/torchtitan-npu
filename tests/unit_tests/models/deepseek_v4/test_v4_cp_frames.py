# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4 context-parallel frames, rebuilt with the V4.1 query-exchange contract.

The queries are chunk ``r`` of every document while the KV streams stay the whole row; the
frames' ``seqused`` is what limits a rank's view.  The compressor is CP-agnostic, so the
ratio-4 overlap's carry -- the predecessor's last block, prepended and trimmed -- is applied by
``CarryExchange`` beside it, and the two halves of this file pin those two contracts: the frames
the kernels grid, and the carry that makes a slab compress like the row.

Neither key stream can stay the whole row under context parallelism: an arch22 fused kernel
right-aligns its mask at the end of the key frame it is given and reads no ``seqused``, so the
options state the reach itself as that frame (``override/a3_workaround.a3_key_frame``) and the
wrapper compacts both gathered rows onto it before the kernel call.  The entry keeps the model's
own row frames at every degree, and ``test_the_selector_is_handed_this_ranks_prefix_pool`` pins
the grids the kernels are actually handed.
"""

from itertools import pairwise

import pytest
import torch
from torch import nn

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.compressor import Compressor
from torchtitan_npu.override.deepseek_v4.metadata import AicpuGeometry
from torchtitan_npu.models.deepseek_v4.metadata import doc_chunk_entry
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import build_aicpu_metadata

_WINDOW = 8
_RATIO = 4
_DIM, _HEAD_DIM, _ROPE_HEAD_DIM = 6, 8, 4
_GEOMETRY = AicpuGeometry(
    num_heads=2,
    head_dim=4,
    window_size=_WINDOW,
    index_topk=4,
    index_num_heads=4,
    index_head_dim=4,
)


def _positions(*documents: int) -> torch.Tensor:
    flat: list[int] = []
    for length in documents:
        flat += list(range(length))
    return torch.tensor([flat])


def _row(positions: torch.Tensor, *, ratio: int = _RATIO):
    config = CompressedSparseInnerAttention.Config(
        window_size=_WINDOW,
        compress_ratio=ratio,
        softmax_scale=0.5,
        index_topk=4,
    )
    return config.build_attention_metadata(positions)


def _entry(positions: torch.Tensor, *, ratio: int = _RATIO):
    return build_aicpu_metadata(_row(positions, ratio=ratio), geometry=_GEOMETRY)


def _masks(bounds: list[int], cp_size: int = 1, cp_rank: int = 0, ratio: int = _RATIO):
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)
    entry = _entry(positions, ratio=ratio)
    if cp_size == 1:
        return entry
    local = doc_chunk_entry(
        _row(positions, ratio=ratio),
        cp_size=cp_size,
        cp_rank=cp_rank,
        slab_len=positions.shape[-1] // cp_size,
    )
    return build_aicpu_metadata(local, geometry=_GEOMETRY)


def _lengths(bounds: list[int]) -> torch.Tensor:
    return torch.tensor([end - begin for begin, end in pairwise(bounds)], dtype=torch.int32)


def _prefix(lengths: torch.Tensor) -> torch.Tensor:
    """The frame of a per-document prefix: the running sum of the reach, from zero."""
    return torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, 0).to(torch.int32)])


def _gather_of(entry, stream: str) -> torch.Tensor | None:
    """The entry's compaction for one key stream, or ``None`` when no row moves.

    arch35 reads each frame's ``seqused`` itself, so it compacts nothing and carries no workaround;
    arch22 states the reach as the frame and gathers the rows onto it.
    """
    workaround = entry.a3_workaround
    if _IS_A5:
        assert workaround is None, "arch35 needs no key workaround"
        return None
    assert workaround is not None, "an arch22 entry states the key workaround"
    return workaround.swa_k_gather if stream == "swa_k" else workaround.cmp_k_gather


def _kernel_grid(reach: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """The frame the kernels are handed for a stream whose reach is ``reach``."""
    return row if _IS_A5 else _prefix(reach)


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("bounds", [[0, 16, 48], [0, 16, 32, 48]], ids=["two-docs", "three-docs"])
def test_one_degree_passes_the_boundary_differences(bounds, ratio):
    entry = _masks(bounds, ratio=ratio)
    lengths = _lengths(bounds)
    row = torch.tensor(bounds, dtype=torch.int32)
    # The row's grid lives on the frame that addresses it.
    assert torch.equal(entry.swa_k.cu_seqlens, row)
    assert entry.cmp_k is None or torch.equal(entry.cmp_k.cu_seqlens, row // ratio)
    assert torch.equal(entry.q.cu_seqlens, row)
    assert torch.equal(entry.q.seqused, lengths)
    assert torch.equal(entry.swa_k.cu_seqlens, row)
    assert torch.equal(entry.swa_k.seqused, lengths)
    if ratio == 1:
        assert entry.cmp_k is None
    else:
        assert torch.equal(entry.cmp_k.cu_seqlens, row // ratio)
        assert torch.equal(entry.cmp_k.seqused, lengths // ratio)
        assert entry.cmp_k.residual is not None and not entry.cmp_k.residual.any()


@pytest.mark.parametrize(
    "cp_size, cp_rank", [(2, 0), (2, 1), (4, 1), (4, 3), (8, 7)], ids=["R2r0", "R2r1", "R4r1", "R4r3", "R8r7"]
)
@pytest.mark.parametrize("ratio", [1, 4])
def test_the_anchor_is_this_ranks_chunk_offset(cp_size, cp_rank, ratio):
    bounds = [0, 32, 64]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    reach = (cp_rank + 1) * chunk
    entry = _masks(bounds, cp_size, cp_rank, ratio=ratio)
    # The row never becomes a rank-local grid: the KV side stays the whole row, which the entry's
    # own boundaries state.  The key frames are the frames the kernels are addressed with: the row
    # itself where they read ``seqused``, the reach where they anchor their mask at the frame's end.
    assert torch.equal(entry.swa_k.cu_seqlens, _kernel_grid(reach, row))
    assert torch.equal(entry.q.cu_seqlens, (row // cp_size).to(torch.int32))
    assert torch.equal(entry.q.seqused, chunk)
    assert torch.equal(entry.swa_k.seqused, reach)
    assert torch.equal(entry.swa_k.seqused - entry.q.seqused, cp_rank * chunk)
    if ratio == 1:
        assert entry.cmp_k is None
    else:
        assert torch.equal(entry.cmp_k.cu_seqlens, _kernel_grid(reach // ratio, row // ratio))
        assert torch.equal(entry.cmp_k.seqused, reach // ratio)
        assert entry.cmp_k.residual is not None and not entry.cmp_k.residual.any()
        if _IS_A5:
            # arch35 reads ``seqused`` itself: nothing is compacted.
            assert entry.a3_workaround is None
        elif cp_rank == cp_size - 1:
            # This rank's reach is every document in full, so no row moves.
            assert _gather_of(entry, "swa_k") is None
            assert _gather_of(entry, "cmp_k") is None
        else:
            starts = row[:-1].to(torch.int64)
            assert torch.equal(
                _gather_of(entry, "swa_k"),
                torch.cat(
                    [
                        torch.arange(int(start), int(start) + int(length))
                        for start, length in zip(starts, reach, strict=True)
                    ]
                ),
            )
            pool_starts = (row // ratio)[:-1].to(torch.int64)
            assert torch.equal(
                _gather_of(entry, "cmp_k"),
                torch.cat(
                    [
                        torch.arange(int(start), int(start) + int(length))
                        for start, length in zip(pool_starts, reach // ratio, strict=True)
                    ]
                ),
            )
    # The kernel-side positions are this rank's slab, restored from the doc-chunk input rows.
    slab = bounds[-1] // cp_size
    held = torch.arange(cp_rank * slab, (cp_rank + 1) * slab)
    docs = torch.searchsorted(row[1:], held, right=True)
    expected = (held - row[docs]).tolist()
    assert entry.cp is not None
    assert entry.position_k.reshape(-1).tolist() == expected


def test_the_selector_is_handed_this_ranks_prefix_pool(dsv4):
    """The kernels are handed this rank's reach; the entry stays the whole row.

    ``lightning_indexer`` right-aligns the compressed causal mask at the end of the key frame it
    is given and reads no ``seqused``, so handing a non-final rank the gathered row makes it pick
    the document-tail window -- keys its queries cannot read.  The options therefore state the
    reach as the frame -- each document's first ``reach // ratio`` blocks for the pool, its first
    ``reach`` tokens for the sliding window -- and both records are built from that grid.  The
    picks stay document-local, so the numbers still address the whole-row pool SMLA/SMLAG read.
    """
    bounds = [0, 32, 48]  # two documents, so every prefix is a per-document prefix
    cp_size, cp_rank = 2, 0
    entry = _masks(bounds, cp_size, cp_rank)
    prefix = (_lengths(bounds) // cp_size * (cp_rank + 1)) // _RATIO
    assert prefix.tolist() == [4, 2]

    # The entry keeps the frames its kernels read: arch35 states the whole row, arch22 hands the
    # kernels this rank's reach as the frame (the branch below pins both).
    assert entry.cmp_k.cu_seqlens.tolist() == ([0, 8, 12] if _IS_A5 else [0, 4, 6])
    assert entry.swa_k.cu_seqlens.tolist() == ([0, 32, 48] if _IS_A5 else [0, 16, 24])

    # The pool the selector reads is each document's prefix, in document order, and the sliding
    # window is this rank's token prefix -- both stated by the entry, both applied to the rows.
    assert entry.cmp_k is not None
    assert entry.cmp_k.seqused.tolist() == [4, 2]
    assert entry.swa_k.seqused.tolist() == [16, 8]
    if _IS_A5:
        # arch35 reads ``seqused`` itself: the frames stay the row's and nothing is compacted.
        assert entry.a3_workaround is None
        assert entry.cmp_k.cu_seqlens.tolist() == [0, 8, 12]
        assert entry.swa_k.cu_seqlens.tolist() == [0, 32, 48]
    else:
        # arch22 is handed the reach as the frame, and the rows are compacted onto it.
        assert _gather_of(entry, "cmp_k").tolist() == [0, 1, 2, 3, 8, 9]
        assert _gather_of(entry, "swa_k").tolist() == [*range(16), *range(32, 40)]
        assert entry.cmp_k.cu_seqlens.tolist() == [0, 4, 6]
        assert entry.swa_k.cu_seqlens.tolist() == [0, 16, 24]

    # Both AI_CPU records are sized for the grid their kernels address: this rank's reach where the
    # kernel anchors at the frame's end, the gathered row where it reads ``seqused``.
    pool_grid = [0, 8, 12] if _IS_A5 else [0, 4, 6]
    token_grid = [0, 32, 48] if _IS_A5 else [0, 16, 24]
    li_calls = [call for call in dsv4.cann_ops.calls if call[0] == "lightning_indexer_metadata"]
    assert li_calls
    assert li_calls[-1][2]["cu_seqlens_k"].tolist() == pool_grid
    assert li_calls[-1][2]["seqused_k"].tolist() == [4, 2]
    slig_calls = [call for call in dsv4.cann_ops.calls if call[0] == "sparse_lightning_indexer_kl_loss_grad_metadata"]
    assert slig_calls
    assert slig_calls[-1][2]["cu_seqlens_k"].tolist() == pool_grid
    assert slig_calls[-1][2]["seqused_k"].tolist() == [4, 2]

    # The compressed attention branch is handed the same pool: the selection, the keys and the grid
    # are one frame in both spellings.  The sliding-window stream keeps ``seqused_ori_kv``.
    smla_calls = [call for call in dsv4.cann_ops.calls if call[0] == "sparse_flash_mla_metadata"]
    assert smla_calls
    assert smla_calls[-1][2]["cu_seqlens_cmp_kv"].tolist() == pool_grid
    assert smla_calls[-1][2]["seqused_cmp_kv"].tolist() == [4, 2]
    assert smla_calls[-1][2]["cu_seqlens_ori_kv"].tolist() == token_grid
    assert smla_calls[-1][2]["seqused_ori_kv"].tolist() == [16, 8]


def test_the_sharded_path_checks_the_launcher_contract():
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _masks([0, 20, 48], 2, 1)


@pytest.mark.parametrize("documents, cp_size", [((20,), 2), ((12,), 4)])
def test_a_document_shorter_than_cp_times_ratio_is_refused(documents, cp_size):
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _masks([0, *documents], cp_size, cp_size - 1)


# ---------------------------------------------------------------------------
# The CP carry: CarryExchange outside the compressor
# ---------------------------------------------------------------------------


class _IdentityRoPE(nn.Module):
    """The pooling oracle applies no rope, so the comparison can be exact."""

    def forward(self, x, key=None, positions=None, *, inverse=False):
        return x if key is None else (x, x)


class _Build:
    def __init__(self, fn):
        self._fn = fn

    def build(self):
        return self._fn()


class _Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _compressor(ratio: int = _RATIO) -> Compressor:
    torch.manual_seed(7)
    coff = 2 if ratio == 4 else 1
    compressor = Compressor(
        Compressor.Config(
            rope=_Build(_IdentityRoPE),
            head_dim=_HEAD_DIM,
            rope_head_dim=_ROPE_HEAD_DIM,
            compress_ratio=ratio,
            carry_exchange=CarryExchange.Config(ratio=ratio),
            wkv=_Build(lambda: nn.Linear(_DIM, coff * _HEAD_DIM, bias=False)),
            wgate=_Build(lambda: nn.Linear(_DIM, coff * _HEAD_DIM, bias=False)),
            norm=_Build(lambda: _Norm(_HEAD_DIM)),
        )
    )
    nn.init.trunc_normal_(compressor.ape, std=0.02)
    return compressor


def _carry(module: Compressor, exchange: CarryExchange, x, positions, cu_seqlens):
    """The call site's composition: exchange -> compress -> trim."""
    carry = exchange(x, positions, cu_seqlens)
    out = module(carry.x, carry.positions, cu_seqlens=carry.cu_seqlens)
    if carry.trim:
        out = out.narrow(1, carry.trim, out.size(1) - carry.trim)
    return out


class _FakeMesh:
    """The two mesh methods the carry reads."""

    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.mark.parametrize("ratio", [4, 128], ids=["overlap", "non-overlap"])
def test_cp_rejects_batched_slabs_before_exchange_or_bounds(ratio):
    exchange = CarryExchange.Config(ratio=ratio).build()
    exchange.group = _FakeMesh(0, 2)
    x = torch.zeros(2, 128, _DIM)
    positions = torch.arange(128).expand(2, -1)
    cu_seqlens = torch.tensor([0, 256, 512], dtype=torch.int32)

    with pytest.raises(ValueError, match="CP>1 暂不支持 local_batch_size>1"):
        exchange(x, positions, cu_seqlens)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_carry_reproduces_the_whole_row_pool(cp_size, monkeypatch):
    """A rank's kept blocks are exactly its block range of the whole row's pool.

    Two equal documents span the slabs, so the ranks past the first carry a predecessor block;
    the one whose slab starts exactly at the second document's edge carries it as a segment of
    its own, the ones starting inside a document merge it into that document's first segment.
    Both cases must reproduce the global pool, which is what pins the merge rule.
    """
    from torchtitan_npu.models.common import carry_exchange as carry_module

    ratio = _RATIO
    total = 16 * ratio
    doc_len = total // 2
    bounds = [0, doc_len, total]
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)
    cu = torch.tensor(bounds, dtype=torch.int32)
    module = _compressor(ratio)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(1, total, _DIM, generator=generator)
    slab = total // cp_size

    with torch.no_grad():
        global_pool = module(x, positions, cu_seqlens=cu)
    assert global_pool.shape == (1, total // ratio, _HEAD_DIM)

    def fake_all_to_all(send, *, output_split_sizes, input_split_sizes, group):
        del group
        # The collective splits dim 0: a caller that hands over a ``[1, rows, D]`` slab is
        # asking for one row of blocks instead of one block of rows.
        assert send.size(0) == sum(input_split_sizes)
        sources = [source for source, split in enumerate(output_split_sizes) if split]
        if not sources:
            return send[:0]
        source = sources[0]
        first = source * slab
        if send.dtype == torch.int32:
            return positions[..., first + slab - ratio : first + slab].reshape(-1).to(torch.int32)
        return x[:, first + slab - ratio : first + slab, :].flatten(0, 1)

    monkeypatch.setattr(carry_module, "all_to_all_single", fake_all_to_all)

    for rank in range(cp_size):
        exchange = CarryExchange(CarryExchange.Config(ratio=ratio))
        exchange.group = _FakeMesh(rank, cp_size)
        first = rank * slab
        with torch.no_grad():
            local = _carry(
                module,
                exchange,
                x[:, first : first + slab, :],
                positions[..., first : first + slab],
                cu,
            )
        expected = global_pool[:, first // ratio : (first + slab) // ratio, :]
        torch.testing.assert_close(local, expected, rtol=1e-5, atol=1e-6)
        assert local.shape == (1, slab // ratio, _HEAD_DIM)


def test_the_rank_that_starts_the_row_carries_nothing(monkeypatch):
    """Rank 0 still enters the collective, but prepends nothing."""
    from torchtitan_npu.models.common import carry_exchange as carry_module

    calls = []

    def fake_all_to_all(send, *, output_split_sizes, input_split_sizes, group):
        del group
        assert send.size(0) == sum(input_split_sizes)
        calls.append((tuple(input_split_sizes), tuple(output_split_sizes)))
        return send[: sum(output_split_sizes)]

    monkeypatch.setattr(carry_module, "all_to_all_single", fake_all_to_all)
    module = _compressor()
    exchange = CarryExchange(CarryExchange.Config(ratio=_RATIO))
    exchange.group = _FakeMesh(0, 2)
    cu = torch.tensor([0, 16, 32], dtype=torch.int32)
    positions = torch.cat([torch.arange(16), torch.arange(16)]).unsqueeze(0)
    x = torch.randn(1, 16, _DIM)

    local = _carry(module, exchange, x, positions, cu)

    # One collective (the tokens alone -- the block's positions are computed, not sent): it sends
    # one block to the successor and receives nothing, but every rank still enters it.
    assert calls == [((0, 4), (0, 0))]
    assert local.shape == (1, 4, _HEAD_DIM)


def test_every_rank_keeps_the_carry_in_the_autograd_graph(monkeypatch):
    """The rank that receives nothing still concats, so its backward exchange is not pruned.

    Autograd drops a collective whose output no output depends on.  Rank 0's carry is empty, so
    building the augmented stream by branching on it would leave the forward looking identical
    and prune *only that rank's* backward exchange -- a hang the next collective would report as
    a timeout.  The empty concat keeps the graph symmetric.
    """
    from torchtitan_npu.models.common import carry_exchange as carry_module

    module = _compressor()
    slab = 16
    x = torch.randn(1, slab, _DIM, generator=torch.Generator().manual_seed(5)).requires_grad_()
    # A leaf of its own: the slab cannot carry the gradient the collective's edge would.
    carry_source = torch.randn(1, 2 * slab, _DIM).requires_grad_()
    positions = torch.cat([torch.arange(slab), torch.arange(slab)]).unsqueeze(0)
    cu = torch.tensor([0, slab, 2 * slab], dtype=torch.int32)
    exchange = CarryExchange(CarryExchange.Config(ratio=_RATIO))
    exchange.group = _FakeMesh(0, 2)

    def fake_all_to_all(send, *, output_split_sizes, input_split_sizes, group):
        del input_split_sizes, group
        if send.dtype == torch.int32:
            return send[:0]
        # Rank 0 receives nothing, but the reply still has to be a graph-connected tensor.
        return carry_source[:, : sum(output_split_sizes), :].flatten(0, 1)

    monkeypatch.setattr(carry_module, "all_to_all_single", fake_all_to_all)

    out = _carry(module, exchange, x, positions[..., :slab], cu)
    out.square().sum().backward()

    # The empty carry flows back into its own leaf: a ``None`` here is the pruned collective.
    assert carry_source.grad is not None
