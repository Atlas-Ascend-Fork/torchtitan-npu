# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.cp_layout import DOC_ZIGZAG_LAYOUT
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override.deepseek_v4 import asc_metadata
from torchtitan_npu.override.deepseek_v4.sparse_attn import asc as v4_attention
from torchtitan_npu.patches.torchtitan.distributed.context_parallel import build_load_balancer
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

_LENGTHS = (32, 48, 16)


def _positions():
    return torch.cat([torch.arange(length) for length in _LENGTHS]).unsqueeze(0)


def _held_tokens(cp_size, rank):
    # Explicitly enumerate each original token and its piece owner, heads before tails.
    heads, tails = [], []
    start = 0
    for length in _LENGTHS:
        for offset in range(length):
            piece = offset // (length // (2 * cp_size))
            if piece == rank:
                heads.append(start + offset)
            if piece == 2 * cp_size - rank - 1:
                tails.append(start + offset)
        start += length
    return torch.tensor(heads + tails)


def _permutation(row, cp_size):
    return build_load_balancer(
        DOC_ZIGZAG_LAYOUT,
        seq_len=sum(_LENGTHS),
        cp_size=cp_size,
        attention_metadata=row,
    ).generate_permutation()


def _metadata(config, *, cp_size, rank):
    row = config.build_attention_metadata(_positions())
    mesh = SimpleNamespace(size=lambda: cp_size, get_local_rank=lambda: rank)
    with cp_shard_context_set(cp_mesh=mesh, load_balancer=DOC_ZIGZAG_LAYOUT):
        return type(row).prepare_cp_metadata(row, permutation=_permutation(row, cp_size))


def _port(ratio):
    return v4_attention(
        CompressedSparseInnerAttention.Config(window_size=8, compress_ratio=ratio, softmax_scale=0.5, index_topk=2),
        num_heads=2, head_dim=4, index_num_heads=2, index_head_dim=4,
    )


@pytest.mark.parametrize("cp_size", [2, 4], ids=["cp2", "cp4"])
def test_zigzag_covers_every_token_and_balances_causal_work(cp_size):
    permutation = _permutation(DeepSeekV4Metadata.from_positions(_positions(), 1), cp_size)
    positions = _positions().squeeze(0)
    work = []
    for rank in range(cp_size):
        held = _held_tokens(cp_size, rank)
        assert torch.equal(permutation.reshape(cp_size, -1)[rank], held)
        work.append(int((positions[held] + 1).sum()))
    assert torch.equal(permutation.flatten().sort().values, torch.arange(sum(_LENGTHS)))
    assert len(set(work)) == 1


@pytest.mark.parametrize("a5", [True, False], ids=["a5-seqused", "a3-prefix"])
@pytest.mark.parametrize("ratio", [0, 4], ids=["swa", "csa"])
def test_query_frames_preserve_global_documents_and_local_swa_windows(monkeypatch, a5, ratio):
    monkeypatch.setattr(asc_metadata, "_IS_A5", a5)
    for rank in range(2):
        entry = _metadata(_port(ratio), cp_size=2, rank=rank)
        assert entry.swa_k.cu_seqlens.tolist() == [0, 32, 80, 96]
        assert torch.equal(entry.position_k, _positions()[:, rank * 48 : (rank + 1) * 48])
        head, tail = entry.head, entry.tail
        assert head.q.cu_seqlens.tolist() == tail.q.cu_seqlens.tolist() == [0, 8, 20, 24]
        for half, chunk in enumerate((head, tail)):
            positions = _positions()[0, _held_tokens(2, rank)][half * 24 : (half + 1) * 24]
            starts, ends = positions[[0, 8, 20]], positions[[7, 19, 23]] + 1
            lengths = torch.tensor([8, 12, 4]) + starts.clamp(max=7)
            expected_cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            assert chunk.cu_seqlens.tolist() == [0, 32, 80, 96]
            assert torch.equal(chunk.swa_k.cu_seqlens, expected_cu)
            assert torch.equal(chunk.swa_k.seqused, lengths)
            if a5:
                assert chunk.a3_workaround is None
            else:
                assert chunk.a3_workaround.swa_k_gather is None
            if ratio:
                assert torch.equal(chunk.cmp_k.seqused * ratio, ends)


@pytest.mark.parametrize("length", [512, 768], ids=["aligned-hca128", "misaligned-hca128"])
def test_hca128_checks_two_cp_compression_alignment(monkeypatch, length):
    monkeypatch.setattr(asc_metadata, "_IS_A5", True)
    config = _port(128)
    row = config.build_attention_metadata(torch.arange(length).unsqueeze(0))
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0)
    permutation = build_load_balancer(
        DOC_ZIGZAG_LAYOUT,
        seq_len=length,
        cp_size=2,
        attention_metadata=row,
    ).generate_permutation()
    with cp_shard_context_set(cp_mesh=mesh, load_balancer=DOC_ZIGZAG_LAYOUT):
        if length == 768:
            with pytest.raises(ValueError, match="per_doc_alignment"):
                type(row).prepare_cp_metadata(row, permutation=permutation)
        else:
            entry = type(row).prepare_cp_metadata(row, permutation=permutation)
            head, tail = entry.head, entry.tail
            assert head.cmp_k.cu_seqlens.tolist() == tail.cmp_k.cu_seqlens.tolist() == [0, 4]
            assert head.cmp_k.seqused.tolist() == [1]
            assert tail.cmp_k.seqused.tolist() == [4]
