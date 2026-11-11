# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized CP metadata follows query rows independently of the restored KV shard."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchao_npu")

from torchao_npu.quantized_modules.deepseek_v4_1 import attention_metadata as provider

from torchtitan_npu.extensions.adaptor.torchao.components.deepseek_v4_1 import _V41QuantAttentionMetadata
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set


def test_zigzag_candidate_lengths_follow_half_queries_not_kv_positions(monkeypatch):
    def build(*, ratio, q, swa_k, cmp_k, num_rows, **numbers):
        lengths = provider.candidate_sparse_length(
            ratio=ratio,
            q=q,
            cmp_k=cmp_k,
            num_rows=num_rows,
            topk_blocks=numbers["topk_blocks"],
            block_size=numbers["block_size"],
        )
        return SimpleNamespace(lightning_indexer=SimpleNamespace(sparse_length=lengths))

    monkeypatch.setattr(provider, "build_attention_metadata", build)
    positions = torch.arange(8).repeat(2).unsqueeze(0)
    global_entry = _V41QuantAttentionMetadata.build_aicpu_metadata(
        DeepSeekV4Metadata.from_positions(positions, ratio=2),
        quant_numbers={"topk_blocks": 2, "block_size": 2, "window_size": 3},
    )
    permutation = torch.tensor([[0, 1, 8, 9, 6, 7, 14, 15, 2, 3, 10, 11, 4, 5, 12, 13]])
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0)

    with cp_shard_context_set(cp_mesh=mesh, load_balancer="doc_zigzag"):
        local = _V41QuantAttentionMetadata.prepare_cp_metadata(global_entry, permutation=permutation)

    assert local.position_k.numel() == 8
    head, tail = local.head, local.tail
    torch.testing.assert_close(head.q.cu_seqlens, torch.tensor([0, 2, 4], dtype=torch.int32))
    for entry, expected, window in [(head, [0, 1, 0, 1], [0, 2, 4]), (tail, [2, 2, 2, 2], [0, 4, 8])]:
        actual = entry.aicpu_metadata.lightning_indexer.sparse_length.flatten()
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))
        torch.testing.assert_close(entry.swa_k.cu_seqlens, torch.tensor(window, dtype=torch.int32))
