# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed teacher plumbing against a dense KL oracle, with CPU kernel substitutes."""

import pytest
import torch

from torchtitan_npu.models.deepseek_v4.cp_layout import DocChunkCPLoadBalancer
from torchtitan_npu.models.deepseek_v4.indexer_loss import IndexerLoss
from torchtitan_npu.override.deepseek_v4.lightning_indexer import ascendc as v4_li
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc as v4_attn
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as v41_li
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as v41_attn
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


@pytest.mark.parametrize("generation", ["v4", "v41"])
@pytest.mark.parametrize("cp_size", [1, 2], ids=["cp1", "cp2"])
def test_packed_teacher_preserves_slots_global_scale_and_compact_gradients(monkeypatch, generation, cp_size):
    # The only substitutes are device kernels. The metadata, gathers, two wrappers and
    # both autograd Functions are real. This does not exercise collective communication.
    old_denominator = AuxLoss._step_denominator
    monkeypatch.setattr(AuxLoss, "_step_denominator", old_denominator)
    AuxLoss.set_step_denominator(torch.tensor(32.0))
    attention_type = v4_attn.AscCompressedSparseInnerAttention if generation == "v4" else v41_attn.AscV41SparseAttention
    ratio = 4 if generation == "v4" else 2
    pool_per_doc = 16 // ratio
    config = attention_type.Config(
        window_size=2,
        compress_ratio=ratio,
        softmax_scale=0.5,
        index_topk=3,
        num_heads=1,
        head_dim=1,
        index_num_heads=1,
        index_head_dim=1,
        aux_loss=IndexerLoss.Config(coeff=0.5),
    )
    for name in (
        "lightning_indexer_metadata",
        "sparse_flash_mla_metadata",
        "sparse_flash_mla_grad_metadata",
        "sparse_lightning_indexer_kl_loss_grad_metadata",
    ):
        monkeypatch.setattr(torch.ops.cann_ops_transformer, name, lambda *a, **kw: torch.zeros(1, dtype=torch.int32))
    positions = torch.cat([torch.arange(16), torch.arange(16)]).unsqueeze(0)
    metadata = config.build_attention_metadata(positions)
    if cp_size > 1:
        permutation = (
            DocChunkCPLoadBalancer.Config()
            .build(
                seq_len=32,
                cp_size=cp_size,
                attention_metadata=metadata,
            )
            .generate_permutation()
        )
        local = metadata.per_doc_cp_metadata(
            permutation=permutation,
            cp_size=cp_size,
            cp_rank=0,
            window_size=config.geometry.window_size,
        )
        metadata = type(metadata).build_aicpu_metadata(local, geometry=config.geometry)
    tokens_per_doc = 16 // cp_size
    total = tokens_per_doc * 2
    selection = torch.full((total, 1, 3), -1, dtype=torch.int32)
    # Preserve the kernel's own slot order. Each document's last local query can
    # see both keys even on the first CP rank; unused slots contain tiling residue.
    selection[tokens_per_doc - 1, 0] = torch.tensor([1, 0, -1])
    selection[-1, 0] = torch.tensor([1, 0, -1])
    teacher = torch.full((total, 1, 3), 7.0)
    teacher[tokens_per_doc - 1, 0, :2] = torch.tensor([0.25, 0.75])
    teacher[-1, 0, :2] = torch.tensor([0.25, 0.75])
    seen = {}

    def select(q, k, w, topk, **kwargs):
        seen["keys"] = k.detach().clone()
        return selection.clone(), torch.zeros(total, 1, topk)

    def attention_forward(q, **kwargs):
        torch.testing.assert_close(kwargs["cmp_sparse_indices"], selection)
        return q.clone(), torch.zeros(1, total, 1)

    def attention_backward(q, *args, **kwargs):
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            teacher,
        )

    def indexer_backward(q, k, w, sparse_indices, attn_softmax_l1_norm, **kwargs):
        # Closed-form SLIKG derivative for positive scalar q/k/w: dI=Z*softmax(I)-p.
        # The independent expected below differentiates dense KL, not this expression.
        seen["teacher"] = attn_softmax_l1_norm.detach().clone()
        dq, dk, dw = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(w)
        cu_q, cu_k = kwargs["cu_seqlens_q"].tolist(), kwargs["cu_seqlens_k"].tolist()
        for doc in range(2):
            row = cu_q[doc + 1] - 1
            ids = sparse_indices[row, 0, :2].long() + cu_k[doc]
            p = attn_softmax_l1_norm[row, 0, :2]
            keys = k[ids, 0, 0]
            logits = q[row, 0, 0] * keys * w[row, 0]
            delta = p.sum() * logits.softmax(0) - p
            dq[row, 0, 0] = (delta * keys * w[row, 0]).sum()
            dw[row, 0] = (delta * keys * q[row, 0, 0]).sum()
            dk[:, 0, 0].index_add_(0, ids, delta * q[row, 0, 0] * w[row, 0])
        return dq, dk, dw, None

    for name, function in (
        ("lightning_indexer", select),
        ("sparse_flash_mla", attention_forward),
        ("sparse_flash_mla_grad", attention_backward),
        ("sparse_lightning_indexer_kl_loss_grad", indexer_backward),
    ):
        monkeypatch.setattr(torch.ops.cann_ops_transformer, name, function)
    selector = (
        v4_li.AscInnerIndexer.Config(index_topk=3).build()
        if generation == "v4"
        else v41_li.AscInnerIndexer.Config(
            mode="full",
            compress_ratio=ratio,
            num_index_heads=1,
            index_head_dim=1,
            index_topk=3,
        ).build()
    )
    idx_q = torch.linspace(0.25, 1.25, total).reshape(1, total, 1, 1).requires_grad_()
    idx_k = torch.arange(1, pool_per_doc * 2 + 1, dtype=torch.float32).reshape(1, pool_per_doc * 2, 1).requires_grad_()
    idx_w = torch.linspace(0.5, 1.5, total).reshape(1, total, 1).requires_grad_()
    result = selector(idx_q, idx_k, idx_w, metadata)
    indices, scores = result[:2]
    attention = config.build()
    q = torch.ones(1, total, 1, 1)
    output = attention(
        q,
        torch.ones(1, 32, 1),
        torch.zeros(1),
        attention_metadata=metadata,
        cmp_k=torch.ones(1, pool_per_doc * 2, 1),
        topk_indices=indices,
        topk_scores=scores,
    )
    output.sum().backward()

    # Dense KL uses original packed-row key coordinates, independent of the compact
    # metadata. Every rank uses 32 global tokens, never its local query count.
    ref_q = idx_q.detach().clone().requires_grad_()
    ref_k = idx_k.detach().clone().requires_grad_()
    ref_w = idx_w.detach().clone().requires_grad_()
    loss = torch.tensor(0.0)
    p = torch.tensor([0.25, 0.75])
    for doc in range(2):
        row = (doc + 1) * tokens_per_doc - 1
        keys = ref_k[0, torch.tensor([doc * pool_per_doc + 1, doc * pool_per_doc]), 0]
        logits = ref_q[0, row, 0, 0] * keys * ref_w[0, row, 0]
        loss = loss + (p * (p.log() - logits.log_softmax(0))).sum() * (0.5 / 32)
    loss.backward()
    for actual, expected in ((idx_q, ref_q), (idx_k, ref_k), (idx_w, ref_w)):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(seen["teacher"], teacher * (0.5 / 32), rtol=0, atol=0)
    # The wrapper gathers the index keys to the row.  A5 reads that row; arch22 compacts it onto
    # the frame this rank's entry may reach first, so the selector sees this rank's slab.
    if _IS_A5:
        expected_keys = list(range(1, pool_per_doc * 2 + 1))
    else:
        expected_keys = [doc * pool_per_doc + offset + 1 for doc in range(2) for offset in range(pool_per_doc // cp_size)]
    assert seen["keys"].flatten().tolist() == expected_keys
