# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real A3 packed sparse teacher kernels and SLIKG gradients against dense KL.

Uses one visible NPU; requires the container's CANN transformer operators.
BF16 attention geometry is 16x512, indexer geometry 64x128, topk=2048,
and two documents of 128 tokens. No distributed collectives are exercised.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from torchtitan_npu.models.deepseek_v4.indexer_loss import IndexerLoss
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


def test_packed_teacher_kernels_match_dense_kl_gradients(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("requires one visible A3 NPU")
    from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscInnerIndexer
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscCompressedSparseInnerAttention

    ratio = 4
    monkeypatch.setattr(AuxLoss, "_step_denominator", AuxLoss._step_denominator)

    generator = torch.Generator(device="npu").manual_seed(1042)
    AuxLoss.set_step_denominator(torch.tensor(256.0, device="npu"))
    config = AscCompressedSparseInnerAttention.Config(
        window_size=128,
        compress_ratio=ratio,
        softmax_scale=512**-0.5,
        index_topk=2048,
        num_heads=16,
        head_dim=512,
        index_num_heads=64,
        index_head_dim=128,
        aux_loss=IndexerLoss.Config(coeff=0.5),
    )
    positions = torch.cat([torch.arange(128), torch.arange(128)]).unsqueeze(0).to("npu")
    meta = config.build_attention_metadata(positions)
    selector_config = AscInnerIndexer.Config(index_topk=2048)
    selector = selector_config.build().to("npu")
    q = (torch.randn(1, 256, 64, 128, device="npu", dtype=torch.bfloat16, generator=generator) * 0.05).requires_grad_()
    k = (
        torch.randn(1, 256 // ratio, 128, device="npu", dtype=torch.bfloat16, generator=generator) * 0.05
    ).requires_grad_()
    w = torch.ones(1, 256, 64, device="npu", dtype=torch.float32, requires_grad=True)
    indices, scores = selector(q, k, w, meta)
    scores.retain_grad()
    attn = config.build().to("npu")
    mainq = torch.randn(1, 256, 16, 512, device="npu", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    swa = torch.randn(1, 256, 512, device="npu", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    cmp = torch.randn(1, 256 // ratio, 512, device="npu", dtype=torch.bfloat16, generator=generator, requires_grad=True)
    out = attn(
        mainq,
        swa,
        torch.zeros(16, device="npu"),
        attention_metadata=meta,
        cmp_k=cmp,
        topk_indices=indices,
        topk_scores=scores,
    )
    out.float().square().mean().backward()
    torch.npu.synchronize()
    for name, x in [
        ("idx_q", q),
        ("idx_k", k),
        ("idx_w", w),
        ("carrier", scores),
        ("mainq", mainq),
        ("swa", swa),
        ("cmp", cmp),
    ]:
        assert x.grad is not None, name
        assert torch.isfinite(x.grad).all().item(), name
        print(name, "grad_absmax", x.grad.abs().max().item())
    print("NPU_SPARSE_TEACHER_SMOKE_PASS", out.shape)
    refq = q.detach().float().cpu().requires_grad_()
    refk = k.detach().float().cpu().requires_grad_()
    refw = w.detach().cpu().requires_grad_()
    ids = indices.detach().cpu()[0]
    p = scores.grad.detach().cpu()[0]
    loss = torch.tensor(0.0)
    for d in range(2):
        begin, end = d * 128, (d + 1) * 128
        dots = torch.einsum("thd,kd->thk", refq[0, begin:end], refk[0, d * (128 // ratio) : (d + 1) * (128 // ratio)])
        logits = (dots.relu() * refw[0, begin:end, :, None]).sum(1)
        visible = torch.zeros(128, 128 // ratio, dtype=torch.bool)
        masses = torch.zeros(128, 128 // ratio)
        for row in range(128):
            ok = ids[begin + row] >= 0
            visible[row, ids[begin + row, ok].long()] = True
            masses[row].index_add_(0, ids[begin + row, ok].long(), p[begin + row, ok])
        log_probs = logits.masked_fill(~visible, float("-inf")).log_softmax(-1)
        loss = loss - torch.where(visible, masses * log_probs, 0.0).sum()
    loss.backward()
    for name, real, expected in [("idx_q", q, refq), ("idx_k", k, refk), ("idx_w", w, refw)]:
        a, b = real.grad.detach().float().cpu(), expected.grad
        print(
            name, "oracle_maxabs", (a - b).abs().max().item(), "oracle_relative_l2", ((a - b).norm() / b.norm()).item()
        )
        torch.testing.assert_close(a, b, rtol=0.03, atol=2e-6)
    print("NPU_DENSE_KL_GRADIENT_ORACLE_PASS")
