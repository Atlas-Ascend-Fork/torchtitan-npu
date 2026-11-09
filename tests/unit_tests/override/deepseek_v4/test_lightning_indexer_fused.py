# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused V4 InnerIndexer: kernel order, teacher edge, SLIKG dispatch.

The NPU kernels are replaced by eager stand-ins with their exact contract (document-local
indices, ``-1`` padding, the teacher arriving on ``topk_scores``' gradient), so these cases
pin the wiring rather than the kernel math: the picks keep the order the kernel reported, the
teacher travels through untouched, and both kernel calls read the metadata the fused attention
precomputed.
"""

import pytest
import torch

from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import (
    AscInnerIndexer,
    kernel_options,
)
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
)

pytestmark = pytest.mark.usefixtures("dsv4")

_TOKENS = 8
_TOPK = 4
_HEADS = 2
_HEAD_DIM = 4
_INDEX_TOP_K = _TOPK
_WINDOW = 8
_GEOMETRY = dict(
    num_heads=2,
    head_dim=8,
    index_num_heads=_HEADS,
    index_head_dim=_HEAD_DIM,
)


def _metadata():
    positions = torch.arange(_TOKENS).unsqueeze(0)
    config = AscCompressedSparseInnerAttention.Config(
        window_size=_WINDOW,
        compress_ratio=4,
        softmax_scale=0.5,
        index_topk=_INDEX_TOP_K,
        **_GEOMETRY,
    )
    return config.build_attention_metadata(positions)


def _indexer():
    return AscInnerIndexer(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))


def _kernel_indices() -> torch.Tensor:
    """A kernel-order selection: the usable picks first, ``-1`` padding at the tail."""
    return torch.tensor(
        [
            [[2, 0, 1, -1]],
            [[3, 1, 0, -1]],
            [[1, 0, 2, -1]],
            [[0, 1, 2, 3]],
            [[3, 2, -1, -1]],
            [[0, 2, -1, -1]],
            [[1, 2, 3, -1]],
            [[2, 0, 1, -1]],
        ],
        dtype=torch.int32,
    )


def _kernel_values() -> torch.Tensor:
    """The scores the kernel reports at those slots, ``-inf`` where it selected nothing.

    A distinct value per slot, so a carrier that dropped or reordered the scores is caught by
    the pairing rather than by the shape.
    """
    indices = _kernel_indices()
    return torch.where(
        indices >= 0,
        torch.arange(indices.numel(), dtype=torch.float32).reshape(indices.shape),
        float("-inf"),
    )


def test_the_selection_is_saved_in_the_kernel_order(monkeypatch, a5_records):
    """The picks and their scores are forwarded as emitted: usable first, ``-1`` at the tail."""
    md = _metadata()
    captured = {}

    def fake_li(q, k, w, topk, *, metadata, return_value, **options):
        captured["metadata"] = metadata
        captured["options"] = options
        return _kernel_indices(), _kernel_values()

    def fake_slig(*, q, k, w, sparse_indices, attn_softmax_l1_norm, metadata, **options):
        captured["indices"] = sparse_indices
        captured["metadata_bwd"] = metadata
        captured["options_bwd"] = options
        captured["teacher"] = attn_softmax_l1_norm
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(w), None

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", fake_li, raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad",
        fake_slig,
        raising=False,
    )

    torch.manual_seed(0)
    idx_q = torch.randn(1, _TOKENS, _HEADS, _HEAD_DIM, requires_grad=True)
    idx_k = torch.randn(1, _TOKENS // 4, _HEAD_DIM, requires_grad=True)
    idx_w = torch.randn(1, _TOKENS, _HEADS, requires_grad=True)
    picks, scores = _indexer()(idx_q, idx_k, idx_w, attention_metadata=md)

    assert picks.shape == (1, _TOKENS, _TOPK) and picks.dtype == torch.int32
    assert scores.shape == picks.shape and scores.dtype == torch.float32
    # The kernel emits the usable picks first and pads the tail, which is the order SLIKG reads
    # and the one the teacher is positioned in, so nothing is reordered here.
    assert torch.equal(picks, _kernel_indices().reshape(1, _TOKENS, _TOPK))
    # The student's scores travel with their slots, in the same order.
    assert scores.dtype == torch.float32
    assert torch.equal(scores.reshape(_TOKENS, _TOPK), _kernel_values().reshape(_TOKENS, _TOPK))
    for row in picks[0]:
        first_pad = (row < 0).nonzero()
        if first_pad.numel():
            assert (row[first_pad[0] :] < 0).all()

    # The forward metadata is the entry's own record; the options are the frame contract.
    assert captured["metadata"] is md.aicpu_metadata.lightning_indexer.fwd
    # The options are the entry's own frame -- the one its keys were compacted onto.
    assert captured["options"] == kernel_options(ratio=md.ratio, q=md.q, cmp_k=md.cmp_k)

    # The value of the carrier is never read: overwrite it with NaN and still drive the edge.
    teacher = torch.zeros_like(scores)
    teacher[..., 0] = 0.1
    teacher[..., 1] = 0.3
    scores.data.fill_(float("nan"))
    scores.backward(teacher)

    # The saved indices are exactly what SMLAG would have been handed, and the teacher reaches
    # SLIKG as the attention's backward scaled it -- nothing is renormalized here.
    assert torch.equal(captured["indices"], picks.reshape(-1, 1, _TOPK))
    assert captured["metadata_bwd"] is md.aicpu_metadata.lightning_indexer.bwd
    assert captured["options_bwd"] == captured["options"]
    torch.testing.assert_close(captured["teacher"], teacher.reshape(-1, 1, _TOPK), rtol=0, atol=0)

    # The kernel's gradients land on the projections the indexer owns.
    for tensor in (idx_q, idx_k, idx_w):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    for row in teacher.reshape(-1, _TOPK):
        assert float(row.abs().sum()) > 0


def test_the_carrier_is_a_function_output(monkeypatch):
    """A caller-made tensor cannot carry the teacher; the returned carrier has a ``grad_fn``."""
    md = _metadata()
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "lightning_indexer",
        lambda *a, **k: (_kernel_indices(), _kernel_values()),
        raising=False,
    )
    idx_q = torch.zeros(1, _TOKENS, _HEADS, _HEAD_DIM, requires_grad=True)
    idx_k = torch.zeros(1, _TOKENS // 4, _HEAD_DIM, requires_grad=True)
    idx_w = torch.zeros(1, _TOKENS, _HEADS, requires_grad=True)
    picks, scores = _indexer()(idx_q, idx_k, idx_w, attention_metadata=md)
    assert scores.requires_grad and scores.grad_fn is not None
    assert not picks.requires_grad


def test_the_override_entry_replaces_the_base_selector():
    """The registry-facing ``asc`` entry derives the fused selector config."""
    from torchtitan_npu.override.deepseek_v4.lightning_indexer import asc

    fused = asc(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))
    assert isinstance(fused, AscInnerIndexer.Config)
    assert fused.index_topk == _INDEX_TOP_K


def test_the_pypto_entry_selects_the_pypto_selector():
    """The PyPTO port is standalone: it implements the selector contract itself."""
    from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
    from torchtitan_npu.override.deepseek_v4.lightning_indexer import pypto
    from torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto import PyPTOInnerIndexer

    fused = pypto(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))
    assert isinstance(fused, PyPTOInnerIndexer.Config)
    # The two ports share the base selector's model contract, not each other's bridge.
    assert issubclass(PyPTOInnerIndexer, InnerIndexer)
    assert not issubclass(PyPTOInnerIndexer, AscInnerIndexer)


@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot-eager"])
def test_pypto_pro_override_dispatches_forward_and_backward(monkeypatch, batch_size, compiled):
    import importlib

    from torchtitan.config import OverrideConfig, apply_overrides

    from torchtitan_npu.models.deepseek_v4 import _debugmodel
    from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
    from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
    from torchtitan_npu.override import a3_workaround
    from torchtitan_npu.override.deepseek_v4.asc_metadata import (
        AicpuGeometry,
        AicpuMetadata,
        AscLightningIndexerMetadata,
        AscSparseAttentionMetadata,
        AscV4Metadata,
    )

    pro = importlib.import_module("torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto_pro")
    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    monkeypatch.setattr(pro, "_IS_A5", True)
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", False)
    monkeypatch.setattr(a3_workaround, "is_a5", lambda: pytest.fail("unexpected runtime device query"))
    model = _debugmodel()
    replacements = apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto_pro"]), model
    )
    selectors = [node for _, node, _, _ in model.traverse(InnerIndexer.Config)]
    assert selectors and len(replacements) == len(selectors)
    assert all(type(node) is pro.PyPTOProInnerIndexer.Config for node in selectors)
    indexer = selectors[0].build()
    assert type(indexer) is pro.PyPTOProInnerIndexer

    tokens, heads, width, topk = 8, 64, 128, 512
    base = DeepSeekV4Metadata.from_positions(torch.arange(tokens).expand(batch_size, -1), ratio=4)
    fwd_record, bwd_record = torch.tensor([11], dtype=torch.int32), torch.tensor([23], dtype=torch.int32)
    metadata = AscV4Metadata(
        ratio=base.ratio,
        q=base.q,
        swa_k=base.swa_k,
        cmp_k=base.cmp_k,
        position_k=base.position_k,
        aicpu_metadata=AicpuMetadata(
            geometry=AicpuGeometry(
                num_heads=16,
                head_dim=512,
                window_size=128,
                index_topk=topk,
                index_num_heads=heads,
                index_head_dim=width,
            ),
            sparse_attn=AscSparseAttentionMetadata(),
            lightning_indexer=AscLightningIndexerMetadata(fwd=fwd_record, bwd=bwd_record),
        ),
    )
    captured = {"li": [], "lig": []}

    def fake_li(
        q,
        k,
        w,
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        cmp_residual_k,
        metadata,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        guard_tensors=None,
        guard_versions=None,
    ):
        captured["li"].append(
            (q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_residual_k, metadata, (layout_q, layout_k, mask_mode, cmp_ratio))
        )
        indices = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32)
        indices[..., 0], indices[..., 2] = 0, 1
        values = torch.full((q.shape[0], 1, topk), float("-inf"), dtype=torch.float32)
        values[..., :2] = 1.0
        return indices, values

    def fake_lig(
        q,
        k,
        w,
        sparse_indices,
        attn_softmax_l1_norm,
        cu_seqlens_q,
        cu_seqlens_k,
        cmp_residual_k,
        metadata,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        guard_tensors=None,
        guard_versions=None,
    ):
        captured["lig"].append(
            (q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_residual_k, metadata, (layout_q, layout_k, mask_mode, cmp_ratio))
        )
        captured["indices"] = sparse_indices
        captured["teacher"] = attn_softmax_l1_norm
        return (
            torch.full_like(q, 2),
            torch.full_like(k, 3, dtype=torch.float32),
            attn_softmax_l1_norm.sum(-1).expand(-1, heads).contiguous(),
            torch.zeros_like(attn_softmax_l1_norm),
        )

    library = torch.library.Library("torchtitan_npu", "IMPL", "CPU")
    try:
        library.impl("pypto_pro_lightning_indexer_indices", fake_li)
        library.impl("pypto_pro_sparse_lightning_indexer_kl_loss_grad", fake_lig)
        idx_q = torch.ones(batch_size, tokens, heads, width, dtype=torch.bfloat16, requires_grad=True)
        idx_k = torch.arange(batch_size * (tokens // 4) * width).reshape(batch_size, tokens // 4, width)
        idx_k = idx_k.bfloat16().requires_grad_()
        idx_w = torch.ones(batch_size, tokens, heads, dtype=torch.bfloat16, requires_grad=True)
        call = torch.compile(indexer, backend="aot_eager", fullgraph=True) if compiled else indexer
        picks, scores = call(idx_q, idx_k, idx_w, attention_metadata=metadata)
        # The port forwards the kernel's order: the fake writes slots 0 and 2.
        expected = torch.full((batch_size, tokens, topk), -1, dtype=torch.int32)
        expected[..., 0], expected[..., 2] = 0, 1
        torch.testing.assert_close(picks, expected, rtol=0, atol=0)
        assert not picks.requires_grad
        assert scores.shape == picks.shape and scores.dtype == torch.float32 and scores.requires_grad
        teacher = torch.zeros_like(scores)
        teacher[..., 0] = torch.arange(1, batch_size * tokens + 1).reshape(batch_size, tokens)
        teacher[..., 1] = 2
        scores.backward(teacher)

        assert len(captured["li"]) == len(captured["lig"]) == 1
        for name, record in (("li", fwd_record), ("lig", bwd_record)):
            q, k, w, cu_q, cu_k, residual, actual_record, options = captured[name][0]
            assert q.shape == (batch_size * tokens, heads, width) and q.dtype == torch.bfloat16
            assert k.shape == (batch_size * (tokens // 4), 1, width) and k.dtype == torch.bfloat16
            assert w.shape == (batch_size * tokens, heads) and w.dtype == torch.float32
            torch.testing.assert_close(k, idx_k.detach().reshape(-1, 1, width), rtol=0, atol=0)
            torch.testing.assert_close(cu_q, base.q.cu_seqlens, rtol=0, atol=0)
            torch.testing.assert_close(cu_k, base.cmp_k.cu_seqlens, rtol=0, atol=0)
            torch.testing.assert_close(residual, base.cmp_k.residual, rtol=0, atol=0)
            torch.testing.assert_close(actual_record, record, rtol=0, atol=0)
            assert options == ("TND", "TND", 3, 4)
        torch.testing.assert_close(captured["indices"], expected.reshape(-1, 1, topk), rtol=0, atol=0)
        # The indexer no longer normalizes: the teacher arrives scaled from the attention's
        # backward and goes to the kernel as handed over.
        torch.testing.assert_close(captured["teacher"], teacher.reshape(-1, 1, topk), rtol=0, atol=0)
        torch.testing.assert_close(idx_q.grad, torch.full_like(idx_q, 2), rtol=0, atol=0)
        torch.testing.assert_close(idx_k.grad, torch.full_like(idx_k, 3), rtol=0, atol=0)
        expected_dw = torch.arange(1, batch_size * tokens + 1) + 2
        expected_dw = expected_dw.reshape(batch_size, tokens, 1).expand_as(idx_w).bfloat16()
        torch.testing.assert_close(idx_w.grad, expected_dw, rtol=0, atol=0)
    finally:
        library._destroy()
        torch._dynamo.reset()
