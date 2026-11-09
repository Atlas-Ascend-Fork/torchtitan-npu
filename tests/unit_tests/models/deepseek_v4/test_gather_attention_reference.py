# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sparse core's own contract: the slots it addresses, and the exit it reserves.

The core reads only the packed offsets the metadata carries, so these cases run it from the
common metadata and spell the rules out again from that grid.  Its numerics against the eager
oracle live in ``test_dsv4.py``.
"""

from itertools import pairwise

import pytest
import torch

pytestmark = pytest.mark.usefixtures("dsv4")

HD = 16
# The metadata is built with the model's full ratio stack (0, 4, 128), so every
# document is a whole number of 128-token blocks (the loader's alignment): four
# one-block documents, and two long three-block documents.
_SHORT = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32)
_LONG = torch.tensor([0, 384, 768], dtype=torch.int32)
_RATIOS = (0, 4, 128)
_WINDOW, _SCALE, _TOPK = 8, 0.25, 8


def _metadata(dsv4, cu):
    """One entry per geometry, the way each layer's own config builds it."""
    positions = torch.cat([torch.arange(end - start) for start, end in pairwise(cu.tolist())]).unsqueeze(0)
    return {ratio: dsv4.metadata.DeepSeekV4Metadata.from_positions(positions, ratio) for ratio in _RATIOS}


def _core(dsv4, ratio, *, aux_loss=None):
    return dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=_WINDOW,
            compress_ratio=ratio,
            softmax_scale=_SCALE,
            index_topk=_TOPK,
            aux_loss=aux_loss,
        )
    )


def _inputs(ratio, cu, seed=0):
    """Query, window keys, compressed pool, and the sink."""
    torch.manual_seed(seed)
    seqlen, heads = int(cu[-1]), 3
    n_cmp = 0 if ratio == 0 else seqlen // ratio
    q = torch.randn(1, seqlen, heads, HD)
    swa_k = torch.randn(1, seqlen, HD)
    cmp_k = None if n_cmp == 0 else torch.randn(1, n_cmp, HD)
    return q, swa_k, cmp_k, torch.randn(heads)


def _visible_local_slots(dsv4, metadata, ratio, seqlen, n_cmp):
    """The HCA rule spelled out again: own document, whole groups, causally complete."""
    entry = metadata[ratio]
    # The pool grid is whichever frame addresses the keys this ratio pools: the compressed
    # frame when the geometry has one, the row's own otherwise.
    starts = entry.swa_k.cu_seqlens if entry.cmp_k is None else entry.cmp_k.cu_seqlens
    row = entry.swa_k.cu_seqlens
    documents = torch.searchsorted(row[1:], torch.arange(seqlen), right=True)
    doc_start = starts[documents]
    local = torch.arange(seqlen) - row[documents]
    ends = torch.minimum(starts[documents + 1], doc_start + (local + 1) // ratio)
    slots = torch.arange(n_cmp).unsqueeze(0)
    visible = (slots >= doc_start.unsqueeze(1)) & (slots < ends.unsqueeze(1))
    return torch.where(visible, slots.expand(seqlen, -1) - doc_start.unsqueeze(1), -1).unsqueeze(0)


def test_the_core_reads_only_the_packed_offsets(dsv4):
    """No reference tier and no block mask: the common metadata is the whole input."""
    common = _metadata(dsv4, _SHORT)
    assert not hasattr(common, "reference")
    seqlen = int(_SHORT[-1])
    q, swa_k, cmp_k, sink = _inputs(4, _SHORT)
    torch.manual_seed(2)
    picks = dsv4.compressor.select_compressed_slots(
        torch.randn(1, seqlen, 4, 8),
        torch.randn(1, seqlen // 4, 8),
        torch.randn(1, seqlen, 4),
        common[4].swa_k.cu_seqlens,
        common[4].cmp_k.cu_seqlens,
        _TOPK,
    )

    with torch.no_grad():
        out = _core(dsv4, 4)(q, swa_k, sink, cmp_k=cmp_k, topk_indices=picks, attention_metadata=common[4])

    assert out.shape == (1, seqlen, 3, HD) and torch.isfinite(out).all()


def test_hca_slots_stay_inside_the_query_document(dsv4):
    """Ratio 128 reaches its whole visible prefix, numbered inside its own document.

    The selection is built in the core's forward (there is no selector to ask), so this pins
    that method against the rule spelled out independently below.
    """
    cu = _LONG
    metadata = _metadata(dsv4, cu)
    seqlen, ratio = int(cu[-1]), 128
    n_cmp = seqlen // ratio

    entry = metadata[ratio]
    core = _core(dsv4, ratio)
    q = torch.zeros(1, seqlen, 1, HD)
    cmp_k = torch.zeros(1, n_cmp, HD)
    selection = core.make_hca_indices(q, cmp_k, entry.q.cu_seqlens, entry.cmp_k.cu_seqlens)
    assert selection.shape == (1, seqlen, n_cmp)
    assert torch.equal(selection, _visible_local_slots(dsv4, metadata, ratio, seqlen, n_cmp))


def test_the_reference_core_ignores_the_selector_scores(dsv4):
    """``topk_scores`` is part of the forward contract but never a reference input."""
    common = _metadata(dsv4, _SHORT)
    seqlen = int(_SHORT[-1])
    q, swa_k, cmp_k, sink = _inputs(4, _SHORT)
    torch.manual_seed(3)
    picks = dsv4.compressor.select_compressed_slots(
        torch.randn(1, seqlen, 4, 8),
        torch.randn(1, seqlen // 4, 8),
        torch.randn(1, seqlen, 4),
        common[4].swa_k.cu_seqlens,
        common[4].cmp_k.cu_seqlens,
        _TOPK,
    )
    scores = torch.randn(1, seqlen, picks.shape[-1], requires_grad=True)

    with torch.no_grad():
        without = _core(dsv4, 4)(q, swa_k, sink, cmp_k=cmp_k, topk_indices=picks, attention_metadata=common[4])
        with_scores = _core(dsv4, 4)(
            q,
            swa_k,
            sink,
            cmp_k=cmp_k,
            topk_indices=picks,
            topk_scores=scores,
            attention_metadata=common[4],
        )

    # Bitwise identical, so the scores never reach the reference gather.
    assert torch.equal(without, with_scores)
    assert scores.grad is None

def test_the_objective_produces_the_reference_indexers_gradient(dsv4):
    """``aux_loss`` gives the selection a gradient; nothing else in the core does.

    The student side of the objective is the scores the indexer emitted, so this pins the
    reference path's whole training edge in one place: it exists, it is scaled by ``coeff`` over
    the step's token count, it leaves the attention output untouched, and the slots the
    selection left unused neither carry a gradient nor influence the rest.
    """
    from torchtitan_npu.models.deepseek_v4.indexer_loss import IndexerLoss
    from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

    common = _metadata(dsv4, _SHORT)
    seqlen = int(_SHORT[-1])
    q, swa_k, cmp_k, sink = _inputs(4, _SHORT)
    torch.manual_seed(3)
    picks = dsv4.compressor.select_compressed_slots(
        torch.randn(1, seqlen, 4, 8),
        torch.randn(1, seqlen // 4, 8),
        torch.randn(1, seqlen, 4),
        common[4].swa_k.cu_seqlens,
        common[4].cmp_k.cu_seqlens,
        _TOPK,
    )
    # Two unused slots, and the scores the selector emits there: ``-inf``, which is what keeps
    # them out of the student's softmax.
    picks[:, :, -2:] = -1
    torch.manual_seed(7)
    base = torch.randn(1, seqlen, picks.shape[-1])
    base[picks < 0] = float("-inf")

    def run(*, coeff, tokens):
        AuxLoss.set_step_denominator(torch.tensor(float(tokens)))
        carried = base.clone().requires_grad_(True)
        out = _core(dsv4, 4, aux_loss=IndexerLoss.Config(coeff=coeff))(
            q,
            swa_k,
            sink,
            cmp_k=cmp_k,
            topk_indices=picks,
            topk_scores=carried,
            attention_metadata=common[4],
        )
        out.sum().backward()
        return out.detach(), carried.grad

    with torch.no_grad():
        plain = _core(dsv4, 4)(q, swa_k, sink, cmp_k=cmp_k, topk_indices=picks, attention_metadata=common[4])

    trained, grad = run(coeff=1.0, tokens=8)
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "the objective must reach the selection"
    # An identity on the attention output: the objective only adds the carrier's edge.
    assert torch.equal(trained, plain)

    # ``coeff`` scales the gradient, the step's token count divides it.
    _, grad_doubled = run(coeff=2.0, tokens=8)
    torch.testing.assert_close(grad_doubled, grad * 2)
    _, grad_halved = run(coeff=1.0, tokens=16)
    torch.testing.assert_close(grad_halved, grad / 2)

    # The slots the selection left unused hold no mass and take no gradient.
    unused = picks < 0
    assert torch.equal(grad[unused], torch.zeros_like(grad[unused]))
