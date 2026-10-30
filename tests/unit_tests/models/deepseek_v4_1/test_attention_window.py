# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The window-only sparse core: no compressed pool, and the pool frame stays unread.

A layer without a compressed KV hands ``cu_seqlens`` to ``gather_attn``'s ``cu_seqlens_k``
because the two offsets travel together, not because the pool frame means anything there.
These tests pin both halves of that statement: the window result matches an explicit
reference, and another document's keys cannot reach the query.
"""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1 import attention
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2


def _window_metadata(bounds: list[int]):
    """The row's own metadata, which is all the window-only branch reads.

    A window-only layer is ``compress_ratio = 0``, and the backend that builds the row folds
    ``cu_seq_k`` there to the row's own boundaries: the core addresses the row with these
    offsets and never reaches for a pooled axis, so this is the whole contract it consumes.
    """
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)])
    # Only the row matters here: the builder reads the positions, not this geometry.
    config = CompressedSparseInnerAttention2.Config(window_size=8, softmax_scale=0.5, compress_ratio=0)
    metadata = config.build_attention_metadata(positions)
    # The window-only geometry has no pool, so its key grid is the row's own -- the property
    # that makes passing both offsets to ``gather_attn`` a no-op here.
    torch.testing.assert_close(metadata.cu_seq_k, metadata.cu_seq_q, rtol=0, atol=0)
    return metadata


def _dense_window_reference(q, kv, sink, bounds, window, scale):
    """Attend over the ``window`` tokens ending at the query, clipped to its own document.

    Written out longhand, independent of the core's helpers: the sink is one more logit in
    the denominator and contributes nothing to the value.
    """
    tokens, heads, dim = q.shape
    out = torch.zeros(tokens, heads, dim, dtype=q.dtype)
    for begin, end in pairwise(bounds):
        for position in range(begin, end):
            first = max(begin, position - window + 1)
            keys = kv[first : position + 1]
            scores = (q[position] @ keys.transpose(0, 1)) * scale
            probs = torch.cat((scores, sink[:, None]), dim=1).softmax(dim=1)
            out[position] = probs[:, : keys.shape[0]] @ keys
    return out


@pytest.mark.parametrize("window", [1, 2, 8], ids=["self-only", "partial", "whole-document"])
def test_window_only_layer_matches_an_explicit_reference(window):
    bounds = [0, 4, 10, 12]  # three documents of 4, 6 and 2 tokens
    metadata = _window_metadata(bounds)
    scale, heads, head_dim = 0.5, 2, 4
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(bounds[-1], heads, head_dim, generator=generator)
    kv = torch.randn(bounds[-1], head_dim, generator=generator)
    sink = torch.randn(heads, generator=generator)

    core = attention.CompressedSparseInnerAttention2.Config(
        window_size=window,
        softmax_scale=scale,
        compress_ratio=0,
    ).build()
    # ``cmp_k``/``topk_indices`` are a pair; omitting both is the window-only layer.
    out = core(q, kv, sink, metadata, cmp_k=None, topk_indices=None)

    torch.testing.assert_close(out, _dense_window_reference(q, kv, sink, bounds, window, scale), rtol=1e-5, atol=1e-6)


def test_window_only_layer_does_not_leak_across_documents():
    """A query's window is clipped at its own document, so other documents cannot reach it."""
    bounds = [0, 4, 10]
    metadata = _window_metadata(bounds)
    window, scale, heads, head_dim = 8, 0.5, 2, 4  # window >= every document: nothing is truncated
    generator = torch.Generator().manual_seed(1)
    q = torch.randn(bounds[-1], heads, head_dim, generator=generator)
    kv = torch.randn(bounds[-1], head_dim, generator=generator)
    sink = torch.randn(heads, generator=generator)

    core = attention.CompressedSparseInnerAttention2.Config(
        window_size=window,
        softmax_scale=scale,
        compress_ratio=0,
    ).build()
    out = core(q, kv, sink, metadata, cmp_k=None, topk_indices=None)

    bumped = kv.clone()
    bumped[bounds[1] :] += 10.0  # disturb every later document
    out_bumped = core(q, bumped, sink, metadata, cmp_k=None, topk_indices=None)

    first_document = slice(0, bounds[1])
    torch.testing.assert_close(out[first_document], out_bumped[first_document], rtol=0, atol=0)


@pytest.mark.parametrize("ratio", (1, 2), ids=("uncompressed-pool", "compressed-pool"))
def test_sparse_attention_outputs_and_gradients_match_dense_document_reference(ratio):
    generator = torch.Generator().manual_seed(71)
    bounds = [0, 4, 8]
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    config = CompressedSparseInnerAttention2.Config(window_size=2, softmax_scale=0.5, compress_ratio=ratio)
    core = config.build()
    metadata = config.build_attention_metadata(positions)
    q = torch.randn(8, 2, 4, generator=generator, requires_grad=True)
    local = torch.randn(8, 4, generator=generator, requires_grad=True)
    compressed = torch.randn(8 // ratio, 4, generator=generator, requires_grad=True)
    sink = torch.randn(2, generator=generator, requires_grad=True)
    picks = torch.full((8, 2), -1, dtype=torch.long)
    for begin, end in pairwise(bounds):
        for token in range(begin, end):
            complete = (token - begin + 1) // ratio
            visible = list(range(max(0, complete - 2), complete))[::-1]
            picks[token, : len(visible)] = torch.tensor(visible, dtype=torch.long)

    actual = core(q, local, sink, metadata, cmp_k=compressed, topk_indices=picks)
    reference_inputs = [x.detach().double().requires_grad_() for x in (q, local, compressed, sink)]
    rq, rl, rc, rs = reference_inputs
    outputs = []
    for begin, end in pairwise(bounds):
        for token in range(begin, end):
            selected = [begin // ratio + int(i) for i in picks[token] if i >= 0]
            keys = torch.cat((rl[max(begin, token - 1) : token + 1], rc[selected]))
            scores = rq[token] @ keys.t() * 0.5
            probabilities = torch.cat((scores, rs[:, None]), dim=-1).softmax(-1)
            outputs.append(probabilities[:, :-1] @ keys)
    expected = torch.stack(outputs)
    upstream = torch.randn(actual.shape, generator=generator)
    gradients = torch.autograd.grad(actual, (q, local, compressed, sink), upstream)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream.double())

    torch.testing.assert_close(actual, expected.float(), rtol=2e-5, atol=2e-6)
    for gradient, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference.float(), rtol=2e-5, atol=2e-6)
