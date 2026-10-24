# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchtitan_npu.models.deepseek_v4_1.vision.encoder import VisionAttention, _vision_rope_batch

pytestmark = pytest.mark.cpu


def _dense_reference(module, x, cos, sin, lengths):
    """Per-image dense attention, excluding padded queries and keys entirely."""

    def rotate(tensor, image_cos, image_sin):
        real, imag = tensor.chunk(2, dim=-1)
        return torch.cat((real * image_cos - imag * image_sin, imag * image_cos + real * image_sin), dim=-1)

    outputs = []
    for index, length in enumerate(lengths):
        qkv = F.linear(x[index], module.wqkv.weight, module.wqkv.bias)
        q, k, v = qkv.reshape(x.shape[1], 3, module.num_heads, module.head_dim).unbind(1)
        q, k = rotate(q, cos[index], sin[index])[:length], rotate(k, cos[index], sin[index])[:length]
        scores = torch.einsum("qhd,khd->hqk", q, k) * module.head_dim**-0.5
        attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[:length])
        attended = F.pad(attended.flatten(1), (0, 0, 0, x.shape[1] - length))
        outputs.append(F.linear(attended, module.wo.weight, module.wo.bias))
    return torch.stack(outputs)


@pytest.mark.parametrize(
    ("unbatched", "grids"),
    [(True, [[2, 3]]), (False, [[2, 3]]), (False, [[1, 3]]), (False, [[2, 3], [1, 3]])],
    ids=["unbatched", "single-image", "padded-single-image", "padded-images"],
)
def test_vision_attention_4d_sdpa_matches_dense_outputs_and_gradients(monkeypatch, unbatched, grids):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        attention = VisionAttention.Config(dim=8, num_heads=2).build()
        attention.init_states(buffer_device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(len(grids), 6, 8, generator=generator)
    lengths = [height * width for height, width in grids]
    valid = torch.arange(6)[None, :] < torch.tensor(lengths)[:, None]
    cos, sin = _vision_rope_batch(torch.tensor(grids), 6, 2, 10000.0, inputs.device)
    reference_inputs = inputs.clone().requires_grad_()
    inputs = (inputs[0] if unbatched else inputs).requires_grad_()
    original_sdpa = F.scaled_dot_product_attention

    def sdpa_spy(q, k, v, **kwargs):
        # Keep the real CPU operation; this only checks the backend-facing layout.
        assert q.shape == k.shape == v.shape == (len(grids), 2, 6, 4)
        return original_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", sdpa_spy)
    actual = attention(
        inputs,
        cos[0] if unbatched else cos,
        sin[0] if unbatched else sin,
        valid[0] if unbatched else valid,
    )
    expected = _dense_reference(attention, reference_inputs, cos, sin, lengths)
    if unbatched:
        expected = expected[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    upstream = torch.randn(actual.shape, generator=generator)
    parameters = tuple(attention.parameters())
    actual_grads = torch.autograd.grad(actual, (inputs, *parameters), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_inputs, *parameters), upstream)
    expected_input_grad = expected_grads[0][0] if unbatched else expected_grads[0]
    torch.testing.assert_close(actual_grads[0], expected_input_grad, atol=1e-6, rtol=1e-5)
    for (name, _), actual_grad, expected_grad in zip(
        attention.named_parameters(), actual_grads[1:], expected_grads[1:], strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5, msg=name)
