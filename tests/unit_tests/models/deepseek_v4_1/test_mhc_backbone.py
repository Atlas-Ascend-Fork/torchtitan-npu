# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folded backbone orchestration with attention and MoE explicitly isolated."""

import pytest
import torch

from tests.unit_tests.models.deepseek_v4_1.mhc_boundary_utils import build_mhc_boundary_model
from tests.unit_tests.models.deepseek_v4_1.mhc_reference import hc_post_reference, hc_pre_reference
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata, build_v41_frames


def _backbone_reference(model, tokens, parameters, input_embeds):
    def norm(x, prefix, eps):
        return x / (x.square().mean(-1, keepdim=True) + eps).sqrt() * parameters[f"{prefix}.weight"]

    hidden = parameters["tok_embeddings.weight"][tokens] if input_embeds is None else input_embeds
    hidden = torch.stack([hidden] * model.hc_mult, dim=1)
    pre_mix = hidden.new_zeros(hidden.shape[:-1])
    pre_mix[:, 0] = 1
    for name, layer in model.layers.items():
        for kind, sublayer_norm in (("attn", layer.attention_norm), ("ffn", layer.ffn_norm)):
            prefix = f"layers.{name}"
            module = getattr(layer, f"hc_{kind}_pre")
            state = {key: parameters[f"{prefix}.hc_{kind}_pre.{key}"] for key in ("hc_fn", "hc_base", "hc_scale")}
            collapsed, pre_mix, post, comb = hc_pre_reference(
                hidden,
                pre_mix,
                state,
                sinkhorn_iters=module.sinkhorn_iters,
                hc_eps=module.hc_eps,
                norm_eps=module.norm_eps,
            )
            normalized = norm(collapsed, f"{prefix}.{'attention' if kind == 'attn' else 'ffn'}_norm", sublayer_norm.eps)
            output = normalized.tanh() * 0.75 if kind == "attn" else normalized * normalized.sigmoid() * 0.5 - 0.125
            hidden = hc_post_reference(output, hidden, post, comb)
    collapsed = torch.stack(
        [sum(pre[m] * branch[m] for m in range(model.hc_mult)) for pre, branch in zip(pre_mix, hidden, strict=True)]
    )
    normalized = norm(collapsed, "norm", model.norm.eps)
    return normalized @ parameters["lm_head.weight"].t()


@pytest.mark.parametrize("supplied_embeddings", (False, True), ids=("token-embedding", "supplied-embedding"))
def test_folded_backbone_outputs_and_gradients_match_reference(supplied_embeddings):
    model = build_mhc_boundary_model()
    tokens = torch.tensor([3, 1, 3])
    positions = torch.tensor([0, 1, 0])
    bounds = torch.tensor([0, 2, 3], dtype=torch.int32)
    q, swa_k, cmp_k = build_v41_frames(bounds, bounds, 0)
    metadata = DeepSeekV41Metadata(
        cu_seq_q=bounds, cu_seq_k=bounds, position_k=positions, ratio=0, q=q, swa_k=swa_k, cmp_k=cmp_k
    )
    generator = torch.Generator().manual_seed(71)
    embeds = torch.randn(3, 4, generator=generator, requires_grad=True) if supplied_embeddings else None
    reference_embeds = embeds.detach().double().requires_grad_() if embeds is not None else None
    parameters = {name: value.detach().double().requires_grad_() for name, value in model.named_parameters()}

    actual = model(tokens, positions, attention_metadata={"mhc_boundary": metadata}, input_embeds=embeds)
    expected = _backbone_reference(model, tokens, parameters, reference_embeds)
    upstream = torch.randn(3, 11, generator=generator)
    actual_inputs = (*model.parameters(), *((embeds,) if embeds is not None else ()))
    reference_inputs = (*parameters.values(), *((reference_embeds,) if reference_embeds is not None else ()))
    gradients = torch.autograd.grad(actual, actual_inputs, upstream, allow_unused=True)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream.double(), allow_unused=True)

    assert actual.shape == (3, 11)
    torch.testing.assert_close(actual, expected.float(), rtol=3e-5, atol=3e-6)
    for value, reference in zip(gradients, reference_gradients, strict=True):
        assert (value is None) == (reference is None)
        if value is not None:
            torch.testing.assert_close(value, reference.float(), rtol=5e-4, atol=3e-6)
