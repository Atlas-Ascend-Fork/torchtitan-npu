# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

from ..testing_utils import target_devices


def test_block_mx_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(),
        activation_config=MXQuantizeConfig(),
    )
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[BlockMXQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, BlockMXTrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.data.activation_config is config.activation_config
    assert result.requires_grad


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_matches_the_training_forward(device):
    """The converted single-axis weight must multiply the way the block-MX wrapped one does."""
    weight_config = BlockMXQuantizeConfig()
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert isinstance(inference_weight, MXTensor), f"got {type(inference_weight).__name__}"
    assert torch.equal(y_training, y_inference), "the inference weight does not reproduce training's forward"


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_with_mxfp4_matches_the_training_forward(device):
    """With mxfp4-QAT the inference weight is FP4, as training's QAT pre-pass produces."""
    weight_config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    )
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert inference_weight.quant_config.elem_dtype is torch.float4_e2m1fn_x2, (
        f"expected an FP4 inference weight, got {inference_weight.quant_config.elem_dtype}"
    )
    assert torch.equal(y_training, y_inference), "the FP4 inference weight does not reproduce training's forward"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "shape",
    [(256, 128), (4, 128, 128)],
    ids=["matrix_rows", "experts"],
)
def test_mxfp4_communication_matches_fp8_payload(device, shape):
    """Real NPU kernels; simulate rank-order all-gather to isolate the format conversion."""
    from dataclasses import replace
    from types import SimpleNamespace

    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16)
    mesh = SimpleNamespace(size=lambda: 2)
    wrappers = [
        BlockMXTrainingWeightWrapperTensor(
            shard.contiguous(), weight_config=config, activation_config=MXQuantizeConfig()
        )
        for shard in weight.chunk(2)
    ]
    results = [
        w.fsdp_pre_all_gather(mesh, weight.shape, weight.stride(), None, SimpleNamespace(param_dtype=torch.bfloat16))
        for w in wrappers
    ]
    gathered = tuple(torch.cat([r[0][i] for r in results]) for i in range(2))
    output, inner = wrappers[0].fsdp_post_all_gather(gathered, results[0][1], torch.bfloat16)
    reference = []
    for w in wrappers:
        old = BlockMXTrainingWeightWrapperTensor(
            w._data,
            weight_config=replace(config, fsdp_prequantize_dtype="fp8"),
            activation_config=MXQuantizeConfig(),
        )
        reference.append(
            old.fsdp_pre_all_gather(
                mesh, weight.shape, weight.stride(), None, SimpleNamespace(param_dtype=torch.bfloat16)
            )[0]
        )
    for actual, expected in zip(inner, (torch.cat([r[i] for r in reference]) for i in range(3)), strict=True):
        torch.testing.assert_close(actual.view(torch.uint8), expected.contiguous().view(torch.uint8), rtol=0, atol=0)
    assert output.is_contiguous()
