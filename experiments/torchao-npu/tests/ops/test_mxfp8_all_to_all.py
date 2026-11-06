# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
import torch.distributed as dist
from torchao_npu.ops.mxfp8_all_to_all import (
    mxfp8_all_to_all,
    mxfp8_all_to_all_quantized,
)
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize
from torchao_npu.quantized_tensors.mx_tensor import MXTensor, _mxfp8_dequantize_tensor


class _ConsumeLogicalTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation):
        return torch.ones(activation.shape, device=activation.device, dtype=activation.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ConsumeMXGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation):
        return activation.clone()

    @staticmethod
    def backward(ctx, grad_output):
        assert isinstance(grad_output, MXTensor)
        return _mxfp8_dequantize_tensor(grad_output)


def test_combine_uses_high_precision_forward_and_mxfp8_backward(mock_distributed_env):
    dist.init_process_group("hccl")
    rows, width = 4, 64
    x = torch.linspace(-1, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    config = MXQuantizeConfig()

    def exchange(input):
        return mxfp8_all_to_all(
            _ConsumeMXGradient.apply(input),
            output_splits=[rows],
            input_splits=[rows],
            group=dist.group.WORLD,
            quant_params=(32, "rint", 1, 0.0),
        )

    compiled_exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    output = compiled_exchange(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    output.sum().backward()
    expected_qdata, expected_scale = mx_quantize(torch.ones_like(x), axis=-1, config=config)
    expected_grad = mx_dequantize(expected_qdata, expected_scale, -1, 32, torch.float8_e4m3fn, x.dtype)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("compile_enabled", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("block_size, scale_alg", [(32, 1), (64, 0)], ids=["block32", "block64_scale0"])
def test_mxfp8_dispatch_returns_wire_tensor_with_inverse_gradient(
    mock_distributed_env, monkeypatch, compile_enabled, block_size, scale_alg
):
    dist.init_process_group("hccl")
    rows, width = 4, 128
    x = torch.linspace(-1, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    splits = [rows]
    module = importlib.import_module("torchao_npu.ops.mxfp8_all_to_all")
    original_exchange = module.all_to_all_single
    wire_dtypes = []

    def record_exchange(input, *args, **kwargs):
        wire_dtypes.append(input.dtype)
        return original_exchange(input, *args, **kwargs)

    if not compile_enabled:
        monkeypatch.setattr(module, "all_to_all_single", record_exchange)

    def exchange(input):
        return mxfp8_all_to_all_quantized(
            input,
            output_splits=splits,
            input_splits=splits,
            group=dist.group.WORLD,
            quant_params=(block_size, "rint", scale_alg, 0.0),
        )

    if compile_enabled:
        exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})

    output = exchange(x)
    expected_qdata, expected_scale = mx_quantize(
        x.detach(), axis=-1, config=MXQuantizeConfig(block_size=block_size, scale_alg=scale_alg)
    )
    assert isinstance(output, MXTensor)
    assert output.shape == x.shape and output.dtype == x.dtype
    assert output.scale.shape == (rows, width // (2 * block_size), 2)
    torch.testing.assert_close(output.qdata, expected_qdata, rtol=0, atol=0)
    torch.testing.assert_close(output.scale, expected_scale, rtol=0, atol=0)
    if not compile_enabled:
        assert wire_dtypes[:2] == [torch.uint8, torch.uint8]

    _ConsumeLogicalTensor.apply(output).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_mxfp8_dispatch_compiled_backward_accepts_quantized_gradient(mock_distributed_env):
    dist.init_process_group("hccl")
    rows, width = 4, 64
    x = torch.linspace(-1, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    gradient = torch.linspace(0.25, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    mx_gradient = MXTensor.from_hp(gradient, MXQuantizeConfig(), axis=-1)

    def exchange(input):
        return mxfp8_all_to_all_quantized(
            input, output_splits=[rows], input_splits=[rows], group=dist.group.WORLD, quant_params=(32, "rint", 1, 0.0)
        )

    compiled_exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    output = compiled_exchange(x)
    output.backward(mx_gradient)

    torch.testing.assert_close(x.grad, mx_gradient.dequantize(), rtol=0, atol=0)


@pytest.mark.parametrize("compile_enabled", [False, True], ids=["eager", "compile"])
def test_mxfp8_dispatch_empty_split(mock_distributed_env, compile_enabled):
    dist.init_process_group("hccl")
    x = torch.empty((0, 64), device="npu", dtype=torch.bfloat16, requires_grad=True)

    def exchange(input):
        return mxfp8_all_to_all_quantized(
            input,
            output_splits=[0],
            input_splits=[0],
            group=dist.group.WORLD,
            quant_params=(32, "rint", 1, 0.0),
        )

    if compile_enabled:
        exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    output = exchange(x)

    assert output.shape == x.shape and output.qdata.dtype == torch.float8_e4m3fn
    assert output.scale.shape == (0, 1, 2) and output.scale.dtype == torch.uint8
    _ConsumeLogicalTensor.apply(output).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("exchange", [mxfp8_all_to_all, mxfp8_all_to_all_quantized])
def test_mxfp8_all_to_all_rejects_non_2d_activation(exchange):
    x = torch.empty((2, 3, 64), device="npu", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="expects a 2D activation"):
        exchange(x, output_splits=[2], input_splits=[2], quant_params=(32, "rint", 1, 0.0))
