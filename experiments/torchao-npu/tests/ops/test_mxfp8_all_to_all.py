# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
import torch.distributed as dist
from torchao_npu.ops.mxfp8_all_to_all import (
    _index_mxfp8_rows,
    mxfp8_all_to_all,
    mxfp8_all_to_all_quantized,
)
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


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
        if not isinstance(grad_output, MXTensor):
            raise AssertionError("Combine backward must supply an MXTensor gradient")
        return grad_output.dequantize()


def test_combine_uses_high_precision_forward_and_mxfp8_backward(mock_distributed_env):
    dist.init_process_group("hccl")
    rows, width = 4, 64
    x = torch.linspace(-1, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    config = MXQuantizeConfig()

    # Let AOTAutograd trace the producer and consumer together. Capturing them
    # as separate Dynamo autograd HOPs can lose the subclass at their boundary.
    @torch.compiler.allow_in_graph
    def exchange(input):
        return mxfp8_all_to_all(
            _ConsumeMXGradient.apply(input),
            output_splits=[rows],
            input_splits=[rows],
            group=dist.group.WORLD,
            quant_params=(32, "rint", 1, 0.0),
        )

    compiled_exchange = torch.compile(
        lambda input: exchange(input), backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"}
    )
    output = compiled_exchange(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    gradient = torch.ones_like(output)
    output.backward(gradient)

    expected_qdata, expected_scale = mx_quantize(gradient, axis=-1, config=config)
    expected_grad = mx_dequantize(expected_qdata, expected_scale, -1, 32, torch.float8_e4m3fn, x.dtype)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)


def test_combine_reorders_high_precision_forward_and_mxfp8_backward(mock_distributed_env):
    from torchao_npu.quantized_modules.token_dispatcher import _MXFP8CombineExchange

    dist.init_process_group("hccl")
    rows, width = 8, 64
    x = torch.arange(rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    # Inverse of the rank-major -> expert-major order for counts [[1, 2], [2, 1], [1, 1]].
    restore_indices = torch.tensor([0, 4, 5, 1, 2, 6, 3, 7], device="npu", dtype=torch.int32)
    token_order = torch.tensor([0, 3, 4, 6, 1, 2, 5, 7], device="npu")
    gradient = torch.arange(1, rows + 1, device="npu", dtype=torch.bfloat16).reshape(rows, 1).expand_as(x)
    quant_params = (32, "rint", 1, 0.0)

    @torch.compiler.allow_in_graph
    def combine(input, restore_indices):
        return _MXFP8CombineExchange.apply(
            _ConsumeMXGradient.apply(input),
            restore_indices,
            dist.group.WORLD,
            [rows],
            [rows],
            quant_params,
        )

    compiled_combine = torch.compile(
        lambda input, indices: combine(input, indices),
        backend="inductor",
        fullgraph=True,
        options={"npu_backend": "ascendc"},
    )
    output = compiled_combine(x, restore_indices)
    torch.testing.assert_close(output, x[restore_indices], rtol=0, atol=0)
    output.backward(gradient)

    config = MXQuantizeConfig()
    expected_qdata, expected_scale = mx_quantize(gradient, axis=-1, config=config)
    expected_gradient = mx_dequantize(
        _index_mxfp8_rows(expected_qdata, token_order),
        expected_scale[token_order],
        -1,
        32,
        torch.float8_e4m3fn,
        x.dtype,
    )
    torch.testing.assert_close(x.grad, expected_gradient, rtol=0, atol=0)


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
        output = mxfp8_all_to_all_quantized(
            input,
            output_splits=splits,
            input_splits=splits,
            group=dist.group.WORLD,
            quant_params=(block_size, "rint", scale_alg, 0.0),
        )
        # Keep the plain-gradient consumer in the graph and expose the wire
        # tensor separately for forward-only checks.
        return _ConsumeLogicalTensor.apply(output).sum(), output.detach()

    if compile_enabled:
        exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})

    loss, output = exchange(x)
    expected_qdata, expected_scale = mx_quantize(
        x.detach(), axis=-1, config=MXQuantizeConfig(block_size=block_size, scale_alg=scale_alg)
    )
    assert isinstance(output, MXTensor)
    assert output.shape == x.shape and output.dtype == x.dtype
    assert output.scale.shape == (rows, width // (2 * block_size), 2)
    torch.testing.assert_close(output.qdata.view(torch.uint8), expected_qdata.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output.scale, expected_scale, rtol=0, atol=0)
    if not compile_enabled:
        assert wire_dtypes[:2] == [torch.uint8, torch.uint8]

    loss.backward()
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


@pytest.mark.parametrize("block_size", [32, 64], ids=["block32", "block64"])
def test_mxfp8_dispatch_and_reroute_compile_with_plain_gradient(mock_distributed_env, block_size):
    from torchao_npu.quantized_modules.token_dispatcher import _MXFP8ReRouting

    dist.init_process_group("hccl")
    rows, width = 4, 128
    x = torch.linspace(-1, 1, rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    x.requires_grad_()
    counts = torch.tensor([[1, 1], [1, 1]], device="npu", dtype=torch.int64)

    def forward(input):
        quant_params = (block_size, "rint", 1, 0.0)
        wire = mxfp8_all_to_all_quantized(input, [rows], [rows], group=dist.group.WORLD, quant_params=quant_params)
        routed, _, _, _ = _MXFP8ReRouting.apply(wire, counts, None, quant_params)
        return _ConsumeLogicalTensor.apply(routed).sum()

    compiled_forward = torch.compile(forward, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    compiled_forward(x).backward()

    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("compile_enabled", [False, True], ids=["eager", "compile"])
def test_mxfp8_reroute_preserves_quantized_rows_and_reverses_gradient(mock_distributed_env, compile_enabled):
    from torchao_npu.quantized_modules.token_dispatcher import _MXFP8ReRouting

    rows, width = 8, 64
    x = torch.arange(rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    row_magnitudes = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=x.device, dtype=x.dtype)
    x = x * row_magnitudes.unsqueeze(-1)
    qdata, scale = mx_quantize(x, -1, MXQuantizeConfig())
    activation = MXTensor(qdata, scale, x.dtype, -1, MXQuantizeConfig(), requires_grad=True)
    counts = torch.tensor([[1, 2], [2, 1], [1, 1]], device="npu", dtype=torch.int64)
    weights = torch.arange(1, rows + 1, device="npu", dtype=torch.bfloat16).reshape(rows, 1)

    def reroute(input, token_counts, weights):
        routed, _, restore, expert_counts = _MXFP8ReRouting.apply(input, token_counts, None, (32, "rint", 1, 0.0))
        loss = (_ConsumeLogicalTensor.apply(routed) * weights).sum()
        return loss, routed.detach(), restore, expert_counts

    if compile_enabled:
        reroute = torch.compile(reroute, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    loss, routed, restore, expert_counts = reroute(activation, counts, weights)

    # Enumerate expert 0's rank segments, then expert 1's; this permutation is not self-inverse.
    expected_order = torch.tensor([0, 3, 4, 6, 1, 2, 5, 7], device="npu")
    restore_order = torch.tensor([0, 4, 5, 1, 2, 6, 3, 7], device="npu", dtype=torch.int32)
    torch.testing.assert_close(routed.qdata.view(torch.uint8), qdata.view(torch.uint8)[expected_order], rtol=0, atol=0)
    torch.testing.assert_close(routed.scale, scale[expected_order], rtol=0, atol=0)
    torch.testing.assert_close(expert_counts, torch.tensor([4, 4], device="npu"))
    torch.testing.assert_close(restore, restore_order, rtol=0, atol=0)

    loss.backward()
    torch.testing.assert_close(activation.grad, weights[restore_order.long()].expand_as(x))


def test_mxfp8_reroute_backward_accepts_quantized_gradient(mock_distributed_env):
    from torchao_npu.quantized_modules.token_dispatcher import _MXFP8ReRouting

    rows, width = 8, 64
    x = torch.arange(rows * width, device="npu", dtype=torch.bfloat16).reshape(rows, width)
    qdata, scale = mx_quantize(x, -1, MXQuantizeConfig())
    activation = MXTensor(qdata, scale, x.dtype, -1, MXQuantizeConfig(), requires_grad=True)
    counts = torch.tensor([[1, 2], [2, 1], [1, 1]], device="npu", dtype=torch.int64)
    gradient = torch.arange(1, rows + 1, device="npu", dtype=torch.bfloat16).reshape(rows, 1).expand_as(x).contiguous()
    mx_gradient = MXTensor.from_hp(gradient, MXQuantizeConfig(), axis=-1)

    def reroute(input, expert_counts):
        return _MXFP8ReRouting.apply(input, expert_counts, None, (32, "rint", 1, 0.0))

    compiled_reroute = torch.compile(reroute, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    routed, _, _, _ = compiled_reroute(activation, counts)
    routed.backward(mx_gradient)

    restore_order = torch.tensor([0, 4, 5, 1, 2, 6, 3, 7], device="npu")
    torch.testing.assert_close(activation.grad, mx_gradient.dequantize()[restore_order], rtol=0, atol=0)


@pytest.mark.parametrize("compile_enabled", [False, True], ids=["eager", "compile"])
def test_mxfp8_dispatch_empty_split(mock_distributed_env, compile_enabled):
    dist.init_process_group("hccl")
    x = torch.empty((0, 64), device="npu", dtype=torch.bfloat16, requires_grad=True)

    def exchange(input):
        output = mxfp8_all_to_all_quantized(
            input,
            output_splits=[0],
            input_splits=[0],
            group=dist.group.WORLD,
            quant_params=(32, "rint", 1, 0.0),
        )
        return _ConsumeLogicalTensor.apply(output).sum(), output.detach()

    if compile_enabled:
        exchange = torch.compile(exchange, backend="inductor", fullgraph=True, options={"npu_backend": "ascendc"})
    loss, output = exchange(x)

    assert output.shape == x.shape and output.qdata.dtype == torch.float8_e4m3fn
    assert output.scale.shape == (0, 1, 2) and output.scale.dtype == torch.uint8
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("exchange", [mxfp8_all_to_all, mxfp8_all_to_all_quantized])
def test_mxfp8_all_to_all_rejects_non_2d_activation(exchange):
    x = torch.empty((2, 3, 64), device="npu", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="expects a 2D activation"):
        exchange(x, output_splits=[2], input_splits=[2], quant_params=(32, "rint", 1, 0.0))
