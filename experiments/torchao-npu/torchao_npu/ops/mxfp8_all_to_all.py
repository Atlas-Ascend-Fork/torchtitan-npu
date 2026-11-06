# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asymmetric MXFP8 all-to-all exchanges for MoE forward and backward."""

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize
from torchao_npu.quantized_tensors.mx_tensor import MXTensor, _mxfp8_dequantize_tensor

# Primitive-only graph signature of the E4M3 activation quantizer.
MXFP8QuantParams = tuple[int, str, int, float]


@torch.library.custom_op(  # pyrefly: ignore [missing-attribute]
    "torchao_npu::mxfp8_quantize_or_empty", mutates_args=(), device_types="npu"
)
def _mxfp8_quantize_or_empty(
    tensor: torch.Tensor, block_size: int, round_mode: str, scale_alg: int, dst_type_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the empty-row branch opaque when compiling dynamic MoE token counts.

    The fake kernel infers metadata without testing an unbacked row count.
    Nonempty inputs still use ``mx_quantize``. The enclosing all-to-all
    ``autograd.Function`` defines the VJP; this internal op has no standalone
    autograd formula.
    """
    config = MXQuantizeConfig(
        block_size=block_size,
        elem_dtype=torch.float8_e4m3fn,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
    if tensor.shape[0] != 0:
        return mx_quantize(tensor, axis=-1, config=config)
    num_blocks = (tensor.shape[-1] + block_size - 1) // block_size
    return (
        torch.empty_like(tensor, dtype=torch.float8_e4m3fn),
        torch.empty((0, (num_blocks + 1) // 2, 2), dtype=torch.uint8, device=tensor.device),
    )


@_mxfp8_quantize_or_empty.register_fake
def _mxfp8_quantize_or_empty_fake(
    tensor: torch.Tensor, block_size: int, round_mode: str, scale_alg: int, dst_type_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    del round_mode, scale_alg, dst_type_max
    num_blocks = (tensor.shape[-1] + block_size - 1) // block_size
    return (
        torch.empty_like(tensor, dtype=torch.float8_e4m3fn),
        tensor.new_empty((tensor.shape[0], (num_blocks + 1) // 2, 2), dtype=torch.uint8),
    )


@torch.library.custom_op(  # pyrefly: ignore [missing-attribute]
    "torchao_npu::mxfp8_dequantize_or_empty", mutates_args=(), device_types="npu"
)
def _mxfp8_dequantize_or_empty(
    qdata: torch.Tensor, scale: torch.Tensor, block_size: int, output_dtype: torch.dtype
) -> torch.Tensor:
    """Decode wire payloads behind the same empty-row compilation boundary.

    Nonempty inputs use ``mx_dequantize``; empty inputs bypass the NPU kernel.
    The fake kernel preserves symbolic row counts without tracing this branch.
    Like the quantizer, this helper has no standalone autograd formula.
    """
    if qdata.shape[0] == 0:
        return torch.empty_like(qdata, dtype=output_dtype)
    return mx_dequantize(
        qdata,
        scale,
        axis=-1,
        block_size=block_size,
        src_dtype=torch.float8_e4m3fn,
        output_dtype=output_dtype,
    )


@_mxfp8_dequantize_or_empty.register_fake
def _mxfp8_dequantize_or_empty_fake(
    qdata: torch.Tensor, scale: torch.Tensor, block_size: int, output_dtype: torch.dtype
) -> torch.Tensor:
    del scale, block_size
    return torch.empty_like(qdata, dtype=output_dtype)


def _exchange_payload(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
    quant_params: MXFP8QuantParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The custom op gives AOTAutograd a FakeTensor contract while MXTensor
    # dispatch lowers the real path to qdata/scale dequantization.
    if isinstance(input, MXTensor):
        input = _mxfp8_dequantize_tensor(input)
    qdata, scale = _mxfp8_quantize_or_empty(input, *quant_params)
    qdata_out = all_to_all_single(
        qdata.view(torch.uint8), output_split_sizes=output_splits, input_split_sizes=input_splits, group=group
    )
    scale_out = all_to_all_single(
        scale.view(torch.uint8), output_split_sizes=output_splits, input_split_sizes=input_splits, group=group
    )
    qdata_out = torch.ops._c10d_functional.wait_tensor(qdata_out)  # pyrefly: ignore [not-callable]
    scale_out = torch.ops._c10d_functional.wait_tensor(scale_out)  # pyrefly: ignore [not-callable]
    # npu_dynamic_mx_quant exposes E8M0 scale bits through uint8 storage.
    return qdata_out.view(torch.float8_e4m3fn), scale_out


def _validate_activation(input: torch.Tensor, quant_params: MXFP8QuantParams) -> None:
    if input.ndim != 2:
        raise ValueError(f"MXFP8 all-to-all expects a 2D activation, got shape {tuple(input.shape)}")
    if quant_params[0] <= 0 or quant_params[0] % 32 != 0:
        raise ValueError(f"MXFP8 block_size must be a positive multiple of 32, got {quant_params[0]}")


def _exchange_high_precision(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    if isinstance(input, MXTensor):
        input = _mxfp8_dequantize_tensor(input)
    output = all_to_all_single(
        input,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return torch.ops._c10d_functional.wait_tensor(output)  # pyrefly: ignore [not-callable]


def _index_mxfp8_rows(qdata: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Reorder E4M3 rows through a bit-preserving byte view supported by NPU indexing."""
    return qdata.view(torch.uint8)[row_indices].view(torch.float8_e4m3fn)


@torch._dynamo.allow_in_graph
def _mxfp8_wire_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    quant_params: MXFP8QuantParams,
    requires_grad: bool,
) -> MXTensor:
    # Keep the Python config inside this constructor: the surrounding compiled
    # autograd path passes only primitive quantization metadata.
    block_size, round_mode, scale_alg, dst_type_max = quant_params
    config = MXQuantizeConfig(
        block_size=block_size,
        elem_dtype=torch.float8_e4m3fn,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
    return MXTensor(qdata, scale, dtype, -1, config, requires_grad=requires_grad)


class _MXFP8AllToAll(torch.autograd.Function):
    """High-precision forward exchange with an MXFP8 inverse VJP."""

    @staticmethod
    def forward(ctx, input, output_splits, input_splits, group, quant_params):  # pyrefly: ignore [bad-override]
        ctx.output_splits, ctx.input_splits, ctx.group, ctx.quant_params = (
            output_splits,
            input_splits,
            group,
            quant_params,
        )
        return _exchange_high_precision(input, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        qdata, scale = _exchange_payload(
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
            ctx.quant_params,
        )
        grad_input = _mxfp8_wire_tensor(  # pyrefly: ignore [not-callable]
            qdata,
            scale,
            grad_output.dtype,
            ctx.quant_params,
            grad_output.requires_grad,
        )
        return grad_input, None, None, None, None


class _MXFP8QuantizedAllToAll(torch.autograd.Function):
    """MXFP8 forward exchange with a high-precision inverse VJP."""

    @staticmethod
    def forward(ctx, input, output_splits, input_splits, group, quant_params):  # pyrefly: ignore [bad-override]
        ctx.output_splits, ctx.input_splits, ctx.group, ctx.quant_params = (
            output_splits,
            input_splits,
            group,
            quant_params,
        )
        qdata, scale = _exchange_payload(input, output_splits, input_splits, group, quant_params)
        return _mxfp8_wire_tensor(  # pyrefly: ignore [not-callable]
            qdata, scale, input.dtype, quant_params, input.requires_grad
        )

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        grad_input = _exchange_high_precision(grad_output, ctx.input_splits, ctx.output_splits, ctx.group)
        return grad_input, None, None, None, None


def mxfp8_all_to_all(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup | None = None,
    *,
    quant_params: MXFP8QuantParams,
) -> torch.Tensor:
    """Exchange BF16 in forward and return MXFP8 from the inverse backward exchange."""
    _validate_activation(input, quant_params)
    group = dist.group.WORLD if group is None else group
    return _MXFP8AllToAll.apply(input, output_splits, input_splits, group, quant_params)


def mxfp8_all_to_all_quantized(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup | None = None,
    *,
    quant_params: MXFP8QuantParams,
) -> MXTensor:
    """Exchange and return the K-axis MXFP8 activation without dequantizing it."""
    _validate_activation(input, quant_params)
    group = dist.group.WORLD if group is None else group
    return _MXFP8QuantizedAllToAll.apply(input, output_splits, input_splits, group, quant_params)


__all__ = ["mxfp8_all_to_all", "mxfp8_all_to_all_quantized"]
