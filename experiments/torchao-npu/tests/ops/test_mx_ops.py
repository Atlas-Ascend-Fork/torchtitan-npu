# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchao.float8.float8_utils import compute_error
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantized_tensors import BlockMXTensor, DualAxisMXTensor, MXTensor

# =========================================================================
# Tests for mxfp4_fake_quantize (autograd.Function wrapper)
# =========================================================================


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("shape", [(32, 256), (1024, 1024), (16, 512, 256)])
def test_mxfp4_fake_quantize(shape, axis):
    """mxfp4_fake_quantize: forward SQNR, STE backward, idempotency."""
    from torchao_npu.ops.mx_ops import mxfp4_fake_quantize

    torch.manual_seed(0)
    x = torch.randn(*shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    y = mxfp4_fake_quantize(x, config, axis=axis)

    # Shape and dtype preserved
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.dtype == x.dtype

    # SQNR: FP4 block32 should exceed 12 dB
    sqnr = compute_error(x, y).item()
    assert sqnr > 12, f"SQNR too low: {sqnr:.1f} dB for axis={axis}, shape={shape}"

    # STE backward
    y.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x.grad)), f"STE gradient failed for axis={axis}, shape={shape}"

    # Idempotency: quantize -> dequantize -> quantize -> dequantize
    y2 = mxfp4_fake_quantize(y, config, axis=axis)
    assert torch.equal(y, y2), f"Not idempotent for axis={axis}, shape={shape}"


@pytest.mark.parametrize("round_mode", ["rint", "floor", "round"])
def test_mxfp4_fake_quantize_round_modes(round_mode):
    """Different round_modes should produce different quantization results."""
    from torchao_npu.ops.mx_ops import mxfp4_fake_quantize

    torch.manual_seed(0)
    x = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, round_mode=round_mode)

    y = mxfp4_fake_quantize(x, config, axis=-1)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    # Round mode should not affect shape/dtype, just numerical accuracy
    sqnr = compute_error(x, y).item()
    assert sqnr > 10, f"SQNR too low: {sqnr:.1f} dB for round_mode={round_mode}"


# =========================================================================
# Tests for mxfp8_dequantize
# =========================================================================


@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("shape", [(32, 256), (1024, 1024)])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mxfp8_dequantize(shape, axis, elem_dtype):
    """mxfp8_dequantize: shape preserved, SQNR vs original, idempotent."""
    from torchao_npu.ops.mx_ops import mxfp8_dequantize

    torch.manual_seed(0)
    x = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    x_q, x_scale = torch_npu.npu_dynamic_mx_quant(
        x,
        axis=axis,
        dst_type=elem_dtype,
        block_size=config.block_size,
        round_mode=config.round_mode,
        scale_alg=config.scale_alg,
    )

    x_dq = mxfp8_dequantize(
        x_q, x_scale, axis=axis, block_size=config.block_size, output_shape=x.shape, output_dtype=x.dtype
    )

    assert x_dq.shape == x.shape, f"Shape mismatch: {x_dq.shape} vs {x.shape}"
    assert x_dq.dtype == x.dtype, f"Dtype mismatch: {x_dq.dtype} vs {x.dtype}"

    sqnr = compute_error(x, x_dq).item()
    threshold = 26 if elem_dtype == torch.float8_e4m3fn else 22
    assert sqnr > threshold, f"SQNR {sqnr:.1f} dB below {threshold} dB for {elem_dtype}"

    # Idempotency: requantize the dequantized output, dequantize again
    x_q2, x_scale2 = torch_npu.npu_dynamic_mx_quant(
        x_dq,
        axis=axis,
        dst_type=elem_dtype,
        block_size=config.block_size,
        round_mode=config.round_mode,
        scale_alg=config.scale_alg,
    )
    x_dq2 = mxfp8_dequantize(
        x_q2, x_scale2, axis=axis, block_size=config.block_size, output_shape=x.shape, output_dtype=x.dtype
    )

    assert torch.equal(x_dq, x_dq2), "Not idempotent"


# =========================================================================
# Tests for to_mx_then_mm (real MX quantized matmul)
# =========================================================================


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 26),
        (torch.float8_e5m2, 21),
    ],
)
def test_to_mx_then_mm_forward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_mm forward produces reasonable SQNR vs bf16 baseline."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    torch.manual_seed(0)
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_mm(A, B, cfg, cfg)
    Y_ref = A @ B

    assert Y.shape == (M, N), f"Shape mismatch: {Y.shape}"
    assert Y.dtype == A.dtype

    sqnr = compute_error(Y, Y_ref).item()
    assert sqnr > sqnr_threshold, f"SQNR {sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 28),
        (torch.float8_e5m2, 23),
    ],
)
def test_to_mx_then_mm_backward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_mm backward produces meaningful gradients vs bf16 baseline."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    torch.manual_seed(1)
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16, requires_grad=True)

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_mm(A, B, cfg, cfg)
    Y_ref = A_ref @ B_ref

    loss = Y.sum()
    loss.backward()
    loss_ref = Y_ref.sum()
    loss_ref.backward()

    assert A.grad is not None and B.grad is not None
    assert A_ref.grad is not None and B_ref.grad is not None

    dA_sqnr = compute_error(A.grad, A_ref.grad).item()
    dB_sqnr = compute_error(B.grad, B_ref.grad).item()
    assert dA_sqnr > sqnr_threshold, f"dA SQNR {dA_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
    assert dB_sqnr > sqnr_threshold, f"dB SQNR {dB_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


# =========================================================================
# Tests for to_mx_then_grouped_mm (real MX quantized grouped matmul)
# =========================================================================


def _make_group_list(M, E, device):
    """Create cumsum group_list for E groups dividing M rows."""
    base = M // E
    rem = M % E
    sizes = [base + 1] * rem + [base] * (E - rem)
    return torch.tensor(sizes, device=device).cumsum(0).to(torch.int64)


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 26),
        (torch.float8_e5m2, 21),
    ],
)
def test_to_mx_then_grouped_mm_forward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_grouped_mm forward produces reasonable SQNR vs bf16 baseline."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    torch.manual_seed(0)
    M, K, N, E = 128, 256, 64, 4
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(E, K, N, device=device, dtype=torch.bfloat16)
    group_list = _make_group_list(M, E, device)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_grouped_mm(A, B, group_list, cfg, cfg)

    # Reference: each expert slice
    Y_ref = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    start = 0
    for i in range(E):
        end = group_list[i].item()
        Y_ref[start:end] = A[start:end] @ B[i]
        start = end

    assert Y.shape == (M, N), f"Shape mismatch: {Y.shape}"
    assert Y.dtype == A.dtype

    sqnr = compute_error(Y, Y_ref).item()
    assert sqnr > sqnr_threshold, f"SQNR {sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 28),
        (torch.float8_e5m2, 23),
    ],
)
def test_to_mx_then_grouped_mm_backward(device, elem_dtype, sqnr_threshold):
    """to_mx_then_grouped_mm backward produces meaningful gradients."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    torch.manual_seed(1)
    M, K, N, E = 128, 256, 64, 4
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    B = torch.randn(E, K, N, device=device, dtype=torch.bfloat16, requires_grad=True)
    group_list = _make_group_list(M, E, device)

    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_grouped_mm(A, B, group_list, cfg, cfg)

    # Reference
    Y_ref = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    start = 0
    for i in range(E):
        end = group_list[i].item()
        Y_ref[start:end] = A_ref[start:end] @ B_ref[i]
        start = end

    loss = Y.sum()
    loss.backward()
    loss_ref = Y_ref.sum()
    loss_ref.backward()

    assert A.grad is not None and B.grad is not None
    assert A_ref.grad is not None and B_ref.grad is not None

    dA_sqnr = compute_error(A.grad, A_ref.grad).item()
    dB_sqnr = compute_error(B.grad, B_ref.grad).item()
    assert dA_sqnr > sqnr_threshold, f"dA SQNR {dA_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
    assert dB_sqnr > sqnr_threshold, f"dB SQNR {dB_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


# =========================================================================
# Tests for to_mx_then_bmm (real MX quantized batched matmul)
# =========================================================================


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, B, M, K, N, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 4, 2048, 4096, 2048, 27.5),
        (torch.float8_e5m2, 4, 2048, 4096, 2048, 21.5),
    ],
)
def test_to_mx_then_bmm_forward(device, elem_dtype, B, M, K, N, sqnr_threshold):
    """to_mx_then_bmm forward produces reasonable SQNR vs bf16 baseline."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    torch.manual_seed(0)
    act = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(B, K, N, device=device, dtype=torch.bfloat16)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_bmm(act, weight, cfg, cfg)
    Y_ref = torch.bmm(act, weight)

    assert Y.shape == (B, M, N), f"Shape mismatch: {Y.shape}"
    assert Y.dtype == act.dtype

    sqnr = compute_error(Y, Y_ref).item()
    assert sqnr > sqnr_threshold, f"SQNR {sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


@pytest.mark.parametrize("device", ["npu"])
@pytest.mark.parametrize(
    "elem_dtype, B, M, K, N, sqnr_threshold",
    [
        (torch.float8_e4m3fn, 4, 2048, 4096, 2048, 30.5),
        (torch.float8_e5m2, 4, 2048, 4096, 2048, 24.5),
    ],
)
def test_to_mx_then_bmm_backward(device, elem_dtype, B, M, K, N, sqnr_threshold):
    """to_mx_then_bmm backward produces meaningful batched gradients."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    torch.manual_seed(1)
    act = torch.randn(B, M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(B, K, N, device=device, dtype=torch.bfloat16, requires_grad=True)

    act_ref = act.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)

    cfg = MXQuantizeConfig(elem_dtype=elem_dtype)
    Y = to_mx_then_bmm(act, weight, cfg, cfg)
    Y_ref = torch.bmm(act_ref, weight_ref)

    loss = Y.sum()
    loss.backward()
    loss_ref = Y_ref.sum()
    loss_ref.backward()

    assert act.grad is not None and weight.grad is not None
    assert act_ref.grad is not None and weight_ref.grad is not None

    dA_sqnr = compute_error(act.grad, act_ref.grad).item()
    dB_sqnr = compute_error(weight.grad, weight_ref.grad).item()
    assert dA_sqnr > sqnr_threshold, f"dA SQNR {dA_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"
    assert dB_sqnr > sqnr_threshold, f"dB SQNR {dB_sqnr:.1f} dB below {sqnr_threshold} dB for {elem_dtype}"


# =========================================================================
# Pre-quantized MXTensor operands
# =========================================================================


def test_mm_accepts_pre_quantized_activation():
    """An activation holding its K-dim form feeds the forward and dgrad unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    torch.manual_seed(0)
    m, k, n = 128, 256, 64
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="npu", dtype=torch.bfloat16)
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_mm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    # The same quantization the high-precision path applies to its A operand.
    mx_a = DualAxisMXTensor.from_hp(a.detach(), cfg).to_mx_tensor(quant_axis=-1)
    mx_a.requires_grad_(True)
    b_mx = b.clone().detach().requires_grad_(True)
    out = to_mx_then_mm(mx_a, b_mx, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized activation changed the forward"
    assert torch.equal(mx_a.grad, a_ref.grad), "the dgrad does not use A and must be identical"
    # The wgrad rebuilds A's dim=-2 form from the dequantized activation, adding one
    # quantization step, so it only has to stay close to the high-precision path.
    wgrad_sqnr = compute_error(b_mx.grad, b_ref.grad).item()
    assert wgrad_sqnr > 20.0, f"the wgrad SQNR vs the HP path is too low: {wgrad_sqnr:.2f} db"


def test_mm_accepts_pre_quantized_weight():
    """A weight holding its K-dim form feeds the forward and wgrad unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    torch.manual_seed(0)
    m, k, n = 128, 256, 64
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    # Stored [n, k] and transposed, as a linear layer's weight is used. This keeps the
    # dim=-2 quant axis unit-stride, which dequantizing it in the backward needs.
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    b = weight.T
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_mm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    # The same quantization the high-precision path applies to its B operand.
    mx_b = DualAxisMXTensor.from_hp(b.detach(), cfg).to_mx_tensor(quant_axis=-2)
    mx_b.requires_grad_(True)
    a_mx = a.clone().detach().requires_grad_(True)
    out = to_mx_then_mm(a_mx, mx_b, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized weight changed the forward"
    assert torch.equal(mx_b.grad, b_ref.grad), "the wgrad does not use B and must be identical"
    # The dgrad rebuilds B's dim=-1 form from the dequantized weight, adding one
    # quantization step, so it only has to stay close to the high-precision path.
    dgrad_sqnr = compute_error(a_mx.grad, a_ref.grad).item()
    assert dgrad_sqnr > 20.0, f"the dgrad SQNR vs the HP path is too low: {dgrad_sqnr:.2f} db"


def test_mm_accepts_dual_axis_quantized_weight():
    """A dual-axis weight supplies both forms, so every output and gradient is unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    torch.manual_seed(0)
    m, k, n = 128, 256, 64
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    b = weight.T
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_mm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    mx_b = DualAxisMXTensor.from_hp(b.detach(), cfg)
    mx_b.requires_grad_(True)
    a_mx = a.clone().detach().requires_grad_(True)
    out = to_mx_then_mm(a_mx, mx_b, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the dual-axis weight changed the forward"
    assert torch.equal(a_mx.grad, a_ref.grad), "the dgrad must be identical"
    assert torch.equal(mx_b.grad, b_ref.grad), "the wgrad must be identical"


def test_bmm_accepts_pre_quantized_activation():
    """A batched activation holding its K-dim form feeds the forward and dgrad unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    torch.manual_seed(0)
    batch, m, k, n = 2, 64, 128, 64
    a = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(batch, k, n, device="npu", dtype=torch.bfloat16)
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_bmm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    mx_a = DualAxisMXTensor.from_hp(a.detach(), cfg).to_mx_tensor(quant_axis=-1)
    mx_a.requires_grad_(True)
    b_mx = b.clone().detach().requires_grad_(True)
    out = to_mx_then_bmm(mx_a, b_mx, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized activation changed the forward"
    assert torch.equal(mx_a.grad, a_ref.grad), "the dgrad does not use A and must be identical"
    wgrad_sqnr = compute_error(b_mx.grad, b_ref.grad).item()
    assert wgrad_sqnr > 20.0, f"the wgrad SQNR vs the HP path is too low: {wgrad_sqnr:.2f} db"


def test_bmm_accepts_dual_axis_quantized_activation():
    """A dual-axis activation supplies both forms, so every output and gradient is unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    torch.manual_seed(0)
    batch, m, k, n = 2, 64, 128, 64
    a = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(batch, k, n, device="npu", dtype=torch.bfloat16)
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_bmm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    mx_a = DualAxisMXTensor.from_hp(a.detach(), cfg)
    mx_a.requires_grad_(True)
    b_mx = b.clone().detach().requires_grad_(True)
    out = to_mx_then_bmm(mx_a, b_mx, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the dual-axis activation changed the forward"
    assert torch.equal(mx_a.grad, a_ref.grad), "the dgrad must be identical"
    assert torch.equal(b_mx.grad, b_ref.grad), "the wgrad must be identical"


def test_bmm_accepts_pre_quantized_weight():
    """A batched weight holding its K-dim form feeds the forward and wgrad unchanged."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    torch.manual_seed(0)
    batch, m, k, n = 2, 64, 128, 64
    a = torch.randn(batch, m, k, device="npu", dtype=torch.bfloat16)
    # Stored [batch, n, k] and transposed, keeping the dim=-2 quant axis unit-stride.
    weight = torch.randn(batch, n, k, device="npu", dtype=torch.bfloat16)
    b = weight.transpose(-1, -2)
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_bmm(a_ref, b_ref, cfg, cfg)
    out_ref.sum().backward()

    mx_b = DualAxisMXTensor.from_hp(b.detach(), cfg).to_mx_tensor(quant_axis=-2)
    mx_b.requires_grad_(True)
    a_mx = a.clone().detach().requires_grad_(True)
    out = to_mx_then_bmm(a_mx, mx_b, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized weight changed the forward"
    assert torch.equal(mx_b.grad, b_ref.grad), "the wgrad does not use B and must be identical"
    dgrad_sqnr = compute_error(a_mx.grad, a_ref.grad).item()
    assert dgrad_sqnr > 20.0, f"the dgrad SQNR vs the HP path is too low: {dgrad_sqnr:.2f} db"


def test_grouped_mm_accepts_pre_quantized_activation():
    """The grouped path uses the pre-quantized activation as its K-dim form."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    torch.manual_seed(0)
    m, k, n, e = 128, 64, 128, 4
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    b = torch.randn(e, k, n, device="npu", dtype=torch.bfloat16)
    group_list = _make_group_list(m, e, "npu")
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_grouped_mm(a_ref, b_ref, group_list, cfg, cfg)
    out_ref.sum().backward()

    # The high-precision path quantizes A into this exact single-axis form.
    mx_a = MXTensor.from_hp(a.detach(), cfg, axis=-1)
    mx_a.requires_grad_(True)
    b_mx = b.clone().detach().requires_grad_(True)
    out = to_mx_then_grouped_mm(mx_a, b_mx, group_list, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized activation changed the forward"
    assert torch.equal(mx_a.grad, a_ref.grad), "the dgrad does not use A and must be identical"
    # The wgrad rebuilds A's group-aware dim=-2 form from the dequantized activation,
    # adding one quantization step, so it only has to stay close to the high-precision path.
    wgrad_sqnr = compute_error(b_mx.grad, b_ref.grad).item()
    assert wgrad_sqnr > 20.0, f"the wgrad SQNR vs the HP path is too low: {wgrad_sqnr:.2f} db"


def test_grouped_mm_accepts_pre_quantized_weight():
    """The grouped forward uses B's K-dim form; the dgrad rebuilds its N-dim form."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    torch.manual_seed(0)
    m, k, n, e = 128, 64, 128, 4
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)
    # Stored [e, n, k] and transposed, keeping the dim=-2 quant axis unit-stride.
    weight = torch.randn(e, n, k, device="npu", dtype=torch.bfloat16)
    b = weight.transpose(-1, -2)
    group_list = _make_group_list(m, e, "npu")
    cfg = MXQuantizeConfig()

    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    out_ref = to_mx_then_grouped_mm(a_ref, b_ref, group_list, cfg, cfg)
    out_ref.sum().backward()

    mx_b = DualAxisMXTensor.from_hp(b.detach(), cfg).to_mx_tensor(quant_axis=-2)
    mx_b.requires_grad_(True)
    a_mx = a.clone().detach().requires_grad_(True)
    out = to_mx_then_grouped_mm(a_mx, mx_b, group_list, cfg, cfg)
    out.sum().backward()

    assert torch.equal(out, out_ref), "the pre-quantized weight changed the forward"
    assert torch.equal(mx_b.grad, b_ref.grad), "the wgrad does not use B and must be identical"
    # The dgrad rebuilds B's dim=-1 form from the dequantized weight, adding one
    # quantization step, so it only has to stay close to the high-precision path.
    dgrad_sqnr = compute_error(a_mx.grad, a_ref.grad).item()
    assert dgrad_sqnr > 20.0, f"the dgrad SQNR vs the HP path is too low: {dgrad_sqnr:.2f} db"


def test_mm_rejects_mx_activation_with_a_different_config():
    """A pre-quantized activation must have been quantized with ``config_A``."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    mx_a = MXTensor.from_hp(a, MXQuantizeConfig(block_size=64), axis=-1)

    with pytest.raises(AssertionError, match="not consistent with ``config_A``"):
        to_mx_then_mm(mx_a, b, MXQuantizeConfig(), MXQuantizeConfig())


def test_mm_rejects_mx_weight_quantized_along_the_wrong_axis():
    """The weight's forward form must be quantized along ``dim=-2``."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    mx_b = MXTensor.from_hp(b, MXQuantizeConfig(), axis=-1)

    with pytest.raises(AssertionError, match="must be quantized along its second-to-last dim"):
        to_mx_then_mm(a, mx_b, MXQuantizeConfig(), MXQuantizeConfig())


def test_mm_rejects_mx_weight_with_a_different_config():
    """A pre-quantized weight must have been quantized with ``config_B``."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    mx_b = MXTensor.from_hp(b, MXQuantizeConfig(block_size=64), axis=-2)

    with pytest.raises(AssertionError, match="not consistent with ``config_B``"):
        to_mx_then_mm(a, mx_b, MXQuantizeConfig(), MXQuantizeConfig())


def test_mm_rejects_pre_quantized_weight_of_another_class():
    """Only an ``MXTensor`` or ``DualAxisMXTensor`` weight is accepted."""
    from torchao_npu.ops.mx_ops import to_mx_then_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(64, 256, device="npu", dtype=torch.bfloat16)
    mx_b = BlockMXTensor.from_hp(b, BlockMXQuantizeConfig())

    with pytest.raises(TypeError, match="B must be a high precision tensor, an ``MXTensor`` or a ``DualAxisMXTensor``"):
        to_mx_then_mm(a, mx_b, MXQuantizeConfig(), MXQuantizeConfig())


def test_bmm_rejects_mx_activation_quantized_along_the_wrong_axis():
    """The batched activation's forward form must be quantized along its last dim."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    b = torch.randn(2, 128, 64, device="npu", dtype=torch.bfloat16)
    mx_a = MXTensor.from_hp(a, MXQuantizeConfig(), axis=-2)

    with pytest.raises(AssertionError, match="must be quantized along its last dim"):
        to_mx_then_bmm(mx_a, b, MXQuantizeConfig(), MXQuantizeConfig())


def test_bmm_rejects_pre_quantized_activation_of_another_class():
    """Only an ``MXTensor`` or ``DualAxisMXTensor`` activation is accepted."""
    from torchao_npu.ops.mx_ops import to_mx_then_bmm

    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)
    b = torch.randn(2, 128, 64, device="npu", dtype=torch.bfloat16)
    mx_a = BlockMXTensor.from_hp(a, BlockMXQuantizeConfig())

    with pytest.raises(TypeError, match="A must be a high precision tensor, an ``MXTensor`` or a ``DualAxisMXTensor``"):
        to_mx_then_bmm(mx_a, b, MXQuantizeConfig(), MXQuantizeConfig())


def test_grouped_mm_rejects_mx_activation_quantized_along_the_wrong_axis():
    """The grouped activation's forward form must be quantized along its last dim."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(4, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = _make_group_list(128, 4, "npu")
    mx_a = MXTensor.from_hp(a, MXQuantizeConfig(), axis=-2)

    with pytest.raises(AssertionError, match="must be quantized along its last dim"):
        to_mx_then_grouped_mm(mx_a, b, group_list, MXQuantizeConfig(), MXQuantizeConfig())


def test_grouped_mm_rejects_pre_quantized_activation_of_another_class():
    """A ``DualAxisMXTensor`` cannot provide the group-aware wgrad form."""
    from torchao_npu.ops.mx_ops import to_mx_then_grouped_mm

    a = torch.randn(128, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(4, 64, 128, device="npu", dtype=torch.bfloat16)
    group_list = _make_group_list(128, 4, "npu")
    mx_a = DualAxisMXTensor.from_hp(a, MXQuantizeConfig())

    with pytest.raises(TypeError, match="A must be a high precision tensor or an MXTensor"):
        to_mx_then_grouped_mm(mx_a, b, group_list, MXQuantizeConfig(), MXQuantizeConfig())
