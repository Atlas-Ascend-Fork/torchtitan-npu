# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor`."""

import pytest
import torch
import torch_npu
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.grouped_mx_tensor import GroupedMXTensor

# =========================================================================
# Construction and validation
# =========================================================================


def test_fp8_construction():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)  # [M//64 + E, K, 2]

    x = GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config)

    assert x.shape == torch.Size([64, 128])
    assert x.dtype is torch.bfloat16
    assert x.quant_axis == 0  # normalized from -2
    assert x.pack_axis is None
    assert x.quant_config is config


def test_fp4_construction_doubles_the_logical_shape():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (64, 64), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)

    x = GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config, pack_axis=-1)

    assert x.shape == torch.Size([64, 128])  # qdata is (64, 64); packing doubles the last dim
    assert x.quant_axis == 0
    assert x.pack_axis == 1  # normalized from -1
    assert x.qdata.dtype is torch.float4_e2m1fn_x2


def test_construction_rejects_qdata_dtype_mismatch():
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)
    fp4_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    with pytest.raises(AssertionError, match="is not consistent with"):
        GroupedMXTensor(qdata, scale, torch.bfloat16, -2, fp4_config)


def test_construction_rejects_qdata_that_is_not_2d():
    """The groups split the rows, so the quantized tensor has to be a matrix."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64).to(torch.float8_e4m3fn)
    scale = torch.full((64, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match=r"requires qdata.ndim == 2"):
        GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config)


def test_construction_rejects_scale_with_wrong_ndim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((64, 2), 2.0, dtype=torch.float8_e8m0fnu)  # no trailing pack-2 dim

    with pytest.raises(ValueError, match=r"scale.ndim must be"):
        GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config)


def test_construction_rejects_pack_axis_on_non_fp4():
    """Values of an FP8 tensor are not packed, so a pack axis is meaningless."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="non-FP4 tensor are not packed"):
        GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config, pack_axis=1)


# =========================================================================
# Shape ops
# =========================================================================


@pytest.mark.parametrize(
    "swap",
    [lambda x: x.t(), lambda x: x.transpose(0, 1), lambda x: x.permute([1, 0])],
    ids=["t", "transpose", "permute"],
)
def test_swap_permutes_the_components(swap):
    """The two dims may be swapped: qdata and its scale follow, and so does ``quant_axis``."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config)

    y = swap(x)

    assert y.shape == torch.Size([128, 64]), f"swapped shape {tuple(y.shape)} != torch.Size([128, 64])"
    assert y.quant_axis == 1, f"quant_axis tracked as {y.quant_axis}, expected 1"
    assert y.pack_axis is None, f"an FP8 tensor gained pack_axis {y.pack_axis}"

    expected_qdata = x.qdata.t()
    assert y.qdata.shape == expected_qdata.shape, (
        f"swapped qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "swapped qdata values differ from the transposed original"
    )

    expected_scale = x.scale.permute([1, 0, 2])  # the leading dims follow the swap, the pack-2 dim stays last
    assert y.scale.shape == expected_scale.shape, (
        f"swapped scale shape {tuple(y.scale.shape)} != the expected {tuple(expected_scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected_scale.view(torch.uint8)), (
        "swapped scale values differ from the transposed original"
    )

    assert x.shape == torch.Size([64, 128]), f"the swap changed the source's shape to {tuple(x.shape)}"


def test_fp4_swap_tracks_the_pack_axis():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (64, 64), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config, pack_axis=-1)

    y = x.t()

    assert y.shape == torch.Size([128, 64]), f"swapped shape {tuple(y.shape)} != torch.Size([128, 64])"
    assert y.quant_axis == 1, f"quant_axis tracked as {y.quant_axis}, expected 1"
    assert y.pack_axis == 0, f"the packed dim moved to {y.pack_axis}, expected 0"


# =========================================================================
# Frozen values
# =========================================================================


def test_values_are_frozen():
    """Values cannot be changed in place."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(64, 128).to(torch.float8_e4m3fn)
    scale = torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = GroupedMXTensor(qdata, scale, torch.bfloat16, -2, config)

    with pytest.raises(NotImplementedError, match=type(x).__name__):
        x.zero_()


# =========================================================================
# Grouped matmul operand rules (no kernel involved)
# =========================================================================


def test_grouped_mm_requires_grouped_mx_operands():
    """Both operands carry the group-aware quantization the kernel reads."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = GroupedMXTensor(
        torch.randn(64, 64).to(torch.float8_e4m3fn),  # [K', N]
        torch.full((3, 64, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
    )
    offs = torch.tensor([16, 32], dtype=torch.int32)

    with pytest.raises(ValueError, match="left grouped_mm operand"):
        torch._grouped_mm(torch.randn(64, 64, dtype=torch.bfloat16), weight, offs=offs)

    with pytest.raises(ValueError, match="right grouped_mm operand"):
        torch._grouped_mm(weight.t(), torch.randn(64, 64, dtype=torch.bfloat16), offs=offs)


def test_grouped_mm_requires_quant_axes_matching_the_matmul():
    """The left operand is passed transposed (its blocks along dim -1); the right one as stored."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = GroupedMXTensor(
        torch.randn(64, 64).to(torch.float8_e4m3fn),  # stored [K, N], quantized along K
        torch.full((3, 64, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
    )
    offs = torch.tensor([16, 32], dtype=torch.int32)

    with pytest.raises(ValueError, match=r"quantized along dim=-1"):
        torch._grouped_mm(weight, weight.t(), offs=offs)  # quantized along dim -2: wrong for the left

    with pytest.raises(ValueError, match=r"quantized along dim=-2"):
        torch._grouped_mm(weight.t(), weight.t(), offs=offs)  # quantized along dim -1: wrong for the right


def test_grouped_mm_checks_the_contracting_dim():
    """``A``'s last dim is the contracted one and ``B``'s second-to-last: the two must match."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = GroupedMXTensor(
        torch.randn(64, 128).to(torch.float8_e4m3fn),  # [M, K]
        torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
    ).t()  # [K, M]: blocks along its last dim, as a left operand needs
    b = GroupedMXTensor(
        torch.randn(32, 64).to(torch.float8_e4m3fn),  # [K', N] with K' = 32, a's K = 64
        torch.full((2, 64, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
    )
    offs = torch.tensor([16, 32], dtype=torch.int32)

    with pytest.raises(ValueError, match="Contracting dim mismatch"):
        torch._grouped_mm(a, b, offs=offs)


def test_grouped_mm_requires_matching_block_size():
    """The kernel pairs blocks of one size, so operands with different ``block_size`` are rejected."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = GroupedMXTensor(
        torch.randn(64, 128).to(torch.float8_e4m3fn),  # [M, K]
        torch.full((3, 128, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
    ).t()  # [K, M]
    b = GroupedMXTensor(
        torch.randn(64, 64).to(torch.float8_e4m3fn),  # [K, N]
        torch.full((3, 64, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, block_size=64),
    )
    offs = torch.tensor([16, 32], dtype=torch.int32)

    with pytest.raises(ValueError, match="agree on ``block_size``"):
        torch._grouped_mm(a, b, offs=offs)


# =========================================================================
# Quantization against the op (NPU kernels)
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_from_hp_matches_npu_grouped_dynamic_mx_quant(elem_dtype):
    """``from_hp`` hands the config's parameters to the op and keeps its output as it is."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    offs = torch.tensor([16, 48, 64], device="npu", dtype=torch.int32)  # uneven groups

    y = GroupedMXTensor.from_hp(x, offs, config)
    qdata, scale = torch_npu.npu_grouped_dynamic_mx_quant(
        x,
        offs.to(torch.int32),
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        blocksize=config.block_size,
        scale_alg=config.scale_alg,
    )

    assert y.shape == x.shape, f"quantized shape {tuple(y.shape)} != the input's {tuple(x.shape)}"
    assert y.quant_axis == 0, f"quant_axis tracked as {y.quant_axis}, expected 0 (dim -2)"
    expected_pack_axis = y.ndim - 1 if elem_dtype is torch.float4_e2m1fn_x2 else None
    assert y.pack_axis == expected_pack_axis, f"pack axis {y.pack_axis}, expected {expected_pack_axis}"

    assert y.qdata.shape == qdata.shape, f"qdata shape {tuple(y.qdata.shape)} != the op's {tuple(qdata.shape)}"
    assert torch.equal(y.qdata.view(torch.uint8), qdata.view(torch.uint8)), "qdata values differ from the op's"

    assert y.scale.shape == scale.shape, f"scale shape {tuple(y.scale.shape)} != the op's {tuple(scale.shape)}"

    # The op writes a group's scale for its ``j``-th 64-row block at
    # ``group_start // 64 + group_idx + j``. Rows reserved for group boundaries
    # but never written are slack and hold uninitialized bytes, so only the
    # meaningful rows are comparable.
    group_ends = offs.tolist()
    rows_per_scale_row = 2 * config.block_size
    meaningful_rows = [
        start // rows_per_scale_row + group_idx + block_idx
        for group_idx, (start, end) in enumerate(zip([0, *group_ends[:-1]], group_ends, strict=True))
        for block_idx in range(-(-(end - start) // rows_per_scale_row))
    ]
    assert torch.equal(y.scale[meaningful_rows].view(torch.uint8), scale[meaningful_rows].view(torch.uint8)), (
        "scale values differ from the op's on the meaningful rows"
    )


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    x = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.tensor([32, 64], device="npu", dtype=torch.int32)

    y = GroupedMXTensor.from_hp(x, offs, config)

    assert y.requires_grad, "from_hp dropped the input's requires_grad"
