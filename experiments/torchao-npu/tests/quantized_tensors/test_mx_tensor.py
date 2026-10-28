# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`torchao_npu.quantized_tensors.mx_tensor.MXTensor`."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.ops.mx_ops import to_mx_then_bmm, to_mx_then_grouped_mm, to_mx_then_mm
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mx_fake_quantize
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

# =========================================================================
# Construction and validation
# =========================================================================


def test_fp8_construction():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert x.shape == torch.Size([4, 64])
    assert x.dtype is torch.bfloat16
    assert x.quant_axis == 1  # normalized from -1
    assert x.pack_axis is None
    assert x.quant_config is config


def test_fp4_construction_doubles_the_logical_shape():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    x = MXTensor(qdata, scale, torch.bfloat16, -2, config, pack_axis=-1)

    assert x.shape == torch.Size([4, 64])  # qdata is (4, 32); packing doubles axis 1
    assert x.quant_axis == 0
    assert x.pack_axis == 1  # normalized from -1
    assert x.qdata.dtype is torch.float4_e2m1fn_x2


def test_construction_rejects_qdata_dtype_mismatch():
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    fp4_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)

    with pytest.raises(AssertionError, match="is not consistent with"):
        MXTensor(qdata, scale, torch.bfloat16, -1, fp4_config)


def test_construction_rejects_scale_with_wrong_ndim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)  # ndim = qdata.ndim + 2

    with pytest.raises(ValueError, match=r"scale.ndim must be"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config)


def test_construction_rejects_pack_axis_on_non_fp4():
    """Values of an FP8 tensor are not packed, so a pack axis is meaningless."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="non-FP4 tensor are not packed"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=1)


def test_construction_rejects_fp4_without_pack_axis():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match="pack_axis should be set"):
        MXTensor(qdata, scale, torch.bfloat16, -1, config)


# =========================================================================
# Axis-preserving shape ops
# =========================================================================


def test_permute_tracks_quant_axis_and_keeps_content():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    perm = [0, 2, 1]

    y = x.permute(perm)

    assert y.quant_axis == 1  # was 2, tracked through the permutation
    assert y.pack_axis is None
    expected_qdata = x.qdata.permute(perm)
    assert y.qdata.shape == expected_qdata.shape, (
        f"permuted qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "permuted qdata values differ from the permuted original"
    )

    expected_scale = x.scale.permute([*perm, x.scale.ndim - 1])
    assert y.scale.shape == expected_scale.shape, (
        f"permuted scale shape {tuple(y.scale.shape)} != the expected {tuple(expected_scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected_scale.view(torch.uint8)), (
        "permuted scale values differ from the permuted original"
    )


def test_transpose_swaps_axes_and_content():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.transpose(-1, -2)

    assert y.shape == torch.Size([64, 4])
    assert y.quant_axis == 0
    expected_qdata = x.qdata.transpose(-1, -2)
    assert y.qdata.shape == expected_qdata.shape, (
        f"transposed qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "transposed qdata values differ from the transposed original"
    )

    expected_scale = x.scale.transpose(0, 1)
    assert y.scale.shape == expected_scale.shape, (
        f"transposed scale shape {tuple(y.scale.shape)} != the expected {tuple(expected_scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected_scale.view(torch.uint8)), (
        "transposed scale values differ from the transposed original"
    )


def test_t_matches_transpose_on_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.t()
    expected = x.transpose(0, 1)

    assert y.qdata.shape == expected.qdata.shape, (
        f"t() gave qdata shape {tuple(y.qdata.shape)}, transpose(0, 1) gave {tuple(expected.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)), (
        "t() and transpose(0, 1) produced different qdata values"
    )
    assert y.quant_axis == expected.quant_axis, (
        f"t() tracked quant_axis {y.quant_axis}, transpose(0, 1) tracked {expected.quant_axis}"
    )


def test_t_rejects_more_than_2d():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(RuntimeError, match="expects a tensor with <= 2 dimensions"):
        x.t()


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize(
    "shape",
    [(2, 64), (2, 3, 64), (2, 3, 4, 64), (2, 3, 4, 5, 64)],  # ..., 64: the quantized dim
    ids=["2d", "3d", "4d", "5d"],
)
@pytest.mark.parametrize(
    "source, destination",
    [(-1, 0), (0, 1), (0, -1)],
    ids=["last-to-first", "first-to-second", "first-to-last"],
)
def test_movedim_matches_the_equivalent_permute(elem_dtype, shape, source, destination):
    """``movedim`` decomposes into a permute, so the axis tracking has to follow it."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    if elem_dtype is torch.float4_e2m1fn_x2:
        # FP4 stores two values per byte along the last dim, so the stored dim is halved
        qdata = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        pack_axis = -1
    else:
        qdata = torch.randn(shape).to(torch.float8_e4m3fn)
        pack_axis = None
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=pack_axis)

    y = torch.movedim(x, source, destination)

    # movedim is defined as taking `source` out of the dims and putting it at `destination`
    perm = list(range(len(shape)))
    perm.insert(destination % len(shape), perm.pop(source))
    expected = x.permute(perm)
    expected_quant_axis = perm.index(len(shape) - 1)

    assert y.shape == expected.shape, (
        f"movedim({source}, {destination}) gave shape {tuple(y.shape)}, permute({perm}) gave {tuple(expected.shape)}"
    )
    assert y.quant_axis == expected_quant_axis, (
        f"movedim({source}, {destination}) tracked quant_axis {y.quant_axis}, expected {expected_quant_axis}"
    )
    assert y.pack_axis == (expected_quant_axis if elem_dtype is torch.float4_e2m1fn_x2 else None), (
        f"movedim({source}, {destination}) tracked pack_axis {y.pack_axis}, expected "
        f"{expected_quant_axis if elem_dtype is torch.float4_e2m1fn_x2 else None}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)), (
        "movedim and the equivalent permute produced different qdata"
    )
    assert torch.equal(y.scale.view(torch.uint8), expected.scale.view(torch.uint8)), (
        "movedim and the equivalent permute produced different scale"
    )


def test_fp4_shape_ops_track_the_pack_axis():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=-1)

    y = x.t()

    expected_qdata = x.qdata.t()

    assert y.pack_axis == 0, f"t() tracked pack_axis {y.pack_axis}, expected the packed dim to move to 0"
    assert y.quant_axis == 0, f"t() tracked quant_axis {y.quant_axis}, expected 0"
    assert y.qdata.shape == expected_qdata.shape, (
        f"transposed FP4 qdata shape {tuple(y.qdata.shape)} != the expected {tuple(expected_qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), expected_qdata.view(torch.uint8)), (
        "transposed FP4 qdata values differ from the transposed original"
    )


# =========================================================================
# Re-shaping (view / reshape / _unsafe_view)
# =========================================================================


@pytest.mark.parametrize(
    "destination",
    [(8, 64), (1, 2, 4, 64), (-1, 64)],
    ids=["merge", "insert-leading", "infer"],
)
def test_view_refactors_the_leading_dims(destination):
    """Leading dims re-factor freely; the components are re-viewed, never copied."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.view(destination)

    expected_qdata = x.qdata.view(destination)
    expected_scale = x.scale.view(*destination[:-1], *x.scale.shape[-2:])
    assert y.shape == expected_qdata.shape, f"view gave shape {tuple(y.shape)}, expected {expected_qdata.shape}"
    assert y.qdata.shape == expected_qdata.shape, (
        f"view gave qdata shape {tuple(y.qdata.shape)}, expected {expected_qdata.shape}"
    )
    assert y.scale.shape == expected_scale.shape, (
        f"view gave scale shape {tuple(y.scale.shape)}, expected {expected_scale.shape}"
    )
    assert y.quant_axis == y.ndim - 1, f"the quant axis must stay last, got {y.quant_axis}"
    assert y.pack_axis is None, f"an FP8 tensor gained pack_axis {y.pack_axis}"
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "view copied qdata instead of re-viewing it"
    assert y.scale.data_ptr() == x.scale.data_ptr(), "view copied scale instead of re-viewing it"


def test_view_refactors_the_leading_dims_with_an_unaligned_last_dim():
    """An unchanged last dim needs no pair alignment: its stored block slots are kept."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 3, 96).to(torch.float8_e4m3fn)  # 3 blocks per row: unaligned, odd
    scale = torch.full((2, 3, 2, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.view(6, 96)

    assert y.qdata.shape == torch.Size([6, 96]), f"qdata shape {tuple(y.qdata.shape)} != (6, 96)"
    assert y.scale.shape == torch.Size([6, 2, 2]), (
        f"scale shape {tuple(y.scale.shape)} != (6, 2, 2); the padded pair slot must be kept"
    )
    assert y.quant_axis == 1, f"quant axis {y.quant_axis} != 1"
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "view copied qdata instead of re-viewing it"


def test_view_merges_rows_when_the_last_dim_is_pair_aligned():
    """With whole block pairs per row, merging rows is a reshape of the flat pair sequence."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.view(-1)

    assert y.shape == torch.Size([256]), f"view gave shape {tuple(y.shape)}, expected (256,)"
    assert y.qdata.shape == torch.Size([256]), f"qdata shape {tuple(y.qdata.shape)} != (256,)"
    assert y.scale.shape == torch.Size([4, 2]), f"scale shape {tuple(y.scale.shape)} != (4, 2)"
    assert y.quant_axis == 0, f"quant axis {y.quant_axis} != 0"
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "view copied qdata instead of re-viewing it"


def test_view_splits_rows_when_the_last_dim_is_pair_aligned():
    """Splitting rows is the inverse: every new row still holds whole block pairs."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 128).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.view(4, 64)

    assert y.qdata.shape == torch.Size([4, 64]), f"qdata shape {tuple(y.qdata.shape)} != (4, 64)"
    assert y.scale.shape == torch.Size([4, 1, 2]), f"scale shape {tuple(y.scale.shape)} != (4, 1, 2)"
    assert y.quant_axis == 1, f"quant axis {y.quant_axis} != 1"
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "view copied qdata instead of re-viewing it"


def test_fp4_view_keeps_the_pack_axis_on_the_quant_axis():
    """FP4 halves the stored last dim; the pack axis follows the quant axis."""
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=-1)

    y = x.view(2, 2, 64)

    assert y.shape == torch.Size([2, 2, 64]), f"view gave shape {tuple(y.shape)}, expected (2, 2, 64)"
    assert y.qdata.shape == torch.Size([2, 2, 32]), f"qdata shape {tuple(y.qdata.shape)} != (2, 2, 32)"
    assert y.quant_axis == 2 and y.pack_axis == 2, (
        f"quant_axis/pack_axis are {y.quant_axis}/{y.pack_axis}, expected 2/2"
    )
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "view copied qdata instead of re-viewing it"


def test_fp4_view_flattens_when_the_last_dim_is_pair_aligned():
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=-1)

    y = x.view(-1)

    assert y.shape == torch.Size([256]), f"view gave shape {tuple(y.shape)}, expected (256,)"
    assert y.qdata.shape == torch.Size([128]), f"qdata shape {tuple(y.qdata.shape)} != (128,)"
    assert y.scale.shape == torch.Size([4, 2]), f"scale shape {tuple(y.scale.shape)} != (4, 2)"
    assert y.quant_axis == 0 and y.pack_axis == 0, (
        f"quant_axis/pack_axis are {y.quant_axis}/{y.pack_axis}, expected 0/0"
    )


def test_unsafe_view_reviews_the_components():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = torch.ops.aten._unsafe_view.default(x, [2, 2, 64])

    assert type(y) is MXTensor, f"_unsafe_view returned a {type(y).__name__}"
    assert y.qdata.shape == torch.Size([2, 2, 64]), f"qdata shape {tuple(y.qdata.shape)} != (2, 2, 64)"
    assert y.scale.shape == torch.Size([2, 2, 1, 2]), f"scale shape {tuple(y.scale.shape)} != (2, 2, 1, 2)"
    assert y.qdata.data_ptr() == x.qdata.data_ptr(), "_unsafe_view copied qdata instead of re-viewing it"


def test_public_reshape_rejects_non_contiguous_components():
    """``reshape`` decomposes to ``aten.view`` before dispatch, so it cannot copy.

    The wrapper's TensorImpl metadata is dense, so the composite takes the view
    path even though the components are strided; the view handler then forwards
    to the components, where torch refuses to merge the strided dims.
    """
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(2, 3, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 3, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    y = x.permute(1, 0, 2)  # quant axis stays last; the leading dims are strided
    assert not y.is_contiguous(), "the fixture should have non-contiguous components"

    with pytest.raises(RuntimeError, match="not compatible with input tensor's size and stride"):
        y.reshape(6, 64)

    # Re-shapes that do not merge the strided dims still work.
    assert y.reshape(3, 2, 64).shape == torch.Size([3, 2, 64])
    assert y.reshape(3, 2, 1, 64).shape == torch.Size([3, 2, 1, 64])


def test_view_rejects_a_quant_axis_that_is_not_last():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="quant axis to be the last dim"):
        x.t().view(-1)


def test_view_rejects_a_quant_axis_that_is_not_innermost_contiguous():
    """A strided quant axis cannot be re-viewed without moving the block boundaries."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn).t()  # [64, 4], stride(-1) == 64
    scale = torch.full((64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert x.qdata.stride(-1) != 1, "the fixture should keep a strided quant axis"
    with pytest.raises(ValueError, match="innermost-contiguous quant axis"):
        x.view(64, 2, 2)


@pytest.mark.parametrize(
    "shape, blocks, destination",
    [
        ((4, 32), 1, (-1,)),
        ((2, 96), 2, (-1,)),
        ((4, 64), 1, (8, 32)),
    ],
    ids=["source-not-pair-aligned", "source-block-count-odd", "destination-not-pair-aligned"],
)
def test_view_rejects_a_last_dim_change_that_breaks_block_pairs(shape, blocks, destination):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], blocks, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="whole packed block pairs"):
        x.view(destination)


def test_view_rejects_a_zero_dimensional_result():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(1, 1).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="at least a 1D"):
        x.view(())


# =========================================================================
# Frozen values
# =========================================================================


@pytest.mark.parametrize("op", ["zero_", "fill_"])
def test_in_place_ops_are_rejected(op):
    """Values are frozen: ``copy_`` is the only in-place op the class registers."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    qdata_before = x.qdata.clone()

    with pytest.raises(NotImplementedError, match=type(x).__name__):
        if op == "zero_":
            x.zero_()
        else:
            x.fill_(1.0)

    assert torch.equal(x.qdata.view(torch.uint8), qdata_before.view(torch.uint8)), f"qdata changed after {op}"


# =========================================================================
# Matmul operand rules (no kernel involved)
# =========================================================================


def test_mm_rejects_non_mx_right_operand():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="must be a MXTensor"):
        torch.mm(x, torch.randn(64, 8, dtype=torch.bfloat16))


def test_mm_requires_right_operand_quantized_along_dim_minus_2():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(64, 8).to(torch.float8_e4m3fn),
        torch.full((64, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,  # must be -2 for a right operand
        config,
        act_quant_config=config,
    )

    with pytest.raises(ValueError, match="quantized along dim=-2"):
        torch.mm(activation, weight)


def test_mm_requires_left_operand_quantized_along_dim_minus_1():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,  # must be -1 for a left operand
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="quantized along dim=-1"):
        torch.mm(activation, weight)


def test_mm_requires_matching_activation_config():
    """The left operand must be quantized with the right operand's act_quant_config."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    other_config = MXQuantizeConfig(elem_dtype=torch.float8_e5m2)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e5m2),  # stored [N, K]
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        other_config,
        act_quant_config=other_config,
    ).transpose(-2, -1)

    with pytest.raises(ValueError, match="act_quant_config"):
        torch.mm(activation, weight)


def test_mm_rejects_a_plain_left_operand_without_an_activation_config():
    """A high-precision left operand needs the right operand's act_quant_config."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,  # no act_quant_config
    ).transpose(-2, -1)

    with pytest.raises(RuntimeError, match="carries no"):
        torch.mm(torch.randn(4, 64, dtype=torch.bfloat16), weight)


def test_mm_requires_matching_block_size():
    """The kernel pairs blocks of one size, so operands with different ``block_size`` are rejected."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, block_size=64),
        act_quant_config=config,  # what the left operand was quantized with
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="agree on ``block_size``"):
        torch.mm(activation, weight)


def test_addmm_checks_operand_ranks():
    """``addmm`` is 2D-only, like the ``mm`` it wraps (plus a bias)."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = torch.randn(1, 4, 64, dtype=torch.bfloat16)  # 3D: rejected before anything else
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K]
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    )
    bias = torch.zeros(8, dtype=torch.float32)

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.addmm(bias, activation, weight.transpose(-2, -1))


def test_mm_checks_operand_ranks_and_shapes():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.mm(activation, weight)


def test_mm_out_dtype_checks_operand_ranks():
    """``torch.mm(..., out_dtype=...)`` is the ``aten.mm.dtype`` overload: same operand rules."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.mm(activation, weight, out_dtype=torch.float16)


def test_addmm_out_dtype_checks_operand_ranks():
    """``out_dtype`` is the fourth positional argument of the ``aten.addmm.dtype`` overload."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(1, 4, 64).to(torch.float8_e4m3fn),
        torch.full((1, 4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives
    bias = torch.zeros(8, dtype=torch.float32)

    with pytest.raises(ValueError, match="expects 2D operands"):
        torch.addmm(bias, activation, weight, out_dtype=torch.float16)


def test_bmm_out_dtype_checks_operand_ranks():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(8, 64).to(torch.float8_e4m3fn),  # stored [N, K], quantized along K
        torch.full((8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [K, N]: what a matmul receives
    activation = torch.randn(4, 64, dtype=torch.bfloat16)  # 2D: bmm's operands must be 3D

    with pytest.raises(ValueError, match="expects 3D operands"):
        torch.bmm(activation, weight, out_dtype=torch.float16)


def test_matmul_rejects_a_batched_weight():
    """``matmul``'s handler supports a 2D right operand only: a batched weight needs ``expand``."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    weight = MXTensor(
        torch.randn(2, 8, 64).to(torch.float8_e4m3fn),  # stored [B, N, K], quantized along K
        torch.full((2, 8, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
        act_quant_config=config,
    ).transpose(-2, -1)  # [B, K, N]: what a matmul receives
    activation = torch.randn(4, 64, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="2D right operand"):
        MXTensor.__torch_dispatch__(torch.ops.aten.matmul.default, (MXTensor,), (activation, weight), {})


# =========================================================================
# Matmul against the training path (NPU kernels)
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_mm_matches_to_mx_then_mm(elem_dtype):
    """A pre-quantized weight must give what the training path computes.

    ``to_mx_then_mm`` quantizes both operands itself; the MXTensor path reuses
    the weight's stored quantization. Matching them is the train/inference
    consistency this class exists for.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.dtype is torch.bfloat16
    assert torch.equal(y, y_ref)


def test_bmm_matches_to_mx_then_bmm():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_bmm(a, w.transpose(-2, -1), config, config)

    assert torch.equal(y, y_ref)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_out_dtype_is_honored(elem_dtype, out_dtype):
    """``torch.mm(..., out_dtype=...)`` takes the ``aten.mm.dtype`` overload into the kernel."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.mm(a, weight.transpose(-2, -1), out_dtype=out_dtype)
    y_default = torch.mm(a, weight.transpose(-2, -1))

    assert y.dtype is out_dtype, f"out_dtype {out_dtype} not honored: got {y.dtype}"
    if out_dtype is y_default.dtype:
        assert torch.equal(y, y_default), "asking for the dtype the product already has changed it"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_bmm_out_dtype_is_honored(elem_dtype):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # [B, M, K]
    w = torch.randn(2, 64, 128, device="npu", dtype=torch.bfloat16)  # stored [B, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.bmm(a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_addmm_out_dtype_is_honored(elem_dtype):
    """``out_dtype`` is the fourth positional argument of the ``aten.addmm.dtype`` overload."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), out_dtype=torch.float16)

    assert y.dtype is torch.float16, f"out_dtype not honored: got {y.dtype}"


@pytest.mark.parametrize(("beta", "alpha"), [(2, 1), (1, 3), (2, 3)])
def test_addmm_scales_the_bias_and_the_product(beta, alpha):
    """``beta``/``alpha`` scale the kernel's output: the fused epilogue has no scaling terms."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    a = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)  # stored [N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(64, device="npu", dtype=torch.float32)

    y = torch.addmm(bias, a, weight.transpose(-2, -1), beta=beta, alpha=alpha)
    y_ref = alpha * to_mx_then_mm(a, w.transpose(-2, -1), config, config) + beta * bias

    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "scaled addmm differs from the scaled training path"
    )


# =========================================================================
# Grouped matmul
# =========================================================================


def test_grouped_mm_checks_the_contracting_dim():
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    activation = MXTensor(
        torch.randn(4, 64).to(torch.float8_e4m3fn),
        torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    offs = torch.tensor([2, 4], dtype=torch.int32)

    weight_mismatched_k = MXTensor(
        torch.randn(2, 32, 8).to(torch.float8_e4m3fn),  # K = 32, activation's K = 64
        torch.full((2, 1, 8, 2), 2.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -2,
        config,
        act_quant_config=config,
    )

    with pytest.raises(ValueError, match="Contracting dim mismatch"):
        torch._grouped_mm(activation, weight_mismatched_k, offs=offs)


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_grouped_mm_matches_to_mx_then_grouped_mm(elem_dtype):
    """A pre-quantized expert weight must give what the training path computes."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    m, k, n, experts = 64, 128, 64, 4
    a = torch.randn(m, k, device="npu", dtype=torch.bfloat16)  # [M, K]
    w = torch.randn(experts, n, k, device="npu", dtype=torch.bfloat16)  # stored [E, N, K]
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    offs = torch.tensor([8, 32, 48, 64], device="npu", dtype=torch.int32)  # uneven groups

    y = torch._grouped_mm(a, weight.transpose(-2, -1), offs=offs)
    y_ref = to_mx_then_grouped_mm(a, w.transpose(-2, -1), offs, config, config)

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert torch.equal(y, y_ref), "grouped matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_matches_to_mx_then_mm(elem_dtype, shape):
    """``F.linear`` takes the stored ``[N, K]`` weight; that must give the training path's result.

    A 3D activation goes through the path ``F.linear`` itself flattens, so the shape it restores
    is checked along with the values. Every trailing dim is a multiple of 32, as the training
    path's dual-axis quantization of either operand requires.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.nn.functional.linear(a, weight)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "linear differs from the training path's"


@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_with_bias_matches_to_mx_then_mm(shape):
    """With a bias, ``F.linear`` routes through ``addmm``, which fuses the bias into the kernel."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)
    bias = torch.randn(n, device="npu", dtype=torch.float32)

    y = torch.nn.functional.linear(a, weight, bias)
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config) + bias

    assert y.dtype is torch.bfloat16, f"output dtype {y.dtype} != torch.bfloat16"
    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.isclose(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2).all().item(), (
        "linear with a bias differs from the training path's plus the bias"
    )


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_linear_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.linear`` handler itself, which ``F.linear`` never reaches.

    ``aten::linear`` is a composite op, decomposed into ``t``/``addmm`` before dispatch, so the
    handler runs only when something dispatches ``linear`` itself -- a backend registering it,
    as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way. The
    activation flattens to a multiple of 32 rows for the training path's sake.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.linear.default, (MXTensor,), (a, weight, None), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the linear handler differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_matches_to_mx_then_mm(elem_dtype, shape):
    """``torch.matmul`` against a stored ``[N, K]`` weight must give the training path's result.

    A 3D activation is folded into the rows by the composite ``matmul`` before its ``mm``, so the
    shape it restores is checked along with the values.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = torch.matmul(a, weight.transpose(-2, -1))
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "matmul differs from the training path's"


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("shape", [(64, 128), (2, 32, 128)])
def test_matmul_handler_matches_to_mx_then_mm(elem_dtype, shape):
    """The ``aten.matmul`` handler itself, which ``torch.matmul`` reaches only through its leaves.

    ``aten::matmul`` is a composite op, decomposed into ``mm``/``bmm``/``expand`` before dispatch,
    so the handler runs only when something dispatches ``matmul`` itself -- a backend registering
    it, as nested tensors do. Calling ``__torch_dispatch__`` directly exercises it either way.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    n, k = 64, 128
    a = torch.randn(*shape, device="npu", dtype=torch.bfloat16)  # [..., M, K]
    w = torch.randn(n, k, device="npu", dtype=torch.bfloat16)  # stored [N, K], quantized along K
    weight = MXTensor.from_hp(w, config, axis=-1, act_quant_config=config)

    y = MXTensor.__torch_dispatch__(torch.ops.aten.matmul.default, (MXTensor,), (a, weight.transpose(-2, -1)), {})
    y_ref = to_mx_then_mm(a, w.transpose(-2, -1), config, config)

    assert y.shape == (*shape[:-1], n), f"output shape {tuple(y.shape)} != {(*shape[:-1], n)}"
    assert torch.equal(y, y_ref), "the matmul handler differs from the training path's"


# =========================================================================
# Quantization (NPU kernels)
# =========================================================================


def test_from_hp_propagates_requires_grad():
    """A weight quantized for training keeps the autograd flag of the tensor it came from."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)

    x = MXTensor.from_hp(tensor, config, axis=-1)

    assert x.requires_grad, "from_hp dropped the input's requires_grad"


# =========================================================================
# Dequantize
# =========================================================================


@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float4_e2m1fn_x2])
def test_dequantize_matches_mx_fake_quantize(elem_dtype):
    """``dequantize`` undoes ``from_hp``'s quantization, reproducing ``mx_fake_quantize``.

    ``mx_fake_quantize`` quantizes and dequantizes the permuted tensor in one op, so for a
    tensor both permute the same way they see the same ``qdata``/``scale`` and must agree bit
    for bit.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    dequantized = x.dequantize()
    fake_quantized = mx_fake_quantize(tensor, -1, config)

    assert dequantized.shape == tensor.shape, f"shape mismatch: {dequantized.shape} vs {tensor.shape}"
    assert dequantized.dtype is tensor.dtype, f"dtype mismatch: {dequantized.dtype} vs {tensor.dtype}"
    assert dequantized.stride() == fake_quantized.stride(), (
        f"stride mismatch: {dequantized.stride()} vs {fake_quantized.stride()}"
    )
    assert torch.equal(dequantized, fake_quantized), "dequantized values differ from mx_fake_quantize"


def test_dequantize_honors_output_dtype():
    """Every MX value fits in bfloat16, so both output dtypes agree bit for bit."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    x = MXTensor.from_hp(tensor, config, axis=-1)

    as_bfloat16 = x.dequantize()
    as_float32 = x.dequantize(torch.float32)

    assert as_bfloat16.dtype is torch.bfloat16, f"dtype {as_bfloat16.dtype} != torch.bfloat16"
    assert as_float32.dtype is torch.float32, f"dtype {as_float32.dtype} != torch.float32"
    assert as_float32.stride() == as_bfloat16.stride(), (
        f"stride mismatch: {as_float32.stride()} vs {as_bfloat16.stride()}"
    )
    assert torch.equal(as_float32, as_bfloat16.to(torch.float32)), "the two output dtypes disagree"


def test_dequantize_rejects_a_strided_quant_axis():
    """A pass-through quantization keeps the quant axis strided, which cannot be dequantized."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    tensor = torch.randn(4, 256, 128, device="npu", dtype=torch.bfloat16)

    x = MXTensor.from_hp(tensor, config, axis=1)  # dense, non-trailing axis: no permutation
    assert x.qdata.stride(x.quant_axis) != 1, "the pass-through route should keep the quant axis strided"

    with pytest.raises(AssertionError, match="unit stride"):
        x.dequantize()


# =========================================================================
# Conversion to Block MX
# =========================================================================


def test_to_block_mx_hands_both_configs_and_the_axis_to_the_primitive(monkeypatch):
    """The tensor's own MX config is the source; the requested Block config is the target."""
    import torchao_npu.quantized_tensors.mx_tensor as mx_tensor_module

    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    act_config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    block_config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=config)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, act_quant_config=act_config, pack_axis=-1)
    seen = []

    def fake(fp4, mxscale, src_config, dst_config, *, quant_axis=None):
        seen.append((fp4, mxscale, src_config, dst_config, quant_axis))
        kernel_shape = (*fp4.shape[:-1], fp4.shape[-1] * 2)
        return (
            torch.zeros(kernel_shape, dtype=torch.float8_e4m3fn),
            torch.zeros(*kernel_shape[:-1], 1, 2, dtype=torch.float8_e8m0fnu),
            torch.zeros(*kernel_shape[:-2], 1, kernel_shape[-1], 2, dtype=torch.float8_e8m0fnu),
        )

    monkeypatch.setattr(mx_tensor_module, "mxfp4_to_block_mx", fake)

    y = x.to_block_mx(block_config)

    fp4, mxscale, src_config, dst_config, quant_axis = seen[0]
    assert fp4 is x.qdata and mxscale is x.scale, "to_block_mx rebuilt the payload instead of passing it"
    assert src_config is x.quant_config, "the source must be the tensor's own MX config"
    assert dst_config is block_config, "the destination must be the requested Block config"
    assert quant_axis == x.quant_axis, f"quant axis {quant_axis} != the tensor's {x.quant_axis}"

    assert type(y) is BlockMXTensor, f"to_block_mx returned a {type(y).__name__}"
    assert y.shape == x.shape, f"converted shape {tuple(y.shape)} != the source's {tuple(x.shape)}"
    assert y.dtype is torch.bfloat16, f"converted dtype {y.dtype} != torch.bfloat16"
    assert y.quant_config is block_config, "the converted tensor must keep the target config"
    assert y.act_quant_config is act_config, "the converted tensor must keep the activation config"
    assert y.pack_axis is None, f"an FP8 result gained pack_axis {y.pack_axis}"
    assert y.qdata.dtype is torch.float8_e4m3fn, f"converted qdata dtype {y.qdata.dtype}"


def test_to_block_mx_requires_the_quant_axis_to_be_the_packed_axis():
    """The conversion doubles the quant axis, so this tensor must be packed along it."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    block_config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)  # FP8: pack_axis is None

    with pytest.raises(AssertionError, match=r"quant_axis == self\.pack_axis"):
        x.to_block_mx(block_config)


def test_to_block_mx_requires_a_matching_mxfp4_config():
    """A target Block config describing another quantization cannot be used for this payload."""
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    other = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, block_size=64)
    block_config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=other)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, pack_axis=-1)

    with pytest.raises(AssertionError, match="not consistent"):
        x.to_block_mx(block_config)


def test_to_block_mx_matches_block_mx_quantize_of_the_same_tensor():
    """``from_hp`` + ``to_block_mx`` must reproduce the block-MX quantization of the same tensor.

    The two paths derive the kernel layout differently (``mx_quantize``'s stride rule vs
    ``_to_block_quant_layout``), so equal components pin the permutation and its inverse.
    """
    fp4_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    block_config = BlockMXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, mxfp4_fake_quantize_config=fp4_config)
    tensor = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)

    converted = MXTensor.from_hp(tensor, fp4_config, axis=-1).to_block_mx(block_config)
    reference = BlockMXTensor.from_hp(tensor, block_config, axis=-1)

    assert converted.pack_axis is None, f"an FP8 result gained pack_axis {converted.pack_axis}"
    for name in ("qdata", "scale1", "scale2"):
        actual = getattr(converted, name)
        expected = getattr(reference, name)
        assert actual.shape == expected.shape, f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            f"{name} values differ from the directly quantized block MX tensor"
        )
