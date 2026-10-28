# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the behavior :class:`BaseQuantizedTensor` gives every subclass."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


def test_torch_function_layer_is_disabled():
    """Ops must bypass the function layer and reach ``__torch_dispatch__``."""
    assert MXTensor.__torch_function__ is torch._C._disabled_torch_function_impl


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize(
    "call",
    [
        lambda x: x.unsqueeze(0),
        lambda x: x.sum(),
        lambda x: x + 1,
        lambda x: x[0],
    ],
)
def test_unimplemented_ops_raise_not_implemented(shape, call):
    """Op support is a whitelist: anything outside it raises, naming the class."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(NotImplementedError, match=type(x).__name__):
        call(x)


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize("op", ["add_", "mul_", "div_"])
def test_in_place_ops_are_rejected(shape, op):
    """Values are frozen: op support is a whitelist, so every mutating op but ``copy_`` is rejected."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    qdata_before = x.qdata.clone()

    with pytest.raises(NotImplementedError, match=type(x).__name__):
        getattr(x, op)(2.0)

    assert x.qdata.shape == qdata_before.shape, (
        f"qdata shape changed after {op}: {tuple(x.qdata.shape)} != {tuple(qdata_before.shape)}"
    )
    assert torch.equal(x.qdata.view(torch.uint8), qdata_before.view(torch.uint8)), f"qdata values changed after {op}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_copy_in_place_writes_into_the_components(shape):
    """``copy_`` is the one in-place op a quantized tensor supports: it copies the components."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    source = MXTensor(
        torch.randn(shape).to(torch.float8_e4m3fn),
        torch.full((*shape[:-1], 1, 2), 4.0, dtype=torch.float8_e8m0fnu),
        torch.bfloat16,
        -1,
        config,
    )
    x_before, qdata_before, scale_before = x, x.qdata, x.scale

    result = x.copy_(source)

    assert result is x_before, f"copy_ returned a {type(result).__name__} instead of the tensor it copied into"
    assert x.qdata is qdata_before, "copy_ replaced qdata instead of writing into it"
    assert x.scale is scale_before, "copy_ replaced scale instead of writing into it"
    assert torch.equal(x.qdata.view(torch.uint8), source.qdata.view(torch.uint8)), "copy_ left qdata untouched"
    assert torch.equal(x.scale.view(torch.uint8), source.scale.view(torch.uint8)), "copy_ left scale untouched"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_clone_and_detach_preserve_class_and_metadata(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    for name, y in (("clone", x.clone()), ("detach", x.detach())):
        assert type(y) is type(x)

        assert y.qdata.shape == x.qdata.shape, (
            f"{name} changed the qdata shape: {tuple(y.qdata.shape)} != {tuple(x.qdata.shape)}"
        )
        assert torch.equal(y.qdata.view(torch.uint8), x.qdata.view(torch.uint8)), f"{name} changed the qdata values"

        assert y.scale.shape == x.scale.shape, (
            f"{name} changed the scale shape: {tuple(y.scale.shape)} != {tuple(x.scale.shape)}"
        )
        assert torch.equal(y.scale.view(torch.uint8), x.scale.view(torch.uint8)), f"{name} changed the scale values"

        assert y.quant_axis == x.quant_axis
        assert y.quant_config is x.quant_config
        assert y.dtype is x.dtype


def test_flatten_unflatten_round_trip():
    """The declared name lists are the serialization contract DCP relies on."""
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    act_config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, act_quant_config=act_config, pack_axis=-1)

    data_names, attributes = x.__tensor_flatten__()
    data = {name: getattr(x, name) for name in data_names}
    y = type(x).__tensor_unflatten__(data, attributes, None, None)

    assert data_names == ["qdata", "scale"]
    assert set(attributes) == {"dtype", "quant_axis", "quant_config", "act_quant_config", "pack_axis"}
    assert y.qdata.shape == x.qdata.shape, (
        f"unflattened qdata shape {tuple(y.qdata.shape)} != the original {tuple(x.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), x.qdata.view(torch.uint8)), (
        "unflattened qdata values differ from the original"
    )

    assert y.scale.shape == x.scale.shape, (
        f"unflattened scale shape {tuple(y.scale.shape)} != the original {tuple(x.scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), x.scale.view(torch.uint8)), (
        "unflattened scale values differ from the original"
    )

    assert y.quant_axis == x.quant_axis
    assert y.pack_axis == x.pack_axis
    assert y.act_quant_config is x.act_quant_config


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_dtype_is_the_tensors_own_dtype(shape):
    """The logical dtype is stored once, as the tensor's own ``dtype``."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert x.dtype is torch.bfloat16, f"the tensor's dtype is {x.dtype}"

    y = x.to(torch.float32)

    assert y.dtype is torch.float32, f".to(torch.float32) ended at {y.dtype}"
    assert y.shape == x.shape, f"the retag changed the logical shape: {tuple(y.shape)} != {tuple(x.shape)}"
    assert x.dtype is torch.bfloat16, f"the retag changed the source's dtype to {x.dtype}"
    assert y.qdata is x.qdata, "the retag replaced the stored qdata instead of sharing it"
    assert y.scale is x.scale, "the retag replaced the stored scale instead of sharing it"


def test_dtype_cannot_be_assigned():
    """The logical dtype is the tensor's own dtype, so ``dtype`` is read-only."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(AttributeError):
        x.dtype = torch.float32

    assert x.dtype is torch.bfloat16, f"the failed assignment changed dtype to {x.dtype}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_to_dtype_changes_only_the_logical_dtype(shape, dtype):
    """``.to(dtype)`` retags the tensor; the stored quantization is shared, not cast."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(dtype)

    assert y.dtype is dtype, f".to({dtype}) left the logical dtype at {y.dtype}"
    assert y.dtype is dtype, f".to({dtype}) left the logical dtype at {y.dtype}"
    assert y.qdata is x.qdata, f".to({dtype}) cast the stored qdata"
    assert y.scale is x.scale, f".to({dtype}) cast the stored scale"
    assert x.dtype is torch.bfloat16, f".to({dtype}) changed the original's dtype to {x.dtype}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_dtype_chains(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(torch.float32).to(torch.float16)

    assert y.dtype is torch.float16, f"chained .to calls ended at {y.dtype}, expected torch.float16"


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.float8_e4m3fn])
def test_to_unsupported_dtype_raises(dtype):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn((4, 64)).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="only supports the logical dtypes"):
        x.to(dtype)


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_same_dtype_is_a_no_op(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(torch.bfloat16)

    assert y.dtype is torch.bfloat16, f".to(same dtype) ended at {y.dtype}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_device_moves_the_components(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to("npu")

    assert y.qdata.device.type == "npu", f".to('npu') left qdata on {y.qdata.device}"
    assert y.scale.device.type == "npu", f".to('npu') left scale on {y.scale.device}"
    assert y.dtype is x.dtype, f".to('npu') changed the logical dtype to {y.dtype}"
    assert x.qdata.device.type == "cpu", f".to('npu') moved the original's qdata to {x.qdata.device}"

    # the move must not alter the values: moving the plain components ourselves
    # (bypassing MXTensor's handler) has to reproduce y's components exactly
    moved_qdata = qdata.to(y.qdata.device)
    assert torch.equal(y.qdata.view(torch.uint8), moved_qdata.view(torch.uint8)), ".to('npu') changed the qdata values"

    moved_scale = scale.to(y.scale.device)
    assert torch.equal(y.scale.view(torch.uint8), moved_scale.view(torch.uint8)), ".to('npu') changed the scale values"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_is_contiguous_follows_the_components(shape):
    """The tensor has no layout of its own: contiguity is the components' answer."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert x.is_contiguous(), f"a freshly built {shape} tensor should be contiguous"

    y = x.transpose(0, 1)

    assert not y.is_contiguous(), f"transposing {shape} left the components contiguous"


def test_is_contiguous_requires_every_component():
    """One strided component is enough: the tensor's answer is the components' ``all``, not either one's."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    contiguous_qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    strided_qdata = torch.randn(64, 4).to(torch.float8_e4m3fn).t()
    contiguous_scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    # A slice keeps the strides of the (4, 2, 2) it came from: a transpose would not do, since
    # torch ignores the strides of size-1 dims and would report this (4, 1, 2) as contiguous.
    strided_scale = torch.full((4, 2, 2), 2.0, dtype=torch.float8_e8m0fnu)[:, :1, :]

    x = MXTensor(contiguous_qdata, strided_scale, torch.bfloat16, -1, config)
    y = MXTensor(strided_qdata, contiguous_scale, torch.bfloat16, -1, config)

    assert x.qdata.is_contiguous() and not x.scale.is_contiguous(), "the premise: only the scale is strided"
    assert not y.qdata.is_contiguous() and y.scale.is_contiguous(), "the premise: only the qdata is strided"

    assert not x.is_contiguous(), "a strided scale must make the tensor non-contiguous"
    assert not y.is_contiguous(), "a strided qdata must make the tensor non-contiguous"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_is_pinned_reports_on_the_components(shape):
    """The tensor holds no storage of its own, so the components' storage answers for it: none of them is pinned."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    assert not x.is_pinned()


def test_is_pinned_requires_the_components_to_agree():
    """One component cannot answer for the other's storage, so a split answer is an error."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata.pin_memory(), scale, torch.bfloat16, -1, config)

    assert x.qdata.is_pinned() and not x.scale.is_pinned(), "the premise: only the qdata is pinned"

    with pytest.raises(RuntimeError, match="agree on ``is_pinned``"):
        x.is_pinned()


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_as_strided_accepts_the_layout_preserving_call(shape):
    """The identity ``as_strided`` is a no-op: the components are handed through unchanged."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.as_strided(x.shape, x.stride(), x.storage_offset())

    assert type(y) is type(x), f"as_strided returned a {type(y).__name__}"
    assert y.qdata is x.qdata, "as_strided replaced qdata instead of passing it through"
    assert y.scale is x.scale, "as_strided replaced scale instead of passing it through"


@pytest.mark.parametrize(
    "size, stride, storage_offset",
    [((4, 64), (64, 2), 0), ((4, 64), (64, 1), 1), ((2, 64), (64, 1), 0)],
    ids=["wider-stride", "later-offset", "fewer-rows"],
)
def test_as_strided_rejects_any_other_layout(size, stride, storage_offset):
    """A different layout is not something the stored components can describe, so it is rejected."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(NotImplementedError, match="layout-preserving"):
        x.as_strided(size, stride, storage_offset)


def test_a_subclass_omitting_dtype_is_rejected_at_class_creation():
    """``dtype`` is mandatory metadata: the check runs when the subclass is created."""

    with pytest.raises(TypeError, match="must declare"):

        class _NoDtype(BaseQuantizedTensor):
            tensor_attribute_names = ["quant_axis", "quant_config", "act_quant_config"]
