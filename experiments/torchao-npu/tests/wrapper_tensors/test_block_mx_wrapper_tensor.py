# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

from ..testing_utils import _SizeOnlyMesh, target_devices


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


# =========================================================================
# fsdp_pre_all_gather
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
def test_fp8_pre_all_gather_payload_contract(device):
    """Three plain, contiguous components and the BlockMXTensor metadata."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    act_config = MXQuantizeConfig()
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=act_config,
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert len(payload) == 3
    qdata, scale1, scale2 = payload
    assert qdata.shape == (64, 128)
    assert scale1.shape == (64, 2, 2)
    assert scale2.shape == (1, 128, 2)
    for component in payload:
        assert component.is_contiguous()
        component.view(-1)  # FSDP flattens every input
    assert metadata["dtype"] is torch.bfloat16
    assert metadata["quant_config"] is config
    assert metadata["act_quant_config"] is act_config
    assert metadata["pack_axis"] is None


@pytest.mark.parametrize("device", target_devices)
def test_mxfp4_pre_all_gather_payload_contract(device):
    """Two plain, contiguous components and the MXTensor metadata."""
    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    act_config = MXQuantizeConfig()
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=act_config,
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert len(payload) == 2
    qdata, scale = payload
    assert qdata.shape == (64, 64)
    assert qdata.dtype is torch.float4_e2m1fn_x2
    assert scale.shape == (64, 2, 2)
    for component in payload:
        assert component.is_contiguous()
        component.view(-1)
    assert metadata["dtype"] is torch.bfloat16
    assert metadata["quant_config"] == config.mxfp4_fake_quantize_config
    assert metadata["act_quant_config"] is act_config
    assert metadata["quant_axis"] == metadata["pack_axis"] == 1


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("param_dtype", [None, torch.float16, torch.float32], ids=["none", "fp16", "fp32"])
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantized_tensor_respects_mixed_precision_policy(device, param_dtype, config_factory):
    """The represented dtype follows ``param_dtype``; the gathered payload stays quantized."""
    config = config_factory()
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=param_dtype),
    )

    expected_dtype = torch.bfloat16 if param_dtype is None else param_dtype
    assert metadata["dtype"] is expected_dtype
    # Only the represented dtype changes; the all-gathered components stay FP8/FP4 + E8M0.
    assert all(component.dtype not in (torch.float16, torch.float32) for component in payload)

    output, _ = wrapper.fsdp_post_all_gather(payload, metadata, expected_dtype)

    expected_data_type = MXTensor if config.fsdp_prequantize_dtype == "mxfp4" else BlockMXTensor
    assert output.dtype is expected_dtype
    assert isinstance(output._data, expected_data_type), (
        f"expected a {expected_data_type.__name__} ``_data``, got {type(output._data).__name__}"
    )
    assert output._data.dtype is expected_dtype


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "shape",
    [(64, 128), (128, 64), (4, 64, 64)],
    ids=["rows_aligned", "cols_aligned", "experts"],
)
def test_fp8_prequantize_accepts_aligned_shards(device, shape):
    """Both trailing dims must divide the 64-element quantization block."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(shape, dtype=torch.bfloat16, device=device),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    payload, _ = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert len(payload) == 3


@pytest.mark.parametrize(
    "shape",
    [(32, 128), (64, 32), (64, 96)],
    ids=["short_rows", "short_cols", "cols_not_block_aligned"],
)
def test_fp8_prequantize_rejects_misaligned_shards(shape):
    """A shard whose trailing dims are not 64-aligned would be padded mid-gather."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(shape),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(RuntimeError, match="Cannot quantize sharded weight before FSDP"):
        wrapper.fsdp_pre_all_gather(
            _SizeOnlyMesh(2),
            wrapper.shape,
            wrapper.stride(),
            None,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("device", target_devices)
def test_fp8_prequantize_alignment_scales_with_block_size(device):
    """block_size=64 requires 128 alignment, not 64."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True, block_size=64)

    misaligned = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    with pytest.raises(RuntimeError, match="Cannot quantize sharded weight before FSDP"):
        misaligned.fsdp_pre_all_gather(
            _SizeOnlyMesh(2),
            misaligned.shape,
            misaligned.stride(),
            None,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )

    aligned = BlockMXTrainingWeightWrapperTensor(
        torch.randn(128, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    payload, _ = aligned.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        aligned.shape,
        aligned.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )
    assert len(payload) == 3


@pytest.mark.parametrize(
    "shape, accepted",
    [((32, 128), True), ((64, 64), True), ((64, 96), False), ((64, 32), False)],
    ids=["short_rows_ok", "aligned", "cols_not_block_aligned", "short_cols"],
)
@pytest.mark.parametrize("device", target_devices)
def test_mxfp4_prequantize_checks_only_the_last_dim(device, shape, accepted):
    """MXFP4 packs along the last dim; dim -2 may be shorter than a block."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(shape, dtype=torch.bfloat16, device=device),
        weight_config=BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
        activation_config=MXQuantizeConfig(),
    )

    if accepted:
        payload, _ = wrapper.fsdp_pre_all_gather(
            _SizeOnlyMesh(2),
            wrapper.shape,
            wrapper.stride(),
            None,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        assert len(payload) == 2
    else:
        with pytest.raises(RuntimeError, match="Cannot quantize sharded weight before FSDP"):
            wrapper.fsdp_pre_all_gather(
                _SizeOnlyMesh(2),
                wrapper.shape,
                wrapper.stride(),
                None,
                MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            )


def test_prequantize_falls_back_on_size_one_mesh():
    """A size-1 mesh has no collective: the HP shard is gathered even when enabled."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(1),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert metadata == ()
    assert len(payload) == 1
    assert payload[0].dtype is torch.bfloat16
    assert payload[0].data_ptr() == wrapper._data.data_ptr()


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantize_quantizes_the_hp_payload_in_post_all_gather(device, config_factory):
    """The size-1-mesh fallback still quantizes ``_data`` in post-all-gather."""
    config = config_factory()
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(1),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )
    assert len(payload) == 1 and payload[0].dtype is torch.bfloat16

    output, inner = wrapper.fsdp_post_all_gather(payload, metadata, torch.bfloat16)

    expected_type = MXTensor if config.fsdp_prequantize_dtype == "mxfp4" else BlockMXTensor
    assert isinstance(output._data, expected_type), (
        f"expected a {expected_type.__name__} ``_data``, got {type(output._data).__name__}"
    )
    assert output._data.shape == wrapper.shape
    assert output._data.dtype is torch.bfloat16
    if config.fsdp_prequantize_dtype == "mxfp4":
        assert output._data.quant_config == config.mxfp4_fake_quantize_config
        assert output._data.qdata.dtype is torch.float4_e2m1fn_x2
    assert inner is payload


def test_prequantize_disabled_falls_back_to_hp_payload():
    """With the flag off, misaligned shards are gathered in high precision."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(32, 32, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=False),
        activation_config=MXQuantizeConfig(),
    )

    payload, metadata = wrapper.fsdp_pre_all_gather(
        _SizeOnlyMesh(2),
        wrapper.shape,
        wrapper.stride(),
        None,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16),
    )

    assert metadata == ()
    assert len(payload) == 1
    assert payload[0].dtype is torch.bfloat16


# =========================================================================
# fsdp_post_all_gather
# =========================================================================


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_post_all_gather_step0_rebuilds_wrapper_from_gathered_components(device):
    """FP8 step 0 returns a wrapper whose components alias the gathered payloads."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    act_config = MXQuantizeConfig()
    source = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=act_config,
    )
    quantized = BlockMXTensor.from_hp(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        config,
        act_quant_config=MXQuantizeConfig(),
    )
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)

    output, inner = source.fsdp_post_all_gather(payload, metadata, torch.bfloat16)

    assert isinstance(output, BlockMXTrainingWeightWrapperTensor)
    assert isinstance(output._data, BlockMXTensor)
    assert output.dtype is torch.bfloat16
    components = (output._data.qdata, output._data.scale1, output._data.scale2)
    for component, gathered in zip(components, payload, strict=True):
        assert component.data_ptr() == gathered.data_ptr()
    assert inner is payload
    assert output.weight_config is config
    assert output.activation_config is act_config

    # FSDP step 0 reshapes the gathered shard with as_strided; the wrapper survives.
    strided = torch.as_strided(output, output.shape, output.stride())
    assert isinstance(strided, BlockMXTrainingWeightWrapperTensor)


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_post_all_gather_mxfp4_step0_keeps_the_mx_payload(device):
    """MXFP4 step 0 keeps the gathered MX payload as ``_data``."""
    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
    quantized = MXTensor.from_hp(
        weight, config.mxfp4_fake_quantize_config, axis=-1, act_quant_config=MXQuantizeConfig()
    )
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)
    source = BlockMXTrainingWeightWrapperTensor(
        weight,
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    output, inner = source.fsdp_post_all_gather(payload, metadata, torch.bfloat16)

    assert isinstance(output._data, MXTensor), f"expected an MXTensor, got {type(output._data).__name__}"
    assert output._data.shape == torch.Size([64, 128])
    # ``__init__`` detaches ``_data``; the detached components still share the payload storages.
    assert output._data.qdata.data_ptr() == payload[0].data_ptr()
    assert output._data.scale.data_ptr() == payload[1].data_ptr()
    assert inner is payload


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_step1_out_rebinds_data_and_preserves_payload_storage(device):
    """FP8 step 1+ rebinds ``out._data`` without copying into the payload buffers."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    act_config = MXQuantizeConfig()
    source = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=act_config,
    )
    quantized = BlockMXTensor.from_hp(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        config,
        act_quant_config=MXQuantizeConfig(),
    )
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)
    sizes = [tensor.untyped_storage().nbytes() for tensor in payload]
    for tensor, size in zip(payload, sizes, strict=True):
        tensor.untyped_storage().resize_(0)
        tensor.untyped_storage().resize_(size)
    pointers = [tensor.data_ptr() for tensor in payload]
    versions = [tensor._version for tensor in payload]

    out = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    out.weight_config = None
    out.activation_config = None
    result = source.fsdp_post_all_gather(payload, metadata, torch.bfloat16, out=out)

    assert result is None
    assert isinstance(out._data, BlockMXTensor)
    assert [out._data.qdata.data_ptr(), out._data.scale1.data_ptr(), out._data.scale2.data_ptr()] == pointers
    assert [tensor.data_ptr() for tensor in payload] == pointers
    assert [tensor._version for tensor in payload] == versions
    assert out.weight_config is config
    assert out.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_step1_out_mxfp4_rebinds_the_mx_data(device):
    """MXFP4 step 1+ rebinds ``out._data`` to the gathered MX payload."""
    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    act_config = MXQuantizeConfig()
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
    quantized = MXTensor.from_hp(weight, config.mxfp4_fake_quantize_config, axis=-1, act_quant_config=act_config)
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)
    pointers = [tensor.data_ptr() for tensor in payload]
    versions = [tensor._version for tensor in payload]

    source = BlockMXTrainingWeightWrapperTensor(
        weight,
        weight_config=config,
        activation_config=act_config,
    )
    out = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    out.weight_config = None
    out.activation_config = None
    result = source.fsdp_post_all_gather(payload, metadata, torch.bfloat16, out=out)

    assert result is None
    assert isinstance(out._data, MXTensor), f"expected an MXTensor, got {type(out._data).__name__}"
    assert [tensor.data_ptr() for tensor in payload] == pointers
    assert [tensor._version for tensor in payload] == versions
    assert out._data.qdata.data_ptr() == payload[0].data_ptr()
    assert out._data.scale.data_ptr() == payload[1].data_ptr()
    assert out.weight_config is config
    assert out.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_post_all_gather_dtensor_local_out(device, mock_distributed_env):
    """The ``out=DTensor`` path restores configs on the wrapped local tensor."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    act_config = MXQuantizeConfig()
    source = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=act_config,
    )
    quantized = BlockMXTensor.from_hp(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        config,
        act_quant_config=MXQuantizeConfig(),
    )
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)
    mesh = DeviceMesh(device, torch.arange(1))
    out_local = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    out = DTensor.from_local(out_local, mesh, [Shard(0)])
    local = out._local_tensor
    local.weight_config = None
    local.activation_config = None

    result = source.fsdp_post_all_gather(payload, metadata, torch.bfloat16, out=out)

    assert result is None
    assert isinstance(local._data, BlockMXTensor)
    assert local.weight_config is config
    assert local.activation_config is act_config


@pytest.mark.parametrize("device", target_devices)
def test_fsdp_post_all_gather_rejects_wrong_out_type(device):
    """A non-wrapper ``out`` must fail loudly, naming the expected type."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    source = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )
    quantized = BlockMXTensor.from_hp(
        torch.randn(64, 128, dtype=torch.bfloat16, device=device),
        config,
        act_quant_config=MXQuantizeConfig(),
    )
    names, metadata = quantized.__tensor_flatten__()
    payload = tuple(getattr(quantized, name) for name in names)

    with pytest.raises(RuntimeError, match="expected out to be BlockMXTrainingWeightWrapperTensor"):
        source.fsdp_post_all_gather(payload, metadata, torch.bfloat16, out=torch.empty(64, 128, device=device))


@pytest.mark.parametrize("num_components", [0, 4], ids=["none", "too_many"])
def test_fsdp_post_all_gather_rejects_unsupported_payload_counts(num_components):
    """Only 1 (HP), 2 (MXFP4), and 3 (FP8) component payloads exist."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(RuntimeError, match=r"`len\(all_gather_outputs\)` in \(1, 2, 3\)"):
        wrapper.fsdp_post_all_gather(tuple(torch.empty(0) for _ in range(num_components)), {}, torch.bfloat16)


def test_fsdp_post_all_gather_single_payload_requires_hp_dtype():
    """The 1-component fallback is only valid for a high-precision gathered shard."""
    wrapper = BlockMXTrainingWeightWrapperTensor(
        torch.randn(64, 128, dtype=torch.bfloat16),
        weight_config=BlockMXQuantizeConfig(fsdp_prequantize=True),
        activation_config=MXQuantizeConfig(),
    )

    with pytest.raises(AssertionError, match="expected to be in high precision"):
        wrapper.fsdp_post_all_gather((torch.empty(64, 128, dtype=torch.float8_e4m3fn),), (), torch.bfloat16)


# =========================================================================
# Real NPU kernels: round trip, dispatch equivalence, autograd
# =========================================================================


def _quant_fsdp(full, config, num_shards=2):
    """Run the pre/post hooks over evenly sharded chunks of ``full``."""
    wrappers = [
        BlockMXTrainingWeightWrapperTensor(
            chunk.contiguous(),
            weight_config=config,
            activation_config=MXQuantizeConfig(),
        )
        for chunk in full.chunk(num_shards)
    ]
    results = [
        wrapper.fsdp_pre_all_gather(
            _SizeOnlyMesh(2),
            wrapper.shape,
            wrapper.stride(),
            None,
            MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        for wrapper in wrappers
    ]
    num_components = len(results[0][0])
    payload = tuple(torch.cat([result[0][i] for result in results]) for i in range(num_components))
    output, _ = wrappers[0].fsdp_post_all_gather(payload, results[0][1], torch.bfloat16)
    return output


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("shape", [(128, 128), (4, 128, 128)], ids=["matrix", "experts"])
def test_fp8_prequantize_round_trip_matches_full_weight_quantization(device, shape):
    """Per-shard quantization + gather must equal quantizing the full weight."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    full = torch.randn(shape, device=device, dtype=torch.bfloat16)

    output = _quant_fsdp(full, config)
    reference = BlockMXTensor.from_hp(full, config, act_quant_config=MXQuantizeConfig())

    assert torch.equal(output._data.qdata.view(torch.uint8), reference.qdata.view(torch.uint8))
    assert torch.equal(output._data.scale1.view(torch.uint8), reference.scale1.view(torch.uint8))
    assert torch.equal(output._data.scale2.view(torch.uint8), reference.scale2.view(torch.uint8))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("shape", [(128, 128), (4, 128, 128)], ids=["matrix", "experts"])
def test_mxfp4_prequantize_round_trip_matches_full_weight_quantization(device, shape):
    """MXFP4 gather must equal quantizing the full weight."""
    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    full = torch.randn(shape, device=device, dtype=torch.bfloat16)
    nested_config = config.mxfp4_fake_quantize_config
    assert nested_config is not None

    output = _quant_fsdp(full, config)
    reference = MXTensor.from_hp(full, nested_config, axis=-1, act_quant_config=MXQuantizeConfig())

    assert isinstance(output._data, MXTensor), f"expected an MXTensor, got {type(output._data).__name__}"
    assert output._data.quant_axis == reference.quant_axis
    assert output._data.pack_axis == reference.pack_axis
    assert torch.equal(output._data.qdata.view(torch.uint8), reference.qdata.view(torch.uint8))
    assert torch.equal(output._data.scale.view(torch.uint8), reference.scale.view(torch.uint8))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("shape", [(128, 128), (4, 128, 128)], ids=["matrix_rows", "experts"])
def test_mxfp4_payload_converts_to_the_fp8_payload(device, shape):
    """The kept MXFP4 payload still converts to the FP8 block-MX payload it replaces."""
    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16)
    output = _quant_fsdp(weight, config)
    # Keep the nested FP4 config: this is the QAT block-quantize path the MXFP4
    # pre-quantize must reproduce once converted.
    reference = BlockMXTensor.from_hp(
        weight,
        replace(config, fsdp_prequantize_dtype="fp8"),
        axis=-1,
        act_quant_config=MXQuantizeConfig(),
    )

    converted = output._data.to_block_mx(config)

    assert torch.equal(converted.qdata.view(torch.uint8), reference.qdata.view(torch.uint8))
    assert torch.equal(converted.scale1.view(torch.uint8), reference.scale1.view(torch.uint8))
    assert torch.equal(converted.scale2.view(torch.uint8), reference.scale2.view(torch.uint8))
    assert converted.is_contiguous()


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantized_forward_matches_on_the_fly(device, config_factory):
    """A pre-quantized weight must multiply exactly like the on-the-fly path."""
    config = config_factory()
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    activation = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    bias = torch.randn(256, device=device, dtype=torch.bfloat16)
    prequantized = _quant_fsdp(weight, config)
    on_the_fly = BlockMXTrainingWeightWrapperTensor(
        weight.clone(),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    assert torch.equal(torch.mm(activation, prequantized.T), torch.mm(activation, on_the_fly.T))
    assert torch.equal(torch.matmul(activation, prequantized.T), torch.matmul(activation, on_the_fly.T))
    assert torch.equal(F.linear(activation, prequantized), F.linear(activation, on_the_fly))
    assert torch.equal(F.linear(activation, prequantized, bias), F.linear(activation, on_the_fly, bias))
    assert torch.equal(torch.addmm(bias, activation, prequantized.T), torch.addmm(bias, activation, on_the_fly.T))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantized_bmm_matches_on_the_fly(device, config_factory):
    """The ``BatchedLinear`` view+transpose sequence multiplies the same either way.

    With the mxfp4 config the gathered MXFP4 weight is converted to block MX in the
    forward, so the batched matmul still runs FP8 x FP8.
    """
    config = config_factory()
    num_heads, out_features, in_features, tokens = 8, 64, 128, 8
    weight = torch.randn(num_heads * out_features, in_features, device=device, dtype=torch.bfloat16)
    activation = torch.randn(num_heads, tokens, in_features, device=device, dtype=torch.bfloat16)
    prequantized = _quant_fsdp(weight, config)
    on_the_fly = BlockMXTrainingWeightWrapperTensor(
        weight.clone(),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    def batched(wrapper):
        return torch.bmm(activation, wrapper.view(num_heads, out_features, in_features).transpose(-2, -1))

    assert torch.equal(batched(prequantized), batched(on_the_fly))


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantized_grouped_mm_matches_on_the_fly(device, config_factory):
    """Expert weights gathered across ranks multiply like the on-the-fly path.

    With the mxfp4 config the gathered MXFP4 weight is converted to block MX in the
    forward, so the grouped matmul still runs FP8 x FP8.
    """
    config = config_factory()
    tokens, num_experts, in_features, out_features = 16, 4, 1024, 2048
    activation = torch.randn(tokens, in_features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(num_experts, out_features, in_features, device=device, dtype=torch.bfloat16)
    offs = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=device)
    prequantized = _quant_fsdp(weight, config)
    on_the_fly = BlockMXTrainingWeightWrapperTensor(
        weight.clone(),
        weight_config=config,
        activation_config=MXQuantizeConfig(),
    )

    assert torch.equal(
        torch._grouped_mm(activation, prequantized.transpose(-2, -1), offs=offs),
        torch._grouped_mm(activation, on_the_fly.transpose(-2, -1), offs=offs),
    )


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: BlockMXQuantizeConfig(fsdp_prequantize=True),
        lambda: BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        ),
    ],
    ids=["fp8", "mxfp4"],
)
def test_prequantized_forward_does_not_requantize_weight(device, monkeypatch, config_factory):
    """The matmul consumer reuses the gathered quantized tensor instead of re-quantizing.

    An MXFP4 weight is still converted to block MX on every forward; the spy checks that
    it is not re-quantized from the high-precision weight.
    """
    config = config_factory()
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    activation = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    prequantized = _quant_fsdp(weight, config)

    quantized_shapes = []

    def spy(original):
        def record(cls, tensor, *args, **kwargs):
            quantized_shapes.append(tuple(tensor.shape))
            return original(cls, tensor, *args, **kwargs)

        return record

    monkeypatch.setattr(BlockMXTensor, "from_hp", classmethod(spy(BlockMXTensor.from_hp.__func__)))
    monkeypatch.setattr(MXTensor, "from_hp", classmethod(spy(MXTensor.from_hp.__func__)))

    F.linear(activation, prequantized)

    assert tuple(weight.shape) not in quantized_shapes


@pytest.mark.parametrize("device", target_devices)
def test_prequantized_backward_keeps_data_grad_free(device):
    """Gradients match the on-the-fly path and land on the wrapper, not ``_data``."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    prequantized_param = nn.Parameter(_quant_fsdp(weight, config))
    on_the_fly_param = nn.Parameter(
        BlockMXTrainingWeightWrapperTensor(
            weight.clone(),
            weight_config=config,
            activation_config=MXQuantizeConfig(),
        )
    )

    # ``_data`` is built detached; autograd tracks the parameter. Check it before
    # the forward: ``unwrap_weight`` returns ``_data`` from inside an autograd
    # Function, which marks the returned tensor as requiring grad in place.
    assert prequantized_param._data.requires_grad is False

    activation = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    F.linear(activation, prequantized_param).sum().backward()

    reference_activation = activation.detach().clone().requires_grad_(True)
    F.linear(reference_activation, on_the_fly_param).sum().backward()

    assert prequantized_param.grad is not None
    assert torch.equal(prequantized_param.grad, on_the_fly_param.grad)
    assert torch.equal(activation.grad, reference_activation.grad)


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize("prequantize_dtype", ["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "experts, world_size",
    [(3, 2), (4, 2)],
    ids=["uneven_3of2", "even_4of2"],
)
def test_fsdp_uneven_shard_round_trip(device, prequantize_dtype, experts, world_size):
    """Uneven dim-0 shards pad every quantized component on dim 0 and trim it back
    on gather, so the rebuilt weight equals the unsharded quantization.

    ``experts=3`` over 2 ranks forces the uneven path (rank 1 holds 1 of the 2
    largest-chunk rows); ``experts=4`` is the even control that must stay put.
    """
    if prequantize_dtype == "fp8":
        config = BlockMXQuantizeConfig(fsdp_prequantize=True, fsdp_prequantize_dtype="fp8")
    else:
        config = BlockMXQuantizeConfig(
            fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
        )
    activation_config = MXQuantizeConfig()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    weight = torch.randn(experts, 128, 128, device=device, dtype=torch.bfloat16)
    padded_dim0 = -(-experts // world_size)
    uneven = experts % world_size != 0

    wrappers = [
        BlockMXTrainingWeightWrapperTensor(
            shard.contiguous(), weight_config=config, activation_config=activation_config
        )
        for shard in weight.chunk(world_size)
    ]
    # Checkpointing reads the local shard's bytes, so pad must live only in the
    # collective's copies: ``fsdp_pre_all_gather`` must leave every wrapper alone
    hp_before = [w._data.clone() for w in wrappers]
    results = [
        w.fsdp_pre_all_gather(_SizeOnlyMesh(world_size), weight.shape, weight.stride(), None, mp_policy)
        for w in wrappers
    ]
    n_components = len(results[0][0])
    for w, before in zip(wrappers, hp_before, strict=True):
        assert torch.equal(w._data, before)

    for inputs, metadata in results:
        assert all(t.size(0) == padded_dim0 for t in inputs)
        if uneven:
            assert metadata["_uneven_logical_dim0"] == experts
        else:
            assert "_uneven_logical_dim0" not in metadata

    gathered = tuple(torch.cat([r[0][i] for r in results]) for i in range(n_components))
    assert all(t.size(0) == padded_dim0 * world_size for t in gathered)

    # Reference: whole-weight quantization, byte-identical to what the uneven
    # pre→gather→post round trip must reconstruct. Post keeps the wire format:
    # fp8 payloads rebuild a BlockMXTensor, mxfp4 payloads stay an MXTensor.
    if prequantize_dtype == "fp8":
        reference = BlockMXTensor.from_hp(weight.contiguous(), config, axis=-1, act_quant_config=activation_config)
    else:
        reference = MXTensor.from_hp(
            weight.contiguous(),
            config.mxfp4_fake_quantize_config,
            axis=-1,
            act_quant_config=activation_config,
        )

    def assert_rebuilt_matches_reference(data):
        assert type(data) is type(reference)
        assert torch.equal(data.qdata.view(torch.uint8), reference.qdata.view(torch.uint8))
        if prequantize_dtype == "mxfp4":
            assert torch.equal(data.scale.view(torch.uint8), reference.scale.view(torch.uint8))
        else:
            assert torch.equal(data.scale1.view(torch.uint8), reference.scale1.view(torch.uint8))
            assert torch.equal(data.scale2.view(torch.uint8), reference.scale2.view(torch.uint8))

    # Step 0 (out=None): logical dim-0 output, full-size inner buffers for reuse.
    output, inner = wrappers[0].fsdp_post_all_gather(gathered, results[0][1], torch.bfloat16)
    assert output.shape == weight.shape
    assert [tuple(t.shape) for t in inner] == [tuple(t.shape) for t in gathered]
    assert_rebuilt_matches_reference(output._data)

    # Step 1+ (out=pre-allocated unsharded wrapper): in-place, same logical size.
    out_alloc = BlockMXTrainingWeightWrapperTensor(
        weight.clone(), weight_config=config, activation_config=activation_config
    )
    assert wrappers[0].fsdp_post_all_gather(gathered, results[0][1], torch.bfloat16, out=out_alloc) is None
    assert out_alloc.shape == weight.shape
    assert_rebuilt_matches_reference(out_alloc._data)
