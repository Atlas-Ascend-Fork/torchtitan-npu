# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.prototype.moe_training.utils import unwrap_weight

from torchao_npu.ops.block_mx_ops import (
    to_block_mx_then_bmm,
    to_block_mx_then_grouped_mm,
    to_block_mx_then_mm,
)
from torchao_npu.quantization import _SUPPORTED_HP_DTYPES
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.transform import register_parameter_swap_handler
from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.base_wrapper_tensor import (
    BaseTrainingWeightWrapperTensor,
    _pad_uneven_shard_dim0,
)


class BlockMXTrainingWeightWrapperTensor(BaseTrainingWeightWrapperTensor):
    """Applies block MX quantized matmul on NPU.

    Performs a **real** block MX matmul using
    ``to_block_mx_then_mm`` / ``to_block_mx_then_grouped_mm``.

    When ``weight_config.mxfp4_fake_quantize_config`` is set, weights are
    pre-quantized to MXFP4 before the block MX matmul.

    When ``weight_config.fsdp_prequantize`` is set, the sharded weight is
    quantized prior to FSDP communication.
    """

    _data: torch.Tensor
    weight_config: BlockMXQuantizeConfig
    activation_config: MXQuantizeConfig

    def __init__(
        self,
        tensor: torch.Tensor,
        weight_config: BlockMXQuantizeConfig | None = None,
        activation_config: MXQuantizeConfig | None = None,
    ):
        if weight_config is None:
            raise ValueError(f"`weight_config` is required for {type(self).__name__}.")

        if activation_config is None:
            raise ValueError(f"`activation_config` is required for {type(self).__name__}.")

        if type(weight_config) is not BlockMXQuantizeConfig:
            raise ValueError(f"Only `BlockMXQuantizeConfig` is supported for `weight_config` in {type(self).__name__}.")

        if type(activation_config) is not MXQuantizeConfig:
            raise ValueError(f"Only `MXQuantizeConfig` is supported for `activation_config` in {type(self).__name__}.")

        super().__init__(tensor, weight_config=weight_config, activation_config=activation_config)

    def to_inference_weight(self) -> MXTensor:
        """Quantize the master weight into its inference-time :class:`MXTensor`.

        Training quantizes the stored weight's last axis (``block_mx_quantize``
        runs on the transposed operand along axis -2, i.e. the stored last axis),
        so quantizing axis -1 here reproduces the quantized weight used in
        training's forward.

        With mxfp4-QAT enabled the weight is served as MXFP4: training's QAT
        pre-pass quantizes the same stored last axis, so an
        :class:`MXTensor` built from the nested FP4 config matches it exactly.
        Otherwise the block-MX FP8 quantization is converted to the equivalent
        single-axis :class:`MXTensor` along the stored last dim (see
        :meth:`~torchao_npu.quantized_tensors.block_mx_tensor.BlockMXTensor.to_mx_tensor`).
        """
        mxfp4_config = self.weight_config.mxfp4_fake_quantize_config
        if mxfp4_config is not None:
            return MXTensor.from_hp(
                self._data,
                quant_config=mxfp4_config,
                axis=-1,
                act_quant_config=self.activation_config,
            )
        else:
            return BlockMXTensor.from_hp(
                self._data,
                quant_config=self.weight_config,
                act_quant_config=self.activation_config,
            ).to_mx_tensor(quant_axis=-1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in (torch.mm, torch.matmul):
            # 2D matmul: A @ B where B is the wrapped weight [K, N]
            return cls._block_mx_mm(args, kwargs)

        elif func is torch._grouped_mm:
            # Grouped matmul: A @ B[E,K,N] with offs=group_list
            return cls._block_mx_grouped_mm(args, kwargs)

        elif func is torch.nn.functional.linear:
            # Linear: A @ weight.T + bias, weight is [N, K]
            return cls._block_mx_linear(args, kwargs)

        elif func is torch.addmm:
            # addmm: bias + A @ B where B is the wrapped weight
            return cls._block_mx_addmm(args, kwargs)

        elif func is torch.bmm:
            # bmm: A @ B where both are 3D and B is the wrapped weight
            return cls._block_mx_bmm(args, kwargs)

        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        """Pre-quantize the local shard before all_gather.

        On a multi-rank mesh with ``fsdp_prequantize`` set, the shard is
        quantized and the quantized results instead of the high precision shard
        are all-gathered and reassembled in :meth:`fsdp_post_all_gather`.
        Otherwise (without this flag, or on a size-1 mesh where no true
        all-gather communication is executed), the high precision shard is
        all-gathered.

        Raises:
            RuntimeError: If ``fsdp_prequantize`` is set on a multi-rank mesh but
                the local shard shape would make the quantization pad zeros, which
                would land in the middle of the gathered tensors.
            ValueError: If ``fsdp_prequantize_dtype`` is unsupported on a
                multi-rank mesh.
        """
        if self.weight_config.fsdp_prequantize is True and mesh.size() > 1:
            shard_shape_error = (
                "Cannot quantize sharded weight before FSDP: local shard shape "
                f"{tuple(self._data.shape)}, fsdp_prequantize_dtype={self.weight_config.fsdp_prequantize_dtype}."
            )

            if self.weight_config.fsdp_prequantize_dtype == "fp8":
                quant_block_size = self.weight_config.block_size * 2
                if self._data.shape[-2] % quant_block_size != 0 or self._data.shape[-1] % quant_block_size != 0:
                    raise RuntimeError(shard_shape_error)

                quantized = BlockMXTensor.from_hp(
                    self._data,
                    quant_config=self.weight_config,
                    axis=-1,
                    act_quant_config=self.activation_config,
                )

            elif self.weight_config.fsdp_prequantize_dtype == "mxfp4":
                assert self.weight_config.mxfp4_fake_quantize_config is not None, "mxfp4_fake_quantize_config is None"

                quant_block_size = self.weight_config.mxfp4_fake_quantize_config.block_size * 2
                if self._data.shape[-1] % quant_block_size != 0:
                    raise RuntimeError(shard_shape_error)

                quantized = MXTensor.from_hp(
                    self._data,
                    quant_config=self.weight_config.mxfp4_fake_quantize_config,
                    axis=-1,
                    act_quant_config=self.activation_config,
                )

            else:
                raise ValueError(
                    f"Unsupported fsdp_prequantize_dtype value '{self.weight_config.fsdp_prequantize_dtype}'"
                )

            quantized = quantized.to(mp_policy.param_dtype)
            tensor_names, tensor_attrs = quantized.__tensor_flatten__()
            all_gather_inputs = tuple(getattr(quantized, name) for name in tensor_names)

            # Uneven dim-0 sharding (e.g. 3 experts over 2 ranks): every
            # component shares the leading expert dim, so pad all of them to the
            # largest chunk and stash the logical dim0 for post to trim back.
            if outer_size[0] % mesh.size() != 0:
                all_gather_inputs = _pad_uneven_shard_dim0(all_gather_inputs, outer_size, mesh.size())
                tensor_attrs = {**tensor_attrs, "_uneven_logical_dim0": outer_size[0]}

            return all_gather_inputs, tensor_attrs

        return super().fsdp_pre_all_gather(mesh, outer_size, outer_stride, module, mp_policy)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        """Rebuild the wrapper tensor from the gathered and potentially quantized components.

        With ``fsdp_prequantize`` set, a high-precision payload -- the size-1-mesh
        fallback of :meth:`fsdp_pre_all_gather` -- is quantized here, so the unsharded
        ``_data`` matches the multi-rank path.
        """
        if len(all_gather_outputs) == 1:
            (data,) = all_gather_outputs
            assert data.dtype in _SUPPORTED_HP_DTYPES, (
                f"When only one tensor is gathered, it is expected to be in high precision "
                f"{_SUPPORTED_HP_DTYPES}, got a {data.dtype} tensor."
            )

            if self.weight_config.fsdp_prequantize is True:
                if self.weight_config.fsdp_prequantize_dtype == "fp8":
                    quantized = BlockMXTensor.from_hp(
                        data,
                        quant_config=self.weight_config,
                        axis=-1,
                        act_quant_config=self.activation_config,
                    )
                elif self.weight_config.fsdp_prequantize_dtype == "mxfp4":
                    mxfp4_config = self.weight_config.mxfp4_fake_quantize_config
                    assert mxfp4_config is not None, "mxfp4_fake_quantize_config is None"
                    quantized = MXTensor.from_hp(
                        data,
                        quant_config=mxfp4_config,
                        axis=-1,
                        act_quant_config=self.activation_config,
                    )
                else:
                    raise ValueError(
                        f"Unsupported fsdp_prequantize_dtype value '{self.weight_config.fsdp_prequantize_dtype}'"
                    )
            else:
                return super().fsdp_post_all_gather(all_gather_outputs, metadata, param_dtype, out=out)

        else:
            # Trim the zero pad tail that ``fsdp_pre_all_gather`` added under uneven
            # dim-0 sharding, before rebuilding.
            logical_dim0 = metadata.get("_uneven_logical_dim0") if isinstance(metadata, dict) else None
            if logical_dim0 is not None:
                gathered: tuple[torch.Tensor, ...] = tuple(t.narrow(0, 0, logical_dim0) for t in all_gather_outputs)
            else:
                gathered = all_gather_outputs

            if len(all_gather_outputs) == 2:
                quantized = MXTensor.__tensor_unflatten__(
                    dict(zip(MXTensor.tensor_data_names, gathered, strict=True)), metadata, None, None
                )

            elif len(all_gather_outputs) == 3:
                # Inverse of ``fsdp_pre_all_gather``'s flatten. ``outer_size`` /
                # ``outer_stride`` are unused by the class's unflatten. So None is passed.
                quantized = BlockMXTensor.__tensor_unflatten__(
                    dict(zip(BlockMXTensor.tensor_data_names, gathered, strict=True)),
                    metadata,
                    None,
                    None,
                )

            else:
                raise RuntimeError(
                    "Expect `len(all_gather_outputs)` in (1, 2, 3), got "
                    f"`len(all_gather_outputs)` == {len(all_gather_outputs)}"
                )

        if out is None:
            # Training step 0: build a new wrapper. The components are also the
            # ``inner_tensors``: FSDP frees and re-allocates their storage and
            # reuses them as the gather buffers, so ``_data`` must keep the very
            # objects handed back here.
            output = type(self)(
                quantized,
                activation_config=self.activation_config,
                weight_config=self.weight_config,
            )
            return output, all_gather_outputs

        # Training step 1+: the pre-allocated unsharded param, bare or wrapped
        # in DTensor. Replace its ``_data`` with the freshly gathered
        # components; the configs are restored from the sharded wrapper (FSDP2
        # hands over a shallow copy of it).
        if isinstance(out, BlockMXTrainingWeightWrapperTensor):
            out._data = quantized
            out.activation_config = self.activation_config
            out.weight_config = self.weight_config
        elif isinstance(out, DTensor) and isinstance(out._local_tensor, BlockMXTrainingWeightWrapperTensor):
            # Bind to a local: pyrefly doesn't propagate isinstance narrowing
            # through attribute accesses reliably.
            local: BlockMXTrainingWeightWrapperTensor = out._local_tensor
            local._data = quantized
            local.activation_config = self.activation_config
            local.weight_config = self.weight_config
        else:
            raise RuntimeError(
                f"expected out to be {type(self).__name__} or DTensor with "
                f"local_tensor={type(self).__name__}, but got {type(out)}"
            )
        return None

    # ------------------------------------------------------------------
    # Per-op helpers
    # ------------------------------------------------------------------

    @classmethod
    def _block_mx_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_grouped_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        group_list = args[2] if len(args) > 2 else kwargs.get("offs")
        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_grouped_mm(A, B_data, group_list, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_linear(cls, args, kwargs):
        # F.linear(A, weight, bias) — weight is [N, K], A @ weight.T + bias
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)
        bias = args[2] if len(args) > 2 else kwargs.get("bias")

        with torch._C.DisableTorchFunctionSubclass():
            result = to_block_mx_then_mm(A, B_data.T, B.activation_config, B.weight_config)  # type: ignore
            if bias is not None:
                result = result + bias
            return result

    @classmethod
    def _block_mx_addmm(cls, args, kwargs):
        # addmm(bias, A, B) — bias + A @ B
        bias, A, B = args[0], args[1], args[2]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            result = to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore
            result = result + bias
            return result

    @classmethod
    def _block_mx_bmm(cls, args, kwargs):
        # bmm(A, B) — both 3D batched matmul, B is the wrapped weight
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_bmm(A, B_data, B.activation_config, B.weight_config)  # type: ignore


@register_parameter_swap_handler(BlockMXQuantizeConfig)
def _(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: tuple[Any, ...] = (),
):
    from torchao_npu.configs import ParamSwapConfig

    config: ParamSwapConfig = extra_args[0]

    if not isinstance(config, ParamSwapConfig):
        raise ValueError(f"extra_args[0] must be a ParamSwapConfig, got {type(config).__name__}.")

    if config.activation_config is not None and type(config.activation_config) is not MXQuantizeConfig:
        raise ValueError(
            f"activation_config must be {MXQuantizeConfig.__name__}, got {type(config.activation_config).__name__}."
        )

    if config.weight_config is not None and type(config.weight_config) is not BlockMXQuantizeConfig:
        raise ValueError(
            f"weight_config must be {BlockMXQuantizeConfig.__name__}, got {type(config.weight_config).__name__}."
        )

    return nn.Parameter(
        data=BlockMXTrainingWeightWrapperTensor(
            param.data.detach(),
            activation_config=config.activation_config,
            weight_config=config.weight_config,
        ),
        requires_grad=param.requires_grad,
    )


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([BlockMXTrainingWeightWrapperTensor])
