# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped MX quantized tensor: MX-quantized with blocks restarted at group boundaries."""

import torch
import torch_npu
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao_npu import normalize_dim
from torchao_npu.quantization import _FP4_DTYPES, _SUPPORTED_MX_ELEM_DTYPES
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors import (
    logical_qdata_shape,
    permutation_from_transpose,
    resolve_pack_axis,
)
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor

aten = torch.ops.aten


class GroupedMXTensor(BaseQuantizedTensor):
    """A group-awarely MX-quantized tensor.

    Shape operation (transpose / t / permute) support is limited to swapping its two
    dims. Grouped matmul is the only supported matmul. Every other op raises
    ``NotImplementedError``, except the handlers inherited from
    :class:`~torchao_npu.quantized_tensors.base_quantized_tensor.BaseQuantizedTensor`.

    Attributes:
        qdata: Quantized values of dtype ``quant_config.elem_dtype`` along ``dim=-2``.
            FP8 (``float8_e4m3fn`` / ``float8_e5m2``) stores one value per byte with
            ``qdata.shape == logical shape``. FP4 (``float4_e2m1fn_x2``) packs two
            values per element along ``pack_axis``, halving that axis in
            ``qdata.shape``.

        scale: uint8 E8M0 group-aware scales along ``dim=-2``, for a logical
            ``[M, K]``: ``[M//64 + E, K, 2]`` with a trailing pack-2 dim and ``E`` the
            number of groups -- the extra block per group is what keeps a block from
            spanning two groups.

        dtype: The logical (high-precision) dtype this tensor presents; the dtype the values had before quantization.

        quant_axis: Non-negative dim the blocks run along; ``dim=-2`` for the tensors
            :meth:`from_hp` returns.

        quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            this tensor was quantized with.

        pack_axis: Non-negative dim ``qdata`` packs its two values per element
            along, or ``None`` for FP8.
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = [
        "dtype",
        "quant_axis",
        "quant_config",
    ]
    optional_tensor_attribute_names = ["pack_axis"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        quant_axis: int,
        quant_config: MXQuantizeConfig,
        pack_axis: int | None = None,
        requires_grad: bool = False,
    ):
        # Leave strides/storage_offset/layout at their defaults derived by torch:
        # the metadata describes the logical (high-precision) tensor as a dense one.
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            logical_qdata_shape(qdata, pack_axis),
            dtype=dtype,
            device=qdata.device,
            requires_grad=requires_grad,
        )
        return self

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        quant_axis: int,
        quant_config: MXQuantizeConfig,
        pack_axis: int | None = None,
        requires_grad: bool = False,
    ):
        elem_dtype = quant_config.elem_dtype
        if elem_dtype not in _SUPPORTED_MX_ELEM_DTYPES:
            raise ValueError(f"{type(self).__name__} supports FP8 or FP4 qdata, got elem_dtype={elem_dtype}.")

        assert qdata.dtype is elem_dtype, (
            f"``qdata`` dtype ({qdata.dtype}) is not consistent with ``quant_config.elem_dtype`` ({elem_dtype})."
        )

        if qdata.ndim != 2:
            raise ValueError(f"{type(self).__name__} requires qdata.ndim == 2, got {qdata.ndim}.")

        if scale.ndim != qdata.ndim + 1:
            raise ValueError(
                f"scale.ndim must be qdata.ndim + 1 = {qdata.ndim + 1}, got {scale.ndim}. "
                "The MX scale carries a per-axis block dim and a trailing pack-2 dim."
            )

        if qdata.dtype not in _FP4_DTYPES and pack_axis is not None:
            raise ValueError(
                f"Values in a non-FP4 tensor are not packed. ``pack_axis`` should not be set, "
                f"``pack_axis``={pack_axis} passed."
            )

        self.qdata = qdata
        self.scale = scale
        self.quant_config = quant_config
        self.quant_axis = normalize_dim(quant_axis, qdata.ndim)
        self.pack_axis = normalize_dim(pack_axis, qdata.ndim) if pack_axis is not None else None

    @classmethod
    def from_hp(
        cls,
        tensor: torch.Tensor,
        offs: torch.Tensor,
        quant_config: MXQuantizeConfig,
    ) -> "GroupedMXTensor":
        """Group-aware MX quantize a high-precision ``tensor`` (along its ``dim=-2``).

        Args:
            tensor: High-precision 2D tensor to quantize group-awarely.
            offs: Cumulative row offsets of the groups, shape ``(E,)`` (no leading zero).
            quant_config: MX quantization parameters.

        Returns:
            A :class:`GroupedMXTensor` holding the quantized ``qdata`` and the
            group-aware scales. The returned tensor inherits ``requires_grad`` from
            ``tensor``.
        """

        qdata, scale = torch_npu.npu_grouped_dynamic_mx_quant(
            tensor,
            offs.to(torch.int32),
            round_mode=quant_config.round_mode,
            dst_type=quant_config.npu_elem_dtype,
            blocksize=quant_config.block_size,
            scale_alg=quant_config.scale_alg,
        )
        qdata = qdata.view(quant_config.elem_dtype)
        pack_axis = resolve_pack_axis(qdata) if qdata.dtype is torch.float4_e2m1fn_x2 else None
        return cls(qdata, scale, tensor.dtype, -2, quant_config, pack_axis, requires_grad=tensor.requires_grad)


implements = GroupedMXTensor.implements


def _permute(x: GroupedMXTensor, perm: list[int]) -> GroupedMXTensor:
    """Permute ``qdata`` and ``scale`` by ``perm``.

    The scale's leading dims mirror the logical ones -- the blocked dim's first, with
    its extent shrunk -- so ``qdata`` and ``scale`` are permuted together, the scale's
    trailing pack-2 dim staying last, and ``quant_axis``/``pack_axis`` following their
    dims.

    Args:
        x: The source :class:`GroupedMXTensor`.
        perm: Permutation of ``x``'s two dims, as non-negative indices.

    Returns:
        A new :class:`GroupedMXTensor` with permuted components.
    """
    # The scale's trailing pack-2 dim rides along unchanged; its leading dims follow
    # the same permutation the qdata's dims do.
    return GroupedMXTensor(
        x.qdata.permute(perm),
        x.scale.permute([*perm, 2]),
        x.dtype,
        perm[x.quant_axis],
        x.quant_config,
        perm[x.pack_axis] if x.pack_axis is not None else None,
    )


@implements(aten.permute.default)
def _(func, types, args, kwargs):
    x, dims = args[0], args[1]
    perm = [normalize_dim(d, x.ndim) for d in dims]
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    x, dim0, dim1 = args[0], args[1], args[2]
    perm = permutation_from_transpose(dim0, dim1, x.ndim)
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


@implements(aten.t.default)
def _(func, types, args, kwargs):
    x = args[0]
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, [1, 0]))


@implements(aten._grouped_mm.default)
def _(func, types, args, kwargs):
    """Grouped MX matmul contracting over ``dim=-2``: the grouped path's wgrad.

    ``dB = A^T @ dY``: both operands are :class:`GroupedMXTensor`, the left one passed
    already transposed (its blocks are then along its ``dim=-1``), the right one as it
    is (blocks along its ``dim=-2``), both quantized with the same ``block_size``.
    ``offs``, when given, has to be the group offsets both operands were quantized with.
    """
    A, B = args[0], args[1]
    offs = args[2] if len(args) > 2 else kwargs.get("offs")
    bias = args[3] if len(args) > 3 else kwargs.get("bias")
    out_dtype = args[4] if len(args) > 4 else kwargs.get("out_dtype")

    if not isinstance(A, GroupedMXTensor):
        raise ValueError(f"The left grouped_mm operand must be a {GroupedMXTensor.__name__}, got {type(A).__name__}.")

    if not isinstance(B, GroupedMXTensor):
        raise ValueError(f"The right grouped_mm operand must be a {GroupedMXTensor.__name__}, got {type(B).__name__}.")

    if A.quant_axis != A.ndim - 1:
        raise ValueError(f"The left matmul operand must be quantized along dim=-1, got quant_axis={A.quant_axis}.")

    if B.quant_axis != B.ndim - 2:
        raise ValueError(f"The right matmul operand must be quantized along dim=-2, got quant_axis={B.quant_axis}.")

    if A.shape[-1] != B.shape[-2]:
        raise ValueError(f"Contracting dim mismatch: {A.shape[-1]} != {B.shape[-2]}.")

    if A.quant_config.block_size != B.quant_config.block_size:
        raise ValueError(
            "The operands must agree on ``block_size``, got "
            f"{A.quant_config.block_size} for ``A`` and {B.quant_config.block_size} for ``B``."
        )

    return torch_npu.npu_grouped_matmul(
        [A.qdata],
        [B.qdata],
        scale=[B.scale],
        per_token_scale=[A.scale],
        group_list=offs.to(torch.int64) if offs is not None else None,
        bias=[bias] if bias is not None else None,
        group_type=2,
        output_dtype=out_dtype if out_dtype is not None else A.dtype,
        group_list_type=0,
        scale_dtype=B.quant_config.npu_scale_dtype,
        per_token_scale_dtype=A.quant_config.npu_scale_dtype,
        x_dtype=A.quant_config.npu_matmul_dtype,
        weight_dtype=B.quant_config.npu_matmul_dtype,
        split_item=3,
    )[0]


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([GroupedMXTensor])
