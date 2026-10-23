# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block MX quantized tensor: ``qdata`` with two block scales, per ``block_mx_quantize``."""

from dataclasses import fields

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao_npu import normalize_dim
from torchao_npu.quantization import _FP4_DTYPES, _SUPPORTED_MX_ELEM_DTYPES
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import block_mx_quantize
from torchao_npu.quantized_tensors import (
    logical_qdata_shape,
    permutation_from_transpose,
    resolve_pack_axis,
    resolve_view_shape,
)
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor
from torchao_npu.quantized_tensors.mx_tensor import MXTensor

aten = torch.ops.aten


class BlockMXTensor(BaseQuantizedTensor):
    """A weight block-MX-quantized in 32x32 blocks, matching ``block_mx_quantize``.

    Stores the quantized ``qdata`` and two E8M0 block scales covering the same
    32x32 blocks from two axes.

    Shape support is limited to the axis-preserving ops -- transpose / t /
    permute / movedim -- and, because both scales cover the trailing two dims,
    they must keep those two dims in the trailing two slots; only a swap between
    them is allowed. A swap transposes ``qdata`` and exchanges the two scales
    (each with its two trailing dims swapped). ``view`` / ``reshape`` re-factor the
    leading dims instead, with limitations (see :func:`_view`). Every other op raises
    ``NotImplementedError``, except the handlers inherited from
    :class:`~torchao_npu.quantized_tensors.base_quantized_tensor.BaseQuantizedTensor`.

    Attributes:
        qdata: Quantized values of dtype ``quant_config.elem_dtype``. FP8
            (``float8_e4m3fn`` / ``float8_e5m2``) stores one value per byte with
            ``qdata.shape == logical shape``. FP4 (``float4_e2m1fn_x2``) packs
            two values per element along ``pack_axis``, halving that axis in
            ``qdata.shape``. The scales are laid out identically for both dtypes.

        scale1: uint8 E8M0 scales along the last dim, for a logical
            ``[..., R, C]``: ``[..., R, ceil(ceil(C/32)/2), 2]`` with a
            trailing pack-2 dim.

        scale2: uint8 E8M0 scales along the second-to-last dim:
            ``[..., ceil(ceil(R/32)/2), C, 2]``.

        dtype: The logical (high-precision) dtype this tensor presents; the dtype the values had before quantization.

        quant_config: The :class:`~torchao_npu.quantization.quant_configs.BlockMXQuantizeConfig`
            this tensor was quantized with.

        act_quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            activations should be quantized with when multiplied against this
            tensor, or ``None`` if unknown/absent.

        pack_axis: Non-negative dim ``qdata`` packs its two values per element
            along, or ``None`` for FP8. Shape ops keep it in sync with the data.
    """

    tensor_data_names = ["qdata", "scale1", "scale2"]
    tensor_attribute_names = [
        "dtype",
        "quant_config",
        "act_quant_config",
    ]
    optional_tensor_attribute_names = ["pack_axis"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale1: torch.Tensor,
        scale2: torch.Tensor,
        dtype: torch.dtype,
        quant_config: BlockMXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
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
        scale1: torch.Tensor,
        scale2: torch.Tensor,
        dtype: torch.dtype,
        quant_config: BlockMXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
        pack_axis: int | None = None,
        requires_grad: bool = False,
    ):
        if qdata.dtype not in _SUPPORTED_MX_ELEM_DTYPES:
            raise ValueError(f"{type(self).__name__} supports FP8 or FP4 qdata, got {qdata.dtype}.")

        assert qdata.dtype is quant_config.elem_dtype, (
            f"``qdata`` dtype ({qdata.dtype}) is not consistent with ``quant_config.elem_dtype`` "
            f"({quant_config.elem_dtype})."
        )

        if qdata.ndim < 2:
            raise ValueError(f"{type(self).__name__} requires qdata.ndim >= 2, got {qdata.ndim}.")

        if scale1.ndim != qdata.ndim + 1 or scale2.ndim != qdata.ndim + 1:
            raise ValueError(
                f"scale1/scale2 ndim must be qdata.ndim + 1 = {qdata.ndim + 1}, got {scale1.ndim} and {scale2.ndim}."
            )

        if qdata.dtype not in _FP4_DTYPES and pack_axis is not None:
            raise ValueError(
                f"Values in a non-FP4 tensor are not packed. ``pack_axis`` should not be set, "
                f"``pack_axis``={pack_axis} passed."
            )

        self.qdata = qdata
        self.scale1 = scale1
        self.scale2 = scale2
        self.act_quant_config = act_quant_config
        self.quant_config = quant_config
        self.pack_axis = normalize_dim(pack_axis, qdata.ndim) if pack_axis is not None else None

    @classmethod
    def from_hp(
        cls,
        tensor: torch.Tensor,
        quant_config: BlockMXQuantizeConfig,
        axis: int | None = None,
        act_quant_config: MXQuantizeConfig | None = None,
    ) -> "BlockMXTensor":
        """Block-MX quantize a high-precision ``tensor`` (along its two trailing dims).

        Args:
            tensor: High-precision tensor to quantize; its two trailing dims are
                quantized in 32x32 blocks.
            quant_config: Block MX quantization parameters. ``quant_config.elem_dtype``
                determines the resulting ``qdata`` dtype.
            axis: The MXFP4 constraint axis, consumed by the mxfp4-QAT path only
                and required there (``quant_config.mxfp4_fake_quantize_config``
                set). The direct path always quantizes both trailing dims and
                ignores ``axis``.
            act_quant_config: The config activations should be quantized with when
                multiplied against the returned tensor, or ``None`` if
                unknown/absent.

        Returns:
            A :class:`~torchao_npu.quantized_tensors.block_mx_tensor.BlockMXTensor`
            holding the quantized ``qdata`` (FP8, or packed FP4 when
            ``quant_config.elem_dtype`` is FP4) and the two block scales. The
            returned tensor inherits ``requires_grad`` from ``tensor``.

        Raises:
            AssertionError: If the mxfp4-QAT path is selected and ``axis`` is
                ``None`` or not one of the two trailing dims.
        """

        qdata, scale1, scale2 = block_mx_quantize(tensor, quant_config, axis)
        pack_axis = resolve_pack_axis(qdata) if qdata.dtype is torch.float4_e2m1fn_x2 else None
        return cls(
            qdata,
            scale1,
            scale2,
            tensor.dtype,
            quant_config,
            act_quant_config,
            pack_axis,
            requires_grad=tensor.requires_grad,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize back to ``dtype``, using whichever block scale's axis is innermost-contiguous.

        Both scales cover the trailing two dims over one shared ``qdata``: ``scale1`` the last dim,
        ``scale2`` the second-to-last. Only a quant axis that is innermost-contiguous
        (``stride == 1``) can back the dequantization op, so ``dim=-1`` is used when
        ``qdata.stride(-1) == 1``, otherwise ``dim=-2`` when ``qdata.stride(-2) == 1``.

        The chosen axis and scale are converted with :meth:`to_mx_tensor` and dequantized as an
        :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`.

        Args:
            output_dtype: The dtype of the returned tensor, ``dtype`` when ``None``;

        Returns:
            A plain tensor of dtype ``output_dtype``

        Raises:
            RuntimeError: If neither trailing dim is innermost-contiguous.
        """
        if self.qdata.stride(-1) == 1:
            quant_axis = self.qdata.ndim - 1
        elif self.qdata.stride(-2) == 1:
            quant_axis = self.qdata.ndim - 2
        else:
            raise RuntimeError(
                f"Neither trailing dim of this {type(self).__name__} is innermost-contiguous: "
                f"stride(-1)={self.qdata.stride(-1)} and stride(-2)={self.qdata.stride(-2)} for ``qdata``; "
                "one of them must be 1 for the dequantization op to consume the data."
            )

        return self.to_mx_tensor(quant_axis).dequantize(output_dtype)

    def to_mx_tensor(self, quant_axis: int) -> MXTensor:
        """Convert to the single-axis :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`
        quantized along ``quant_axis``.

        ``qdata``, ``scale1``/``scale2`` are used directly. No additional re-quantization or copy
        are used. A :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig` is constructed
        on the fly based on this tensor's
        :class:`~torchao_npu.quantization.quant_configs.BlockMXQuantizeConfig`. The returned
        :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` inherits the
        ``act_quant_config``.

        Args:
            quant_axis: The dim the returned
                :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` is quantized along;
                must be one of the two trailing dims.

        Returns:
            An :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor` sharing this tensor's
            ``qdata`` and the matching scale, carrying an
            :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            equivalent to ``quant_config``.

        Raises:
            ValueError: If ``quant_axis`` is not one of the two trailing dims.
        """
        quant_axis = normalize_dim(quant_axis, self.qdata.ndim)
        if quant_axis not in (self.qdata.ndim - 2, self.qdata.ndim - 1):
            raise ValueError(
                f"Only the last two dims are supported for ``quant_axis``, quant_axis={quant_axis} passed."
            )

        scale = self.scale1 if quant_axis == self.qdata.ndim - 1 else self.scale2
        # A fresh MX config: the block config's extra mxfp4_fake_quantize_config field does not describe an MX tensor.
        quant_config = MXQuantizeConfig(
            **{f.name: getattr(self.quant_config, f.name) for f in fields(MXQuantizeConfig)}
        )
        return MXTensor(
            self.qdata,
            scale,
            self.dtype,  # pyrefly: ignore [missing-attribute]
            quant_axis,
            quant_config,
            self.act_quant_config,
            self.pack_axis,
        )


implements = BlockMXTensor.implements


def _permute(x: BlockMXTensor, perm: list[int]) -> BlockMXTensor:
    """Permute ``qdata`` and both scales; the last two dims must stay in the trailing two slots.

    ``pack_axis`` is tracked onto its permuted position. Leading dims permute freely;
    a swap of the trailing two exchanges ``scale1``/``scale2`` roles (each with its
    two spatial dims swapped).

    Args:
        x: The source :class:`BlockMXTensor`.
        perm: Full permutation of ``x``'s dims, as non-negative indices.

    Returns:
        A new :class:`BlockMXTensor` with permuted components.

    Raises:
        ValueError: If the permutation moves a dim across the leading /
            trailing-two boundary.
    """
    if {perm[-2], perm[-1]} != {x.ndim - 2, x.ndim - 1}:
        raise ValueError(
            f"{type(x).__name__} only supports permutations that keep the last two dims "
            f"trailing (optionally swapped); got {perm}."
        )

    swap = perm[-2] == x.ndim - 1
    new_qdata = x.qdata.permute(perm)
    new_pack_axis = perm.index(x.pack_axis) if x.pack_axis is not None else None

    # ``scale1``/``scale2`` carry an extra trailing pack-2 dim, which stays last.
    scale_perm = [*perm, x.scale1.ndim - 1]

    if not swap:
        new_s1 = x.scale1.permute(scale_perm)
        new_s2 = x.scale2.permute(scale_perm)
    else:
        # Swap the two trailing dims: scale1/scale2 exchange roles.
        new_s1 = x.scale2.permute(scale_perm)
        new_s2 = x.scale1.permute(scale_perm)

    return BlockMXTensor(new_qdata, new_s1, new_s2, x.dtype, x.quant_config, x.act_quant_config, new_pack_axis)


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
    # Match plain torch semantics: the <=2D check lives in the C++ native impl,
    # which subclass dispatch bypasses, so enforce it here. For 2D it is
    # transpose(0, 1); for <=1D a no-op.
    if x.ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {x.ndim}D")
    perm = permutation_from_transpose(0, 1, x.ndim) if x.ndim == 2 else list(range(x.ndim))
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


def _view(func, x: BlockMXTensor, new_shape: tuple[int, ...]) -> BlockMXTensor:
    """Re-shape ``x``, allowing only the changes that its quantization block scales can follow.

    A re-shape is only supported when

    * the tensor is contiguous (every component, see :meth:`is_contiguous`);
    * for FP4, the pack axis is the last dim -- its packed pairs may not be split;
    * the last dim keeps its size -- its block boundaries may not move;
    * ``dim -2`` is a multiple of ``2 * block_size`` both before and after, since the
      blocks of ``dim -2`` are read in packed pairs.

    The leading dims then re-factor freely and both scales follow with plain views: the
    block count of ``dim -2`` divides evenly, so 2 x block_size consecutive rows cover
    the same blocks before and after. The components are re-viewed with no copy, never
    materialized into a copy of the quantized data.

    Args:
        func: The ``aten`` op being handled, named in the messages.
        x: The tensor to re-shape.
        new_shape: The requested shape, possibly holding one ``-1``.

    Returns:
        A :class:`BlockMXTensor` whose components are views of ``x``'s.

    Raises:
        ValueError: If an unsupported re-shape is requested.
    """
    src_shape = x.shape
    dst_shape = resolve_view_shape(x.numel(), new_shape)
    block_size = x.quant_config.block_size
    packed_block_size = 2 * block_size

    if not x.is_contiguous():
        raise ValueError(f"{type(x).__name__}'s {func.name()} requires contiguous components.")

    if x.qdata.dtype in _FP4_DTYPES and x.pack_axis != normalize_dim(-1, x.ndim):
        raise ValueError(f"pack_axis must be the last dim for an FP4 quantized tensor, got pack_axis={x.pack_axis}")

    if len(dst_shape) < 2:
        raise ValueError(f"{type(x).__name__}'s {func.name()} must yield at least a 2D tensor, got {dst_shape}.")

    if dst_shape[-1] != src_shape[-1]:
        raise ValueError(
            f"{type(x).__name__} only supports re-shapes that keep the last dim ({src_shape[-1]}); "
            f"got {src_shape} -> {dst_shape}."
        )
    if src_shape[-2] % packed_block_size or dst_shape[-2] % packed_block_size:
        raise ValueError(
            f"{type(x).__name__} cannot represent {src_shape} -> {dst_shape}: dim -2 must be a "
            f"multiple of {packed_block_size} both before and after, got {src_shape[-2]} and {dst_shape[-2]}."
        )

    # ``qdata``'s last dim is stored, not logical: FP4 packs two values per element, while
    # FP8 stores one. The scales below are laid out per logical value, so they take the
    # logical ``dst_shape``.
    qdata = x.qdata.view(*dst_shape[:-1], x.qdata.shape[-1])
    # ``scale1`` carries one block row per ``dim -2`` row, so it takes the same leading split.
    scale1 = x.scale1.view(*dst_shape[:-1], *x.scale1.shape[-2:])
    # ``scale2``'s block dim covers ``dim -2``, whose blocks now divide evenly.
    scale2 = x.scale2.view(*dst_shape[:-2], dst_shape[-2] // packed_block_size, dst_shape[-1], x.scale2.shape[-1])
    # FP4's pack axis is the (unit-stride) last dim, which a re-shape never resizes.
    pack_axis = -1 if x.pack_axis is not None else None

    return BlockMXTensor(qdata, scale1, scale2, x.dtype, x.quant_config, x.act_quant_config, pack_axis)


@implements([aten.view.default, aten.reshape.default])
def _(func, types, args, kwargs):
    x, new_shape = args[0], args[1]
    return return_and_correct_aliasing(func, args, kwargs, _view(func, x, tuple(new_shape)))


@implements(aten._unsafe_view.default)
def _(func, types, args, kwargs):
    # Not an aliasing op per its schema; the re-view shares the components either way.
    x, new_shape = args[0], args[1]
    return _view(func, x, tuple(new_shape))


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([BlockMXTensor])
