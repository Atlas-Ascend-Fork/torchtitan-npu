# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MX quantized tensor: a single-axis MX-quantized tensor"""

from typing import TYPE_CHECKING

import torch
import torch_npu
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao_npu import normalize_dim
from torchao_npu.quantization import _FP4_DTYPES, _SUPPORTED_MX_ELEM_DTYPES
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.block_mx import mxfp4_to_block_mx
from torchao_npu.quantization.quant_primitives.mx import mx_dequantize, mx_quantize
from torchao_npu.quantized_tensors import (
    logical_qdata_shape,
    permutation_from_transpose,
    resolve_pack_axis,
    resolve_view_shape,
)
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor

if TYPE_CHECKING:
    from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor

aten = torch.ops.aten


class MXTensor(BaseQuantizedTensor):
    """A single-axis MX-quantized tensor.

    Stores the quantized ``qdata`` and its E8M0 ``scale``, together with the
    configs it was built from and the quantization axis.

    Shape support covers the axis-preserving ops -- transpose / t / permute /
    movedim -- which ``__torch_dispatch__`` translates onto ``qdata`` and
    ``scale``, keeping ``quant_axis`` and ``pack_axis`` in sync (those ops may
    not merge or split dims). ``view`` / ``_unsafe_view`` instead re-factor the
    leading dims, only when the quant axis is the last dim and
    innermost-contiguous (and, for FP4, the pack axis is that same last dim);
    the last dim may change only between multiples of ``2 * block_size`` (whole
    packed block pairs). ``mm`` / ``bmm`` / ``addmm`` / ``_grouped_mm`` /
    ``linear`` / ``matmul`` are intercepted too (see below). Every other op
    raises ``NotImplementedError``, except the handlers inherited from
    :class:`~torchao_npu.quantized_tensors.base_quantized_tensor.BaseQuantizedTensor`.

    Matmul: the right operand is the weight and must be an ``MXTensor``
    quantized along ``dim=-2`` (the contracting dim); the left operand is either
    an ``MXTensor`` quantized along ``dim=-1`` (the contracting dim), which must
    match the right operand's ``act_quant_config`` when it is set, or a
    high-precision tensor, which is quantized on the fly along ``dim=-1`` with
    that config -- a ``RuntimeError`` is raised when it is unset. The product is
    computed by the NPU kernels from the stored ``qdata``/``scale``, without
    re-quantizing, and returned as a plain high-precision tensor. There is no
    backward.

    ``torch.matmul`` is only partially supported: it may apply various shape
    operations to an ``MXTensor`` operand, and raises an error for any of them
    that is not supported.

    ``qdata`` is FP8 (``float8_e4m3fn`` / ``float8_e5m2``) or FP4
    (``float4_e2m1fn_x2``). FP4 packs two values per element along ``pack_axis``
    (which may differ from ``quant_axis``, e.g. a dense input quantized along a
    non-last axis packs on the last dim), so ``qdata.shape`` differs from the
    logical shape.

    Attributes:
        qdata: Quantized values. For FP8, a dense ``qdata.shape == logical
            shape`` tensor of dtype ``quant_config.elem_dtype``. For FP4, ``float4_e2m1fn_x2``
            with two values packed per element along ``pack_axis``, so that
            axis is halved.

        scale: ``float8_e8m0fnu`` block scales with ndim = ``qdata.ndim + 1``, readers are
            referred to the doc of ``torch_npu.npu_dynamic_mx_quant`` for its shape details.

        dtype: The logical (high-precision) dtype this tensor presents; the dtype the values had before quantization.

        quant_axis: Non-negative dim along which the tensor is quantized.

        pack_axis: Non-negative dim ``qdata`` packs its two values per element
            along, or ``None`` for FP8. Shape ops keep it in sync with the data,
            so it need not equal ``quant_axis``.

        quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            this tensor was quantized with.

        act_quant_config: The :class:`~torchao_npu.quantization.quant_configs.MXQuantizeConfig`
            activations should be quantized with when multiplied against this
            tensor, or ``None`` if unknown/absent.
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = [
        "dtype",
        "quant_axis",
        "quant_config",
        "act_quant_config",
    ]
    optional_tensor_attribute_names = ["pack_axis"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
        quant_axis: int,
        quant_config: MXQuantizeConfig,
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
        scale: torch.Tensor,
        dtype: torch.dtype,
        quant_axis: int,
        quant_config: MXQuantizeConfig,
        act_quant_config: MXQuantizeConfig | None = None,
        pack_axis: int | None = None,
        requires_grad: bool = False,
    ):
        elem_dtype = quant_config.elem_dtype
        if elem_dtype not in _SUPPORTED_MX_ELEM_DTYPES:
            raise ValueError(f"{type(self).__name__} supports FP8 or FP4 qdata, got elem_dtype={elem_dtype}.")

        assert qdata.dtype is elem_dtype, (
            f"``qdata`` dtype ({qdata.dtype}) is not consistent with ``quant_config.elem_dtype`` ({elem_dtype})."
        )

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
        self.act_quant_config = act_quant_config
        self.quant_config = quant_config
        self.quant_axis = normalize_dim(quant_axis, qdata.ndim)
        self.pack_axis = normalize_dim(pack_axis, qdata.ndim) if pack_axis is not None else None

    @classmethod
    def from_hp(
        cls,
        tensor: torch.Tensor,
        quant_config: MXQuantizeConfig,
        axis: int,
        act_quant_config: MXQuantizeConfig | None = None,
    ) -> "MXTensor":
        """Quantize a high-precision ``tensor`` along ``axis`` into an ``MXTensor``.

        The returned tensor inherits ``requires_grad`` from ``tensor``.
        """
        axis = normalize_dim(axis, tensor.ndim)
        qdata, scale = mx_quantize(tensor, axis, quant_config)
        pack_axis = resolve_pack_axis(qdata) if qdata.dtype is torch.float4_e2m1fn_x2 else None
        return cls(
            qdata,
            scale,
            tensor.dtype,
            axis,
            quant_config,
            act_quant_config,
            pack_axis,
            requires_grad=tensor.requires_grad,
        )

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize back to ``dtype``, or to ``output_dtype`` when given.

        Args:
            output_dtype: The dtype of the returned tensor, ``dtype`` when ``None``. Must be
                a dtype the dequantization op supports: ``torch.float32``, ``torch.bfloat16`` or
                ``torch.float16``.

        Returns:
            A plain tensor of dtype ``output_dtype``
        """
        return mx_dequantize(
            self.qdata,
            self.scale,
            self.quant_axis,
            block_size=self.quant_config.block_size,
            src_dtype=self.quant_config.elem_dtype,
            # pyrefly: ignore [missing-attribute]
            output_dtype=self.dtype if output_dtype is None else output_dtype,
        )

    def to_block_mx(self, quant_config: BlockMXQuantizeConfig) -> "BlockMXTensor":
        """Convert this MXFP4 tensor into its Block MX form.

        Args:
            quant_config: The Block MX configuration of the returned tensor.

        Returns:
            A :class:`BlockMXTensor` with this tensor's logical shape,
            ``dtype`` and ``act_quant_config``.
        """
        assert self.quant_axis == self.pack_axis, (
            "The conversion is supported only when self.quant_axis == self.pack_axis, "
            f"got self.quant_axis={self.quant_axis}, self.pack_axis={self.pack_axis}"
        )
        if quant_config.mxfp4_fake_quantize_config is not None:
            assert self.quant_config == quant_config.mxfp4_fake_quantize_config, (
                f"self.quant_config ({self.quant_config}) is not consistent with the target quant_config "
                f"{quant_config.mxfp4_fake_quantize_config}"
            )

        q, s1, s2 = mxfp4_to_block_mx(
            self.qdata, self.scale, self.quant_config, quant_config, quant_axis=self.quant_axis
        )

        # Imported here because ``block_mx_tensor`` imports this module.
        from torchao_npu.quantized_tensors.block_mx_tensor import BlockMXTensor

        return BlockMXTensor(
            q,
            s1,
            s2,
            self.dtype,  # pyrefly: ignore [missing-attribute]
            quant_config,
            self.act_quant_config,
            None,
            requires_grad=self.requires_grad,  # pyrefly: ignore [missing-attribute]
        )


implements = MXTensor.implements


def _permute(x: MXTensor, perm: list[int]) -> MXTensor:
    """Apply a full permutation to ``qdata`` and ``scale`` and track ``quant_axis``/``pack_axis``.

    Args:
        x: The source :class:`MXTensor`.
        perm: Full permutation of ``x``'s dims, as non-negative indices.

    Returns:
        A new :class:`MXTensor` with permuted ``qdata``/``scale`` and the
        ``quant_axis``/``pack_axis`` updated to their permuted positions.
    """
    new_axis = perm.index(x.quant_axis)
    new_pack_axis = perm.index(x.pack_axis) if x.pack_axis is not None else None
    # The scale's trailing pack-2 dim (at index ndim) rides along unchanged.
    scale_perm = [*perm, x.scale.ndim - 1]
    return MXTensor(
        x.qdata.permute(perm),
        x.scale.permute(scale_perm),
        x.dtype,
        new_axis,
        x.quant_config,
        x.act_quant_config,
        new_pack_axis,
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
    # Match plain torch semantics: the <=2D check lives in the C++ native impl,
    # which subclass dispatch bypasses, so enforce it here. For 2D it is
    # transpose(0, 1); for <=1D a no-op.
    if x.ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {x.ndim}D")
    perm = permutation_from_transpose(0, 1, x.ndim) if x.ndim == 2 else list(range(x.ndim))
    return return_and_correct_aliasing(func, args, kwargs, _permute(x, perm))


def _view(func, x: MXTensor, new_shape: tuple[int, ...]) -> MXTensor:
    """Re-shape ``x``, allowing only the changes its quantization axis can follow.

    A re-shape is only supported when

    * the quant axis is the last dim -- its block boundaries may not move;
    * that axis is innermost-contiguous in ``qdata`` (``stride == 1``), the layout
      the MX kernels consume;
    * for FP4, the pack axis is that same last dim -- its packed pairs may not be split;
    * the last dim either keeps its size, or changes between sizes that are both
      multiples of ``2 * block_size`` -- every row then holds whole packed block pairs,
      so merging or splitting rows preserves the flat pair sequence.

    ``scale`` follows the same split: with the last dim unchanged it keeps its trailing
    ``(blocks, 2)`` dims; when the last dim changes, both sides are pair-aligned, so the
    flat pair sequence is preserved and it takes ``(dst[-1] // (2 * block_size), 2)``.
    Both ``view`` and ``_unsafe_view`` re-view the components strictly; neither copies.

    Args:
        func: The ``aten`` op being handled, named in the messages.
        x: The tensor to re-shape.
        new_shape: The requested shape, possibly holding one ``-1``.

    Returns:
        An :class:`MXTensor` whose components are views of ``x``'s.

    Raises:
        ValueError: If an unsupported re-shape is requested.
    """
    src_shape = x.shape
    dst_shape = resolve_view_shape(x.numel(), new_shape)  # logical numel; FP4's qdata is halved

    if x.quant_axis != x.ndim - 1:
        raise ValueError(
            f"{type(x).__name__}'s {func.name()} requires the quant axis to be the last dim, "
            f"got quant_axis={x.quant_axis} for a {x.ndim}D tensor."
        )
    if x.qdata.stride(x.quant_axis) != 1:
        raise ValueError(
            f"{type(x).__name__}'s {func.name()} requires an innermost-contiguous quant axis, "
            f"got stride({x.quant_axis})={x.qdata.stride(x.quant_axis)} for ``qdata``."
        )
    if x.qdata.dtype in _FP4_DTYPES and x.pack_axis != normalize_dim(-1, x.ndim):
        raise ValueError(
            f"{type(x).__name__}'s {func.name()} requires pack_axis to be the last dim for FP4, "
            f"got pack_axis={x.pack_axis}."
        )
    if not dst_shape:
        raise ValueError(f"{type(x).__name__}'s {func.name()} must yield at least a 1D tensor, got {dst_shape}.")
    packed_block_size = 2 * x.quant_config.block_size
    if dst_shape[-1] != src_shape[-1] and (src_shape[-1] % packed_block_size or dst_shape[-1] % packed_block_size):
        raise ValueError(
            f"{type(x).__name__}'s {func.name()} can only change the last dim between multiples of "
            f"{packed_block_size} (whole packed block pairs); got {src_shape[-1]} -> {dst_shape[-1]}."
        )

    # ``qdata``'s last dim is stored, not logical: FP4 packs two values per element
    # along the quant axis, so the target's last dim is halved.
    # ``scale`` keeps its trailing ``(blocks, 2)`` dims when the last dim is unchanged;
    # otherwise both sides are pair-aligned and its blocks-per-row count is recomputed.
    # Forwarding ``func`` re-views each component with the same op.
    qdata_last = dst_shape[-1] // 2 if x.qdata.dtype in _FP4_DTYPES else dst_shape[-1]
    scale_blocks = x.scale.shape[-2] if dst_shape[-1] == src_shape[-1] else dst_shape[-1] // packed_block_size
    qdata = func(x.qdata, (*dst_shape[:-1], qdata_last))
    scale = func(x.scale, (*dst_shape[:-1], scale_blocks, x.scale.shape[-1]))
    # The quant axis stays last; FP4's pack axis is that same last dim.
    pack_axis = -1 if x.pack_axis is not None else None
    return MXTensor(qdata, scale, x.dtype, -1, x.quant_config, x.act_quant_config, pack_axis)


@implements(aten.view.default)
def _(func, types, args, kwargs):
    x, new_shape = args[0], args[1]
    return return_and_correct_aliasing(func, args, kwargs, _view(func, x, tuple(new_shape)))


@implements(aten._unsafe_view.default)
def _(func, types, args, kwargs):
    # Not an aliasing op per its schema; the re-view shares the components either way.
    x, new_shape = args[0], args[1]
    return _view(func, x, tuple(new_shape))


def _matmul_operands(A: torch.Tensor, B: torch.Tensor) -> tuple[MXTensor, MXTensor]:
    """Validate a matmul operand pair and return both operands as :class:`MXTensor`.

    The right operand is the weight: it must already be an :class:`MXTensor`
    quantized along ``dim=-2`` (the contracting dim). The left operand is either
    an :class:`MXTensor` quantized along ``dim=-1`` (the contracting dim) and
    matching the right operand's ``act_quant_config`` when set, or a high-precision tensor,
    which is quantized here along ``dim=-1`` with that config.

    Args:
        A: Left operand.
        B: Right operand.

    Returns:
        The ``(A, B)`` pair, both as :class:`MXTensor`.

    Raises:
        ValueError:
            - If the contracting dims do not match;
            - If ``B`` is not an :class:`MXTensor`;
            - If either operand is quantized along the wrong dim;
            - If an :class:`MXTensor` ``A`` was not quantized with
              ``B.act_quant_config`` when that config is set;
            - If ``A`` is a quantized tensor of another class;
            - If the two operands' ``quant_config.block_size`` differ.

        RuntimeError: If ``A`` is a high-precision tensor and ``B`` carries no
            ``act_quant_config`` to quantize it with.
    """
    if A.shape[-1] != B.shape[-2]:
        raise ValueError(f"Contracting dim mismatch: {A.shape[-1]} != {B.shape[-2]}.")

    if not isinstance(B, MXTensor):
        raise ValueError(f"The right matmul operand must be a {MXTensor.__name__}, got {type(B).__name__}.")

    if B.quant_axis != B.ndim - 2:
        raise ValueError(f"The right matmul operand must be quantized along dim=-2, got quant_axis={B.quant_axis}.")

    if isinstance(A, MXTensor):
        if A.quant_axis != A.ndim - 1:
            raise ValueError(f"The left matmul operand must be quantized along dim=-1, got quant_axis={A.quant_axis}.")

        if B.act_quant_config is not None and A.quant_config != B.act_quant_config:
            raise ValueError(
                f"The left matmul operand must be quantized with the right operand's ``act_quant_config``; "
                f"got quant_config={A.quant_config} and act_quant_config={B.act_quant_config}."
            )

    elif isinstance(A, BaseQuantizedTensor):
        raise ValueError(
            f"The left matmul operand must be a {MXTensor.__name__} or a high-precision tensor, got {type(A).__name__}."
        )

    else:
        if B.act_quant_config is None:
            raise RuntimeError(
                "The left operand is a high-precision tensor, but the right operand carries no "
                "``act_quant_config`` to quantize it with. Set ``act_quant_config`` on the right operand."
            )

        A = MXTensor.from_hp(A, B.act_quant_config, axis=-1)

    if A.quant_config.block_size != B.quant_config.block_size:
        raise ValueError(
            "The operands must agree on ``block_size``, got "
            f"{A.quant_config.block_size} for ``A`` and {B.quant_config.block_size} for ``B``."
        )

    return A, B


def _quant_matmul(
    mat1: MXTensor,
    mat2: MXTensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``mat1 @ mat2 + bias`` through ``torch_npu.npu_quant_matmul``.

    Args:
        mat1: Left operand, quantized along ``dim=-1``.
        mat2: Right operand, quantized along ``dim=-2``.
        bias: Per-column bias the kernel adds to the product, or ``None`` for a plain product.
        output_dtype: The dtype of the returned tensor, ``mat1.dtype`` when ``None``. It is
            passed to the kernel as-is, so it must be a dtype the kernel supports.

    Returns:
        The product plus ``bias``, a plain tensor of dtype ``output_dtype``.
    """
    return torch_npu.npu_quant_matmul(
        mat1.qdata,
        mat2.qdata,
        mat2.scale,
        pertoken_scale=mat1.scale,
        bias=bias,
        output_dtype=output_dtype or mat1.dtype,
        group_sizes=[1, 1, mat1.quant_config.block_size],
        scale_dtype=mat2.quant_config.npu_scale_dtype,
        pertoken_scale_dtype=mat1.quant_config.npu_scale_dtype,
        x1_dtype=mat1.quant_config.npu_matmul_dtype,
        x2_dtype=mat2.quant_config.npu_matmul_dtype,
    )


def _mx_addmm(
    input: torch.Tensor | None,
    mat1: MXTensor,
    mat2: MXTensor,
    *,
    beta: float = 1,
    alpha: float = 1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``beta * input + alpha * (mat1 @ mat2)`` through ``torch_npu.npu_quant_matmul``.

    ``input`` is fused into the kernel as its per-column ``bias`` when it is added as-is
    (``beta == 1``). The kernel has no scaling terms, so a non-default ``alpha``/``beta``
    is applied to the kernel's output in high precision instead of being fused into it.

    Args:
        input: The added operand, or ``None`` for a plain product.
        mat1: Left operand, quantized along ``dim=-1``.
        mat2: Right operand, quantized along ``dim=-2``.
        beta: The coefficient of ``input``.
        alpha: The coefficient of the product.
        output_dtype: The dtype of the returned tensor, ``mat1.dtype`` when ``None``; passed to
            the kernel as-is, so the scaling above is carried out in it.

    Returns:
        The sum, a plain tensor of dtype ``output_dtype``.
    """
    if input is None or beta == 0:
        if alpha == 1:
            return _quant_matmul(mat1, mat2, output_dtype=output_dtype)
        else:
            return alpha * _quant_matmul(mat1, mat2, output_dtype=output_dtype)

    elif beta == 1:
        if alpha == 1:
            return _quant_matmul(mat1, mat2, input, output_dtype=output_dtype)
        else:
            return alpha * _quant_matmul(mat1, mat2, output_dtype=output_dtype) + input

    else:
        if alpha == 1:
            return _quant_matmul(mat1, mat2, output_dtype=output_dtype) + beta * input
        else:
            return alpha * _quant_matmul(mat1, mat2, output_dtype=output_dtype) + beta * input


@implements([aten.mm.default, aten.mm.dtype])
def _(func, types, args, kwargs):
    A, B = args[0], args[1]
    out_dtype = args[2] if func._schema.overload_name == "dtype" else None

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"``mm`` expects 2D operands, got {A.ndim}D and {B.ndim}D.")

    A_mx, B_mx = _matmul_operands(A, B)
    return _mx_addmm(None, A_mx, B_mx, output_dtype=out_dtype)


@implements([aten.addmm.default, aten.addmm.dtype])
def _(func, types, args, kwargs):
    input, A, B = args[0], args[1], args[2]
    out_dtype = args[3] if func._schema.overload_name == "dtype" else None
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"``addmm`` expects 2D operands, got {A.ndim}D and {B.ndim}D.")

    A_mx, B_mx = _matmul_operands(A, B)
    return _mx_addmm(input, A_mx, B_mx, beta=beta, alpha=alpha, output_dtype=out_dtype)


@implements([aten.bmm.default, aten.bmm.dtype])
def _(func, types, args, kwargs):
    A, B = args[0], args[1]
    out_dtype = args[2] if func._schema.overload_name == "dtype" else None

    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(f"``bmm`` expects 3D operands, got {A.ndim}D and {B.ndim}D.")

    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch dim mismatch: {A.shape[0]} != {B.shape[0]}.")

    A_mx, B_mx = _matmul_operands(A, B)
    return _mx_addmm(None, A_mx, B_mx, output_dtype=out_dtype)


@implements(aten.linear.default)
def _(func, types, args, kwargs):
    input, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    if weight.ndim != 2:
        raise ValueError(f"``linear`` expects a 2D weight, got {weight.ndim}D.")

    input_2d = input if input.ndim == 2 else input.view(-1, input.shape[-1])
    A_mx, B_mx = _matmul_operands(input_2d, weight.t())
    out = _mx_addmm(bias, A_mx, B_mx)

    return out if input.ndim == 2 else out.view(*input.shape[:-1], out.shape[-1])


@implements(aten.matmul.default)
def _(func, types, args, kwargs):
    A, B = args

    if B.ndim != 2:
        raise ValueError(f"``matmul`` expects a 2D right operand, got {B.ndim}D; batched weights go through ``bmm``.")

    A_2d = A if A.ndim == 2 else A.view(-1, A.shape[-1])
    A_mx, B_mx = _matmul_operands(A_2d, B)
    out = _mx_addmm(None, A_mx, B_mx)

    return out if A.ndim == 2 else out.view(*A.shape[:-1], out.shape[-1])


@implements(aten._grouped_mm.default)
def _(func, types, args, kwargs):
    A, B = args[0], args[1]
    offs = args[2] if len(args) > 2 else kwargs.get("offs")
    bias = args[3] if len(args) > 3 else kwargs.get("bias")
    out_dtype = args[4] if len(args) > 4 else kwargs.get("out_dtype")

    A_mx, B_mx = _matmul_operands(A, B)

    # ``group_type=0`` contracts over K; ``group_list_type=0`` takes the
    # cumulative group offsets ``offs`` as-is.
    return torch_npu.npu_grouped_matmul(
        [A_mx.qdata],
        [B_mx.qdata],
        scale=[B_mx.scale],
        per_token_scale=[A_mx.scale],
        group_list=offs.to(torch.int64) if offs is not None else None,
        bias=[bias] if bias is not None else None,
        group_type=0,
        output_dtype=out_dtype if out_dtype is not None else A_mx.dtype,
        group_list_type=0,
        scale_dtype=B_mx.quant_config.npu_scale_dtype,
        per_token_scale_dtype=A_mx.quant_config.npu_scale_dtype,
        x_dtype=A_mx.quant_config.npu_matmul_dtype,
        weight_dtype=B_mx.quant_config.npu_matmul_dtype,
        split_item=3,
    )[0]


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([MXTensor])
