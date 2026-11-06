# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import mxfp4_dequantize
from torchao_npu.quantized_tensors import DualAxisMXTensor, GroupedMXTensor, MXTensor
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor

__all__ = [
    "mxfp4_fake_quantize",
    "mxfp8_dequantize",
    "to_mx_then_bmm",
    "to_mx_then_grouped_mm",
    "to_mx_then_mm",
]


class MXFP4FakeQuantize(torch.autograd.Function):
    """MX FP4 fake quantize via autograd.Function.

    Forward:  quantize to FP4 via ``npu_dynamic_mx_quant``,
              then dequantize back via ``mxfp4_dequantize``.
    Backward: straight-through estimator (gradient passes through unchanged).

    Takes an ``MXQuantizeConfig`` as a single extra argument.
    """

    @staticmethod
    def forward(ctx, hp: torch.Tensor, config: MXQuantizeConfig, axis: int = -1):  # pyrefly: ignore [bad-override]
        weight_mx, w_scale = torch_npu.npu_dynamic_mx_quant(
            hp,
            dst_type=config.npu_elem_dtype,
            axis=axis,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        return mxfp4_dequantize(
            weight_mx,
            w_scale,
            axis=axis,
            block_size=config.block_size,
            output_shape=hp.shape,
            output_dtype=hp.dtype,
        )

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        return grad_output, None, None


def mxfp4_fake_quantize(
    hp: torch.Tensor,
    config: MXQuantizeConfig,
    axis: int = -1,
) -> torch.Tensor:
    """Thin wrapper around ``MXFP4FakeQuantize.apply``."""
    assert config.elem_dtype is torch.float4_e2m1fn_x2, (
        f"mxfp4_fake_quantize only supports elem_dtype=torch.float4_e2m1fn_x2, got {config.elem_dtype}"
    )
    return MXFP4FakeQuantize.apply(hp, config, axis)


def mxfp8_dequantize(
    data: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    block_size: int,
    output_shape: torch.Size,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP8 data from ``torch_npu.npu_dynamic_mx_quant`` output.

    Unlike ``mxfp4_dequantize``, FP8 uses 1 byte per value so no LUT-based
    nibble unpacking is needed. The scale format (E8M0, paired) is the same.

    Args:
        data: FP8 tensor (output y from npu_dynamic_mx_quant).
              Shape matches the original input shape.
        scale: uint8 E8M0 tensor (output mxscale_out).
               ndim = data.ndim + 1, with a trailing 2 dim packing scale pairs.
        axis: Quantization axis used in npu_dynamic_mx_quant.
        block_size: Block size used in npu_dynamic_mx_quant.
        output_dtype: Target dtype for the dequantized output.
    """
    assert output_shape[axis] % block_size == 0, f"quant dim must be divisible by block_size ({block_size})"

    orig_shape = output_shape
    pos_axis = axis if axis >= 0 else axis + data.ndim
    qdim = orig_shape[pos_axis]
    num_blocks = qdim // block_size

    # Convert FP8 → output dtype (no nibble unpacking needed)
    values = data.to(output_dtype)

    # Unpack scale (same paired E8M0 format as mxfp4_dequantize)
    s = scale.to(torch.uint8)
    s = s.movedim(-1, pos_axis + 1)
    s = s.flatten(pos_axis, pos_axis + 1)  # [..., packed_blocks*2, ...]
    s = s.narrow(pos_axis, 0, num_blocks)  # trim padding when num_blocks is odd

    # E8M0 uint8 → bf16 scale: 2^(e - 127)
    s = torch.exp2(s.to(torch.bfloat16) - 127.0)

    # Block-wise broadcast multiply
    values = values.unflatten(pos_axis, (num_blocks, block_size))
    s = s.unsqueeze(pos_axis + 1)

    result = values * s  # broadcasts over block_size
    result = result.reshape(orig_shape)

    return result.to(output_dtype)


# =========================================================================
# Real MX quantized matmul ops
# =========================================================================


def _maybe_quantize_activation_operand(
    A: torch.Tensor,
    config_A: MXQuantizeConfig,
) -> tuple[MXTensor, MXTensor]:
    """Return the activation operand ``A`` as its forward / backward forms.

    The forward form is quantized along ``dim=-1`` (the contracting dim), the
    backward form along ``dim=-2`` (the wgrad's contracting dim).

    - a pre-quantized ``MXTensor`` must hold the forward form and is reused; its
      ``dim=-2`` form is rebuilt by
      :func:`_retrieve_backward_operands_and_may_requantize`;
    - a pre-quantized ``DualAxisMXTensor`` provides both forms through
      ``to_mx_tensor``;
    - a high-precision ``A`` is dual-axis quantized with
      ``DualAxisMXTensor.from_hp``.

    Returns:
        ``A``'s ``dim=-1`` and ``dim=-2`` forms.

    Raises:
        AssertionError: If a pre-quantized ``A`` carries a different
            ``quant_config``, or an ``MXTensor`` ``A`` is not quantized along
            its last dim.
        TypeError: If ``A`` is pre-quantized but neither an ``MXTensor`` nor a
            ``DualAxisMXTensor``.
    """
    if isinstance(A, (MXTensor, DualAxisMXTensor)):
        assert A.quant_config == config_A, (
            "The quantization configuration of the pre-quantized ``A`` is not consistent with ``config_A``, "
            f"``A.quant_config`` is {A.quant_config}, "
            f"``config_A`` is {config_A}."
        )

    if isinstance(A, MXTensor):
        assert A.quant_axis == A.ndim - 1, (
            "The pre-quantized ``MXTensor`` ``A`` must be quantized along its last dim, "
            f"got quant_axis={A.quant_axis} for a {A.ndim}D tensor."
        )
        forward_A = A
        backward_A = A

    elif isinstance(A, DualAxisMXTensor):
        forward_A = A.to_mx_tensor(quant_axis=-1)
        backward_A = A.to_mx_tensor(quant_axis=-2)

    elif isinstance(A, BaseQuantizedTensor):
        raise TypeError(
            f"A must be a high precision tensor, an ``MXTensor`` or a ``DualAxisMXTensor``, got {type(A).__name__}."
        )

    else:
        dual_axis_mx_A = DualAxisMXTensor.from_hp(A, config_A)
        forward_A = dual_axis_mx_A.to_mx_tensor(quant_axis=-1)
        backward_A = dual_axis_mx_A.to_mx_tensor(quant_axis=-2)

    return forward_A, backward_A


def _maybe_quantize_weight_operand(
    B: torch.Tensor,
    config_B: MXQuantizeConfig,
) -> tuple[MXTensor, MXTensor]:
    """Return the weight operand ``B`` as its forward / backward forms.

    The forward form is quantized along ``dim=-2`` (the contracting dim), the
    backward form along ``dim=-1`` (the dgrad's contracting dim).

    - a pre-quantized ``MXTensor`` must hold the forward form and is reused; its
      ``dim=-1`` form is rebuilt in the backward by
      :func:`_retrieve_backward_operands_and_may_requantize` (``mm`` / ``bmm``) or
      :func:`_retrieve_gmm_backward_operands_and_may_requantize` (``grouped_mm``);
    - a pre-quantized ``DualAxisMXTensor`` provides both forms through
      ``to_mx_tensor``;
    - a high-precision ``B`` is dual-axis quantized with
      ``DualAxisMXTensor.from_hp``.

    Returns:
        ``B``'s ``dim=-2`` and ``dim=-1`` forms.

    Raises:
        AssertionError: If a pre-quantized ``B`` carries a different
            ``quant_config``, or an ``MXTensor`` ``B`` is not quantized along
            ``dim=-2``.
        TypeError: If ``B`` is pre-quantized but neither an ``MXTensor`` nor a
            ``DualAxisMXTensor``.
    """
    if isinstance(B, (MXTensor, DualAxisMXTensor)):
        assert B.quant_config == config_B, (
            "The quantization configuration of the pre-quantized ``B`` is not consistent with ``config_B``, "
            f"``B.quant_config`` is {B.quant_config}, "
            f"``config_B`` is {config_B}."
        )

    if isinstance(B, MXTensor):
        assert B.quant_axis == B.ndim - 2, (
            "The pre-quantized ``MXTensor`` ``B`` must be quantized along its second-to-last dim, "
            f"got quant_axis={B.quant_axis} for a {B.ndim}D tensor."
        )
        forward_B = B
        backward_B = B

    elif isinstance(B, DualAxisMXTensor):
        forward_B = B.to_mx_tensor(quant_axis=-2)
        backward_B = B.to_mx_tensor(quant_axis=-1)

    elif isinstance(B, BaseQuantizedTensor):
        raise TypeError(
            f"B must be a high precision tensor, an ``MXTensor`` or a ``DualAxisMXTensor``, got {type(B).__name__}."
        )

    else:
        dual_axis_mx_B = DualAxisMXTensor.from_hp(B, config_B)
        forward_B = dual_axis_mx_B.to_mx_tensor(quant_axis=-2)
        backward_B = dual_axis_mx_B.to_mx_tensor(quant_axis=-1)

    return forward_B, backward_B


def _maybe_quantize_operands_then_save_for_backward(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> tuple[MXTensor, MXTensor]:
    """Quantize the operands unless already quantized, return the forward forms
    and save the backward forms.

    ``A`` goes through :func:`_maybe_quantize_activation_operand` and ``B`` through
    :func:`_maybe_quantize_weight_operand`. The unbatched ``mm`` flattens ``A`` to
    2D before this call, so its row blocks run across every leading dim.

    Saves ``A``'s ``dim=-2`` form and ``B``'s ``dim=-1`` form, both as
    :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`, for the backward.

    Returns:
        ``A``'s ``dim=-1`` and ``B``'s ``dim=-2`` form.

    Raises:
        AssertionError: If a pre-quantized operand carries a different
            ``quant_config``, or an ``MXTensor`` operand is not quantized along
            the dim the forward needs.
        TypeError: If an operand is pre-quantized but neither an ``MXTensor`` nor
            a ``DualAxisMXTensor``.
    """
    forward_A, backward_A = _maybe_quantize_activation_operand(A, config_A)
    forward_B, backward_B = _maybe_quantize_weight_operand(B, config_B)

    ctx.save_for_backward(backward_A, backward_B)

    return forward_A, forward_B


def _retrieve_backward_operands_and_may_requantize(ctx) -> tuple[MXTensor, MXTensor]:
    """Return the saved ``A``/``B`` in the forms the backward GEMMs need.

    The saved ``A`` must be quantized along ``dim=-2`` and the saved ``B`` along
    ``dim=-1``; the ``DualAxisMXTensor`` and high-precision paths save exactly those
    forms. A saved ``dim=-1`` ``A`` (or ``dim=-2`` ``B``) means the operand arrived
    pre-quantized as an ``MXTensor`` holding only its forward form: the backward form
    does not exist, so it is rebuilt by dequantizing along the stored axis and
    requantizing along the other one.
    """
    backward_A, backward_B = ctx.saved_tensors

    assert isinstance(backward_A, MXTensor), (
        f"Expected the saved ``A`` to be an ``MXTensor``, got {type(backward_A).__name__}."
    )
    assert isinstance(backward_B, MXTensor), (
        f"Expected the saved ``B`` to be an ``MXTensor``, got {type(backward_B).__name__}."
    )

    if backward_A.quant_axis == backward_A.ndim - 1:
        backward_A = MXTensor.from_hp(backward_A.dequantize(), quant_config=backward_A.quant_config, axis=-2)

    if backward_B.quant_axis == backward_B.ndim - 2:
        backward_B = MXTensor.from_hp(backward_B.dequantize(), quant_config=backward_B.quant_config, axis=-1)

    return backward_A, backward_B


class _MXQuantMM(torch.autograd.Function):
    """Per-axis MX matrix multiply: ``A[M,K] @ B[K,N] = Y[M,N]``.

    - A can be a high precision tensor or an MXTensor quantized along its last dim
      (it is flattened to 2D first, so it can be n-dimensional)
    - B can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along dim=-2

    If either operand is in high precision, it is quantized first, then multiplied.

    Forward:
        - Low-precision matmul contracting over K.
        - The quantized operands are saved in ctx for the backward.


    Backward (``dY = grad_output``):
        - dY is dual-axis quantized using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = A^T @ dY`` (contract over M)
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        forward_A, forward_B = _maybe_quantize_operands_then_save_for_backward(
            ctx, A.reshape(-1, A.shape[-1]), B, config_A, config_B
        )

        # Low-precision matmul, contracting over K. The operand classes own the
        # kernel call and its per-config dtypes.
        Y = torch.mm(forward_A, forward_B)

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        mx_A, mx_B = _retrieve_backward_operands_and_may_requantize(ctx)

        # dY is quantized along both axes with A's config, providing both GEMMs'
        # dY operands below.
        dual_axis_mx_Y = DualAxisMXTensor.from_hp(
            dY.reshape(-1, dY.shape[-1]), mx_A.quant_config, act_quant_config=mx_A.quant_config
        )
        mx_dY_1 = dual_axis_mx_Y.to_mx_tensor(quant_axis=-1)
        mx_dY_2 = dual_axis_mx_Y.to_mx_tensor(quant_axis=-2)

        # dgrad: dA = dY @ B^T (contract over N) -- dY's -1 form against the saved
        # B form transposed.
        dA = torch.mm(mx_dY_1, mx_B.transpose(-1, -2))

        # wgrad: dB = A^T @ dY (contract over M) -- the saved A form transposed
        # against dY's -2 form.
        dB = torch.mm(mx_A.transpose(-1, -2), mx_dY_2)

        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_mx_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized matrix multiply: ``A @ B``.

    ``A`` can be n-dimensional (n >= 2); leading dimensions are flattened
    before quantization and restored in the output.  ``B`` must be 2D.

    ``A`` may be high precision or an ``MXTensor`` quantized along its last dim;
    ``B`` may be high precision, a ``DualAxisMXTensor``, or an ``MXTensor``
    quantized along ``dim=-2``. A pre-quantized operand that lacks the other
    backward form is dequantized and re-quantized once in the backward.

    Quantization configs are drawn from ``config_A`` (for A and dY) and
    ``config_B`` (for B).

    Args:
        A: Shape ``(..., M, K)`` (n >= 2).
        B: Shape ``(K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(..., M, N)``, dtype matching ``A``.
    """
    return _MXQuantMM.apply(A, B, config_A, config_B)


def _maybe_quantize_gmm_operands_then_save_for_backward(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> tuple[MXTensor, MXTensor]:
    """Quantize the grouped matmul's operands, return the forward forms and save the
    backward forms.

    ``A`` is either high precision or a pre-quantized ``MXTensor`` quantized along
    its last dim. A high-precision ``A`` is quantized along ``dim=-1`` (the
    contracting dim) for the forward, and along ``dim=-2`` with a group-aware quant
    op for the wgrad. A pre-quantized ``A`` provides its ``dim=-1`` form only; its
    group-aware ``dim=-2`` form is rebuilt in
    :func:`_retrieve_gmm_backward_operands_and_maybe_requantize`.

    ``B`` goes through :func:`_maybe_quantize_weight_operand`.

    Saves ``A``'s ``dim=-2`` form -- a group-aware
    :class:`~torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor` for a
    high-precision ``A``, or the ``MXTensor`` itself -- and ``B``'s ``dim=-1`` form
    (a pre-quantized ``MXTensor`` ``B`` is saved as its ``dim=-2`` forward form
    instead and rebuilt in the backward), plus the group offsets, for the backward.

    Returns:
        ``A``'s ``dim=-1`` and ``B``'s ``dim=-2`` form, for the forward grouped matmul.

    Raises:
        AssertionError: If a pre-quantized ``A`` carries a different
            ``quant_config``, or an ``MXTensor`` ``A`` is not quantized along
            its last dim.
        TypeError: If ``A`` is pre-quantized but not an ``MXTensor`` -- notably a
            ``DualAxisMXTensor``, whose stored forms cannot provide the group-aware
            wgrad form.
    """
    if isinstance(A, MXTensor):
        assert A.quant_config == config_A, (
            "The quantization configuration of the pre-quantized ``A`` is not consistent with ``config_A``, "
            f"``A.quant_config`` is {A.quant_config}, "
            f"``config_A`` is {config_A}."
        )
        assert A.quant_axis == A.ndim - 1, (
            "The pre-quantized ``MXTensor`` ``A`` must be quantized along its last dim, "
            f"got quant_axis={A.quant_axis} for a {A.ndim}D tensor."
        )
        forward_A = A
        backward_A = A

    elif not isinstance(A, BaseQuantizedTensor):
        # A's contracting-dim form, for the forward grouped matmul.
        forward_A = MXTensor.from_hp(A, config_A, axis=-1)

        # A's M-dim form, for the wgrad: a grouped quant with zero-boundary padding.
        # A [M, K] → A_s2 (M-dim scale) [M//64 + E, K, 2]
        backward_A = GroupedMXTensor.from_hp(A, offs, config_A)

    else:
        raise TypeError(f"A must be a high precision tensor or an MXTensor, got {type(A).__name__}.")

    forward_B, backward_B = _maybe_quantize_weight_operand(B, config_B)

    ctx.save_for_backward(backward_A, backward_B, offs)

    return forward_A, forward_B


def _retrieve_gmm_backward_operands_and_maybe_requantize(ctx) -> tuple[GroupedMXTensor, MXTensor, torch.Tensor]:
    """Return the saved ``A``/``B``/offsets in the forms the grouped backward needs.

    A saved ``GroupedMXTensor`` (the high-precision path) is reused as is. A saved
    ``MXTensor`` means ``A`` arrived pre-quantized along its last dim: its group-aware
    ``dim=-2`` form is rebuilt from its dequantized values.

    The dgrad needs ``B`` quantized along ``dim=-1`` (that dim becomes the contracting
    one after the transpose); the high-precision and ``DualAxisMXTensor`` paths save
    exactly that form, while a pre-quantized ``MXTensor`` ``B`` holds its ``dim=-2``
    forward form and is rebuilt the same way as ``A``.
    """
    backward_A, mx_B, offs = ctx.saved_tensors

    if isinstance(backward_A, GroupedMXTensor):
        grouped_mx_A = backward_A

    elif isinstance(backward_A, MXTensor):
        grouped_mx_A = GroupedMXTensor.from_hp(backward_A.dequantize(), offs, quant_config=backward_A.quant_config)

    else:
        raise TypeError(
            f"Expected the saved ``A`` to be an ``MXTensor`` or a ``GroupedMXTensor``, got {type(backward_A).__name__}."
        )

    assert isinstance(mx_B, MXTensor), f"Expected the saved ``B`` to be an ``MXTensor``, got {type(mx_B).__name__}."

    if mx_B.quant_axis == mx_B.ndim - 2:
        mx_B = MXTensor.from_hp(mx_B.dequantize(), quant_config=mx_B.quant_config, axis=-1)

    return grouped_mx_A, mx_B, offs


class _MXQuantGroupedMM(torch.autograd.Function):
    """Per-axis MX quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``). Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``.

    - A must be a high precision tensor or an MXTensor quantized along its last dim
    - B can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along dim=-2

    If either operand is in high precision, it is quantized first, then multiplied.

    Quantization:
    - A's K-dim (contraction):  ``npu_dynamic_mx_quant(axis=-1)`` (params from
                                ``config_A``); skipped when ``A`` is already an
                                ``MXTensor``
    - A's M-dim (for wgrad):    ``npu_grouped_dynamic_mx_quant`` (zero-boundary padded;
                                params from ``config_A``)
    - B:                        ``npu_dynamic_mx_quant_with_dual_axis`` (params from
                                ``config_B``)

    Forward:
        - Low-precision grouped matmul contracting over K.
        - Quantized version of A and B are saved in ctx for backward.
        - A's M-dim form is quantized by a group-aware op, so it is a
          :class:`~torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor`.


    Backward (``dY = grad_output``):
        - A last-axis MXTensor dY is reused; a high-precision dY is quantized
          using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = A^T @ dY`` (contract over M, ``group_type=2``).
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        group_list: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        forward_A, forward_B = _maybe_quantize_gmm_operands_then_save_for_backward(
            ctx, A, B, group_list, config_A, config_B
        )

        # Grouped low-precision matmul, group_type=0 (contract over K). The operand
        # classes own the kernel call and its per-config dtypes.
        Y = torch.nn.functional.grouped_mm(forward_A, forward_B, offs=group_list)

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        grouped_mx_A, mx_B, offs = _retrieve_gmm_backward_operands_and_maybe_requantize(ctx)

        # The inverse combine exchange can provide dY already quantized along N.
        # Reuse that wire representation for dgrad; wgrad still needs the logical
        # values to build its group-aware M-dim form.
        if isinstance(dY, MXTensor):
            assert dY.quant_config == grouped_mx_A.quant_config, (
                "The quantization configuration of the pre-quantized ``dY`` is not consistent with that of ``A``."
            )
            assert dY.quant_axis == dY.ndim - 1, (
                "The pre-quantized ``MXTensor`` ``dY`` must be quantized along its last dim, "
                f"got quant_axis={dY.quant_axis} for a {dY.ndim}D tensor."
            )
            mx_dY = dY
            logical_dY = dY.dequantize()
        else:
            # dY [M, N] → dY_s1 (N-dim scale) [M, ceil(ceil(N/32)/2), 2]
            mx_dY = MXTensor.from_hp(dY, grouped_mx_A.quant_config, axis=-1)
            logical_dY = dY

        # dY [M, N] → dY_s2 (M-dim scale) [M//64 + E, N, 2]
        grouped_mx_dY = GroupedMXTensor.from_hp(logical_dY, offs, grouped_mx_A.quant_config)

        # dgrad: dA = dY @ B^T (group_type=0, contract over N)
        # - dY's N-dim form against the saved B form transposed.
        dA = torch.nn.functional.grouped_mm(mx_dY, mx_B.transpose(-1, -2), offs=offs)

        # wgrad: dB = A^T @ dY (group_type=2, contract over each group's M dim)
        # - A's group-aware M-dim form transposed against dY's group-aware form.
        dB = torch.nn.functional.grouped_mm(grouped_mx_A.t(), grouped_mx_dY, offs=offs)

        return dA, dB, None, None, None


def to_mx_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``). Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``.

    ``A`` must be high precision or an ``MXTensor`` quantized along its last dim;
    ``B`` may be high precision, a ``DualAxisMXTensor``, or an ``MXTensor``
    quantized along ``dim=-2``. The wgrad always needs A's group-aware form, so a
    pre-quantized activation is dequantized and re-quantized there.

    Args:
        A: Shape ``(M, K)``.
        B: Shape ``(E, K, N)``.
        group_list: Cumulative row offsets per group, shape ``(E,)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _MXQuantGroupedMM.apply(A, B, group_list, config_A, config_B)


class _MXQuantBMM(torch.autograd.Function):
    """Per-axis MX quantized batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and share the leading batch dimension.

    - A can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along its last dim
    - B can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along dim=-2

    If either operand is in high precision, it is quantized first, then multiplied.

    Forward:
        - Low-precision batched matmul contracting over K.
        - The quantized operands are saved in ctx for the backward.


    Backward (``dY = grad_output``):
        - dY is dual-axis quantized using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = A^T @ dY`` (contract over M)
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: MXQuantizeConfig,
    ):
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[0] == B.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B[0]={B.shape[0]}"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        forward_A, forward_B = _maybe_quantize_operands_then_save_for_backward(ctx, A, B, config_A, config_B)

        # Low-precision batched matmul, contracting over K. The operand classes own
        # the kernel call and its per-config dtypes.
        Y = torch.bmm(forward_A, forward_B)

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        mx_A, mx_B = _retrieve_backward_operands_and_may_requantize(ctx)

        # dY is quantized along both axes with A's config, providing both GEMMs'
        # dY operands below.
        dual_axis_mx_Y = DualAxisMXTensor.from_hp(dY, mx_A.quant_config, act_quant_config=mx_A.quant_config)
        mx_dY_1 = dual_axis_mx_Y.to_mx_tensor(quant_axis=-1)
        mx_dY_2 = dual_axis_mx_Y.to_mx_tensor(quant_axis=-2)

        # dgrad: dA = dY @ B^T (contract over N)
        # - dY's -1 form against the saved B form transposed.
        dA = torch.bmm(mx_dY_1, mx_B.transpose(-1, -2))

        # wgrad: dB = A^T @ dY (contract over M)
        # - the saved A form transposed against dY's -2 form.
        dB = torch.bmm(mx_A.transpose(-1, -2), mx_dY_2)

        return dA, dB, None, None


def to_mx_then_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: MXQuantizeConfig,
) -> torch.Tensor:
    """Per-axis MX quantized batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and must share the leading batch dimension.
    ``A`` may be high precision, a ``DualAxisMXTensor``, or an ``MXTensor``
    quantized along its last dim; ``B`` may be high precision, a
    ``DualAxisMXTensor``, or an ``MXTensor`` quantized along ``dim=-2``.
    Quantization configs are drawn from ``config_A`` (for A and dY) and
    ``config_B`` (for B).

    Args:
        A: Shape ``(B, M, K)``.
        B: Shape ``(B, K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(B, M, N)``, dtype matching ``A``.
    """
    return _MXQuantBMM.apply(A, B, config_A, config_B)
