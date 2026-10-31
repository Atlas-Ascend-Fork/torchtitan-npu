# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block MX low-precision matmul ops for training on NPU.

Each op quantizes its high precision operands into the tensors of
:mod:`torchao_npu.quantized_tensors`, which own the layouts and the operand checks,
and multiplies those.

Quantization scheme (non-grouped ``to_block_mx_then_mm`` / ``to_block_mx_then_bmm``):
- A (activation):           a ``DualAxisMXTensor``, or a pre-quantized ``MXTensor``
                            quantized along its last dim
- B ((transposed) weight):  a ``BlockMXTensor``
- dY:                       a ``DualAxisMXTensor``

Quantization scheme (grouped ``to_block_mx_then_grouped_mm``):
- A (activation), K-dim:    an ``MXTensor``
- A (activation), M-dim:    a ``GroupedMXTensor``
- B (weight):               a ``BlockMXTensor``
- dY, N-dim:                an ``MXTensor``
- dY, M-dim:                a ``GroupedMXTensor``
"""

import torch

from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantized_tensors import BlockMXTensor, DualAxisMXTensor, GroupedMXTensor, MXTensor
from torchao_npu.quantized_tensors.base_quantized_tensor import BaseQuantizedTensor

__all__ = [
    "to_block_mx_then_bmm",
    "to_block_mx_then_grouped_mm",
    "to_block_mx_then_mm",
]


def _maybe_quantize_weight_operand(
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
    axis: int,
) -> tuple[MXTensor, MXTensor]:
    """Return the weight operand ``B`` as its forward / backward :class:`MXTensor` forms.

    ``B`` is brought into the block-MX form first:

    - a pre-quantized ``MXTensor`` is converted with ``B.to_block_mx(config_B)``, after its
      ``quant_config`` and ``act_quant_config`` are validated against
      ``config_B.mxfp4_fake_quantize_config`` and ``config_A``;
    - a pre-quantized ``BlockMXTensor`` is reused as-is after its ``quant_config`` is
      validated to be the caller's ``config_B`` and its ``act_quant_config`` to be
      either ``None`` or ``config_A``;
    - a high-precision ``B`` is quantized with ``config_B`` -- an MXFP4 QAT config still
      applies its FP4 fake-quantization pre-pass -- recording ``config_A`` as its
      ``act_quant_config``.

    Returns:
        ``B``'s ``dim=-2`` and ``dim=-1`` forms, for the forward and the backward.

    Raises:
        AssertionError: If a pre-quantized ``B`` carries a different ``quant_config``
            (the nested MXFP4 config for an ``MXTensor``), or an ``act_quant_config``
            that is neither ``None`` nor ``config_A``.
    """
    if isinstance(B, MXTensor):
        assert config_B.mxfp4_fake_quantize_config == B.quant_config, (
            "The quantization configuration of the pre-quantized ``B`` is not consistent with "
            f"``config_B.mxfp4_fake_quantize_config``, {B.quant_config=}, "
            f"{config_B.mxfp4_fake_quantize_config=}"
        )

        if B.act_quant_config is not None:
            assert B.act_quant_config == config_A, (
                "The activation quantization configuration of the pre-quantized ``B`` is not consistent with "
                f"that of the wrapper tensor, ``B.act_quant_config`` is {B.act_quant_config}, that of the "
                f"wrapper tensor is {config_A}."
            )

        B = B.to_block_mx(config_B)

    if isinstance(B, BlockMXTensor):
        assert B.quant_config is config_B, (
            "The quantization configuration of the pre-quantized ``B`` is not consistent with ``config_B``, "
            f"``B.quant_config`` is {B.quant_config}, "
            f"``config_B`` is {config_B}."
        )

        if B.act_quant_config is not None:
            assert B.act_quant_config == config_A, (
                "The activation quantization configuration of the pre-quantized ``B`` is not consistent with "
                f"that of the wrapper tensor. ``B.act_quant_config`` is {B.act_quant_config}, that of the "
                f"wrapper tensor is {config_A}."
            )

        quantized_B = B

    else:
        quantized_B = BlockMXTensor.from_hp(B, config_B, axis, config_A)

    return quantized_B.to_mx_tensor(-2), quantized_B.to_mx_tensor(-1)


def _maybe_quantize_operands_then_save_for_backward(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
    axis: int,
) -> tuple[MXTensor, MXTensor]:
    """Quantize the operands unless already quantized, return the forward forms
    and save the backward forms.

    A pre-quantized ``DualAxisMXTensor`` or ``MXTensor`` ``A`` is validated against
    ``config_A`` and reused. The ``MXTensor`` must be quantized along its last dim
    (the forward form); its ``dim=-2`` backward form is rebuilt by
    :func:`_retrieve_backward_operands_and_maybe_requantize`. A ``DualAxisMXTensor``
    saves its ``dim=-2`` form directly. ``B`` goes through
    :func:`_maybe_quantize_weight_operand`. Saves ``A``'s backward form and ``B``'s
    ``dim=-1`` form, both as
    :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`.

    Returns:
        ``A``'s ``dim=-1`` and ``B``'s ``dim=-2`` form.

    Raises:
        AssertionError: If a pre-quantized ``A`` carries a different ``quant_config``,
            or an ``MXTensor`` ``A`` is not quantized along its last dim.
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

    else:
        dual_axis_mx_A = A if isinstance(A, DualAxisMXTensor) else DualAxisMXTensor.from_hp(A, config_A)
        forward_A = dual_axis_mx_A.to_mx_tensor(quant_axis=-1)
        backward_A = dual_axis_mx_A.to_mx_tensor(quant_axis=-2)

    forward_B, backward_B = _maybe_quantize_weight_operand(B, config_A, config_B, axis)

    ctx.save_for_backward(backward_A, backward_B)

    return forward_A, forward_B


def _retrieve_backward_operands_and_maybe_requantize(ctx) -> tuple[MXTensor, MXTensor]:
    """Return the saved ``A``/``B`` in the forms the backward GEMMs need.

    A saved ``dim=-2`` form (the ``DualAxisMXTensor`` path) is reused as is. A saved
    ``dim=-1`` form means ``A`` arrived pre-quantized as an ``MXTensor``: its
    ``dim=-2`` form does not exist, so it is rebuilt by dequantizing along the last
    dim and requantizing along ``dim=-2``.
    """
    backward_A, mx_B = ctx.saved_tensors

    assert isinstance(backward_A, MXTensor), (
        f"Expected the saved ``A`` to be an ``MXTensor``, got {type(backward_A).__name__}."
    )
    assert isinstance(mx_B, MXTensor), f"Expected the saved ``B`` to be an ``MXTensor``, got {type(mx_B).__name__}."

    if backward_A.quant_axis == backward_A.ndim - 1:
        dequant_A = backward_A.dequantize()
        mx_A = MXTensor.from_hp(dequant_A, quant_config=backward_A.quant_config, axis=-2)
    else:
        mx_A = backward_A

    return mx_A, mx_B


class _BlockMXQuantMM(torch.autograd.Function):
    """Block MX matrix multiply: ``A[..., M, K] @ B[K, N] = Y[..., M, N]``.

    - A can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along its last dim
    - B can be a high precision tensor, a BlockMXTensor, or a pre-quantized MXTensor
      (converted to block MX)

    If either A or B is in high precision, it will be quantized first then multiplied.

    Forward:
        - Low-precision matmul contracting over K.
        - Quantized version of A and B are saved in ctx for backward.


    Backward (``dY = grad_output``):
        - dY is quantized using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = (dY^T @ A)^T`` (contract over M). The final transpose
          is a view, exposing contiguous dW after the outer weight transpose backward.
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim >= 2, f"A must be >=2D, got {A.ndim}D"
        assert B.ndim == 2, f"B must be 2D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        mx_A, mx_B = _maybe_quantize_operands_then_save_for_backward(
            ctx, A.reshape(-1, A.shape[-1]), B, config_A, config_B, axis=-2
        )

        Y = torch.mm(mx_A, mx_B)

        if A.ndim != 2:
            Y = Y.reshape(*A.shape[:-1], *Y.shape[1:])

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        mx_A, mx_B = _retrieve_backward_operands_and_maybe_requantize(ctx)

        # dY is quantized along both axes with the same config as A's, providing
        # both GEMMs' dY operands below.
        dual_axis_mx_Y = DualAxisMXTensor.from_hp(
            dY.reshape(-1, dY.shape[-1]), mx_A.quant_config, act_quant_config=mx_A.quant_config
        )

        # dgrad: dA = dY @ B^T (contract over N) -- dY's -1 form against the saved
        # B form transposed.
        dA = torch.mm(dual_axis_mx_Y.to_mx_tensor(quant_axis=-1), mx_B.transpose(-1, -2))

        # wgrad: dB = (dY^T @ A)^T, contracting over M.
        # Match B = W.T so the outer transpose backward exposes contiguous dW.
        dB = torch.mm(dual_axis_mx_Y.to_mx_tensor(quant_axis=-2).transpose(-1, -2), mx_A).transpose(-1, -2)

        # Restore dA to original shape
        if dY.ndim != 2:
            dA = dA.reshape(*dY.shape[:-1], *dA.shape[1:])

        return dA, dB, None, None


def to_block_mx_then_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX matrix multiply: ``A @ B`` with mixed-format quantization.

    - ``A`` may be n-dimensional (n >= 2): leading dimensions are flattened before
      quantization and restored in the output. It may already be a
      ``DualAxisMXTensor`` or an ``MXTensor`` quantized along its last dim, validated
      against ``config_A``.
    - ``B`` must be 2D. It may already be a ``BlockMXTensor`` or an ``MXTensor``
      converted to block MX, validated against ``config_B``.

    High-precision operands are quantized before the matmul.

    See ``_BlockMXQuantMM`` for details of the three GEMMs.

    Args:
        A: Shape ``(..., M, K)`` (n >= 2).
        B: Shape ``(K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(..., M, N)``, dtype matching ``A``.
    """
    return _BlockMXQuantMM.apply(A, B, config_A, config_B)


def _maybe_quantize_gmm_operands_then_save_for_backward(
    ctx,
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> tuple[MXTensor, MXTensor]:
    """Quantize the grouped matmul's operands, return the forward forms and save the
    backward forms.

    ``A`` is either high precision or a pre-quantized ``MXTensor`` quantized along its
    last dim. A high-precision ``A`` is quantized along both axes here: along ``dim=-1``
    (the contracting dim) for the forward, and along ``dim=-2`` with a group-aware quant
    op for the wgrad. An ``MXTensor`` ``A`` is used as its ``dim=-1`` form; its
    group-aware ``dim=-2`` form is rebuilt in
    :func:`_retrieve_gmm_backward_operands_and_maybe_requantize`.

    ``B`` goes through :func:`_maybe_quantize_weight_operand`.

    Saves ``A``'s ``dim=-2`` form -- a group-aware
    :class:`~torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor` for a
    high-precision ``A``, or the ``MXTensor`` itself -- and ``B``'s ``dim=-1`` form,
    plus the group offsets, for the backward.

    Returns:
        ``A``'s ``dim=-1`` and ``B``'s ``dim=-2`` form, for the forward grouped matmul.

    Raises:
        AssertionError: If a pre-quantized ``A`` carries a different ``quant_config``,
            or an ``MXTensor`` ``A`` is not quantized along its last dim.
        TypeError: If ``A`` is pre-quantized but not a :class:`~torchao_npu.quantized_tensors.mx_tensor.MXTensor`.
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
        forward_A = MXTensor.from_hp(A, config_A, -1)

        # A's M-dim form, for the wgrad: a grouped quant with zero-boundary padding.
        # A [M, K] → A_s2 (M-dim scale) [M//64 + E, K, 2]
        backward_A = GroupedMXTensor.from_hp(A, offs, config_A)

    else:
        raise TypeError(f"A must be a high precision tensor or an MXTensor, got {type(A).__name__}.")

    forward_B, backward_B = _maybe_quantize_weight_operand(B, config_A, config_B, axis=-2)

    ctx.save_for_backward(backward_A, backward_B, offs)

    return forward_A, forward_B


def _retrieve_gmm_backward_operands_and_maybe_requantize(ctx) -> tuple[GroupedMXTensor, MXTensor, torch.Tensor]:
    """Return the saved ``A``/``B``/offsets in the forms the grouped backward needs.

    A saved ``GroupedMXTensor`` (the high-precision path) is reused as is. A saved
    ``MXTensor`` means ``A`` arrived pre-quantized along its last dim: its group-aware
    ``dim=-2`` form is rebuilt from its dequantized values.
    """
    backward_A, mx_B, offs = ctx.saved_tensors

    if isinstance(backward_A, MXTensor):
        dequant_A = backward_A.dequantize()
        mx_A = GroupedMXTensor.from_hp(dequant_A, offs, quant_config=backward_A.quant_config)

    elif isinstance(backward_A, GroupedMXTensor):
        mx_A = backward_A

    else:
        raise TypeError(
            f"Expected the saved ``A`` to be an ``MXTensor`` or a ``GroupedMXTensor``, got {type(backward_A).__name__}."
        )

    return mx_A, mx_B, offs


class _BlockMXQuantGroupedMM(torch.autograd.Function):
    """Block MX grouped matmul: ``A[M,K] @ B[E,K,N] = Y[M,N]``.

    Rows of ``A`` are partitioned into ``E`` groups via ``group_list``
    (cumsum offsets, length ``E``).  Each slice ``A[group_list[i-1]:group_list[i]]``
    is multiplied by ``B[i]``.

    - A must be a high precision tensor or an MXTensor quantized along its last dim
    - B can be a high precision tensor, a BlockMXTensor, or a pre-quantized MXTensor
      (converted to block MX)

    If B is in high precision, it will be quantized first then multiplied.

    Quantization:
    - A's K-dim (contraction):  ``npu_dynamic_mx_quant(axis=-1)`` (params from
                                ``config_A``); skipped when ``A`` is already an
                                ``MXTensor``
    - A's M-dim (for wgrad):    ``npu_grouped_dynamic_mx_quant`` (zero-boundary padded;
                                params from ``config_A``)
    - B:                        ``npu_dynamic_block_mx_quant``, 32×32 blocks
                                (params from ``config_B``); a pre-quantized
                                ``MXTensor`` is converted with ``MXTensor.to_block_mx``

    Forward:
        - Low-precision grouped matmul contracting over K.
        - Quantized version of A and B are saved in ctx for backward. The saved A is
          the group-aware
          :class:`~torchao_npu.quantized_tensors.grouped_mx_tensor.GroupedMXTensor`
          for a high-precision ``A``, or the ``MXTensor`` itself for a pre-quantized
          ``A``; its group-aware form is rebuilt in the backward.


    Backward (``dY = grad_output``):
        - dY is quantized using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = (dY^T @ A)^T`` (contract over M, ``group_type=2``).
          Group boundaries are unchanged; the final transpose is a view.
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        group_list: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim == 2, f"A must be 2D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        mx_A, mx_B = _maybe_quantize_gmm_operands_then_save_for_backward(ctx, A, B, group_list, config_A, config_B)

        return torch.nn.functional.grouped_mm(mx_A, mx_B, offs=group_list)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        assert dY.ndim == 2, f"dY must be 2D, got {dY.ndim}D"

        grouped_mx_A, mx_B, offs = _retrieve_gmm_backward_operands_and_maybe_requantize(ctx)

        # dY is quantized along both axes with the same config as A's, providing both
        # GEMMs' dY operands below: the N-dim form for the dgrad, the group-aware M-dim
        # form for the wgrad.
        # dY [M, N] → dY_s1 (N-dim scale) [M, ceil(ceil(N/32)/2), 2]
        mx_dY = MXTensor.from_hp(dY, grouped_mx_A.quant_config, axis=-1)

        # dY [M, N] → dY_s2 (M-dim scale) [M//64 + E, N, 2]
        grouped_mx_dY = GroupedMXTensor.from_hp(dY, offs, grouped_mx_A.quant_config)

        # dgrad: dA = dY @ B^T (contract over N)
        # - dY's N-dim form against the saved B form transposed.
        dA = torch.nn.functional.grouped_mm(
            mx_dY,
            mx_B.transpose(-1, -2),
            offs=offs,
        )

        # wgrad: dB = (dY^T @ A)^T, contracting over each group's M dim.
        # Match B = W.T so the outer transpose backward exposes contiguous dW.
        dB = torch.nn.functional.grouped_mm(grouped_mx_dY.t(), grouped_mx_A, offs=offs).transpose(-1, -2)

        return dA, dB, None, None, None


def to_block_mx_then_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    group_list: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX grouped matrix multiply with mixed-format quantization.

    ``A`` is a 2D tensor ``(M, K)`` whose rows are partitioned among
    ``E`` groups via ``group_list`` (cumsum offsets, length ``E``).
    ``B`` is a 3D tensor ``(E, K, N)``.  Computes the equivalent of
    ``A @ B`` where each group's assigned rows are multiplied by its
    corresponding slice of ``B``.

    - ``A`` is a high precision tensor or a pre-quantized ``MXTensor`` quantized
      along its last dim, validated against ``config_A``.
    - ``B`` may already be a ``BlockMXTensor`` or a pre-quantized ``MXTensor``, validated against ``config_B``.

    High-precision operands are quantized before the matmul.

    See ``_BlockMXQuantGroupedMM`` for the quantization scheme and details of the
    three GEMMs.

    Args:
        A: Input tensor, shape ``(M, K)``.
        B: Tensor, shape ``(E, K, N)``.
        group_list: Cumulative row offsets per group, shape ``(E,)``
            (no leading zero, length equals number of groups ``E``).
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(M, N)``, dtype matching ``A``.
    """
    return _BlockMXQuantGroupedMM.apply(A, B, group_list, config_A, config_B)


class _BlockMXQuantBMM(torch.autograd.Function):
    """Block MX batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    - A can be a high precision tensor, a DualAxisMXTensor, or an MXTensor quantized
      along its last dim
    - B can be a high precision tensor, a BlockMXTensor, or a pre-quantized MXTensor
      (converted to block MX)

    If either A or B is in high precision, it will be quantized first then multiplied.

    Forward:
        - Low-precision batched matmul contracting over K.
        - Quantized version of A and B are saved in ctx for backward.


    Backward (``dY = grad_output``):
        - dY is quantized using A's config.
        - dgrad ``dA = dY @ B^T`` (contract over N)
        - wgrad ``dB = (dY^T @ A)^T`` (contract over M). The final transpose
          is a view, exposing contiguous dW after the outer weight transpose backward.
        - All multiplications are done in low precision.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config_A: MXQuantizeConfig,
        config_B: BlockMXQuantizeConfig,
    ):
        assert A.ndim == 3, f"A must be 3D, got {A.ndim}D"
        assert B.ndim == 3, f"B must be 3D, got {B.ndim}D"
        assert A.shape[0] == B.shape[0], f"batch dim mismatch: A[0]={A.shape[0]} != B[0]={B.shape[0]}"
        assert A.shape[-1] == B.shape[-2], f"contracting dim mismatch: A[-1]={A.shape[-1]} != B[-2]={B.shape[-2]}"

        mx_A, mx_B = _maybe_quantize_operands_then_save_for_backward(ctx, A, B, config_A, config_B, axis=-2)

        return torch.bmm(mx_A, mx_B)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):  # pyrefly: ignore [bad-override]
        mx_A, mx_B = _retrieve_backward_operands_and_maybe_requantize(ctx)

        # dY is quantized along both axes with the same config as A's, providing
        # both GEMMs' dY operands below.
        dual_axis_mx_Y = DualAxisMXTensor.from_hp(dY, mx_A.quant_config, act_quant_config=mx_A.quant_config)

        # dgrad: dA = dY @ B^T (contract over N)
        # - dY's -1 form against the saved B form transposed.
        dA = torch.bmm(dual_axis_mx_Y.to_mx_tensor(quant_axis=-1), mx_B.transpose(-1, -2))

        # wgrad: dB = (dY^T @ A)^T, contracting over M.
        # Match B = W.T so the outer transpose backward exposes contiguous dW.
        dB = torch.bmm(dual_axis_mx_Y.to_mx_tensor(quant_axis=-2).transpose(-1, -2), mx_A).transpose(-1, -2)

        return dA, dB, None, None


def to_block_mx_then_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    config_A: MXQuantizeConfig,
    config_B: BlockMXQuantizeConfig,
) -> torch.Tensor:
    """Block MX batched matmul: ``A[B,M,K] @ B[B,K,N] = Y[B,M,N]``.

    Both operands are 3D and must share the leading batch dimension.

    - ``A`` may already be a ``DualAxisMXTensor`` or an ``MXTensor`` quantized along
      its last dim, validated against ``config_A``.
    - ``B`` may already be a ``BlockMXTensor`` or a pre-quantized ``MXTensor``,
      validated against ``config_B``.

    High-precision operands are quantized before the matmul.

    See ``_BlockMXQuantBMM`` for details of the three GEMMs.

    Args:
        A: Shape ``(B, M, K)``.
        B: Shape ``(B, K, N)``.
        config_A: Config for A's quantization.
        config_B: Config for B's quantization.

    Returns:
        Output tensor, shape ``(B, M, N)``, dtype matching ``A``.
    """
    return _BlockMXQuantBMM.apply(A, B, config_A, config_B)
