# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 token exchange for TorchTitan's existing AllToAll experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchao_npu.configs import ParamSwapConfig
from torchao_npu.ops.mxfp8_all_to_all import (
    MXFP8QuantParams,
    _exchange_high_precision,
    _exchange_payload,
    _index_mxfp8_rows,
    _mxfp8_wire_tensor,
    mxfp8_all_to_all_quantized,
)
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.mx_tensor import MXTensor, _mxfp8_dequantize_tensor
from torchtitan_npu.ops.ascendc.moe_re_routing import npu_moe_re_routing
from torchtitan_npu.ops.ascendc.moe_token_unpermute import npu_moe_token_unpermute
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def mxfp8_quant_params_from_experts(experts_config: object) -> MXFP8QuantParams:
    """Use the experts' activation quantizer as the communication wire format."""
    param_swap = getattr(experts_config, "_torchao_npu_config", None)
    activation_config = param_swap.activation_config if isinstance(param_swap, ParamSwapConfig) else None
    if not isinstance(activation_config, MXQuantizeConfig) or activation_config.elem_dtype is not torch.float8_e4m3fn:
        raise ValueError("MXFP8 all-to-all requires an E4M3 MX activation config on the experts")
    if activation_config.scale_alg is None:
        raise ValueError("MXFP8 all-to-all requires an explicit activation scale_alg")
    return (
        activation_config.block_size,
        activation_config.round_mode,
        activation_config.scale_alg,
        activation_config.dst_type_max,
    )


class _MXFP8ReRouting(torch.autograd.Function):
    """Reroute byte-identical MXFP8 payloads; reverse the row map for BF16 dA."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        activation: MXTensor,
        expert_token_num_per_rank: torch.Tensor,
        scores: torch.Tensor | None,
        quant_params: MXFP8QuantParams,
    ):
        if activation.quant_axis != activation.ndim - 1 or activation.qdata.dtype is not torch.float8_e4m3fn:
            raise ValueError("MXFP8 expert rerouting requires last-axis float8_e4m3fn activations")

        # The CANN kernel accepts int8 rather than FP8. A bitcast preserves every
        # quantized value and lets the kernel produce the expert-major row map.
        qdata, routed_scores, restore_indices, expert_counts = npu_moe_re_routing(
            activation.qdata.view(torch.int8), expert_token_num_per_rank, per_token_scales=scores
        )
        # The NPU wrapper returns the inverse of the kernel's gather order.
        # Recover that forward order so each scale follows its qdata row.
        token_order = torch.argsort(restore_indices.to(torch.float32))
        scale = activation.scale[token_order]
        routed = _mxfp8_wire_tensor(  # pyrefly: ignore [not-callable]
            qdata.view(activation.qdata.dtype),
            scale,
            activation.dtype,
            quant_params,
            activation.requires_grad,
        )
        ctx.save_for_backward(restore_indices)
        ctx.has_scores = scores is not None
        ctx.mark_non_differentiable(restore_indices, expert_counts)
        if scores is None:
            ctx.mark_non_differentiable(routed_scores)
        return routed, routed_scores, restore_indices, expert_counts

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_routed, grad_scores, _grad_restore_indices, _grad_expert_counts
    ):
        (restore_indices,) = ctx.saved_tensors
        if isinstance(grad_routed, MXTensor):
            # The inverse NPU permutation consumes logical BF16 gradients.
            grad_routed = _mxfp8_dequantize_tensor(grad_routed)
        grad_activation = (
            npu_moe_token_unpermute(grad_routed, restore_indices, None) if grad_routed is not None else None
        )
        grad_input_scores = None
        if ctx.has_scores and grad_scores is not None:
            grad_input_scores = npu_moe_token_unpermute(grad_scores.reshape(-1, 1), restore_indices, None).reshape(-1)
        return grad_activation, None, grad_input_scores, None


class _MXFP8CombineExchange(torch.autograd.Function):
    """Use BF16 for forward combine and preserve MXFP8 in its inverse VJP."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        routed_output_expert_major: torch.Tensor,
        restore_indices: torch.Tensor,
        group,
        combine_output_splits: list[int],
        combine_input_splits: list[int],
        quant_params: MXFP8QuantParams,
    ) -> torch.Tensor:
        routed_output_rank_major = npu_moe_token_unpermute(
            routed_output_expert_major,
            restore_indices,
            probs=None,
        )
        ctx.save_for_backward(restore_indices)
        ctx.group = group
        ctx.combine_output_splits = combine_output_splits
        ctx.combine_input_splits = combine_input_splits
        ctx.quant_params = quant_params
        return _exchange_high_precision(
            routed_output_rank_major,
            combine_output_splits,
            combine_input_splits,
            group,
        )

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        (restore_indices,) = ctx.saved_tensors
        qdata, scale = _exchange_payload(
            grad_output,
            ctx.combine_input_splits,
            ctx.combine_output_splits,
            ctx.group,
            ctx.quant_params,
        )

        # The inverse collective returns rank-major rows. Apply the same
        # rank-major -> expert-major map used by dispatch to both wire tensors.
        # Forward unpermute gathers by restore_indices, so its VJP gathers by
        # the inverse permutation, not by restore_indices itself.
        token_order = torch.argsort(restore_indices.to(torch.float32))
        qdata = _index_mxfp8_rows(qdata, token_order)
        scale = scale[token_order]
        grad_input = _mxfp8_wire_tensor(  # pyrefly: ignore [not-callable]
            qdata,
            scale,
            grad_output.dtype,
            ctx.quant_params,
            grad_output.requires_grad,
        )
        return grad_input, None, None, None, None, None


_dispatcher_cache: dict[type, type] = {}


def get_mxfp8_alltoall_dispatcher_cls(parent_cls: type[AllToAllTokenDispatcher]) -> type[AllToAllTokenDispatcher]:
    """Extend the selected dispatcher, promoting plain standard configs to Asc.

    Plain standard configs need not enable the Asc override beforehand: the
    promotion supplies the dispatch/combine hooks this replacement overrides.
    Existing Asc subclasses keep their dispatch/combine implementation.
    """

    from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher

    if parent_cls in _dispatcher_cache.values():
        return parent_cls
    if parent_cls is AllToAllTokenDispatcher:
        parent_cls = AscAllToAllTokenDispatcher
    if not issubclass(parent_cls, AscAllToAllTokenDispatcher):
        raise TypeError("MXFP8 all-to-all requires an AscAllToAllTokenDispatcher subclass")
    if parent_cls in _dispatcher_cache:
        return _dispatcher_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class MXFP8AllToAllTokenDispatcher(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc, valid-type]
            mxfp8_quant_params: MXFP8QuantParams

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.mxfp8_quant_params = config.mxfp8_quant_params

        def _dispatch_activation_exchange(self, routed_input_ND, pg, output_splits, input_splits):
            return mxfp8_all_to_all_quantized(
                routed_input_ND, output_splits, input_splits, group=pg, quant_params=self.mxfp8_quant_params
            )

        def _reroute_activation(self, routed_input_RD, expert_token_num_per_rank, routed_scores_rank_major_R):
            return _MXFP8ReRouting.apply(
                routed_input_RD, expert_token_num_per_rank, routed_scores_rank_major_R, self.mxfp8_quant_params
            )

        def _combine_activation_exchange(
            self,
            routed_output_RD,
            restore_indices,
            pg,
            input_splits,
            output_splits,
        ):
            return _MXFP8CombineExchange.apply(
                routed_output_RD,
                restore_indices,
                pg,
                input_splits,
                output_splits,
                self.mxfp8_quant_params,
            )

    MXFP8AllToAllTokenDispatcher.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8AllToAllTokenDispatcher.__qualname__ = f"MXFP8{parent_cls.__qualname__}"
    _dispatcher_cache[parent_cls] = MXFP8AllToAllTokenDispatcher
    return MXFP8AllToAllTokenDispatcher


__all__ = ["get_mxfp8_alltoall_dispatcher_cls", "mxfp8_quant_params_from_experts"]
