# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/5543

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Clamping and pre-W2 score absorption for common MoE.

The pinned router owns score activation/normalization and the native MoE owns
flat-token dispatch, expert counters and shared-expert composition.

Only the pre-W2 score path is adapted from PR 5543; its rematerialization
changes are not included. Grouped/shared experts retain the local DeepSeek-V4
SwiGLU clamp and BF16 behavior. Routed experts
consume dispatcher-carried router scores before W2, as required by the local
BF16 rounding contract. Config factories wire those optional extensions.

``config_utils`` is imported inside ``apply()``, after the class swaps, so its
module-level imports bind the patched classes. Keep that activation order.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import (
    GroupedExperts,
    RoutedExperts,
)


class _RouterScoreAbsorbingRoutedExperts(RoutedExperts):
    """RoutedExperts that absorbs dispatcher-aligned router scores before w2."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        dispatcher = self.token_dispatcher
        routed_input_RD, num_global_tokens_per_local_expert_e, metadata = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD,
                num_global_tokens_per_local_expert_e,
                routed_scores_R=getattr(metadata, "routed_scores_R", None),
            )
        return dispatcher.combine(routed_output_RD, metadata, x_TD)


class _ClampGroupedExperts(GroupedExperts):
    """GroupedExperts with the optional DeepSeek-V4 SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        # SwiGLU limit on the gate/up activations; 0 disables the clamp.
        # DeepSeek-V4 uses 10.0 (transformers config default).
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw expert computation; the gate/up grouped-mms are clamped
        before the SiLU when ``swiglu_limit > 0``."""
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        g_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            weight_EOI=w1_EFD.bfloat16(),
            offs=offsets_E,
        )
        u_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            weight_EOI=w3_EFD.bfloat16(),
            offs=offsets_E,
        )
        if self.swiglu_limit > 0:
            u_RF = torch.clamp(u_RF, min=-self.swiglu_limit, max=self.swiglu_limit)
            g_RF = torch.clamp(g_RF, max=self.swiglu_limit)
        h_RF = F.silu(g_RF) * u_RF
        if routed_scores_R is not None:
            h_RF = (h_RF.float() * routed_scores_R.float().reshape(-1, 1)).to(h_RF.dtype)
        return self._grouped_mm(
            A=h_RF,
            weight_EOI=w2_EDF.bfloat16(),
            offs=offsets_E,
        ).type_as(x_RD)


class _ClampFeedForward(FeedForward):
    """FeedForward (shared expert) with the optional SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.w2(F.silu(gate) * up)


def _as_config(config, config_type, **overrides):
    """Rebuild a config while retaining fields shared across upstream schemas."""
    field_names = {config_field.name for config_field in dataclasses.fields(config_type)}
    values = {name: getattr(config, name) for name in field_names if hasattr(config, name)}
    values.update(overrides)
    return config_type(**values)


def _clamp_make_routed_experts_config(
    *,
    swiglu_limit: float = 0.0,
    absorb_router_scores: bool = True,
    **kwargs,
):
    """``make_routed_experts_config`` with the clamp passthrough."""
    cfg = _original_make_routed_experts_config(**kwargs)
    dispatcher_config = cfg.token_dispatcher
    # Only the dispatcher configs extended by the NPU patch understand this
    # option.  HybridEP and MinimalAsyncEP intentionally retain upstream
    # schemas, so passing the field to them would fail during config
    # construction before a model can report which backend is unsupported.
    dispatcher_fields = {field.name for field in dataclasses.fields(dispatcher_config)}
    if "absorb_router_scores" in dispatcher_fields:
        dispatcher_config = dataclasses.replace(
            dispatcher_config,
            absorb_router_scores=absorb_router_scores,
        )
    inner_experts = cfg.inner_experts
    if isinstance(inner_experts, _ClampGroupedExperts.Config):
        inner_experts = dataclasses.replace(inner_experts, swiglu_limit=swiglu_limit)
    else:
        inner_experts = _as_config(inner_experts, _ClampGroupedExperts.Config, swiglu_limit=swiglu_limit)
    return dataclasses.replace(cfg, inner_experts=inner_experts, token_dispatcher=dispatcher_config)


def _clamp_make_ffn_config(*, swiglu_limit: float = 0.0, **kwargs):
    """``make_ffn_config`` with the clamp passthrough."""
    cfg = _original_make_ffn_config(**kwargs)
    if isinstance(cfg, _ClampFeedForward.Config):
        return dataclasses.replace(cfg, swiglu_limit=swiglu_limit)
    return _as_config(cfg, _ClampFeedForward.Config, swiglu_limit=swiglu_limit)


# Assigned in ``apply()`` before the clamp factories are ever called; declared
# at module scope so type checkers see the names.
_original_make_routed_experts_config: Callable[..., Any] = lambda **kwargs: None
_original_make_ffn_config: Callable[..., Any] = lambda **kwargs: None


# Swap the originals so ``torchtitan.models.common.moe`` resolves to the
# extended classes (see the module docstring).  Modules imported after this
# patch binds the new classes; the default 0.0 clamp keeps upstream behavior.


def apply() -> None:
    import torchtitan.models.common.feed_forward
    import torchtitan.models.common.moe

    # Swap the classes BEFORE ``config_utils`` is imported: its module-level
    # ``from torchtitan.models.common.moe import GroupedExperts`` (and
    # ``from ...feed_forward import FeedForward``) then binds the patched
    # classes, so the factories build the clamped configs.  Importing it
    # earlier would freeze the originals into its namespace and the
    # factories would keep building unpatched configs.
    torchtitan.models.common.moe.GroupedExperts = _ClampGroupedExperts
    torchtitan.models.common.moe.RoutedExperts = _RouterScoreAbsorbingRoutedExperts
    torchtitan.models.common.feed_forward.FeedForward = _ClampFeedForward

    import torchtitan.models.common.config_utils

    torchtitan.models.common.config_utils.GroupedExperts = _ClampGroupedExperts
    torchtitan.models.common.config_utils.RoutedExperts = _RouterScoreAbsorbingRoutedExperts

    global _original_make_routed_experts_config, _original_make_ffn_config
    _original_make_routed_experts_config = torchtitan.models.common.config_utils.make_routed_experts_config
    _original_make_ffn_config = torchtitan.models.common.config_utils.make_ffn_config
    torchtitan.models.common.config_utils.make_routed_experts_config = _clamp_make_routed_experts_config
    torchtitan.models.common.config_utils.make_ffn_config = _clamp_make_ffn_config


apply()
