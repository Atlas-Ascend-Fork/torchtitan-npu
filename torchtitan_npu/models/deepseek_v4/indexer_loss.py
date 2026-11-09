# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distilling the sparse indexer from the attention it selects for.

With ``p`` the attention's head-averaged probability at the selected slots and ``y = softmax(I)``
the indexer's own distribution over them, one query row's objective is

    sum_j p_j * (log t_j - log y_j),   t = p / Z,   Z = sum_j p_j

i.e. ``Z * D_KL(t || y)``: the row's KL weighted by the mass those slots actually hold. ``Z`` is
below one whenever the window or the sink takes part of the row, so ``p`` is never renormalized.
The teacher comes from the attention's own log-sum-exp, which is why the augmentation is requested
where the attention is computed.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan_npu.models.deepseek_v4.document_intervals import document_intervals
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

# An unused slot in *logit* space.  ``-inf`` would make ``log_softmax`` of an all-padding row NaN
# on the backward; a finite sentinel far below every real logit gives the same distribution.
_MASKED_LOGIT = -1e30


class IndexerLoss(AuxLoss):
    """The selection's distillation objective: ``coeff`` scales the gradient only."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        # Additive over query rows, so ranks contribute distinct data: reduce over dp+cp.
        reduce_mesh: str = "loss"

    def forward(  # pyrefly: ignore [bad-override]
        self,
        q_THD: torch.Tensor,
        cmp_k_SD: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        attn_lse_TH: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        softmax_scale: float,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Score the microbatch and inject the objective on the attention's output.

        The teacher side is constant (the trunk must not receive this gradient); only the student
        logits stay live.
        """
        p_TK = self._teacher(
            q_THD,
            cmp_k_SD,
            topk_indices_TK,
            attn_lse_TH,
            cu_seqlens,
            cu_seqlens_k,
            softmax_scale=softmax_scale,
        )
        return self.inject(self._logged_kl(p_TK, topk_scores_TK), carrier=carrier)

    @torch.no_grad()
    def _teacher(
        self,
        q_THD: torch.Tensor,
        cmp_k_SD: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        attn_lse_TH: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> torch.Tensor:
        """The attention's mass at the selected slots, head-averaged, ``[T, K]`` fp32.

        ``exp(logit - lse)`` is the row's probability, not a renormalized approximation: the
        log-sum-exp already covers window, selection and sink.  Picks are document-local here and
        the pooled keys packed-row indexed, so their numbering is bridged at this one point.
        """
        pool_start_T1 = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=q_THD.size(0))[0]
        # A ``-1`` slot may name a real entry once shifted; the gathered row is masked out below.
        rows_TK = torch.where(topk_indices_TK >= 0, topk_indices_TK + pool_start_T1, 0)
        keys_TKD = cmp_k_SD[rows_TK].float()
        logits_THK = torch.einsum("thd,tkd->thk", q_THD.float(), keys_TKD)
        logits_THK = logits_THK * softmax_scale - attn_lse_TH[..., None].float()
        return logits_THK.exp().mean(dim=1).masked_fill(topk_indices_TK < 0, 0.0)

    def _logged_kl(self, p_TK: torch.Tensor, logits_TK: torch.Tensor) -> torch.Tensor:
        """``sum_j p_j * (log t_j - log y_j)`` over the caller's slots; an unused slot has ``p = 0``."""
        # The student logits arrive in the model's dtype (the reference selector's einsum is
        # bf16); the distribution and its gradient are scored in fp32, the space the teacher is
        # computed in.
        masked_TK = logits_TK.float().masked_fill(~torch.isfinite(logits_TK), _MASKED_LOGIT)
        log_y_TK = F.log_softmax(masked_TK, dim=-1)
        # An all-padding row has ``Z == 0``: ``t`` stays zero and ``xlogy(0, 0) == 0`` avoids NaN.
        z_T1 = p_TK.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        t_TK = p_TK / z_T1
        return (torch.special.xlogy(p_TK, t_TK) - p_TK * log_y_TK).sum()
