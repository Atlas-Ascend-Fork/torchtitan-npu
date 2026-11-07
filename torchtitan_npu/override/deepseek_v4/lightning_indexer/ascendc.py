# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 indexer's BF16 port: InnerIndexer fused with SLIKG.

The forward runs the CANN ``lightning_indexer`` selection and returns the document-local
indices plus a ``topk_scores`` carrier.  Nothing reads the carrier's value: it exists so
autograd has an output to carry the teacher that SMLAG's backward produces
(``sparse_flash_mla_grad``'s ``cmp_softmax_l1_norm``).  This Function's backward consumes
that edge with ``sparse_lightning_indexer_kl_loss_grad`` and emits ``dq/dk/dw``.

The metadata is the entry the fused attention already built per geometry (``AscV4Metadata``):
the indexer's two AI_CPU metadata (``lightning_indexer`` and SLIKG) live beside the attention's,
because the teacher edge crosses both kernels' backwards.

This module owns the whole AscendC indexer -- the autograd bridge, the input preparation,
the canonical sort, the teacher carrier and the backward -- so its kernel pair is stated once,
here.  The PyPTO port is a separate implementation of the same contract in
``lightning_indexer.pypto``; the two do not share code, so each can follow its runtime's
calling conventions without a common abstraction between them.
"""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
from torchtitan_npu.models.deepseek_v4.metadata import KernelFrame
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround
from torchtitan_npu.override.deepseek_v4.asc_metadata import AicpuGeometry, AscV4Metadata


def kernel_options(*, ratio: int, q: KernelFrame, cmp_k: KernelFrame | None) -> dict:
    """The options the indexer's metadata calls and its kernels must agree on.

    They are read off this geometry's frames, so the metadata call, the InnerIndexer
    forward and the SLIKG backward cannot drift option-for-option.  ``cmp_k`` is the entry's frame
    -- on arch22 the pool this rank's reach covers, which its keys are compacted onto.
    """
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_k=None if cmp_k is None else cmp_k.cu_seqlens,
        seqused_q=q.seqused,
        seqused_k=None if cmp_k is None else cmp_k.seqused,
        # A3's kernels take the zero-filled empty group; A5 rejects a value at ratio 1.
        cmp_residual_k=(
            None
            if cmp_k is None
            else cmp_k.residual
            if cmp_k.residual is not None or _IS_A5
            else torch.zeros(cmp_k.seqused.shape, dtype=torch.int32, device=cmp_k.seqused.device)
        ),
        layout_q="TND",
        layout_k="TND",
        mask_mode=3,
        cmp_ratio=ratio,
    )


def kernel_aicpu_metadata(
    *,
    q: KernelFrame,
    cmp_k: KernelFrame | None,
    ratio: int,
    geometry: AicpuGeometry,
    has_indexer: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The indexer's forward metadata and its SLIG backward metadata, for one geometry.

    arch22's (A3) ``lightning_indexer`` schedules itself and is launched with ``metadata=None``,
    so its forward metadata is not built there; SLIKG's backward metadata is required on both SoCs.
    """
    if not has_indexer or cmp_k is None:
        return None, None
    frames = kernel_options(ratio=ratio, q=q, cmp_k=cmp_k)
    # ``max_seqlen_q``/``max_seqlen_k`` are pinned to ``None`` because both operators take them as
    # optional ints; passing them makes the call independent of whatever default the operator ships.
    geometry_kwargs = dict(
        num_heads_q=geometry.index_num_heads,
        num_heads_k=1,
        head_dim=geometry.index_head_dim,
        topk=geometry.index_topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
    )
    li_record = (
        torch.ops.cann_ops_transformer.lightning_indexer_metadata(**geometry_kwargs, **frames) if _IS_A5 else None
    )
    slig_metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
        **geometry_kwargs, **frames
    )
    return li_record, slig_metadata


class _InnerIndexerTND(torch.autograd.Function):
    """InnerIndexer forward fused with its SLIKG backward, over the CANN kernel pair.

    The kernels address the packed TND row: ``idx_q`` is ``[T, Hi, Di]``, ``idx_k`` is
    ``[Tk, 1, Di]`` and ``idx_w`` is ``[T, Hi]`` float32, with document-local ``-1`` padding.
    ``topk_scores`` is an input only so the teacher SMLAG emits on its gradient has a consumer.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_metadata,
        num_global_queries,
    ):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("the fused indexer needs AscV4Metadata")
        options = kernel_options(ratio=ratio, q=attention_metadata.q, cmp_k=attention_metadata.cmp_k)
        topk_indices, _ = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=attention_metadata.aicpu_metadata.lightning_indexer.fwd,
            return_value=1,
            **options,
        )
        # The kernel's order is unspecified, so the selection is canonicalised here, before
        # anything is saved: the teacher arrives on ``topk_scores``' gradient positioned per slot
        # by the selection SMLAG was handed, and SLIKG consumes it next to these indices.
        # Descending gives both properties with one sort -- valid entries keep position order and
        # ``-1`` lands behind every one of them.
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        # Shape, dtype and the autograd edge are the carrier's whole contract: SMLAG replaces the
        # value, SMLA accepts and ignores it.
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices, num_global_queries)
        ctx.ratio = ratio
        # A dataclass cannot go through ``save_for_backward``; it holds only the frames the graph
        # already keeps, and is released when this backward has run.
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(topk_indices)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        del grad_indices
        if attn_softmax_l1_norm is None:
            return (None,) * 7
        idx_q, idx_k, idx_w, topk_indices, num_global_queries = ctx.saved_tensors
        # The teacher is normalized once, by the row's query count: under CP every rank owns a
        # slab of the row and the parameter gradients are summed across the CP axis, so a local
        # denominator would scale every indexer parameter's gradient by the CP degree.
        attn_softmax_l1_norm = attn_softmax_l1_norm / num_global_queries
        metadata = ctx.attention_metadata
        slig_metadata = metadata.aicpu_metadata.lightning_indexer.bwd
        if slig_metadata is None:
            raise ValueError("the fused indexer needs the entry's backward metadata")
        options = kernel_options(ratio=ctx.ratio, q=metadata.q, cmp_k=metadata.cmp_k)
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **options,
        )
        return dq.to(idx_q.dtype), dk.to(idx_k.dtype), dw.to(idx_w.dtype), None, None, None, None


class AscIndexerMixin:
    """The fused LI/SLIKG port both stacks share, in V4.1's shape.

    The caller may hand candidate block indices and receives a third tuple slot back; V4 states
    neither (its reference indexer has no candidate stage), it passes no candidates and drops the
    slot.  The denominator the KL grad divides by is the microbatch's global query count; a stack
    that carries one on the indexer states it, and otherwise the entry's row supplies it.
    """

    def _num_global_queries(self, attention_metadata):
        configured = getattr(self, "num_global_queries", None)
        if configured is None:
            configured = attention_metadata.swa_k.cu_seqlens[-1].to(torch.float32)
        return configured

    # The parameter names are a contract: each stack's sharding config states this node's
    # redistribution by name (in_src_shardings/in_dst_shardings with "idx_q"/"idx_k"/"idx_w"),
    # and the module binds those placements by parameter name. Renaming them silently drops the
    # row-wide gather of idx_k, which leaves the selector with this rank's slab while
    # ``A3KeyWorkaround``'s gather still addresses the whole row.
    index_topk: int

    def _select(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        attention_metadata: Any,
        *,
        candidates_BL1C: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        del candidates_BL1C
        topk = self.index_topk
        batch, seqlen = idx_q.shape[:2]
        topk_indices, topk_scores = _InnerIndexerTND.apply(
            idx_q.flatten(0, 1),
            # The keys enter the Function already compacted (arch22): the gather is
            # differentiable, so the teacher's gradient scatters back onto the gathered row and
            # the wrapper's ``S(1) -> R`` reduce-scatter lands it on the compressor.
            A3KeyWorkaround.indexer_keys(idx_k.flatten(0, 1), attention_metadata, a5=_IS_A5).unsqueeze(1).contiguous(),
            idx_w.flatten(0, 1).float(),
            topk,
            attention_metadata.ratio,
            attention_metadata,
            self._num_global_queries(attention_metadata),
        )
        return (
            topk_indices.reshape(batch, seqlen, topk),
            topk_scores.reshape(batch, seqlen, topk),
            None,
        )


class AscInnerIndexer(AscIndexerMixin, InnerIndexer):
    """The fused BF16 LI/SLIKG indexer for the V4 indexer layers.

    ``topk_indices`` crosses the model boundary in the kernels' own document-local numbering,
    with ``-1`` for an unused slot, so neither side translates.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerIndexer.Config):
        pass

    def forward(  # pyrefly: ignore [bad-override]
        self,
        idx_q_BLHiDi,
        idx_k_BNDi,
        idx_w_BLHi,
        attention_metadata,
        *,
        candidates_BL1C=None,
    ):
        """V4's node states indices and scores: the third slot is V4.1's candidate pool."""
        topk_indices, topk_scores, _candidates = self._select(
            idx_q_BLHiDi, idx_k_BNDi, idx_w_BLHi, attention_metadata, candidates_BL1C=candidates_BL1C
        )
        return topk_indices, topk_scores
