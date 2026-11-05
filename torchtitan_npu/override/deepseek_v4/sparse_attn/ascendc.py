# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused AscendC TND kernels.

This module holds the AscendC backend: the AI_CPU metadata tier the fused path consumes
(``AscV4Metadata`` + the ``*_metadata`` fills, built by
``AscCompressedSparseInnerAttention.Config.build_attention_metadata``, upstream
#4897's seam) and the fused TND kernel (``AscCompressedSparseInnerAttention``).
The ``sparse_attn`` package ``__init__`` defines the registration, so the
override path stays ``override.deepseek_v4.sparse_attn.asc``.

The fused path only bridges the AscendC TND kernels' forward/backward with a
slim ``torch.autograd.Function``.  The indexer is the separate
``lightning_indexer.asc`` override: this bridge consumes its ``topk_indices``,
and SMLAG's backward emits the teacher on the indexer's ``topk_scores`` edge.
All AscendC AI_CPU metadata is precomputed at the backend's own metadata boundary and
carried in the registered ``AscV4Metadata``, so this module never builds or
caches metadata per layer; because they describe *frames*, the CP
transform rebuilds them for this rank (``prepare_cp_metadata``).  The CP path's
``S(1) -> R`` sharding gathers the KV streams to the whole row at the wrapper
boundary, and the frames' ``seqused`` limits each rank's view.  Every fused
kernel of arch22, however, anchors its mask at the *end* of the key frame it is
given (it reads no ``seqused`` there), so on that SoC the options state the reach
itself as the frame and the wrapper compacts both gathered rows onto it before the
kernel call -- the entry's ``a3_workaround``, built once per microbatch beside the
AI_CPU metadata (``override/a3_workaround.py``).  The kernels therefore see the reach as
their key frame's end, while the picks and indices keep the model's
document-local numbering.  There is no container or ``_assemble_tnd`` step.

The kernel inputs follow the training autocast contract: q/k/v are bf16 and
the sink is fp32 — an eager-fp32 caller fails loudly at the kernel's dtype
check.
"""

from dataclasses import dataclass

import cann_ops_transformer  # noqa: F401  (registers the torch.ops below)
import torch

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata, KernelFrame
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.asc_metadata import AicpuGeometry, AscV4Metadata

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


def kernel_options(
    *,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    ratio: int,
    window_size: int,
):
    """The options the SMLA/SMLAG kernels and their metadata builders share.

    ``swa_k`` / ``cmp_k`` are the frames the kernels are addressed with: the entry's own, which on
    arch22 are the per-document reach this rank may read (``a3_workaround``).
    """
    has_cmp = cmp_k is not None
    return dict(
        cu_seqlens_q=q.cu_seqlens,
        cu_seqlens_ori_kv=swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if cmp_k is None else cmp_k.cu_seqlens,
        seqused_q=q.seqused,
        seqused_ori_kv=swa_k.seqused,
        seqused_cmp_kv=None if cmp_k is None else cmp_k.seqused,
        cmp_residual_kv=None if cmp_k is None else cmp_k.residual,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=_ORI_MASK_MODE,
        cmp_mask_mode=0 if _IS_A5 and not has_cmp else _CMP_MASK_MODE,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q=_LAYOUT,
        layout_kv=_LAYOUT,
    )


def kernel_aicpu_metadata(
    *,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    ratio: int,
    geometry: AicpuGeometry,
    has_indexer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SMLA/SMLAG AI_CPU metadata for these frames."""
    options = kernel_options(q=q, swa_k=swa_k, cmp_k=cmp_k, ratio=ratio, window_size=geometry.window_size)
    has_cmp_kv = cmp_k is not None
    geometry_kwargs = dict(
        ori_topk=0,
        # Only a geometry with an indexer selects compressed entries.  A ratio-128 HCA layer
        # addresses the whole causal prefix, and the kernel rejects a non-zero ``cmp_topk``
        # with a ratio above 4 (the error the A3 integration run surfaced).
        cmp_topk=geometry.index_topk if has_indexer else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp_kv,
    )
    smla_metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        ori_topk_length=None,
        cmp_topk_length=None,
        **options,
        **geometry_kwargs,
    )
    smla_grad_metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        **options,
        **geometry_kwargs,
    )
    return smla_metadata, smla_grad_metadata


class _SparseMLA(torch.autograd.Function):
    """SMLA forward with SMLAG backward, in the TND layout of one packed row.

    ``cmp_k`` is the pooled container (a ratio above one, addressed by ``topk_indices``) or the
    full-resolution second stream (ratio one).  ``topk_scores`` is an input only so the teacher
    SMLAG emits on its gradient has a consumer: the indexer's backward reads that edge and it is
    never read here.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        sinks,
        attention_metadata,
        softmax_scale,
        ratio,
        window_size,
        topk_scores,
    ):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("the fused core needs AscV4Metadata")
        options = kernel_options(
            q=attention_metadata.q,
            swa_k=attention_metadata.swa_k,
            cmp_k=attention_metadata.cmp_k,
            ratio=ratio,
            window_size=window_size,
        )
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=topk_indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=attention_metadata.aicpu_metadata.sparse_attn.fwd,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(q, swa_k, cmp_k, topk_indices, sinks, output, lse)
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        ctx.has_carrier = topk_scores is not None
        # A dataclass cannot go through ``save_for_backward``; it holds only the frames the graph
        # already keeps, and is released when this backward has run.
        ctx.attention_metadata = attention_metadata
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse
        q, swa_k, cmp_k, topk_indices, sinks, output, lse = ctx.saved_tensors
        options = kernel_options(
            q=ctx.attention_metadata.q,
            swa_k=ctx.attention_metadata.swa_k,
            cmp_k=ctx.attention_metadata.cmp_k,
            ratio=ctx.ratio,
            window_size=ctx.window_size,
        )
        dq, dswa_k, dcmp_k, dsinks, _, teacher = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=topk_indices,
            sinks=sinks,
            metadata=ctx.attention_metadata.aicpu_metadata.sparse_attn.bwd,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.softmax_scale,
            **options,
        )
        # One gradient per forward input, in order.  ``cmp_k`` and the carrier are answered only
        # where the forward carried them.
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            teacher if ctx.has_carrier else None,
        )


def _compact_key_rows(swa_k, cmp_k, metadata):
    """arch22 reads only this rank's reach, so both key streams are compacted onto it.

    The gathers are differentiable, so the kernels' gradients scatter back onto the gathered
    rows; on the rank whose reach is every document in full they are the identity.
    """
    if _IS_A5:
        return swa_k, cmp_k
    workaround = metadata.a3_workaround
    assert workaround is not None, "an arch22 entry states the key workaround"
    if workaround.swa_k_gather is not None:
        swa_k = swa_k.index_select(0, workaround.swa_k_gather)
    if cmp_k is not None and workaround.cmp_k_gather is not None:
        cmp_k = cmp_k.index_select(0, workaround.cmp_k_gather)
    return swa_k, cmp_k


class AscSparseMLAMixin:
    """The kernel-facing half both stacks' fused cores share.

    The reference cores differ (V4's single indexer against V4.1's indexer modes and kv
    sources), so the ports share behaviour here rather than a base class: the row/TND
    translation, the arch22 compaction and the Function call.
    """

    training: bool
    softmax_scale: float
    compress_ratio: int
    window_size: int

    def __init__(self, config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-count]
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim

    def forward(  # pyrefly: ignore [bad-override]
        self,
        q,
        swa_k,
        attn_sink: torch.Tensor,
        attention_metadata,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
    ) -> torch.Tensor:
        batch_size, seqlen = q.shape[0], q.shape[1]
        swa_k, cmp_k = _compact_key_rows(
            swa_k.flatten(0, 1),
            None if cmp_k is None else cmp_k.flatten(0, 1),
            attention_metadata,
        )
        swa_k = swa_k.unsqueeze(1).contiguous()
        if cmp_k is not None:
            cmp_k = cmp_k.unsqueeze(1).contiguous()
        indices = None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1])
        # The carrier is a training-only by-product: its gradient is the indexer's only edge.
        carrier = None
        if self.training and topk_scores is not None:
            carrier = topk_scores.reshape(-1, 1, topk_scores.shape[-1])
        output, _lse = _SparseMLA.apply(
            q.flatten(0, 1),
            swa_k,
            cmp_k,
            indices,
            attn_sink.float(),
            attention_metadata,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            carrier,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])


class AscCompressedSparseInnerAttention(AscSparseMLAMixin, CompressedSparseInnerAttention):
    """Run SMLA/SMLAG in a local TND layout, consuming the indexer's selection.

    The core reads this geometry's frames and calls the AscendC ops directly; the indexer is the
    separate ``lightning_indexer.asc`` override, which emits ``topk_indices`` and the teacher
    carrier that this core's backward writes onto.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        """The fused config: the reference core's fields plus the geometry its AI_CPU metadata need."""

        num_heads: int  # pyrefly: ignore [bad-override]
        head_dim: int
        """The main attention geometry the SMLA/SMLAG AI_CPU metadata are sized with."""

        index_num_heads: int
        index_head_dim: int
        """The indexer geometry its metadata and SLIKG backward are sized with."""

        @property
        def geometry(self) -> AicpuGeometry:
            """The numbers the AI_CPU operators are called with, read off this layer's config."""
            return AicpuGeometry(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                window_size=self.window_size,
                index_topk=self.index_topk,
                index_num_heads=self.index_num_heads,
                index_head_dim=self.index_head_dim,
            )

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> AscV4Metadata:
            """This backend's per-microbatch metadata (upstream #4897).

            The model-dir plain row plus the AI_CPU metadata tier: this geometry's four ``*_metadata``
            kernel outputs, computed here once per microbatch for every layer of the geometry.
            """
            del padding_mask, max_num_documents, max_context_length
            metadata = DeepSeekV4Metadata.from_positions(positions, self.compress_ratio)
            return AscV4Metadata.build_aicpu_metadata(metadata, geometry=self.geometry)
