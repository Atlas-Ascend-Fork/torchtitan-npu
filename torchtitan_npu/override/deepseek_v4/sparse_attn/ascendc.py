# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 DSA with fused AscendC TND kernels.

This module holds the AscendC backend: the record tier the fused path consumes
(``AscV4Metadata`` + the ``*_metadata`` fills, built by
``AscCompressedSparseInnerAttention.Config.build_attention_metadata``, upstream
#4897's seam) and the fused TND kernel (``AscCompressedSparseInnerAttention``).
The ``sparse_attn`` package ``__init__`` defines the registrations, so the
override paths stay ``override.deepseek_v4.sparse_attn.*``.

The fused path only bridges the AscendC TND kernels' forward/backward with a
slim ``torch.autograd.Function``.  All AscendC metadata is precomputed at the
backend's own metadata boundary and carried in the registered
``AscV4Metadata``, so this module never builds or caches metadata
per layer; because the records describe *frames*, the CP transform rebuilds
them for this rank (``prepare_cp_metadata``).  The container-grid compressed KV
is converted to the packed TND stream by slicing the leading ``n_blocks``
slots (the identity ``storage_indices`` mapping under the B=1 contract).

The kernel inputs follow the training autocast contract: q/k/v are bf16,
``idx_w`` and the sink are fp32 — an eager-fp32 caller fails loudly at the
kernel's dtype check.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer
from torchtitan_npu.models.deepseek_v4.metadata import (
    CompressedBlockLayout,
    DeepSeekV4Metadata,
)
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_CMP_RATIO as _LI_CMP_RATIO,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_LAYOUT as _LI_LAYOUT,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_MASK_MODE as _LI_MASK_MODE,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    AicpuGeometry,
    AscAicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata as AscV4Metadata,
)
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


@dataclass(frozen=True, slots=True)
class _SparseAttentionHooks:
    """Operator implementation bundle used by the fused attention bridge."""

    lightning_indexer: Callable[..., Any]
    sparse_flash_mla: Callable[..., Any]
    sparse_flash_mla_grad: Callable[..., Any]
    sparse_lightning_indexer_kl_loss_grad: Callable[..., Any]


_ASC_SPARSEATTN_HOOK = _SparseAttentionHooks(
    lightning_indexer=torch.ops.cann_ops_transformer.lightning_indexer,
    sparse_flash_mla=torch.ops.cann_ops_transformer.sparse_flash_mla,
    sparse_flash_mla_grad=torch.ops.cann_ops_transformer.sparse_flash_mla_grad,
    sparse_lightning_indexer_kl_loss_grad=torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad,
)


# ---------------------------------------------------------------------------
# AscendC metadata layer
# ---------------------------------------------------------------------------
#
# The AscendC backend's metadata is the model-dir packed contract plus its own
# record tier: after the document-packed layout is built, this backend fills
# the precomputed AscendC ``*_metadata`` kernel outputs (lightning_indexer,
# sparse_flash_mla, sparse_flash_mla_grad,
# sparse_lightning_indexer_kl_loss_grad) into an ``AscV4Metadata`` wrapper,
# keeping the model-dir metadata NPU-free.  Those kernels take only shape
# information, so their results are layer-invariant and reused by every DSA
# layer and by the backward pass.  ``AscCompressedSparseInnerAttention.Config``
# builds the whole thing (upstream #4897); because a record describes the frames
# it was derived from, ``prepare_cp_metadata`` rebuilds them per rank.


def shard_compressed_varlen_metadata(
    attention_metadata: DeepSeekV4Metadata,
) -> DeepSeekV4Metadata:
    """This rank's copy of one geometry's contract, derived from the pre-shard documents.

    The queries are sharded while the keys stay the gather of the row, so the rank-local plan is
    derived from the **global** documents, in frame and with no communication.  Each rank owns
    one contiguous slab of the row: this stack shards contiguous slabs, so a load balancer's
    permutation is refused rather than folded into the chunk grid.

    Only the fused entry's ``prepare_cp_metadata`` calls this -- the reference contract refuses
    context parallel.  The records are not part of this contract (the caller refills them over
    the returned entry).
    """
    context = cp_shard_context()
    cp_mesh, positions = context.cp_mesh, context.positions
    if cp_mesh is None or positions is None:
        raise ValueError("the context-parallel transform needs the CP mesh and the pre-shard positions")
    if attention_metadata.window_size is None:
        raise ValueError("the context-parallel transform needs the window size the plan was built with.")
    if positions.ndim != 2:
        raise ValueError(f"positions must be a [batch, seq_len] row, got shape {tuple(positions.shape)}.")
    if context.load_balancer_type is not None:
        raise ValueError(
            f"the DeepSeek-V4 fused stack shards contiguous slabs, but the context-parallel load "
            f"balancer is {context.load_balancer_type!r}: run with "
            "--parallelism.context-parallel-load-balancer None."
        )
    from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan

    seq_len = positions.shape[-1]
    cp_size = cp_mesh.size(0)
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size}).")
    cp_meta, plans, window = build_cp_plan(
        attention_metadata.varlen,
        rank=cp_mesh.get_local_rank(),
        cp_size=cp_size,
        shard_len=seq_len // cp_size,
        window_size=attention_metadata.window_size,
        ratios=[attention_metadata.ratio],
    )
    return DeepSeekV4Metadata(
        varlen=cp_meta,
        ratio=attention_metadata.ratio,
        plan=plans[attention_metadata.ratio],
        window=window,
        window_size=attention_metadata.window_size,
    )


def _reject_incomplete_blocks(ratio: int, plan: CompressedBlockLayout) -> None:
    """A ratio above 1 needs at least one complete block in the batch.

    Doc-packed sequences shorter than ``ratio`` tokens produce no compressed
    entry at all, and every consumer of the plan (the kernels' block tables, the
    dispatcher's gather, the compressor) would read an empty pool: refuse the
    batch here, before any AscendC op is called for it.  One entry is one
    geometry, so the check is this geometry's own.
    """
    if ratio > 1 and cast("torch.Tensor", plan.gather_indices).numel() == 0:
        raise ValueError(
            f"batch has no complete compression block for ratio={ratio}; "
            "doc-packed sequences must be long enough to produce at least "
            "one full block per sequence."
        )


def _selector_records(
    metadata: DeepSeekV4Metadata,
    geometry: AicpuGeometry,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The selector's forward record and its SLIG backward record, for one geometry.

    The geometry that pools at the selector's ratio is the only one that runs a selector, so
    every other entry leaves both fields empty.  The records are built from this entry's frames:
    one geometry's build serves every layer that reads them.
    """
    # The selector runs on the ratio-4 geometry: this is the stack's long-term convention
    # (the ratio-4 CSA layers are the ones that own an indexer); the V4.1-style indexer
    # refactor will make the source layer explicit instead of inferring it from the ratio.
    if metadata.ratio != _LI_CMP_RATIO:
        return None, None
    plan = metadata.plan
    # Refuse the batch here too: the selector record is the first thing a geometry's entry
    # fills, so this is where "no complete block" fails before any operator ran for it.
    _reject_incomplete_blocks(metadata.ratio, plan)
    frames = dict(
        cu_seqlens_q=metadata.varlen.cu_seq_q,
        cu_seqlens_k=plan.cu_seqlens_cmp_k,
        cmp_residual_k=plan.block_remainder,
        layout_q=_LI_LAYOUT,
        layout_k=_LI_LAYOUT,
        mask_mode=_LI_MASK_MODE,
        cmp_ratio=_LI_CMP_RATIO,
    )
    li_record = lightning_indexer_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        geometry.index_topk,
        **frames,
    )
    slig_metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
        geometry.index_num_heads,
        1,
        geometry.index_head_dim,
        topk=geometry.index_topk,
        **frames,
    )
    return li_record, slig_metadata


def _sparse_attention_records(
    metadata: DeepSeekV4Metadata,
    geometry: AicpuGeometry,
    cu_seqlens_ori_kv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SMLA/SMLAG records for these frames.

    ``num_heads`` / ``head_dim`` describe the main attention; the selector's widths are not
    part of these two kernels.  The indexer score sums over the head dimension, so tensor
    parallel is fixed at 1 for the DSA path.  Under context parallel ``cu_seqlens_ori_kv`` is
    the per-segment window-pack cumsum; otherwise the bare ``cu_seq_k``.
    """
    ratio = metadata.ratio
    has_cmp_kv = ratio > 1
    ori_cu = cu_seqlens_ori_kv if cu_seqlens_ori_kv is not None else metadata.varlen.cu_seq_k
    smla_kwargs = {
        "cu_seqlens_q": metadata.varlen.cu_seq_q,
        "cu_seqlens_ori_kv": ori_cu,
        "cu_seqlens_cmp_kv": metadata.plan.cu_seqlens_cmp_k if has_cmp_kv else None,
        "cmp_residual_kv": metadata.plan.block_remainder if has_cmp_kv else None,
        "ori_topk_length": None,
        "cmp_topk_length": None,
        "ori_topk": 0,
        "cmp_topk": geometry.index_topk if ratio == _LI_CMP_RATIO else 0,
        "cmp_ratio": ratio,
        "ori_mask_mode": _ORI_MASK_MODE,
        "cmp_mask_mode": 0 if _IS_A5 and not has_cmp_kv else _CMP_MASK_MODE,
        "ori_win_left": geometry.window_size - 1,
        "ori_win_right": 0,
        "layout_q": _LAYOUT,
        "layout_kv": _LAYOUT,
        "has_ori_kv": True,
        "has_cmp_kv": has_cmp_kv,
    }
    smla_metadata = sparse_flash_mla_metadata(geometry.num_heads, 1, geometry.head_dim, **smla_kwargs)
    grad_kwargs: dict[str, Any] = dict(smla_kwargs)
    grad_kwargs.pop("ori_topk_length")
    grad_kwargs.pop("cmp_topk_length")
    smla_grad_metadata = sparse_flash_mla_grad_metadata(geometry.num_heads, 1, geometry.head_dim, **grad_kwargs)
    return smla_metadata, smla_grad_metadata


def build_aicpu_metadata(
    metadata: DeepSeekV4Metadata,
    *,
    geometry: AicpuGeometry,
) -> AscV4Metadata:
    """The AscendC record tier over one geometry's packed contract (upstream #4897).

    The AI_CPU metadata of ``metadata.plan`` is computed from the frames of that
    object, which is why a re-cut frame — the CP transform's rank-local plan, a batch
    chunk's local documents — runs this again rather than reusing the records of the
    frame they came from.
    """
    ratio, plan = metadata.ratio, metadata.plan
    # The second of the two refusals on this path: the record builders check their own ratio,
    # this one covers the geometry the entry describes.
    _reject_incomplete_blocks(ratio, plan)
    window = metadata.window
    ori_cu = window.cu_seqlens_ori_kv if window is not None else None
    li_record, slig_metadata = _selector_records(metadata, geometry)
    smla_metadata, smla_grad_metadata = _sparse_attention_records(metadata, geometry, ori_cu)
    record = AscAicpuMetadata(
        geometry=geometry,
        sparse_attn=AscSparseAttentionMetadata(fwd=smla_metadata, bwd=smla_grad_metadata),
        lightning_indexer=AscLightningIndexerMetadata(fwd=li_record, bwd=slig_metadata),
    )
    result = AscV4Metadata(
        varlen=metadata.varlen,
        ratio=ratio,
        plan=plan,
        window=window,
        window_size=metadata.window_size,
        asc=record,
    )
    # The fused compressor reads this plan's state tables, and a table's rows and width follow
    # this batch's documents -- so it is filled here, at the eager metadata boundary, never
    # inside a traced layer.  A non-fused attention backend leaves the table to the
    # compressor's own `_state_inputs`, which sizes it per layer from the batch it runs on.
    from torchtitan_npu.override.deepseek_v4.compressor.ascendc import build_compressor_state_tables

    build_compressor_state_tables(result)
    _mark_dynamic(result)
    return result


def _mark_dynamic(metadata: AscV4Metadata) -> None:
    """Mark batch-dependent leading dimensions so Dynamo does not specialize.

    The entry carries only the kernel tensors; the model-dir reference tier is never materialized
    here.  Under context parallel the geometry's plan (``metadata.plan``) and the window plan
    (``metadata.window``) carry the dispatcher machinery too.
    """
    tensors: list[torch.Tensor] = [metadata.varlen.cu_seq_q]
    for name in (
        "cu_seqlens_cmp_k",
        "block_remainder",
        "gather_indices",
        "block_positions",
        "first_indices",
        "compressed_rows",
        "cmp_k_global_gather_indices",
    ):
        tensor = getattr(metadata.plan, name, None)
        if tensor is not None and tensor.numel() > 0:
            tensors.append(tensor)
    if metadata.plan.exchange is not None:
        tensors += [metadata.plan.exchange.send_indices, metadata.plan.exchange.recv_offsets]
    if metadata.asc is not None and metadata.asc.lightning_indexer.fwd is not None:
        # The selector's grid follows the batch's compressed block count, exactly as a plan
        # tensor does; the other records are opaque kernel layouts the tracer keeps whole.
        tensors.append(metadata.asc.lightning_indexer.fwd)
    if metadata.window is not None:
        window = metadata.window
        tensors += [
            window.exchange.send_indices,
            window.exchange.recv_offsets,
            window.gather_indices,
            window.cu_seqlens_ori_kv,
        ]
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)
        # Written via setattr: "_dynamo_unbacked_indices" is the dynamo shape
        # annotation the GraphTrainer tracer reads (torchtitan
        # experiments/graph_trainer/dynamic_shapes.py); the underscore name is
        # deliberate and must stay off the codecheck protected-member list.
        setattr(tensor, "_dynamo_unbacked_indices", {0})  # noqa: B010


class AscLightningIndexer(LightningIndexer):
    @dataclass(kw_only=True, slots=True)
    class Config(LightningIndexer.Config):
        pass

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError("AscLightningIndexer requires AscV4Metadata.")
        plan = attention_metadata.plan
        if plan.cu_seqlens_cmp_k is None:
            raise ValueError("AscLightningIndexer requires a ratio-4 plan.")
        flat_k = idx_k.flatten(0, 1)
        if plan.cmp_k_global_gather_indices is not None:
            idx_k_tnd = flat_k[plan.cmp_k_global_gather_indices].unsqueeze(1).contiguous()
        else:
            if plan.block_positions is None:
                raise ValueError("AscLightningIndexer requires plan.block_positions.")
            # The marked tensor dimension tracks the current batch on replay;
            # the host count would remain fixed at graph capture.
            idx_k_tnd = flat_k[: plan.block_positions.numel()].unsqueeze(1).contiguous()
        asc = attention_metadata.asc
        li_metadata = None if asc is None else asc.lightning_indexer.fwd
        if li_metadata is None:
            raise ValueError("AscLightningIndexer requires the entry's selector record.")
        sparse_indices, _ = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q.flatten(0, 1),
            idx_k_tnd,
            idx_w.flatten(0, 1).float(),
            self.index_topk,
            cu_seqlens_q=attention_metadata.varlen.cu_seq_q,
            cu_seqlens_k=plan.cu_seqlens_cmp_k,
            cmp_residual_k=plan.block_remainder,
            metadata=li_metadata,
            layout_q=_LI_LAYOUT,
            layout_k=_LI_LAYOUT,
            mask_mode=_LI_MASK_MODE,
            cmp_ratio=_LI_CMP_RATIO,
            return_value=1,
        )
        return sparse_indices.reshape(idx_q.shape[0], idx_q.shape[1], 1, -1)


def _compute_li_loss(
    indexer_softmax: torch.Tensor,
    teacher_mass: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Recover the reference LI loss from SLIG outputs."""
    student = indexer_softmax.float().clamp_min(1e-10)
    target = teacher_mass.float().clamp_min(0)
    target_sum = target.sum(dim=-1, keepdim=True)
    valid_target = target_sum > 1e-10
    student = torch.where(valid_target, student, torch.ones_like(student))
    teacher = target / target_sum.clamp_min(1e-10)
    log_teacher = teacher.clamp_min(1e-10).log()
    loss = (teacher * (log_teacher - student.log())).sum(dim=-1)
    return (target_sum.squeeze(-1) * loss).mean() * softmax_scale


class _SparseFlashMLATND(torch.autograd.Function):
    """Bridge SparseFlashMLA forward and SMLAG + SLIG backward.

    The ``cann_ops_transformer`` kernels carry no native autograd
    registration (unlike dsv3.2's older ``torch_npu`` ops), so the manual
    Function runs SMLA forward, SMLAG backward, and the SLIG indexer-loss
    gradient (which consumes SMLAG's ``cmp_softmax_l1``) in one bridge.
    All AscendC metadata tensors are precomputed in
    ``CompressedBlockLayout``.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        cmp_sparse_indices,
        sinks,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        attention_metadata,
        idx_q,
        idx_k,
        idx_w,
        softmax_scale,
        ratio,
        window_size,
        indexer_loss_coeff,
        indexer_loss_accumulator,
        hooks,
    ):
        # Non-CP self-attention uses the same cumsum for Q and original KV.
        # Resolve the alias inside the Function instead of passing one Tensor
        # object twice to autograd.Function.apply, which Dynamo rejects.
        if cu_seqlens_ori_kv is None:
            cu_seqlens_ori_kv = cu_seqlens_q
        has_compressed = ratio > 1
        torch._check(swa_k.shape[0] > 0, lambda: "SparseFlashMLA original KV must not be empty")
        asc = attention_metadata.asc
        if asc is None:
            raise ValueError("the fused core needs this geometry's AI_CPU records")
        # The precomputed records are read here instead of re-running the AI_CPU operator per
        # layer; the backward reads its own off the same entry.
        result, softmax_lse = hooks.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            cmp_sparse_indices=(cmp_sparse_indices if cmp_sparse_indices is not None else None),
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            sinks=sinks,
            metadata=asc.sparse_attn.fwd,
            softmax_scale=softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=0 if _IS_A5 and not has_compressed else _CMP_MASK_MODE,
            ori_win_left=window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
            return_softmax_lse=True,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            idx_q,
            idx_k,
            idx_w,
            result,
            softmax_lse,
        )
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.window_size = window_size
        ctx.indexer_loss_coeff = indexer_loss_coeff
        ctx.indexer_loss_accumulator = indexer_loss_accumulator
        ctx.hooks = hooks
        # A dataclass cannot go through ``save_for_backward``; it is a context attribute
        # instead, holding the same records the graph already keeps for the backward.
        ctx.attention_metadata = attention_metadata
        return result

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        (
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            idx_q,
            idx_k,
            idx_w,
            result,
            softmax_lse,
        ) = ctx.saved_tensors
        has_compressed = ctx.ratio > 1
        has_sparse_indices = cmp_sparse_indices is not None

        (
            dquery,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = ctx.hooks.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(cmp_sparse_indices if has_sparse_indices else None),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=(cu_seqlens_cmp_kv if has_compressed else None),
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=(cmp_residual_kv if has_compressed else None),
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=ctx.attention_metadata.asc.sparse_attn.bwd,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.ratio,
            ori_mask_mode=_ORI_MASK_MODE,
            cmp_mask_mode=0 if _IS_A5 and not has_compressed else _CMP_MASK_MODE,
            ori_win_left=ctx.window_size - 1,
            ori_win_right=0,
            layout_q=_LAYOUT,
            layout_kv=_LAYOUT,
        )
        if not has_compressed:
            dcmp_kv = None

        didx_q = didx_k = didx_w = None
        if ctx.ratio == 4 and ctx.indexer_loss_coeff != 0:
            didx_q, didx_k, didx_w = _SparseFlashMLATND._indexer_loss_grad(
                ctx,
                cmp_softmax_l1,
                idx_q,
                idx_k,
                idx_w,
                cmp_sparse_indices,
                cmp_residual_kv,
                cu_seqlens_q,
                cu_seqlens_cmp_kv,
            )

        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_ori_kv
            None,  # cu_seqlens_cmp_kv
            None,  # cmp_residual_kv
            None,  # attention_metadata
            didx_q,
            didx_k,
            didx_w,
            None,  # softmax_scale
            None,  # ratio
            None,  # window_size
            None,  # indexer_loss_coeff
            None,  # indexer_loss_accumulator
            None,  # hooks
        )

    @staticmethod
    def _indexer_loss_grad(
        ctx,
        cmp_softmax_l1,
        idx_q,
        idx_k,
        idx_w,
        cmp_sparse_indices,
        cmp_residual_kv,
        cu_seqlens_q,
        cu_seqlens_cmp_kv,
    ):
        """The LightningIndexer KL-loss gradient (SLIG), mirroring dsv3.2's
        ``AscSparseIndexerLoss`` role: runs in the backward (it consumes
        SMLAG's ``cmp_softmax_l1``), scales the indexer grads, and
        accumulates the detached LI loss for logging."""
        slig_metadata = ctx.attention_metadata.asc.lightning_indexer.bwd
        if any(x is None for x in (idx_q, idx_k, idx_w, slig_metadata)):
            raise RuntimeError("ratio-4 asc requires the LI tensors and the selector's backward record.")
        (
            didx_q,
            didx_k,
            didx_w,
            indexer_softmax,
        ) = ctx.hooks.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w.float(),
            sparse_indices=cmp_sparse_indices,
            attn_softmax_l1_norm=cmp_softmax_l1,
            cmp_residual_k=cmp_residual_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_cmp_kv,
            metadata=slig_metadata,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_CMP_MASK_MODE,
            cmp_ratio=4,
        )
        if ctx.indexer_loss_accumulator is not None:
            li_loss = _compute_li_loss(
                indexer_softmax,
                cmp_softmax_l1,
                ctx.softmax_scale,
            )
            ctx.indexer_loss_accumulator.add_(li_loss.detach())
        query_rows = cmp_softmax_l1.sum(dim=-1).numel()
        # Keep the row count symbolic while GraphTrainer captures sequence
        # chunks; converting it to float creates a data-shape guard.
        grad_scale = ctx.indexer_loss_coeff * ctx.softmax_scale / query_rows
        didx_q = (didx_q * grad_scale).to(idx_q.dtype)
        didx_k = (didx_k * grad_scale).to(idx_k.dtype)
        didx_w = (didx_w * grad_scale).to(idx_w.dtype)
        return didx_q, didx_k, didx_w


class AscCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Run LI and SMLA/SMLAG/SLIG in a local TND layout.

    The flow mirrors ``AscSparseInnerAttention`` (dsv3.2): the core
    contains the **kgather** — the compressed-level gather assembling the
    per-segment packed TND streams from the all-gathered padded containers
    (``cmp_k_global_gather_indices``; the plain identity slice without context
    parallel) — and then calls the AscendC ops directly (LightningIndexer,
    then SparseFlashMLA).  The difference from dsv3.2 is the op set: the
    ``cann_ops_transformer`` SMLA/SMLAG/SLIG kernels carry no native
    autograd registration (unlike the older ``torch_npu``
    ``npu_sparse_flash_attention``), so the fused core needs the manual
    ``_SparseFlashMLATND`` bridge — SMLA forward, SMLAG backward, and the
    indexer KL-loss gradient (SLIG, the dsv3.2 ``AscSparseIndexerLoss``
    role) in the same backward because it consumes SMLAG's
    ``cmp_softmax_l1``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        """The fused config: the reference core's fields plus the geometry its records need.

        That geometry is the *port's*: the reference core neither reads it nor knows it exists.
        The widths are declared without defaults, so an entry can only size a record from a
        value some override stated; the window and the selection width come from this config's
        own inherited fields, so they are stated once, on the model config, and never duplicated.
        """

        indexer_loss_coeff: float = 0.0
        """Scales the LightningIndexer KL-loss gradient produced by SLIG."""

        num_heads: int  # pyrefly: ignore [bad-override]
        head_dim: int
        """The main attention geometry the SMLA/SMLAG records are sized with."""

        index_num_heads: int
        index_head_dim: int
        """The selector geometry its record and SLIG backward are sized with."""

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

            The model-dir packed contract plus the record tier: this geometry's four
            ``*_metadata`` kernel outputs, computed here once per microbatch for every layer of
            the geometry, and the ``_mark_dynamic`` marks land on the object the layers
            actually read.
            """
            del padding_mask, max_num_documents, max_context_length
            # The fused builder owns its packed contract: materialize the host boundaries once
            # here so the metadata builder and the compressor's state tables read them without a
            # second device-to-host sync.  The reference builder stays free of NPU host caches.
            from torchtitan.models.common.attention import create_varlen_metadata_for_document

            from torchtitan_npu.models.deepseek_v4.metadata import build_compressed_varlen_metadata

            varlen = create_varlen_metadata_for_document(positions, include_host_offsets=True)
            metadata = build_compressed_varlen_metadata(varlen, self.compress_ratio, window_size=self.window_size)
            return build_aicpu_metadata(metadata, geometry=self.geometry)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim
        self.indexer_loss_coeff = config.indexer_loss_coeff
        self.register_buffer(
            "_indexer_loss_acc",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.hooks = _ASC_SPARSEATTN_HOOK

    @staticmethod
    def _assemble_tnd(container, plan) -> torch.Tensor:
        """The kgather: the gathered container -> the per-segment packed
        TND stream (``[T_cmp, 1, D]``)."""
        asm = plan.cmp_k_global_gather_indices
        flat = container.flatten(0, 1)
        if asm is not None:
            return flat[asm].unsqueeze(1).contiguous()
        if plan.block_positions is None:
            raise ValueError("TND assembly requires plan.block_positions.")
        # Keep K's logical length in step with the current cumulative lengths.
        return flat[: plan.block_positions.numel()].unsqueeze(1).contiguous()

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        *,
        attn_sink: torch.Tensor | None = None,
        sparse_indices=None,
        attention_metadata: DeepSeekV4Metadata | None = None,
    ) -> torch.Tensor:
        hooks = self.hooks
        if attention_metadata is None:
            raise TypeError(
                "the fused core received no attention metadata: the decoder hands each layer the "
                "entry its attention_metadata_key names, so this layer's backend config did not "
                "build one (check the override.imports selection and the layer's ratio)."
            )
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError(f"asc requires AscV4Metadata attention metadata, got {type(attention_metadata).__name__}.")
        if attn_sink is None:
            raise ValueError("AscCompressedSparseInnerAttention requires attn_sink")
        metadata = attention_metadata
        plan = metadata.plan
        if self.compress_ratio <= 1:
            if cmp_k is not None and cmp_k.numel() != 0:
                raise ValueError("ratio-1 asc must not receive compressed KV.")
        elif cmp_k is None or cmp_k.ndim != 3:
            raise ValueError("ratio>1 asc requires compressed KV in the container layout [B, S//ratio, D].")
        # q / swa_k shape consistency and the kernel input layouts are
        # validated by the aclnn interface itself.
        if metadata.asc is None:
            raise ValueError("asc requires the AI_CPU records the fused port's builder fills.")
        # The records are sized from the config's stated geometry, while the layers run from the
        # model's: a recipe that states other widths would otherwise hand the kernels a record
        # built for a different shape.  Both checks are on static head dims, so they are plain
        # Python booleans at trace time.
        if q.shape[2] != self.num_heads or q.shape[3] != self.head_dim:
            raise ValueError(
                f"the fused records were sized for num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"but the layer runs {tuple(q.shape[2:])}"
            )
        if idx_q is not None and tuple(idx_q.shape[2:]) != (self.index_num_heads, self.index_head_dim):
            raise ValueError(
                f"the selector records were sized for index_num_heads={self.index_num_heads}, "
                f"index_head_dim={self.index_head_dim}, but the layer runs {tuple(idx_q.shape[2:])}"
            )
        batch_size, seqlen, _, _ = q.shape

        q = q.flatten(0, 1)
        swa_k = swa_k.flatten(0, 1).unsqueeze(1).contiguous()
        # The ori cumsum: the kernel's ``cu_seqlens_ori_kv`` — under context
        # parallel the window plan's packed cumsum, otherwise the bare
        # ``cu_seq_k`` (identical to ``cu_seq_q`` without CP).
        cu_seqlens_ori_kv = metadata.window.cu_seqlens_ori_kv if metadata.window is not None else None
        if cmp_k is not None:
            # The container is the ShardingConfig all-gather's output (under
            # CP the padded per-rank containers concatenated); the kgather
            # assembles the per-segment packed stream.
            cmp_k = self._assemble_tnd(cmp_k, plan)

        cmp_sparse_indices = None if sparse_indices is None else sparse_indices.flatten(0, 1)
        if self.compress_ratio == 4:
            if cmp_sparse_indices is None:
                raise ValueError("ratio-4 asc requires sparse_indices.")
            if idx_q is None or idx_k is None or idx_w is None:
                raise ValueError("ratio-4 asc requires all LI projection tensors for backward.")
            idx_q = idx_q.flatten(0, 1)
            idx_k = self._assemble_tnd(idx_k, plan)
            idx_w = idx_w.flatten(0, 1)

        indexer_loss_coeff = float(self.indexer_loss_coeff) if self.training else 0.0
        indexer_loss_accumulator = self._indexer_loss_acc

        output = _SparseFlashMLATND.apply(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            attn_sink.float(),
            metadata.varlen.cu_seq_q,
            cu_seqlens_ori_kv,
            plan.cu_seqlens_cmp_k,
            plan.block_remainder,
            metadata,
            idx_q,
            idx_k,
            idx_w,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            indexer_loss_coeff,
            indexer_loss_accumulator,
            hooks,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # Device declared at the creation point instead of an ambient device
        # context; ``buffer_device=None`` keeps the to_empty() protocol (the
        # emptied buffer already records the target device).
        self._indexer_loss_acc = torch.zeros(
            (), dtype=torch.float32, device=buffer_device or self._indexer_loss_acc.device
        )
