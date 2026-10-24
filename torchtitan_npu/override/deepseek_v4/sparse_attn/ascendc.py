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
The ``sparse_attn`` package ``__init__`` defines the registration, so the
override path stays ``override.deepseek_v4.sparse_attn.asc``.

The fused path only bridges the AscendC TND kernels' forward/backward with a
slim ``torch.autograd.Function``.  The selector is the separate
``lightning_indexer.asc`` override: this bridge consumes its ``topk_indices``,
and SMLAG's backward emits the teacher on the selector's ``topk_scores`` edge.
All AscendC metadata is precomputed at the backend's own metadata boundary and
carried in the registered ``AscV4Metadata``, so this module never builds or
caches metadata per layer; because the records describe *frames*, the CP
transform rebuilds them for this rank (``prepare_cp_metadata``).  The CP path's
``S(1) -> R`` sharding gathers the KV streams to the whole row at the wrapper
boundary, and the frames' ``seqused`` limits each rank's view.  Every fused
kernel, however, anchors its mask at the *end* of the key frame it is given (the
selector reads no ``seqused`` at all), so a rank whose reach is a document prefix
then compacts each stream onto that prefix: ``selector_k`` /
``selector_k_gather`` for the compressed pool and ``swa_k_prefix`` /
``swa_k_gather`` for the sliding window.  The kernels therefore see the reach as
their key frame's end, while the picks and indices keep the model's
document-local numbering.  There is no container or ``_assemble_tnd`` step
anymore.

The kernel inputs follow the training autocast contract: q/k/v are bf16 and
the sink is fp32 — an eager-fp32 caller fails loudly at the kernel's dtype
check.
"""

from dataclasses import dataclass

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_flash_mla_grad_metadata,
    sparse_flash_mla_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.common.packing import require_document_alignment
from torchtitan_npu.models.common.query_exchange import ExchangeMetadata, cp_plan
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.deepseek_v4.metadata import (
    LI_CMP_RATIO,
    AicpuGeometry,
    AicpuMetadata,
    AscLightningIndexerMetadata,
    AscSparseAttentionMetadata,
    lightning_indexer_options,
)
from torchtitan_npu.override.deepseek_v4.metadata import (
    AscV4Metadata as AscV4Metadata,
)

_LAYOUT = "TND"
_ORI_MASK_MODE = 4
_CMP_MASK_MODE = 3


# ---------------------------------------------------------------------------
# AscendC metadata layer
# ---------------------------------------------------------------------------
#
# The AscendC backend's metadata is the model-dir row plus its own record
# tier: after the document-packed row is built, this backend fills
# the precomputed AscendC ``*_metadata`` kernel outputs (lightning_indexer,
# sparse_flash_mla, sparse_flash_mla_grad,
# sparse_lightning_indexer_kl_loss_grad) into an ``AscV4Metadata`` wrapper,
# keeping the model-dir metadata NPU-free.  Those kernels take only shape
# information, so their results are layer-invariant and reused by every DSA
# layer and by the backward pass.  ``AscCompressedSparseInnerAttention.Config``
# builds the whole thing (upstream #4897); because a record describes the frames
# it was derived from, ``prepare_cp_metadata`` rebuilds them per rank.


def shard_asc_v4_metadata(
    attention_metadata: AscV4Metadata,
    *,
    cp_size: int,
    cp_rank: int,
    batch_size: int,
    seq_len: int,
) -> AscV4Metadata:
    """One rank's frames and records: the row re-cut into this rank's chunks.

    The queries are chunk ``cp_rank`` of every document while the KV streams stay the **whole
    row** -- gathered declaratively at the fused attention's boundary -- so the KV frames'
    ``cu_seqlens`` address that row and their ``seqused = K_d = (cp_rank + 1) * Q_d`` is how far
    these queries reach into each document.  That is what keeps the kernels' end-aligned window
    and causal limit anchored at this rank's chunk rather than at the row's end.

    ``seq_len`` is the **slab** width -- the shape of the inputs this rank holds, not of the row.
    The compressor is not re-cut here: its carry is applied by ``CarryExchange`` at
    the call site, from the same row grid and this rank's slab.
    """
    global_cu = attention_metadata.cu_seq_q
    ratio = attention_metadata.ratio
    # The strictest packing check: every document a multiple of cp_size x this ratio.
    require_document_alignment(global_cu, cp_size=cp_size, ratio=ratio)
    seqused = torch.diff(global_cu).to(torch.int32)
    chunk = (seqused // cp_size).to(torch.int32)
    if int(chunk.sum()) != batch_size * seq_len:
        raise ValueError(
            f"the sliced rows ({batch_size * seq_len}) are not the chunks' total "
            f"({int(chunk.sum())}): the boundaries describe a different row than the one the "
            "inputs were sliced from."
        )
    reach = ((cp_rank + 1) * chunk).to(torch.int32)
    exchange = cp_plan(global_cu, cp_size, cp_rank, seq_len)
    q = KernelFrame(cu_seqlens=(global_cu // cp_size).to(torch.int32), seqused=chunk)
    swa_k = KernelFrame(cu_seqlens=global_cu.to(torch.int32), seqused=reach)
    cmp_k = None
    if ratio > 1:
        cmp_k = KernelFrame(
            cu_seqlens=attention_metadata.cu_seq_k.to(torch.int32),
            seqused=(reach // ratio).to(torch.int32),
            residual=(reach % ratio).to(torch.int32),
        )
    return build_aicpu_metadata(
        attention_metadata,
        geometry=attention_metadata.aicpu_metadata.geometry,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        positions_q=exchange.positions_q.reshape(batch_size, seq_len),
        exchange=exchange,
    )


def _selector_key_frame(cmp_k: KernelFrame) -> tuple[KernelFrame, torch.Tensor] | None:
    """The rank-local prefix pool the compressed kernels must be handed, or ``None``.

    The compressed kernels -- ``lightning_indexer`` and ``sparse_flash_mla``'s compressed
    branch -- right-align their compressed causal mask at the end of the key frame they are
    given (the selector reads no ``seqused`` at all), so a rank whose queries reach only a
    prefix of every document would otherwise select the document-tail window: future keys for
    its own queries.  The compressed stream is therefore handed the prefix pool itself:
    ``prefix_d = reach_d // ratio`` blocks per document, laid out in document order by
    ``gather``, with this frame as its grid.  The picks stay document-local, so the same
    numbers address the whole-row pool ``cmp_k`` describes and the attention needs no
    translation.  Wider-ratio (HCA) layers read the same frame without a selector.

    A frame that already addresses every block of every document (one degree, or the final
    rank, whose reach is the whole document) is its own compressed frame and returns ``None``.
    """
    if cmp_k.residual is None:
        raise ValueError("the selector's key frame needs a compressed axis")
    full = (cmp_k.cu_seqlens[1:] - cmp_k.cu_seqlens[:-1]).to(torch.int64)
    prefix = cmp_k.seqused.to(torch.int64)
    if torch.equal(prefix, full):
        return None
    device = prefix.device
    starts = cmp_k.cu_seqlens[:-1].to(torch.int64)
    # The rows ``index_select`` picks: document ``d``'s first ``prefix_d`` blocks, documents in
    # order.  A document's blocks are the contiguous rows its boundary pair names, so this is
    # ``starts[d] + j`` for ``j < prefix_d``.
    documents = torch.repeat_interleave(torch.arange(prefix.numel(), device=device), prefix)
    within = torch.arange(int(prefix.sum()), device=device) - torch.repeat_interleave(
        torch.cumsum(prefix, 0) - prefix, prefix
    )
    gather = starts[documents] + within
    cu_seqlens = torch.cat([prefix.new_zeros(1), torch.cumsum(prefix, 0)]).to(torch.int32)
    return KernelFrame(cu_seqlens=cu_seqlens, seqused=prefix.to(torch.int32), residual=cmp_k.residual), gather


def _window_key_frame(swa_k: KernelFrame) -> tuple[KernelFrame, torch.Tensor] | None:
    """The rank-local token prefix the sliding-window kernels must be handed, or ``None``.

    ``sparse_flash_mla`` anchors its window at the end of the key frame it is given, the same
    contract the compressed kernels state.  A rank whose reach is a document prefix therefore
    hands the sliding window that prefix: ``reach_d`` tokens per document, laid out in document
    order by ``gather``.  Every key a chunk-``r`` query can attend to lies inside it -- the
    chunk is exactly the prefix's last ``chunk_d`` rows -- so a right-aligned window on the
    prefix lands on this rank's chunk.  ``seqused`` equals the frame's boundary differences,
    which makes the mask independent of whether the kernel reads ``seqused`` or the frame end.

    A frame that already is the whole document (one degree, or the final rank) needs no
    compaction and returns ``None``.
    """
    reach = swa_k.seqused.to(torch.int64)
    full = (swa_k.cu_seqlens[1:] - swa_k.cu_seqlens[:-1]).to(torch.int64)
    if torch.equal(reach, full):
        return None
    device = reach.device
    starts = swa_k.cu_seqlens[:-1].to(torch.int64)
    documents = torch.repeat_interleave(torch.arange(reach.numel(), device=device), reach)
    within = torch.arange(int(reach.sum()), device=device) - torch.repeat_interleave(
        torch.cumsum(reach, 0) - reach, reach
    )
    gather = starts[documents] + within
    cu_seqlens = torch.cat([reach.new_zeros(1), torch.cumsum(reach, 0)]).to(torch.int32)
    return KernelFrame(cu_seqlens=cu_seqlens, seqused=reach.to(torch.int32), residual=None), gather


def _selector_records(
    q: KernelFrame,
    key_frame: KernelFrame | None,
    geometry: AicpuGeometry,
    ratio: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The selector's forward record and its SLIG backward record, for one geometry."""
    if ratio != LI_CMP_RATIO or key_frame is None:
        return None, None
    frames = lightning_indexer_options(ratio=ratio, q=q, cmp_k=key_frame)
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


def _sparse_attention_options(
    *,
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    ratio: int,
    window_size: int,
):
    """The options the SMLA/SMLAG kernels and their metadata builders share."""
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


def _sparse_attention_records(
    q: KernelFrame,
    swa_k: KernelFrame,
    cmp_k: KernelFrame | None,
    geometry: AicpuGeometry,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The SMLA/SMLAG records for these frames."""
    has_cmp_kv = cmp_k is not None
    options = _sparse_attention_options(
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        ratio=ratio,
        window_size=geometry.window_size,
    )
    geometry_kwargs = dict(
        ori_topk=0,
        # Only the ratio-4 CSA geometry selects compressed entries.  A ratio-128 HCA layer
        # addresses the whole causal prefix, and the kernel rejects a non-zero ``cmp_topk``
        # with a ratio above 4 (the error the A3 integration run surfaced).
        cmp_topk=geometry.index_topk if ratio == LI_CMP_RATIO else 0,
        has_ori_kv=True,
        has_cmp_kv=has_cmp_kv,
    )
    smla_metadata = sparse_flash_mla_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        ori_topk_length=None,
        cmp_topk_length=None,
        **options,
        **geometry_kwargs,
    )
    smla_grad_metadata = sparse_flash_mla_grad_metadata(
        geometry.num_heads,
        1,
        geometry.head_dim,
        **options,
        **geometry_kwargs,
    )
    return smla_metadata, smla_grad_metadata


def build_aicpu_metadata(
    metadata: DeepSeekV4Metadata,
    *,
    geometry: AicpuGeometry,
    q: KernelFrame | None = None,
    swa_k: KernelFrame | None = None,
    cmp_k: KernelFrame | None = None,
    positions_q: torch.Tensor | None = None,
    exchange: ExchangeMetadata | None = None,
) -> AscV4Metadata:
    """The AscendC record tier over one geometry's frames (upstream #4897).

    Frames not supplied are the row's own (``metadata.q`` / ``swa_k`` / ``cmp_k``).  The AI_CPU
    metadata is computed from exactly these frames, which is why a re-cut frame -- the CP
    transform's rank-local chunks -- runs this again rather than reusing records of the frame
    they came from.
    """
    ratio = metadata.ratio
    q = metadata.q if q is None else q
    swa_k = metadata.swa_k if swa_k is None else swa_k
    cmp_k = metadata.cmp_k if cmp_k is None else cmp_k
    if cmp_k is not None and ratio <= 1:
        cmp_k = None
    if cmp_k is not None and ratio > 1 and int(cmp_k.cu_seqlens[-1]) == 0:
        raise ValueError(
            f"batch has no complete compression block for ratio={ratio}; doc-packed sequences "
            "must be long enough to produce at least one full block."
        )
    # The compressed and sliding-window key sides are rank-local: the attention's own row frames
    # stay the whole row, while a rank whose reach is a document prefix gets that prefix as its
    # pool.  The records below are built from whichever frames the kernels actually address, and
    # the gathers are what turn the gathered rows into those frames' layouts.
    # Every compressed geometry is handed its reach as the key frame: the ratio-4 selector picks
    # against it and the wider-ratio (HCA) layers run their full-causal compressed branch on it.
    selector = _selector_key_frame(cmp_k) if cmp_k is not None and ratio > 1 else None
    selector_k, selector_k_gather = selector if selector is not None else (None, None)
    # The sliding window gets the same rank-local treatment: its kernels anchor the window at
    # the key frame's end, so a prefix-reaching rank hands them its token prefix.
    window = _window_key_frame(swa_k)
    swa_k_prefix, swa_k_gather = window if window is not None else (None, None)
    window_frame = swa_k if swa_k_prefix is None else swa_k_prefix
    li_record, slig_metadata = _selector_records(
        q,
        cmp_k if selector_k is None else selector_k,
        geometry,
        ratio,
    )
    smla_metadata, smla_grad_metadata = _sparse_attention_records(
        q,
        window_frame,
        cmp_k if selector_k is None else selector_k,
        geometry,
        ratio,
    )
    record = AicpuMetadata(
        geometry=geometry,
        sparse_attn=AscSparseAttentionMetadata(fwd=smla_metadata, bwd=smla_grad_metadata),
        lightning_indexer=AscLightningIndexerMetadata(fwd=li_record, bwd=slig_metadata),
    )
    result = AscV4Metadata(
        cu_seq_q=metadata.cu_seq_q,
        cu_seq_k=metadata.cu_seq_k,
        ratio=ratio,
        positions_q=positions_q if positions_q is not None else metadata.positions_q,
        exchange=exchange,
        q=q,
        swa_k=swa_k,
        cmp_k=cmp_k,
        aicpu_metadata=record,
        selector_k=selector_k,
        selector_k_gather=selector_k_gather,
        swa_k_prefix=swa_k_prefix,
        swa_k_gather=swa_k_gather,
    )
    _mark_dynamic(result)
    return result


def _mark_dynamic(metadata: AscV4Metadata) -> None:
    """Mark batch-dependent leading dimensions so Dynamo does not specialize."""
    tensors: list[torch.Tensor] = [metadata.cu_seq_q, metadata.cu_seq_k, metadata.positions_q]
    tensors += [metadata.q.cu_seqlens, metadata.q.seqused, metadata.swa_k.cu_seqlens, metadata.swa_k.seqused]
    if metadata.cmp_k is not None:
        tensors += [metadata.cmp_k.cu_seqlens, metadata.cmp_k.seqused]
        if metadata.cmp_k.residual is not None:
            tensors.append(metadata.cmp_k.residual)
    if metadata.selector_k is not None:
        tensors += [metadata.selector_k.cu_seqlens, metadata.selector_k.seqused]
        if metadata.selector_k.residual is not None:
            tensors.append(metadata.selector_k.residual)
    if metadata.selector_k_gather is not None:
        tensors.append(metadata.selector_k_gather)
    if metadata.swa_k_prefix is not None:
        tensors += [metadata.swa_k_prefix.cu_seqlens, metadata.swa_k_prefix.seqused]
    if metadata.swa_k_gather is not None:
        tensors.append(metadata.swa_k_gather)
    if metadata.exchange is not None:
        tensors += [metadata.exchange.send_index, metadata.exchange.positions_q]
    if metadata.aicpu_metadata.lightning_indexer.fwd is not None:
        tensors.append(metadata.aicpu_metadata.lightning_indexer.fwd)
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)
        # Written via setattr: "_dynamo_unbacked_indices" is the dynamo shape
        # annotation the GraphTrainer tracer reads (torchtitan
        # experiments/graph_trainer/dynamic_shapes.py); the underscore name is
        # deliberate and must stay off the codecheck protected-member list.
        setattr(tensor, "_dynamo_unbacked_indices", {0})  # noqa: B010


def _gathered_rows(rows: torch.Tensor, gather: torch.Tensor | None) -> torch.Tensor:
    """``rows[gather]`` when this rank addresses a prefix, the rows themselves otherwise.

    The gather is differentiable, so a consumer's gradient is scattered back onto the gathered
    row, which the wrapper's ``S(1) -> R`` backward then reduce-scatters onto the producer.
    """
    return rows if gather is None else rows.index_select(0, gather)


def _compressed_row(cmp_k: torch.Tensor, metadata: AscV4Metadata) -> torch.Tensor:
    """The compressed stream in the frame the fused kernels grid it with.

    The attention's compressed keys are gathered to the whole row, but a rank that reaches a
    document prefix addresses only that prefix -- the same pool the selector picked from and
    the same frame its records were built with (``selector_k``).  Compacting the row here is
    what keeps the kernel's key tensor, its grid and the selection one coordinate system.
    """
    return _gathered_rows(cmp_k, metadata.selector_k_gather)


def _window_row(swa_k: torch.Tensor, metadata: AscV4Metadata) -> torch.Tensor:
    """The sliding-window stream in the frame its kernels are given (``swa_k_prefix``)."""
    return _gathered_rows(swa_k, metadata.swa_k_gather)


def _selector_keys(idx_k: torch.Tensor, metadata: AscV4Metadata) -> torch.Tensor:
    """The compressed keys in the coordinate frame the selector addresses.

    ``idx_k`` is the gathered whole row ``[B, N, D]``.  Under context parallelism a non-final
    rank's selector addresses only each document's prefix, so the row is compacted with the
    entry's ``selector_k_gather``; without one the row already is the selector's pool and comes
    back unchanged.  Both the selector's forward and its SLIKG backward read their keys through
    this function, so the tensor and its frame can never come from different pools.
    """
    flat = idx_k.flatten(0, 1)
    gather = metadata.selector_k_gather
    return flat if gather is None else flat.index_select(0, gather)


class _SparseFlashMLATND(torch.autograd.Function):
    """Bridge SparseFlashMLA forward with SMLAG backward and its teacher by-product.

    The ``cann_ops_transformer`` kernels carry no native autograd registration, so the manual
    Function runs SMLA forward and SMLAG backward.  It does **not** train the indexer: SMLAG's
    backward emits the raw teacher on ``topk_scores``' gradient and the selector's own
    ``_LightningIndexerTND`` backward consumes that edge with SLIKG.  ``topk_scores`` is an
    input here only so an incoming gradient on it exists and is routed to the emitter.

    The kernels are addressed by this geometry's frames, not by a plan: the query frame is this
    rank's chunks, the two KV frames are the whole row, and their ``seqused`` is what limits a
    rank's view.
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
        metadata = attention_metadata
        if not isinstance(metadata, AscV4Metadata):
            raise TypeError("the fused core needs AscV4Metadata")
        asc = metadata.aicpu_metadata
        has_compressed = metadata.cmp_k is not None
        key_frame = metadata.cmp_k if metadata.selector_k is None else metadata.selector_k
        window_frame = metadata.swa_k if metadata.swa_k_prefix is None else metadata.swa_k_prefix
        options = _sparse_attention_options(
            q=metadata.q,
            swa_k=window_frame,
            cmp_k=key_frame,
            ratio=ratio,
            window_size=window_size,
        )
        result, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            cmp_sparse_indices=(topk_indices if has_compressed else None),
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=asc.sparse_attn.fwd,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            result,
            softmax_lse,
        )
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.window_size = window_size
        # The carrier is an input only while training; its position in the backward's
        # per-input list must still be answered with ``None`` when it was not threaded.
        ctx.has_carrier = topk_scores is not None
        # A dataclass cannot go through ``save_for_backward``; it is a context attribute
        # instead, holding the same frames the graph already keeps for the backward.
        ctx.attention_metadata = metadata
        # The LSE is a teacher signal, never a loss input.
        ctx.mark_non_differentiable(softmax_lse)
        return result, softmax_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse
        (
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            result,
            softmax_lse,
        ) = ctx.saved_tensors
        metadata = ctx.attention_metadata
        has_compressed = metadata.cmp_k is not None
        key_frame = metadata.cmp_k if metadata.selector_k is None else metadata.selector_k
        window_frame = metadata.swa_k if metadata.swa_k_prefix is None else metadata.swa_k_prefix
        options = _sparse_attention_options(
            q=metadata.q,
            swa_k=window_frame,
            cmp_k=key_frame,
            ratio=ctx.ratio,
            window_size=ctx.window_size,
        )
        (
            dquery,
            dori_kv,
            dcmp_kv,
            dsinks,
            _,
            cmp_softmax_l1,
        ) = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            result,
            softmax_lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k if has_compressed else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=(topk_indices if has_compressed else None),
            sinks=sinks,
            metadata=metadata.aicpu_metadata.sparse_attn.bwd,
            softmax_scale=ctx.softmax_scale,
            **options,
        )
        if not has_compressed:
            dcmp_kv = None

        # The teacher edge: SMLAG's raw marginal mass, with every ``-1`` slot zeroed (the
        # kernel may write junk there, and SLIKG sums every slot into ``p_reduce``), scaled by
        # ``softmax_scale`` exactly once here; the selector's backward only divides by the
        # microbatch's query-row count.  Only a layer that threads a carrier has an edge to
        # emit on: an HCA layer's compressed branch has no selector, and an eval forward
        # passes no carrier.
        teacher = None
        if ctx.has_carrier and cmp_softmax_l1 is not None:
            teacher = cmp_softmax_l1.masked_fill(topk_indices < 0, 0.0) * ctx.softmax_scale

        # One gradient per forward input, in order: (q, swa_k, cmp_k, topk_indices, sinks,
        # attention_metadata, softmax_scale, ratio, window_size, topk_scores).
        return (
            dquery,
            dori_kv,
            dcmp_kv,
            None,  # topk_indices
            dsinks,
            None,  # attention_metadata
            None,  # softmax_scale
            None,  # ratio
            None,  # window_size
            teacher,  # topk_scores
        )


class AscCompressedSparseInnerAttention(CompressedSparseInnerAttention):
    """Run SMLA/SMLAG in a local TND layout, consuming the selector's selection.

    The flow mirrors ``AscSparseInnerAttention`` (dsv3.2): the core reads this geometry's
    frames (queries on this rank's chunks) and calls the AscendC ops directly.  The selector's
    forward/backward is **not** here: it is the separate ``lightning_indexer.asc`` override,
    which emits ``topk_indices`` and the teacher carrier.  This core consumes both, and its
    SMLAG backward writes the raw teacher onto the carrier's gradient.  The container/plan
    assembly is gone: the compressor produces the whole compressed row through the core's
    ``S(1) -> R`` gather, and the frames' ``seqused`` limits each rank's view.  Both KV streams
    are then compacted onto this rank's reach (``selector_k`` / ``selector_k_gather`` for the
    compressed pool, ``swa_k_prefix`` / ``swa_k_gather`` for the sliding window), because every
    fused kernel anchors its mask at the key frame's end -- the same frame the selector picked
    in -- and the gradients scatter back through those gathers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention.Config):
        """The fused config: the reference core's fields plus the geometry its records need."""

        num_heads: int  # pyrefly: ignore [bad-override]
        head_dim: int
        """The main attention geometry the SMLA/SMLAG records are sized with."""

        index_num_heads: int
        index_head_dim: int
        """The selector geometry its record and SLIKG backward are sized with."""

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

            The model-dir plain row plus the record tier: this geometry's four ``*_metadata``
            kernel outputs, computed here once per microbatch for every layer of the geometry.
            """
            del padding_mask, max_num_documents, max_context_length
            from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata

            # The fused core never reads ``slots``: its HCA branch runs full-causal on the
            # compressed pool, so the reference-only S x (S / ratio) selection is skipped here.
            metadata = build_v4_metadata(positions, self.compress_ratio, build_slots=False)
            return build_aicpu_metadata(metadata, geometry=self.geometry)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim

    def forward(  # pyrefly: ignore [bad-override]
        self,
        q,
        swa_k,
        attn_sink: torch.Tensor,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        attention_metadata: AscV4Metadata,
    ) -> torch.Tensor:
        # The single-source contract (see ``docs/feature_guides/fused_ops.md``): the fused
        # kernels read the records their own builder attached, so an entry built by any other
        # backend is refused here rather than launched with missing metadata.  There is no
        # inline fallback.
        if not isinstance(attention_metadata, AscV4Metadata):
            raise TypeError(
                "asc requires the attention metadata its own config built, got "
                f"{type(attention_metadata).__name__}: the decoder hands each layer the entry "
                "its attention_metadata_key names, so this layer's backend did not build one."
            )
        if self.compress_ratio == 4 and (cmp_k is None or topk_indices is None):
            raise ValueError("ratio-4 asc requires cmp_k and topk_indices together.")
        metadata = attention_metadata
        batch_size, seqlen, _, _ = q.shape

        q = q.flatten(0, 1)
        # Both KV streams arrive as the gathered whole row; a rank whose reach is a document
        # prefix compacts each onto the frame its records were built with.
        swa_k = _window_row(swa_k.flatten(0, 1).unsqueeze(1).contiguous(), metadata)
        cmp_k = _compressed_row(cmp_k.flatten(0, 1).unsqueeze(1).contiguous(), metadata) if cmp_k is not None else None

        topk_indices_tnd = None
        if topk_indices is not None:
            # The model carries the selection as ``[B, L, K]``; a TND kernel wants
            # ``[T, 1, K]``.  Order is left exactly as the selector emitted it: it already
            # canonicalised position order, and the teacher's slots follow that order.
            topk_indices_tnd = topk_indices.reshape(-1, 1, topk_indices.shape[-1])

        # The carrier is a training-only by-product: its gradient is the selector's only edge,
        # and SMLAG's backward writes the teacher on it.  In eval there is no indexer gradient
        # to feed, so the carrier is not threaded through.
        topk_scores_tnd = None
        if self.training and topk_scores is not None:
            topk_scores_tnd = topk_scores.reshape(-1, 1, topk_scores.shape[-1])

        output, _lse = _SparseFlashMLATND.apply(
            q,
            swa_k,
            cmp_k,
            topk_indices_tnd,
            attn_sink.float(),
            metadata,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            topk_scores_tnd,
        )
        return output.reshape(batch_size, seqlen, *output.shape[1:])
