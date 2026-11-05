# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU QuantLightningIndexer module behavior."""

__all__ = [
    "QuantizedLightningIndexerInputs",
    "quantized_lightning_indexer_forward",
]

from dataclasses import dataclass

import torch

from torchao_npu.ops.li_ops import quant_lightning_indexer


@dataclass(frozen=True, slots=True)
class QuantizedLightningIndexerInputs:
    """Host-normalized inputs consumed by the quantized LI kernel."""

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    topk: int
    metadata: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cmp_residual_k: torch.Tensor
    num_global_queries: torch.Tensor
    """The KL term's denominator: the whole row's query-token count at every degree.

    Every rank divides by the row's count, not by its own chunk's: the parameters' gradients are
    summed across ranks, so a rank-local divisor would scale them by the parallel degree.
    """
    seqused_q: torch.Tensor
    seqused_k: torch.Tensor
    """The per-document counts each side is masked by: this rank's reach under CP.

    Required, not optional: the kernels mask right-down causal against them, and the
    zero-filled default a missing value used to fall back to lets a rank select keys
    outside its own frame.
    """
    backward_metadata: torch.Tensor | None = None


class _QuantLightningIndexerTND(torch.autograd.Function):
    """Keep quantized selection attached to the BF16 SLIKG teacher edge."""

    @staticmethod
    def forward(ctx, query, key, weights, inputs, config):  # pyrefly: ignore [bad-override]
        ctx.set_materialize_grads(False)
        indices, _ = quant_lightning_indexer(
            query,
            key,
            weights,
            config.query_config,
            config.key_config,
            topk=inputs.topk,
            quant_mode=config.quant_mode,
            layout_q=config.layout_q,
            layout_k=config.layout_k,
            mask_mode=config.mask_mode,
            cmp_ratio=config.cmp_ratio,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            seqused_q=inputs.seqused_q,
            seqused_k=inputs.seqused_k,
            cmp_residual_k=inputs.cmp_residual_k,
            metadata=inputs.metadata,
        )
        # The saved slots must match the order SMLAG uses for its teacher.
        indices = indices.sort(dim=-1, descending=True).values
        scores = indices.new_empty(indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(query, key, weights, indices)
        ctx.inputs = inputs
        ctx.config = config
        ctx.mark_non_differentiable(indices)
        return indices, scores

    @staticmethod
    def backward(ctx, grad_indices, teacher):  # pyrefly: ignore [bad-override]
        del grad_indices
        if teacher is None:
            raise RuntimeError("Quantized LI backward is missing the teacher gradient from sparse attention")
        query, key, weights, indices = ctx.saved_tensors
        inputs, config = ctx.inputs, ctx.config
        if inputs.backward_metadata is None:
            raise RuntimeError("Quantized LI training requires the SLIKG backward record")
        # SMLAG already masked padding and applied softmax_scale. Match the BF16 selector's
        # normalization by the row's query-token count: under context parallelism the rank's own
        # ``cu_seqlens_q[-1]`` is one chunk of it, and the ranks' gradients are summed.
        teacher = teacher / inputs.num_global_queries.to(torch.float32)
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=query,
            k=key,
            w=weights,
            sparse_indices=indices,
            attn_softmax_l1_norm=teacher,
            metadata=inputs.backward_metadata,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=inputs.cu_seqlens_k,
            seqused_q=inputs.seqused_q,
            seqused_k=inputs.seqused_k,
            cmp_residual_k=inputs.cmp_residual_k,
            layout_q=config.layout_q,
            layout_k=config.layout_k,
            mask_mode=config.mask_mode,
            cmp_ratio=config.cmp_ratio,
        )
        return dq.to(query.dtype), dk.to(key.dtype), dw.to(weights.dtype), None, None


def quantized_lightning_indexer_forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
    """Run quantized LI with the fused attention's teacher gradient carrier."""

    config = self._torchao_npu_module_swap_config
    if config.input_adapter is None:
        raise RuntimeError("Quantized LI requires a host input adapter")
    inputs = config.input_adapter(self, idx_q, idx_k, idx_w, attention_metadata)
    indices, scores = _QuantLightningIndexerTND.apply(inputs.query, inputs.key, inputs.weights, inputs, config)
    batch, seqlen = idx_q.shape[:2]
    return indices.reshape(batch, seqlen, -1), scores.reshape(batch, seqlen, -1)
