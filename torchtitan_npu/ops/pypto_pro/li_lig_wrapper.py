# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe LI indices and auxiliary KL-gradient launches for PyPTO-Pro.

The kernels are imported only by the real NPU callbacks. Fake execution retains
symbolic token axes and needs neither the optional compiler nor a device.
The LightningIndexer autograd bridge supplies the SMLAG teacher, already
scaled by the attention softmax scale and the row token count. LIG consumes
that teacher directly and returns dq, dk, dw and the student softmax, not a KL
loss scalar. Integer LI selection and LIG launches do not expose double backward.
"""

import importlib
from functools import cache

import torch


@cache
def _load_operators():
    # Check the operator's mixed SIMT launch ABI before launching the first LI.
    lig = importlib.import_module(
        "torchtitan_npu.ops.pypto_pro.sparse_lightning_indexer_kl_loss_grad.sparse_lightning_indexer_kl_loss_grad"
    )
    li = importlib.import_module("torchtitan_npu.ops.pypto_pro.lightning_indexer.lightning_indexer")
    return li.lightning_indexer_pro, lig.sparse_li_klgrad_pro


def _check_device(q: torch.Tensor) -> None:
    if q.device.type != "npu" or q.device.index is None:
        raise ValueError(f"PyPTO-Pro LI/LIG requires a concrete NPU device, got {q.device}")


def _check_compact_guard(guard_tensors: list[torch.Tensor] | None, guard_versions: list[int] | None) -> None:
    """Check host version counters only in real opaque callbacks, before launch."""
    if guard_tensors is None and guard_versions is None:
        return
    if guard_tensors is None or guard_versions is None or len(guard_tensors) != len(guard_versions):
        raise ValueError("PyPTO-Pro compact-frame tensors and versions must be supplied together")
    try:
        changed = any(tensor._version != version for tensor, version in zip(guard_tensors, guard_versions, strict=True))
    except RuntimeError as error:
        raise ValueError(
            "PyPTO-Pro compact-frame guard needs tracked tensor versions; rebuild outside inference_mode"
        ) from error
    if changed:
        raise ValueError("PyPTO-Pro compact-frame tensors changed in place; rebuild metadata before launching LI/LIG")


@torch.library.custom_op(
    "torchtitan_npu::pypto_pro_lightning_indexer_indices",
    mutates_args=(),
    device_types="npu",
    schema="(Tensor q, Tensor k, Tensor w, SymInt topk, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
    "Tensor cmp_residual_k, Tensor? metadata, str layout_q, str layout_k, SymInt mask_mode, "
    "SymInt cmp_ratio, Tensor[]? guard_tensors=None, SymInt[]? guard_versions=None) -> Tensor",
)
def lightning_indexer_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cmp_residual_k: torch.Tensor,
    metadata: torch.Tensor | None,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    guard_tensors: list[torch.Tensor] | None = None,
    guard_versions: list[int] | None = None,
) -> torch.Tensor:
    _check_compact_guard(guard_tensors, guard_versions)
    _check_device(q)
    with torch.get_device_module("npu").device(q.device):
        li, _ = _load_operators()
        indices, _ = li(
            q.contiguous(),
            k.contiguous(),
            w.contiguous(),
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            return_value=1,
        )
        return indices


@lightning_indexer_indices.register_fake
def _lightning_indexer_indices_fake(
    q,
    k,
    w,
    topk,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    guard_tensors=None,
    guard_versions=None,
):
    return q.new_empty((q.shape[0], k.shape[1], topk), dtype=torch.int32)


@torch.library.custom_op(
    "torchtitan_npu::pypto_pro_sparse_lightning_indexer_kl_loss_grad",
    mutates_args=(),
    device_types="npu",
    schema="(Tensor q, Tensor k, Tensor w, Tensor sparse_indices, Tensor attn_softmax_l1_norm, "
    "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor cmp_residual_k, Tensor? metadata, "
    "str layout_q, str layout_k, SymInt mask_mode, SymInt cmp_ratio, "
    "Tensor[]? guard_tensors=None, SymInt[]? guard_versions=None) -> (Tensor, Tensor, Tensor, Tensor)",
)
def sparse_lightning_indexer_kl_loss_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    sparse_indices: torch.Tensor,
    attn_softmax_l1_norm: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cmp_residual_k: torch.Tensor,
    metadata: torch.Tensor | None,
    layout_q: str,
    layout_k: str,
    mask_mode: int,
    cmp_ratio: int,
    guard_tensors: list[torch.Tensor] | None = None,
    guard_versions: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_compact_guard(guard_tensors, guard_versions)
    _check_device(q)
    with torch.get_device_module("npu").device(q.device):
        _, lig = _load_operators()
        return lig(
            query=q.contiguous(),
            key=k.contiguous(),
            weights=w.contiguous(),
            sparse_indices=sparse_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )


@sparse_lightning_indexer_kl_loss_grad.register_fake
def _sparse_lightning_indexer_kl_loss_grad_fake(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    guard_tensors=None,
    guard_versions=None,
):
    return (
        q.new_empty(q.shape, dtype=torch.bfloat16),
        k.new_empty(k.shape, dtype=torch.float32),
        w.new_empty(w.shape, dtype=torch.float32),
        attn_softmax_l1_norm.new_empty(attn_softmax_l1_norm.shape, dtype=torch.float32),
    )
