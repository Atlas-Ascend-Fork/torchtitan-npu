# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430

"""The CP shard, plus the doc-chunk layout as one of its load balancers.

The pinned ``cp_shard`` shards by sharding alone, or by a load-balancer permutation the framework
owns (``headtail``, ``ptrr``).  This repository's stacks shard by a third one: the row's
**doc-chunk** order -- rank ``r``'s window is chunk ``r`` of every document
(``doc_chunk_permutation``) -- which the DeepSeek stacks' rank-local attention frames describe.

It is written the way the framework's own balancers are: ``cp_shard`` builds it from the row it is
about to shard (``_HeadTailLoadBalancer`` needs the length, this one the document boundaries) and
the balancer computes its own ``_generate_indices``.  Nothing upstream of the shard precomputes the
permutation, which is what keeps the move free: the framework permutes the replicated row *before*
slicing it, so a rank receives its chunk rows without a collective, where an exchange between
slab-sharded ranks would need one.
"""

import functools
import logging
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
)
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _LoadBalancer,
)
from torchtitan.distributed.context_parallel.api import cp_shard as original_cp_shard
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata

from torchtitan_npu.models.common.restore import (
    DOC_CHUNK_LAYOUT,
    doc_chunk_permutation,
    document_starts,
)
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)

logger = logging.getLogger(__name__)


def _varlen_from_masks(masks):
    """Return the shared VarlenMetadata behind a bare mask or an all-varlen dict."""
    if isinstance(masks, VarlenMetadata):
        return masks
    if isinstance(masks, dict):
        values = [value for value in masks.values() if value is not None]
        if values and all(isinstance(value, VarlenMetadata) for value in values):
            return values[0]
    return None


class DocChunkLoadBalancer(_LoadBalancer):
    """The doc-chunk layout, stated as a load balancer (``DOC_CHUNK_LAYOUT``).

    Rank ``r``'s window of the rearranged row is chunk ``r`` of every document, in document order
    -- rank-major, as the sharding contract requires -- so the layout *is* the balancer's
    permutation.  It is a function of the row's document boundaries, the degree and the row's
    length alone: device arithmetic, zero communication, nothing to agree on across ranks.

    The constructor takes the **row** the shard is about to slice (that is what a shard knows);
    the layout helpers speak in slab widths, so the slab is derived here, once.

    ``restore`` is the inverse permutation, which is what the framework's
    ``_generate_indices(restore=True)`` promises (and what the unshard path asks for).
    """

    def __init__(self, cu_seqlens: torch.Tensor, row_len: int, cp_size: int) -> None:
        if row_len % cp_size:
            raise ValueError(f"the row ({row_len} tokens) does not split into {cp_size} slabs.")
        self.cu_seqlens = cu_seqlens
        self.slab_len = row_len // cp_size
        self.cp_size = cp_size

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        permutation = doc_chunk_permutation(self.cu_seqlens, self.cp_size, self.slab_len)
        if restore:
            return torch.argsort(permutation, dim=-1)
        return permutation


_LAYOUT_BALANCERS = {DOC_CHUNK_LAYOUT: DocChunkLoadBalancer}
"""The layouts this repository states as load balancers, by their CLI name."""


def layout_needs_positions(load_balancer_type: str | None) -> bool:
    """Whether this layout is derived from where the documents start.

    A boundary-driven layout is a function of the row's document boundaries, so a caller has to
    hand it the positions; the pinned ones (torch's two, and "no balancer") are functions of the
    length alone, and a model that carries no positions shards exactly as it did before these
    layouts existed.
    """
    return bool(load_balancer_type) and load_balancer_type in _LAYOUT_BALANCERS


@functools.wraps(original_cp_shard)
def patched_cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
    ptrr_mask_key: str | None = None,
    *,
    positions: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | CPVarlenMetadata | None]:
    """Shard inputs for CP, with this repository's layouts beside the framework's.

    ``positions`` is the row's own positions, and it is what a boundary-driven layout is built
    from: torch's two balancers are functions of the length alone, while ``DOC_CHUNK_LAYOUT`` is a
    function of where the documents start.  The shard derives the boundaries, builds the balancer
    and lets it state the permutation -- the caller passes the row, not a layout.

    The rank-local metadata of these layouts is the model's own transform -- it reads the row's
    boundaries, not the sharded rows -- so nothing is returned in place of the masks.

    Without a repository layout the pinned behaviour is unchanged: shard the inputs and, for a
    varlen stack, build the rank-local varlen metadata after sharding.
    """
    balancer_cls = _LAYOUT_BALANCERS.get(load_balancer_type) if load_balancer_type else None
    if balancer_cls is not None:
        if attention_masks is not None:
            raise NotImplementedError(
                "a boundary-driven CP layout shards tensors only: the attention masks of this "
                "repository's stacks are varlen metadata built by the model from the row's "
                "boundaries, not BlockMask objects the shard could slice."
            )
        if positions is None:
            raise ValueError(
                f"the {load_balancer_type!r} layout is a function of the row's document boundaries: "
                "pass the row's positions to cp_shard."
            )
        sharded = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=tuple(inputs),
            seq_dims=tuple(input_seq_dim for _ in inputs),
            load_balancer=balancer_cls(
                document_starts(positions),
                inputs[0].size(input_seq_dim),
                cp_mesh.size(),
            ),
        )
        # ``_context_parallel_shard`` is typed for the union upstream shards; a boundary-driven
        # layout is handed tensors only, so the tuple is one of tensors.
        return cast("tuple[torch.Tensor, ...]", tuple(sharded)), attention_masks

    varlen_metadata = _varlen_from_masks(attention_masks)
    is_varlen = varlen_metadata is not None
    batch_size = inputs[0].size(0)
    seq_len = inputs[0].size(input_seq_dim)

    inputs, output_masks = original_cp_shard(
        cp_mesh,
        inputs,
        None if is_varlen else attention_masks,
        load_balancer_type,
        input_seq_dim,
        ptrr_mask_key=ptrr_mask_key,
    )

    if varlen_metadata is not None:
        assert load_balancer_type in (
            None,
            "headtail",
        ), f"varlen only support headtail as load balancer, got ({load_balancer_type})"

        cp_metadata = CPVarlenMetadata.from_global(
            varlen_metadata,
            cp_mesh,
            batch_size,
            seq_len,
            load_balancer=(
                _HeadTailLoadBalancer(seq_len, cp_mesh.size(0), cp_mesh.device_type)
                if load_balancer_type == "headtail"
                else None
            ),
        )
        if isinstance(attention_masks, dict):
            # Hybrid attention stacks carry one shared varlen metadata under
            # multiple mask keys; keep the dict shape so each block can still
            # select its mask by key after CP sharding.
            output_masks = {
                key: cp_metadata if isinstance(value, VarlenMetadata) else value
                for key, value in attention_masks.items()
            }
        else:
            output_masks = cp_metadata

    return inputs, output_masks  # pyrefly: ignore [bad-return]


def apply() -> None:
    try:
        import torchtitan.distributed.context_parallel.api as context_parallel_api
    except ImportError:
        import torchtitan.distributed.context_parallel as context_parallel_api

    # Qwen3.5 vision masks are created after NPU initialization.  Torch's
    # compiled helper starts a worker process, which cannot reinitialize NPU;
    # use the eager helper for this CP path.
    try:
        import torch.distributed.tensor.experimental._context_parallel._attention as cp_attention
        from torch.nn.attention.flex_attention import create_block_mask

        def _eager_create_block_mask(*args, **kwargs):
            kwargs.pop("separate_full_blocks", None)
            kwargs["_compile"] = False
            return create_block_mask(*args, **kwargs)

        cp_attention._compiled_create_block_mask = _eager_create_block_mask  # pyrefly: ignore [bad-assignment]
    except (ImportError, AttributeError):
        logger.debug("CP eager block-mask workaround is unavailable", exc_info=True)

    logger.info("[PATCH] torchtitan.distributed.context_parallel.api.cp_shard -> patched_cp_shard")
    context_parallel_api.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]

    try:
        import torchtitan.distributed.context_parallel as context_parallel_root

        context_parallel_root.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]
    except ImportError:
        pass


apply()
