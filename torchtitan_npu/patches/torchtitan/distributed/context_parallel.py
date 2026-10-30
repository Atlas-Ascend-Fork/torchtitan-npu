# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4617
#
# #4617 made the load balancer a configurable object built from the batch it is about to
# shard, and moved `prepare_cp_metadata` onto the backend.  0.3.0's `cp_shard` takes a
# balancer *name* and builds it internally from the sequence length alone, so a
# boundary-driven layout has nothing to read: the ``positions`` keyword below is #4617's
# balancer input in this pin's shape.  It disappears with the pin, not with this repository.

"""The CP shard, with upstream's balancer abstraction and a registry for this repository's layouts.

The pinned ``cp_shard`` shards by sharding alone, or by a load-balancer permutation the framework
owns (``headtail``, ``ptrr``).  This module adds the pieces #4617 gave the shard -- an abstract
:class:`ContextParallelLoadBalancer` that states a permutation, a registry that resolves a layout
name to one, :func:`shard_tensors`, and the boundary derivation a packed row needs -- and leaves the
pinned path below unchanged for everything that is not registered.

The split is deliberate.  A layout is a property of a model's attention contract, so the model
package defines and registers it; the mechanism is generic, so it lives here, with no model name and
no model import.  This file touches upstream only at the names it must: the entry point it replaces
and the pinned fallback it preserves.
"""

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from torchtitan.config import Configurable
from torchtitan.distributed.context_parallel.api import cp_shard as original_cp_shard
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)

logger = logging.getLogger(__name__)


class ContextParallelLoadBalancer(Configurable, ABC):
    """Generate a context-parallel token permutation for one batch.

    Upstream #4617's abstraction: a balancer is built from the batch it is about to shard and
    states the global token order the shard cuts by.  The two framework balancers (``headtail``,
    ``ptrr``) balance work; a layout of this repository states an order its attention contract
    requires.  Both answer the same question, which is why they share this base.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for a context-parallel load balancer."""

    @abstractmethod
    def generate_permutation(self) -> torch.Tensor:
        """Generate the global token permutation (``[1, num_tokens]``) for this batch."""
        raise NotImplementedError


LAYOUT_BALANCERS: dict[str, type[ContextParallelLoadBalancer]] = {}
"""The CP layouts this repository adds, by their ``--parallelism.context-parallel-load-balancer``
name.

The model package that owns one registers it at import time (:func:`register_layout`), because a
layout is a property of that model's attention contract, not of the shard; this registry is the
shard's half of the seam and names no model.  On this pin the CLI field is a string, so the name
stands in for the config object upstream resolves.
"""


def register_layout(name: str, balancer: type[ContextParallelLoadBalancer]) -> None:
    """Register ``balancer`` under the CLI name ``name``.

    Registering the same class twice is a no-op (a module imported twice, or a test that reloads
    it); a different class under a taken name is a programming error.
    """
    if not issubclass(balancer, ContextParallelLoadBalancer):
        raise TypeError(f"{balancer.__qualname__} is not a ContextParallelLoadBalancer")
    previous = LAYOUT_BALANCERS.get(name)
    if previous is not None and previous is not balancer:
        raise ValueError(f"the CP layout {name!r} is already registered by {previous.__qualname__}")
    LAYOUT_BALANCERS[name] = balancer


def check_layout_supported(
    stack: str,
    load_balancer_type: str | None,
    *,
    supported: tuple[str, ...] | None = None,
) -> None:
    """Refuse a layout this stack's rank-local metadata does not describe.

    Upstream #4617's support matrix, transposed: there it maps a balancer class to the inner
    attentions it can drive, because every inner attention is a class there; here a stack states
    the layouts it can consume.  The check runs at the metadata boundary, before a layer runs, and
    catches a run sharded by a layout whose frames this stack cannot build.
    """
    allowed = supported if supported is not None else tuple(sorted(LAYOUT_BALANCERS))
    if load_balancer_type not in allowed:
        raise ValueError(
            f"{stack} shards by one of {sorted(allowed)}, but "
            f"--parallelism.context-parallel-load-balancer is {load_balancer_type!r}: run with one "
            "of them."
        )


def document_starts(positions: torch.Tensor) -> torch.Tensor:
    """The row's document boundaries (``n_documents + 1``), read off its own positions.

    The stack's positions restart at every document, so a row's document start is its own index
    minus its position; the distinct starts are the boundaries, and the row's end closes the last
    one.  Device arithmetic throughout -- no ``nonzero``, no ``tolist`` -- because the CP shard
    runs this on every microbatch and a round-trip there would drain the pipeline.

    This is the one place the layout learns where the documents are when only the positions are
    at hand (the shard runs before any metadata consumer); it must agree with the metadata's
    ``cu_seqlens`` -- the packing grid the positions were built from -- which ``tests`` pins.
    """
    flat = positions.reshape(-1)
    rows = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    # Non-decreasing, one distinct value per document, so ``unique_consecutive`` is the boundary
    # set (a full ``unique`` would sort the row again for nothing).
    doc_start = rows - flat.to(torch.int64)
    return torch.cat((torch.unique_consecutive(doc_start), rows.new_tensor([rows.numel()])))


class _FixedPermutation(_LoadBalancer):
    """Replay a permutation a balancer already generated.

    The pinned shard takes a balancer and not a permutation, so :func:`shard_tensors` hands it
    this two-line adapter; upstream's ``spmd.shard`` takes the permutation directly.
    """

    def __init__(self, permutation: torch.Tensor) -> None:
        self._permutation = permutation

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        if restore:
            return torch.argsort(self._permutation, dim=-1)
        return self._permutation


def shard_tensors(
    cp_mesh: DeviceMesh,
    tensors: tuple[torch.Tensor, ...],
    *,
    permutation: torch.Tensor | None,
    input_seq_dim: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Permute and shard the tensors over the CP axis (upstream #4617's ``shard_tensors``).

    ``permutation`` is the global token order applied before slicing; ``None`` shards the row
    contiguously.  Upstream shards a declared ``input_dict``; this pin's decoder hands the shard an
    explicit tuple and has no sharding table to declare, so the tensors and one sequence dimension
    stand in for it.
    """
    sharded = _context_parallel_shard(
        mesh=cp_mesh,
        buffers=tuple(tensors),
        seq_dims=tuple(input_seq_dim for _ in tensors),
        load_balancer=None if permutation is None else _FixedPermutation(permutation),
    )
    # ``_context_parallel_shard`` is typed for the union upstream shards; this seam hands it
    # tensors only, so the tuple is one of tensors.
    return cast("tuple[torch.Tensor, ...]", tuple(sharded))


def shard_for_layout(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks,
    load_balancer_type: str | None,
    *,
    input_seq_dim: int = 1,
    positions: torch.Tensor | None = None,
):
    """Shard by one of this repository's layouts, or return ``None`` when it is not one.

    The balancer is built from the batch (:class:`ContextParallelLoadBalancer`), states the
    permutation, and the shard applies it -- #4617's two steps.  ``positions`` is what a
    boundary-driven balancer is built from on this pin; upstream hands the balancer the batch's
    attention metadata instead.

    The rank-local metadata of these layouts is the model's own transform -- it reads the row's
    boundaries, not the sharded rows -- so nothing is returned in place of the masks, and the
    caller keeps them as they are.
    """
    balancer_cls = LAYOUT_BALANCERS.get(load_balancer_type) if load_balancer_type else None
    if balancer_cls is None:
        return None
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
    balancer = balancer_cls.Config().build(
        seq_len=inputs[0].size(input_seq_dim),
        cu_seqlens=document_starts(positions),
        cp_size=cp_mesh.size(),
    )
    sharded = shard_tensors(
        cp_mesh,
        tuple(inputs),
        permutation=balancer.generate_permutation(),
        input_seq_dim=input_seq_dim,
    )
    return sharded, attention_masks


def _varlen_from_masks(masks):
    """Return the shared VarlenMetadata behind a bare mask or an all-varlen dict."""
    if isinstance(masks, VarlenMetadata):
        return masks
    if isinstance(masks, dict):
        values = [value for value in masks.values() if value is not None]
        if values and all(isinstance(value, VarlenMetadata) for value in values):
            return values[0]
    return None


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
    """Shard inputs for CP: a registered layout first, the pinned path otherwise.

    ``positions`` is the row's own positions, consumed only by a registered layout (see
    :func:`shard_for_layout`); the pinned layouts are functions of the length alone, so a stack
    that carries no positions shards as it did before these layouts existed.
    """
    sharded = shard_for_layout(
        cp_mesh,
        inputs,
        attention_masks,
        load_balancer_type,
        input_seq_dim=input_seq_dim,
        positions=positions,
    )
    if sharded is not None:
        return sharded
    if load_balancer_type not in (None, "headtail", "ptrr"):
        raise ValueError(
            f"no CP layout is registered under {load_balancer_type!r}: the model package that "
            "defines it must be imported (it calls register_layout), and the pinned shard only "
            "knows None, 'headtail' and 'ptrr'."
        )

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
