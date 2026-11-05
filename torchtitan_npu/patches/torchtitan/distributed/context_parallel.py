# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4617
#
# #4617 made the load balancer a configurable object built from the batch it is about to
# shard, and moved `prepare_cp_metadata` onto the backend.  0.3.0's `cp_shard` takes a
# balancer *name* and builds it internally, so the pieces below -- the balancer object, the
# registry that resolves the name to one, and the permutation-driven ``shard_tensors`` --
# exist to give this pin #4617's split.  The whole module sunsets with the pin bump, which
# brings the flat ``context_parallel`` module that has no ``cp_shard`` to rebind: nothing
# here is expected to survive it, so nothing here defends against it.

"""The CP shard, with upstream's balancer abstraction and a registry for this repository's layouts.

The pinned ``cp_shard`` shards by sharding alone, or by a load-balancer permutation the framework
owns (``headtail``, ``ptrr``).  This module adds the pieces #4617 gave the shard -- an abstract
:class:`ContextParallelLoadBalancer` that states a permutation, a registry that resolves a layout
name to one, and :func:`shard_tensors` -- and leaves the pinned path below unchanged for everything
that is not registered.

The split is deliberate.  A layout is a property of a model's attention contract, so the model
package defines and registers it; the mechanism is generic, so it lives here, with no model name and
no model import.  This file touches upstream only at the names it must: the entry point it replaces
and the pinned fallback it preserves.
"""

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import _context_parallel_shard
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _LoadBalancer,
)
from torchtitan.config import Configurable
from torchtitan.distributed.context_parallel.api import cp_shard as original_cp_shard
from torchtitan.models.common.attention import AttentionMasksType

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


def build_load_balancer(
    name: str | None,
    *,
    seq_len: int,
    cp_size: int,
    attention_metadata: Any,
) -> ContextParallelLoadBalancer | None:
    """This repository's layout named ``name``, built from the batch it is about to shard.

    ``None`` means the name is not one of this repository's layouts -- a pinned balancer
    (``headtail``, ``ptrr``) or no balancer at all -- and the caller leaves the shard to the
    pinned path.  ``attention_metadata`` is one entry of the batch's metadata map -- the layer the
    layout addresses -- where the row's boundaries are, as upstream's balancers read them; the
    degree is a parameter because this pin has no ambient CP mesh.
    """
    balancer_cls = LAYOUT_BALANCERS.get(name) if name else None
    if balancer_cls is None:
        return None
    return balancer_cls.Config().build(seq_len=seq_len, cp_size=cp_size, attention_metadata=attention_metadata)


@functools.wraps(original_cp_shard)
def patched_cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
    ptrr_mask_key: str | None = None,
    *,
    permutation: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    """Shard inputs for CP by a permutation the caller built, or by the pinned path.

    Upstream #4617 builds the balancer in ``Decoder._cp_shard`` and hands the shard its
    permutation; this entry keeps that split.  ``permutation=None`` means the caller has no
    balancer of its own, so the pinned shard -- which builds ``headtail``/``ptrr`` from the
    length and the masks -- runs exactly as upstream's shard does.
    """
    if permutation is None:
        if load_balancer_type in LAYOUT_BALANCERS:
            raise ValueError(
                f"the {load_balancer_type!r} layout is built from the batch's document boundaries, "
                "but the batch carries no attention metadata to read them from."
            )
        return original_cp_shard(
            cp_mesh, inputs, attention_masks, load_balancer_type, input_seq_dim, ptrr_mask_key=ptrr_mask_key
        )
    if attention_masks is not None:
        raise NotImplementedError(
            "a permuted CP shard moves the inputs only: the attention masks of this repository's "
            "stacks are varlen metadata built by the model from the row's boundaries, not "
            "BlockMask objects the shard could slice."
        )
    return shard_tensors(cp_mesh, inputs, permutation=permutation, input_seq_dim=input_seq_dim), attention_masks


def apply() -> None:
    # Every caller of the pinned shard -- the models here, and Flux upstream -- imports it from the
    # package path, so the package's re-exported name is the one to rebind.  The defining ``api``
    # module is reached only by this module's own ``original_cp_shard``, which must stay original.
    import torchtitan.distributed.context_parallel as context_parallel

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

    logger.info("[PATCH] torchtitan.distributed.context_parallel.cp_shard -> patched_cp_shard")
    context_parallel.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]


apply()
