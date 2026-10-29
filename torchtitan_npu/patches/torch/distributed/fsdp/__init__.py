# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file wraps PyTorch composable FSDP internals
# (torch/distributed/fsdp/_fully_shard/_fsdp_param.py and
# _fsdp_collectives.py, torch 2.14.0.dev+git194fcb011f) as the downstream part
# of https://github.com/pytorch/pytorch/issues/156784.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-parameter mixed-precision policies for composable FSDP.

A ``MixedPrecisionPolicy`` holds one dtype for a whole parameter group, but
DeepSeek-V4 keeps a few parameters in FP32 while the rest of the model computes
in the training dtype.  :func:`policy_overrides` installs an exact-FQN policy
mapping for the duration of a block, and every policy built inside that block
carries the mapping, so it reaches ``fully_shard`` through the ordinary
``mp_policy`` argument.  ``init_dtype_attrs`` then resolves each parameter's own
policy during lazy initialization, and ``foreach_reduce`` unifies the resulting
mixed gradients before the native single-dtype reduce-scatter.

The ``all_gather_inputs`` relaxation lets quantized tensor subclasses return
multiple all-gather inputs (data + scales) under uneven sharding: the patched
check requires padded dim0 per input instead of the full padded shape.  Native
torch rejects those payloads; tracked upstream as
https://github.com/pytorch/pytorch/issues/202040, with the same relaxation
proposed in https://github.com/pytorch/pytorch/pull/202101.  Retire this patch
once a released torch carries the fix.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
    _to_dtype_if_needed,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_ORIGINAL_POLICY_INIT = MixedPrecisionPolicy.__init__
_ORIGINAL_INIT_DTYPE_ATTRS = FSDPParam.init_dtype_attrs
_ORIGINAL_FOREACH_REDUCE = _fsdp_collectives.foreach_reduce
_ORIGINAL_ALL_GATHER_INPUTS = FSDPParam.all_gather_inputs
# ``property.fget`` is statically ``Callable | None``; this named property has a getter.
_ORIGINAL_ALL_GATHER_INPUTS_FGET = cast("Callable[[FSDPParam], Any]", _ORIGINAL_ALL_GATHER_INPUTS.fget)


@contextmanager
def policy_overrides(overrides: Mapping[str, MixedPrecisionPolicy]) -> Iterator[None]:
    """Attach ``overrides`` to every policy built inside the block."""

    def init(self, *args, **kwargs) -> None:
        _ORIGINAL_POLICY_INIT(self, *args, **kwargs)
        object.__setattr__(self, "overrides", dict(overrides))

    previous, MixedPrecisionPolicy.__init__ = MixedPrecisionPolicy.__init__, init
    try:
        yield
    finally:
        MixedPrecisionPolicy.__init__ = previous


@wraps(_ORIGINAL_INIT_DTYPE_ATTRS)
def _patched_init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy) -> None:
    overrides = getattr(mp_policy, "overrides", None)
    if not overrides:
        return _ORIGINAL_INIT_DTYPE_ATTRS(self, mp_policy)

    fqn = self._param_fqn
    if "_checkpoint_wrapped_module" in fqn:
        # Activation checkpointing module is prefixed with
        # ``_checkpoint_wrapped_module``
        fqn = ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")
    policy = overrides.get(fqn) or mp_policy
    self.mp_policy = policy  # extension hooks read the parameter's own policy
    _ORIGINAL_INIT_DTYPE_ATTRS(self, policy)
    # Use the group's (default) reduce dtype so all parameters in the group
    # share a single reduce-scatter dtype regardless of per-parameter override.
    self.reduce_dtype = mp_policy.reduce_dtype


@wraps(_ORIGINAL_FOREACH_REDUCE)
@torch.no_grad()
def _patched_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
    # Backward computes in each parameter's own dtype, so the gradients are mixed
    # (FP32 for the overridden families, BF16 for the rest) while native
    # ``foreach_reduce`` asserts a single dtype: unify them first.
    reduce_dtype = fsdp_params[0].reduce_dtype
    if reduce_dtype is not None and len({grad.dtype for grad in unsharded_grads}) > 1:
        unsharded_grads[:] = [grad if grad.dtype == reduce_dtype else grad.to(reduce_dtype) for grad in unsharded_grads]
    return _ORIGINAL_FOREACH_REDUCE(fsdp_params, unsharded_grads, *args, **kwargs)


def _check_all_gather_inputs(fsdp_param: FSDPParam, all_gather_inputs) -> None:
    """Uneven-sharding contract: multi-input payloads (data + scales) are
    checked for padded dim0 only, keeping all-gather buffers equal-sized.

    Stands in for the native check pending upstream pytorch/pytorch#202040
    (fix proposed in pytorch/pytorch#202101).
    """
    local_tensor = fsdp_param._sharded_local_tensor
    padded = fsdp_param.padded_sharded_param_size
    if local_tensor.size() == padded:
        return  # even sharding (or already padded): nothing to check
    if len(all_gather_inputs) == 1:
        ok = all_gather_inputs[0].size() == padded
    else:
        ok = all(t.size(0) == padded[0] for t in all_gather_inputs)
    if not ok:
        # Raised only on padding ranks; elsewhere it manifests as an HCCL timeout.
        raise AssertionError(
            "When a parameter is unevenly sharded by FSDP "
            f"(orig size={fsdp_param._orig_size}, FSDP world size={fsdp_param.mesh_info.mesh.size()}), "
            "fsdp_pre_all_gather must return all-gather inputs padded to the "
            f"padded sharded size {padded} (single input: full shape; multiple "
            f"inputs: dim0) but got {[t.size() for t in all_gather_inputs]}"
        )


@wraps(_ORIGINAL_ALL_GATHER_INPUTS_FGET)
def _patched_all_gather_inputs(self: FSDPParam):
    # Mirrors the native property (torch 2.14.0.dev); uneven-sharding check delegated below.
    self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
    if self.sharded_state == ShardedState.SHARDED:
        if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
            sharded_local_tensor = self._sharded_local_tensor
            if self.offload_to_cpu:
                sharded_local_tensor = sharded_local_tensor.to(self.device, non_blocking=True)
            pre_all_gather_signature = inspect.signature(
                # pyrefly: ignore [missing-attribute]
                sharded_local_tensor.fsdp_pre_all_gather
            )
            num_fn_params = len(pre_all_gather_signature.parameters)
            # Old signature only passes mesh; keep for BC for now
            if num_fn_params not in (1, 5):
                raise AssertionError(
                    f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                    "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                    "outer_size: torch.Size, outer_stride: tuple[int, ...], "
                    "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
                )
            if num_fn_params == 1:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                    # pyrefly: ignore [missing-attribute]
                ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh_from_root)
            else:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                    # pyrefly: ignore [missing-attribute]
                ) = sharded_local_tensor.fsdp_pre_all_gather(
                    self.shard_mesh_from_root,
                    self._orig_size,
                    self._contiguous_orig_stride,
                    self._module_info.module,
                    self.mp_policy,
                )
                _check_all_gather_inputs(self, all_gather_inputs)
            self._extensions_data.all_gather_input_sizes = [t.size() for t in all_gather_inputs]
            return [t.view(-1) for t in all_gather_inputs]
        sharded_param_data = self._sharded_param_data
        if self.offload_to_cpu:
            sharded_param_data = sharded_param_data.to(self.device, non_blocking=True)
        return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
    return _ORIGINAL_ALL_GATHER_INPUTS_FGET(self)


def apply() -> None:
    FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
    # ``_fsdp_param_group`` binds ``foreach_reduce`` by value at import time,
    # so both references must be patched.
    _fsdp_collectives.foreach_reduce = _patched_foreach_reduce
    _fsdp_param_group.foreach_reduce = _patched_foreach_reduce
    FSDPParam.all_gather_inputs = property(_patched_all_gather_inputs)


apply()
