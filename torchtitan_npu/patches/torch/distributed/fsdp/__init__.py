# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file wraps PyTorch composable FSDP internals
# (torch/distributed/fsdp/_fully_shard/_fsdp_param.py and
# _fsdp_collectives.py, torch 2.14.0.dev+git194fcb011f) as the downstream part
# of https://github.com/pytorch/pytorch/issues/156784.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixed-precision and NPU buffer-lifetime adaptations for composable FSDP.

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

Optional AG/RS compute-stream allocation adapts the pinned FSDP communication
interfaces to NPU stream-local memory reuse. This module owns allocation,
copy-out and cleanup together; the upstream issues above concern precision
and input validation, not this allocation policy. AG results retain the native
ownership protocol and return side-stream usage before storage can be reused.
Both options are disabled by default and read once at import time. Backend
requirements and supported paths are documented in
``docs/feature_guides/fsdp_compute_stream_allocation.md``.
"""

from __future__ import annotations

import inspect
import os
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
_ORIGINAL_FOREACH_ALL_GATHER = _fsdp_collectives.foreach_all_gather
_ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT = _fsdp_collectives.foreach_all_gather_copy_out
_ORIGINAL_FINALIZE_BACKWARD = _fsdp_param_group.FSDPParamGroup.finalize_backward
_ORIGINAL_RESET_ITER_STATE = _fsdp_param_group.FSDPParamGroup._reset_iter_state
_ORIGINAL_ALL_GATHER_INPUTS = FSDPParam.all_gather_inputs
# ``property.fget`` is statically ``Callable | None``; this named property has a getter.
_ORIGINAL_ALL_GATHER_INPUTS_FGET = cast("Callable[[FSDPParam], Any]", _ORIGINAL_ALL_GATHER_INPUTS.fget)
_FOREACH_REDUCE_SIGNATURE = inspect.signature(_ORIGINAL_FOREACH_REDUCE)
_FOREACH_ALL_GATHER_SIGNATURE = inspect.signature(_ORIGINAL_FOREACH_ALL_GATHER)
_RS_ALLOC_ON_COMPUTE_STREAM = os.getenv("TORCHTITAN_NPU_FSDP_RS_ALLOC_ON_COMPUTE_STREAM", "0") == "1"
_AG_ALLOC_ON_COMPUTE_STREAM = os.getenv("TORCHTITAN_NPU_FSDP_AG_ALLOC_ON_COMPUTE_STREAM", "0") == "1"


class _ComputeStreamAllocator:
    """Allocate on compute; preserve native copy and communication streams."""

    def __init__(self, comm, device_handle, kind):
        self._comm = comm
        self._device_handle = device_handle
        self._compute_stream = device_handle.current_stream()
        self._kind = kind
        self._all_gather_output = None

    def allocate(self, size, *, dtype, device):
        caller = self._device_handle.current_stream()
        with self._device_handle.stream(self._compute_stream):
            output = self._comm.allocate(size, dtype=dtype, device=device)
            if self._kind == "AG":
                # Keep storage alive until the native result takes ownership.
                self._all_gather_output = output
        if caller != self._compute_stream:
            # Order recycled storage and allocation work before side-stream use.
            caller.wait_stream(self._compute_stream)
            if self._kind != "AG":
                output.record_stream(caller)
        return output

    def __call__(self, *args, **kwargs):
        return self._comm(*args, **kwargs)


class _ComputeStreamAllGatherResult(_fsdp_collectives.AllGatherResult):
    """Preserve the native six-field result and its existing keepalive lifetime."""

    _compute_stream: torch.Stream
    _side_streams: tuple[torch.Stream, ...]

    def _record_side_streams(self, *streams):
        for stream in (*self._side_streams, *streams):
            if stream != self._compute_stream:
                self.all_gather_output.record_stream(stream)

    def _wait_on_compute_stream(self):
        # Unconsumed prefetch: return all side-stream usage before native cleanup
        # drops the result. Async Work may cover an internal HCCL stream too.
        handle = torch.get_device_module(self.all_gather_output.device)
        with handle.stream(self._compute_stream):
            if self.all_gather_event is not None:
                self._compute_stream.wait_event(self.all_gather_event)
            if self.all_gather_work is not None:
                self.all_gather_work.wait()


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
    if _RS_ALLOC_ON_COMPUTE_STREAM:
        bound = _FOREACH_REDUCE_SIGNATURE.bind(fsdp_params, unsharded_grads, *args, **kwargs)
        values = bound.arguments
        values["reduce_scatter_comm"] = _ComputeStreamAllocator(
            values["reduce_scatter_comm"],
            torch.get_device_module(values["device"]),
            "RS",
        )
        return _ORIGINAL_FOREACH_REDUCE(*bound.args, **bound.kwargs)
    return _ORIGINAL_FOREACH_REDUCE(fsdp_params, unsharded_grads, *args, **kwargs)


@wraps(_ORIGINAL_FOREACH_ALL_GATHER)
def _patched_foreach_all_gather(fsdp_params, *args, **kwargs):
    if not _AG_ALLOC_ON_COMPUTE_STREAM:
        return _ORIGINAL_FOREACH_ALL_GATHER(fsdp_params, *args, **kwargs)
    bound = _FOREACH_ALL_GATHER_SIGNATURE.bind(fsdp_params, *args, **kwargs)
    values = bound.arguments
    allocator = _ComputeStreamAllocator(
        values["all_gather_comm"],
        torch.get_device_module(values["device"]),
        "AG",
    )
    values["all_gather_comm"] = allocator
    try:
        native_result = _ORIGINAL_FOREACH_ALL_GATHER(*bound.args, **bound.kwargs)
        if native_result is None:
            raise RuntimeError("Expected a native all-gather result")
        result = _ComputeStreamAllGatherResult(*native_result)
        result._compute_stream = allocator._compute_stream
        result._side_streams = (values["all_gather_copy_in_stream"], values["all_gather_stream"])
        allocator._all_gather_output = None
        return result
    except BaseException:
        # A failure before Result construction must still protect queued work.
        if allocator._all_gather_output is not None:
            for stream in (values["all_gather_copy_in_stream"], values["all_gather_stream"]):
                if stream != allocator._compute_stream:
                    allocator._all_gather_output.record_stream(stream)
        raise


@wraps(_ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT)
def _patched_foreach_all_gather_copy_out(all_gather_result, *args, **kwargs):
    if not isinstance(all_gather_result, _ComputeStreamAllGatherResult):
        return _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT(all_gather_result, *args, **kwargs)
    handle = torch.get_device_module(all_gather_result.all_gather_output.device)
    consumer = handle.current_stream()
    try:
        output = _ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT(all_gather_result, *args, **kwargs)
        # Native copy-out already waits for AG and reads on the compute stream.
        # Only a different consumer needs to return its final read to the owner.
        if consumer != all_gather_result._compute_stream:
            all_gather_result._compute_stream.wait_event(consumer.record_event())
        return output
    except BaseException:
        all_gather_result._record_side_streams(consumer)
        raise


@wraps(_ORIGINAL_FINALIZE_BACKWARD)
def _patched_finalize_backward(self):
    if isinstance(self._all_gather_result, _ComputeStreamAllGatherResult):
        self._all_gather_result._wait_on_compute_stream()
    return _ORIGINAL_FINALIZE_BACKWARD(self)


@wraps(_ORIGINAL_RESET_ITER_STATE)
def _patched_reset_iter_state(self):
    if isinstance(self._all_gather_result, _ComputeStreamAllGatherResult):
        self._all_gather_result._wait_on_compute_stream()
    return _ORIGINAL_RESET_ITER_STATE(self)


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
    _fsdp_collectives.foreach_all_gather = _patched_foreach_all_gather
    _fsdp_param_group.foreach_all_gather = _patched_foreach_all_gather
    _fsdp_collectives.foreach_all_gather_copy_out = _patched_foreach_all_gather_copy_out
    _fsdp_param_group.foreach_all_gather_copy_out = _patched_foreach_all_gather_copy_out
    _fsdp_param_group.FSDPParamGroup.finalize_backward = _patched_finalize_backward
    _fsdp_param_group.FSDPParamGroup._reset_iter_state = _patched_reset_iter_state
    FSDPParam.all_gather_inputs = property(_patched_all_gather_inputs)


apply()
