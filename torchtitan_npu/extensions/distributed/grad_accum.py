# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2006 Idiap Research Institute (Samy Bengio)
# Copyright (c) 2013 the respective contributors
# Licensed under the BSD 3-Clause License (see LICENSE for details).

"""NPU-side accumulation for CPU-offloaded FSDP gradients.

FSDP owns the canonical gradient object and its host allocation. On the
second and later microbatch, this module only changes where the addition is
performed: a bounded scratch slice is filled from the pinned host gradient,
updated with the NPU reduce-scatter result, and copied back to that same host
slice. On the first microbatch, eligible pinned gradient destinations are
collected into one batched D2H dispatch. DTensor metadata remains unchanged.

Patching strategy
-----------------
Instead of rewriting upstream source text, four thin wrappers replace the
upstream entry points and *delegate to the untouched originals* whenever CPU
offload is disabled or not applicable, so non-offload training runs pristine
upstream code:

* ``FSDPParamGroup.unshard`` / ``FSDPParamGroup.wait_for_unshard``: singleton
  (world-size 1) groups whose parameters are CPU-offloaded prefetch their
  parameters to the NPU through the all-gather copy-in stream, replacing
  upstream's synchronous copy in ``wait_for_unshard``.
* ``foreach_reduce``: reduced shards that must be accumulated into a
  CPU-canonical gradient are added on the NPU (see ``accumulate_cpu_grad``)
  instead of through upstream's blocking D2H copy plus host add. The
  eligible first-microbatch pinned destinations in a reduce-scatter group are
  also collected into one ``torch._foreach_copy_`` D2H dispatch; ineligible
  destinations keep the per-parameter path.
* ``foreach_all_gather``: CPU-offloaded groups (any world size) stage their
  parameters in direct-copy and storage-to-compute-cast subsets, bypassing the
  per-parameter ``.to(device)`` inside ``FSDPParam.all_gather_inputs`` and the
  D2D second hop through ``all_gather_copy_in``. With a uniform destination
  dtype, a direct-only group uses one batched H2D dispatch, a cast-only group
  uses one H2D into NPU scratch plus one D2D cast, and a direct+cast group uses
  all three serial dispatches. A raw-byte mixed-dtype destination adds one D2D
  byte-pack dispatch when casts are present.

Each wrapper carries a marker attribute so installation is idempotent, keeps
``__wrapped__`` pointing at the original for debugging, and the upstream
signatures are validated at install time so incompatible PyTorch releases
fail loudly instead of silently misbehaving.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from inspect import signature
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch import Tensor
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_device_handle,
    _get_gradient_divide_factors,
    _get_param_all_gather_inputs,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    TrainingState,
    _disable_functorch_if_active,
    _get_dim0_padded_size,
    _to_dtype_if_needed,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from torch.distributed.tensor import DTensor
from torch.profiler import record_function

from torchtitan_npu.extensions.cpu_offload.runtime import local_tensor

_PATCH_MARKER = "_torchtitan_npu_fsdp_grad_accum_v1"
_CHUNK_BYTES = 64 * 1024 * 1024  # NPU bounce-buffer granularity for grad accumulation
_PREFETCH_MARKER = "_torchtitan_npu_cpu_offload_prefetch_v2"

_LOGGER = logging.getLogger(__name__)
_BATCHED_COPY_IN_LOGGED = False
_CPU_OFFLOAD_ENABLED = False


@dataclass(slots=True)
class _AccumSlot:
    buffer: Tensor
    release: Any | None = None


class _AccumPipeline:
    """Bounded H2D/compute/D2H pipeline for one NPU device."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)  # pyrefly: ignore [read-only]
        rank = os.environ.get("RANK", "0")
        from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging

        self.staging = CpuStaging(self.device, owner=f"grad-accum-rank{rank}")
        self.compute_stream = torch_npu.npu.Stream(device=self.device)
        self._slots: dict[tuple[torch.dtype, int], tuple[_AccumSlot, _AccumSlot]] = {}
        self._next_slot: dict[tuple[torch.dtype, int], int] = {}

    def slots(
        self, dtype: torch.dtype, chunk_numel: int
    ) -> tuple[tuple[_AccumSlot, _AccumSlot], tuple[torch.dtype, int]]:
        key = (dtype, chunk_numel)
        slots = self._slots.get(key)
        if slots is None:
            slots = (
                _AccumSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
                _AccumSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
            )
            self._slots[key] = slots
        return slots, key

    def next_slot(self, key: tuple[torch.dtype, int]) -> int:
        index = self._next_slot.get(key, 0)
        self._next_slot[key] = index + 1
        return index

    def close(self) -> None:
        self.staging.close()
        self._slots.clear()
        self._next_slot.clear()


_PIPELINES_BY_DEVICE: dict[torch.device, _AccumPipeline] = {}


def register_cpu_offload_hooks() -> None:
    """Enable the CPU-offload FSDP hooks owned by this module."""
    global _CPU_OFFLOAD_ENABLED
    _CPU_OFFLOAD_ENABLED = True


def unregister_cpu_offload_hooks() -> None:
    """Disable the CPU-offload FSDP hooks (symmetric to register).

    The wrappers stay installed; they delegate to the untouched upstream
    functions while disabled.
    """
    global _CPU_OFFLOAD_ENABLED
    _CPU_OFFLOAD_ENABLED = False


def _cpu_offload_enabled() -> bool:
    return _CPU_OFFLOAD_ENABLED


def _pipeline_for(device: torch.device) -> _AccumPipeline:
    pipeline = _PIPELINES_BY_DEVICE.get(device)
    if pipeline is None:
        pipeline = _AccumPipeline(device)
        _PIPELINES_BY_DEVICE[device] = pipeline
    return pipeline


@torch.no_grad()
def accumulate_cpu_grad(
    fsdp_param: Any,
    incoming: Tensor,
    compute_device: torch.device,
    *,
    stream: Any,
    synchronous: bool = False,
) -> None:
    """Accumulate one reduced shard in CPU canonical storage via NPU.

    incoming is already the post-reduce dtype and is a view of FSDP's
    reduce-scatter output. The operation is stream ordered: the scratch
    slice is reused only after its queued D2H copy on the same stream.
    """
    grad = getattr(getattr(fsdp_param, "sharded_param", None), "grad", None)
    if grad is None:
        raise RuntimeError("FSDP gradient accumulation requires an existing canonical CPU gradient")
    canonical = local_tensor(grad)
    compute = torch.device(compute_device)
    if canonical.device.type != "cpu":
        raise RuntimeError(f"FSDP gradient accumulation requires CPU canonical storage, got {canonical.device}")
    if incoming.device != compute:
        raise RuntimeError(f"FSDP gradient accumulation input is on {incoming.device}, expected {compute}")
    if incoming.numel() != canonical.numel():
        raise RuntimeError(
            "FSDP gradient accumulation shape mismatch: "
            f"incoming={tuple(incoming.shape)} canonical={tuple(canonical.shape)}"
        )
    if incoming.dtype != canonical.dtype:
        raise RuntimeError(
            f"FSDP gradient accumulation dtype mismatch: incoming={incoming.dtype} canonical={canonical.dtype}"
        )
    if incoming.numel() == 0:
        return
    if not canonical.is_contiguous() or not incoming.is_contiguous():
        raise RuntimeError("FSDP NPU gradient accumulation requires contiguous canonical and reduced shards")

    canonical_flat = canonical.view(-1)
    incoming_flat = incoming.view(-1)
    chunk_numel = max(1, _CHUNK_BYTES // canonical.element_size())
    previous_offload = getattr(fsdp_param, "grad_offload_event", None)
    if synchronous or not canonical.is_pinned():
        if previous_offload is not None:
            previous_offload.synchronize()
        canonical.add_(incoming.to(device="cpu"))
        return

    pipeline = _pipeline_for(compute)
    transfer_stream = pipeline.staging.stream
    if previous_offload is not None:
        # A second microbatch can enter before FSDP's finalization fence.  The
        # H2D must observe the first microbatch's D2H before reading the same
        # canonical slice.
        transfer_stream.wait_event(previous_offload)
    slots, slot_key = pipeline.slots(
        canonical.dtype,
        min(chunk_numel, canonical.numel()),
    )
    producer_ready = torch_npu.npu.Event()
    producer_ready.record(stream)
    pipeline.compute_stream.wait_event(producer_ready)
    incoming_flat.record_stream(pipeline.compute_stream)
    last_complete = None
    for offset in range(0, canonical.numel(), chunk_numel):
        count = min(chunk_numel, canonical.numel() - offset)
        slot = slots[pipeline.next_slot(slot_key) % 2]
        destination = canonical_flat[offset : offset + count]
        source = incoming_flat[offset : offset + count]
        work = slot.buffer[:count]
        with torch_npu.npu.stream(transfer_stream):
            if slot.release is not None:
                transfer_stream.wait_event(slot.release.event)
        h2d = pipeline.staging.submit_h2d(
            destination,
            work,
            stream=transfer_stream,
            track=False,
        )
        h2d.wait_on(pipeline.compute_stream)
        with torch_npu.npu.stream(pipeline.compute_stream):
            work.add_(source)
        d2h = pipeline.staging.submit_d2h(
            work,
            destination,
            producer_stream=pipeline.compute_stream,
            stream=transfer_stream,
            track=False,
        )
        slot.release = d2h
        last_complete = d2h
    if last_complete is None:
        raise AssertionError("gradient accumulation did not enqueue a copy")
    fsdp_param.grad_offload_event = last_complete.event


def clear() -> None:
    """Release process-local staging resources at shutdown or test teardown."""
    global _BATCHED_COPY_IN_LOGGED, _DECLINE_LOGGED
    for pipeline in _PIPELINES_BY_DEVICE.values():
        pipeline.close()
    _PIPELINES_BY_DEVICE.clear()
    _CAST_SCRATCH.clear()
    _BATCHED_COPY_IN_LOGGED = False
    _DECLINE_LOGGED = 0


# ---------------------------------------------------------------------------
# Prefetch for singleton, CPU-offloaded parameter groups
# ---------------------------------------------------------------------------


def _is_single_rank_cpu_offload(group: Any) -> bool:
    if not _cpu_offload_enabled():
        return False
    if not isinstance(group.mesh_info, FSDPMeshInfo):
        return False
    return bool(
        group._all_gather_process_group.size() == 1
        and group.fsdp_params
        and all(fsdp_param.offload_to_cpu for fsdp_param in group.fsdp_params)
    )


def _uses_cpu_param_offload(fsdp_params: list[Any], device: torch.device) -> bool:
    """Whether this all-gather stages CPU-canonical (offloaded) parameters."""
    return (
        _cpu_offload_enabled()
        and device.type == "npu"
        and bool(fsdp_params)
        and all(fsdp_param.offload_to_cpu for fsdp_param in fsdp_params)
    )


# NPU-side scratch for storage→compute dtype conversions, per device and stream:
# grow-only byte buffer whose aligned views hold cast sources and, for raw
# mixed-dtype outputs, converted values before byte packing. FSDP may select
# either its prefetch stream or the current stream for explicit unshard.
# Separate buffers prevent cross-stream overwrite; FIFO ordering protects
# reuse within a stream. Stream equality/hash identify the underlying stream,
# including distinct Python wrappers for the same stream. Freed in ``clear()``.
_CAST_SCRATCH: dict[tuple[torch.device, Any], torch.Tensor] = {}


def _aligned_scratch_layout(
    sizes: list[tuple[int, torch.dtype]],
) -> tuple[list[tuple[int, int, torch.dtype]], int]:
    """Return aligned ``(offset, nbytes, dtype)`` entries and total bytes."""
    layout: list[tuple[int, int, torch.dtype]] = []
    offset = 0
    for numel, dtype in sizes:
        alignment = dtype.itemsize
        offset = (offset + alignment - 1) // alignment * alignment
        nbytes = numel * alignment
        layout.append((offset, nbytes, dtype))
        offset += nbytes
    return layout, offset


def _cast_scratch_views(
    device: torch.device,
    sizes: list[tuple[int, torch.dtype]],
    copy_in_stream: Any,
) -> list[torch.Tensor]:
    """Carve aligned typed views in scratch owned by this device and stream."""
    key = (device, copy_in_stream)
    layout, size_bytes = _aligned_scratch_layout(sizes)
    buffer = _CAST_SCRATCH.get(key)
    if buffer is None or buffer.numel() < size_bytes:
        buffer = torch.empty(max(size_bytes, 1), dtype=torch.uint8, device=device)
        _CAST_SCRATCH[key] = buffer
    return [buffer.narrow(0, offset, nbytes).view(dtype) for offset, nbytes, dtype in layout]


def _batched_h2d(
    sources: list[torch.Tensor],
    destination: torch.Tensor,
    split_sizes: list[int],
    destination_dtypes: list[torch.dtype],
    *,
    copy_in_stream: Any,
) -> None:
    """Batch CPU→NPU copy-in by direct-copy and dtype-cast subsets.

    * Same-dtype pairs: one cross-device ``torch._foreach_copy_`` (the
      officially supported mixed CPU/NPU form; executed as one batched-DMA
      call).
    * Storage→compute dtype casts (e.g. fp32 master weights under a bf16
      mixed-precision policy — every parameter when the model is built in
      fp32) cannot convert inside a cross-device foreach (the conversion
      routes to an ACLNN compute kernel that requires NPU inputs), and
      converting on the host costs 50-200ms per group single-threaded. They
      cross in two batched segments instead: one foreach H2D into an NPU
      scratch at the storage dtype (pure batched DMA), then one same-device
      foreach D2D cast into the communication buffer. Measured: ~0.5ms of
      host dispatch per group.
    * With a uniform destination dtype, a direct-only group uses one dispatch,
      a cast-only group uses two, and a direct+cast group uses three serial
      dispatches on the caller's copy-in stream.
    * Mixed destination dtypes (e.g. a non-floating-point parameter whose
      ``param_dtype`` FSDP clamps to ``None``): the caller allocates a raw
      uint8 buffer and this function carves one byte range per parameter —
      same-dtype pairs ride a single byte-level foreach. Cast pairs use
      aligned typed scratch for both H2D and conversion, then one additional
      D2D byte-pack dispatch writes the tight raw layout without making an
      unaligned typed view. Such cast-only and direct+cast groups therefore
      use three and four dispatches, respectively.
    * Each device/stream pair owns its scratch, so prefetch and explicit
      unshard can use different streams without overwriting pending work.
    * Zero-numel pairs carry no data and are dropped: a zero-length entry
      fails the batched-DMA parameter check.
    """
    direct_dsts: list[torch.Tensor] = []
    direct_srcs: list[torch.Tensor] = []
    cast_dsts: list[torch.Tensor] = []
    cast_byte_dsts: list[torch.Tensor] = []
    cast_destination_dtypes: list[torch.dtype] = []
    cast_srcs: list[torch.Tensor] = []
    if destination.dtype == torch.uint8:
        # Mixed-destination group: the buffer is raw bytes; carve one byte
        # range per parameter. Same-dtype pairs ride a single byte-level
        # foreach (dtype-agnostic as uint8). Cast pairs convert in aligned
        # typed scratch, then copy bytes into the tight raw layout.
        offset = 0
        for src, numel, dst_dtype in zip(sources, split_sizes, destination_dtypes, strict=True):
            nbytes = numel * dst_dtype.itemsize
            byte_range = destination.narrow(0, offset, nbytes)
            offset += nbytes
            if src.numel() == 0:
                continue
            if src.dtype == dst_dtype:
                direct_dsts.append(byte_range)
                direct_srcs.append(src if src.dtype == torch.uint8 else src.view(torch.uint8))
            else:
                cast_byte_dsts.append(byte_range)
                cast_destination_dtypes.append(dst_dtype)
                cast_srcs.append(src)
    else:
        destinations = torch.split_with_sizes(destination, split_sizes)
        for dst, src in zip(destinations, sources, strict=True):
            if src.numel() == 0:
                continue
            if src.dtype == dst.dtype:
                direct_dsts.append(dst)
                direct_srcs.append(src)
            else:
                cast_dsts.append(dst)
                cast_srcs.append(src)
    if cast_srcs:
        source_sizes = [(src.numel(), src.dtype) for src in cast_srcs]
        output_sizes = (
            [(src.numel(), dtype) for src, dtype in zip(cast_srcs, cast_destination_dtypes, strict=True)]
            if cast_byte_dsts
            else []
        )
        scratch = _cast_scratch_views(
            destination.device,
            source_sizes + output_sizes,
            copy_in_stream,
        )
        source_scratch = scratch[: len(cast_srcs)]
        # Segment 1: batched H2D at the storage dtype — no conversion, the
        # batched-DMA fast path. Segment 2: same-device dtype conversion. For
        # raw mixed-dtype output, segment 3 packs converted bytes into the
        # tight communication-buffer layout without an unaligned typed view.
        torch._foreach_copy_(source_scratch, cast_srcs, True)
        if cast_byte_dsts:
            output_scratch = scratch[len(cast_srcs) :]
            torch._foreach_copy_(output_scratch, source_scratch)
            torch._foreach_copy_(
                cast_byte_dsts,
                [tensor.view(torch.uint8) for tensor in output_scratch],
            )
        else:
            torch._foreach_copy_(cast_dsts, source_scratch)
    if direct_dsts:
        torch._foreach_copy_(direct_dsts, direct_srcs, True)  # non_blocking
    global _BATCHED_COPY_IN_LOGGED
    if not _BATCHED_COPY_IN_LOGGED:
        _BATCHED_COPY_IN_LOGGED = True
        _LOGGER.info(
            "CPU-offload batched copy-in active (first group: %d direct + %d "
            "cast params; direct H2D when present, cast H2D + device-side "
            "cast when present%s)",
            len(direct_dsts),
            len(cast_srcs),
            " + byte-pack" if cast_byte_dsts else "",
        )


def _batched_d2h(sources: list[torch.Tensor], destinations: list[torch.Tensor]) -> None:
    """One batched NPU→CPU swap: a single ``torch._foreach_copy_`` dispatch
    for the eligible pooled subset supplied by the caller. Sources and
    destinations are dtype-matched; zero-numel pairs carry no data and are
    dropped (zero-length entries fail the batched-DMA parameter check).

    This fast path requires backend support for mixed CPU/NPU foreach lists.
    Backend errors propagate; no per-pair ``copy_`` fallback is attempted.
    """
    batch = [
        (destination, source) for destination, source in zip(destinations, sources, strict=True) if source.numel() > 0
    ]
    if not batch:
        return
    torch._foreach_copy_([dst for dst, _ in batch], [src for _, src in batch], True)


@dataclass(slots=True)
class _AllGatherCopyInPlan:
    """Shared dtype and buffer-layout contract for batched all-gather copy-in."""

    sources: list[torch.Tensor]
    param_dtypes: list[list[torch.dtype]]
    param_numels: list[list[int]]
    destination_dtypes: list[torch.dtype]
    element_numels: list[int]
    buffer_dtype: torch.dtype
    input_split_sizes: list[int]
    local_numel: int


def _build_all_gather_copy_in_plan(
    sources: list[torch.Tensor],
    param_dtypes: list[list[torch.dtype]],
    param_numels: list[list[int]],
) -> _AllGatherCopyInPlan:
    """Build the shared dtype and byte-layout plan for batched copy-in.

    Uniform destination dtypes use element-count split sizes. Mixed
    destination dtypes use a raw ``uint8`` communication buffer and byte-count
    split sizes, matching the upstream all-gather copy-out contract.
    """
    destination_dtypes = [dtype for dtypes in param_dtypes for dtype in dtypes]
    element_numels = [numel for numels in param_numels for numel in numels]
    if not (len(sources) == len(destination_dtypes) == len(element_numels)):
        raise AssertionError(
            "CPU-offload all-gather metadata must describe one destination "
            f"per source: sources={len(sources)}, dtypes={len(destination_dtypes)}, "
            f"numels={len(element_numels)}"
        )
    if len(set(destination_dtypes)) == 1:
        buffer_dtype = destination_dtypes[0]
        input_split_sizes = element_numels
    else:
        buffer_dtype = torch.uint8
        input_split_sizes = [
            numel * destination_dtype.itemsize
            for numel, destination_dtype in zip(element_numels, destination_dtypes, strict=True)
        ]
    return _AllGatherCopyInPlan(
        sources=sources,
        param_dtypes=param_dtypes,
        param_numels=param_numels,
        destination_dtypes=destination_dtypes,
        element_numels=element_numels,
        buffer_dtype=buffer_dtype,
        input_split_sizes=input_split_sizes,
        local_numel=sum(input_split_sizes),
    )


# Log the first few collector declines so a training run directly shows
# which groups keep the upstream per-parameter path and why.
_DECLINE_LOG_LIMIT = 8
_DECLINE_LOGGED = 0


def _decline(fsdp_params: list[Any], index: int, reason: str) -> None:
    global _DECLINE_LOGGED
    if _DECLINE_LOGGED >= _DECLINE_LOG_LIMIT:
        return
    _DECLINE_LOGGED += 1
    param = fsdp_params[index] if 0 <= index < len(fsdp_params) else None
    fqn = getattr(param, "_param_fqn", None) or "?"
    _LOGGER.info(
        "CPU-offload batched copy-in declined for group with %d params "
        "(param #%d %r): %s — keeping upstream per-parameter path",
        len(fsdp_params),
        index,
        fqn,
        reason,
    )


def _collect_cpu_ag_inputs(
    fsdp_params: list[Any],
) -> _AllGatherCopyInPlan | None:
    """Collect CPU-canonical all-gather inputs directly, bypassing the
    per-parameter ``.to(device)`` H2D staging inside ``FSDPParam.all_gather_inputs``.

    Upstream stages every offloaded parameter with an individual
    ``.to(device)`` (one host dispatch and one NPU temporary per parameter)
    and then copies the staged tensors into the communication buffer with a
    second D2D foreach hop. Callers that can use the returned sources issue
    batched direct H2D copies into the communication buffer and, when casts
    are needed, batched H2D copies through NPU scratch followed by D2D casts
    (see ``_batched_h2d``).

    Returns a shared copy-in plan with the destination (compute) dtype and
    split-size contract used by both singleton and multi-rank paths, or
    ``None`` when the group must keep the upstream path (mixed offload,
    pre-all-gather extensions, non-SHARDED states, or unavailable CPU
    storage). Mixed destination dtypes use the plan's raw-byte buffer layout.
    """
    sources: list[torch.Tensor] = []
    dtypes: list[list[torch.dtype]] = []
    numels: list[list[int]] = []
    for index, fsdp_param in enumerate(fsdp_params):
        if not fsdp_param.offload_to_cpu:
            return _decline(fsdp_params, index, "param not offloaded")
        local_tensor = getattr(fsdp_param, "_sharded_local_tensor", None)
        if local_tensor is not None and hasattr(local_tensor, "fsdp_pre_all_gather"):
            return _decline(fsdp_params, index, "pre-all-gather extension")
        if fsdp_param.sharded_state != ShardedState.SHARDED:
            # SHARDED_POST_FORWARD data is device-resident (no H2D to save);
            # other states have no all-gather inputs to stage.
            return _decline(fsdp_params, index, f"sharded_state={fsdp_param.sharded_state}")
        source = getattr(fsdp_param, "_sharded_param_data", None)
        if source is None or source.device.type != "cpu":
            return _decline(fsdp_params, index, "sharded param data not on CPU")
        destination_dtype = fsdp_param.param_dtype
        if destination_dtype is None:
            destination_dtype = source.dtype
        sources.append(source.view(-1))
        dtypes.append([destination_dtype])
        numels.append([source.numel()])
    if not sources:
        return _decline(fsdp_params, 0, "empty group")
    # Mixed destination dtypes are fine: callers mirror upstream's uint8 byte
    # path (e.g. non-floating-point parameters whose ``param_dtype`` FSDP
    # clamps to ``None`` ride as raw bytes alongside the cast pairs).
    return _build_all_gather_copy_in_plan(sources, dtypes, numels)


def _allocate_and_stage_all_gather_input(
    plan: _AllGatherCopyInPlan,
    all_gather_comm: Any,
    device: torch.device,
    *,
    world_size: int,
    rank: int,
    copy_in_stream: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the all-gather output and stage this rank's planned input."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise AssertionError(f"invalid all-gather rank layout: world_size={world_size}, rank={rank}")
    output = all_gather_comm.allocate(
        (plan.local_numel * world_size,),
        dtype=plan.buffer_dtype,
        device=device,
    )
    local_input = output.narrow(
        0,
        plan.local_numel * rank,
        plan.local_numel,
    )
    _batched_h2d(
        plan.sources,
        local_input,
        plan.element_numels,
        plan.destination_dtypes,
        copy_in_stream=copy_in_stream,
    )
    return output, local_input


@torch.no_grad()
def _copy_only_all_gather(
    fsdp_params: list[Any],
    group: Any,
    async_op: bool,
    all_gather_copy_in_stream: Any,
    all_gather_stream: Any,
    device: torch.device,
    all_gather_comm: Any,
) -> Any:
    """Stage singleton CPU-offloaded parameters without issuing a collective.

    Copy-in section derived from PyTorch ``_fsdp_collectives.foreach_all_gather``
    (BSD 3-Clause, Copyright (c) Facebook, Inc. and its affiliates). When the
    collector accepts the whole CPU-offloaded group, ``_batched_h2d`` stages
    direct-copy and cast subsets. Uniform-dtype destinations use one, two, or
    three dispatches depending on subset composition; raw mixed-dtype output
    with casts adds a D2D byte-pack dispatch. Cast sources first cross into NPU
    scratch at their storage dtype and are converted on-device.
    If the collector declines the group, the upstream-derived per-parameter
    path runs unchanged, including its ``_to_dtype_if_needed`` result.
    """
    del async_op, all_gather_stream
    if group.size() != 1:
        raise RuntimeError("copy-only all-gather requires a singleton process group")

    device_handle = _get_device_handle(device.type)
    if device_handle is None:
        raise RuntimeError(f"no device module is registered for {device.type}")
    with device_handle.stream(all_gather_copy_in_stream):
        plan = _collect_cpu_ag_inputs(fsdp_params) if _uses_cpu_param_offload(fsdp_params, device) else None
        if plan is not None:
            output, _ = _allocate_and_stage_all_gather_input(
                plan,
                all_gather_comm,
                device,
                world_size=1,
                rank=0,
                copy_in_stream=all_gather_copy_in_stream,
            )
            input_dtypes = plan.param_dtypes
            input_numels = plan.param_numels
            input_split_sizes = plan.input_split_sizes
        else:
            param_inputs = _get_param_all_gather_inputs(fsdp_params)
            input_dtypes, input_numels, dtype = _get_all_gather_input_metadatas(param_inputs)
            if dtype == torch.uint8:
                inputs = [tensor.view(torch.uint8) for tensors in param_inputs for tensor in tensors]
            else:
                inputs = [*chain.from_iterable(param_inputs)]
            input_split_sizes = [tensor.numel() for tensor in inputs]
            input_numel = sum(input_split_sizes)
            output = all_gather_comm.allocate((input_numel,), dtype=dtype, device=device)
            _, output = torch.ops.fsdp.all_gather_copy_in(
                inputs,
                output,
                input_split_sizes,
                input_numel,
                0,
            )
        ready_event = all_gather_copy_in_stream.record_event()

    return AllGatherResult(
        output,
        ready_event,
        None,
        input_dtypes,
        input_numels,
        input_split_sizes,
    )


def _is_copy_only_result(group: Any) -> bool:
    result = group._all_gather_result
    return bool(
        _cpu_offload_enabled()
        and result is not None
        and result.all_gather_work is None
        and result.all_gather_input_split_sizes
    )


def _unshard_already_satisfied(group: Any) -> bool:
    """Mirror upstream ``unshard``'s early returns before its main body."""
    if group._all_gather_result is not None:  # already called, pending wait
        return True
    if group.is_unsharded:
        return True  # no-op
    return not group.unshard_in_backward and group._training_state == TrainingState.PRE_BACKWARD


# Wraps PyTorch _fsdp_param_group.FSDPParamGroup.unshard (BSD 3-Clause).
@_disable_functorch_if_active
def _patched_unshard(self: Any, async_op: bool = False) -> None:
    """Wrap ``FSDPParamGroup.unshard`` with a copy-only prefetch path.

    Upstream's singleton path stages nothing in ``unshard`` and pays a
    synchronous copy in ``wait_for_unshard``. For CPU-offloaded parameters we
    reuse the regular copy-in machinery on the all-gather copy-in stream so
    the H2D transfer overlaps with compute; every other case delegates to the
    original method.
    """
    if not _is_single_rank_cpu_offload(self) or _unshard_already_satisfied(self):
        return _ORIGINAL_UNSHARD(self, async_op)
    if self._reshard_after_forward_event is not None:
        # Resharded parameter data is allocated in the default stream and
        # used in the all-gather streams
        self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
        self._reshard_after_forward_event = None
    with record_function(self._with_fqn("FSDP::all_gather_copy_in")):
        copy_in_stream, gather_stream = self.comm_ctx.get_all_gather_streams(async_op, self._training_state)
        self._all_gather_result = _copy_only_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            copy_in_stream,
            gather_stream,
            self.device,
            self._all_gather_comm,
        )


def _wait_for_copy_only_unshard(group: Any) -> None:
    """Consume a copy-only all-gather result like the multi-rank copy-out."""
    result = group._all_gather_result
    if group._training_state == TrainingState.FORWARD and (  # implicit prefetch
        prev_all_gather_state := group.comm_ctx.all_gather_state
    ):
        group._wait_all_gather_streams_on_event(prev_all_gather_state.event)
        group.comm_ctx.all_gather_state = None  # free the all-gather result
    with record_function(group._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out(
            result,
            group.fsdp_params,
            group._all_gather_process_group,
        )
    for fsdp_param in group.fsdp_params:
        fsdp_param.init_unsharded_param()

    group._to_unsharded()
    all_gather_copy_out_event = group.device_handle.Event()
    all_gather_copy_out_event.record()
    # Upstream only defers the free for multi-rank all-gathers; copy-only
    # groups are single-rank, so always wait on the copy-out event.
    group._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    group._all_gather_result = None


# Wraps PyTorch _fsdp_param_group.FSDPParamGroup.wait_for_unshard (BSD 3-Clause).
@_disable_functorch_if_active
def _patched_wait_for_unshard(self: Any) -> None:
    """Wrap ``FSDPParamGroup.wait_for_unshard`` for copy-only results."""
    if _is_copy_only_result(self):
        _wait_for_copy_only_unshard(self)
        return None
    return _ORIGINAL_WAIT_FOR_UNSHARD(self)


# ---------------------------------------------------------------------------
# Batched copy-in for multi-rank CPU-offloaded parameter groups
# ---------------------------------------------------------------------------


# Mirrors PyTorch _fsdp_collectives.foreach_all_gather (BSD 3-Clause).
@torch.no_grad()
def _cpu_offload_foreach_all_gather(
    fsdp_params: list[Any],
    group: Any,
    async_op: bool,
    all_gather_copy_in_stream: Any,
    all_gather_stream: Any,
    device: torch.device,
    all_gather_comm: Any,
    *,
    plan: _AllGatherCopyInPlan,
) -> Any:
    """Mirror upstream ``foreach_all_gather`` with batched copy-in subsets.

    The rank-offset narrow matches upstream ``all_gather_copy_in``
    (``narrow(0, all_gather_input_numel * rank, ...)``). The pinned CPU
    shards are partitioned into direct-copy and cast subsets by
    ``_batched_h2d``. For a uniform destination dtype, direct-only, cast-only,
    and direct+cast groups use one, two, and three serial dispatches. Raw
    mixed-dtype output with casts adds a D2D byte-pack dispatch. These replace
    upstream's N per-parameter ``.to(device)`` dispatches plus the D2D second
    hop through ``all_gather_copy_in``.
    """
    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    if device_handle is None:
        raise RuntimeError(f"no device module is registered for {device.type}")
    with device_handle.stream(all_gather_copy_in_stream):
        all_gather_output, all_gather_input = _allocate_and_stage_all_gather_input(
            plan,
            all_gather_comm,
            device,
            world_size=world_size,
            rank=rank,
            copy_in_stream=all_gather_copy_in_stream,
        )
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    with device_handle.stream(all_gather_stream):
        all_gather_work = all_gather_comm(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = all_gather_stream.record_event()
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            plan.param_dtypes,
            plan.param_numels,
            plan.input_split_sizes,
        )


def _patched_foreach_all_gather(*args: Any, **kwargs: Any) -> Any:
    """Wrap ``foreach_all_gather`` to batch the CPU-offload copy-in.

    ``args[0]`` is ``fsdp_params`` and ``args[5]`` is ``device`` (the
    upstream positional signature); keyword calls are forwarded verbatim.
    Groups that are not entirely CPU-offloaded, or that the collector
    declines (extensions, non-SHARDED states, or unavailable CPU storage),
    keep the untouched upstream implementation. Mixed destination dtypes use
    the collector's raw-byte path instead of forcing a fallback.
    """
    fsdp_params = args[0] if args else kwargs["fsdp_params"]
    device = args[5] if len(args) > 5 else kwargs["device"]
    if _uses_cpu_param_offload(fsdp_params, device):
        plan = _collect_cpu_ag_inputs(fsdp_params)
        if plan is not None:
            return _cpu_offload_foreach_all_gather(*args, **kwargs, plan=plan)
    return _ORIGINAL_FOREACH_ALL_GATHER(*args, **kwargs)


# ---------------------------------------------------------------------------
# NPU accumulation of reduced shards into CPU-canonical gradients
# ---------------------------------------------------------------------------


def _uses_cpu_canonical_grads(fsdp_params: list[Any], device: torch.device) -> bool:
    """Whether this reduction feeds CPU-canonical gradients on the NPU."""
    return (
        _cpu_offload_enabled() and device.type == "npu" and any(fsdp_param.offload_to_cpu for fsdp_param in fsdp_params)
    )


# Mirrors PyTorch _fsdp_collectives.foreach_reduce (BSD 3-Clause).
def _cpu_offload_foreach_reduce(
    fsdp_params: list[Any],
    unsharded_grads: list[Tensor],
    reduce_scatter_group: Any,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: Any,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: Any,  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Tensor | None,  # only used for HSDP
    all_reduce_hook: Any,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    Tensor,
    Any,
    torch.Stream,
    Any,
    Tensor | None,
    Any,
    Tensor | None,
]:
    """Upstream ``foreach_reduce`` with NPU accumulation for CPU-canonical grads.

    This mirrors
    ``torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce``
    with two post-reduce changes: an existing CPU-canonical gradient is
    accumulated on the NPU through :func:`accumulate_cpu_grad`, and eligible
    first-microbatch pinned destinations are collected into one batched D2H
    dispatch. The HSDP all-reduce, all-reduce-hook, and partial-reduction
    paths follow upstream.
    """
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = _get_gradient_divide_factors(
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device.type,
        gradient_divide_factor,
        force_sum_reduction_for_comms,
    )

    world_size = 1 if reduce_scatter_group is None else reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    if device_handle is None:
        raise RuntimeError("NPU is the only supported backend")
    current_stream = device_handle.current_stream()

    # CPU-offloaded models can carry hardcoded fp32 parameters alongside
    # bf16 mixed-precision weights, producing mixed-dtype gradients within
    # one param group. Upstream FSDP expects uniform dtype at the copy-in;
    # cast everything to the reduce dtype (fp32 under mixed-precision-reduce)
    # before entering the reduce-scatter pipeline.
    if len({grad.dtype for grad in unsharded_grads}) > 1:
        unsharded_grads = [grad.to(reduce_dtype) if grad.dtype != reduce_dtype else grad for grad in unsharded_grads]

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads, strict=False)):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if world_size > 1:
            reduce_scatter_comm(
                output_tensor=reduce_output,
                input_tensor=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        else:
            # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
            # account for a possible gradient_divide_factor.
            if gradient_divide_factor is not None:
                reduce_output.copy_(reduce_scatter_input / gradient_divide_factor)
            else:
                reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                # Keep refs to the reduce-dtype AR buffer + completion event
                # so FSDPParamGroup._all_reduce_state can hold them across
                # layers; see upstream PR #140044 / #180900.
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        # Rebinds to a new orig_dtype tensor when reduce_dtype != orig_dtype
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        # Collect eligible pre-carved D2H offloads and issue one batched
        # ``torch._foreach_copy_`` after the loop. Parameters that do not meet
        # the pooled-path conditions keep their per-parameter copy.
        pooled_offloads: list[tuple[Tensor, Tensor, Any]] = []
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params, strict=False):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            # In-backward CPU consumers of the grad must observe completed
            # transfers, hence the synchronous mode below.
            has_post_acc_grad_hook = bool(getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", None))
            if fsdp_param.offload_to_cpu:
                if to_accumulate_grad:
                    # Accumulate into CPU-canonical storage on the NPU:
                    # chunked H2D, add, and D2H through pinned bounce
                    # buffers, all stream ordered.
                    accumulate_cpu_grad(
                        fsdp_param,
                        new_sharded_grad,
                        device,
                        stream=post_reduce_stream,
                        synchronous=has_post_acc_grad_hook,
                    )
                else:
                    # First microbatch for this parameter: install a CPU-
                    # canonical gradient. Eligible pooled destinations are
                    # copied in one batch below; others use the per-parameter
                    # upstream-derived staging path.
                    non_blocking = fsdp_param.pin_memory and not has_post_acc_grad_hook
                    grad_dest = getattr(fsdp_param, "_pooled_grad_dest", None)
                    deferred = False
                    if (
                        non_blocking
                        and grad_dest is not None
                        and grad_dest.shape == new_sharded_grad.shape
                        and grad_dest.dtype == new_sharded_grad.dtype
                    ):
                        # Pre-carved pinned slot from the flat pool: avoids the
                        # per-gradient allocation inside ``.to(non_blocking=True)``,
                        # which the host pinned allocator rounds up to a
                        # power-of-two size class (~5 MB average per gradient).
                        # The copy itself is deferred to the single batched
                        # foreach below for the eligible pooled subset.
                        pooled_offloads.append((grad_dest, new_sharded_grad, fsdp_param))
                        new_sharded_grad = grad_dest
                        deferred = True
                    else:
                        new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)
                    if non_blocking and not deferred:
                        # Record an event on which to block the CPU thread to
                        # ensure that the D2H copy finishes before the optimizer
                        fsdp_param.grad_offload_event = post_reduce_stream.record_event()
                    fsdp_param.sharded_param.grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            elif to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        "Expected fsdp_param.sharded_param.grad to be DTensor, "
                        f"got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                fsdp_param.sharded_param.grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            for hook in (getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        if pooled_offloads:
            # One batched D2H dispatch for the eligible pooled subset:
            # destinations are pre-carved pinned slots (truly asynchronous)
            # and sources are views of the flat reduce-scatter output. Backend
            # support for mixed CPU/NPU foreach lists is required; this helper
            # has no per-pair fallback.
            _batched_d2h(
                [source for _, source, _ in pooled_offloads],
                [destination for destination, _, _ in pooled_offloads],
            )
            # The whole batch shares one completion event, replacing the
            # per-parameter ``grad_offload_event``; the next microbatch's
            # ``accumulate_cpu_grad`` still waits each parameter's event on
            # the transfer stream, and no consumer depends on cross-parameter
            # event ordering.
            pooled_done = post_reduce_stream.record_event()
            for _, _, fsdp_param in pooled_offloads:
                fsdp_param.grad_offload_event = pooled_done
        post_reduce_event = post_reduce_stream.record_event()
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def _patched_foreach_reduce(*args: Any, **kwargs: Any) -> Any:
    """Wrap ``foreach_reduce`` to accumulate CPU-canonical grads on the NPU.

    ``args[0]`` is ``fsdp_params`` and ``args[7]`` is ``device`` (the
    upstream positional signature); keyword calls are forwarded verbatim.
    """
    fsdp_params = args[0] if args else kwargs["fsdp_params"]
    device = args[7] if len(args) > 7 else kwargs["device"]
    if _uses_cpu_canonical_grads(fsdp_params, device):
        return _cpu_offload_foreach_reduce(*args, **kwargs)
    return _ORIGINAL_FOREACH_REDUCE(*args, **kwargs)


# Installation
# ---------------------------------------------------------------------------

_ORIGINAL_FOREACH_REDUCE: Any = _collectives.foreach_reduce
_ORIGINAL_UNSHARD: Any = _param_group.FSDPParamGroup.unshard
_ORIGINAL_WAIT_FOR_UNSHARD: Any = _param_group.FSDPParamGroup.wait_for_unshard
_ORIGINAL_FOREACH_ALL_GATHER: Any = _collectives.foreach_all_gather

_EXPECTED_FOREACH_REDUCE_PARAMS = (
    "fsdp_params",
    "unsharded_grads",
    "reduce_scatter_group",
    "reduce_scatter_stream",
    "reduce_scatter_comm",
    "orig_dtype",
    "reduce_dtype",
    "device",
    "gradient_divide_factor",
    "all_reduce_group",
    "all_reduce_stream",
    "all_reduce_grads",
    "partial_reduce_output",
    "all_reduce_hook",
    "force_sum_reduction_for_comms",
)

_EXPECTED_FOREACH_ALL_GATHER_PARAMS = (
    "fsdp_params",
    "group",
    "async_op",
    "all_gather_copy_in_stream",
    "all_gather_stream",
    "device",
    "all_gather_comm",
)


# Validates against PyTorch _fsdp_collectives/_fsdp_param_group signatures (BSD 3-Clause).
def _validate_upstream_signatures() -> None:
    """Fail loudly on PyTorch releases the wrappers were not built against."""
    foreach_reduce_params = tuple(signature(_ORIGINAL_FOREACH_REDUCE).parameters)
    if foreach_reduce_params != _EXPECTED_FOREACH_REDUCE_PARAMS:
        raise RuntimeError(
            "Unsupported PyTorch FSDP foreach_reduce signature "
            f"{foreach_reduce_params}; CPU-canonical NPU gradient accumulation "
            "was not installed"
        )
    unshard_params = tuple(signature(_ORIGINAL_UNSHARD).parameters)
    if unshard_params != ("self", "async_op"):
        raise RuntimeError(
            f"Unsupported PyTorch FSDP unshard signature {unshard_params}; CPU-offload prefetch was not installed"
        )
    wait_for_unshard_params = tuple(signature(_ORIGINAL_WAIT_FOR_UNSHARD).parameters)
    if wait_for_unshard_params != ("self",):
        raise RuntimeError(
            f"Unsupported PyTorch FSDP wait_for_unshard signature {wait_for_unshard_params}; "
            "CPU-offload prefetch was not installed"
        )
    foreach_all_gather_params = tuple(signature(_ORIGINAL_FOREACH_ALL_GATHER).parameters)
    if foreach_all_gather_params != _EXPECTED_FOREACH_ALL_GATHER_PARAMS:
        raise RuntimeError(
            "Unsupported PyTorch FSDP foreach_all_gather signature "
            f"{foreach_all_gather_params}; batched CPU-offload copy-in was not installed"
        )


def _wrap_metadata(wrapper: Any, original: Any, marker: str) -> None:
    """Give the wrapper the original's identity plus a marker and escape hatch."""
    functools.update_wrapper(
        wrapper,
        original,
        assigned=("__module__", "__name__", "__qualname__"),
        updated=(),
    )
    setattr(wrapper, marker, True)


_wrap_metadata(_patched_foreach_reduce, _ORIGINAL_FOREACH_REDUCE, _PATCH_MARKER)
_wrap_metadata(_patched_unshard, _ORIGINAL_UNSHARD, _PREFETCH_MARKER)
_wrap_metadata(_patched_wait_for_unshard, _ORIGINAL_WAIT_FOR_UNSHARD, _PREFETCH_MARKER)
_wrap_metadata(_patched_foreach_all_gather, _ORIGINAL_FOREACH_ALL_GATHER, _PATCH_MARKER)


def install() -> None:
    """Install the wrappers once per process, leaving originals reachable.

    Called explicitly by the CPU-offload optimizer container
    (:class:`torchtitan_npu.override.common.optimizer.CpuOffloadOptimizersContainer`)
    when the user selects the ``cpu_offload`` override; importing this module
    alone has no side effects. Without the installation, upstream PyTorch
    FSDP and the CPU-offload gradients are not patched.
    """
    foreach_reduce_patched = getattr(_collectives.foreach_reduce, _PATCH_MARKER, False)
    unshard_patched = getattr(_param_group.FSDPParamGroup.unshard, _PREFETCH_MARKER, False)
    wait_for_unshard_patched = getattr(_param_group.FSDPParamGroup.wait_for_unshard, _PREFETCH_MARKER, False)
    foreach_all_gather_patched = getattr(_collectives.foreach_all_gather, _PATCH_MARKER, False)
    if foreach_reduce_patched and unshard_patched and wait_for_unshard_patched and foreach_all_gather_patched:
        return
    _validate_upstream_signatures()
    if not foreach_reduce_patched:
        _collectives.foreach_reduce = _patched_foreach_reduce
        # ``_fsdp_param_group`` binds the function with a from-import.
        _param_group.foreach_reduce = _patched_foreach_reduce
    if not unshard_patched:
        _param_group.FSDPParamGroup.unshard = _patched_unshard
    if not wait_for_unshard_patched:
        _param_group.FSDPParamGroup.wait_for_unshard = _patched_wait_for_unshard
    if not foreach_all_gather_patched:
        _collectives.foreach_all_gather = _patched_foreach_all_gather
        # ``_fsdp_param_group`` binds the function with a from-import.
        _param_group.foreach_all_gather = _patched_foreach_all_gather
