# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read torchao-protocol MX checkpoints through the standard DCP flow.

The export pipeline flattens each quantized tensor to ``<module>._<weight>_qdata`` /
``..._scale`` with per-instance records in every shard header ``__metadata__``.
This reader trusts those header records only -- the ``config.json`` ``quant_type``
plan is never read here. It presents plain fp32 logical tensors keyed by
torchtitan HF FQNs, dequantized on the fly.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os  # noqa: TC003
import re
import struct
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint._hf_utils import SUFFIX, _HFStorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem  # noqa: TC002
from torch.futures import Future

from .common import get_safetensors_dtype, make_slices
from .mx_dequant_backend import dequantize_mx_on_npu

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_TENSOR_NAMES_KEY = "tensor_names"
_MX_TYPE = "MXTensor"

_SUPPORTED_BLOCK_SIZE = 32
# Scale components are identified by dtype (E8M0 bytes), not by ``tensor_data_names``.
_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)

# Dense renames, HF checkpoint FQNs -> logical DCP FQNs; inverse of
# ``_LOGICAL_TO_HF`` in export_quantized_hf.py.
_HF_TO_LOGICAL = (
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$"), r"layers.\1.attn.wq_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$"), r"layers.\1.attn.wq_b.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_proj\.weight$"), r"layers.\1.attn.wkv.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_a_proj\.weight$"), r"layers.\1.attn.wo_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_b_proj\.weight$"), r"layers.\1.attn.wo_b.weight"),
    (
        re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.q_b_proj\.weight$"),
        r"layers.\1.attn.indexer.wq_b.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$"),
        r"layers.\1.ffn.shared_experts.w1.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"),
        r"layers.\1.ffn.shared_experts.w2.weight",
    ),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$"),
        r"layers.\1.ffn.shared_experts.w3.weight",
    ),
)
# Merged experts; ``gate_up_proj`` rows are ``[w1; w3]`` per expert.
_GATE_UP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_DOWN_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")

# quant_config.elem_dtype name -> elements packed per storage byte.
_QDATA_PACKING = {
    "float4_e2m1fn_x2": 2,
    "float8_e4m3fn": 1,
}


@dataclass(frozen=True, slots=True)
class _PhysicalTensor:
    key: str
    relative_path: str
    size: torch.Size
    dtype: torch.dtype
    byte_offset: int
    byte_length: int

    @property
    def ndim(self) -> int:
        return len(self.size)


@dataclass(frozen=True, slots=True)
class AOQuantizedRegion:
    """Logical fp32 view: a row band of a physical qdata/scale pair; ``expert`` set for merged 3D weights."""

    qdata: _PhysicalTensor
    scale: _PhysicalTensor
    expert: int | None
    row_start: int
    row_count: int
    packing: int
    block_size: int

    @property
    def cols(self) -> int:
        return int(self.qdata.size[-1])

    @property
    def scale_pair_cols(self) -> int:
        """Columns one 2-byte scale pair covers: the kernel input packs two E8M0 scales."""
        return 2 * self.block_size

    @property
    def logical_size(self) -> torch.Size:
        return torch.Size((self.row_count, self.cols))

    def row_window(self, row_offset: int, row_count: int) -> tuple[slice | int, ...]:
        rows = slice(self.row_start + row_offset, self.row_start + row_offset + row_count)
        return (self.expert, rows) if self.expert is not None else (rows,)


def _aligned_cols(req: ReadItem, region: AOQuantizedRegion) -> tuple[int, int]:
    """Column window of ``req`` grown to whole MX scale pairs."""
    pair_cols = region.scale_pair_cols
    k_start = int(req.storage_offsets[-1])
    k_end = k_start + int(req.lengths[-1])
    aligned_start = (k_start // pair_cols) * pair_cols
    aligned_end = min(region.cols, ((k_end + pair_cols - 1) // pair_cols) * pair_cols)
    return aligned_start, aligned_end


def _reject_remote_path(path: object) -> None:
    # ``://`` mirrors DCP's own FileSystem.validate_checkpoint_id URL check.
    if "://" in str(path):
        raise ValueError(
            f"AO MX checkpoints must live on the local filesystem, got {path!r}; "
            "stage the shards on a local or mounted path first"
        )


def _read_raw_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse the safetensors JSON header; returns (header, offset where tensor data starts)."""
    with open(path, "rb") as f:
        (header_length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_length))
    return header, 8 + header_length


def _ao_protocol_entries(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return {tensor_fqn: decoded metadata entry} for an AO shard header, else ``None``."""
    if not metadata or _TENSOR_NAMES_KEY not in metadata:
        return None
    raw_names = metadata[_TENSOR_NAMES_KEY]
    try:
        names = json.loads(raw_names) if isinstance(raw_names, str) else raw_names
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(names, list):
        return None
    entries: dict[str, Any] = {}
    for name in names:
        blob = metadata.get(name)
        if not isinstance(blob, str):
            return None
        try:
            entries[name] = json.loads(blob)
        except json.JSONDecodeError:
            return None
    return entries


class MxAOHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Serve AO-protocol MX checkpoints to DCP as plain logical fp32 tensors.

    Local filesystem only: headers and tensor bytes are read by raw byte offset
    and ``safe_open`` needs a real file to mmap, so URL paths (``scheme://``)
    are rejected up front instead of failing later with opaque errors.
    """

    def __init__(
        self,
        path: str,
        *,
        thread_count: int = 4,
        npu_max_inflight: int = 2,
    ) -> None:
        _reject_remote_path(path)
        super().__init__(path=path, thread_count=thread_count)
        if npu_max_inflight <= 0:
            raise ValueError("npu_max_inflight must be positive")

        self.npu_max_inflight = npu_max_inflight
        self._regions: dict[str, AOQuantizedRegion] = {}
        self._scan: tuple[str, Metadata, dict[MetadataIndex, _HFStorageInfo], dict[str, AOQuantizedRegion]] | None = (
            None
        )

    def detect(self) -> bool:
        """True if a shard header carries the AO protocol; shards repeat metadata, so the first decides."""
        for safetensors_file in self.fs.ls(self.path):
            if not safetensors_file.endswith(SUFFIX):
                continue
            with safe_open(safetensors_file, framework="pt") as f:
                entries = _ao_protocol_entries(f.metadata())
            if entries is not None:
                return any(entry.get("_type") == _MX_TYPE for entry in entries.values())
        return False

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        if checkpoint_id is not None:
            _reject_remote_path(checkpoint_id)
        super().reset(checkpoint_id)
        # Same-path resets only restamp ``load_id`` (done on the cache hit), so
        # the scan survives; drop it only when ``path`` actually changes.
        if self._scan is not None and self._scan[0] != str(self.path):
            self._scan = None

    def read_metadata(self) -> Metadata:
        """Parse the shard headers once per path; repeat calls reuse the cached scan."""
        if self._scan is not None and self._scan[0] == str(self.path):
            _, metadata, storage_data, regions = self._scan
            self.storage_data = storage_data
            self._regions = regions
            if metadata.storage_meta is not None:
                metadata.storage_meta.load_id = self.load_id
            return metadata

        ao_entries: dict[str, Any] = {}
        physical: dict[str, _PhysicalTensor] = {}
        skipped: list[str] = []

        for safetensors_file in self.fs.ls(self.path):
            if not safetensors_file.endswith(SUFFIX):
                continue
            # ``__metadata__`` lives in the raw header, so no separate safe_open here.
            header, data_start = _read_raw_header(safetensors_file)
            shard_entries = _ao_protocol_entries(header.get("__metadata__"))
            if shard_entries is None:
                skipped.append(safetensors_file)
                continue
            for name, entry in shard_entries.items():
                ao_entries.setdefault(name, entry)
            for key, info in header.items():
                if key == "__metadata__" or not isinstance(info, dict):
                    continue
                physical[key] = _PhysicalTensor(
                    key=key,
                    relative_path=safetensors_file,
                    size=torch.Size(info["shape"]),
                    dtype=get_safetensors_dtype(info["dtype"]),
                    byte_offset=data_start + int(info["data_offsets"][0]),
                    byte_length=int(info["data_offsets"][1]) - int(info["data_offsets"][0]),
                )

        if not ao_entries:
            raise ValueError(f"No torchao-protocol metadata found under {self.path}")
        if skipped:
            # The exporter stamps every shard; an un-stamped one means a hand-assembled
            # mixed directory whose tensors would otherwise vanish silently.
            preview = ", ".join(skipped[:4]) + (", ..." if len(skipped) > 4 else "")
            logger.warning("Ignoring %d shard(s) without AO protocol metadata: %s", len(skipped), preview)

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}
        logical_regions: dict[str, AOQuantizedRegion] = {}
        quantized_keys = self._quantized_physical_keys(ao_entries)

        for key, tensor in physical.items():
            if key in quantized_keys:
                continue
            state_dict_metadata[key] = TensorStorageMetadata(
                properties=TensorProperties(dtype=tensor.dtype),
                size=tensor.size,
                chunks=[ChunkStorageMetadata(offsets=torch.Size((0,) * len(tensor.size)), sizes=tensor.size)],
            )
            storage_data[MetadataIndex(key, torch.Size((0,) * len(tensor.size)))] = _HFStorageInfo(
                relative_path=tensor.relative_path,
                shape=tensor.size,
                dtype=tensor.dtype,
            )

        for fqn, entry in ao_entries.items():
            if entry.get("_type") != _MX_TYPE:
                continue
            regions = self._expand_mx_tensor(fqn, entry, physical)
            for logical_fqn, region in regions.items():
                size = region.logical_size
                state_dict_metadata[logical_fqn] = TensorStorageMetadata(
                    properties=TensorProperties(dtype=torch.float32),
                    size=size,
                    chunks=[ChunkStorageMetadata(offsets=torch.Size((0, 0)), sizes=size)],
                )
                storage_data[MetadataIndex(logical_fqn, torch.Size((0, 0)))] = _HFStorageInfo(
                    relative_path=region.qdata.relative_path,
                    shape=size,
                    dtype=torch.float32,
                )
                logical_regions[logical_fqn] = region

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # pyrefly: ignore [bad-argument-type]
            storage_data=storage_data,
        )
        if metadata.storage_meta is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        self.storage_data = storage_data
        self._regions = logical_regions
        self._scan = (str(self.path), metadata, storage_data, logical_regions)
        return metadata

    @staticmethod
    def _quantized_physical_keys(ao_entries: dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        for fqn, entry in ao_entries.items():
            if entry.get("_type") != _MX_TYPE:
                continue
            module_fqn, weight_name = fqn.rsplit(".", 1)
            prefix = f"{module_fqn}._{weight_name}_"
            data_names = entry.get("_tensor_data_names")
            if not data_names:
                raise ValueError(f"AO tensor {fqn}: MXTensor metadata lacks _tensor_data_names")
            keys.update(f"{prefix}{data_name}" for data_name in data_names)
        return keys

    @staticmethod
    def _mx_geometry(entry: dict[str, Any], qdata_ndim: int) -> tuple[int, int]:
        """``(qdata elements per byte, MX block size)`` from the per-instance ``quant_config``.

        Dequant below assumes blocks (and FP4 pairs) run along the last dim, but
        MXTensor also permits ``quant_axis``/``pack_axis`` at ``ndim - 2`` (matmul
        right operands); validate them here so such tensors fail explicitly instead
        of mis-dequantizing silently when the shapes happen to line up.
        """
        data = entry["_data"]
        config_data = data["quant_config"]["_data"]
        elem_dtype = config_data["elem_dtype"]
        name = elem_dtype["_data"] if isinstance(elem_dtype, dict) else elem_dtype
        packing = _QDATA_PACKING.get(name)
        if packing is None:
            raise ValueError(f"Unsupported MX quant_config elem_dtype: {name!r}")
        block_size = config_data.get("block_size")
        if block_size != _SUPPORTED_BLOCK_SIZE:
            raise ValueError(
                f"Unsupported MX quant_config block_size {block_size!r}; the npu_anti_mx_quant "
                f"dequant kernel implements only {_SUPPORTED_BLOCK_SIZE}-element blocks"
            )
        last_axis = qdata_ndim - 1
        quant_axis = data.get("quant_axis")
        if quant_axis != last_axis:
            raise ValueError(
                f"Unsupported MX quant_axis {quant_axis!r}; dequant reads blocks along the "
                f"last dim ({last_axis} for {qdata_ndim}D qdata) only"
            )
        pack_axis = data.get("pack_axis")
        if pack_axis is not None and pack_axis != last_axis:
            raise ValueError(f"Unsupported MX pack_axis {pack_axis!r}; FP4 pairs are unpacked along the last dim only")
        return packing, block_size

    @staticmethod
    def _resolve_components(
        fqn: str,
        prefix: str,
        entry: dict[str, Any],
        physical: dict[str, _PhysicalTensor],
    ) -> tuple[_PhysicalTensor, _PhysicalTensor]:
        """Split the record's ``_tensor_data_names`` components into (qdata, scale)."""
        data_names = entry.get("_tensor_data_names")
        if not data_names:
            raise ValueError(f"AO tensor {fqn}: MXTensor metadata lacks _tensor_data_names")
        roles: dict[str, list[_PhysicalTensor]] = {"qdata": [], "scale": []}
        for name in data_names:
            tensor = physical.get(f"{prefix}{name}")
            if tensor is None:
                raise ValueError(f"AO tensor {fqn}: {prefix}{name} listed in _tensor_data_names is absent on disk")
            roles["scale" if tensor.dtype in _SCALE_DTYPES else "qdata"].append(tensor)
        if len(roles["qdata"]) != 1 or len(roles["scale"]) != 1:
            raise ValueError(
                f"AO tensor {fqn}: expected one qdata and one scale component, "
                f"_tensor_data_names={data_names!r} resolved {len(roles['qdata'])} payloads "
                f"and {len(roles['scale'])} scales"
            )
        return roles["qdata"][0], roles["scale"][0]

    def _expand_mx_tensor(
        self,
        fqn: str,
        entry: dict[str, Any],
        physical: dict[str, _PhysicalTensor],
    ) -> dict[str, AOQuantizedRegion]:
        module_fqn, weight_name = fqn.rsplit(".", 1)
        prefix = f"{module_fqn}._{weight_name}_"
        qdata, scale = self._resolve_components(fqn, prefix, entry, physical)
        packing, block_size = self._mx_geometry(entry, qdata.ndim)
        self._validate_pair(fqn, qdata, scale, packing, block_size)

        def make_region(expert: int | None, row_start: int, row_count: int) -> AOQuantizedRegion:
            return AOQuantizedRegion(
                qdata=qdata,
                scale=scale,
                expert=expert,
                row_start=row_start,
                row_count=row_count,
                packing=packing,
                block_size=block_size,
            )

        gate_up = _GATE_UP_RE.match(fqn)
        if gate_up:
            layer = gate_up.group(1)
            if qdata.ndim != 3:
                raise ValueError(f"AO tensor {fqn}: merged routed experts must be 3D, got shape {qdata.size}")
            num_experts, rows, _ = qdata.size
            if rows % 2:
                raise ValueError(f"AO tensor {fqn}: gate_up_proj row count {int(rows)} is not even")
            half = int(rows // 2)
            return {
                f"layers.{layer}.ffn.experts.{e}.w1.weight": make_region(e, 0, half) for e in range(int(num_experts))
            } | {
                f"layers.{layer}.ffn.experts.{e}.w3.weight": make_region(e, half, half) for e in range(int(num_experts))
            }

        down = _DOWN_RE.match(fqn)
        if down:
            layer = down.group(1)
            if qdata.ndim != 3:
                raise ValueError(f"AO tensor {fqn}: merged routed experts must be 3D, got shape {qdata.size}")
            num_experts, rows, _ = qdata.size
            return {
                f"layers.{layer}.ffn.experts.{e}.w2.weight": make_region(e, 0, int(rows))
                for e in range(int(num_experts))
            }

        for pattern, replacement in _HF_TO_LOGICAL:
            if pattern.match(fqn):
                if qdata.ndim != 2:
                    raise ValueError(f"AO tensor {fqn}: dense MX weight must be 2D, got shape {qdata.size}")
                return {pattern.sub(replacement, fqn): make_region(None, 0, int(qdata.size[-2]))}
        raise ValueError(f"Unsupported AO quantized tensor name: {fqn}")

    @staticmethod
    def _validate_pair(
        fqn: str,
        qdata: _PhysicalTensor,
        scale: _PhysicalTensor,
        packing: int,
        block_size: int,
    ) -> None:
        if qdata.dtype != (torch.float4_e2m1fn_x2 if packing == 2 else torch.float8_e4m3fn):
            raise ValueError(
                f"AO tensor {fqn}: qdata dtype {qdata.dtype} disagrees with quant_config packing={packing}"
            )
        if qdata.dtype == torch.float4_e2m1fn_x2 and qdata.byte_length * 2 != int(qdata.size.numel()):
            raise ValueError(f"AO tensor {fqn}: F4 qdata byte length {qdata.byte_length} does not pack {qdata.size}")
        if qdata.ndim not in (2, 3):
            raise ValueError(f"AO tensor {fqn}: qdata must be 2D dense or merged 3D, got shape {qdata.size}")
        if scale.dtype not in _SCALE_DTYPES:
            raise ValueError(f"AO tensor {fqn}: MX scale must contain E8M0 bytes, got {scale.dtype}")
        if len(scale.size) != qdata.ndim + 1 or scale.size[-1] != 2 or scale.size[:-2] != qdata.size[:-1]:
            raise ValueError(
                f"AO tensor {fqn}: scale shape {scale.size} does not match qdata rows/blocks "
                f"(*{qdata.size[:-1]}, blocks/2, 2)"
            )
        num_blocks = (int(qdata.size[-1]) + block_size - 1) // block_size
        if int(scale.size[-2]) != (num_blocks + 1) // 2:
            raise ValueError(f"AO tensor {fqn}: scale pairs {int(scale.size[-2])} != ceil({num_blocks} blocks / 2)")

    def _read_ao_inputs(
        self,
        qdata_file: Any,
        raw_file: BinaryIO,
        qdata_path: str,
        req: ReadItem,
        region: AOQuantizedRegion,
        spill_handle: Callable[[str], Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        aligned_start, _ = _aligned_cols(req, region)
        qdata = self._read_qdata(qdata_file, raw_file, req, region)
        scale = self._read_scale(qdata_file, qdata_path, req, region, spill_handle)
        crop_start = int(req.storage_offsets[-1]) - aligned_start
        return qdata, scale, crop_start

    def _read_qdata(
        self,
        qdata_file: Any,
        raw_file: BinaryIO,
        req: ReadItem,
        region: AOQuantizedRegion,
    ) -> torch.Tensor:
        row_offset = int(req.storage_offsets[-2])
        row_count = int(req.lengths[-2])
        aligned_start, aligned_end = _aligned_cols(req, region)
        if region.packing == 2:
            # Lazy safetensors slices cannot materialize F4 (0.8.0 bug); read the
            # packed row band straight from the file instead.
            row_bytes = region.cols // 2
            first_row = (region.expert or 0) * int(region.qdata.size[-2]) + region.row_start + row_offset
            raw_file.seek(region.qdata.byte_offset + first_row * row_bytes)
            band = raw_file.read(row_count * row_bytes)
            if len(band) != row_count * row_bytes:
                raise ValueError(
                    f"Incomplete F4 qdata for {region.qdata.relative_path}: "
                    f"read {len(band)} of {row_count * row_bytes} bytes"
                )
            rows = torch.frombuffer(bytearray(band), dtype=torch.uint8).view(row_count, row_bytes)
            byte_start, byte_end = aligned_start // 2, aligned_end // 2
            return rows[:, byte_start:byte_end]
        index = (*region.row_window(row_offset, row_count), slice(aligned_start, aligned_end))
        return qdata_file.get_slice(region.qdata.key)[index]

    def _read_scale(
        self,
        qdata_file: Any,
        qdata_path: str,
        req: ReadItem,
        region: AOQuantizedRegion,
        spill_handle: Callable[[str], Any],
    ) -> torch.Tensor:
        aligned_start, aligned_end = _aligned_cols(req, region)
        pair_cols = region.scale_pair_cols
        pair_start = aligned_start // pair_cols
        pair_end = (aligned_end + pair_cols - 1) // pair_cols
        index = (
            *region.row_window(int(req.storage_offsets[-2]), int(req.lengths[-2])),
            slice(pair_start, pair_end),
            slice(0, 2),
        )
        if region.scale.relative_path == qdata_path:
            scale_file = qdata_file
        else:
            # Greedy shard packing spills the scale into a sibling shard, not qdata_file.
            scale_file = spill_handle(region.scale.relative_path)
        scale = scale_file.get_slice(region.scale.key)[index]
        return scale.reshape(int(req.lengths[-2]), -1)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        has_mx = any(req.storage_index.fqn in self._regions for req in plan.items)
        if not has_mx:
            return super().read_data(plan, planner)
        return self._read_data_npu(plan, planner)

    def _read_data_npu(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        requests = list(plan.items)
        targets = {id(req): planner.resolve_tensor(req).detach() for req in requests}
        mx_requests = [req for req in requests if req.storage_index.fqn in self._regions]

        by_device: dict[torch.device, list[str]] = {}
        for req in mx_requests:
            by_device.setdefault(targets[id(req)].device, []).append(req.storage_index.fqn)
        if len(by_device) > 1:
            conflicts = "; ".join(f"{device}: [{', '.join(fqns)}]" for device, fqns in by_device.items())
            raise ValueError(f"AO MX loading needs all MX targets on one NPU device per rank, got {conflicts}")
        device = next(iter(by_device))

        # Plain logical tensors land via a blocking copy_ and need no device module or
        # dedicated stream, so they may sit on any device (e.g. CPU) next to the MX ones.
        plain_per_file: dict[str, list[ReadItem]] = {}
        for req in requests:
            if req.storage_index.fqn not in self._regions:
                info = self.storage_data[req.storage_index]
                plain_per_file.setdefault(info.relative_path, []).append(req)
        for file_name, file_requests in plain_per_file.items():
            with safe_open(filename=file_name, framework="pt", device="cpu") as data_file:
                for req in file_requests:
                    tensor = data_file.get_slice(req.storage_index.fqn)[make_slices(req.storage_offsets, req.lengths)]
                    target = targets[id(req)]
                    target.copy_(tensor)
                    planner.commit_tensor(req, target)

        device_module = torch.get_device_module(device)
        stream = device_module.Stream(device=device)
        stream.wait_stream(device_module.current_stream(device))
        inflight: deque[tuple[Any, ReadItem, torch.Tensor, tuple[torch.Tensor, ...]]] = deque()

        def finish_oldest() -> None:
            event, request, target, _keepalive = inflight.popleft()
            event.synchronize()
            planner.commit_tensor(request, target)

        per_file: dict[str, list[ReadItem]] = {}
        for req in mx_requests:
            storage_info = self.storage_data[req.storage_index]
            per_file.setdefault(storage_info.relative_path, []).append(req)

        # Spilled-scale files may back many requests in this plan; open each at most once.
        spill_files: dict[str, Any] = {}
        spill_stack = contextlib.ExitStack()

        def spill_handle(path: str) -> Any:
            if path not in spill_files:
                spill_files[path] = spill_stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            return spill_files[path]

        try:
            for file_name, file_requests in per_file.items():
                with (
                    safe_open(filename=file_name, framework="pt", device="cpu") as data_file,
                    open(file_name, "rb") as raw_file,
                ):
                    for req in file_requests:
                        target = targets[id(req)]
                        region = self._regions[req.storage_index.fqn]
                        with device_module.stream(stream):
                            qdata, scale, crop_start = self._read_ao_inputs(
                                data_file, raw_file, file_name, req, region, spill_handle
                            )
                            resources = dequantize_mx_on_npu(
                                qdata,
                                scale,
                                crop_start=crop_start,
                                requested_length=int(req.lengths[-1]),
                                target=target,
                            )
                            keepalive = (qdata, scale, *resources)
                            event = device_module.Event()
                            event.record(stream)
                        inflight.append((event, req, target, keepalive))
                        if len(inflight) >= self.npu_max_inflight:
                            finish_oldest()

            while inflight:
                finish_oldest()
        finally:
            spill_stack.close()
            # Keep pending host buffers alive until all submitted copies complete.
            stream.synchronize()
        future: Future[None] = Future()
        future.set_result(None)
        return future
