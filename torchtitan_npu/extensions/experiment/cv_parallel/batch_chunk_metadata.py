# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prebuild full and half-batch attention metadata for CV chunking."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4.metadata import (
    DeepSeekV4Metadata,
    build_compressed_varlen_metadata,
)
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscV4Metadata,
    build_aicpu_metadata,
)

from . import _BATCH_CHUNK_PAIR_ATTR


@dataclass(kw_only=True, slots=True)
class BatchChunkAscV4Metadata(AscV4Metadata):
    batch_chunk_tensors: tuple[tuple[torch.Tensor, ...], ...] | None = None


if BatchChunkAscV4Metadata not in SUPPORTED_NODES:
    register_dataclass(
        BatchChunkAscV4Metadata,
        serialized_type_name=(f"{BatchChunkAscV4Metadata.__module__}.{BatchChunkAscV4Metadata.__name__}"),
    )


def _plain_batch_chunk_metadata(
    metadata: DeepSeekV4Metadata,
) -> tuple[DeepSeekV4Metadata, DeepSeekV4Metadata]:
    """Build two local plain-stream contracts at an aligned token boundary."""
    if metadata.window is not None:
        raise ValueError("batch chunk metadata does not support context parallelism")
    varlen = metadata.varlen
    if not torch.equal(varlen.cu_seq_q, varlen.cu_seq_k):
        raise ValueError("batch chunk metadata requires cu_seq_q == cu_seq_k")
    cu_host = getattr(varlen, "cu_seq_q_host", None)
    if cu_host is None:
        cu_host = tuple(int(value) for value in varlen.cu_seq_q.cpu().tolist())
    total_tokens = cu_host[-1]
    if total_tokens < 2 or total_tokens % 2:
        raise ValueError(f"batch chunk metadata requires an even positive token extent, got {total_tokens}")
    split_token = total_tokens // 2
    if split_token not in cu_host:
        raise ValueError(
            "batch chunk metadata requires documents not to cross the equal "
            f"batch boundary at token {split_token}; boundaries={cu_host}"
        )
    split_index = cu_host.index(split_token)
    # One entry is one geometry, so both chunks re-cut the same ratio's plan.
    ratio = metadata.ratio
    qk_share_storage = varlen.cu_seq_k is varlen.cu_seq_q
    chunks = []
    for start_index, end_index, token_offset in (
        (0, split_index, 0),
        (split_index, len(cu_host) - 1, split_token),
    ):
        local_host = tuple(boundary - token_offset for boundary in cu_host[start_index : end_index + 1])
        local_cu = (varlen.cu_seq_q[start_index : end_index + 1] - token_offset).contiguous()
        local_lengths = tuple(right - left for left, right in pairwise(local_host))
        local_max = max(local_lengths)
        local_varlen = VarlenMetadata(
            cu_seq_q=local_cu,
            # Preserve the full contract's pytree alias structure. A shared
            # q/k tensor contributes one runtime role; distinct tensors
            # contribute two roles and are remapped independently.
            cu_seq_k=local_cu if qk_share_storage else local_cu.clone(),
            max_q=local_max,
            max_k=local_max,
            cu_seq_q_host=local_host,
        )
        # The chunks are derived streams: a document may straddle the equal batch
        # boundary, so their fragments are not required to be whole compression blocks.
        chunks.append(
            build_compressed_varlen_metadata(
                local_varlen, ratio, window_size=metadata.window_size, require_aligned=False
            )
        )
    return cast(
        "tuple[DeepSeekV4Metadata, DeepSeekV4Metadata]",
        tuple(chunks),
    )


def _metadata_tensor_fields(
    metadata: AscV4Metadata,
) -> dict[str, torch.Tensor]:
    """Return tensor fields whose full/chunk values have identical roles."""
    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensors[name] = value

    add("varlen.cu_seq_q", metadata.varlen.cu_seq_q)
    if metadata.varlen.cu_seq_k is not metadata.varlen.cu_seq_q:
        add("varlen.cu_seq_k", metadata.varlen.cu_seq_k)
    for field_name in (
        "cu_seqlens_cmp_k",
        "block_remainder",
        "gather_indices",
        "block_positions",
        "first_indices",
        "compressed_rows",
        "cmp_k_global_gather_indices",
        "compressor_state_block_table",
    ):
        add(f"plan.{field_name}", getattr(metadata.plan, field_name))
    # The role names stay geometry-free on purpose: the full batch and its chunks carry the same
    # geometry, so the pairing is by role, and the ratio is part of the metadata, not the name.
    assert metadata.asc is not None, "batch chunk metadata requires the AscendC record tier"
    # The records are grouped by the kernel that reads them, one field per direction.
    for group_name in ("sparse_attn", "lightning_indexer"):
        group = getattr(metadata.asc, group_name)
        add(f"asc.{group_name}.fwd", group.fwd)
        add(f"asc.{group_name}.bwd", group.bwd)
    return tensors


def _tag_batch_chunk_metadata(
    full: AscV4Metadata,
    chunks: tuple[AscV4Metadata, AscV4Metadata],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Tag and pack only tensor roles consumed by the full-batch graph.

    Vendor builders may produce additional optional tensors for a chunk's
    document layout. Select only full-batch roles so those extra tensors and
    data-dependent host fields cannot change the runtime pytree signature.
    """
    full_fields = _metadata_tensor_fields(full)
    chunk_fields = tuple(_metadata_tensor_fields(chunk) for chunk in chunks)
    missing_by_chunk = {
        chunk_id: sorted(set(full_fields) - set(fields))
        for chunk_id, fields in enumerate(chunk_fields)
        if set(full_fields) - set(fields)
    }
    if missing_by_chunk:
        raise RuntimeError(f"batch-chunk metadata is missing full-batch tensor roles: {missing_by_chunk}")
    for pair_id, full_tensor in full_fields.items():
        setattr(full_tensor, _BATCH_CHUNK_PAIR_ATTR, (pair_id, -1))
        for chunk_id, fields in enumerate(chunk_fields):
            setattr(
                fields[pair_id],
                _BATCH_CHUNK_PAIR_ATTR,
                (pair_id, chunk_id),
            )
    return cast(
        "tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]",
        tuple(tuple(fields[pair_id] for pair_id in full_fields) for fields in chunk_fields),
    )


def _batch_chunk_metadata(metadata: AscV4Metadata) -> BatchChunkAscV4Metadata:
    """Add the two batch chunks' records beside the full batch's.

    The fused core's own config built the full microbatch's metadata (upstream
    #4897); this re-cuts that contract at the equal batch boundary and rebuilds the
    local records from the geometry that rides on it.  A quantized selector's entry
    is refused: its records are its own backend's operator, and this re-cut has no
    backend to ask (mirroring ``QuantV4Metadata.prepare_cp_metadata``).
    """
    if type(metadata) is not AscV4Metadata:
        raise TypeError(
            "batch chunk metadata cannot re-cut "
            f"{type(metadata).__name__}: its records are its own backend's "
            "(the quantized selector refuses the re-cut); run without cv_parallel "
            "or with the BF16 fused selector."
        )
    first, second = _plain_batch_chunk_metadata(metadata)

    asc = metadata.asc
    if asc is None:
        raise ValueError("batch chunk metadata requires the fused entry's AI_CPU records")
    geometry = asc.geometry

    def chunk_records(chunk: DeepSeekV4Metadata) -> AscV4Metadata:
        """This tier's records: the selector's and the attention's, plus the state tables."""
        return build_aicpu_metadata(chunk, geometry=geometry)

    chunks = (chunk_records(first), chunk_records(second))
    return BatchChunkAscV4Metadata(
        varlen=metadata.varlen,
        ratio=metadata.ratio,
        plan=metadata.plan,
        window=metadata.window,
        window_size=metadata.window_size,
        asc=metadata.asc,
        batch_chunk_tensors=_tag_batch_chunk_metadata(metadata, chunks),
    )


def chunk_attention_metadata(extra_kwargs: dict[str, Any]) -> None:
    """Re-cut this step's built entries into the full batch plus its two chunks, in place.

    The fused core's own config built one entry per geometry (upstream #4897); the
    re-cut is a property of this trainer's batch, so it happens at the step boundary,
    before anything traces or runs.  An entry that already carries the chunks passes
    through, so a boundary entered twice cannot cut the same entry twice.
    """
    metadata = extra_kwargs.get("attention_metadata")
    if not metadata:
        return
    for key, entry in cast("dict[object, AscV4Metadata]", metadata).items():
        if not isinstance(entry, BatchChunkAscV4Metadata):
            metadata[key] = _batch_chunk_metadata(entry)
