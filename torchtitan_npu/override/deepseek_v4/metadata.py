# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 fused attention's metadata hierarchy.

Each level is named for what it is, and the AI_CPU metadata is grouped by the consumer that
launches it -- one group per pair of kernels, one field per direction::

    aicpu_metadata
    + sparse_attn
      + fwd   -> sparse_flash_mla
      + bwd   -> sparse_flash_mla_grad
    + lightning_indexer
      + fwd   -> lightning_indexer / quant_lightning_indexer (the selector's own backend states it)
      + bwd   -> sparse_lightning_indexer_kl_loss_grad

``AicpuGeometry``
    the static numbers the AI_CPU operators are called with (heads, head dim, window, and the
    selector's widths).  It lives beside them because the context-parallel transform is handed
    the metadata alone and must rebuild it for this rank's frames; nothing in the forward or the
    backward reads it.

``AscV4Metadata``
    one geometry's entry: the model row (``DeepSeekV4Metadata``) plus the operator outputs of
    exactly those frames, computed at the eager boundary and rebuilt per rank under context
    parallelism.

The quantized selector is another backend: it owns an entry of its own (a subclass of this one)
and rebuilds the selector group with its own contract.  Nothing here knows about quantization.
"""

from dataclasses import dataclass

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4.metadata import CompressedBlockLayout, DeepSeekV4Metadata
from torchtitan_npu.models.deepseek_v4.token_dispatcher import ExchangePlan, WindowPlan
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata

__all__ = [
    "AicpuGeometry",
    "AscAicpuMetadata",
    "AscLightningIndexerMetadata",
    "AscSparseAttentionMetadata",
    "AscV4Metadata",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class AscSparseAttentionMetadata:
    """What the fused attention's two kernels are called with: forward and backward."""

    fwd: torch.Tensor | None = None
    """``sparse_flash_mla_metadata`` output (opaque, layer-invariant)."""

    bwd: torch.Tensor | None = None
    """``sparse_flash_mla_grad_metadata`` output (opaque, layer-invariant)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AscLightningIndexerMetadata:
    """What the selector's two kernels are called with: forward and backward.

    ``fwd`` is what the selector grids its scores with -- the record of whichever selector
    backend built this entry -- and ``bwd`` what ``sparse_lightning_indexer_kl_loss_grad``
    consumes.  A layer with no compressed axis runs no selector, so both are ``None`` there.
    """

    fwd: torch.Tensor | None = None
    bwd: torch.Tensor | None = None


# The V4 selector's wire contract, shared by the fused record builder, the quantized
# backend and the converter: a TND pool of ratio-4 blocks with right-down causal masking.
LI_LAYOUT = "TND"
LI_MASK_MODE = 3
LI_CMP_RATIO = 4


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuGeometry:
    """The static numbers the AI_CPU operators of one geometry are called with."""

    num_heads: int
    head_dim: int
    window_size: int
    index_topk: int
    index_num_heads: int
    index_head_dim: int


@dataclass(frozen=True, kw_only=True, slots=True)
class AscAicpuMetadata:
    """This geometry's AI_CPU operator outputs, grouped by the kernel that launches them.

    They depend only on the frames of one forward and the layer's static geometry -- never on
    the operands -- so they are computed once per microbatch (per rank under context parallelism)
    at the eager boundary and read by every layer of the geometry instead of being re-run per
    layer inside the compiled region.  A record describes the frames it was computed for, which
    is why a re-cut frame (the CP transform's rank-local plan, a batch chunk) rebuilds them.
    """

    geometry: AicpuGeometry
    sparse_attn: AscSparseAttentionMetadata
    lightning_indexer: AscLightningIndexerMetadata


@dataclass(kw_only=True, slots=True)
class AscV4Metadata(DeepSeekV4Metadata):
    """The fused path's entry: the model row plus the AscendC records of its frames."""

    asc: AscAicpuMetadata | None = None
    """This geometry's AI_CPU records, grouped by the kernel that reads them."""

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        # The seam dispatches on ``type(metadata)``, so this hook's contract is its own class;
        # widening the parameter would accept a rejecting row and lose this geometry's records.
        attention_metadata: "AscV4Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "AscV4Metadata":
        """This rank's plan *and* its AscendC records (upstream #4617).

        The rank-local plan comes from the pre-shard documents
        (``shard_compressed_varlen_metadata``); the records do not carry over, because a record
        describes the frames it was computed for (block tables, host offsets, kernel layouts).
        So every record is rebuilt for the rank-local frames by the same builder that built the
        global ones -- a quantized selector's entry overrides this hook, because its record
        contract is its own backend's.
        """
        del permutation
        # Lazy: the builder imports this module's classes, so the two cannot import each other
        # eagerly; the hook runs at the eager metadata boundary, never inside a graph.
        from .sparse_attn.ascendc import build_aicpu_metadata, shard_compressed_varlen_metadata

        asc = attention_metadata.asc
        if asc is None:
            raise ValueError("the fused entry must carry its AI_CPU records")
        common = shard_compressed_varlen_metadata(attention_metadata)
        return build_aicpu_metadata(common, geometry=asc.geometry)


# The entry is a traced input (the decoder hands it to every layer), so the plan it carries --
# and the plans' own exchange routes -- flatten into tensors and primitives as well.
# These registrations run only when the fused tier is imported.  The reference entry stays
# opaque by design: the graph trainer is a fused-kernel path (its tracer needs every leaf to
# be a tensor/primitive, and an unregistered dataclass is one non-primitive leaf).  A stack
# that runs the reference core under the graph trainer would have to register its own entry.
# ``register_dataclass`` registers *all* fields: a future eager-only host cache on these
# dataclasses must be added to ``drop_field_names`` (or it silently becomes a graph input).
if CPVarlenMetadata not in SUPPORTED_NODES:
    register_dataclass(
        CPVarlenMetadata, serialized_type_name=f"{CPVarlenMetadata.__module__}.{CPVarlenMetadata.__name__}"
    )

"""A CP-local entry carries the rank's own varlen stream; it is a traced leaf too."""
if CompressedBlockLayout not in SUPPORTED_NODES:
    register_dataclass(
        CompressedBlockLayout,
        serialized_type_name=f"{CompressedBlockLayout.__module__}.{CompressedBlockLayout.__name__}",
    )

if ExchangePlan not in SUPPORTED_NODES:
    register_dataclass(ExchangePlan, serialized_type_name=f"{ExchangePlan.__module__}.{ExchangePlan.__name__}")

if WindowPlan not in SUPPORTED_NODES:
    register_dataclass(WindowPlan, serialized_type_name=f"{WindowPlan.__module__}.{WindowPlan.__name__}")

if AscSparseAttentionMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscSparseAttentionMetadata,
        serialized_type_name=f"{AscSparseAttentionMetadata.__module__}.{AscSparseAttentionMetadata.__name__}",
    )

if AscLightningIndexerMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscLightningIndexerMetadata,
        serialized_type_name=f"{AscLightningIndexerMetadata.__module__}.{AscLightningIndexerMetadata.__name__}",
    )

if AicpuGeometry not in SUPPORTED_NODES:
    register_dataclass(AicpuGeometry, serialized_type_name=f"{AicpuGeometry.__module__}.{AicpuGeometry.__name__}")

if AscAicpuMetadata not in SUPPORTED_NODES:
    register_dataclass(
        AscAicpuMetadata, serialized_type_name=f"{AscAicpuMetadata.__module__}.{AscAicpuMetadata.__name__}"
    )

if AscV4Metadata not in SUPPORTED_NODES:
    register_dataclass(AscV4Metadata, serialized_type_name=f"{AscV4Metadata.__module__}.{AscV4Metadata.__name__}")
