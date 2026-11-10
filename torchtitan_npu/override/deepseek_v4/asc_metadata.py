# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4 fused attention's AI_CPU metadata and compact-indexer proof.

The model-dir metadata (``models/deepseek_v4/metadata.py``) owns the frames; this module owns
exactly what the AscendC kernels add to it, grouped by the consumer that launches it -- one group
per pair of kernels, one field per direction::

    aicpu_metadata
    + sparse_attn
      + fwd   -> sparse_flash_mla
      + bwd   -> sparse_flash_mla_grad
    + lightning_indexer
      + fwd   -> lightning_indexer / quant_lightning_indexer (the indexer's own backend states it)
      + bwd   -> sparse_lightning_indexer_kl_loss_grad

``AicpuGeometry``
    the static numbers the AI_CPU operators are called with (heads, head dim, window, and the
    indexer's widths).  It lives beside them because the context-parallel transform is handed the
    metadata alone and must rebuild it for this rank's frames; nothing in the forward or the
    backward reads it.

``AicpuMetadata``
    the AI_CPU operator outputs of one geometry, grouped by consumer, plus the geometry they were
    sized with.

``AscV4Metadata``
    the model metadata with this geometry's AI_CPU metadata and optional compact-indexer proof
    attached.  Under context parallelism its ``prepare_cp_metadata`` rebuilds the frames, AI_CPU
    metadata and proof for this rank.

The compressor has no frames here: the ratio-4 overlap's carry is applied outside the compressor
(``models/deepseek_v4/cp_compressor.CarryExchange``), which builds the augmented
boundaries and the kernel's state table at the call site and passes them as arguments.  The
quantized indexer is another backend: it owns a metadata class of its own (a subclass of this
one) and rebuilds the indexer group with its own contract.  Nothing here knows about
quantization.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Self

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.override import _IS_A5
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround

__all__ = [
    "AicpuGeometry",
    "AicpuMetadata",
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
    """What the indexer's two kernels are called with: forward and backward.

    ``fwd`` is what the indexer grids its scores with -- the metadata of whichever indexer
    backend built this one -- and ``bwd`` what ``sparse_lightning_indexer_kl_loss_grad``
    consumes.  A layer with no compressed axis runs no indexer, so both are ``None`` there.
    """

    fwd: torch.Tensor | None = None
    bwd: torch.Tensor | None = None


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
class AicpuMetadata:
    """This geometry's AI_CPU operator outputs, grouped by the kernel that launches them.

    They depend only on the frames of one forward and the layer's static geometry -- never on
    the operands -- so they are computed once per microbatch (per rank under context parallelism)
    at the eager boundary and read by every layer of the geometry instead of being re-run per
    layer inside the compiled region.  The metadata describes the frames it was computed for, which
    is why a re-cut frame (the CP transform's rank-local chunks) rebuilds them.
    """

    geometry: AicpuGeometry
    sparse_attn: AscSparseAttentionMetadata
    lightning_indexer: AscLightningIndexerMetadata


@dataclass(kw_only=True, slots=True)
class AscV4Metadata(DeepSeekV4Metadata):
    """The fused metadata: the model's frames, plus this geometry's AscendC AI_CPU metadata.

    The frames and the doc-chunk plan are the model's (they describe the tensors, not the
    kernels); this class adds what the fused kernels need on top of them:

    - ``aicpu_metadata``: the operator outputs computed for exactly these frames at the eager
      boundary and rebuilt per rank under context parallelism;
    - ``a3_workaround``: arch22's compaction of the two key streams, because every fused kernel
      right-aligns its mask at the end of the key frame it is given.

    Consumers read the fields directly (``cmp_k.seqused``, ``aicpu_metadata.sparse_attn.bwd``);
    one model runs one selection width, so there is nothing to look up.
    """

    aicpu_metadata: AicpuMetadata
    """This geometry's AI_CPU metadata, grouped by the kernel that reads them."""

    a3_workaround: A3KeyWorkaround | None = None
    """The two key streams as arch22 must be handed them, or ``None`` where the kernels read the
    model's own spelling (arch35, and every metadata that was not built under ``if not _IS_A5:``).

    Built once per microbatch beside the AI_CPU metadata, so the forward, the backward and the indexer read
    one frame and one gather each instead of re-deriving them per layer.
    """

    indexer_ratios: ClassVar[tuple[int, ...]] = (4,)
    """The ratios whose layers run an indexer, and so carry its forward metadata.

    V4 selects at ratio four only: a window-only layer has no compressed axis, and an HCA layer
    addresses its whole causal prefix instead of selecting.
    """

    @classmethod
    def build_aicpu_metadata(cls, metadata: DeepSeekV4Metadata, *, geometry: AicpuGeometry) -> Self:
        """This geometry's AscendC AI_CPU metadata over one metadata's frames (upstream #4897).

        The frames are the metadata's own (``q`` / ``swa_k`` / ``cmp_k``); because the operator
        outputs describe the frames they were computed for, a re-cut metadata -- the CP transform's
        rank-local chunks -- runs this again rather than reusing the outputs of the frame it came
        from.  A stack whose indexer metadata adds fields subclasses this and wraps the result.
        """
        ratio = metadata.ratio
        q, swa_k, cmp_k = metadata.q, metadata.swa_k, metadata.cmp_k
        cmp_tokens = int(cmp_k.cu_seqlens[-1]) if cmp_k is not None else 0
        if cmp_k is not None and cmp_tokens == 0:
            raise ValueError(
                f"batch has no complete compression block for ratio={ratio}; doc-packed sequences "
                "must be long enough to produce at least one full block."
            )
        # arch22 anchors its mask at the key frame's end and reads no ``seqused``, so its key
        # frames are the per-document reach this rank may read.  The workaround is built here,
        # once per microbatch (the walk inside it synchronises once), and the metadata keeps it.
        workaround = None
        if not _IS_A5:
            workaround, swa_k, cmp_k = A3KeyWorkaround.build(swa_k=swa_k, cmp_k=cmp_k)
        from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import (
            kernel_aicpu_metadata as indexer_metadata,
        )
        from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
            kernel_aicpu_metadata as smla_metadata,
        )

        has_indexer = ratio in cls.indexer_ratios
        fwd, bwd = indexer_metadata(q=q, cmp_k=cmp_k, ratio=ratio, geometry=geometry, has_indexer=has_indexer)
        smla_fwd, smla_bwd = smla_metadata(
            q=q, swa_k=swa_k, cmp_k=cmp_k, ratio=ratio, geometry=geometry, has_indexer=has_indexer
        )
        aicpu = AicpuMetadata(
            geometry=geometry,
            sparse_attn=AscSparseAttentionMetadata(fwd=smla_fwd, bwd=smla_bwd),
            lightning_indexer=AscLightningIndexerMetadata(fwd=fwd, bwd=bwd),
        )
        result = cls(
            ratio=ratio,
            q=q,
            swa_k=swa_k,
            cmp_k=cmp_k,
            position_k=metadata.position_k,
            xchg=metadata.xchg,
            aicpu_metadata=aicpu,
            a3_workaround=workaround,
        )
        _mark_dynamic(result)
        return result

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        # The seam dispatches on ``type(metadata)``, so this hook's contract is its own class;
        # widening the parameter would accept a rejecting metadata and lose this geometry's AI_CPU metadata.
        attention_metadata: "AscV4Metadata",
        *,
        permutation: torch.Tensor | None,
    ) -> "AscV4Metadata":
        """This rank's frames and its AscendC AI_CPU metadata (upstream #4617).

        The rank-local metadata comes from the pre-shard one with the per-document layout
        (``DeepSeekV4Metadata.per_doc_cp_metadata``); the AI_CPU operator outputs do not carry over,
        because they describe the frames they were computed for, so they are rebuilt by the same
        builder that built the global ones.

        A subclass that states its own ``build_aicpu_metadata`` overrides this hook as well (or
        passes its own ``build=``): the hook names ``AscV4Metadata``'s builder explicitly, so a
        subclass that inherits this one rebuilds with the parent's builder, not its own.
        """
        local = attention_metadata.rebuild_cp_metadata(
            permutation,
            build=partial(AscV4Metadata.build_aicpu_metadata, geometry=attention_metadata.aicpu_metadata.geometry),
        )
        # The rank-local entry is a new object: its frames, ``position_k`` slice and exchange
        # tables are none the global build marked, so mark them here; ZigZag's ``head`` / ``tail``
        # are marked by their own builders.
        _mark_dynamic(local)
        return local


def _mark_dynamic(metadata: Any) -> None:
    """Mark the rank-local entry's batch-dependent dimensions; the metadata owns the rule."""
    metadata.mark_dynamic()


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

if AicpuMetadata not in SUPPORTED_NODES:
    register_dataclass(AicpuMetadata, serialized_type_name=f"{AicpuMetadata.__module__}.{AicpuMetadata.__name__}")

if AscV4Metadata not in SUPPORTED_NODES:
    register_dataclass(AscV4Metadata, serialized_type_name=f"{AscV4Metadata.__module__}.{AscV4Metadata.__name__}")
