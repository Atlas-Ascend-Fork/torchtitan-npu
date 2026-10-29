# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4897
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4116
# Remove this module (and its activation import in
# ``torchtitan_npu/patches/torchtitan/__init__.py``) once the pinned
# ``torchtitan`` dependency carries both PRs natively.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport of upstream's decoder-owned input and attention-metadata seam.

Two upstream changes are carried here, both onto the pinned 0.3.0 decoder:

- **#4897** — attention metadata is owned by the attention backends:
  ``Decoder._get_attention_metadata`` builds one entry per distinct backend
  key, and ``Decoder.forward`` hands every layer the entry its own attention
  names.  The map is built once per microbatch, so a stack whose backends
  precompute kernel metadata (the DeepSeek stacks' CANN ``*_metadata``
  operators) pays for it once, not once per layer.
- **#4116** — the model owns the preprocessing seam:
  ``Decoder.preprocess_inputs`` turns the dataloader's ``input_dict`` into the
  forward's ``(inputs, labels, extra_kwargs)``, including the CP shard and the
  SPMD annotation.  It is the upstream home for what this repository carried
  as the model-level ``build_attention_masks`` hook, and it is what makes the
  trainer's and the validator's duplicated post-dataloading logic unnecessary
  (upstream 0.3.0 asks for exactly that dedup in a TODO).
- **#4617**, the CP half of that seam — the model owns the rank-local
  attention-metadata transform as well: ``Decoder._cp_shard`` is the upstream
  landing site, and the transform lives there instead of in the CP patch.  The
  map's entries carry this rank's frames and the kernel records built from
  them, so they are prepared after the shard, on this rank's rows.  0.3.0 has
  neither ``ParallelismContext`` nor ``shard_tensors``, so the shard inside
  ``_cp_shard`` is the pinned ``cp_shard`` entry point.

0.3.0 adaptation, because the pinned version predates #4116/#4617:
``labels`` is passed separately (0.3.0 keeps it out of ``input_dict``), and
``parallel_dims``/``device``/``parallelism`` stand in for #4116's
``parallelism_context`` (0.3.0 has no ``ParallelismContext``).  The
``build_attention_masks`` branch is transitional: it is the pre-#4116 shape of
the same seam, and disappears with the DeepSeek models' migration onto
``build_attention_metadata``.
"""

from typing import Any

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import BaseAttention, FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder

from torchtitan_npu.patches.torchtitan.distributed.context_parallel import layout_needs_positions
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    AttentionMetadataMap,
    cp_shard_context_set,
    prepare_cp_metadata_map,
    resolve_cp_layout,
)

__all__ = ["apply"]


def _get_attention_metadata(
    self,
    positions: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> dict[Any, Any]:
    """Build one metadata entry per distinct attention-backend key.

    Upstream #4897's body.  The traversal differs only in where it finds the
    inner attention: 0.3.0 has no ``InnerAttention.Config`` base to traverse
    for, so it walks the attention wrapper configs and reads their
    ``inner_attention``.  Entries whose builder returns ``None`` are absent,
    exactly as upstream.
    """
    attention_metadata: dict[Any, Any] = {}
    # MTP depths are layers too: a decoder that keeps them in their own config list (upstream's
    # ``MTPDecoder.Config.mtp_layers``) runs attention backends whose geometries the map has to
    # cover, and with one entry per geometry a depth whose ratio no main layer pools at would
    # otherwise have no entry at all.
    for layer_config in (
        layer for layers in (self.config.layers, getattr(self.config, "mtp_layers", None) or []) for layer in layers
    ):
        traverse = getattr(layer_config, "traverse", None)
        if traverse is None:
            continue
        for _, attention_config, _, _ in traverse(BaseAttention.Config):
            inner_attention = getattr(attention_config, "inner_attention", None)
            if inner_attention is None:
                continue
            key = inner_attention.attention_metadata_key
            if key in attention_metadata:
                continue
            metadata = inner_attention.build_attention_metadata(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if metadata is not None:
                attention_metadata[key] = metadata
    return attention_metadata


def forward(
    self,
    tokens: torch.Tensor,
    positions: torch.Tensor | None = None,
    attention_masks: Any = None,
    *,
    attention_metadata: AttentionMetadataMap | None = None,
):
    """Upstream #4897's forward: select each layer's metadata by its key.

    ``attention_masks`` keeps its 0.3.0 meaning (one object broadcast to every
    layer) until the stacks that still build a single mask type are migrated;
    passing ``attention_metadata`` selects per layer instead.  The rest of the
    body is 0.3.0's, unchanged.
    """
    # positions is listed before attention_masks so AutoParallel's input_fn,
    # which returns (tokens, positions) and binds them positionally, maps
    # positions to the right parameter (it would otherwise land in the
    # attention_masks slot and break the maskless SDPA backend).
    # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

    for layer in self.layers.values():
        layer_attention_metadata = (
            attention_masks
            if attention_metadata is None
            else attention_metadata.get(layer.attention.attention_metadata_key)
        )
        h = layer(h, layer_attention_metadata, positions)

    h = self.norm(h) if self.norm is not None else h

    # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
    # calls .requires_grad on all stage inputs, which fails on bool kwargs.
    # TODO: fix PP backward upstream to skip non-tensor inputs
    if self._skip_lm_head:
        return h
    output = self.lm_head(h) if self.lm_head is not None else h
    return output


def _cp_shard(
    self,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    *,
    cp_mesh: Any,
    device: torch.device,
    load_balancer_type: str | None = None,
    ptrr_mask_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Shard for CP, then prepare this rank's attention metadata (upstream #4617).

    Upstream #4116/#4617 put the CP half of the metadata seam on the model, and
    ``_cp_shard`` is its name.  The map carries this rank's frames and the
    kernel records built from them, so it is rank-dependent exactly as
    ``attention_masks`` is: its transform runs *after* the shard, on this rank's
    rows, with the row's positions the map was built from.  Upstream calls the
    backend's ``prepare_cp_metadata`` per entry because the mesh is ambient
    there; 0.3.0 hands the mesh, the load-balancer type and those positions
    along as well.

    The shard itself stays 0.3.0's: the pinned version has neither
    ``ParallelismContext`` nor ``shard_tensors``, so the pinned ``cp_shard``
    remains the entry point.

    ``load_balancer_type`` names the layout the inputs are sharded by; ``None``
    asks the stack, whose metadata declares the layout it can describe
    (``cp_layout``).  A repository layout -- ``doc_chunk`` -- is built *inside*
    ``cp_shard`` from the row's boundaries, so nothing here precomputes a
    permutation; the transform is handed none, because a rank-local entry of
    these layouts is a function of the row's boundaries alone.  A pinned layout
    (``headtail``) still hands its permutation over, as upstream does.
    """
    attention_metadata = extra_kwargs.get("attention_metadata")
    # The shard below replaces ``positions`` with this rank's slice; the
    # transform describes the row the shard started from, so keep the global
    # one before that happens.
    row_positions = extra_kwargs.get("positions")
    # Only a boundary-driven layout reads the positions: it is a function of where the documents
    # start.  The pinned ones are functions of the length alone, so a model that carries none
    # keeps sharding exactly as it did before this repository's layouts existed.
    layout = resolve_cp_layout(load_balancer_type, attention_metadata)
    if layout_needs_positions(layout) and row_positions is None:
        raise ValueError(f"the {layout!r} layout needs the row's positions: they are what it is derived from.")
    permutation = None
    if attention_metadata and layout == "headtail" and row_positions is not None:
        permutation = _HeadTailLoadBalancer(
            row_positions.shape[-1], cp_mesh.size(), cp_mesh.device_type
        )._generate_indices()

    attention_masks = extra_kwargs.get("attention_masks")
    buffers: tuple[torch.Tensor, ...] = (inputs, labels) if row_positions is None else (inputs, labels, row_positions)
    sharded, attention_masks = cp_shard(
        cp_mesh,
        buffers,
        attention_masks,
        layout,
        ptrr_mask_key=ptrr_mask_key,
        positions=row_positions,  # pyrefly: ignore [unexpected-keyword]
    )
    inputs, labels = sharded[0], sharded[1]
    if len(sharded) > 2:
        extra_kwargs["positions"] = sharded[2]
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    if attention_metadata:
        # Upstream's loop, with the pinned stand-in for its ambient parallelism
        # context published around it (see ``attention_metadata``).
        with cp_shard_context_set(
            cp_mesh=cp_mesh,
            load_balancer_type=layout,
            positions=row_positions,
        ):
            extra_kwargs["attention_metadata"] = prepare_cp_metadata_map(
                attention_metadata,
                permutation=permutation,
            )
    return inputs, labels, extra_kwargs


def preprocess_inputs(
    self,
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    parallel_dims: Any,
    device: torch.device,
    parallelism: Any,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build masks, shard for CP, and annotate SPMD types (upstream #4116).

    The metadata map is preferred when the stack's attention backends provide
    builders; the 0.3.0 single-mask path stays for the stacks whose inner
    attention is a plain Flex/Varlen config, and a model-level
    ``build_attention_masks`` override (transitional) still owns the whole
    metadata + CP step.  The CP step itself is delegated to ``_cp_shard``,
    upstream's method for it, which is also where the map becomes rank-local.
    """
    if parallelism.spmd_backend == "full_dtensor":
        # Fail before any mask build, shard or wrapping: the config is stale.
        raise ValueError(
            "upstream torchtitan removed full_dtensor's input path (the pinned 0.3.0 still accepts the "
            "field), and this seam does not support it; use spmd_types"
        )
    inputs = input_dict["input"]
    extra_kwargs: dict[str, Any] = {k: v for k, v in input_dict.items() if k != "input"}

    build_attention_masks = getattr(self, "build_attention_masks", None)
    if build_attention_masks is not None:
        # Transitional model-level seam (pre-#4116 shape): the model owns the
        # metadata construction *and* its CP sharding.
        cp_mesh = parallel_dims.get_mesh("cp") if parallel_dims.cp_enabled else None
        inputs, labels, extra_kwargs = build_attention_masks(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh=cp_mesh,
            load_balancer_type=parallelism.context_parallel_load_balancer,
        )
    else:
        positions = extra_kwargs.get("positions", None)
        if positions is not None:
            attention_metadata = self._get_attention_metadata(
                positions,
                padding_mask=extra_kwargs.get("padding_mask", None),
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if attention_metadata:
                extra_kwargs["attention_metadata"] = attention_metadata
            else:
                # positions and attention_masks are optional (Decoder.forward
                # defaults both to None). Build masks only for the masked
                # backends (Flex/Varlen), which is where get_attention_masks is
                # defined. A maskless backend (e.g. the SDPA config used by the
                # graph_trainer tests) still receives positions for RoPE but no
                # masks — it relies on is_causal instead.
                inner_attention = getattr(self.config.first_attention, "inner_attention", None)
                if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
                    extra_kwargs["attention_masks"] = self.get_attention_masks(positions=positions)

        if parallel_dims.cp_enabled:
            cp_mesh = parallel_dims.get_mesh("cp")
            inputs, labels, extra_kwargs = self._cp_shard(
                inputs,
                labels,
                extra_kwargs,
                cp_mesh=cp_mesh,
                device=device,
                load_balancer_type=parallelism.context_parallel_load_balancer,
                ptrr_mask_key=parallelism.context_parallel_ptrr_mask_key,
            )

    if parallelism.spmd_backend == "spmd_types":
        inputs, labels, extra_kwargs = annotate_input_spmd_types(parallel_dims, inputs, labels, extra_kwargs)

    return inputs, labels, extra_kwargs


def apply() -> None:
    if getattr(Decoder, "_npu_metadata_seam", False):
        return
    Decoder._get_attention_metadata = _get_attention_metadata  # pyrefly: ignore [bad-assignment]
    Decoder._cp_shard = _cp_shard  # pyrefly: ignore [bad-assignment]
    Decoder.forward = forward  # pyrefly: ignore [bad-assignment]
    Decoder.preprocess_inputs = preprocess_inputs  # pyrefly: ignore [bad-assignment]
    Decoder._npu_metadata_seam = True


apply()
