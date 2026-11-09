# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit routed-expert storage policy for decoder FSDP setup."""

from dataclasses import replace

from .fsdp import apply_fsdp_to_decoder


def shard_experts_across_replicas(model, dp_mesh, *, edp_mesh_dims, **kwargs):
    """Fold the replica axis only for routed experts managed by decoder FSDP."""
    if edp_mesh_dims is None:
        raise ValueError("Expert replica sharding requires explicit expert mesh dimensions")
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        edp_mesh_dims=replace(edp_mesh_dims, shard=("dp_replicate", "efsdp"), replicate=None),
        **kwargs,
    )
