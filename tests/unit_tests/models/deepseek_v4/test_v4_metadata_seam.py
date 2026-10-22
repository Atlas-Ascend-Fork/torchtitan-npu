# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract of DeepSeek-V4's backend-owned metadata seam.

The sparse-attention core's own config builds what it consumes (upstream #4897): the
reference tier the packed contract, the fused AscendC tier that contract plus this
geometry's kernel records.  The decoder builds one entry per backend geometry for the
whole microbatch, and the fused entry's own ``prepare_cp_metadata`` re-derives this
rank's copy (upstream #4617).  The test keeps the contract's two ends -- per-geometry
entries and the rank-local rebuild -- and drops the per-field detail.
"""

import pytest
import torch

from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
from torchtitan_npu.models.deepseek_v4.sharding import set_deepseek_v4_sharding_config
from torchtitan_npu.override.deepseek_v4.metadata import AicpuGeometry, AscV4Metadata
from torchtitan_npu.override.deepseek_v4.sparse_attn import asc
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscCompressedSparseInnerAttention
from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context_set

pytestmark = pytest.mark.usefixtures("dsv4")

_DIM, _HEADS, _HEAD_DIM = 32, 4, 8
_INDEX_N_HEADS, _INDEX_HEAD_DIM, _INDEX_TOPK, _WINDOW = 2, 8, 16, 128


def _positions_for(doc_lens) -> torch.Tensor:
    return torch.cat([torch.arange(length) for length in doc_lens]).unsqueeze(0)


def _model_config(ratios: tuple[int, ...], *, ascendc: bool, geometry: dict | None = None):
    config = _make_v4_config(
        dim=_DIM,
        n_layers=len(ratios),
        vocab_size=64,
        n_heads=_HEADS,
        head_dim=_HEAD_DIM,
        rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=ratios,
        window_size=_WINDOW,
        norm_eps=1e-6,
        index_n_heads=_INDEX_N_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
        index_topk=_INDEX_TOPK,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=_WINDOW * 2,
        compress_rope_theta=10000.0,
        original_seq_len=_WINDOW * 2,
        num_mtp_layers=0,
    )
    # The sharding pass runs at config resolution, before the overrides.
    set_deepseek_v4_sharding_config(config, enable_sp=False, enable_ep=False)
    if ascendc:
        for _, inner, parent, attr in list(config.traverse(CompressedSparseInnerAttention.Config)):
            setattr(
                parent,
                attr,
                asc(
                    inner,
                    **(geometry or {
                        "num_heads": _HEADS,
                        "head_dim": _HEAD_DIM,
                        "index_num_heads": _INDEX_N_HEADS,
                        "index_head_dim": _INDEX_HEAD_DIM,
                    }),
                ),
            )
    return config


def _entry(model, ratio: int = 4, positions=None):
    metadata = model._get_attention_metadata(_positions_for((_WINDOW, _WINDOW)) if positions is None else positions)
    matches = [entry for entry in metadata.values() if entry.ratio == ratio]
    assert len(matches) == 1
    return matches[0]


def test_entries_are_built_per_geometry():
    """One entry per (backend class, ratio), and every layer dispatches by its own key."""
    model = _model_config((1, 1, 4, 128), ascendc=False).build()

    metadata = model._get_attention_metadata(_positions_for((128, 128)))

    assert {ratio for _, ratio in metadata} == {1, 4, 128}
    entry = metadata[(CompressedSparseInnerAttention.Config, 4)]
    assert type(entry) is DeepSeekV4Metadata and entry.ratio == 4
    assert entry.varlen.cu_seq_q.tolist() == [0, 128, 256]
    # Only the pooled geometries carry a compressed plan.
    assert entry.plan.cu_seqlens_cmp_k is not None
    assert metadata[(CompressedSparseInnerAttention.Config, 1)].plan.cu_seqlens_cmp_k is None
    keys = {layer.attention.attention_metadata_key for layer in model.layers.values()}
    assert {key[1] for key in keys} == {1, 4, 128}
    assert all(metadata[key].ratio == key[1] for key in keys)


def test_the_fused_entry_carries_its_geometry_and_records(dsv4):
    """The fused core's config builds the packed contract and its kernel records."""
    dsv4.cann_ops.calls.clear()
    model = _model_config((1, 4, 128), ascendc=True).build()

    metadata = model._get_attention_metadata(_positions_for((128, 128, 128, 128)))

    geometry = AicpuGeometry(
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        window_size=_WINDOW,
        index_topk=_INDEX_TOPK,
        index_num_heads=_INDEX_N_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
    )
    for (cls, ratio), entry in metadata.items():
        assert cls is AscCompressedSparseInnerAttention.Config and entry.ratio == ratio
        assert isinstance(entry, AscV4Metadata)
        assert entry.asc is not None and entry.asc.geometry == geometry
        assert entry.asc.sparse_attn.fwd is not None and entry.asc.sparse_attn.bwd is not None
        if ratio == 4:
            assert entry.asc.lightning_indexer.fwd is not None
        else:
            assert entry.asc.lightning_indexer.fwd is None
    smla = [call for call in dsv4.cann_ops.calls if call[0] == "sparse_flash_mla_metadata"]
    assert len(smla) == 3
    assert {call[2]["cmp_ratio"] for call in smla} == {1, 4, 128}


class _FakeCPMesh:
    """The 1-D mesh contract the metadata's CP transform reads."""

    ndim = 1
    device_type = "cpu"

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _prepare_for_rank(attention_metadata, positions, *, rank: int, cp_size: int = 2):
    with cp_shard_context_set(
        cp_mesh=_FakeCPMesh(rank, cp_size),
        load_balancer_type=None,
        positions=positions,
    ):
        return type(attention_metadata).prepare_cp_metadata(attention_metadata, permutation=None)


def test_the_fused_cp_transform_rebuilds_rank_local_metadata(dsv4, monkeypatch):
    """The CP half rebuilds this rank's plan and records from its own frames."""
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    built: list[torch.Tensor] = []

    def recording_smla_metadata(*args, **kwargs):
        built.append(kwargs["cu_seqlens_q"])
        return torch.tensor([kwargs["cu_seqlens_q"][-1].item()], dtype=torch.int32)

    monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", recording_smla_metadata)
    positions = _positions_for((_WINDOW, _WINDOW // 2))
    model = _model_config((1, 4), ascendc=True).build()

    global_metadata = _entry(model, 4, positions)

    rank_metadata = [_prepare_for_rank(global_metadata, positions, rank=rank) for rank in range(2)]
    for metadata in rank_metadata:
        assert int(metadata.varlen.cu_seq_q[-1]) == (_WINDOW + _WINDOW // 2) // 2
        assert metadata.ratio == 4
        assert metadata.asc.sparse_attn.fwd is not global_metadata.asc.sparse_attn.fwd
        assert metadata.asc.geometry == global_metadata.asc.geometry
        assert metadata.plan.cu_seqlens_cmp_k is not global_metadata.plan.cu_seqlens_cmp_k
    assert rank_metadata[0].plan.gather_indices.tolist() != rank_metadata[1].plan.gather_indices.tolist()
