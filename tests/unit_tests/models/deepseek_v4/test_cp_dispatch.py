# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torchtitan.config import apply_overrides

from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel
from torchtitan_npu.models.deepseek_v4.cp_dispatch import CPAttention, CPIndexer
from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_stripped
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


@pytest.mark.parametrize("model", ["v4", "v41"])
def test_configured_cp_attention_preserves_checkpoint_and_backend_metadata(monkeypatch, model):
    monkeypatch.setattr(AuxLoss, "_group_counts", AuxLoss._group_counts.copy())
    baseline = deepseek_v4_debugmodel(num_mtp_layers=1) if model == "v4" else deepseek_v4_1_stripped()
    family = "deepseek_v4" if model == "v4" else "deepseek_v4_1"
    baseline.override.imports = [
        f"torchtitan_npu.override.{family}.sparse_attn.asc",
        f"torchtitan_npu.override.{family}.lightning_indexer.asc",
    ]
    candidate = copy.deepcopy(baseline)
    candidate.parallelism.context_parallel_degree = 2
    candidate.parallelism.context_parallel_load_balancer = "doc_zigzag"
    for config in (baseline, candidate):
        config.model_spec.model.update_from_config(config=config)
        apply_overrides(config.override, config)
    base_config = baseline.model_spec.model.layers[2].attention
    cp_config = candidate.model_spec.model.layers[2].attention
    with torch.random.fork_rng(devices=[]):
        original = base_config.build()
        attention = cp_config.build()
    for i, value in enumerate(original.state_dict().values()):
        value.fill_((i % 7 - 3) / 8 if value.is_floating_point() else 0)
    original_state = original.state_dict()
    attention.load_state_dict(original_state, strict=True)
    actual_state = attention.state_dict()
    assert actual_state.keys() == original_state.keys()
    for name, value in original_state.items():
        torch.testing.assert_close(actual_state[name], value, rtol=0, atol=0)
    assert attention.attention_metadata_key == original.attention_metadata_key
    assert attention.attention_metadata_key == cp_config.inner_attention.attention_metadata_key
    assert isinstance(attention.inner_attention, CPAttention)
    assert isinstance(attention.indexer.inner_indexer, CPIndexer)
    if model == "v4":
        with torch.device("meta"):
            depth = candidate.model_spec.model.mtp_layers[0].attention.build()
        assert isinstance(depth.inner_attention, CPAttention)
