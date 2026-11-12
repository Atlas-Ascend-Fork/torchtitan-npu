# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for DeepEP overlap configuration at the GraphTrainer entry point."""

from types import SimpleNamespace

import pytest

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.ep_overlap_deepep_guard import (
    disable_ep_overlap_for_deepep,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.override.common.token_dispatcher import AscDeepEPTokenDispatcher, asc_deepep
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import graph_trainer_runtime_context
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher


def _config(*, backend="deepep", pipeline="npu_auto_overlap", enabled=True, npu_override=False):
    config = GraphTrainerEx.Config(model_spec=model_registry("debugmodel", moe_comm_backend=backend))
    config.compile.ep_overlap.enabled = enabled
    config.compile.pass_pipeline = pipeline
    if npu_override:
        for layer in config.model_spec.model.layers:
            if layer.moe is not None:
                routed = layer.moe.routed_experts
                routed.token_dispatcher = asc_deepep(routed.token_dispatcher)
    return config


@pytest.mark.parametrize("npu_override", [False, True], ids=["deepep", "asc-deepep"])
@pytest.mark.parametrize(
    ("pipeline", "expected"),
    [
        ("npu_auto_overlap", "default"),
        ("mutation-functionalization+npu_auto_overlap", "mutation-functionalization"),
        ("default", "default"),
        ("custom-pipeline", "custom-pipeline"),
    ],
    ids=["auto", "mutation-auto", "default", "custom"],
)
def test_deepep_guard_reaches_parent_with_base_pipeline(monkeypatch, caplog, npu_override, pipeline, expected):
    config = _config(pipeline=pipeline, npu_override=npu_override)
    dispatchers = [layer.moe.routed_experts.token_dispatcher for layer in config.model_spec.model.layers if layer.moe]
    expected_type = AscDeepEPTokenDispatcher.Config if npu_override else DeepEPTokenDispatcher.Config
    assert dispatchers and all(isinstance(dispatcher, expected_type) for dispatcher in dispatchers)
    calls = []

    def runtime_patch_spy():
        calls.append("runtime")

    def parent_init_stub(self, received):
        # Isolate device/model allocation, while observing the real entry point's handoff.
        calls.append((received, received.compile.ep_overlap.enabled, received.compile.pass_pipeline))

    monkeypatch.setattr(graph_trainer_runtime_context, "apply", runtime_patch_spy)
    monkeypatch.setattr(TrainerEx, "__init__", parent_init_stub)

    GraphTrainerEx(config)

    assert calls == [(config, False, expected)]
    assert "Disabling compile.ep_overlap" in caplog.text
    assert all(
        layer.moe.routed_experts.token_dispatcher is dispatcher
        for layer, dispatcher in zip(
            [layer for layer in config.model_spec.model.layers if layer.moe], dispatchers, strict=True
        )
    )


@pytest.mark.parametrize(
    ("backend", "enabled"),
    [("standard", True), ("deepep", False)],
    ids=["all-to-all", "already-disabled"],
)
def test_guard_preserves_other_configs_at_runtime_and_parent(monkeypatch, caplog, backend, enabled):
    config = _config(backend=backend, enabled=enabled)
    received_states = []

    def observe_state():
        received_states.append((config.compile.ep_overlap.enabled, config.compile.pass_pipeline))

    def parent_init_stub(self, received):
        assert received is config
        observe_state()

    monkeypatch.setattr(graph_trainer_runtime_context, "apply", observe_state)
    monkeypatch.setattr(TrainerEx, "__init__", parent_init_stub)

    GraphTrainerEx(config)

    assert received_states == [(enabled, "npu_auto_overlap"), (enabled, "npu_auto_overlap")]
    assert "Disabling compile.ep_overlap" not in caplog.text


def test_deepep_guard_is_idempotent(caplog):
    config = _config()

    disable_ep_overlap_for_deepep(config)
    disable_ep_overlap_for_deepep(config)

    assert config.compile.ep_overlap.enabled is False
    assert config.compile.pass_pipeline == "default"
    assert caplog.text.count("Disabling compile.ep_overlap") == 1


@pytest.mark.parametrize(
    "model_spec", [None, SimpleNamespace(model=SimpleNamespace(layers=[]))], ids=["no-model", "empty-layers"]
)
def test_guard_preserves_pipeline_without_moe_layers(model_spec, caplog):
    config = GraphTrainerEx.Config(model_spec=model_spec)
    config.compile.ep_overlap.enabled = True
    config.compile.pass_pipeline = "npu_auto_overlap"

    disable_ep_overlap_for_deepep(config)

    assert config.compile.ep_overlap.enabled is True
    assert config.compile.pass_pipeline == "npu_auto_overlap"
    assert "Disabling compile.ep_overlap" not in caplog.text
