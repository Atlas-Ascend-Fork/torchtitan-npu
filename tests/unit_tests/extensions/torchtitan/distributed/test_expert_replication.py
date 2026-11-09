# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert FSDP placements on real CPU meshes with a fake process group."""

import importlib
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.trainer import Trainer

from torchtitan_npu.config import ParallelismConfig
from torchtitan_npu.extensions.torchtitan.distributed import fsdp
from torchtitan_npu.extensions.torchtitan.distributed.expert_replication import shard_experts_across_replicas
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.parallelize import _decoder_fsdp_binding

mtp = importlib.import_module("torchtitan.models.deepseek_v3.mtp")
upstream_fsdp = importlib.import_module("torchtitan.distributed.fsdp")
parallel_dims_module = importlib.import_module("torchtitan.distributed.parallel_dims")


def _config(tmp_path, *args, model_name="deepseek_v4", config_name=None):
    if config_name is None:
        config_name = "deepseek_v4_debugmodel" if model_name == "deepseek_v4" else "deepseek_v4_1_debugmodel_text"
    model_args = ["--no-engram-enabled"] if model_name == "deepseek_v4_1" else []
    return ConfigManager().parse_args([
        "--module", f"torchtitan_npu.models.{model_name}",
        "--config", config_name,
        "--hf-assets-path", str(tmp_path),
        "--training.disable-cuda-graphs",
        "--compile.no-enable",
        "--parallelism.data-parallel-shard-degree", "4",
        "--parallelism.data-parallel-replicate-degree", "2",
        "--parallelism.expert-parallel-degree", "4",
        *model_args, *args,
    ])


@pytest.fixture
def cpu_meshes(monkeypatch, tmp_path):
    monkeypatch.setattr(parallel_dims_module, "device_type", "cpu")
    dist.init_process_group("fake", rank=0, world_size=8)
    try:
        config = _config(tmp_path)
        dims = ParallelDims.from_config(config.parallelism, world_size=8)
        dims.build_mesh()
        yield dims
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_name", ["deepseek_v4", "deepseek_v4_1"], ids=["v4", "v4.1"])
@pytest.mark.parametrize("enabled", [False, True], ids=["replicated-experts", "sharded-experts"])
def test_cli_policy_sets_expert_storage_without_changing_dense_placement(
    monkeypatch, tmp_path, cpu_meshes, model_name, enabled
):
    args = ["--parallelism.disable-expert-replication"] if enabled else []
    config = _config(tmp_path, *args, model_name=model_name)
    with torch.device("meta"):
        model = config.model_spec.model.build()
    block = next(iter(model.layers.values()))
    expert = block.moe.routed_experts.inner_experts.w1_EFD
    dense = block.attention.wq_a.weight
    placements = {}

    def record_fully_shard(module, *, shard_placement_fn=None, **kwargs):
        if module is block:
            placements["expert"] = shard_placement_fn(expert)
            placements["dense"] = shard_placement_fn(dense)
        for item in module if isinstance(module, list) else [module]:
            item.set_modules_to_forward_prefetch = lambda *args, **kwargs: None
            item.set_modules_to_backward_prefetch = lambda *args, **kwargs: None

    monkeypatch.setattr(fsdp, "fully_shard", record_fully_shard)
    monkeypatch.setattr(upstream_fsdp, "fully_shard", record_fully_shard)
    # CPU checks the FSDP consumer; activation/EP compute sharding is tested on NPU.
    monkeypatch.setattr(model, "parallelize", lambda *args, **kwargs: None)
    original = mtp.apply_fsdp_to_decoder

    kwargs = {"decoder_fsdp": shard_experts_across_replicas} if enabled else {}
    config.model_spec.parallelize_fn(
        model, parallel_dims=cpu_meshes, training=config.training, parallelism=config.parallelism,
        compile_config=config.compile, ac_config=None, dump_folder=str(tmp_path), **kwargs,
    )

    assert mtp.apply_fsdp_to_decoder is original
    expert_info = placements["expert"].mesh_info
    dense_info = placements["dense"].mesh_info
    assert expert_info.shard_mesh_size == (2 if enabled else 1)
    if enabled:
        assert expert_info.replicate_mesh_dim is None
    else:
        assert expert_info.replicate_mesh_size == 2
    # V4 has 4 experts (< EP4 * FSDP2); V4.1 has 16, so it shards the expert axis.
    assert placements["expert"].placement == Shard(1 if enabled and model_name == "deepseek_v4" else 0)
    assert dense_info.shard_mesh_size == 4
    assert dense_info.replicate_mesh_size == 2
    assert placements["dense"].placement == Shard(0)


def test_default_policy_keeps_upstream_binding():
    original = mtp.apply_fsdp_to_decoder

    with _decoder_fsdp_binding(None):
        assert mtp.apply_fsdp_to_decoder is original


def test_policy_restores_binding_after_setup_failure():
    original = mtp.apply_fsdp_to_decoder

    with pytest.raises(RuntimeError, match="setup failed"), _decoder_fsdp_binding(shard_experts_across_replicas):
        raise RuntimeError("setup failed")

    assert mtp.apply_fsdp_to_decoder is original


@contextmanager
def _raises_cli_config_error(capsys, message):
    # Tyro 1.0.16 prints constructor ValueErrors and exits with status 2.
    with pytest.raises((ValueError, SystemExit)) as error:
        yield
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
        diagnostic = " ".join(capsys.readouterr().err.replace("│", " ").split())
    else:
        diagnostic = str(error.value)
    assert message in diagnostic


def test_cli_rejects_graph_trainer(tmp_path, capsys):
    with _raises_cli_config_error(capsys, "not supported by GraphTrainer"):
        _config(
            tmp_path, "--parallelism.disable-expert-replication", "--parallelism.spmd-backend", "full_dtensor",
            config_name="graph_trainer_deepseek_v4_debugmodel",
        )


def test_cli_rejects_other_models(monkeypatch, tmp_path, capsys):
    source = _config(tmp_path)
    source.model_spec.name = "other_model"
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4.config_registry")
    monkeypatch.setattr(registry, "deepseek_v4_debugmodel", lambda: source)

    with _raises_cli_config_error(capsys, "requires DeepSeek-V4"):
        _config(tmp_path, "--parallelism.disable-expert-replication")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"data_parallel_replicate_degree": 1}, "data_parallel_replicate_degree"),
        ({"expert_parallel_degree": 1}, "EP"),
        ({"spmd_backend": "partial_dtensor"}, "full_dtensor or spmd_types"),
        ({"context_parallel_degree": 2}, "TP=CP=PP=1"),
        ({"tensor_parallel_degree": 2}, "TP=CP=PP=1"),
        ({"pipeline_parallel_degree": 2}, "TP=CP=PP=1"),
    ],
    ids=["no-dense-replicas", "no-ep", "partial-dtensor", "cp", "tp", "pp"],
)
def test_rejects_unsupported_replication_configuration(kwargs, message):
    values = {"data_parallel_replicate_degree": 2, "expert_parallel_degree": 4, **kwargs}

    with pytest.raises(ValueError, match=message):
        ParallelismConfig(disable_expert_replication=True, **values)


def test_cli_rejects_muon_replica_dedup(tmp_path, capsys):
    with _raises_cli_config_error(capsys, "incompatible with muon_enable_hsdp_replica_dedup"):
        _config(
            tmp_path, "--parallelism.disable-expert-replication",
            "--optimizer.name", "Muon", "--optimizer.muon-enable-hsdp-replica-dedup",
        )


@pytest.mark.parametrize("enabled", [False, True], ids=["replicated-experts", "sharded-experts"])
def test_trainer_injects_policy_without_mutating_recipe(monkeypatch, tmp_path, enabled):
    config = _config(tmp_path, *(["--parallelism.disable-expert-replication"] if enabled else []))
    original_spec = config.model_spec
    original_parallelize = original_spec.parallelize_fn
    captured = []

    def capture(self, configured):
        captured.append(configured)
        raise RuntimeError("captured trainer input")

    monkeypatch.setattr(Trainer, "__init__", capture)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="captured trainer input"):
            TrainerEx(config)
    assert config.model_spec is original_spec
    assert config.model_spec.parallelize_fn is original_parallelize
    for configured in captured:
        if enabled:
            assert configured.model_spec is not original_spec
            assert configured.model_spec.parallelize_fn.func is original_parallelize
            assert configured.model_spec.parallelize_fn.keywords["decoder_fsdp"] is shard_experts_across_replicas
        else:
            assert configured.model_spec.parallelize_fn is original_parallelize
