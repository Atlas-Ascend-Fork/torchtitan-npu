# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.components.optimizer import ParamGroupConfig

from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer_recovery import build_optimizer_recovery
from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx


def make_optimizer(accepted):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    manager = Mock()
    manager._pg = ProcessGroupHCCLEx(recovery_timeout=timedelta(seconds=1))
    manager.should_commit.return_value = accepted
    ft = SimpleNamespace(manager=manager, process_group=manager._pg, use_async_quorum=False)
    config = TorchFTOptimizersContainerEx.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 0.1,
                    "betas": (0.0, 0.0),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            )
        ],
        implementation="for-loop",
    )
    optimizer = config.build(model_parts=[model], ft_manager=ft)
    return model, optimizer, manager


def test_accepted_step_updates_parameter_once_and_advances_scheduler_once(monkeypatch):
    model, optimizer, manager = make_optimizer(True)
    monkeypatch.setattr("torch_npu.npu.current_stream", lambda: Mock(synchronize=Mock()))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizers[0], step_size=1, gamma=0.5)
    hooks = []
    optimizer.register_step_pre_hook(lambda *_: hooks.append("step"))

    assert optimizer.hccl_recovery is manager._pg.recovery
    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    scheduler.step()

    torch.testing.assert_close(model.weight, torch.tensor([[0.9]]))
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.05
    assert hooks == ["step"]
    manager.should_commit.assert_called_once_with()
    manager._begin_step.assert_called_once_with()


def test_rejected_step_skips_optimizer_and_preserves_upstream_scheduler_advancement():
    model, optimizer, manager = make_optimizer(False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizers[0], step_size=1, gamma=0.5)
    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()

    optimizer.step()
    with pytest.warns(UserWarning, match=r"lr_scheduler\.step"):
        scheduler.step()

    torch.testing.assert_close(model.weight, torch.ones(1, 1))
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.05
    assert not optimizer.hccl_recovery.optimizer_update_completed
    manager.should_commit.assert_called_once_with()


def _recovery_validation_worker(rank, store, corrupt):
    from unittest.mock import patch

    import torch.distributed as dist
    from torchtitan.distributed.flex_shard.dist_muon import DistMuon
    from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer

    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        muon = object.__new__(DistMuon)
        muon.param_groups = [{"params": [torch.nn.Parameter(torch.ones(1))]}]
        checks = []

        def validate():
            checks.append(True)
            signature = torch.tensor(rank if corrupt else 7)
            peers = [torch.empty_like(signature) for _ in range(2)]
            dist.all_gather(peers, signature)
            if any(value.item() != signature.item() for value in peers):
                raise RuntimeError("optimizer bucket plans differ across ranks")

        muon._validate_plan_across_ranks = validate
        container = object.__new__(TorchFTOptimizersContainerEx)
        container.optimizers = [muon]
        container._optimizer_recovery = build_optimizer_recovery(container.optimizers)
        container.hccl_recovery = Mock()
        container._transaction_manager = Mock()

        def load(_container, state):
            muon.state = state
            muon._validate_plan_across_ranks()

        with patch.object(TorchFTOptimizersContainer, "load_state_dict", load):
            if rank == 0:
                container.load_state_dict({"momentum": torch.tensor([3.0])})
                assert muon.state["momentum"].item() == 3.0
        assert not checks
        assert muon._validate_plan_across_ranks is validate
        with patch.object(TorchFTOptimizersContainer, "zero_grad", lambda *_args, **_kwargs: None):
            if corrupt:
                with pytest.raises(RuntimeError, match="bucket plans differ"):
                    container.zero_grad()
            else:
                container.zero_grad()
                assert len(checks) == 1  # Donor rank must validate too.
                assert not container._optimizer_recovery._pending
                container.zero_grad()
                assert len(checks) == 1  # No new load, no repeated validation.
            container._transaction_manager._begin_step.assert_called()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("corrupt", [False, True], ids=["matching-plans", "corrupt-plan"])
def test_rank_local_healing_validates_muon_on_all_replica_ranks(tmp_path, corrupt):
    torch.multiprocessing.start_processes(
        _recovery_validation_worker,
        args=((tmp_path / "store").as_uri(), corrupt),
        nprocs=2,
        start_method="fork",
    )
