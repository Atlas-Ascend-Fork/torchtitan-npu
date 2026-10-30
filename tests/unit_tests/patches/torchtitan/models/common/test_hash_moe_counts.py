# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router
from torchtitan_npu.patches.torchtitan.models.common.moe import (
    _ClampGroupedExperts,
    _RouterScoreAbsorbingRoutedExperts,
)
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


class _RoutedExpertsBoundaryStub(torch.nn.Module):
    """Leave expert computation outside this router/statistics boundary test."""

    def forward(self, x, scores, expert_ids, counts):
        return x


def test_v4_hash_moe_keeps_upstream_balancing_buffer():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    config = model_registry("debugmodel").model.layers[0].moe
    with torch.random.fork_rng(devices=[]):
        moe = config.build()

    assert type(moe) is MoE
    assert moe.router.hash
    assert moe.load_balance_coeff == config.load_balance_coeff
    torch.testing.assert_close(moe.expert_bias_E, torch.zeros(config.num_experts))
    assert "expert_bias_E" in moe.state_dict()


def _count_worker(rank, rendezvous):
    set_spmd_backend("partial_dtensor")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        moe = MoE.Config(
            num_experts=2,
            load_balance_coeff=0.1,
            router=DeepSeekV41Router.Config(num_experts=2, gate=Linear.Config(in_features=2, out_features=2), top_k=1),
            routed_experts=_RouterScoreAbsorbingRoutedExperts.Config(
                inner_experts=_ClampGroupedExperts.Config(dim=2, hidden_dim=2, num_experts=2),
                token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1),
            ),
        ).build()
        moe.routed_experts = _RoutedExpertsBoundaryStub()
        moe.router.gate.weight = torch.nn.Parameter(distribute_tensor(torch.eye(2), mesh, [Replicate()]))
        moe.expert_bias_E = distribute_tensor(torch.zeros(2), mesh, [Replicate()])
        moe.tokens_per_expert_E = distribute_tensor(torch.zeros(2), mesh, [Replicate()])
        x = distribute_tensor(torch.tensor([[3.0, 0], [2.0, 0], [1.0, 0], [0, 2.0]]), mesh, [Replicate()])
        block = torch.nn.Module()
        block.moe_enabled, block.moe = True, moe
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict({"0": block})
        # CPU-only step: torch_npu replaces torch.optim's foreach-device probe
        # with one that queries the current NPU, so the default
        # ``foreach=None`` would initialize a driver this gloo/CPU worker does
        # not have.  The balancing hook and the update itself are unaffected.
        optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, foreach=False)
        parallel_dims = SimpleNamespace(ep_enabled=False, tp=2, get_optional_mesh=lambda name: None)
        register_moe_load_balancing_hook(optimizer, [model], parallel_dims)

        for multiplier in (1, 2):
            output = moe(x)
            torch.testing.assert_close(output.to_local(), x.to_local(), rtol=0, atol=0)
            assert isinstance(moe.tokens_per_expert_E, DTensor)
            assert moe.tokens_per_expert_E.device_mesh == mesh
            assert moe.tokens_per_expert_E.placements == (Replicate(),)
            torch.testing.assert_close(moe.tokens_per_expert_E.to_local(), torch.tensor([3.0, 1.0]) * multiplier)

        optimizer.step()

        torch.testing.assert_close(moe.expert_bias_E.to_local(), torch.tensor([-0.1, 0.1]), rtol=0, atol=0)
        torch.testing.assert_close(moe.tokens_per_expert_E.to_local(), torch.zeros(2), rtol=0, atol=0)
        assert moe.expert_bias_E.placements == (Replicate(),)
    finally:
        dist.destroy_process_group()


def test_hash_moe_keeps_dtensor_counts_and_balancing_updates_on_two_tp_ranks(tmp_path):
    mp.spawn(_count_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_moe_forwards_image_mask_to_router():
    with torch.random.fork_rng(devices=[]):
        moe = MoE.Config(
            num_experts=2,
            load_balance_coeff=0.1,
            router=DeepSeekV41Router.Config(num_experts=2, gate=Linear.Config(in_features=2, out_features=2), top_k=1),
            routed_experts=_RouterScoreAbsorbingRoutedExperts.Config(
                inner_experts=_ClampGroupedExperts.Config(dim=2, hidden_dim=2, num_experts=2),
                token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1),
            ),
        ).build()
    moe.routed_experts = _RoutedExpertsBoundaryStub()
    with torch.no_grad():
        moe.router.gate.weight.copy_(torch.eye(2))
    inputs = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    image_mask = torch.tensor([False, True])
    observed = []
    hook = moe.router.register_forward_pre_hook(lambda module, args, kwargs: observed.append(kwargs), with_kwargs=True)
    kwargs = {"image_mask": image_mask}
    try:
        output = moe(inputs, **kwargs)
        torch.testing.assert_close(output, inputs, rtol=0, atol=0)
        torch.testing.assert_close(moe.tokens_per_expert_E, torch.ones(2), rtol=0, atol=0)
    finally:
        hook.remove()
    assert len(observed) == 1
    for key, value in kwargs.items():
        assert observed[0][key] is value


def _cpu_grouped_mm(self, *, A, weight_EOI, offs):
    """CPU primitive reference; router, local_map, dispatch and expert flow stay real."""
    boundaries = [0, *offs.tolist()]
    return torch.cat(
        [
            A[start:end] @ weight.bfloat16().T
            for start, end, weight in zip(boundaries[:-1], boundaries[1:], weight_EOI, strict=True)
        ]
    )


def _parallelized_routing_worker(rank, rendezvous):
    import torchtitan.distributed.parallel_dims as parallel_dims_module
    from torchtitan.distributed.parallel_dims import ParallelDims

    from torchtitan_npu.models.deepseek_v4 import model_registry as v4_registry
    from torchtitan_npu.models.deepseek_v4.sharding import set_deepseek_v4_layer_sharding
    from torchtitan_npu.models.deepseek_v4_1 import model_registry
    from torchtitan_npu.models.deepseek_v4_1.sharding import set_v41_layer_sharding

    torch.accelerator.is_available = lambda: False
    torch.set_num_threads(1)
    parallel_dims_module.device_type = "cpu"
    set_spmd_backend("partial_dtensor")
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    original_grouped_mm = _ClampGroupedExperts._grouped_mm
    _ClampGroupedExperts._grouped_mm = _cpu_grouped_mm
    try:
        x_TD = torch.tensor([[3.0, -1.0], [2.0, 0.5], [-1.0, 2.0], [0.0, 3.0]])
        ids_T = torch.tensor([0, 2, 5, 7])
        mask_T = torch.tensor([False, True, True, False])
        gate_ED = torch.tensor([[1.0, -0.5], [-0.2, 1.0], [0.4, 0.7], [-0.8, -0.3]])
        initial_bias_E = torch.tensor([0.2, -0.1, 0.0, 0.05])
        vision_bias_E = torch.tensor([-0.6, 0.7, -0.3, 0.2])
        hash_table_VK = torch.tensor([[3, 1], [0, 2], [2, 0], [1, 3], [3, 0], [1, 2], [2, 3], [0, 1]])
        debug_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        debug_router = TokenChoiceTopKRouter.Config(
            num_experts=4, gate=Linear.Config(in_features=2, out_features=4), top_k=2
        ).build()
        debug_scores_TE = distribute_tensor(torch.arange(24.0).reshape(6, 4).requires_grad_(), debug_mesh, [Shard(0)])
        debug_ids_TK, debug_topk_TK = debug_router._debug_force_load_balance_routing(debug_scores_TE)
        # Three tokens/rank makes local round-robin restart differ from a global
        # arange on rank 1: 3*K is not divisible by the expert count.
        expected_debug_ids_TK = torch.tensor([[0, 1], [2, 3], [0, 1]])
        torch.testing.assert_close(debug_ids_TK.to_local(), expected_debug_ids_TK, rtol=0, atol=0)
        torch.testing.assert_close(
            debug_topk_TK.to_local(), debug_scores_TE.to_local().gather(-1, expected_debug_ids_TK), rtol=0, atol=0
        )
        debug_topk_TK.to_local().sum().backward()
        torch.testing.assert_close(
            debug_scores_TE.grad.to_local(), torch.zeros(3, 4).scatter_(-1, expected_debug_ids_TK, 1.0), rtol=0, atol=0
        )
        for enable_ep in (False, True):
            dims = ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=2,
                pp=1,
                ep=2 if enable_ep else 1,
                world_size=2,
                spmd_backend="partial_dtensor",
            )
            dims.build_mesh()
            mesh = dims.get_optional_mesh("tp")
            for route in ("ordinary", "hash", "vision", "debug"):
                config = MoE.Config(
                    num_experts=4,
                    load_balance_coeff=0.1,
                    router=(DeepSeekV4Router.Config if route == "hash" else DeepSeekV41Router.Config)(
                        num_experts=4,
                        gate=Linear.Config(in_features=2, out_features=4),
                        top_k=2,
                        **({"vocab_size": 8, "layer_id": 0, "n_hash_layers": 1} if route == "hash" else {}),
                        **({} if route == "hash" else {"vision_enabled": route == "vision", "sorted_topk": True}),
                        _debug_force_load_balance=route == "debug",
                    ),
                    routed_experts=_RouterScoreAbsorbingRoutedExperts.Config(
                        inner_experts=_ClampGroupedExperts.Config(dim=2, hidden_dim=4, num_experts=4),
                        token_dispatcher=(AllToAllTokenDispatcher if enable_ep else LocalTokenDispatcher).Config(
                            num_experts=4, top_k=2, absorb_router_scores=True
                        ),
                    ),
                )
                layer = (
                    v4_registry("debugmodel") if route == "hash" else model_registry("deepseek_v4_1_debugmodel_text")
                ).model.layers[0]
                layer.moe = config
                if route == "hash":
                    set_deepseek_v4_layer_sharding(layer, enable_sp=True, enable_ep=enable_ep)
                else:
                    set_v41_layer_sharding(layer, enable_sp=True, enable_ep=enable_ep)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(17)
                    moe = layer.moe.build()
                    for parameter in moe.routed_experts.parameters():
                        torch.nn.init.uniform_(parameter, -0.4, 0.4)
                with torch.no_grad():
                    moe.router.gate.weight.copy_(gate_ED)
                    moe.expert_bias_E.copy_(initial_bias_E)
                    if route == "hash":
                        moe.router.tid2eid.copy_(hash_table_VK)
                    if route == "vision":
                        moe.router.bias_vl.copy_(vision_bias_E)
                moe.parallelize(dims)

                observed = []
                routed = moe.routed_experts.token_dispatcher
                original_dispatch = routed.dispatch

                def dispatch_local(*args, observed=observed, original_dispatch=original_dispatch):
                    assert all(isinstance(value, torch.Tensor) and not isinstance(value, DTensor) for value in args)
                    observed.append(tuple(value.shape for value in args))
                    return original_dispatch(*args)

                routed.dispatch = dispatch_local
                router_outputs = []
                handle = moe.router.register_forward_hook(
                    lambda module, args, output, router_outputs=router_outputs: router_outputs.append(output)
                )
                inputs_TD = distribute_tensor(x_TD.clone().requires_grad_(), mesh, [Shard(0)])
                raw_scores_TE = torch.sigmoid(x_TD @ gate_ED.T)
                choice_bias_TE = torch.where(mask_T[:, None], vision_bias_E, initial_bias_E)
                if route == "hash":
                    expected_ids_TK = hash_table_VK[ids_T]
                elif route == "debug":
                    local_tokens = 2 if enable_ep else 4
                    expected_ids_TK = torch.arange(local_tokens * 2).reshape(local_tokens, 2) % 4
                    if enable_ep:
                        expected_ids_TK = expected_ids_TK.repeat(2, 1)
                else:
                    expected_ids_TK = (
                        (raw_scores_TE + (choice_bias_TE if route == "vision" else initial_bias_E))
                        .topk(2, dim=-1, sorted=True)
                        .indices
                    )
                expected_map_TE = torch.zeros(4, 4, dtype=torch.bool).scatter_(-1, expected_ids_TK, True)
                local_map_TE = expected_map_TE.chunk(2)[rank] if enable_ep else expected_map_TE
                expected_count_E = local_map_TE.sum(0).float()
                for multiplier in (1, 2):
                    output_TD = moe(
                        inputs_TD, **({"input_ids_T": ids_T} if route == "hash" else {"image_mask": mask_T})
                    )
                    assert torch.isfinite(output_TD.to_local()).all()
                    output_TD.to_local().float().square().sum().backward()
                    scores_TK, selected_TK, scores_TE = router_outputs[-1]
                    layout = (Shard(0),) if enable_ep else (Replicate(),)
                    assert scores_TE.placements == selected_TK.placements == scores_TK.placements == layout
                    torch.testing.assert_close(selected_TK.full_tensor(), expected_ids_TK, rtol=0, atol=0)
                    torch.testing.assert_close(scores_TK.full_tensor(), raw_scores_TE.gather(-1, expected_ids_TK))
                    assert moe.tokens_per_expert_E.placements == ((Partial(),) if enable_ep else (Replicate(),))
                    torch.testing.assert_close(moe.tokens_per_expert_E.to_local(), expected_count_E * multiplier)
                handle.remove()
                assert len(observed) == 2
                assert moe.router.gate.weight.grad is not None
                assert torch.isfinite(moe.router.gate.weight.grad.to_local()).all()
                assert torch.count_nonzero(moe.router.gate.weight.grad.to_local()) > 0
                if route == "hash":
                    assert moe.router.tid2eid.placements == (Replicate(),)
                block = torch.nn.Module()
                block.moe_enabled, block.moe = True, moe
                model = torch.nn.Module()
                model.layers = torch.nn.ModuleDict({"0": block})
                optimizer = torch.optim.SGD(moe.parameters(), lr=0.01, foreach=False, fused=False)
                register_moe_load_balancing_hook(optimizer, [model], dims)
                global_count_E = expected_map_TE.sum(0).float()
                delta_E = 0.1 * (global_count_E.mean() - global_count_E).sign()
                optimizer.step()
                torch.testing.assert_close(moe.expert_bias_E.to_local(), initial_bias_E + delta_E - delta_E.mean())
                torch.testing.assert_close(moe.tokens_per_expert_E.to_local(), torch.zeros(4))
                if route == "vision":
                    torch.testing.assert_close(moe.router.bias_vl.to_local(), vision_bias_E)
    finally:
        _ClampGroupedExperts._grouped_mm = original_grouped_mm
        dist.destroy_process_group()


def test_hash_moe_parallelized_token_metadata_dispatch_gradients_and_bias_hook_on_two_tp_ranks(tmp_path):
    mp.spawn(_parallelized_routing_worker, args=(str(tmp_path / "parallelized"),), nprocs=2, join=True)
