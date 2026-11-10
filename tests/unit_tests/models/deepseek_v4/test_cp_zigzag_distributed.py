# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ZigZag KV restoration, gather and adjoints on real two-rank Gloo."""

import pytest
import torch
import torch.distributed as dist


def _gather_worker(rank, rendezvous, kind):
    from types import SimpleNamespace
    from unittest.mock import patch

    import spmd_types as spmd
    from torch.distributed.device_mesh import DeviceMesh
    from torchtitan.distributed.parallel_dims import MeshAxisName
    from torchtitan.distributed.spmd_types import set_current_spmd_mesh
    from torchtitan.protocols import module as framework
    from torchtitan.protocols.module import Module
    from torchtitan.protocols.sharding import ShardingConfig, SpmdLayout

    from torchtitan_npu.models.deepseek_v4.cp_attention import ExchangeMetadata
    from torchtitan_npu.models.deepseek_v4.cp_dispatch import CPAttention, CPIndexer
    from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata
    from torchtitan_npu.patches.torchtitan.distributed.context_parallel import build_load_balancer

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = DeviceMesh("cpu", torch.arange(2), mesh_dim_names=("cp",))
        held = torch.tensor([0, 4, 3, 7] if rank == 0 else [1, 5, 2, 6], dtype=torch.float32)
        q = held.reshape(1, 4, 1, 1).clone().requires_grad_()
        source = (held + 1).reshape(1, 4, 1).requires_grad_()
        row = DeepSeekV4Metadata.from_positions(torch.arange(4).repeat(2).unsqueeze(0), 1)
        permutation = build_load_balancer(
            "doc_zigzag", seq_len=8, cp_size=2, attention_metadata=row
        ).generate_permutation()
        plan = ExchangeMetadata.from_permutation(permutation, cp_size=2, cp_rank=rank)
        key = plan.apply(source, group=dist.group.WORLD)
        torch.testing.assert_close(key.flatten(), torch.arange(rank * 4 + 1, rank * 4 + 5).float())
        weights = torch.ones(1, 4, 1, requires_grad=True)
        seen = []

        class PackedAttentionOracle(Module):
            def forward(self, q, swa_k, attn_sink, attention_metadata, **kwargs):
                seen.append(swa_k)
                torch.testing.assert_close(kwargs["topk_indices"], q[..., 0].int())
                torch.testing.assert_close(kwargs["topk_scores"], q[..., 0])
                return q * (swa_k.square().sum() / 2) * attention_metadata.factor + kwargs["topk_scores"].unsqueeze(-1)

        class PackedIndexerOracle(Module):
            def forward(self, idx_q_BLHiDi, idx_k_BNDi, idx_w_BLHi, *, attention_metadata, candidates_BL1C):
                seen.append(idx_k_BNDi)
                scores = (idx_q_BLHiDi[..., 0] * (idx_k_BNDi.square().sum() / 2) + idx_w_BLHi) * attention_metadata.factor
                torch.testing.assert_close(candidates_BL1C, idx_q_BLHiDi.int())
                return idx_q_BLHiDi[..., 0].int(), scores, candidates_BL1C

        inner = PackedAttentionOracle() if kind == "attention" else PackedIndexerOracle()
        key_name = "swa_k" if kind == "attention" else "idx_k_BNDi"
        inner._sharding_config = ShardingConfig(
            in_src_shardings={key_name: SpmdLayout({MeshAxisName.CP: spmd.S(1)})},
            in_dst_shardings={key_name: SpmdLayout({MeshAxisName.CP: spmd.R})},
        )
        module = CPAttention(inner) if kind == "attention" else CPIndexer(inner)
        module.parallelize(SimpleNamespace(spmd_backend="spmd_types"))
        metadata = SimpleNamespace(head=SimpleNamespace(factor=1), tail=SimpleNamespace(factor=3))
        real_redistribute = framework.spmd_redistribute_per_axis
        with set_current_spmd_mesh(mesh), patch.object(
            framework, "spmd_redistribute_per_axis", wraps=real_redistribute
        ) as redistribution:
            if kind == "attention":
                scores = held.reshape(1, 4, 1).clone().requires_grad_()
                output = module(
                    q, key, torch.zeros(1), attention_metadata=metadata,
                    topk_indices=scores.detach().int(), topk_scores=scores,
                )
            else:
                candidates = q.detach().int()
                indices, output, returned = module(
                    q, key, weights, attention_metadata=metadata, candidates_BL1C=candidates
                )
                assert returned is candidates
                torch.testing.assert_close(indices, held.reshape(1, 4, 1).int())
            assert redistribution.call_count == 1
            upstream = (held + 1).reshape_as(output)
            output.backward(upstream)

        factor = torch.tensor([1, 1, 3, 3], dtype=torch.float32).reshape_as(output)
        expected = (held.reshape_as(output) * 102 + (1 if kind == "indexer" else 0)) * factor
        if kind == "attention":
            expected = expected + held.reshape_as(output)
            torch.testing.assert_close(scores.grad, upstream.reshape_as(scores), rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert len(seen) == 2 and seen[0].data_ptr() == seen[1].data_ptr()
        for gathered in seen:
            torch.testing.assert_close(gathered.flatten(), torch.arange(1, 9).float(), rtol=0, atol=0)
        torch.testing.assert_close(q.grad, (102 * factor * upstream).reshape_as(q), rtol=0, atol=0)
        # Head contributions sum to 52; tail contributions sum to 3 * 116.
        torch.testing.assert_close(source.grad.flatten(), 400 * (held + 1), rtol=0, atol=0)
        if kind == "indexer":
            torch.testing.assert_close(weights.grad, factor * upstream, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("kind", ["attention", "indexer"])
def test_cp_dispatch_gathers_once_before_both_halves_and_reduces_shared_gradient(tmp_path, kind):
    torch.multiprocessing.start_processes(
        _gather_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", kind),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
