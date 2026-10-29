# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract tests for the patched ``FSDPParam.all_gather_inputs``.

The patch relaxes the uneven-sharding check to padded dim 0 for multi-input
pre-quantize payloads (upstream https://github.com/pytorch/pytorch/issues/202040,
fix proposed in https://github.com/pytorch/pytorch/pull/202101); fake FSDP
internals pin that contract on CPU without a process group or NPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState

import torchtitan_npu.patches.torch.distributed.fsdp as fsdp_patch

# 5 logical rows over a 2-rank mesh: rank 0 holds 3 (== padded), rank 1 holds
# 2; the fakes below stand in for rank 1, local (2, 4) vs padded (3, 4).
PADDED = torch.Size((3, 4))


def _fake_param(local, padded=PADDED, *, state=ShardedState.SHARDED, param_dtype=None):
    seen = []
    mesh = SimpleNamespace(size=lambda: 2)
    return (
        SimpleNamespace(
            _sharded_local_tensor=local,
            _sharded_param_data=local,
            padded_sharded_param_size=padded,
            sharded_state=state,
            param_dtype=param_dtype,
            offload_to_cpu=False,
            device=local.device,
            shard_mesh_from_root=mesh,
            _orig_size=torch.Size((5, 4)),
            _contiguous_orig_stride=(4, 1),
            _module_info=SimpleNamespace(module=None),
            mp_policy=None,
            mesh_info=SimpleNamespace(mesh=mesh),
            _extensions_data=SimpleNamespace(all_gather_metadata=None, all_gather_input_sizes=None),
            _assert_in_states=lambda *states: seen.append(states),
        ),
        seen,
    )


class TestCheckAllGatherInputs:
    """``_check_all_gather_inputs`` enforces the padded-buffer all-gather contract."""

    def test_even_sharding_skips_the_check(self):
        even = torch.randn(3, 4)
        param, _ = _fake_param(even, padded=torch.Size((3, 4)))
        # Logical size == padded size, unpadded inputs accepted by early return.
        assert fsdp_patch._check_all_gather_inputs(param, (torch.randn(2, 4),)) is None

    def test_single_input_must_match_full_padded_shape(self):
        param, _ = _fake_param(torch.randn(2, 4))
        assert fsdp_patch._check_all_gather_inputs(param, (torch.randn(3, 4),)) is None
        with pytest.raises(AssertionError, match="padded sharded size"):
            fsdp_patch._check_all_gather_inputs(param, (torch.randn(2, 4),))

    def test_multi_inputs_only_need_padded_dim0(self):
        """The relaxation vs native: quantized payloads carry per-component
        trailing shapes (scales), only dim 0 is shared with the weight shard."""
        param, _ = _fake_param(torch.randn(2, 4))
        inputs = (torch.randn(3, 4), torch.randn(3, 1))
        assert fsdp_patch._check_all_gather_inputs(param, inputs) is None

    def test_multi_input_with_short_dim0_raises(self):
        param, _ = _fake_param(torch.randn(2, 4))
        with pytest.raises(AssertionError, match="dim0"):
            fsdp_patch._check_all_gather_inputs(param, (torch.randn(3, 4), torch.randn(2, 1)))


class TestPatchedAllGatherInputs:
    """The patched property: native fallback for plain parameters, pre-hook
    contract for tensor subclasses, delegation to the original otherwise."""

    def test_plain_param_fallback_casts_to_param_dtype(self):
        local = torch.randn(3, 4)
        param, seen = _fake_param(local, padded=torch.Size((3, 4)), param_dtype=torch.bfloat16)
        out = fsdp_patch._patched_all_gather_inputs(param)
        assert seen == [(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)]
        # Native returns the shard data without flattening here (the pre-hook
        # branch is the one that views to 1D).
        assert len(out) == 1
        assert out[0].dtype is torch.bfloat16
        assert out[0].shape == torch.Size((3, 4))

    def test_plain_param_fallback_without_param_dtype_keeps_dtype(self):
        param, _ = _fake_param(torch.randn(3, 4, dtype=torch.float16), padded=torch.Size((3, 4)))
        (out,) = fsdp_patch._patched_all_gather_inputs(param)
        assert out.dtype is torch.float16

    def test_five_arg_pre_hook_flattens_and_records_sizes(self):
        calls = []

        def pre(mesh, outer_size, outer_stride, module, mp_policy):
            calls.append((mesh, outer_size, outer_stride, module, mp_policy))
            return (torch.full((3, 4), 7.0), torch.ones(3, 1)), ("meta",)

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        out = fsdp_patch._patched_all_gather_inputs(param)
        assert calls == [(param.shard_mesh_from_root, torch.Size((5, 4)), (4, 1), None, None)]
        assert [t.numel() for t in out] == [12, 3]
        assert torch.equal(out[0], torch.full((12,), 7.0))
        assert param._extensions_data.all_gather_input_sizes == [torch.Size((3, 4)), torch.Size((3, 1))]
        assert param._extensions_data.all_gather_metadata == ("meta",)

    def test_five_arg_pre_hook_unpadded_payload_raises(self):
        def pre(mesh, outer_size, outer_stride, module, mp_policy):
            return (torch.randn(2, 4),), ()

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        with pytest.raises(AssertionError, match="padded sharded size"):
            fsdp_patch._patched_all_gather_inputs(param)

    def test_one_arg_pre_hook_is_passed_through_unchecked(self):
        """Old mesh-only signature: kept for BC, native skips the size check."""

        def pre(mesh):
            return (torch.randn(2, 4),), ()

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        out = fsdp_patch._patched_all_gather_inputs(param)
        assert [t.numel() for t in out] == [8]

    def test_invalid_hook_signature_raises(self):
        def pre(mesh, outer_size):
            return (mesh,), ()

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        with pytest.raises(AssertionError, match="Invalid fsdp_pre_all_gather"):
            fsdp_patch._patched_all_gather_inputs(param)

    def test_non_sharded_state_delegates_to_original(self, monkeypatch):
        param, _ = _fake_param(torch.randn(2, 4), state=ShardedState.SHARDED_POST_FORWARD)
        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_ALL_GATHER_INPUTS_FGET", lambda self: "original")
        assert fsdp_patch._patched_all_gather_inputs(param) == "original"


class TestApplyWiring:
    """Importing the patch module runs ``apply()``; ``FSDPParam.all_gather_inputs``
    itself must be the replacement, so the cases below read through the live
    class property rather than the module-level patched function."""

    def test_apply_replaced_the_upstream_property(self):
        prop = FSDPParam.all_gather_inputs
        assert isinstance(prop, property)
        assert prop.fget is fsdp_patch._patched_all_gather_inputs
        # The pre-patch native property is kept under a separate name: the swap
        # really happened, apply() is not a no-op on the class.
        assert prop is not fsdp_patch._ORIGINAL_ALL_GATHER_INPUTS
        assert fsdp_patch._ORIGINAL_ALL_GATHER_INPUTS.fget is not fsdp_patch._patched_all_gather_inputs

    def test_padded_multi_input_payload_passes_through_the_class_property(self):
        """Native rejects this payload (scale trailing shape != padded param
        shape); passing through the class property pins the patched behavior."""

        def pre(mesh, outer_size, outer_stride, module, mp_policy):
            return (torch.full((3, 4), 7.0), torch.ones(3, 1)), ()

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        out = FSDPParam.all_gather_inputs.__get__(param)
        assert [t.numel() for t in out] == [12, 3]

    def test_unpadded_payload_raises_through_the_class_property(self):
        def pre(mesh, outer_size, outer_stride, module, mp_policy):
            return (torch.randn(2, 4),), ()

        local = torch.randn(2, 4)
        local.fsdp_pre_all_gather = pre
        param, _ = _fake_param(local)
        with pytest.raises(AssertionError, match="padded sharded size"):
            FSDPParam.all_gather_inputs.__get__(param)
