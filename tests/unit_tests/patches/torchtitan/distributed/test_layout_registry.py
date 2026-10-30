# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CP layout registry: the shard names a layout, the model package registers it.

The registry is the shard's half of the seam -- it names no model -- and the layout class is the
model's half.  These tests pin the two things that make that split safe: the model's import
registers its layout, and a name nobody registered is refused instead of silently falling through
to the pinned balancers.
"""

from dataclasses import dataclass

import pytest
import torch

from torchtitan_npu.patches.torchtitan.distributed.context_parallel import (
    LAYOUT_BALANCERS,
    ContextParallelLoadBalancer,
    check_layout_supported,
    register_layout,
)


class _TestLayout(ContextParallelLoadBalancer):
    """A second layout, owned by this test: only its identity matters here."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancer.Config):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def generate_permutation(self) -> torch.Tensor:
        return torch.arange(0, dtype=torch.int64).unsqueeze(0)


def test_the_model_package_registers_its_layout():
    """Importing the model is what makes ``doc_chunk`` a name the shard can find."""
    import torchtitan_npu.models.deepseek_v4.cp_layout  # noqa: F401  imported for its registration

    assert LAYOUT_BALANCERS["doc_chunk"].__name__ == "DocChunkCPLoadBalancer"


def test_a_stack_refuses_a_layout_it_does_not_describe():
    """The support check is the stack's matrix, not the registry's contents."""
    import torchtitan_npu.models.deepseek_v4.cp_layout  # noqa: F401  imported for its registration

    check_layout_supported("a stack", "doc_chunk", supported=("doc_chunk",))  # a no-op when it matches
    with pytest.raises(ValueError, match="shards by one of"):
        check_layout_supported("a stack", None, supported=("doc_chunk",))
    with pytest.raises(ValueError, match="shards by one of"):
        check_layout_supported("a stack", "headtail", supported=("doc_chunk",))


def test_a_taken_name_cannot_be_rebound_to_another_class():
    """Two models claiming one CLI name would shard the same run two ways."""
    import torchtitan_npu.models.deepseek_v4.cp_layout as layouts

    register_layout("a-layout", _TestLayout)
    register_layout("a-layout", _TestLayout)  # the same class twice is a no-op
    with pytest.raises(ValueError, match="already registered"):
        register_layout("a-layout", layouts.DocChunkCPLoadBalancer)
    finally_pop = LAYOUT_BALANCERS.pop("a-layout")
    assert finally_pop is _TestLayout


def test_a_name_can_only_hold_a_load_balancer():
    """The registry is typed: a class that cannot state a permutation is refused up front."""
    with pytest.raises(TypeError, match="not a ContextParallelLoadBalancer"):
        register_layout("a-non-layout", object)
