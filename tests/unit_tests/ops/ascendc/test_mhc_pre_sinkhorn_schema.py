# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin the ``mhc_pre_sinkhorn`` autograd registration to torch's call conventions.

torch's ``make_autograd_impl`` calls ``setup_context`` with keyword arguments and
adds ``keyword_only_inputs`` once the schema declares any keyword-only argument.
ops-transformer added ``*, bool enable_hf32=True`` next to the optional
``pre_mix`` to ``mhc_pre_sinkhorn``, so the registration has to serve both
conventions while deployments still run the legacy nine-argument schema.

The kernels are CPU stand-ins: they only prove how the registration hands
operands to ``mhc_pre_sinkhorn_backward``; the ACLNN kernels are covered on
hardware.
"""

import pytest
import torch

from torchtitan_npu.ops.ascendc import mhc as mhc_ops

# Deployed schema (CANN 9.2.0 and earlier): no keyword-only argument, so torch
# calls ``setup_context(ctx, inputs, output)``.
LEGACY_SCHEMA = (
    "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
    "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# ops-transformer ``mhc/mhc_pre_sinkhorn`` (2026-11): ``pre_mix`` stays
# positional because ``register_autograd`` rejects keyword-only tensors, while
# ``enable_hf32`` is keyword-only, which also makes torch pass
# ``keyword_only_inputs`` to the callback.
KEYWORD_ONLY_SCHEMA = (
    "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
    "int hcMult, int numIters, float hcEps, float normEps, bool outFlag, "
    "Tensor? pre_mix=None, *, bool enable_hf32=True) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# Dropping a ``torch.library.Library`` handle deregisters its ops.
_LIBRARIES = []

# Values the stand-in forward stamps on ``h_pre`` .. ``norm_out`` so the
# backward can prove which operand slot each saved tensor came from.
_SAVED_OUTPUT_VALUES = (3.0, 4.0, 5.0, 6.0, 7.0)


@pytest.fixture
def backward_calls(monkeypatch):
    """Record ``mhc_pre_sinkhorn_backward`` calls on the CANN op namespace."""
    calls = []

    def mhc_pre_sinkhorn_backward(*args):
        calls.append(args)
        x, phi, alpha, bias = args[3:7]
        return (
            torch.zeros_like(x),
            torch.ones_like(phi),
            torch.full_like(alpha, 2.0),
            torch.full_like(bias, 3.0),
        )

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "mhc_pre_sinkhorn_backward", mhc_pre_sinkhorn_backward, raising=False
    )
    return calls


def _declare_op(namespace, schema):
    """Register ``mhc_pre_sinkhorn`` with ``schema`` and a CPU stand-in kernel."""
    lib = torch.library.Library(namespace, "FRAGMENT")
    lib.define(schema)

    def mhc_pre_sinkhorn(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag, pre_mix=None, enable_hf32=True
    ):
        batch, seq = x.shape[:2]
        n = hc_mult
        saved = (
            torch.full((batch, seq, n), _SAVED_OUTPUT_VALUES[0]),
            torch.full((batch, seq, n * n + 2 * n), _SAVED_OUTPUT_VALUES[1]),
            torch.full((batch, seq, 1), _SAVED_OUTPUT_VALUES[2]),
            torch.full((2 * num_iters, batch, seq, n), _SAVED_OUTPUT_VALUES[3]),
            torch.full((2 * num_iters, batch, seq, n, n), _SAVED_OUTPUT_VALUES[4]),
        )
        return x.clone(), torch.ones(batch, seq, n), torch.ones(batch, seq, n * n), *saved

    lib.impl("mhc_pre_sinkhorn", mhc_pre_sinkhorn, "CPU")
    _LIBRARIES.append(lib)


def _run_registered_op(namespace, schema, backward_calls):
    _declare_op(namespace, schema)
    torch.library.register_autograd(
        f"{namespace}::mhc_pre_sinkhorn",
        mhc_ops._mhc_pre_sinkhorn_backward,
        setup_context=mhc_ops._mhc_pre_sinkhorn_setup_context,
    )

    n, batch, seq, dim = 2, 2, 3, 8
    x = torch.randn(batch, seq, dim, requires_grad=True)
    phi = torch.randn((2 + n) * n, dim * n, requires_grad=True)
    alpha = torch.randn(3, requires_grad=True)
    bias = torch.randn((2 + n) * n, requires_grad=True)

    # ``numIters``, ``hcEps`` and ``normEps`` all differ so the assertions below
    # can tell a shifted operand apart from the intended one.
    outputs = getattr(torch.ops, namespace).mhc_pre_sinkhorn(x, phi, alpha, bias, n, 3, 1e-6, 2e-6, True)
    assert len(outputs) == 8
    sum(output.sum() for output in outputs).backward()

    # ``hcEps`` still lives at index 6 of the positional prefix, whatever the
    # optional tail looks like; the saved operands keep their documented order.
    assert len(backward_calls) == 1
    (call,) = backward_calls
    assert call[12] == pytest.approx(1e-6)
    assert torch.equal(call[3], x.detach())
    for offset, value in enumerate(_SAVED_OUTPUT_VALUES):
        saved = call[7 + offset]
        assert torch.equal(saved, torch.full_like(saved, value))

    # Each gradient lands on the input it was produced for.
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(phi.grad, torch.ones_like(phi))
    assert torch.equal(alpha.grad, torch.full_like(alpha, 2.0))
    assert torch.equal(bias.grad, torch.full_like(bias, 3.0))


def test_setup_context_accepts_keyword_only_inputs():
    """torch passes keyword-only args by name, next to the positional ones."""

    class _Context:
        def save_for_backward(self, *tensors):
            self.saved = tensors

        def set_materialize_grads(self, enabled):
            self.materialize_grads = enabled

        def mark_non_differentiable(self, *tensors):
            self.non_differentiable = tensors

    ctx = _Context()
    inputs = [torch.zeros(1) for _ in range(4)] + [2, 3, 1e-6, 2e-6, True, None]
    output = [torch.zeros(1) for _ in range(8)]
    mhc_ops._mhc_pre_sinkhorn_setup_context(
        ctx=ctx,
        inputs=inputs,
        keyword_only_inputs={"enable_hf32": True},
        output=output,
    )

    assert ctx.hc_eps == pytest.approx(1e-6)
    assert len(ctx.saved) == 9
    # The trailing five outputs are backward-only bookkeeping: the
    # registration must stop PyTorch from materializing their gradients.
    assert ctx.materialize_grads is False
    assert ctx.non_differentiable == tuple(output[3:])


@pytest.mark.parametrize(
    "namespace,schema",
    [
        pytest.param("tnt_mhc_legacy", LEGACY_SCHEMA, id="legacy"),
        pytest.param("tnt_mhc_keyword_only", KEYWORD_ONLY_SCHEMA, id="keyword_only"),
    ],
)
def test_autograd_registration_matches_schema_convention(namespace, schema, backward_calls):
    _run_registered_op(namespace, schema, backward_calls)
