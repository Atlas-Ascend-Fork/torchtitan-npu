# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin the fused mHC forward call to the installed ``mhc_pre_sinkhorn`` schema.

ops-transformer (2026-11) appended an optional ``pre_mix`` and the keyword-only
``enable_hf32=True`` to ``mhc_pre_sinkhorn``.  Deployments still run both
schemas: the override passes ``enable_hf32`` only when the installed schema
carries it (older packages reject the kwarg outright) and sources its value
from the run-wide ACLNN HF32 switch, so ``set_allow_hf32(False)`` also covers
the fused mHC path instead of silently falling back to the schema default.

The kernels are CPU stand-ins: they only pin how the forward dispatches; the
ACLNN kernels themselves are covered on hardware.
"""

import itertools

import pytest
import torch
import torch_npu

from torchtitan_npu.override.deepseek_v4.mhc import ascendc

LEGACY_SCHEMA = (
    "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
    "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
KEYWORD_ONLY_SCHEMA = (
    "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
    "int hcMult, int numIters, float hcEps, float normEps, bool outFlag, "
    "Tensor? pre_mix=None, *, bool enable_hf32=True) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)

# Dropping a torch.library.Library handle deregisters its ops.
_LIBRARIES = []
_NS_COUNTER = itertools.count()


def _declare(schema):
    """Register ``mhc_pre_sinkhorn`` with ``schema`` and a CPU stand-in kernel."""
    namespace = f"ttn_mhc_{next(_NS_COUNTER)}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    lib.define(schema)

    def mhc_pre_sinkhorn(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag, pre_mix=None, enable_hf32=True
    ):
        batch, seq = x.shape[:2]
        saved = (
            torch.zeros(batch, seq, hc_mult),
            torch.zeros(batch, seq, hc_mult * hc_mult + 2 * hc_mult),
            torch.zeros(batch, seq, 1),
            torch.zeros(2 * num_iters, batch, seq, hc_mult),
            torch.zeros(2 * num_iters, batch, seq, hc_mult, hc_mult),
        )
        return x.clone(), torch.ones(batch, seq, hc_mult), torch.ones(batch, seq, hc_mult * hc_mult), *saved

    lib.impl("mhc_pre_sinkhorn", mhc_pre_sinkhorn, "CPU")
    _LIBRARIES.append(lib)
    return getattr(torch.ops, namespace).mhc_pre_sinkhorn


class _RecordingPacket:
    """A stand-in for the CANN packet that records the dispatched kwargs.

    Keeps ``.default`` (with ``_schema``) so the schema probe reads the pinned
    schema while the forward call itself is recorded.
    """

    def __init__(self, packet, calls):
        self.default = packet.default
        self._packet = packet
        self._calls = calls

    def __call__(self, *args, **kwargs):
        self._calls.append(kwargs)
        return self._packet(*args, **kwargs)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    yield
    ascendc._mhc_pre_sinkhorn_accepts_hf32.cache_clear()


def _make_hc_pre(dim: int = 8, hc_mult: int = 2):
    config = ascendc.AscHcPre.Config(dim=dim, hc_mult=hc_mult, sinkhorn_iters=2, eps=1e-6, norm_eps=1e-6)
    return ascendc.AscHcPre(config)


@pytest.mark.parametrize(
    "schema,accepts",
    [
        pytest.param(LEGACY_SCHEMA, False, id="legacy"),
        pytest.param(KEYWORD_ONLY_SCHEMA, True, id="keyword_only"),
    ],
)
def test_probe_matches_schema(schema, accepts, monkeypatch):
    """The probe reports exactly whether ``enable_hf32`` is in the schema."""
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "mhc_pre_sinkhorn", _declare(schema), raising=False)
    ascendc._mhc_pre_sinkhorn_accepts_hf32.cache_clear()
    assert ascendc._mhc_pre_sinkhorn_accepts_hf32() is accepts


def test_forward_omits_hf32_on_legacy_schema(monkeypatch):
    """Packages without ``enable_hf32`` must keep receiving the legacy call."""
    calls = []
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "mhc_pre_sinkhorn",
        _RecordingPacket(_declare(LEGACY_SCHEMA), calls),
        raising=False,
    )
    ascendc._mhc_pre_sinkhorn_accepts_hf32.cache_clear()

    hc_pre = _make_hc_pre()
    h_in, h_post, h_res = hc_pre.forward(torch.randn(2, 3, 8))

    assert calls == [{}]
    assert h_in.shape == (2, 3, 8)
    assert h_post.shape == (2, 3, 2)
    assert h_res.shape == (2, 3, 2, 2)


def test_forward_follows_aclnn_hf32_on_new_schema(monkeypatch):
    """With ``enable_hf32`` in the schema, its value is the ACLNN switch."""
    calls = []
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "mhc_pre_sinkhorn",
        _RecordingPacket(_declare(KEYWORD_ONLY_SCHEMA), calls),
        raising=False,
    )
    ascendc._mhc_pre_sinkhorn_accepts_hf32.cache_clear()

    monkeypatch.setattr(torch_npu.npu.aclnn, "allow_hf32", False, raising=False)
    hc_pre = _make_hc_pre()
    hc_pre.forward(torch.randn(2, 3, 8))
    assert calls == [{"enable_hf32": False}]

    monkeypatch.setattr(torch_npu.npu.aclnn, "allow_hf32", True, raising=False)
    hc_pre.forward(torch.randn(2, 3, 8))
    assert calls[-1] == {"enable_hf32": True}


def test_aclnn_hf32_falls_back_to_true(monkeypatch):
    """Without the ACLNN switch (older torch_npu), keep the pre-switch default."""
    monkeypatch.delattr(torch_npu.npu, "aclnn", raising=False)
    assert ascendc._aclnn_allow_hf32() is True
