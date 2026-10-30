# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the folded Qwen3.5 GDN override and its CP refusal."""

import pytest
import torch

_CP_REFUSAL = "not migrated to the folded token layout"


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _reference_varlen_conv(x_TD, weight_C1W, cu_seqlens):
    """Independent per-document causal depthwise conv + SiLU oracle."""
    num_channels, _, kernel_width = weight_C1W.shape
    output = torch.empty_like(x_TD)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        channels_T = x_TD[start:end].transpose(0, 1)
        padded_C_1T = torch.nn.functional.pad(channels_T.unsqueeze(0), (kernel_width - 1, 0))
        convolved = torch.nn.functional.conv1d(padded_C_1T, weight_C1W, None, groups=num_channels)
        output[start:end] = torch.nn.functional.silu(convolved).squeeze(0).transpose(0, 1)
    return output


@pytest.mark.parametrize("offsets", ([0, 7], [0, 3, 7], [0, 1, 2, 7]))
def test_folded_varlen_conv_matches_per_document_reference(offsets):
    from torchtitan_npu.override.qwen3_5.gated_delta import _causal_conv1d_varlen_folded

    torch.manual_seed(0)
    x_TD = torch.randn(7, 3, dtype=torch.float64)
    weight_C1W = torch.randn(3, 1, 4, dtype=torch.float64)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)

    expected = _reference_varlen_conv(x_TD, weight_C1W, cu_seqlens)

    torch.testing.assert_close(_causal_conv1d_varlen_folded(x_TD, weight_C1W, cu_seqlens), expected)


def test_folded_varlen_conv_rejects_other_kernel_widths():
    from torchtitan_npu.override.qwen3_5.gated_delta import _causal_conv1d_varlen_folded

    x_TD = torch.randn(4, 2)
    weight_C1W = torch.randn(2, 1, 3)

    with pytest.raises(ValueError, match="kernel=4, stride=1, padding=0, dilation=1"):
        _causal_conv1d_varlen_folded(x_TD, weight_C1W, torch.tensor([0, 4], dtype=torch.int32))


def test_folded_kernel_adapter_feeds_one_packed_row(monkeypatch):
    from torchtitan_npu.override.qwen3_5 import gated_delta

    captured = {}

    def fake_run_gdn(xq, xk, xv, g, beta, reset=None):
        captured.update(q=xq, k=xk, v=xv, g=g, beta=beta, reset=reset)
        return xv

    monkeypatch.setattr(gated_delta, "run_gdn", fake_run_gdn)
    kernel = gated_delta.TritonGatedDeltaKernel(gated_delta.TritonGatedDeltaKernel.Config())
    xq_THK = torch.randn(7, 2, 4)
    xk_THK = torch.randn(7, 2, 4)
    xv_THV = torch.randn(7, 2, 4)
    g_TH = torch.randn(7, 2)
    beta_TH = torch.randn(7, 2)

    output = kernel(xq_THK, xk_THK, xv_THV, g_TH, beta_TH, cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32))

    assert output.shape == xv_THV.shape
    assert captured["q"].shape == (1, 7, 2, 4)
    assert captured["v"].shape == (1, 7, 2, 4)
    assert captured["g"].shape == (1, 7, 2)
    # Document starts (0 and 3) reset the recurrence; the op wants [batch, tokens].
    assert torch.equal(
        captured["reset"],
        torch.tensor([[True, False, False, True, False, False, False]]),
    )


def test_folded_kernel_adapter_passes_single_stream_without_reset(monkeypatch):
    from torchtitan_npu.override.qwen3_5 import gated_delta

    captured = {}

    def fake_run_gdn(xq, xk, xv, g, beta, reset=None):
        captured.update(q=xq, reset=reset)
        return xv

    monkeypatch.setattr(gated_delta, "run_gdn", fake_run_gdn)
    kernel = gated_delta.TritonGatedDeltaKernel(gated_delta.TritonGatedDeltaKernel.Config())
    xq_THK = torch.randn(7, 2, 4)

    output = kernel(xq_THK, xq_THK, xq_THK, torch.randn(7, 2), torch.randn(7, 2))

    assert output.shape == xq_THK.shape
    assert captured["q"].shape == (1, 7, 2, 4)
    assert captured["reset"] is None


@pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
)
def test_context_parallel_gated_delta_factory_refuses_unmigrated_path():
    from torchtitan_npu.override.qwen3_5 import gated_delta

    with pytest.raises(NotImplementedError, match=_CP_REFUSAL):
        gated_delta.context_parallel(None)


@pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
)
def test_context_parallel_gated_delta_forward_refuses_unmigrated_path():
    from torchtitan_npu.override.qwen3_5 import gated_delta

    with pytest.raises(NotImplementedError, match=_CP_REFUSAL):
        gated_delta.ContextParallelGatedDeltaNet.forward(object(), torch.zeros(4, 8))
