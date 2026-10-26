# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU coverage for the compare_tensors non-finite branch (issue #173).

Locks the contract that NaN/Inf inputs return structured non-finite stats and
FAIL via ``failed_logits_metrics`` instead of raising, while finite and
integer paths keep their exact metrics.
"""

import torch
import pytest

from scripts.numerical_alignment.utils import metrics

pytestmark = pytest.mark.tooling


def test_non_finite_candidate_reports_stats_and_fails():
    result = metrics.compare_tensors(
        torch.tensor([1.0, float("nan"), 2.0]),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert result["non_finite"] == 1
    assert result["candidate_nan_count"] == 1
    assert result["candidate_inf_count"] == 0
    assert result["reference_nan_count"] == 0
    assert result["reference_inf_count"] == 0
    assert result["first_non_finite_index"] == 1
    assert metrics.failed_logits_metrics(result) == ("non_finite",)


def test_non_finite_reference_reports_stats_and_fails():
    result = metrics.compare_tensors(
        torch.tensor([1.0, 2.0, float("inf")]),
        torch.tensor([1.0, float("inf"), 3.0]),
    )
    assert result["non_finite"] == 1
    assert result["candidate_inf_count"] == 1
    assert result["reference_inf_count"] == 1
    # candidate side is inspected first when both sides are non-finite
    assert result["first_non_finite_index"] == 2
    assert metrics.failed_logits_metrics(result) == ("non_finite",)


def test_finite_path_metrics_unchanged():
    result = metrics.compare_tensors(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.5]),
    )
    assert result["mean_ae"] == pytest.approx(0.25)
    assert result["max_re"] == pytest.approx(0.2)
    assert result["mean_re"] == pytest.approx(0.1)
    assert result["cos"] == pytest.approx(0.9965457582448797)


def test_integer_path_unchanged():
    result = metrics.compare_tensors(
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 4]),
    )
    assert result["exact_match"] == 0
    assert result["mismatch_count"] == 1
    assert result["mismatch_rate"] == pytest.approx(1 / 3)
