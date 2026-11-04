# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placeholder for Nightly-layer cases; inheritance lives in .github/ci/suites.json."""

from tests.integration_tests import OverrideDefinitions


def build_test_list() -> list[OverrideDefinitions]:
    """Return no cases until the Nightly-layer NPU validation is complete."""
    return []
