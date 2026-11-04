# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-owned definitions for base, PR and Nightly training."""

from tests.integration_tests import OverrideDefinitions
from . import base, pr, nightly


def build_test_list() -> list[OverrideDefinitions]:
    return [*base.build_test_list(), *pr.build_test_list(), *nightly.build_test_list()]
