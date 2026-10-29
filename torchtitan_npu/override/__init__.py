# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in NPU overrides for TorchTitan."""

from torchtitan_npu.patches.torch_npu.device import is_a5

_IS_A5 = is_a5()
"""Whether this run is on an A5 (arch35) part.  The detector lives in ``patches`` because the
model layer needs the same fact and may not import ``override``."""
