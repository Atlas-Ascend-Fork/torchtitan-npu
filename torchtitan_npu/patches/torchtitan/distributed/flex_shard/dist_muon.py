# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Upstream-Issue: https://github.com/pytorch/torchtitan/issues/4405

"""Allow NPU-local DTensor storage in TorchTitan DistMuon."""

import torch
import torchtitan.distributed.flex_shard.dist_muon
from torch.distributed.tensor import DTensor


def _validate_parameter_storage(self) -> torch.device:
    local_devices = set()
    for group in self.param_groups:
        for param in group["params"]:
            if not isinstance(param, DTensor):
                raise TypeError("DistMuon requires DTensor parameters")
            local_device = param.to_local().device
            if local_device.type != "npu":
                raise ValueError("DistMuon requires NPU parameters")
            local_devices.add(local_device)
    if len(local_devices) != 1:
        raise ValueError("DistMuon requires one device per process")
    return local_devices.pop()


def apply() -> None:
    torchtitan.distributed.flex_shard.dist_muon.DistMuon._validate_parameter_storage = _validate_parameter_storage


apply()
