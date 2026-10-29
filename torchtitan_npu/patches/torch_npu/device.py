# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which Ascend architecture this process runs on, stated once.

The fused DeepSeek stacks branch on the architecture: arch22 (A3, 910) right-aligns its key
masks and compacts the key streams, arch35 (A5, 950) reads the frames as the model states them,
and arch22 cannot compile a region that contains a device row count.  Every layer that needs the
fact reads it from here, so there is one detector rather than one per consumer: ``models`` may
depend on ``patches`` (it does so for several upstream shims), while ``override`` may not be
reached from below.
"""

import torch_npu

__all__ = ["is_a5"]


def is_a5() -> bool:
    """Whether the current device is an A5 (arch35, ``Ascend950``) part.

    Queried per call rather than at import: a process may import this before it has selected a
    device, and a run without one (CPU unit tests) must answer "not A5" instead of raising.
    """
    try:
        return torch_npu.npu.get_device_name().startswith("Ascend950")
    except (AttributeError, TypeError, RuntimeError):
        return False
