# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Own the debug-router DTensor compatibility hook.

This extension activates before the trainer and preserves rank-local debug routing.
"""

import torch
import torchtitan.models.common.moe
from torch.distributed.tensor import DTensor

_original_debug_routing = torchtitan.models.common.moe.TokenChoiceTopKRouter._debug_force_load_balance_routing


def _debug_routing(self, scores_TE: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(scores_TE, DTensor):
        return _original_debug_routing(self, scores_TE)
    expert_ids_TK, topk_scores_TK = _original_debug_routing(self, scores_TE.to_local())
    return (
        DTensor.from_local(expert_ids_TK, scores_TE.device_mesh, scores_TE.placements, run_check=False),
        DTensor.from_local(topk_scores_TK, scores_TE.device_mesh, scores_TE.placements, run_check=False),
    )


def apply() -> None:
    torchtitan.models.common.moe.TokenChoiceTopKRouter._debug_force_load_balance_routing = _debug_routing


apply()
