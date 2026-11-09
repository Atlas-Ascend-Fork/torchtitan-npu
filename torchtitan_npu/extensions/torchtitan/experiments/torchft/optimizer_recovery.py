# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt optimizer collective validation to rank-local TorchFT recovery."""

from contextlib import ExitStack, contextmanager

import torch
import torch.distributed as dist
from torchtitan.distributed.flex_shard.dist_muon import DistMuon


class OptimizerRecovery:
    """Defer collective checks until all replica-local ranks finish healing."""

    def __init__(self, checks):
        self._checks = checks
        self._pending = False

    def _mark_pending(self):
        self._pending = True

    @contextmanager
    def defer_validation(self):
        # Only these optimizer instances are changed, for the duration of loading.
        with ExitStack() as restore:
            for optimizer, method in self._checks:
                original = getattr(optimizer, method)
                setattr(optimizer, method, self._mark_pending)
                restore.callback(setattr, optimizer, method, original)
            yield

    def validate(self):
        if not self._checks:
            return
        optimizer, _ = self._checks[0]
        parameter = optimizer.param_groups[0]["params"][0]
        pending = torch.tensor(int(self._pending), device=parameter.device)
        # WORLD is one TorchFT replica, not the outer elastic replica group.
        dist.all_reduce(pending, op=dist.ReduceOp.MAX)
        if pending.item():
            for optimizer, method in self._checks:
                getattr(optimizer, method)()
            self._pending = False


def build_optimizer_recovery(optimizers):
    # Build on every rank, including donors that never load state during healing.
    # DistMuon rebuilds local plans during loading; only its collective check waits.
    return OptimizerRecovery(
        [(optimizer, "_validate_plan_across_ranks") for optimizer in optimizers if isinstance(optimizer, DistMuon)]
    )
