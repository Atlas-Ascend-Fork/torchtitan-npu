# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ratio-4 overlap's cross-rank block: an autograd-capable ``batch_send_recv``.

Under context parallelism a rank's slab usually starts inside a document, and the ratio-4
compressor's first local block pools tokens from the predecessor rank's last block.  This
module rings that block over -- raw tokens and their positions -- and returns the augmented
stream together with the document grid clipped to the augmented window and the number of
output blocks the carry owns.  It is the *only* part of the stack that knows the compressor's
overlap continues on another rank: the compressor itself just compresses
``(x, positions, cu_seqlens)`` at every degree.

Like :class:`~torchtitan_npu.models.common.query_exchange.QueryExchange`, the process group is
the module's only state (wired by the framework's ``Module.parallelize`` recursion) and the
collectives are functional, so the backward of the prepended block is the reverse exchange.
At one degree the module is a no-op: it hands its inputs back, and the caller runs the very
same compressor instance it runs under CP.

The augmented boundaries come from the row's grid clipped to ``[slab_start - ratio, slab_end)``:
when the slab starts inside a document no boundary falls at the carry/slab seam, so the carry
completes that document's first local segment; when it starts at a document edge the row's own
boundary is kept and the carry is a segment of its own.  "Mask every segment start" is the same
rule in both cases, and the carry's output blocks are dropped by the caller, so nothing
special-cases the carry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.protocols.module import Module

__all__ = ["CarryExchange", "CarryStream"]


@dataclass(frozen=True, slots=True)
class CarryStream:
    """One rank's stream after the carry: what to compress, and what to trim afterwards."""

    x: torch.Tensor
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    trim: int
    """Compressed blocks at the head of the output that belong to the carry (0 or 1)."""


class CarryExchange(Module):
    """The predecessor's last block, prepended to this rank's slab (V4's ratio-4 overlap)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        """The layer's compression ratio; only a ratio-4 geometry has an overlapping block."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio
        self.group = None

    def parallelize(self, parallel_dims) -> None:
        """Take the CP mesh the framework hands down to every module."""
        self.group = parallel_dims.get_optional_mesh("cp")
        super().parallelize(parallel_dims)

    @property
    def active(self) -> bool:
        """Whether this run shards the sequence: at one degree the exchange is a no-op."""
        return self.group is not None and self.group.size() > 1

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> CarryStream:
        """This rank's augmented stream (``cu_seqlens`` are the **row's** boundaries)."""
        group = self.group
        if group is None or group.size() <= 1:
            return CarryStream(x=x, positions=positions, cu_seqlens=cu_seqlens, trim=0)
        if x.size(0) > 1:
            raise ValueError("CP>1 暂不支持 local_batch_size>1")
        ratio = self.ratio
        cp_size = group.size()
        cp_rank = group.get_local_rank()
        slab_len = x.size(1)
        slab_start = cp_rank * slab_len
        slab_end = slab_start + slab_len
        carried = ratio == 4 and cp_rank > 0
        if ratio == 4:
            # Every rank enters the ring, and every rank prepends what it received: on rank 0
            # that is an empty block, but the concat keeps the collective's output in the graph.
            # Dropping the empty case instead would prune the collective's *backward* on rank 0
            # only, and the rank would then wait for a gradient exchange its peer still issues.
            x_carry, positions_carry = self._exchange(
                x,
                positions,
                group=group,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
            x = torch.cat([x_carry, x], dim=1)
            positions = torch.cat([positions_carry, positions], dim=1)
            if carried:
                slab_start -= ratio
        bounds = self._bounds(cu_seqlens, slab_start, slab_end, x=x)
        return CarryStream(x=x, positions=positions, cu_seqlens=bounds, trim=1 if carried else 0)

    def _exchange(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        group,
        cp_rank: int,
        cp_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ring the predecessor's last block over, empty on the rank that starts the row.

        Every rank enters both collectives with an explicit split (the ranks that have nothing to
        send or receive pass an empty one), so the ring cannot deadlock on them.
        """
        ratio = self.ratio
        successor = cp_rank + 1
        predecessor = cp_rank - 1
        send_split = [ratio if peer == successor else 0 for peer in range(cp_size)]
        recv_split = [ratio if peer == predecessor else 0 for peer in range(cp_size)]
        if sum(send_split):
            # ``all_to_all_single`` splits **dim 0**, so the batch axis has to come off: the ring
            # moves one block of rows, not one row of blocks.
            send = x[:, -ratio:, :].flatten(0, 1).contiguous()
            send_positions = positions[..., -ratio:].reshape(-1).to(torch.int32).contiguous()
        else:
            send = x[:, :0, :].flatten(0, 1)
            send_positions = positions[..., :0].reshape(-1).to(torch.int32)
        received = all_to_all_single(
            send,
            output_split_sizes=recv_split,
            input_split_sizes=send_split,
            group=group,
        )
        received_positions = all_to_all_single(
            send_positions,
            output_split_sizes=recv_split,
            input_split_sizes=send_split,
            group=group,
        )
        return received.unsqueeze(0), received_positions.reshape(1, -1).to(positions.dtype)

    def _bounds(self, cu_seqlens: torch.Tensor, first: int, stop: int, *, x: torch.Tensor) -> torch.Tensor:
        """The row's document boundaries clipped to ``[first, stop)`` and rebased onto it."""
        host = [int(value) for value in cu_seqlens.cpu().tolist()]
        inside = [boundary - first for boundary in host if first < boundary < stop]
        return torch.tensor([0, *inside, stop - first], dtype=torch.int32, device=x.device)
