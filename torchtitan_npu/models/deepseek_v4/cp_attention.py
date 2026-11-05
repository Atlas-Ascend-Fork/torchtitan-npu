# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The doc-chunk layout: the movement between one rank's flatten slice and its chunks.

Shape legend: ``R = cp_size``; ``r = rank``; ``N`` the row's tokens, ``C = N // R`` the flatten
width; ``L_d`` document ``d``'s length and ``Q_d = L_d // R`` its chunk.

The dataloader packs one row; the context-parallel shard applies this layout as the **permutation**
its balancer states, so every rank receives chunk ``r`` of every document before any hidden state
exists.  That is what makes the per-document reach ``K_d = (r+1)·Q_d`` one
problem on every rank.

:meth:`ExchangeMetadata.from_permutation` inverts the order the shard applied, so the movement
between the two frames is derived from that one statement of the layout: pure, and it moves nothing
until a layer asks.  The move happens once, in the shard, for the inputs, and once per layer for
the KV side (``restore_key_stream``); the query side never moves.

Everything here is stated on python ints (the degree and the rank are host values of the run), so
no layout arithmetic ever meets a symbolic rank; the only host round-trips are the split lists the
collective requires, derived once per microbatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single

__all__ = [
    "ExchangeMetadata",
    "restore_key_stream",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class ExchangeMetadata:
    """One batch's routing between the flatten frame and the doc-chunk frame."""

    send_index: torch.Tensor
    """Reorder of this rank's rows (int64, ``[C]``): the rows addressed to peer ``i`` form one
    contiguous slice, peers in order and ascending slot order within a peer."""

    send_split: list[int]
    """``input_split_sizes``: rows this rank addresses to peer ``i``, one entry per peer."""

    recv_split: list[int]
    """``output_split_sizes``: rows this rank receives from peer ``i``.  The receiver's buffer is
    already in held-row order, so there is no scatter index and no second reorder."""

    @classmethod
    def from_permutation(cls, perm: torch.Tensor, *, cp_size: int, cp_rank: int) -> ExchangeMetadata:
        """Derive one rank's movement from the token order the shard applied.

        That order is what a layout states and all it states, so its inverse is where every rank's
        rows go: position ``q`` of the row belongs to rank ``q // C``, and the rows this rank holds
        are its slice of the permutation.  Reading them needs a rank's held rows to ascend inside
        its slice -- true of every document layout, which keeps documents in order -- and the
        movement is then index arithmetic, with no document boundaries involved.
        """
        # A layout states ``[1, N]``; the arithmetic below wants the row axis alone.
        perm = perm.reshape(-1)
        width = perm.numel() // cp_size
        mine = slice(cp_rank * width, (cp_rank + 1) * width)
        # ``perm`` is a permutation of the row's positions, so sorting it inverts it: the value at
        # position ``g`` is the slot row ``g`` landed in, hence the rank that holds it.
        dst = torch.argsort(perm)[mine] // width
        send_index = torch.argsort(dst, stable=True)
        send_split = torch.bincount(dst, minlength=cp_size)
        src = perm[mine] // width
        recv_split = torch.bincount(src, minlength=cp_size)
        return cls(send_index=send_index, send_split=send_split.tolist(), recv_split=recv_split.tolist())

    def apply(self, x: torch.Tensor, *, group) -> torch.Tensor:
        """The chunks' rows back onto this rank's flatten slice.

        A permutation, so the adjoint *is* the inverse: the reply is scattered through
        ``send_index`` and nothing accumulates.  Differentiable, so the key side's gradient goes
        back to the doc-chunk rows that produced it.
        """
        if group is None:
            raise RuntimeError("the doc-chunk layout has no process group; it must be wired before a sharded forward")
        if x.size(0) != 1:
            raise ValueError(f"the doc-chunk layout addresses one packed row per batch, got {tuple(x.shape)}.")
        received = all_to_all_single(
            x.squeeze(0),
            output_split_sizes=self.send_split,
            input_split_sizes=self.recv_split,
            group=group,
        )
        out = torch.empty_like(received)
        out[self.send_index] = received
        return out.unsqueeze(0)


def restore_key_stream(xchg: ExchangeMetadata | None, x: torch.Tensor, *, group) -> torch.Tensor:
    """The stream this entry's key projections read: the input's, or this rank's window.

    At one degree the layout left the row in place, so the input *is* the stream; under context
    parallelism it is the window this rank's chunks are restored onto.  A consumer hands over the
    stream as the layout produced it and never branches on the degree.
    """
    if xchg is None:
        return x
    return xchg.apply(x, group=group)
