# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The per-document CP layouts: the movement between one rank's flatten slice and its pieces.

Shape legend: ``R = cp_size``; ``r = rank``; ``N`` the row's tokens, ``C = N // R`` the flatten
width; ``L_d`` document ``d``'s length, and ``L_d // R`` (``doc_chunk``) or ``L_d // (2R)``
(``doc_zigzag``) its piece of that document.

The dataloader packs one row; the context-parallel shard applies a layout as the **permutation**
its balancer states, so every rank receives whole pieces of every document before any hidden state
exists -- chunk ``r`` under ``doc_chunk``, chunk ``r`` and its mirrored chunk under ``doc_zigzag``.
That is what makes the per-document reach ``K_d = (r+1)·Q_d`` one problem on every rank.

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
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

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
    """Rows held by this rank whose flat KV owner is peer ``i`` (the restore's input splits).

    ``recv_index`` puts those rows in peer order for ZigZag before the collective.
    """

    recv_index: torch.Tensor | None = None
    """Sort held query rows into flat-token order before sending them back to their KV owners.

    Doc-chunk rows are already sorted; ZigZag rows contain separate head/tail document lists.
    """

    @classmethod
    def from_permutation(cls, perm: torch.Tensor, *, cp_size: int, cp_rank: int) -> ExchangeMetadata:
        """Derive one rank's movement from the token order the shard applied.

        That order is what a layout states and all it states, so its inverse is where every rank's
        rows go: position ``q`` of the row belongs to rank ``q // C``, and the rows this rank holds
        are its slice of the permutation. Sorting the held rows groups both head and tail by
        their flat KV owner, then by document position within that owner.
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
        recv_index = torch.argsort(perm[mine])
        if torch.equal(recv_index, torch.arange(width, device=perm.device)):
            recv_index = None
        return cls(
            send_index=send_index,
            send_split=send_split.tolist(),
            recv_split=recv_split.tolist(),
            recv_index=recv_index,
        )

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
        tokens = x.squeeze(0)
        if self.recv_index is not None:
            tokens = tokens.index_select(0, self.recv_index)
        received = all_to_all_single(
            tokens,
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


if ExchangeMetadata not in SUPPORTED_NODES:
    register_dataclass(
        ExchangeMetadata, serialized_type_name=f"{ExchangeMetadata.__module__}.{ExchangeMetadata.__name__}"
    )
