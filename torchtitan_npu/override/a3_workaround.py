# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The key gather arch22 ("A3", ``ascend910b``/``910_93``) kernels need, and the frames they read.

The model states a key stream the way a reference attention states it: one ragged grid
(``cu_seqlens``), the per-document prefix a consumer addresses (``seqused``) and, for a pooled axis,
the trailing partial group (``residual``).  A TND kernel of arch22 reads that differently -- it
ignores ``seqused`` and right-aligns its mask at the end of the key frame it is given -- and it
insists on a ``residual`` even when there is no partial group.

``a3_key_frame`` states the reach as the frame; ``A3KeyWorkaround`` holds the ``index_select``
indices that compact the gathered rows onto it.  The frames themselves are not kept here: the fused
entry stores them as its own key frames, so the kernels' options, the records and the rows are one
answer.  The workaround is built once per microbatch in ``build_attention_metadata`` -- inside
``if not _IS_A5:`` -- and read by the forward, the backward and the selector.  Two consequences are
deliberate:

* the walk that produces the indices ends in ``nonzero``, whose row count is a device value, so it
  synchronises once; building it here pays that once per microbatch instead of once per layer; and
* arch35 never builds one, so the model's own frames are what its kernels read, unchanged.

``A3KeyWorkaround.build`` returns the object and the frames it belongs to."""

from dataclasses import dataclass

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.patches.torch_npu.device import is_a5

__all__ = ["A3KeyWorkaround", "a3_key_frame", "a3_key_gather"]


def a3_key_frame(
    cu_seqlens: torch.Tensor,
    seqused: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The frame arch22 needs for a stream whose per-document reach is ``seqused``.

    ``seqused`` is the per-document reach -- how far into each document this rank's queries can look
    -- and arch22 anchors its mask at the end of the key frame instead of reading it, so the reach
    has to *be* the frame: the boundaries are its running sum, which makes the frame's end this
    rank's chunk.

    ``residual`` is ``None`` for a pooled axis that is not folded (ratio 1); arch22 rejects that
    spelling, and the honest value -- no document has a partial trailing group -- is all zeros.

    Returns ``(cu_seqlens, seqused, residual)``, each of them a tensor the kernels accept.
    """
    reach = seqused.to(torch.int64)
    boundaries = torch.cat([reach.new_zeros(1), torch.cumsum(reach, 0)]).to(torch.int32)
    if residual is None:
        residual = torch.zeros_like(seqused)
    return boundaries, seqused.to(torch.int32), residual


def a3_key_gather(cu_seqlens: torch.Tensor, seqused: torch.Tensor) -> torch.Tensor:
    """The rows this rank may read, as ``index_select`` indices onto the stream's own rows.

    ``cu_seqlens`` are the stream's own coordinates -- they start at zero and label the rows it is
    packed from -- and every ``seqused[d]`` is at most document ``d``'s length.  A document keeps a
    prefix of its rows, so a row is kept exactly when it lies before its document's reach ends:
    ``start_d + seqused[d]``, expanded to one end per row.  Document order is preserved and the
    result is 1-D ``int64``; when the reach is every document in full, it is the identity.

    The mask is device tensor arithmetic, but the *number* of kept rows is a device value, so eager
    reads it back before ``nonzero`` can allocate: one synchronisation, paid once per microbatch by
    :meth:`A3KeyWorkaround.build`.  ``torch.nonzero_static`` cannot replace it here (this backend has
    no NPU kernel for it and falls back to the CPU).
    """
    # One end per row.  ``repeat_interleave`` is what keeps this flat in the stream's length: a
    # binary search per row (``bucketize``) costs 5.3 ms against 0.33 ms for this at 1M rows.
    ends = torch.repeat_interleave(cu_seqlens[:-1] + seqused, torch.diff(cu_seqlens))
    keep = torch.arange(ends.numel(), device=cu_seqlens.device) < ends
    return torch.nonzero(keep, as_tuple=True)[0]


@dataclass(frozen=True, kw_only=True, slots=True)
class A3KeyWorkaround:
    """Where each gathered key row has to be compacted, for one geometry and one rank.

    ``None`` means the frame the kernels are handed already is the gathered row, so the wrapper
    copies nothing -- which is every case at one degree and on the rank whose reach is every
    document in full.  The frames themselves live on the entry; this object only states the rows
    that have to move to match them.
    """

    swa_k_gather: torch.Tensor | None = None
    """``index_select`` indices that compact the gathered window row onto the entry's ``swa_k``."""

    cmp_k_gather: torch.Tensor | None = None
    """The same for the gathered compressed row onto the entry's ``cmp_k``."""

    @classmethod
    def build(
        cls,
        *,
        swa_k: KernelFrame,
        cmp_k: KernelFrame | None = None,
    ) -> tuple["A3KeyWorkaround", KernelFrame, KernelFrame | None]:
        """The workaround of these key frames, and the frames those rows are compacted onto.

        Returns ``(workaround, swa_k, cmp_k)``: the caller stores the frames on its entry and keeps
        the object for the wrapper.  Built once per microbatch, because the walk that produces the
        indices synchronises once.
        """
        window_frame, swa_k_gather = _frame_and_gather(swa_k)
        if cmp_k is None:
            return cls(swa_k_gather=swa_k_gather), window_frame, None
        pool_frame, cmp_k_gather = _frame_and_gather(cmp_k)
        return cls(swa_k_gather=swa_k_gather, cmp_k_gather=cmp_k_gather), window_frame, pool_frame

    @staticmethod
    def indexer_keys(keys: torch.Tensor, metadata, *, a5: bool | None = None) -> torch.Tensor:
        """The gathered index keys, compacted onto the frame this entry's selector addresses.

        ``keys`` is the gathered row -- flattened, ``[N, D]``.  On arch22 the entry states how far
        this rank may read, so the row moves onto that frame here with the same gather the
        selector's options were derived with; where the frame already is the row (arch35, one
        degree, the rank that reaches every document) nothing moves.  The gather is
        differentiable, so the teacher's gradient scatters back onto the row.
        """
        if a5 is None:
            a5 = is_a5()
        if a5:
            return keys
        workaround = getattr(metadata, "a3_workaround", None)
        assert workaround is not None, "an arch22 entry states the key workaround"
        if workaround.cmp_k_gather is None:
            return keys
        return keys.index_select(0, workaround.cmp_k_gather)


def _frame_and_gather(frame: KernelFrame) -> tuple[KernelFrame, torch.Tensor | None]:
    """One frame, converted: the arch22 frame, plus the gather when any row has to move."""
    cu_seqlens, seqused, residual = a3_key_frame(frame.cu_seqlens, frame.seqused, frame.residual)
    gather = None
    if not torch.equal(seqused, torch.diff(frame.cu_seqlens).to(torch.int32)):
        gather = a3_key_gather(frame.cu_seqlens, frame.seqused)
    return KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused, residual=residual), gather


if A3KeyWorkaround not in SUPPORTED_NODES:
    register_dataclass(
        A3KeyWorkaround,
        serialized_type_name=f"{A3KeyWorkaround.__module__}.{A3KeyWorkaround.__name__}",
    )
