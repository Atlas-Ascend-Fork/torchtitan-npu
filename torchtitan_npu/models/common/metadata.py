# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ragged coordinate frame the DeepSeek attention stacks are addressed with.

Every operand of these stacks is a packed row of documents, so the operator that reads one is
told where each document starts and, for a consumer that reads only a prefix of its operand,
how far into each document it goes.  ``KernelFrame`` is that description: one per addressed
tensor, carried by the model-dir attention metadata (``models/deepseek_v4/metadata.py`` and the
V4.1 entry) so the reference core and a fused port grid one set of boundaries.  A vendor backend
only *binds* the frames its kernels take (``cu_seqlens_*`` / ``seqused_*`` arguments); it does
not define what a frame is.

The frame is a pytree node so it travels with the metadata through the compile boundary, exactly
like the tensors it describes.
"""

from dataclasses import dataclass

import torch
from torch.utils._pytree import SUPPORTED_NODES, register_dataclass

__all__ = ["KernelFrame"]


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """One tensor's coordinate frame: the boundaries needed to address it.

    ``cu_seqlens`` are the ragged document boundaries (``int32``, ``n_documents + 1``), the form
    the kernels take as ``cu_seqlens_*``.  ``seqused`` is the per-document prefix this frame's
    consumer addresses -- how an operator is told to read a prefix of its operand -- and
    ``residual`` the trailing partial group of a compressed axis (``None`` where the axis is not
    compressed; a backend whose kernels need a value substitutes zeros for it).
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


if KernelFrame not in SUPPORTED_NODES:
    register_dataclass(KernelFrame, serialized_type_name=f"{KernelFrame.__module__}.{KernelFrame.__name__}")
