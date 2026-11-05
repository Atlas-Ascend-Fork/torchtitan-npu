# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The indexer's BF16 port: LightningIndexer with its SLIKG edge.

Its metadata tensors are sized from the frames and the geometry the attention override states, and
none of them is launched from a layer.  A backend that runs another
indexer (the quantized one) builds its own entry in its own package instead of extending this one.
"""

from dataclasses import dataclass

from torchtitan_npu.models.deepseek_v4_1.indexer import Selector
from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscIndexerMixin


class AscSelector(AscIndexerMixin, Selector):  # pyrefly: ignore [inconsistent-inheritance]
    """The fused BF16 LI/SLIKG indexer for the V4.1 indexer layers.

    It emits the kernel's own document-local ``topk_indices``, which the fused attention consumes
    in the same coordinate system; only ``-1`` marks an unused slot.  ``candidates_BL1C`` is
    accepted for the quantized selector's candidate stage and unused here.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Selector.Config):
        pass

    def forward(  # pyrefly: ignore [inconsistent-inheritance]
        self,
        idx_q_BLHiDi,
        idx_k_BNDi,
        weights_BLHi,
        attention_metadata,
        *,
        candidates_BL1C=None,
    ):
        return self._select(idx_q_BLHiDi, idx_k_BNDi, weights_BLHi, attention_metadata, candidates_BL1C=candidates_BL1C)
