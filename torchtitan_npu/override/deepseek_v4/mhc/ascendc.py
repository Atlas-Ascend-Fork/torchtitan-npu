# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 MHC pre/post with NPU operators."""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch_npu

import torchtitan_npu.ops.ascendc.mhc  # noqa: F401
from torchtitan_npu.models.deepseek_v4.mhc import HcPost, HcPre


@lru_cache(maxsize=1)
def _mhc_pre_sinkhorn_accepts_hf32() -> bool:
    """Whether the installed ``mhc_pre_sinkhorn`` schema carries ``enable_hf32``.

    ops-transformer (2026-11) appended ``Tensor? pre_mix`` and the keyword-only
    ``bool enable_hf32=True`` next to the legacy nine positional arguments;
    older CANN packages reject the kwarg, so probe once before passing it.
    """
    schema = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn.default._schema
    return any(arg.name == "enable_hf32" for arg in schema.arguments)


def _aclnn_allow_hf32() -> bool:
    """The run-wide ACLNN HF32 switch; falls back to ``True`` when absent.

    ``True`` matches the behavior of the packages that predate the switch,
    mirroring the fallback ops-transformer itself applies.
    """
    try:
        return bool(torch_npu.npu.aclnn.allow_hf32)
    except AttributeError:
        return True


class AscHcPre(HcPre):
    """HcPre backed by the hardware-specific AscendC implementation.

    The modern model-dir ``HcPre`` owns its mixing parameters and calls
    ``forward(x)``. Both Ascend A3 and A5 run the fused
    ``cann_ops_transformer.mhc_pre_sinkhorn`` path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        kwargs: dict[str, bool] = {}
        if _mhc_pre_sinkhorn_accepts_hf32():
            # Follow the run-wide ACLNN HF32 switch instead of the schema
            # default so that set_allow_hf32(False) also covers the fused mHC.
            kwargs["enable_hf32"] = _aclnn_allow_hf32()
        h_in, h_post, h_res, *_ = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(
            x,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
            self.norm_eps,
            True,
            **kwargs,
        )
        h_res = h_res.reshape(*x.shape[:2], self.hc_mult, self.hc_mult)
        return h_in.to(dtype), h_post, h_res


class AscHcPost(HcPost):
    """HcPost backed by ``cann_ops_transformer.ops.mhc_post``."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)
