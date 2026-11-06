# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Official FLA backend, explicitly selected through ``override.imports``."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override

try:
    from fla.modules.causal_conv1d import causal_conv1d  # pyrefly: ignore [missing-import]
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # pyrefly: ignore [missing-import]
except ImportError as error:
    raise ImportError(
        "Failed to import Flash Linear Attention (FLA) for the Qwen FLA override. "
        "Install the supported flash-linear-attention and fla-core packages as documented in "
        "torchtitan_npu/models/qwen3_5/README.md."
    ) from error

from .gated_delta import ContextParallelGatedDeltaNet, GatedDeltaKernel, _npu_causal_conv, qwen3_5
from .parallelize import shard_local_heads


class FLAGatedDeltaKernel(GatedDeltaKernel):
    """Dispatch GDN and its gradients to the official FLA implementation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaKernel.Config):
        force_fp32: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        if config.backend != "fla_chunked":
            raise ValueError("Qwen FLA training requires the fla_chunked backend")
        self.force_fp32 = config.force_fp32

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = xv_BLNV.shape
        inputs = (xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN)
        if self.force_fp32:
            inputs = tuple(tensor.float() for tensor in inputs)
        if cu_seqlens is not None:
            inputs = tuple(tensor.flatten(0, 1).unsqueeze(0) for tensor in inputs)
        gated_delta_fn = cast("Any", chunk_gated_delta_rule)
        output, _ = gated_delta_fn(
            *inputs,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=64,
        )
        return output.to(xv_BLNV.dtype).reshape(shape)


def _convolution(tensor, weight, cu_seqlens=None, cu_seqlens_cpu=None, *, force_fp32=False):
    output_dtype = tensor.dtype
    if force_fp32:
        tensor, weight = tensor.float(), weight.float()
    output, _ = cast("Any", causal_conv1d)(
        x=tensor,
        weight=weight.reshape(weight.shape[0], -1),
        bias=None,
        activation="silu",
        backend="triton",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return output.to(output_dtype)


class FLAGatedDeltaNet(qwen3_5.GatedDeltaNet):
    """Use official operators while inheriting the upstream forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(qwen3_5.GatedDeltaNet.Config):
        causal_conv1d: bool = True
        force_fp32: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_fla_convolution = config.causal_conv1d
        self.force_fp32 = config.force_fp32

    def _causal_conv(self, x_BLD, conv, cu_seqlens=None, cu_seqlens_cpu=None):
        if not self.use_fla_convolution:
            return _npu_causal_conv(self, x_BLD, conv, cu_seqlens, cu_seqlens_cpu)
        if isinstance(x_BLD, DTensor):

            def local_convolution(tensor_local, weight_local, offsets_local=None):
                return _convolution(
                    tensor_local, weight_local, offsets_local, cu_seqlens_cpu, force_fp32=self.force_fp32
                )

            offsets = () if cu_seqlens is None else (cu_seqlens,)
            return self._local_map_conv(x_BLD, conv, local_convolution, *offsets)
        return _convolution(x_BLD, conv.weight, cu_seqlens, cu_seqlens_cpu, force_fp32=self.force_fp32)


class FLAConvContextParallelGatedDeltaNet(ContextParallelGatedDeltaNet):
    """Replace only convolution; inherit the production CP forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelGatedDeltaNet.Config):
        force_fp32: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.force_fp32 = config.force_fp32

    def causal_convolution(self, tensor, convolution, metadata):
        output = _convolution(
            tensor.flatten(0, 1).unsqueeze(0),
            shard_local_heads(convolution.weight, self.context_parallel_mesh),
            metadata.cu_seqlens,
            metadata.cu_seqlens_cpu,
            force_fp32=self.force_fp32,
        )
        return output.reshape_as(tensor)


@override(
    target=qwen3_5.GatedDeltaNet.Config,
    exact=True,
    description="Use official FLA GDN and optional convolution for Qwen training",
)
def operators(
    cfg: qwen3_5.GatedDeltaNet.Config,
    *,
    context_parallel: bool = False,
    causal_conv1d: bool = True,
    force_fp32: bool = False,
):
    """Select official operators, optionally promoting their inputs to FP32.

    Promotion is a training-side numerical strategy: parameters and returned
    activations retain their configured dtype. The default follows the recipe.
    """
    kernel = derive(cfg.kernel, FLAGatedDeltaKernel.Config, force_fp32=force_fp32)
    if context_parallel and causal_conv1d:
        return derive(cfg, FLAConvContextParallelGatedDeltaNet.Config, kernel=kernel, force_fp32=force_fp32)
    if context_parallel:
        return derive(cfg, ContextParallelGatedDeltaNet.Config, kernel=kernel)
    return derive(cfg, FLAGatedDeltaNet.Config, kernel=kernel, causal_conv1d=causal_conv1d, force_fp32=force_fp32)
