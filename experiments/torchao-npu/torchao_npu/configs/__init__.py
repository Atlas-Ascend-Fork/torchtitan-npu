# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public quantization configurations for torchao-npu."""

from torchao.core.config import ALLOWED_AO_MODULES

from torchao_npu.configs.module_swap import ModuleSwapConfig
from torchao_npu.configs.module_swap_configs import (
    DeepSeekV4QuantLightningIndexerConfig,
    DeepSeekV41QuantCompressorConfig,
    DeepSeekV41QuantLightningIndexerConfig,
    DeepSeekV41QuantSparseAttentionConfig,
    quant_mode_for_config,
)
from torchao_npu.configs.param_swap import ParamSwapConfig

__all__ = [
    "DeepSeekV4QuantLightningIndexerConfig",
    "DeepSeekV41QuantCompressorConfig",
    "DeepSeekV41QuantLightningIndexerConfig",
    "DeepSeekV41QuantSparseAttentionConfig",
    "ModuleSwapConfig",
    "ParamSwapConfig",
    "quant_mode_for_config",
]

# ``config_from_dict`` resolves ``_type`` names only inside ALLOWED_AO_MODULES;
# register every module whose configs land in exported config.json. ``qat.api``
# backfills torchao's own gap: ``config_to_dict`` emits QATConfig/QATStep but
# their module is absent from the decode set (tracked upstream as
# https://github.com/pytorch/ao/issues/5022). Drop the ``qat.api`` entry once
# a torchao release ships it.
ALLOWED_AO_MODULES.update(
    {
        "torchao.quantization.qat.api",
        "torchao_npu.configs.module_swap",
        "torchao_npu.configs.module_swap_configs",
        "torchao_npu.configs.param_swap",
        "torchao_npu.quantization.quant_configs",
    }
)
