# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


@pytest.mark.parametrize("backend", ["asc", "pypto", "pypto_pro"])
def test_model_config_accepts_paired_teacher_ports(backend):
    config = deepseek_v4_debugmodel()
    config.override.imports = [
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"num_heads": 16}),
        (f"torchtitan_npu.override.deepseek_v4.lightning_indexer.{backend}", {}),
    ]

    config.model_spec.model.update_from_config(config=config)

    assert config.model_spec.model.compress_ratios == (0, 0, 4, 128)


@pytest.mark.parametrize(
    "port",
    ["sparse_attn.asc", "lightning_indexer.asc", "lightning_indexer.pypto", "lightning_indexer.pypto_pro"],
    ids=["provider-only", "asc-only", "pypto-only", "pypto-pro-only"],
)
def test_model_config_rejects_half_teacher_before_build(port):
    config = deepseek_v4_debugmodel()
    config.override.imports = [("torchtitan_npu.override.deepseek_v4." + port, {})]

    with pytest.raises(ValueError, match="SMLAG provides the teacher"):
        config.model_spec.model.update_from_config(config=config)


def test_model_config_accepts_reference_teacher_path():
    config = deepseek_v4_debugmodel()
    config.override.imports = []

    config.model_spec.model.update_from_config(config=config)

    assert config.model_spec.model.compress_ratios == (0, 0, 4, 128)


@pytest.mark.parametrize("mode", ["fp8", "mxfp4", "mxfp8", "hif8"])
def test_model_config_accepts_quantized_teacher_ports(mode):
    config = deepseek_v4_debugmodel()
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.li_quantization = mode
    config.override.imports = [
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {}),
        ("torchtitan_npu.override.deepseek_v4.lightning_indexer.asc", {}),
    ]

    config.model_spec.model.update_from_config(config=config)

    assert config.model_spec.model.compress_ratios == (0, 0, 4, 128)
