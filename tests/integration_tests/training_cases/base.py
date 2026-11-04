# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cases introduced by the base layer; inheritance lives in .github/ci/suites.json."""

from tests.integration_tests import OverrideDefinitions
from .common import ASSETS, COMMON, EAGER, CP_ARGS, V4_ENV, V41_ENV, case_defaults


def build_test_list() -> list[OverrideDefinitions]:
    common = case_defaults()
    return [
        OverrideDefinitions(
            test_name="dsv4_flash_a3_16p_cp2_ep8_eager",
            test_descr="base_01: DSV4 Flash A3 1x16P CP2 EP8 Muon eager",
            device="a3",
            nnodes=1,
            ngpu=16,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
            env_vars={**V4_ENV, "CP": "2"},
            override_args=[(*COMMON, *EAGER, *CP_ARGS)],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a3_16p_cp2_ep8_compile",
            test_descr="base_02: DSV4 Flash A3 1x16P CP2 EP8 Muon compile",
            device="a3",
            nnodes=1,
            ngpu=16,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
            env_vars={**V4_ENV, "CP": "2"},
            override_args=[(*COMMON, "--compile.enable", *CP_ARGS)],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_cp1_ep8_graphtrainer",
            test_descr="base_03: DSV4 Flash A3 1x8P CP1 EP8 GraphTrainer compile",
            device="a3",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh",
            env_vars={**V4_ENV, "CONFIG": "graph_trainer_deepseek_v4_flash_43layers_16experts", "CP": "1"},
            override_args=[COMMON],
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv41_flash_a3_8p_multimodal_cpt_eager",
            test_descr="base_04: DSV4.1 Flash A3 1x8P vision-language CPT eager",
            device="a3",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh",
            env_vars={**V41_ENV, "DATASET": "cc12m-test"},
            override_args=[(*COMMON, *EAGER)],
            **{**common, "required_env": (*ASSETS, "DATASET_PATH")},
        ),
        OverrideDefinitions(
            test_name="dsv41_flash_a3_8p_multimodal_cpt_compile",
            test_descr="base_05: DSV4.1 Flash A3 1x8P vision-language CPT compile",
            device="a3",
            nnodes=1,
            ngpu=8,
            train_script="examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh",
            env_vars={**V41_ENV, "DATASET": "cc12m-test"},
            override_args=[COMMON],
            **{**common, "required_env": (*ASSETS, "DATASET_PATH")},
        ),
    ]
