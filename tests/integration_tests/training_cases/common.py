# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared defaults for the layered training cases; site assets are injected."""

STEPS = 10
METRICS = ("--metrics.enable_tensorboard", "--metrics.log-freq=1")
ASSETS = ("ASCEND_SET_ENV_PATH", "HF_ASSETS_PATH")
EAGER = ("--compile.no-enable",)
COMMON = ("--training.steps=10", "--debug.seed=42", "--debug.print-config")
RESUME = (
    "--checkpoint.enable",
    "--checkpoint.no-load-only",
    "--checkpoint.interval=5",
    "--checkpoint.no-last-save-model-only",
    "--checkpoint.keep-latest-k=0",
    "--lr-scheduler.total-steps=10",
)


V4_ENV = {
    "CONFIG": "deepseek_v4_flash_43layers_16experts",
    "USE_GOLDEN": "0",
    "DATASET": "c4_test",
    "DATASET_PATH": "tests/assets/c4_test",
}

V41_ENV = {
    "CONFIG": "deepseek_v4_1_flash_40layers_16experts_multimodal",
    "TORCHTITAN_ENGRAM_TABLE_ROWS": "100000",
    "USE_GOLDEN": "0",
}

CP_ARGS = (
    "--parallelism.context-parallel-degree=2",
    "--parallelism.expert-parallel-degree=8",
    "--parallelism.data-parallel-shard-degree=8",
    "--parallelism.data-parallel-replicate-degree=1",
    "--dataloader.per-doc-alignment=256",
)


def case_defaults() -> dict:
    return dict(
        train_args=METRICS,
        expected_steps=(tuple(range(1, STEPS + 1)),),
        check_loss=False,  # No independent frozen golden exists for these recipes yet.
        check_grad_norm=True,
        use_golden=False,
        required_env=ASSETS,
        timeout=14400,
        unset_env=(
            "MODULE",
            "CONFIG",
            "TRAIN_FILE",
            "COMM_MODE",
            "CLI_OVERRIDES",
            "OPTIMIZER_OVERRIDES",
            "USE_TILELANG",
            "MHC_POST_OVERRIDE",
            "CP",
            "EP",
            "DP_SHARD",
            "MBS",
            "GBS",
            "STEPS",
            "DATASET",
            "TORCHTITAN_ENGRAM_TABLE_ROWS",
        ),
    )
