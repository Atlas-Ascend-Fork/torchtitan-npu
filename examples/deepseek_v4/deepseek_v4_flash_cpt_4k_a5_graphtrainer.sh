#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Full-model A5 GraphTrainer recipe with sequence-based EP overlap.
# Run this script on each participating node, as with the A5 eager recipe.
# Append CLI arguments to override defaults, for example --training.steps 5.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-graph_trainer_deepseek_v4_flash}"
# The GraphTrainer config builds the standard all-to-all dispatcher, so this entry
# selects token_dispatcher.asc instead of the A5 default asc_deepep.
if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                                           torchtitan_npu.override.deepseek_v4.compressor.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
                                           torchtitan_npu.override.common.token_dispatcher.asc}"
fi
export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1
export TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

SEQ_LEN="${SEQ_LEN:-4096}"
MBS="${MBS:-2}"
GBS="${GBS:-512}"
STEPS="${STEPS:-20}"

EP_OVERLAP_ARGS=(
    --compile.ep_overlap.enabled
    --compile.ep_overlap.chunk_dim seq
    --compile.ep_overlap.strategy graph
    --compile.ep_overlap.module_fqn "layers.*.moe"
    --compile.pass_pipeline mutation-functionalization+npu_auto_overlap
)

# To check performance under load-balanced MoE routing, append --debug.moe-force-load-balance.
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a5.sh" \
    --parallelism.spmd-backend partial_dtensor \
    --training.local-batch-size "${MBS}" \
    --training.global-batch-size "${GBS}" \
    --training.seq-len "${SEQ_LEN}" \
    --training.steps "${STEPS}" \
    "${EP_OVERLAP_ARGS[@]}" \
    "$@"
