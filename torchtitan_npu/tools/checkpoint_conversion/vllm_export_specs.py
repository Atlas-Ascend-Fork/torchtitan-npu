# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative name rules for loading, quantization, and frozen-weight recovery.

These rules identify source tensors: quantization targets/exclusions, vision and
MTP names, and DCP training-state entries. The written artifact -- its names,
dtypes, description entries, and loadability checks -- lives in
``vllm_export_contract``.

Quantization filters match adapter-output tensor names, not model-config paths.
"""

import re

# Quantization targets.

# Select DeepSeek attention and shared-expert Linear weights by logical-name suffix.
DEEPSEEK_V4_DENSE_WEIGHT_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
)
# Select Qwen3.5 attention and dense MLP Linear weights by HF-name suffix.
QWEN35_DENSE_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
# Select DeepSeek routed-expert w1/w2/w3 weights; fused w13 is split before matching.
ROUTED_WEIGHT_PATTERN = r".*\.ffn\.experts\.\d+\.w[123]\.weight"


# Quantization exclusions and frozen-weight recovery.

# Vision tower and aligner names, in local and HF spellings. The marker vectors are
# listed separately: a checkpoint carrying only markers is not a complete tower.
VISION_TOWER_PREFIXES = (
    "model.visual.",
    "model.visual_merger.",
    "visual.",
    "vision.",
    "vision_encoder.",
    "aligner.",
)
# V4.1 release checkpoints spell the frozen marker vectors as bare top-level names.
VISION_MARKER_NAMES = ("image_start", "image_end", "image_newline")
# Skip vision quantization and select original HF vision weights for restoration;
# also allow extra vision entries during DCP validation.
VISION_PREFIXES = (
    *VISION_TOWER_PREFIXES,
    "image_marker_embeddings.",
    *VISION_MARKER_NAMES,
)
# Vision-owned routing bias that lives in the language layers' HF namespace. A
# text-only flavor builds no such parameter, but vLLM Ascend builds one per MoE
# layer whenever the served model has a vision tower.
VISION_BIAS_RE = re.compile(r"^layers\.\d+\.ffn\.gate\.bias_vl$")
# Skip MTP quantization even when its weights match the target suffixes.
MTP_PREFIXES = ("mtp.", "mtp_layers.", "mtp.layers.", "model.mtp.")


# DCP load validation.

# Allow these frozen vision weights to be absent from the training checkpoint.
LOCAL_VISION_PREFIXES = ("vision_encoder.", "image_marker_embeddings.")
# Allow and skip extra training-state entries when loading model weights.
DCP_TRAINING_STATE_PREFIXES = (
    "optimizer.",
    "lr_scheduler.",
    "dataloader.",
    "train_state.",
    "rng_state.",
    "ema_optimizer.",
)


# Expert layout.

# Extract layer, expert, and projection IDs for DeepSeek expert validation and w13 splitting.
EXPERT_FQN_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w(13|[123])\.weight$")
