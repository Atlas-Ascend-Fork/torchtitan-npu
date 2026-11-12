# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vLLM Ascend export contract: names, description, config, and loadability checks.

The producer and the post-write validator share these helpers so that the
tensor names, dtype tags, description entries, and config vision switch have
one source of truth.
"""

import json
import logging
import re
from pathlib import Path

import torch

if __package__:
    from .vllm_export_specs import EXPERT_FQN_RE, VISION_PREFIXES, VISION_TOWER_PREFIXES
else:
    # Script mode: the sibling module is on sys.path at runtime.
    from vllm_export_specs import (  # pyrefly: ignore [missing-import]
        EXPERT_FQN_RE,
        VISION_PREFIXES,
        VISION_TOWER_PREFIXES,
    )

logger = logging.getLogger(__name__)


def effective_weight_config(weight_config):
    """The config whose dtype and block size produce the inference weight.

    MXFP4 QAT stores the FP4 config on ``BlockMXQuantizeConfig``, and
    ``to_inference_weight`` quantizes with it.
    """
    return getattr(weight_config, "mxfp4_fake_quantize_config", None) or weight_config


def quant_description_type(weight_config) -> str:
    """Derive the vLLM Ascend format from the inference weight config.

    Both ``W8A8_MXFP8`` (E4M3) and ``W4A8_MXFP`` are 32-element MX formats, so
    a recipe with any other block size or with E5M2 weights has no vLLM Ascend
    description.
    """
    weight_config = effective_weight_config(weight_config)
    if weight_config.block_size != 32:
        raise ValueError(f"vLLM Ascend MX formats use 32-element blocks, got block_size={weight_config.block_size}")
    if weight_config.elem_dtype == torch.float4_e2m1fn_x2:
        return "W4A8_MXFP"
    if weight_config.elem_dtype is torch.float8_e4m3fn:
        return "W8A8_MXFP8"
    raise ValueError(f"no vLLM Ascend description for weight dtype {weight_config.elem_dtype}")


def scale_name(fqn: str) -> str:
    return fqn[: -len(".weight")] + ".weight_scale" if fqn.endswith(".weight") else fqn + ".weight_scale"


# DeepSeek-V4.1 Engram tables are written as E4M3 rows with one E8M0 scale per
# 32-column block; the table itself carries a metadata-only ``quant_type`` entry.
ENGRAM_TABLE_RE = re.compile(r"^layers\.\d+\.engram\.embed\.weight$")
ENGRAM_MX_BLOCK_SIZE = 32
ENGRAM_DESCRIPTION_QUANT_TYPE = "MXFP8"


def engram_scale_name(fqn: str) -> str:
    # vLLM's Engram loader looks for ``<prefix>.scale``, not the ModelSlim
    # ``weight_scale`` suffix.
    return fqn.removesuffix(".weight") + ".scale"


def engram_quant_type_name(fqn: str) -> str:
    # Official release convention: metadata-only ``embed.quant_type`` entry,
    # with no per-weight description entry for the table itself.
    return fqn.removesuffix(".weight") + ".quant_type"


def build_quant_model_description(hf_state_dict, quantized_fqns, plan, engram_fqns=()) -> dict:
    """Build the flat ``quant_model_description.json`` consumed by vLLM Ascend.

    Engram tables use the official metadata-only ``embed.quant_type`` entry;
    vLLM's Engram loader reads the safetensors dtype and index directly.
    """
    engram_fqns = set(engram_fqns)
    engram_tensor_names = engram_fqns | {engram_scale_name(fqn) for fqn in engram_fqns}
    quantized_scales: dict[str, int] = {}
    for fqn, idx in quantized_fqns.items():
        weight_scale = scale_name(fqn)
        if weight_scale in hf_state_dict:
            quantized_scales[weight_scale] = idx
    quant_types = [quant_description_type(swap.weight_config) for swap, _ in plan]
    # Mixed exports keep the FP8 model label; each weight records its own format.
    quant_type = "W4A8_MXFP" if set(quant_types) == {"W4A8_MXFP"} else "W8A8_MXFP8"
    description: dict[str, object] = {
        "model_quant_type": quant_type,
        "version": "1.0.0",
        "group_size": 0,
        "metadata": {},
        "optional": {},
    }
    for fqn in sorted(hf_state_dict):
        if fqn in engram_tensor_names or fqn in quantized_scales:
            continue
        description[fqn] = quant_types[quantized_fqns[fqn]] if fqn in quantized_fqns else "FLOAT"
    for fqn, idx in sorted(quantized_scales.items()):
        description[fqn] = quant_types[idx]
    for fqn in sorted(engram_fqns):
        description[engram_quant_type_name(fqn)] = ENGRAM_DESCRIPTION_QUANT_TYPE

    # ``quant_description_type`` rejects every MX block size other than 32, so
    # this set has at most one element.
    block_sizes = {
        effective_weight_config(plan[idx][0].weight_config).block_size for idx in set(quantized_fqns.values())
    }
    if block_sizes:
        description["group_size"] = next(iter(block_sizes))
    return description


def write_quant_model_description(output_dir: Path, description: dict) -> None:
    path = output_dir / "quant_model_description.json"
    path.write_text(json.dumps(description, indent=2) + "\n")
    logger.info("Wrote vLLM Ascend quantization description to %s (%d entries)", path, len(description))


# safetensors dtype tags for the dtypes the exporter can write; an unmapped dtype
# only skips the header dtype comparison, not the shape check.
SAFETENSORS_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.float8_e8m0fnu: "F8_E8M0",
}

# HF config ``model_type`` vLLM Ascend dispatches on, per model spec name.
EXPECTED_CONFIG_MODEL_TYPE = {
    "deepseek_v4": "deepseek_v4",
    "deepseek_v4_1": "deepseek_v41",
    "qwen3_5": "qwen3_5",
}
_MODEL_NAMES = frozenset(EXPECTED_CONFIG_MODEL_TYPE)


def normalize_model_name(model_name: str) -> str:
    """Reduce a dotted module path to the canonical model name."""
    short = model_name.rsplit(".", 1)[-1]
    if short not in _MODEL_NAMES:
        raise ValueError(f"unsupported model name {model_name!r}; expected one of {sorted(_MODEL_NAMES)}")
    return short


def block_count(size: int, block_size: int) -> int:
    """Number of ``block_size``-sized groups covering ``size``."""
    return -(-size // block_size)


def expected_scale_shape(
    output_size: int, input_size: int, block_size: int, *, keep_scale_tail_dim: bool
) -> tuple[int, ...]:
    blocks = block_count(input_size, block_size)
    if keep_scale_tail_dim:
        return (output_size, -(-blocks // 2), 2)
    return (output_size, blocks)


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Tensor records of a safetensors file, read from the header alone (stdlib only)."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def has_vision_tower(state_dict: dict[str, torch.Tensor]) -> bool:
    """Whether the exported state contains a vision tower tensor."""
    return any(fqn.startswith(VISION_TOWER_PREFIXES) for fqn in state_dict)


def sync_vision_config(output_dir: Path, *, model_name: str, export_has_vision: bool) -> None:
    """Align the copied ``config.json`` vision switch with the exported weights.

    Only the top-level ``vision_n_layers`` field is rewritten; every other entry
    is kept as copied. Qwen3.5 expresses vision through ``vision_config`` and the
    model architecture, so a text-only Qwen export fails instead of guessing.
    """
    model_name = normalize_model_name(model_name)
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    if model_name in ("deepseek_v4", "deepseek_v4_1"):
        config_has_vision = (config.get("vision_n_layers") or 0) > 0
        if not export_has_vision and config_has_vision:
            config["vision_n_layers"] = 0
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            logger.info("config.json vision_n_layers -> 0: the export contains no vision tower")
        elif export_has_vision and not config_has_vision:
            logger.warning(
                "the export contains vision weights, but config.json has vision_n_layers=0; "
                "vLLM Ascend skips them during text-only serving"
            )
        return
    if model_name == "qwen3_5":
        config_has_vision = isinstance(config.get("vision_config"), dict) and bool(config["vision_config"])
        if not export_has_vision and config_has_vision:
            raise ValueError(
                "the export has no vision weights, but config.json declares vision_config; "
                "provide a text-only Qwen config (--needs_vision only fills DCP exports)"
            )
        if export_has_vision and not config_has_vision:
            logger.warning(
                "the export contains vision weights, but config.json has no vision_config; "
                "vLLM Ascend will not build a vision tower"
            )


def _quantized_weight_problems(
    state_dict: dict[str, torch.Tensor],
    quantized_fqns: dict[str, int],
    plan,
    description: dict,
    *,
    keep_scale_tail_dim: bool,
) -> list[str]:
    problems: list[str] = []
    for fqn, idx in sorted(quantized_fqns.items()):
        weight_config = plan[idx][0].weight_config
        effective_config = effective_weight_config(weight_config)
        qdata = state_dict.get(fqn)
        weight_scale = state_dict.get(scale_name(fqn))
        if not isinstance(qdata, torch.Tensor) or not isinstance(weight_scale, torch.Tensor):
            problems.append(f"{fqn}: missing quantized weight or weight_scale")
            continue
        if qdata.ndim != 2:
            problems.append(f"{fqn}: quantized weight must be 2D, got shape {tuple(qdata.shape)}")
            continue
        if effective_config.elem_dtype is torch.float4_e2m1fn_x2:
            expected_dtype, input_size = torch.uint8, qdata.shape[-1] * 2
        else:
            expected_dtype, input_size = effective_config.elem_dtype, qdata.shape[-1]
        if qdata.dtype is not expected_dtype:
            problems.append(f"{fqn}: weight dtype {qdata.dtype} != {expected_dtype}")
        if weight_scale.dtype is not torch.uint8:
            problems.append(f"{fqn}: weight_scale dtype {weight_scale.dtype} != torch.uint8")
        shape = expected_scale_shape(
            qdata.shape[0], input_size, effective_config.block_size, keep_scale_tail_dim=keep_scale_tail_dim
        )
        if tuple(weight_scale.shape) != shape:
            problems.append(f"{fqn}: weight_scale shape {tuple(weight_scale.shape)} != {shape}")
        try:
            expected_type = quant_description_type(weight_config)
        except ValueError as exc:
            problems.append(f"{fqn}: {exc}")
        else:
            if description.get(fqn) != expected_type:
                problems.append(f"{fqn}: description type {description.get(fqn)!r} != {expected_type!r}")
            if description.get(scale_name(fqn)) != expected_type:
                problems.append(
                    f"{fqn}: weight_scale description type {description.get(scale_name(fqn))!r} != {expected_type!r}"
                )
    return problems


def _engram_problems(
    output_dir: Path,
    shard_names: list[str],
    state_dict: dict[str, torch.Tensor],
    description: dict,
) -> list[str]:
    problems: list[str] = []
    engram_fqns = sorted(fqn for fqn in state_dict if ENGRAM_TABLE_RE.match(fqn))
    for fqn in engram_fqns:
        weight = state_dict[fqn]
        weight_scale = state_dict.get(engram_scale_name(fqn))
        if weight.ndim != 2:
            problems.append(f"{fqn}: Engram table must be 2D, got shape {tuple(weight.shape)}")
            continue
        if weight.dtype is not torch.float8_e4m3fn:
            problems.append(f"{fqn}: Engram table dtype {weight.dtype} != torch.float8_e4m3fn")
        shape = (weight.shape[0], block_count(weight.shape[1], ENGRAM_MX_BLOCK_SIZE))
        if weight_scale is None:
            problems.append(f"{fqn}: missing Engram MXFP8 scale")
        elif weight_scale.dtype is not torch.float8_e8m0fnu:
            problems.append(f"{fqn}: Engram scale dtype {weight_scale.dtype} != torch.float8_e8m0fnu")
        elif tuple(weight_scale.shape) != shape:
            problems.append(f"{fqn}: Engram scale shape {tuple(weight_scale.shape)} != {shape}")
        if description.get(engram_quant_type_name(fqn)) != ENGRAM_DESCRIPTION_QUANT_TYPE:
            problems.append(f"{fqn}: description has no Engram quant_type entry")

    if engram_fqns:
        index_path = output_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            problems.append("Engram tables require model.safetensors.index.json; vLLM Ascend loads them from the index")
        else:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict):
                problems.append("model.safetensors.index.json has no weight_map object")
            else:
                shard_headers: dict[str, dict] = {}
                for fqn in engram_fqns:
                    for tensor_name in (fqn, engram_scale_name(fqn)):
                        target = weight_map.get(tensor_name)
                        if target is None:
                            problems.append(f"{tensor_name}: missing from model.safetensors.index.json")
                        elif not isinstance(target, str) or target not in shard_names:
                            problems.append(
                                f"{tensor_name}: index targets {target!r}, which is not one of the exported shards"
                            )
                        else:
                            if target not in shard_headers:
                                shard_headers[target] = read_safetensors_header(output_dir / target)
                            if tensor_name not in shard_headers[target]:
                                problems.append(
                                    f"{tensor_name}: index targets {target!r}, which does not contain the tensor"
                                )
        # vLLM Ascend probes the ModelSlim index name first; a stale one in the output
        # directory would shadow the index this exporter just wrote.
        modelslim_index = output_dir / "quant_model_weights.safetensors.index.json"
        if modelslim_index.is_file():
            modelslim_map = json.loads(modelslim_index.read_text()).get("weight_map", {})
            shadowed = sorted(
                name for fqn in engram_fqns for name in (fqn, engram_scale_name(fqn)) if name in modelslim_map
            )
            if shadowed:
                problems.append(
                    "quant_model_weights.safetensors.index.json shadows the exported index for Engram "
                    f"tensor(s) {shadowed[:4]}; vLLM Ascend probes it first, so remove it from the output directory"
                )
    return problems


def _model_contract_problems(
    state_dict: dict[str, torch.Tensor],
    description: dict,
    *,
    model_name: str,
    needs_vision: bool,
    export_has_vision: bool,
) -> list[str]:
    problems: list[str] = []
    if needs_vision and not export_has_vision:
        problems.append(
            "vision weights were requested, but the export contains no vision tower tensor; "
            "pass --hf_assets_path with the release safetensors so the frozen tower can be restored"
        )
    if model_name == "deepseek_v4" and "hc_head_fn" not in description:
        problems.append(
            "deepseek_v4 description lacks 'hc_head_fn'; vLLM Ascend only rewrites the "
            "checkpoint-named description keys when that key is present"
        )
    if model_name == "deepseek_v4_1" and any(key.startswith(VISION_PREFIXES) for key in state_dict):
        for marker in ("image_start", "image_end", "image_newline"):
            if marker not in state_dict:
                problems.append(f"deepseek_v4_1 multimodal export is missing {marker!r}")
    if model_name == "deepseek_v4_1" and export_has_vision:
        # vLLM Ascend builds one gate.bias_vl per MoE layer whenever the served
        # model has a vision tower, so a text-only flavor's export must have them
        # filled from the release checkpoint as well. The checkpoint may carry the
        # tower even when --needs_vision stays off, so gate on the export itself.
        moe_layers = sorted({int(match[1]) for fqn in state_dict if (match := EXPERT_FQN_RE.match(fqn))})
        for layer in moe_layers:
            bias_name = f"layers.{layer}.ffn.gate.bias_vl"
            if bias_name not in state_dict:
                problems.append(f"{bias_name}: missing; vLLM Ascend needs it for every MoE layer with vision")
    return problems


def _config_problems(output_dir: Path, *, model_name: str, export_has_vision: bool) -> list[str]:
    problems: list[str] = []
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        problems.append("config.json is missing; vLLM Ascend cannot load the directory")
        return problems
    config = json.loads(config_path.read_text())
    expected_model_type = EXPECTED_CONFIG_MODEL_TYPE.get(model_name)
    if not config.get("model_type"):
        problems.append("config.json has no model_type")
    elif expected_model_type is not None and config["model_type"] != expected_model_type:
        problems.append(
            f"config.json model_type {config['model_type']!r} != expected {expected_model_type!r} for {model_name}"
        )
    if not config.get("architectures"):
        problems.append("config.json has no architectures")
    if model_name in ("deepseek_v4", "deepseek_v4_1"):
        if not export_has_vision and (config.get("vision_n_layers") or 0) > 0:
            problems.append(
                "config.json declares vision_n_layers>0, but the export contains no vision weights; it must be set to 0"
            )
    elif model_name == "qwen3_5" and not export_has_vision and config.get("vision_config"):
        problems.append(
            "config.json declares vision_config, but the export contains no vision weights; "
            "provide a text-only Qwen config"
        )
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict) and isinstance(config.get("text_config"), dict):
        quant_config = config["text_config"].get("quantization_config")
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")
        if quant_method is not None and quant_method != "ascend":
            logger.warning(
                "config.json declares quantization_config.quant_method=%r, but the exported weights are "
                "described by quant_model_description.json (ModelSlim); serve with '--quantization ascend' "
                "so vLLM does not select the config's scheme",
                quant_method,
            )
    return problems


def _shard_problems(
    output_dir: Path,
    shard_names: list[str],
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], dict[str, dict]]:
    problems: list[str] = []
    written: dict[str, dict] = {}
    for shard_name in shard_names:
        header = read_safetensors_header(output_dir / shard_name)
        duplicate = set(written).intersection(header)
        if duplicate:
            problems.append(f"tensor(s) written to multiple shards: {sorted(duplicate)[:4]}")
        written.update(header)
    missing = sorted(set(state_dict) - set(written))
    extra = sorted(set(written) - set(state_dict))
    if missing:
        problems.append(f"tensor(s) missing from the shards: {missing[:4]}")
    if extra:
        problems.append(f"unexpected tensor(s) in the shards: {extra[:4]}")
    for name in sorted(set(written) & set(state_dict)):
        header = written[name]
        tensor = state_dict[name]
        expected_dtype = SAFETENSORS_DTYPES.get(tensor.dtype)
        if expected_dtype is not None and header.get("dtype") != expected_dtype:
            problems.append(f"{name}: shard dtype {header.get('dtype')} != {expected_dtype}")
        if tuple(header.get("shape", ())) != tuple(tensor.shape):
            problems.append(f"{name}: shard shape {tuple(header.get('shape', ()))} != {tuple(tensor.shape)}")
    return problems, written


# Non-tensor top-level entries that ``build_quant_model_description`` writes.
_DESCRIPTION_METADATA_KEYS = frozenset({"model_quant_type", "version", "group_size", "metadata", "optional"})


def _description_problems(
    output_dir: Path,
    state_dict: dict[str, torch.Tensor],
    description: dict,
    written: dict[str, dict],
) -> list[str]:
    """Check the description file vLLM Ascend actually reads, not only the in-memory dict."""
    path = output_dir / "quant_model_description.json"
    if not path.is_file():
        return ["quant_model_description.json is missing; vLLM Ascend reads it to choose each weight's quantization"]
    try:
        on_disk = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"quant_model_description.json cannot be parsed: {exc}"]
    if not isinstance(on_disk, dict):
        return ["quant_model_description.json is not a JSON object"]

    problems: list[str] = []
    if on_disk != description:
        problems.append("quant_model_description.json on disk differs from the description the export produced")
    # Engram tables carry a metadata-only ``.quant_type`` entry instead of a tensor.
    metadata_names = {engram_quant_type_name(fqn) for fqn in state_dict if ENGRAM_TABLE_RE.match(fqn)}
    for name in sorted(set(on_disk) - _DESCRIPTION_METADATA_KEYS - metadata_names):
        if name not in written:
            problems.append(f"{name}: description entry has no tensor in the written shards")
    return problems


def validate_export(
    output_dir: Path,
    shard_names: list[str],
    state_dict: dict[str, torch.Tensor],
    quantized_fqns: dict[str, int],
    plan,
    description: dict,
    *,
    model_name: str,
    keep_scale_tail_dim: bool,
    needs_vision: bool,
) -> None:
    """Fail unless the written directory matches the vLLM Ascend export contract.

    With ``keep_scale_tail_dim=True`` the NPU kernel's packed
    ``(..., ceil(blocks / 2), 2)`` scale layout is preserved for inspection; that
    layout is not readable by vLLM Ascend, so the check warns instead of
    guaranteeing loadability. Otherwise the written directory must be directly
    loadable.

    No consumer validates these names before serving, so the export checks its own
    contract: per-weight dtype and scale geometry, the description entries and the
    ``quant_model_description.json`` written to disk, the shard tensor set, the
    vision tower and per-MoE vision bias the export requires, and the
    ``config.json`` vLLM dispatches on.
    """
    model_name = normalize_model_name(model_name)
    if keep_scale_tail_dim:
        logger.warning(
            "keep_scale_tail_dim preserves the packed (..., ceil(blocks / 2), 2) weight_scale "
            "layout; the exported checkpoint is for inspection and is not directly loadable by vLLM Ascend"
        )
    export_has_vision = has_vision_tower(state_dict)
    shard_problems, written = _shard_problems(output_dir, shard_names, state_dict)
    problems = [
        *_quantized_weight_problems(
            state_dict, quantized_fqns, plan, description, keep_scale_tail_dim=keep_scale_tail_dim
        ),
        *_engram_problems(output_dir, shard_names, state_dict, description),
        *_model_contract_problems(
            state_dict,
            description,
            model_name=model_name,
            needs_vision=needs_vision,
            export_has_vision=export_has_vision,
        ),
        *_config_problems(output_dir, model_name=model_name, export_has_vision=export_has_vision),
        *_description_problems(output_dir, state_dict, description, written),
        *shard_problems,
    ]
    if problems:
        raise ValueError("exported checkpoint is not directly loadable by vLLM Ascend:\n  " + "\n  ".join(problems))
    logger.info(
        "Validated %d quantized tensor(s), %d shard(s), the description, and config.json%s",
        len(quantized_fqns),
        len(shard_names),
        " (packed inspection scale layout; not vLLM-loadable)" if keep_scale_tail_dim else "",
    )
