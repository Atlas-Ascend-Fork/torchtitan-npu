# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe the final config before build, without changing override selection.

This module owns the apply_overrides extension. It activates after package patches
and before trainer extensions, refreshing already captured upstream bindings too.
The pinned upstream's read-only claim helpers keep diagnostics aligned with its
selection rules; upstream still owns every mutation and successful replacement log.
"""

import importlib
import sys
from collections import defaultdict
from dataclasses import fields, is_dataclass
from functools import wraps

from torchtitan.config.configurable import Configurable
from torchtitan.tools.logging import logger

_override = importlib.import_module("torchtitan.config.override")
_original_apply_overrides = _override.apply_overrides
_FIELDS = (
    "backend",
    "kernel",
    "recipe",
    "li_quantization",
    "enable_quantized_training",
    "enable_sparse_attention_quantization",
    "num_heads",
    "head_dim",
    "index_num_heads",
    "index_head_dim",
    "index_topk",
    "window_size",
    "compression_ratio",
    "context_parallel_degree",
    "tensor_parallel_degree",
    "expert_parallel_degree",
)


def _config_summary(config_root):
    """Group equal configured implementations; no device execution is implied."""
    quantization = getattr(getattr(config_root, "extension", None), "quantization", None)
    if quantization is not None and is_dataclass(quantization):
        settings = {
            item.name: value
            for item in fields(quantization)
            if isinstance((value := getattr(quantization, item.name)), (str, int, float, bool, list, type(None)))
        }
        logger.info("[Override quantization] settings=%s", settings)
    groups = defaultdict(list)
    for fqn, cfg, _, _ in config_root.traverse(Configurable.Config, recurse=True):
        values = tuple(
            (name, value)
            for name in _FIELDS
            if isinstance((value := getattr(cfg, name, None)), (str, int, float, bool))
        )
        owner = getattr(cfg, "_owner", None)
        implementation = (
            f"{owner.__module__}.{owner.__qualname__}"
            if owner is not None
            else f"{type(cfg).__module__}.{type(cfg).__qualname__}"
        )
        groups[(implementation, values)].append(fqn or "<root>")
    for (implementation, values), fqns in groups.items():
        logger.info(
            "[Override configured] implementation=%s count=%d fqns=%s settings=%s",
            implementation,
            len(fqns),
            fqns,
            dict(values),
        )


def _report_override_entries(override_config, config_root):
    """Inspect claims without applying factories or mutating the config tree."""
    entries = [(e, {}) if isinstance(e, str) else e for e in override_config.imports]
    try:
        targets = [(*_override._resolve_target(name), kwargs) for name, kwargs in entries]
        resolved = _override._resolve_active(targets)
    except (ImportError, ValueError):
        logger.error("[Override resolution failed] imports=%s", [name for name, _ in entries])
        return
    claims = _override._collect_claims(resolved, config_root)
    try:
        _override._check_node_conflicts(claims)
    except ValueError:
        logger.error("[Override conflict] imports=%s", [name for name, _ in entries])
        return
    for ov, _ in resolved:
        fqns = [claim.fqn for claim in claims if claim.ov.name == ov.name]
        if fqns:
            logger.info("[Override entry] %s matched=%d fqns=%s", ov.name, len(fqns), fqns)
        else:
            candidates = [
                (fqn, f"{type(cfg).__module__}.{type(cfg).__qualname__}")
                for fqn, cfg, _, _ in config_root.traverse(ov.target_cls, recurse=True)
            ]
            logger.warning(
                "[Override zero match] %s matched=0 target=%s.%s exact=%s fqns=%s "
                "candidate_configs=%s; check FQN/type filters or quantization takeover",
                ov.name,
                ov.target_cls.__module__,
                ov.target_cls.__qualname__,
                ov.exact,
                ov.fqns,
                candidates,
            )


@wraps(_original_apply_overrides)
def apply_overrides(override_config, config_root):
    """Report effective configs; upstream remains the authority for validation."""
    # Private inspection helpers may change when the pinned upstream is upgraded.
    # Diagnostics must not prevent its public apply function from validating a run.
    try:
        _report_override_entries(override_config, config_root)
    except Exception:
        logger.warning(
            "[Override diagnostics unavailable] entry inspection failed; delegating validation "
            "and replacement to upstream apply_overrides",
            exc_info=True,
        )

    # Keep mutations and factory exceptions outside the diagnostic fallback.
    result = _original_apply_overrides(override_config, config_root)
    try:
        _config_summary(config_root)
    except Exception:
        logger.warning(
            "[Override diagnostics unavailable] config summary failed after upstream replacements completed",
            exc_info=True,
        )
    return result


def apply() -> None:
    """Install the unique owner and refresh existing from-import bindings."""
    # The package exports a decorator named ``override``, shadowing the module.
    # Address that one module explicitly rather than assigning to the decorator.
    sys.modules["torchtitan.config.override"].apply_overrides = apply_overrides  # pyrefly: ignore [missing-attribute]
    sys.modules["torchtitan.config"].apply_overrides = apply_overrides  # pyrefly: ignore [missing-attribute]
    if "torchtitan.trainer" in sys.modules:
        sys.modules["torchtitan.trainer"].apply_overrides = apply_overrides  # pyrefly: ignore [missing-attribute]
    if "torchtitan.experiments.torchft.trainer" in sys.modules:
        sys.modules["torchtitan.experiments.torchft.trainer"].apply_overrides = (  # pyrefly: ignore [missing-attribute]
            apply_overrides
        )
    if "torchtitan_npu.patches.torchtitan.experiments.torchft.trainer" in sys.modules:
        sys.modules[  # pyrefly: ignore [missing-attribute]
            "torchtitan_npu.patches.torchtitan.experiments.torchft.trainer"
        ].apply_overrides = apply_overrides


apply()
