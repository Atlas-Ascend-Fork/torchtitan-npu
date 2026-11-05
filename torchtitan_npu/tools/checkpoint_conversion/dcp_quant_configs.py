# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read the quantization configs a training run embedded in its DCP checkpoint.

Training pickles a wrapped parameter as the whole wrapper object into the
``.distcp`` stream at the offset ``.metadata`` records, so reading it there
recovers the training-time configs that ``dcp.load`` discards. A cheap byte
probe on the stream head skips plain-tensor streams; CPU-only, side-effect
free. ``.metadata`` itself is read with plain ``pickle.load`` -- exactly what
``dcp.load`` (and ``FileSystemReader.read_metadata``) does on the same file,
which this tool performs right after extraction -- so the trust boundary is
the checkpoint directory itself, not this reader.
"""

from __future__ import annotations

import io
import logging
import pickle
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Wrapper streams carry the pickled class path in their first GLOBAL opcodes.
_WRAPPER_STREAM_MARKER = b"torchao_npu.wrapper_tensors"
_PROBE_BYTES = 8192


def register_wrapper_safe_globals() -> None:
    """Allow the wrapper classes for ``weights_only`` unpickling (needed by any ``dcp.load``).

    HiF8's module registers no safe globals; add it here so a HiF8 checkpoint
    fails at ``to_inference_weight`` (owner-defined) instead of an opaque ``UnpicklingError``.
    """
    from torchao_npu.wrapper_tensors import (  # noqa: F401  (import = registration)
        BlockMXTrainingWeightWrapperTensor,
        Float8TrainingWeightWrapperTensor,
        HiF8TrainingWeightWrapperTensor,
        MXTrainingWeightWrapperTensor,
    )

    torch.serialization.add_safe_globals([HiF8TrainingWeightWrapperTensor])


def extract_wrapper_configs(input_dir: Path) -> dict[str, tuple[Any, Any]]:
    """Map each FQN saved as a quantization wrapper (training namespace) to ``(weight_config, activation_config)``.

    Plain streams are absent; an FQN's first recorded shard decides it (all
    shards serialize the same object).
    """
    metadata_path = input_dir / ".metadata"
    if not metadata_path.is_file():
        raise ValueError(f"{input_dir} is not a DCP checkpoint directory (missing .metadata)")
    register_wrapper_safe_globals()
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    configs: dict[str, tuple[Any, Any]] = {}
    decided: set[str] = set()  # first shard decides an FQN; see docstring
    shards_checked = 0
    for index, storage in metadata.storage_data.items():
        fqn = getattr(index, "fqn", None)
        if fqn is None or fqn in decided:
            continue
        decided.add(fqn)
        shard_path = input_dir / storage.relative_path
        if not shard_path.is_file():
            raise ValueError(f"DCP shard {shard_path} referenced by .metadata is missing")
        with shard_path.open("rb") as f:
            # ``.distcp`` files concatenate torch.save archives: reads must
            # stay inside this item's ``length`` or the zip reader loads the wrong object.
            f.seek(storage.offset)
            if _WRAPPER_STREAM_MARKER not in f.read(min(_PROBE_BYTES, storage.length)):
                continue
            f.seek(storage.offset)
            stream = io.BytesIO(f.read(storage.length))
            obj = torch.load(stream, map_location="cpu", weights_only=True)
        shards_checked += 1
        weight_config = getattr(obj, "weight_config", None)
        if weight_config is None:
            logger.warning(
                "Stream for %s looks like a quantization wrapper but carries no weight_config; skipping it",
                fqn,
            )
            continue
        configs[fqn] = (weight_config, getattr(obj, "activation_config", None))
    if shards_checked and not configs:
        logger.warning(
            "%d wrapper stream(s) found but none yielded a quantization config; replay mode will be unavailable",
            shards_checked,
        )
    logger.info(
        "DCP config extraction: %d wrapper stream(s) read, %d FQN(s) with configs",
        shards_checked,
        len(configs),
    )
    return configs
