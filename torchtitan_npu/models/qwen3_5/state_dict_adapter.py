# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

from safetensors import safe_open
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter as _UpstreamAdapter
from torchtitan.tools.logging import logger

_MISSING_VISION_WEIGHTS = (
    "Cannot restore untrained vision/MTP weights without an initial HF checkpoint. "
    "Provide its weights directory for a complete HF export."
)


class Qwen35StateDictAdapter(_UpstreamAdapter):
    """Restore untrained vision and MTP tensors from the initial HF checkpoint."""

    def get_hf_storage_reader(self, path, from_quantized=False):
        reader = super().get_hf_storage_reader(path, from_quantized)
        self._initial_hf_path = path
        return reader

    def to_hf(self, state_dict):
        converted = super().to_hf(state_dict)
        # CheckpointManager converts the model state before DCP splits it into shards.
        self._restore_vision = getattr(self.model_config, "vision_encoder", None) is not None and not any(
            key.startswith("vision_encoder.") for key in state_dict
        )
        prefixes = ("mtp.", "model.visual.") if self._restore_vision else ("mtp.",)
        source = getattr(self, "_initial_hf_path", self.hf_assets_path)
        if source is None:
            if self._restore_vision:
                logger.warning(_MISSING_VISION_WEIGHTS)
            return converted
        root = Path(source)
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            try:
                mapping = json.loads(index_path.read_text())["weight_map"]
                if not isinstance(mapping, dict):
                    raise TypeError("weight_map must be an object")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid initial HF weight index: {index_path}") from error
        else:
            single_path = root / "model.safetensors"
            if not single_path.is_file():
                # HF loading converts shapes before it provides the storage reader.
                if not self._restore_vision or not hasattr(self, "_initial_hf_path"):
                    if self._restore_vision:
                        logger.warning(_MISSING_VISION_WEIGHTS)
                    return converted
                raise ValueError(f"The initial HF checkpoint at {root} has no vision weights")
            with safe_open(single_path, framework="pt", device="cpu") as shard:
                mapping = dict.fromkeys(shard.keys(), single_path.name)
        if self._restore_vision and not any(key.startswith("model.visual.") for key in mapping):
            logger.warning(
                "TorchTitan received a Qwen HF checkpoint at %s without the expected vision weights "
                "included in the official Qwen release, which may indicate incomplete or modified third-party weights. "
                "Verify the model folder carefully before and after training.",
                root,
            )
        filenames = {filename for key, filename in mapping.items() if key.startswith(prefixes) and key not in converted}
        for filename in sorted(filenames):
            with safe_open(root / filename, framework="pt", device="cpu") as shard:
                for key, shard_name in mapping.items():
                    if key.startswith(prefixes) and shard_name == filename:
                        converted.setdefault(key, shard.get_tensor(key))
        return converted

    def from_hf(self, hf_state_dict):
        converted = super().from_hf(hf_state_dict)
        if getattr(self, "_restore_vision", False):
            converted = {key: value for key, value in converted.items() if not key.startswith("vision_encoder.")}
        return converted
