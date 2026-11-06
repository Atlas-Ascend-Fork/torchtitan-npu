# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Qwen checkpoint lifecycle on small CPU model states."""

import pytest
import torch


def _adapter(source=None):
    from torchtitan_npu.models.qwen3_5 import model_registry

    spec = model_registry("debugmodel")
    spec.model.enable_weight_tying = False
    return spec.state_dict_adapter(spec.model, source)


def _model(vision):
    model = torch.nn.Module()
    model.tok_embeddings = torch.nn.Embedding(2, 3)
    model.lm_head = torch.nn.Linear(3, 2, bias=False)
    if vision:
        model.vision_encoder = torch.nn.Module()
        model.vision_encoder.register_parameter("pos_embed", torch.nn.Parameter(torch.zeros(2, 3)))
    return model


@pytest.mark.parametrize("index", [None, "{}", "{"])
def test_qwen_missing_initial_weights(tmp_path, caplog, index):
    adapter = _adapter(str(tmp_path))
    state = _model(False).state_dict()
    adapter.to_hf(state)
    assert "Provide its weights directory" in caplog.text
    if index is not None:
        (tmp_path / "model.safetensors.index.json").write_text(index)
    adapter.get_hf_storage_reader(str(tmp_path))
    with pytest.raises(ValueError, match=r"[Ii]nitial HF"):
        adapter.to_hf(state)
