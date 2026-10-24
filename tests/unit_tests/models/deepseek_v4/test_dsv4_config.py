# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_debugmodel_uses_the_documented_per_layer_compression_ratios():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")

    assert model_spec.model.compress_ratios == (1, 1, 4, 128)
    assert len(model_spec.model.layers) == len(model_spec.model.compress_ratios)


def test_flash_mtp_4k_model_flops():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry(
        "deepseek_v4_flash",
        num_mtp_layers=1,
    ).model

    with torch.device("meta"):
        model = model_config.build()

    assert model_config.get_nparams_and_flops(model, seq_len=4096) == (
        290_942_278_866,
        92_762_352_876,
    )


def test_debugmodel_uses_cann_profiler_extension():
    from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()

    assert isinstance(config.profiler, CANNProfiler.Config)
    assert isinstance(config.profiler.build(), CANNProfiler)


def test_every_layer_pins_the_core_config_the_overrides_target():
    """The sparse core's config type is the anchor the fused overrides replace.

    ``asc``/``pypto`` are registered against exactly this type, so a model that
    moved to another Config class would quietly run the reference core instead of the fused
    one.  The core's implementation may change; this anchor may not.
    """
    from torchtitan_npu.models.deepseek_v4 import model_registry
    from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention

    model_config = model_registry("debugmodel").model
    cores = [layer.attention.inner_attention for layer in model_config.layers]

    assert len(cores) == len(model_config.compress_ratios) == 4
    for core in cores:
        assert type(core) is CompressedSparseInnerAttention.Config
        assert isinstance(core, CompressedSparseInnerAttention.Config)


def test_flash_rope_configs_pin_split_per_site():
    """Every rope site's config carries the split matching its tensor width."""
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry("deepseek_v4_flash", num_mtp_layers=0).model
    rd = 64

    for layer in model_config.layers:
        attn = layer.attention
        # Attention q/kv/o rotate the tail of a head_dim-wide (512) tensor.
        assert attn.rope.dim == rd
        assert attn.rope.split == attn.head_dim - rd
        if attn.compress_ratio > 1:
            assert attn.compressor.rope.split == attn.compressor.head_dim - rd
        if attn.compress_ratio == 4:
            # Indexer q is index_head_dim-wide (128): a different split even
            # though it reuses the same base rope flavor.
            indexer = attn.indexer
            assert indexer.rope.dim == rd
            assert indexer.rope.split == indexer.index_head_dim - rd
            return
    raise AssertionError("flash spec must contain compress_ratio == 4 layers")


def test_dataloader_pads_documents_to_the_compression_alignment():
    """Every pooled document must be a whole number of blocks at every ratio.

    The loader pads each document to ``lcm(compress_ratios)``, so a compressed
    block never straddles a document edge -- the property the compressor's
    "complete blocks only" contract and the kernels' ``cmp_residual_k`` rely on.
    """
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()
    assert config.dataloader.per_doc_alignment == 128
    assert config.training.seq_len % config.dataloader.per_doc_alignment == 0


def test_aligned_dataloader_only_inserts_pads():
    """Alignment must not drop, reorder or rewrite real tokens.

    The aligned loader is the plain loader plus per-document padding: stripping
    the pads (their labels are ``IGNORE_INDEX``) must give the plain loader's
    stream, token for token.  That makes the ST loss shift a packing-density
    (and row-cut placement) effect rather than a data rewrite.
    """
    import torch
    from torchtitan.components.loss import IGNORE_INDEX
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import AlignedTextDataset

    tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/deepseek_v3")

    def rows(alignment):
        dataset = AlignedTextDataset(
            dataset_name="c4_test",
            dataset_path="tests/assets/c4_test",
            tokenizer=tokenizer,
            seq_len=1024,
            per_doc_alignment=alignment,
            infinite=True,
        )
        rows_out = []
        for index, (sample, labels) in enumerate(dataset):
            if index == 2:
                break
            rows_out.append((sample["input"], labels))
        return torch.cat([row[0] for row in rows_out]), torch.cat([row[1] for row in rows_out])

    plain_ids, _ = rows(1)
    aligned_ids, aligned_labels = rows(128)
    real_ids = aligned_ids[aligned_labels != IGNORE_INDEX]
    assert (aligned_labels == IGNORE_INDEX).sum() > 0
    assert torch.equal(real_ids, plain_ids[: real_ids.numel()])
