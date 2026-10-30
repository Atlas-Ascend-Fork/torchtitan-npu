# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)

from torchtitan_npu.models.common.alignment import AlignedTextProcessor


def aligned_text_config(path, *, alignment=2, repeat=False):
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={"data_files": str(path)},
                ),
                processor=AlignedTextProcessor.Config(per_doc_alignment=alignment),
                post_filters=(lambda sample: sample is not None,),
            ),
        ),
        shuffle=False,
        repeat=repeat,
    )


class IntegerTextTokenizer:
    """CPU fixture for whitespace-separated integer tokens, BOS=1 and EOS=2."""

    def encode(self, text, *, add_bos, add_eos):
        return ([1] if add_bos else []) + [int(token) for token in text.split()] + ([2] if add_eos else [])
