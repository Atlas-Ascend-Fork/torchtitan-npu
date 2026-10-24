# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shared packing-grid checks: one contract for the loader and every CP entry."""

import pytest
import torch

from torchtitan_npu.models.common.packing import (
    compression_alignment,
    require_document_alignment,
    require_per_doc_alignment,
)


def _bounds(*lengths: int) -> torch.Tensor:
    return torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)


def test_compression_alignment_is_the_stack_lcm():
    assert compression_alignment((1, 4, 128)) == 128
    assert compression_alignment((0, 0, 2, 4)) == 4
    assert compression_alignment((1, 1)) == 1


def test_require_document_alignment_returns_the_strict_multiple():
    assert require_document_alignment(_bounds(256, 128)) == 1
    assert require_document_alignment(_bounds(256, 128), cp_size=2) == 2
    assert require_document_alignment(_bounds(256, 512), cp_size=2, ratio=128) == 256


@pytest.mark.parametrize(
    "lengths, cp_size, ratio, required",
    [
        ((64, 64), 2, 128, 256),
        ((128, 128), 4, 128, 512),
        ((0, 128), 1, 128, 128),
    ],
    ids=["cp2-ratio128-doc128-is-intentionally-rejected", "cp4-ratio128", "zero-length"],
)
def test_require_document_alignment_rejects_partial_chunks(lengths, cp_size, ratio, required):
    with pytest.raises(ValueError, match=rf"positive multiples of {required}") as exc:
        require_document_alignment(_bounds(*lengths), cp_size=cp_size, ratio=ratio)
    assert "per_doc_alignment" in str(exc.value)
    assert "partial compression blocks" in str(exc.value)


def test_require_per_doc_alignment_checks_the_cli_value():
    # V4's lcm is 128: CP=2 needs 256, CP=4 needs 512, CP=1 accepts 128.
    assert require_per_doc_alignment(128, (1, 4, 128), 1) == 128
    assert require_per_doc_alignment(256, (1, 4, 128), 2) == 256
    assert require_per_doc_alignment(1024, (1, 4, 128), 4) == 512
    with pytest.raises(ValueError, match=r"--dataloader.per_doc_alignment=128") as exc:
        require_per_doc_alignment(128, (1, 4, 128), 2)
    assert "multiple of 256" in str(exc.value)
    assert "cp_degree=2" in str(exc.value)
