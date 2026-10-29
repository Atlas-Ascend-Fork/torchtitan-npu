# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whether a run may compile: the arch x degree table, and what happens when it may not.

The rule is a conjunction of two facts -- arch22's fused metadata carries a device row count, so a
block cannot be one graph there, and at one degree nothing is rebuilt per rank.  The architecture
is passed in rather than queried, so both halves are pinned without an NPU, and the refusal is
pinned to be loud: a skipped compile must never be silent.
"""

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan_npu.models.common.compile_policy import (
    compiles_under_cp,
    skip_compile_under_cp,
)


@dataclass
class _CompileConfig:
    enable: bool = True


def _dims(cp: int) -> SimpleNamespace:
    # The policy reads one attribute of ``ParallelDims``; the rest is the caller's business.
    return SimpleNamespace(cp=cp)


@pytest.mark.parametrize(
    ("cp", "a5", "expected"),
    [
        (1, True, True),
        (1, False, True),
        (2, True, True),
        (2, False, False),
        (8, False, False),
    ],
)
def test_compiles_table(cp, a5, expected):
    assert compiles_under_cp(_dims(cp), a5=a5) is expected


def test_skip_compile_turns_it_off_and_says_why(caplog):
    config = _CompileConfig(enable=True)

    with caplog.at_level(logging.WARNING):
        returned = skip_compile_under_cp(object(), config, _dims(2), a5=False)

    assert returned is not config
    assert returned.enable is False
    assert "Skipping torch.compile" in caplog.text
    # The reason names the architecture fact, not just the degree.
    assert "arch22" in caplog.text


def test_skip_compile_keeps_the_object_when_it_may_compile():
    config = _CompileConfig(enable=True)
    # Same object back: the caller's config is shared with the recipe that built it.
    assert skip_compile_under_cp(object(), config, _dims(2), a5=True) is config
    assert skip_compile_under_cp(object(), config, _dims(1), a5=False) is config


def test_skip_compile_is_silent_when_not_asked():
    config = _CompileConfig(enable=False)
    assert skip_compile_under_cp(object(), config, _dims(2), a5=False) is config
