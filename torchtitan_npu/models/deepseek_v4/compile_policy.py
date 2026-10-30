# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whether a fused DeepSeek stack can be compiled at this degree, and what to say when it cannot.

arch22's fused attention and selector size their kernel options by a **device** row count -- the key
compaction is a ``nonzero`` -- so a compiled region cannot guard on it
(``Could not guard on data-dependent expression``, raised from inside the vendor op's meta
function).  There is nowhere to break the graph either: torchtitan compiles every TransformerBlock
with ``fullgraph=True``, and under full activation checkpointing the block is traced inside the
``checkpoint`` higher-order operator, which admits no break of its own.

arch35 reads ``seqused``, its metadata carries no device row count, and it compiles as usual; a
single degree moves nothing, so ``CP=1`` compiles on every architecture.

So the stack states the fact where its degree and its arch are both known -- in the model's
``parallelize`` -- and the flag stays accepted, which is what lets one recipe serve both
architectures: on A5 it compiles, on arch22 it warns once and runs eagerly.
"""

from dataclasses import replace

from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torch_npu.device import is_a5

__all__ = ["compiles_under_cp", "skip_compile_under_cp"]


def compiles_under_cp(parallel_dims, *, a5: bool | None = None) -> bool:
    """Whether this run's blocks can each be one graph (see the module docstring).

    ``a5`` states the architecture; ``None`` asks the device.  The parameter exists so the
    decision can be pinned in a test without an NPU.
    """
    return (is_a5() if a5 is None else a5) or parallel_dims.cp <= 1


def skip_compile_under_cp(model, compile_config, parallel_dims, *, a5: bool | None = None):
    """``compile_config``, with compilation turned off when this run cannot support it.

    Logs once, with the reason, when it does: a skipped compile must never be silent.  Returns the
    config unchanged -- same object -- when compilation is possible or was not asked for.
    """
    if not compile_config.enable or compiles_under_cp(parallel_dims, a5=a5):
        return compile_config
    logger.warning(
        "Skipping torch.compile for %s: arch22's fused kernels read device row counts under context "
        "parallel, so a block cannot be one graph. Running eagerly; --compile.enable stays accepted "
        "so one recipe serves both architectures.",
        type(model).__name__,
    )
    return replace(compile_config, enable=False)
