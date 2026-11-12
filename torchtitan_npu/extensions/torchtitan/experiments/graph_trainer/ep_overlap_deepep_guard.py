# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fix: skip the all-to-all-specific EP overlap stack when MoE uses DeepEP.

Root cause: the graph-trainer EP overlap subsystem discovers MoE regions via
``_c10d_functional.all_to_all_single`` nodes (``ep_chunk_pass._ep_roots`` with
``require_all_to_all=True`` when EP>1) and both the upstream scheduler and the
NPU MoE auto-overlap scheduler treat annotated all-to-all launches as their
only scheduling markers. DeepEP token exchange runs through the fused
``torch.ops.deepep.dispatch/combine`` kernels backed by a process-global
ElasticBuffer, so the traced graph contains no all-to-alls and
``ep_overlap_chunk_pass`` raises
``ValueError: No EP all-to-all regions matched 'layers.*.moe'``.

Chunking DeepEP regions is not a safe alternative: it would duplicate the
buffer-bound dispatch/combine calls per chunk with no concurrency guarantee.
DeepEP performs its own comm/compute overlap inside the fused kernels, so the
correct behavior is to run GraphTrainer without the FX-level EP overlap
passes for this comm backend.
"""

from torchtitan.tools.logging import logger


def _model_uses_deepep_dispatch(config) -> bool:
    """Return whether any routed-experts layer dispatches tokens via DeepEP."""
    from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
        DeepEPTokenDispatcher,
    )

    layers = getattr(getattr(getattr(config, "model_spec", None), "model", None), "layers", None)
    if not layers:
        return False
    for layer_cfg in layers:
        moe_cfg = getattr(layer_cfg, "moe", None)
        routed_cfg = getattr(moe_cfg, "routed_experts", None)
        dispatcher_cfg = getattr(routed_cfg, "token_dispatcher", None)
        if isinstance(dispatcher_cfg, DeepEPTokenDispatcher.Config):
            return True
    return False


def disable_ep_overlap_for_deepep(config) -> None:
    """Disable the all-to-all EP overlap stack when DeepEP is the MoE backend.

    The graph EP chunking pass, the upstream EP overlap scheduler, and the NPU
    MoE auto-overlap scheduler all key off ``_c10d_functional.all_to_all_single``
    token exchanges. With ``moe_comm_backend="deepep"`` those nodes do not
    exist, so the stack cannot apply; without this guard graph build fails at
    ``ep_overlap_chunk_pass``. DeepEP overlaps communication inside its fused
    ElasticBuffer kernels instead.
    """
    compile_config = getattr(config, "compile", None)
    ep_overlap = getattr(compile_config, "ep_overlap", None)
    if ep_overlap is None or not ep_overlap.enabled:
        return
    if not _model_uses_deepep_dispatch(config):
        return

    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.auto_overlap import (
        revert_npu_auto_overlap_pipeline,
    )

    logger.warning(
        "DeepEP MoE token dispatch detected: EP overlap graph chunking and "
        "scheduling (including the NPU MoE auto-overlap scheduler) are built "
        "on _c10d_functional.all_to_all_single token exchanges and do not "
        "apply to torch.ops.deepep.dispatch/combine. Disabling "
        "compile.ep_overlap for this run; DeepEP overlaps communication "
        "inside its fused ElasticBuffer kernels."
    )
    ep_overlap.enabled = False
    if compile_config is not None:
        compile_config.pass_pipeline = revert_npu_auto_overlap_pipeline(compile_config.pass_pipeline)
