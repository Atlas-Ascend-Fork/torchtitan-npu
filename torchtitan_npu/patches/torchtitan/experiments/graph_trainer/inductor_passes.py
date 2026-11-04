# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/4224

"""Preserve loss ordering for the TorchTitan v0.3.0 full compiler.

Compatibility bridge for the explicit subgraph boundaries introduced in #4224,
not a direct backport: retain the loss interval as FX and compile its surroundings.
Remove after the pinned runtime enables equivalent native chunk-loss regions.
"""

import operator
from contextlib import nullcontext
from functools import wraps

import torch
import torch.fx as fx
import torch.fx.passes.regional_inductor as regional_inductor
import torchtitan.experiments.graph_trainer.inductor_passes
import torchtitan.experiments.graph_trainer.passes
from torch.fx.graph_module import _get_attr
from torch.fx.passes.split_module import split_module
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    set_graph_module_boxed_codegen,
)
from torchtitan.experiments.graph_trainer.cudagraph import is_cudagraph_compatible
from torchtitan.tools.logging import logger

_original_full_compile = torchtitan.experiments.graph_trainer.inductor_passes.full_inductor_compilation_pass


def _compile_boxed_region(gm):
    inputs = [node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder"]
    options = {}
    for module in gm.modules():
        if isinstance(module, fx.GraphModule):
            for node in module.graph.nodes:
                if node.op not in {"placeholder", "output"}:
                    annotation = node.meta.setdefault("custom", {}).setdefault(
                        "compile_with_inductor", {"inductor_configs": {}}
                    )
                    node_options = annotation.get("inductor_configs") if isinstance(annotation, dict) else None
                    if node_options:
                        if options and options != node_options:
                            raise ValueError("Conflicting non-empty inductor_configs in a full Inductor region")
                        options = dict(node_options)
    graph = fx.Graph()
    args = graph.placeholder("args")
    # Use Inductor's native flatten_graph_inputs/GmWrapper path. Its steal_arg
    # contract clears the caller's list before entering the compiled function.
    args.meta.update(val=inputs, steal_arg=True)
    env = {}
    index = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node] = graph.call_function(operator.getitem, (args, index))
            env[node].meta = node.meta.copy()
            index += 1
        else:
            env[node] = graph.node_copy(node, env.__getitem__)
    region = fx.GraphModule(gm, graph)
    disable_remat = getattr(regional_inductor, "_disable_remat_for_regional_subcompile", nullcontext)
    with (
        torch._guards.tracing(torch._guards.TracingContext(torch._guards.detect_fake_mode(inputs))),
        torch._inductor.config.patch({"eager_numerics.division_rounding": True, "reorder_for_peak_memory": True}),
        torch._inductor.config.patch(options),
        disable_remat(),
    ):
        # compile_fx supports nested inputs via flatten_graph_inputs, but the
        # standalone_compile InputType annotation only lists flat input types.
        compiled = torch._inductor.standalone_compile(
            region,
            [inputs],  # pyrefly: ignore [bad-argument-type]
            dynamic_shapes="from_tracing_context",
            options={},
            donate_graph_module=True,
        )
    wrapper = fx.Graph()
    args = wrapper.placeholder("args")
    output = wrapper.call_function(regional_inductor._dummy_wrapper(compiled), (args,))
    wrapper.output(output)
    return fx.GraphModule({}, wrapper)


def _is_loss_node(node: fx.Node) -> bool:
    return _get_module_fqn(node).split(".", 1)[0] in {"loss", "lm_head"}


@wraps(_original_full_compile)
def full_inductor_compilation_pass(gm, example_inputs, *, boxed_codegen=False):
    nodes = list(gm.graph.nodes)
    loss_nodes = {
        node for node in nodes if node.op not in {"placeholder", "get_attr", "output"} and _is_loss_node(node)
    }
    # A HOP can carry loss in a nested graph without a top-level loss FQN.
    for node in nodes:
        if node.op not in {"get_attr", "call_module"} or not isinstance(node.target, str):
            continue
        child = _get_attr(gm, node.target)
        if isinstance(child, fx.GraphModule) and any(
            _is_loss_node(inner)
            for module in child.modules()
            if isinstance(module, fx.GraphModule)
            for inner in module.graph.nodes
        ):
            loss_nodes.update(node.users if node.op == "get_attr" else (node,))
    if not loss_nodes:
        return _original_full_compile(gm, example_inputs, boxed_codegen=boxed_codegen)

    compatible = is_cudagraph_compatible(gm, skip_flex_attention_check=True)
    # split_module can lift loss constants into the parent graph. Migrate the
    # whole graph (including nested graphs) before any partition is extracted.
    torchtitan.experiments.graph_trainer.inductor_passes._migrate_cpu_get_attrs_to_cuda(gm)
    positions = {node: index for index, node in enumerate(nodes)}
    first, last = min(positions[node] for node in loss_nodes), max(positions[node] for node in loss_nodes)
    # Include unannotated gradient accumulation between chunks. Separate FX
    # call boundaries prevent Inductor from merging across this whole interval.
    partitions = {node: 0 if index < first else 1 if index <= last else 2 for index, node in enumerate(nodes)}
    result = split_module(gm, gm, partitions.__getitem__, keep_original_order=True)
    for name, child in list(result.named_children()):
        if not isinstance(child, fx.GraphModule):
            continue
        if name == "submod_1":
            for module in child.modules():
                if isinstance(module, fx.GraphModule):
                    for node in module.graph.nodes:
                        node.meta.get("custom", {}).pop("compile_with_inductor", None)
            set_graph_module_boxed_codegen(child, boxed=True)
            continue
        if name not in {"submod_0", "submod_2"}:
            continue
        result.add_submodule(name, _compile_boxed_region(child))
    for node in result.graph.nodes:
        if node.op == "call_module" and node.target in {"submod_0", "submod_1", "submod_2"}:
            # Packing must be a separate node: FX then drops last-use locals
            # BEFORE entering the consumer, rather than after it returns.
            with result.graph.inserting_before(node):
                args = result.graph.call_function(list, (node.args,))
            node.args = (args,)
    result.meta.update(gm.meta)
    result.meta.pop("graph_trainer_boxed_codegen", None)
    result.meta["cudagraph_compatible"] = compatible
    logger.info("Full Inductor: retaining sequential FX execution for loss; compiling the surrounding model regions")
    return set_graph_module_boxed_codegen(result, boxed=boxed_codegen)


def apply() -> None:
    """Own the full-compile override, including the pass factory's captured alias."""
    if (
        torchtitan.experiments.graph_trainer.inductor_passes.full_inductor_compilation_pass
        is full_inductor_compilation_pass
    ):
        return
    torchtitan.experiments.graph_trainer.inductor_passes.full_inductor_compilation_pass = full_inductor_compilation_pass
    torchtitan.experiments.graph_trainer.passes.full_inductor_compilation_pass = full_inductor_compilation_pass


apply()
