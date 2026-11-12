# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve the real chunk-loss schedule around compiled model regions."""

import operator
from types import SimpleNamespace

import pytest
import torch
from torch._inductor.standalone_compile import CompiledArtifact
from torch.fx.experimental.proxy_tensor import make_fx
from torch.multiprocessing.reductions import StorageWeakRef
from torchtitan.experiments.graph_trainer.chunked_loss import ChunkedLossWrapperWithParamGrads
from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns, compute_annotated_loss
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced
from torchtitan.experiments.graph_trainer.passes import final_inductor_compile_passes

from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import inductor_passes


@pytest.fixture(autouse=True)
def cpu_inductor(monkeypatch):
    # Observe device migration on the original graph without requiring CUDA.
    migrated = []
    monkeypatch.setattr(
        inductor_passes.torchtitan.experiments.graph_trainer.inductor_passes,
        "_migrate_cpu_get_attrs_to_cuda",
        migrated.append,
    )
    # Keep real Inductor partitioning/codegen; use ATen CPU fallbacks without a C++ toolchain.
    with torch._inductor.config.patch(disable_cpp_codegen=True), torch.random.fork_rng():
        torch.manual_seed(0)
        yield migrated


def test_full_migrates_loss_constants_before_partitioning(cpu_inductor, monkeypatch):
    x = torch.arange(8, dtype=torch.float32)

    def step(x):
        loss = compute_annotated_loss(lambda h, _: (h * torch.tensor([2.0])).sum(), x.sin(), None)
        return loss.neg()

    with torch.fx.traceback.preserve_node_meta():
        gm = make_fx(step, tracing_mode="fake")(x)
    constants = [node.target for node in gm.graph.nodes if node.op == "get_attr"]
    assert constants
    assert all(isinstance(getattr(gm, name), torch.Tensor) for name in constants)
    split = inductor_passes.split_module

    def split_after_migration(module, *args, **kwargs):
        assert module is gm
        assert cpu_inductor == [gm]
        return split(module, *args, **kwargs)

    monkeypatch.setattr(inductor_passes, "split_module", split_after_migration)
    (compile_pass,) = final_inductor_compile_passes(SimpleNamespace(inductor_compilation="full"))

    compiled = compile_pass(gm, (x,))

    assert cpu_inductor == [gm]
    assert any(node.op == "get_attr" for node in compiled.graph.nodes)
    torch.testing.assert_close(compiled(x), -(x.sin() * 2).sum(), rtol=0, atol=0)


@pytest.mark.parametrize("annotations", [0, 1, 2], ids=["defaults", "nonempty-after-default", "matching"])
def test_full_preserves_region_compile_options(monkeypatch, annotations):
    x = torch.arange(8, dtype=torch.float32)

    def step(x):
        return compute_annotated_loss(lambda h, _: h.sum(), x.sin().cos().neg(), None).neg()

    with torch.fx.traceback.preserve_node_meta():
        gm = make_fx(step, tracing_mode="fake")(x)
    options = {"max_fusion_size": 1}
    default_max_fusion_size = torch._inductor.config.max_fusion_size
    candidates = [
        node for node in gm.graph.nodes if node.target in {torch.ops.aten.cos.default, torch.ops.aten.neg.default}
    ]
    for node in candidates[:annotations]:
        node.meta.setdefault("custom", {})["compile_with_inductor"] = {"inductor_configs": dict(options)}
    received = []
    standalone_compile = torch._inductor.standalone_compile

    def compile_with_options(*args, **kwargs):
        received.append(torch._inductor.config.max_fusion_size)
        return standalone_compile(*args, **kwargs)

    monkeypatch.setattr(torch._inductor, "standalone_compile", compile_with_options)
    (compile_pass,) = final_inductor_compile_passes(SimpleNamespace(inductor_compilation="full"))

    compiled = compile_pass(gm, (x,))

    assert received == [1 if annotations else default_max_fusion_size, default_max_fusion_size]
    assert torch._inductor.config.max_fusion_size == default_max_fusion_size
    torch.testing.assert_close(compiled(x), x.sin().cos().sum(), rtol=0, atol=0)


def test_full_rejects_conflicting_region_compile_options():
    x = torch.arange(8, dtype=torch.float32)

    def step(x):
        return compute_annotated_loss(lambda h, _: h.sum(), x.sin().cos(), None)

    with torch.fx.traceback.preserve_node_meta():
        gm = make_fx(step, tracing_mode="fake")(x)
    for node, value in zip(
        [node for node in gm.graph.nodes if node.target in {torch.ops.aten.sin.default, torch.ops.aten.cos.default}],
        [True, False],
        strict=True,
    ):
        node.meta.setdefault("custom", {})["compile_with_inductor"] = {"inductor_configs": {"max_autotune": value}}
    (compile_pass,) = final_inductor_compile_passes(SimpleNamespace(inductor_compilation="full"))

    with pytest.raises(ValueError, match="Conflicting non-empty inductor_configs"):
        compile_pass(gm, (x,))


class _LossOrderInterpreter(torch.fx.Interpreter):
    def __init__(self, module, events):
        super().__init__(module)
        self.events = events

    def run_node(self, node):
        if node.target == torch.ops.aten.mm.default and node.meta.get("custom", {}).get("module_fqn") == "lm_head":
            self.events.append("head_backward" if node.meta.get("autograd_backward") else "head_forward")
        elif node.target == torch.ops.aten._log_softmax.default:
            self.events.append("forward")
        elif node.target == torch.ops.aten._log_softmax_backward_data.default:
            self.events.append("backward")
        return super().run_node(node)


@pytest.mark.parametrize("num_chunks", [2, 4])
def test_full_keeps_chunk_forward_backward_order_and_gradients(num_chunks):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(8, 8, bias=False)
    model.lm_head = torch.nn.Linear(8, 17, bias=False)
    annotate_module_fqns(model)
    loss = ChunkedLossWrapperWithParamGrads(ChunkedLossWrapperWithParamGrads.Config(num_chunks=num_chunks))
    loss.set_lm_head(model.lm_head)
    x = torch.randn(1, 8, 8, requires_grad=True)
    labels = torch.arange(8).reshape(1, 8)

    def step(x, labels):
        value = compute_annotated_loss(loss, model.backbone(x), labels)
        return (value, *torch.autograd.grad(value, (x, *model.parameters())))

    expected = step(x, labels)
    traced = minimal_fx_tracer(step, module=model)(x, labels)
    (compile_pass,) = final_inductor_compile_passes(SimpleNamespace(inductor_compilation="full"))
    traced.gm = compile_pass(traced.gm, traced.example_inputs)
    # Both surrounding model graphs must really compile; interpreting the whole step is not enough.
    for name in ("submod_0", "submod_2"):
        assert any(
            isinstance(getattr(node.target, "__wrapped__", None), CompiledArtifact)
            for node in traced.gm.get_submodule(name).graph.nodes
        )
    events = []
    loss_graph = traced.gm.submod_1
    loss_graph.forward = _LossOrderInterpreter(loss_graph, events).boxed_run
    actual = run_traced(traced, module=model)(x, labels)

    assert events == ["head_forward", "forward", "backward", "head_backward", "head_backward"] * num_chunks
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.fixture
def residency_probe():
    storages, released = [], []

    def record(x):
        result = x.clone()
        storages.append(StorageWeakRef(result.untyped_storage()))
        return result

    def probe(x):
        released.append([storage.expired() for storage in storages])
        return x.clone()

    # Keep the probes opaque to real Inductor; fake execution must not record lifetimes.
    with torch.library._scoped_library("chunk_loss_residency", "FRAGMENT") as lib:
        lib.define("record(Tensor x) -> Tensor")
        lib.define("probe(Tensor x) -> Tensor")
        lib.impl("record", record, "CPU")
        lib.impl("probe", probe, "CPU")
        for name in ("record", "probe"):
            torch.library.register_fake(f"chunk_loss_residency::{name}", torch.empty_like, lib=lib)
        yield torch.ops.chunk_loss_residency, released


@pytest.mark.parametrize("probe_region", ["loss", "backward"])
@pytest.mark.parametrize("boxed_codegen", [False, True])
def test_full_releases_partition_input_storage_before_partition_returns(residency_probe, probe_region, boxed_codegen):
    ops, released = residency_probe
    x = torch.arange(1024, dtype=torch.float32)

    def step(x):
        saved = ops.record(x)
        if probe_region == "loss":
            loss = compute_annotated_loss(lambda h, _: ops.probe(h.sum()), saved, None)
            return loss.neg()
        loss = compute_annotated_loss(lambda h, _: h.sum(), x, None)
        return ops.probe(saved.sum() + loss)

    with torch.fx.traceback.preserve_node_meta():
        gm = make_fx(step, tracing_mode="fake")(x)
    (compile_pass,) = final_inductor_compile_passes(
        SimpleNamespace(inductor_compilation="full"), boxed_codegen=boxed_codegen
    )
    compiled = compile_pass(gm, (x,))
    assert any(
        isinstance(getattr(node.target, "__wrapped__", None), CompiledArtifact)
        for node in compiled.submod_2.graph.nodes
    )

    with torch.no_grad():
        actual = compiled([x]) if boxed_codegen else compiled(x)

    torch.testing.assert_close(actual, -x.sum() if probe_region == "loss" else 2 * x.sum(), rtol=0, atol=0)
    # Inspect storage while the consumer is STILL running, not after the whole step exits.
    assert released == [[True]]


@pytest.mark.parametrize("boxed_codegen", [False, True], ids=["unboxed", "boxed"])
def test_full_recovers_non_tensor_attrs_with_recomputed_node_names(boxed_codegen):
    x = torch.arange(8, dtype=torch.float32)

    def step(x):
        loss = compute_annotated_loss(lambda h, _: h.sum(), x.sin() * 2.0, None)
        return loss.neg() * 2.0

    with torch.fx.traceback.preserve_node_meta():
        gm = make_fx(step, tracing_mode="fake")(x)
    # A tuple represents non-tensor metadata without requiring a CANN opaque class.
    # Both regions read the same attribute, with the recomputed node using a suffix.
    gm._opaque_obj0 = (2.0,)
    for index, node in enumerate([node for node in gm.graph.nodes if node.target == torch.ops.aten.mul.Tensor]):
        with gm.graph.inserting_before(node):
            attr = gm.graph.get_attr("_opaque_obj0")
            scalar = gm.graph.call_function(operator.getitem, (attr, 0))
        assert attr.name == ("_opaque_obj0" if index == 0 else "_opaque_obj0_1")
        # Only the lifted attribute lacks val; the getitem's scalar metadata is known.
        scalar.meta["val"] = 2.0
        node.args = (node.args[0], scalar)
    gm.recompile()
    (compile_pass,) = final_inductor_compile_passes(
        SimpleNamespace(inductor_compilation="full"), boxed_codegen=boxed_codegen
    )

    compiled = compile_pass(gm, (x,))
    runtime_metadata = tuple([2.0])
    compiled._opaque_obj0 = runtime_metadata
    received = []

    def observe_region_inputs(module, args):
        received.append(any(value is runtime_metadata for value in args[0]))

    handles = [
        compiled.get_submodule(name).register_forward_pre_hook(observe_region_inputs)
        for name in ("submod_0", "submod_2")
    ]
    try:
        actual = compiled([x]) if boxed_codegen else compiled(x)
    finally:
        for handle in handles:
            handle.remove()

    assert received == [True, True]
    torch.testing.assert_close(actual, -4.0 * x.sin().sum(), rtol=0, atol=0)
    for name in ("submod_0", "submod_2"):
        assert any(
            isinstance(getattr(node.target, "__wrapped__", None), CompiledArtifact)
            for node in compiled.get_submodule(name).graph.nodes
        )


def test_boxed_region_reports_unrecoverable_placeholder():
    graph = torch.fx.Graph()
    missing = graph.placeholder("missing_metadata")
    graph.output(missing)
    gm = torch.fx.GraphModule({}, graph)

    with pytest.raises(KeyError, match="missing_metadata.*no recoverable module attribute"):
        inductor_passes._compile_boxed_region(gm, {})
