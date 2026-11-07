# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts for stream ordering and ownership; not NPU allocator tests."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import torchtitan_npu.patches.torch.distributed.fsdp as fsdp_patch


@pytest.fixture
def streams(monkeypatch):
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

        def wait_event(self, event):
            events.append(("wait_event", self.name, event))

        def record_event(self):
            events.append(("event", self.name))
            return self.name

    owner, copy, gather, consumer = [Stream(name) for name in ("owner", "copy", "gather", "consumer")]
    state = SimpleNamespace(current=owner)

    @contextmanager
    def use_stream(stream):
        previous, state.current = state.current, stream
        try:
            yield
        finally:
            state.current = previous

    handle = SimpleNamespace(current_stream=lambda: state.current, stream=use_stream)
    output = SimpleNamespace(device=torch.device("cpu"), record_stream=Mock())

    def allocate(size, *, dtype, device):
        assert size == (8,) and dtype == torch.float32 and device == torch.device("cpu")
        events.append(("allocate", state.current.name))
        return output

    comm = Mock(allocate=Mock(side_effect=allocate))
    monkeypatch.setattr(torch, "get_device_module", lambda device: handle)
    monkeypatch.setattr(fsdp_patch, "_AG_ALLOC_ON_COMPUTE_STREAM", True)
    return SimpleNamespace(
        owner=owner,
        copy=copy,
        gather=gather,
        consumer=consumer,
        state=state,
        handle=handle,
        output=output,
        comm=comm,
        events=events,
    )


def test_disabled_path_preserves_native_arguments(monkeypatch):
    native = Mock(return_value=object())
    monkeypatch.setattr(fsdp_patch, "_AG_ALLOC_ON_COMPUTE_STREAM", False)
    allocator = Mock(side_effect=AssertionError("Disabled path must not wrap the comm"))
    monkeypatch.setattr(fsdp_patch, "_ComputeStreamAllocator", allocator)
    monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_ALL_GATHER", native)
    args = ([object()],)
    assert fsdp_patch._patched_foreach_all_gather(*args, marker="native") is native.return_value
    native.assert_called_once_with(*args, marker="native")
    allocator.assert_not_called()


def _gather_arguments(s):
    return dict(
        fsdp_params=[SimpleNamespace()],
        group=object(),
        async_op=False,
        all_gather_copy_in_stream=s.copy,
        all_gather_stream=s.gather,
        device=torch.device("cpu"),
        all_gather_comm=s.comm,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_gather_transfers_ownership_or_protects_failed_construction(monkeypatch, streams, fail):
    s = streams
    native_result = fsdp_patch._fsdp_collectives.AllGatherResult(
        s.output, "gather", None, [[torch.float32]], [[8]], [8]
    )

    def native(fsdp_params, group, async_op, all_gather_copy_in_stream, all_gather_stream, device, all_gather_comm):
        with s.handle.stream(all_gather_copy_in_stream):
            assert all_gather_comm.allocate((8,), dtype=torch.float32, device=device) is s.output
            assert s.state.current is s.copy
            assert all_gather_comm._all_gather_output is s.output
            all_gather_comm(output_tensor=s.output)
        assert s.events == [("allocate", "owner"), ("wait_stream", "copy", "owner")]
        s.comm.assert_called_once_with(output_tensor=s.output)
        if fail:
            # Work was already submitted before the native result could be returned.
            raise RuntimeError("injected construction failure")
        return native_result

    monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_ALL_GATHER", native)
    arguments = _gather_arguments(s)
    if fail:
        with pytest.raises(RuntimeError, match="injected construction failure"):
            fsdp_patch._patched_foreach_all_gather(**arguments)
        assert [call.args[0] for call in s.output.record_stream.call_args_list] == [s.copy, s.gather]
    else:
        result = fsdp_patch._patched_foreach_all_gather(**arguments)
        assert isinstance(result, fsdp_patch._ComputeStreamAllGatherResult)
        assert tuple(result) == tuple(native_result)
        assert len(result) == 6
        assert result._compute_stream is s.owner
        s.output.record_stream.assert_not_called()


def _managed_result(s, work=None):
    result = fsdp_patch._ComputeStreamAllGatherResult(s.output, "gather", work, [[torch.float32]], [[8]], [8])
    result._compute_stream = s.owner
    result._side_streams = (s.copy, s.gather)
    return result


@pytest.mark.parametrize(
    ("different_consumer", "fail"),
    [(False, False), (True, False), (True, True)],
    ids=["same-stream", "other-stream", "partial-copyout-failure"],
)
def test_copyout_returns_consumer_event_or_records_on_failure(monkeypatch, streams, different_consumer, fail):
    s = streams
    result = _managed_result(s)
    consumer = s.consumer if different_consumer else s.owner

    def native(received, *, marker):
        assert received is result and marker == "payload"
        s.events.append(("copyout", s.state.current.name))
        if fail:
            raise RuntimeError("injected partial copyout")
        return "copied"

    monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_ALL_GATHER_COPY_OUT", native)
    with s.handle.stream(consumer):
        if fail:
            with pytest.raises(RuntimeError, match="injected partial copyout"):
                fsdp_patch._patched_foreach_all_gather_copy_out(result, marker="payload")
            expected = [s.copy, s.gather] + ([consumer] if different_consumer else [])
            assert [call.args[0] for call in s.output.record_stream.call_args_list] == expected
        else:
            assert fsdp_patch._patched_foreach_all_gather_copy_out(result, marker="payload") == "copied"
            expected = [("copyout", consumer.name)]
            if different_consumer:
                expected += [("event", "consumer"), ("wait_event", "owner", "consumer")]
            assert s.events == expected
            s.output.record_stream.assert_not_called()


@pytest.mark.parametrize(
    ("method", "asynchronous"),
    [("finalize_backward", True), ("reset_iter_state", False)],
    ids=["finalize-async", "reset-sync"],
)
def test_unconsumed_cleanup_waits_before_native_release(monkeypatch, streams, method, asynchronous):
    s = streams
    work = Mock() if asynchronous else None
    if work is not None:
        work.wait.side_effect = lambda: s.events.append(("work", s.state.current.name))
    group = SimpleNamespace(_all_gather_result=_managed_result(s, work))

    def native(received):
        assert received is group
        s.events.append(("release", s.state.current.name))
        group._all_gather_result = None
        return "cleaned"

    monkeypatch.setattr(fsdp_patch, f"_ORIGINAL_{method.upper()}", native)
    call = getattr(fsdp_patch, f"_patched_{method}")
    with s.handle.stream(s.consumer):
        assert call(group) == "cleaned"
        assert call(group) == "cleaned"
    expected = [("wait_event", "owner", "gather")]
    if asynchronous:
        expected.append(("work", "owner"))
    expected += [("release", "consumer"), ("release", "consumer")]
    assert s.events == expected
    assert group._all_gather_result is None
    s.output.record_stream.assert_not_called()


@pytest.mark.parametrize("method", ["finalize_backward", "reset_iter_state"])
def test_failed_work_wait_keeps_result_owned(monkeypatch, streams, method):
    work = Mock()
    work.wait.side_effect = RuntimeError("injected wait failure")
    result = _managed_result(streams, work)
    group = SimpleNamespace(_all_gather_result=result)
    native = Mock()
    monkeypatch.setattr(fsdp_patch, f"_ORIGINAL_{method.upper()}", native)
    with pytest.raises(RuntimeError, match="injected wait failure"):
        getattr(fsdp_patch, f"_patched_{method}")(group)
    assert group._all_gather_result is result
    native.assert_not_called()
