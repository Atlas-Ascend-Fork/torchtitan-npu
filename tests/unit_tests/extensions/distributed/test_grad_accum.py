# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan_npu.extensions.distributed import grad_accum


class _RecordingAllGatherComm:
    """CPU fake for the allocation surface used by the copy-in planner."""

    def __init__(self) -> None:
        self.request: tuple[tuple[int, ...], torch.dtype, torch.device] | None = None

    def allocate(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        self.request = (shape, dtype, device)
        return torch.empty(shape, dtype=dtype, device=device)


def test_copy_in_plan_uses_element_splits_for_uniform_dtype() -> None:
    sources = [torch.empty(2, dtype=torch.float32), torch.empty(3, dtype=torch.float16)]

    plan = grad_accum._build_all_gather_copy_in_plan(
        sources,
        [[torch.bfloat16], [torch.bfloat16]],
        [[2], [3]],
    )

    assert plan.sources is sources
    assert plan.destination_dtypes == [torch.bfloat16, torch.bfloat16]
    assert plan.element_numels == [2, 3]
    assert plan.buffer_dtype == torch.bfloat16
    assert plan.input_split_sizes == [2, 3]
    assert plan.local_numel == 5


def test_copy_in_plan_uses_byte_splits_for_mixed_dtype() -> None:
    sources = [torch.empty(2, dtype=torch.float32), torch.empty(1, dtype=torch.float64)]

    plan = grad_accum._build_all_gather_copy_in_plan(
        sources,
        [[torch.bfloat16], [torch.float32]],
        [[2], [1]],
    )

    assert plan.destination_dtypes == [torch.bfloat16, torch.float32]
    assert plan.element_numels == [2, 1]
    assert plan.buffer_dtype == torch.uint8
    assert plan.input_split_sizes == [4, 4]
    assert plan.local_numel == 8


def test_allocate_and_stage_uses_rank_slice_and_plan_metadata(monkeypatch) -> None:
    sources = [torch.empty(2, dtype=torch.float32), torch.empty(3, dtype=torch.float16)]
    plan = grad_accum._build_all_gather_copy_in_plan(
        sources,
        [[torch.bfloat16], [torch.bfloat16]],
        [[2], [3]],
    )
    comm = _RecordingAllGatherComm()
    staged: dict[str, object] = {}
    copy_in_stream = object()

    def record_batched_h2d(
        actual_sources: list[torch.Tensor],
        destination: torch.Tensor,
        split_sizes: list[int],
        destination_dtypes: list[torch.dtype],
        *,
        copy_in_stream: object,
    ) -> None:
        staged.update(
            sources=actual_sources,
            destination=destination,
            split_sizes=split_sizes,
            destination_dtypes=destination_dtypes,
            copy_in_stream=copy_in_stream,
        )

    monkeypatch.setattr(grad_accum, "_batched_h2d", record_batched_h2d)

    output, local_input = grad_accum._allocate_and_stage_all_gather_input(
        plan,
        comm,
        torch.device("cpu"),
        world_size=4,
        rank=2,
        copy_in_stream=copy_in_stream,
    )

    assert comm.request == ((20,), torch.bfloat16, torch.device("cpu"))
    assert output.shape == (20,)
    assert local_input.shape == (5,)
    assert local_input.storage_offset() == 10
    assert staged["sources"] is sources
    assert staged["destination"] is local_input
    assert staged["split_sizes"] == [2, 3]
    assert staged["destination_dtypes"] == [torch.bfloat16, torch.bfloat16]
    assert staged["copy_in_stream"] is copy_in_stream


def test_cast_scratch_views_align_mixed_item_sizes() -> None:
    grad_accum.clear()
    stream = object()
    try:
        views = grad_accum._cast_scratch_views(
            torch.device("cpu"),
            [
                (1, torch.float16),
                (1, torch.float32),
                (1, torch.float64),
            ],
            stream,
        )

        base_pointer = views[0].data_ptr()
        assert [view.dtype for view in views] == [
            torch.float16,
            torch.float32,
            torch.float64,
        ]
        assert [view.data_ptr() - base_pointer for view in views] == [0, 4, 8]
    finally:
        grad_accum.clear()


@dataclass(frozen=True)
class _FakeStream:
    """CPU stand-in for stream wrappers with native-handle equality/hash."""

    handle: int


def test_cast_scratch_isolates_streams_and_reuses_equivalent_wrappers(monkeypatch) -> None:
    monkeypatch.setattr(grad_accum, "_CAST_SCRATCH", {})
    device = torch.device("cpu")
    stream = _FakeStream(1)
    first = grad_accum._cast_scratch_views(device, [(3, torch.float32)], stream)[0]
    first.fill_(1.5)
    second = grad_accum._cast_scratch_views(device, [(3, torch.float32)], _FakeStream(2))[0]
    second.fill_(-2.5)
    alias = _FakeStream(1)
    reused = grad_accum._cast_scratch_views(device, [(3, torch.float32)], alias)[0]

    assert alias is not stream
    assert reused.data_ptr() == first.data_ptr()
    torch.testing.assert_close(reused, torch.full((3,), 1.5))
    torch.testing.assert_close(second, torch.full((3,), -2.5))


def test_batched_h2d_uniform_direct_and_cast_uses_three_dispatches(
    monkeypatch,
) -> None:
    grad_accum.clear()
    stream = object()
    direct = torch.tensor([2], dtype=torch.bfloat16)
    cast = torch.tensor([1.5], dtype=torch.float32)
    destination = torch.empty(2, dtype=torch.bfloat16)
    dispatches = 0
    foreach_copy = torch._foreach_copy_

    def record_foreach_copy(*args, **kwargs):
        nonlocal dispatches
        dispatches += 1
        return foreach_copy(*args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_copy_", record_foreach_copy)
    try:
        grad_accum._batched_h2d(
            [direct, cast],
            destination,
            [1, 1],
            [torch.bfloat16, torch.bfloat16],
            copy_in_stream=stream,
        )

        assert dispatches == 3
        torch.testing.assert_close(destination, torch.tensor([2, 1.5], dtype=torch.bfloat16))
    finally:
        grad_accum.clear()


def test_batched_h2d_packs_unaligned_mixed_dtype_output(monkeypatch) -> None:
    grad_accum.clear()
    stream = object()
    direct_uint8 = torch.tensor([7], dtype=torch.uint8)
    cast_float = torch.tensor([1.5], dtype=torch.float16)
    direct_int = torch.tensor([2, -3, 4], dtype=torch.int16)
    destination = torch.empty(11, dtype=torch.uint8)
    dispatches = 0
    foreach_copy = torch._foreach_copy_

    def record_foreach_copy(*args, **kwargs):
        nonlocal dispatches
        dispatches += 1
        return foreach_copy(*args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_copy_", record_foreach_copy)
    try:
        grad_accum._batched_h2d(
            [direct_uint8, cast_float, direct_int],
            destination,
            [1, 1, 3],
            [torch.uint8, torch.float32, torch.int16],
            copy_in_stream=stream,
        )

        expected = torch.cat(
            [
                direct_uint8.view(torch.uint8),
                cast_float.to(torch.float32).view(torch.uint8),
                direct_int.view(torch.uint8),
            ]
        )
        assert dispatches == 4
        assert torch.equal(destination, expected)
    finally:
        grad_accum.clear()


def test_clear_resets_copy_in_diagnostics() -> None:
    grad_accum._BATCHED_COPY_IN_LOGGED = True
    grad_accum._DECLINE_LOGGED = 3
    grad_accum._CAST_SCRATCH[(torch.device("cpu"), object())] = torch.empty(1, dtype=torch.uint8)

    grad_accum.clear()

    assert not grad_accum._BATCHED_COPY_IN_LOGGED
    assert grad_accum._DECLINE_LOGGED == 0
    assert not grad_accum._CAST_SCRATCH
