# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The arch22 frame conversions, without a kernel.

``a3_key_frame`` states the per-document reach as the frame and spells an absent ``residual`` as
zeros; ``a3_key_gather`` picks the rows that reach covers.  Both are tensor arithmetic over the
tensors a caller already holds, so these tests pin the arithmetic and the dtypes a caller passes
straight to a kernel: the identity case (one degree, or the rank that reaches every document), each
document's prefix in document order, the invariants the walk assumes (boundaries from zero since
they *are* the row's coordinates), and the fact that the gather is a differentiable ``index_select``
whose gradient scatters back onto the gathered row.  ``A3KeyWorkaround`` is the object the fused
entry carries, so its build/compaction behaviour is pinned last.
"""

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.override.a3_workaround import A3KeyWorkaround, a3_key_frame, a3_key_gather


def _frame(bounds: list[int], reach: list[int], residual: list[int] | None = None):
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    seqused = torch.tensor(reach, dtype=torch.int32)
    return cu_seqlens, seqused, None if residual is None else torch.tensor(residual, dtype=torch.int32)


def _kernel_frame(bounds: list[int], reach: list[int], residual: list[int] | None = None) -> KernelFrame:
    cu_seqlens, seqused, residual = _frame(bounds, reach, residual)
    return KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused, residual=residual)


def test_a_whole_row_states_the_model_s_own_grid():
    """At one degree -- and on the rank that owns the row's tail -- the reach *is* the row."""
    cu_seqlens, seqused, residual = _frame([0, 8, 24], [8, 16], residual=[0, 0])

    boundaries, reach, out_residual = a3_key_frame(cu_seqlens, seqused, residual)

    assert boundaries.tolist() == cu_seqlens.tolist()
    assert reach.tolist() == seqused.tolist()
    assert out_residual is residual
    # ... and the gather is the identity, so the copy it drives is a no-op.
    gather = a3_key_gather(cu_seqlens, seqused)
    assert torch.equal(gather, torch.arange(24))


def test_a_prefix_is_the_running_sum_of_the_reach():
    """``cu_seqlens = cumsum(seqused)``, so the frame's end is this rank's chunk."""
    cu_seqlens, seqused, residual = _frame([0, 16, 48], [8, 12], residual=[0, 0])

    boundaries, reach, out_residual = a3_key_frame(cu_seqlens, seqused, residual)

    assert boundaries.dtype == torch.int32
    assert boundaries.tolist() == [0, 8, 20]
    assert reach.tolist() == [8, 12]
    assert torch.equal(reach, torch.diff(boundaries))
    assert out_residual.tolist() == [0, 0]


def test_an_unfolded_axis_states_no_partial_group_as_zeros():
    """A ratio-1 axis has no trailing group; arch22 rejects ``None`` and accepts zeros."""
    cu_seqlens, seqused, _ = _frame([0, 8, 24], [8, 16])

    _, _, residual = a3_key_frame(cu_seqlens, seqused)

    assert residual.dtype == torch.int32
    assert residual.tolist() == [0, 0]


@pytest.mark.parametrize(
    "bounds, reach, expected",
    [
        ([0, 16, 48], [8, 12], [*range(8), *range(16, 28)]),
        ([0, 32, 64], [16, 16], [*range(16), *range(32, 48)]),
        # The tail of one document may be shorter than the next one's prefix.
        ([0, 32, 64], [8, 24], [*range(8), *range(32, 56)]),
        # An empty document contributes nothing and shifts nothing.
        ([0, 8, 8, 32], [4, 0, 16], [*range(4), *range(8, 24)]),
    ],
    ids=["two-documents", "equal-chunks", "uneven", "empty-document"],
)
def test_the_gather_keeps_each_documents_prefix_in_document_order(bounds, reach, expected):
    cu_seqlens, seqused, _ = _frame(bounds, reach)

    gather = a3_key_gather(cu_seqlens, seqused)

    assert gather.tolist() == expected
    assert gather.dtype == torch.int64


def test_the_gather_is_differentiable_and_drops_the_unreached_rows():
    """A consumer's gradient scatters back onto the gathered row, which the wrapper reduces."""
    cu_seqlens, seqused, _ = _frame([0, 4, 12], [2, 3])
    gather = a3_key_gather(cu_seqlens, seqused)
    rows = torch.arange(24, dtype=torch.float32).reshape(12, 2).requires_grad_()

    compacted = rows.index_select(0, gather)
    assert compacted.shape == (5, 2)
    compacted.sum().backward()

    assert rows.grad is not None
    expected = torch.zeros_like(rows)
    # Every reached row carries exactly the gradient it was handed; the rows this rank may not read
    # carry none.
    expected[gather] = 1.0
    assert torch.equal(rows.grad, expected)


def _host_walk(cu_seqlens: torch.Tensor, seqused: torch.Tensor) -> list[int]:
    """The same indices, walked with host integers: the oracle the device-side version must match."""
    gather: list[int] = []
    for start, keep in zip(cu_seqlens[:-1].tolist(), seqused.tolist(), strict=True):
        gather += list(range(start, start + keep))
    return gather


@pytest.mark.parametrize(
    "bounds, reach",
    [
        ([0, 16, 48], [16, 32]),  # every document in full: the identity
        ([0, 16, 48], [8, 12]),
        ([0, 32, 64], [16, 16]),
        ([0, 32, 64], [8, 24]),
        ([0, 8, 8, 32], [4, 0, 16]),
        ([0, 4, 12], [2, 3]),
    ],
    ids=["whole-row", "two-documents", "equal-chunks", "uneven", "empty-document", "one-block"],
)
def test_the_gather_matches_a_host_walk_of_the_same_frame(bounds, reach):
    """Only a row length is read on the host; the indices come from the tensors themselves."""
    cu_seqlens, seqused, _ = _frame(bounds, reach)

    gather = a3_key_gather(cu_seqlens, seqused)

    assert gather.tolist() == _host_walk(cu_seqlens, seqused)
    assert gather.dtype == torch.int64


def test_the_gather_is_addressed_by_the_frame_it_is_derived_from():
    """``rows[gather]`` lands on the boundaries ``a3_key_frame`` states."""
    cu_seqlens, seqused, _ = _frame([0, 32, 64], [8, 24])
    boundaries, _, _ = a3_key_frame(cu_seqlens, seqused)
    rows = torch.arange(128, dtype=torch.float32).reshape(-1, 2)
    gather = a3_key_gather(cu_seqlens, seqused)

    assert gather.numel() == int(boundaries[-1])
    expected = torch.cat([rows[start : start + keep] for start, keep in zip([0, 32], [8, 24], strict=True)])
    assert torch.equal(rows.index_select(0, gather), expected)


def test_the_gather_reads_the_boundaries_as_the_rows_own_coordinates():
    """The walk assumes ``cu_seqlens`` starts at zero and ``row`` ends where the frame does.

    Both are structural: the boundaries are the packed row's coordinates, and the row a caller hands
    over *is* the gathered stream.  A frame starting elsewhere (a slab-relative one) would keep the
    leading rows of the first document twice over, so this pins the contract the callers keep.
    """
    cu_seqlens, seqused, _ = _frame([0, 16, 48], [8, 12])

    gather = a3_key_gather(cu_seqlens, seqused)

    assert int(cu_seqlens[0]) == 0
    assert gather.max() < cu_seqlens[-1]
    assert gather.numel() == int(seqused.sum())


def test_the_workaround_states_both_streams_for_the_kernels():
    """The entry's object is one frame and one gather per stream, derived from the frames it is given."""
    workaround, swa_k, cmp_k = A3KeyWorkaround.build(
        swa_k=_kernel_frame([0, 16, 48], [8, 12]),
        cmp_k=_kernel_frame([0, 4, 12], [2, 3], residual=[0, 0]),
    )

    assert swa_k.cu_seqlens.tolist() == [0, 8, 20]
    assert workaround.swa_k_gather.tolist() == [*range(8), *range(16, 28)]
    assert cmp_k is not None
    assert cmp_k.cu_seqlens.tolist() == [0, 2, 5]
    assert cmp_k.residual is not None
    assert cmp_k.residual.tolist() == [0, 0]
    assert workaround.cmp_k_gather.tolist() == [0, 1, 4, 5, 6]


def test_a_whole_row_compacts_to_nothing():
    """At one degree -- and on the rank that owns the row's tail -- no row moves."""
    workaround, swa_k, cmp_k = A3KeyWorkaround.build(
        swa_k=_kernel_frame([0, 8, 24], [8, 16]),
        cmp_k=_kernel_frame([0, 4, 12], [4, 8], residual=[0, 0]),
    )

    assert workaround.swa_k_gather is None
    assert cmp_k is not None and workaround.cmp_k_gather is None
    # The frames are stated either way: it is the rows that need no moving.
    assert swa_k.cu_seqlens.tolist() == [0, 8, 24]
    rows = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    assert rows.index_select(0, torch.arange(12)).shape == rows.shape
    # The compaction is then the identity, so a wrapper carries no branch of its own.
    rows = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    # No gather means no copy: the wrapper's ``if ... is not None`` is what skips it.


def test_a_geometry_without_a_pool_states_no_selector_stream():
    workaround, _, cmp_k = A3KeyWorkaround.build(swa_k=_kernel_frame([0, 8, 24], [8, 16]), cmp_k=None)

    assert cmp_k is None
    assert workaround.swa_k_gather is None and workaround.cmp_k_gather is None


def test_the_workaround_travels_with_the_entry_as_a_pytree_node():
    """The entry is a pytree node, so the workaround has to flatten into its tensors."""
    workaround, _, _ = A3KeyWorkaround.build(
        swa_k=_kernel_frame([0, 16, 48], [8, 12]),
        cmp_k=_kernel_frame([0, 4, 12], [2, 3], residual=[0, 0]),
    )

    leaves, spec = tree_flatten(workaround)
    assert {id(leaf) for leaf in leaves} == {id(workaround.swa_k_gather), id(workaround.cmp_k_gather)}
    rebuilt = tree_unflatten(leaves, spec)

    assert isinstance(rebuilt, A3KeyWorkaround)
    assert torch.equal(rebuilt.swa_k_gather, workaround.swa_k_gather)
    assert torch.equal(rebuilt.cmp_k_gather, workaround.cmp_k_gather)


def test_the_compaction_matches_the_gather_it_was_built_with():
    """The gather applied per stream: each document's prefix, in document order."""
    workaround, swa_k, cmp_k = A3KeyWorkaround.build(
        swa_k=_kernel_frame([0, 16, 48], [8, 12]),
        cmp_k=_kernel_frame([0, 4, 12], [2, 3], residual=[0, 0]),
    )
    rows = torch.arange(96, dtype=torch.float32).reshape(-1, 2)  # the 48-token row
    pool = torch.arange(24, dtype=torch.float32).reshape(-1, 2)  # the 12-block pool

    compacted_rows = rows.index_select(0, workaround.swa_k_gather)
    compacted_pool = pool.index_select(0, workaround.cmp_k_gather)
    assert compacted_rows.shape[0] == int(swa_k.cu_seqlens[-1])
    assert compacted_pool.shape[0] == int(cmp_k.cu_seqlens[-1])
    # The rows each was compacted from are the frames' own coordinates: document by document.
    assert torch.equal(compacted_rows, torch.cat([rows[0:8], rows[16:28]]))
    assert torch.equal(compacted_pool, torch.cat([pool[0:2], pool[4:7]]))
