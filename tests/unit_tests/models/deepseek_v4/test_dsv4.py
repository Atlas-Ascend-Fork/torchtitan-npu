"""DSV4 host-logic CPU tests: the AscendC metadata layer, the packed layout,
the compressor/indexer/attention numerics, and the ported DSA semantics.

The AscendC override host wiring (``test_asc_metadata_*``) and the model-dir
contract (``test_layout_*``, ``test_compressor_*``, ``test_attention_*``,
``test_indexer_*``, ``test_ported_*``) all import against the real
torchtitan checkout with the plugin's patches applied; only the
``cann_ops_transformer`` surface is faked (the conftest's recorder).

Numerics verification criterion: a match is only accepted when it is
*bitwise*, or the residual sits at the fp32 rounding floor of the outputs
(at most a few ulps of the output scale) — a floating-point artifact, not
an arithmetic difference.  The compressor and the indexer are bitwise; the
reference attention core vs the eager golden stays within 1.8 ulps (the golden
is exact-fp32 by design, so its own f32/f64 delta is zero and the remaining
residual is the flex path's fp32 rounding).

Every comparison runs single-threaded (``_reproducible_numerics`` below): the
pool gradient's backward accumulates duplicate contributions in a
thread-scheduling dependent order, which makes the residual a sample of a
distribution rather than a number, and that distribution crosses this file's
4-ulp budget often enough to flake.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

DIM, HD, RD = 8, 16, 4
_GRAD_ULPS = 4
# Two documents longer than one compression block, with different tails (at
# ratio 128 neither is a whole number of blocks).
_GRADIENT_DOCS = (200, 133)
_EPS32 = 1.1920929e-7  # fp32 machine epsilon


@pytest.fixture(autouse=True)
def _reproducible_numerics():
    """Pin the intra-op thread count so the rounding-floor residuals are reproducible.

    The core and the golden both differentiate the *same* compressed-KV pool, and
    both read it through advanced indexing (the golden's ``kv[batch, index, :]``,
    attn-gym's ``kv[b, h, indices]``).  The pool's rows are read by many query
    rows, so that gather's backward is an accumulate scatter (ATen lowers it to
    ``index_put_`` with ``accumulate=True``), and its CPU kernel sums the
    duplicate contributions in a thread-scheduling dependent order: measured in
    isolation, ``index_put_(accumulate=True)`` and both advanced-index backwards
    give different last bits from run to run at 16 threads with a non-uniform
    upstream gradient.

    So with more than one intra-op thread each consumer drifts against *itself*
    by up to ~3.5 ulps of ``cmp_k``'s scale, and the core-vs-golden residual
    wanders between ~2.3 and ~5.2 ulps against this file's 4-ulp budget (an
    in-process 100-rep loop of the comparison below: median 3.46, max 5.19, 25
    runs over budget — only ``cmp_k`` moves; ``out``, ``q``, ``swa_k`` and
    ``sink`` are bitwise constant).  Single-threaded every residual is bitwise
    reproducible, with measured worsts of 1.8 (out), 1.1 (q), 1.4 (swa_k), 1.2
    (cmp_k) and 0.0 (sink) ulps.  ``torch.use_deterministic_algorithms(True)``
    does not help (torch does not call these kernels nondeterministic), and
    neither does pinning ``MKL_DYNAMIC``/``OMP_NUM_THREADS`` to 16.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class IdentityRoPE(nn.Module):
    """Identity rope — the compressor-side counterpart of ``manual_compress``.

    The pooling reference applies NO rope, so the compressor's
    ``self.rope(...)`` step must be a no-op for the comparison to be exact.
    The real rope numerics are covered by the rope patch's own on-device
    verification.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        dim: int = 1
        original_max_position_embeddings: int = 1
        rope_theta: float = 1.0

    def __init__(self, config=None):
        super().__init__()

    def forward(self, x, key=None, positions=None, *, inverse=False):
        if key is None:
            return x
        return x, x


class BuildCfg:
    def __init__(self, fn):
        self._fn = fn

    def build(self):
        return self._fn()


class RMS(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def make_linear(n, m):
    lin = nn.Linear(n, m, bias=False)
    nn.init.normal_(lin.weight, std=0.02)
    return lin


# B=1, S=64; docs A(10) B(17) C(20) D(17)
_CU = torch.tensor([0, 10, 27, 47, 64], dtype=torch.int32)


def _metadata_for(dsv4, varlen, ratios, **kwargs):
    """One entry per geometry, the way each layer's own config builds it."""
    # The four-document fixture below is deliberately off the 128-block grid: these are
    # the partial-block probes, not the loader's aligned stream.
    kwargs.setdefault("require_aligned", False)
    return {ratio: dsv4.metadata.build_compressed_varlen_metadata(varlen, ratio, **kwargs) for ratio in ratios}


def _build_metadata(dsv4):
    """The 64-token four-document layout (ratios 1/4/128) as the model builds it."""
    v = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=17, max_k=17)
    return _metadata_for(dsv4, v, (1, 4, 128))


def _pack_container(dsv4, pooled, plan):
    """The plan-driven container pack (the dispatcher's ``select``)."""
    disp = dsv4.token_dispatcher.CPTokenDispatcher(dsv4.token_dispatcher.CPTokenDispatcher.Config())
    return disp.select(pooled, plan)


def build_compressor(dsv4, ratio):
    coff = 2 if ratio == 4 else 1
    comp = dsv4.compressor.Compressor(
        dsv4.compressor.Compressor.Config(
            rope=BuildCfg(lambda: IdentityRoPE()),
            head_dim=HD,
            rope_head_dim=RD,
            compress_ratio=ratio,
            wkv=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            wgate=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            norm=BuildCfg(lambda: RMS(HD)),
        )
    )
    nn.init.trunc_normal_(comp.ape, std=0.02)
    return comp


def manual_compress(x, md, comp, ratio):
    """Independent per-document pooling reference (no rope).

    ``md`` is the entry of the geometry being pooled: its own plan, its own ratio.
    """
    plan = md.plan
    flat_x = x.flatten(0, 1).float()
    out = torch.zeros(md.batch_size * plan.out_width, HD)
    for doc_id, (qs, qe) in enumerate(
        zip(
            md.varlen.cu_seq_q[:-1].tolist(),
            md.varlen.cu_seq_q[1:].tolist(),
            strict=True,
        )
    ):
        length = qe - qs
        c_len = length // ratio
        if c_len == 0:
            continue
        tokens = flat_x[qs : qs + c_len * ratio].view(c_len, ratio, DIM)
        kv = F.linear(tokens, comp.wkv.weight.float())
        score = F.linear(tokens, comp.wgate.weight.float()) + comp.ape
        if ratio == 4:
            c_start = plan.cu_seqlens_cmp_k[doc_id].item()
            c_end = plan.cu_seqlens_cmp_k[doc_id + 1].item()
            valid = (torch.arange(c_end - c_start) != 0).view(-1, 1, 1)
            prev_idx = (torch.arange(c_end - c_start) - 1).clamp_min(0)
            kv_prev = torch.where(valid, kv[prev_idx, :, :HD], torch.zeros_like(kv[:, :, :HD]))
            score_prev = torch.where(
                valid,
                score[prev_idx, :, :HD],
                torch.full_like(score[:, :, :HD], float("-inf")),
            )
            kv = torch.cat([kv_prev, kv[:, :, HD:]], dim=1)
            score = torch.cat([score_prev, score[:, :, HD:]], dim=1)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = comp.norm(kv.to(x.dtype))
        start = plan.cu_seqlens_cmp_k[doc_id].item()
        out[start : start + c_len] = kv
    return out.view(md.batch_size, plan.out_width, HD)


def _pool_blocks_where(comp, block_tokens, block_positions, overlap_valid, dtype, ratio):
    """The pre-refactor pooling math (two ``where`` masks) — the bitwise
    reference for the scatter-masking implementation (exact-zero softmax
    weights make the two forms identical)."""
    kv = comp.wkv(block_tokens)
    score = comp.wgate(block_tokens) + comp.ape
    if ratio == 4:
        head_dim = comp.head_dim
        n = kv.shape[0]
        overlap_kv = kv.new_zeros(n, 2 * ratio, head_dim)
        overlap_score = score.new_full((n, 2 * ratio, head_dim), float("-inf"))
        overlap_kv[:, ratio:] = kv[:, :, head_dim:]
        overlap_score[:, ratio:] = score[:, :, head_dim:]
        prev = (torch.arange(n) - 1).clamp_min(0)
        valid = overlap_valid.view(-1, 1, 1)
        overlap_kv[:, :ratio] = torch.where(valid, kv[prev, :, :head_dim], overlap_kv[:, :ratio])
        overlap_score[:, :ratio] = torch.where(valid, score[prev, :, :head_dim], overlap_score[:, :ratio])
        kv, score = overlap_kv, overlap_score
    kv = (kv * score.softmax(dim=1)).sum(dim=1)
    kv = comp.norm(kv.to(dtype))
    nope_dim = comp.head_dim - comp.rope_head_dim
    kv_nope, kv_rope = torch.split(kv, [nope_dim, comp.rope_head_dim], dim=-1)
    kv_rope = comp.rope(kv_rope.unsqueeze(0).unsqueeze(2), positions=block_positions.unsqueeze(0)).squeeze(0).squeeze(1)
    return torch.cat([kv_nope, kv_rope], dim=-1)


def _positions_for(doc_lens):
    """A packed row whose documents have the given lengths (positions reset)."""
    return torch.cat([torch.arange(length) for length in doc_lens]).unsqueeze(0)


def _build_varlen(doc_lens):
    cu = torch.tensor([0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32)
    return VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=17, max_k=17)


def _asc_core_config(ratio, **shape):
    """The fused core's own config for one geometry, as the model registry fills it.

    The four widths are stated on the config -- the port config has no defaults -- and the
    window/top-k come from the layer's own fields, exactly as the ``asc`` override derives them.
    """
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
        AscCompressedSparseInnerAttention,
    )

    return AscCompressedSparseInnerAttention.Config(
        window_size=shape.get("window_size", 128),
        compress_ratio=ratio,
        softmax_scale=0.1,
        index_topk=shape.get("index_topk", 512),
        num_heads=shape.get("num_heads", 16),
        head_dim=shape.get("head_dim", 512),
        index_num_heads=shape.get("index_num_heads", 8),
        index_head_dim=shape.get("index_head_dim", 128),
    )


def _run_extension(dsv4, doc_lens, ratio, **shape):
    """The core config's own per-microbatch build of one geometry (the seam under test)."""
    return _asc_core_config(ratio, **shape).build_attention_metadata(_positions_for(doc_lens))


def _run_all(dsv4, doc_lens, ratios=(1, 4, 128), **shape):
    """Every geometry's entry, the way the decoder's map holds them."""
    return {ratio: _run_extension(dsv4, doc_lens, ratio, **shape) for ratio in ratios}


@pytest.fixture
def inputs(dsv4):
    torch.manual_seed(0)
    md = _build_metadata(dsv4)
    x = torch.randn(1, 64, DIM)
    pos = torch.arange(64).unsqueeze(0)
    return dsv4, md, x, pos


# ---------------------------------------------------------------------------
# The AscendC metadata layer (handler + kernel fills)
# ---------------------------------------------------------------------------


def _assert_rounding_floor(diff, scale, *, ulps: int = 4):
    """The residual must sit at the fp32 rounding floor of the outputs.

    ``ulps=4`` is the suite-wide budget; the measured worsts are 1.8 (attention
    outputs) and 1.2 (attention gradients) ulps, reproducible because the
    comparisons run single-threaded.  An arithmetic difference would exceed the
    budget by orders of magnitude.
    """
    assert scale > 0
    assert diff.max().item() <= ulps * _EPS32 * scale, (diff.max().item(), scale)


def test_compressor_numerics_ratio4(inputs):
    """C4A pooling vs the independent manual reference.

    The residual must sit at the fp32 rounding floor: the two paths project
    with different matmul batching (all blocks at once vs per document), so
    the CPU kernel's shape-dependent reduction order yields 1-ulp residuals
    (the documented batch-size artifact; the same-size calls are
    deterministic).  ``_assert_rounding_floor`` accepts that — bitwise would
    only hold on shape-invariant kernels; the suite-wide 4-ulp budget keeps
    a 2.2x margin over the measured worst (at most 1.8 ulps, single-threaded)
    while still flagging any multi-ulp arithmetic regression.
    """
    dsv4, md, x, _pos = inputs
    comp4 = build_compressor(dsv4, 4)
    pooled = comp4(x, md[4])
    cmp4 = _pack_container(dsv4, pooled, md[4].plan)
    # 2+4+5+4 complete blocks: the 64-token row has 15 whole blocks, not 64 // 4.
    assert cmp4.shape == (1, 15, 16)
    ref = manual_compress(x, md[4], comp4, 4)
    _assert_rounding_floor((cmp4 - ref).abs(), ref.abs().max().item())


def test_compressor_numerics_ratio128(inputs):
    """C128A with zero complete blocks — shape/emptiness only (no numerics)."""
    dsv4, md, x, _pos = inputs
    comp128 = build_compressor(dsv4, 128)
    cmp128 = comp128(x, md[128])
    assert cmp128.shape == (0, 16)
    assert cmp128.numel() == 0


def test_attention_numerics_ratio4_zero_blocks(dsv4):
    """The ratio-4 reference path with zero complete blocks — shape/finite
    (no numerics to compare; the golden is not exercised)."""
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=5, max_k=5)
    md_s = _metadata_for(dsv4, v, (4,))[4]
    comp_s = build_compressor(dsv4, 4)
    cmp_s = _pack_container(dsv4, comp_s(torch.randn(1, 8, DIM), md_s), md_s.plan)
    # Three and five tokens: only the second document contributes a complete block.
    assert cmp_s.shape == (1, 1, 16)
    core_s = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=4,
            softmax_scale=0.25,
            index_topk=4,
        )
    )
    with torch.no_grad():
        sparse_s = dsv4.compressor.select_compressed_slots(
            torch.randn(1, 8, 4, 8),
            torch.randn(1, 2, 8),
            torch.randn(1, 8, 4),
            md_s.varlen.cu_seq_q,
            md_s.plan.cu_seqlens_cmp_k,
            4,
        )
        out_s = core_s(
            torch.randn(1, 8, 3, HD),
            torch.randn(1, 8, HD),
            cmp_s,
            attn_sink=torch.randn(3),
            sparse_indices=sparse_s,
            attention_metadata=md_s,
        )
    # The first document has no complete group, so all of its picks are padding; the second
    # has one, and the rest of its width is padding.
    assert (sparse_s[0, :3] == -1).all()
    assert (sparse_s[0, 3:] >= 0).any()
    assert out_s.shape == (1, 8, 3, HD) and torch.isfinite(out_s).all()


def _gradient_case(dsv4, ratio=4):
    """The gradient comparison's inputs: one pool, one selection, one draw per consumer.

    One input set is built for both consumers, so any difference in the outputs
    or the gradients is arithmetic, not a different draw.  Raises if the pool is
    empty, i.e. if the case stops exercising the compressed path.
    """
    torch.manual_seed(0)
    docs = _GRADIENT_DOCS
    mds = _metadata_for(dsv4, _build_varlen(docs), (4, 128))
    md = mds[ratio]
    seqlen = sum(docs)
    sink = torch.randn(3)
    pool = _pack_container(dsv4, build_compressor(dsv4, ratio)(torch.randn(1, seqlen, DIM), md), md.plan)
    assert pool.shape[1] > 0, "the fixture must exercise a non-empty compressed pool"

    sparse = None
    if ratio == 4:
        n_cmp = int(mds[4].plan.cu_seqlens_cmp_k[-1])
        sparse = dsv4.compressor.select_compressed_slots(
            torch.randn(1, seqlen, 4, 8),
            torch.randn(1, n_cmp, 8),
            torch.randn(1, seqlen, 4),
            mds[4].varlen.cu_seq_q,
            mds[4].plan.cu_seqlens_cmp_k,
            8,
        )

    base = (
        torch.randn(1, seqlen, 3, HD),
        torch.randn(1, seqlen, HD),
        pool.detach().clone(),
        sink,
    )
    core = dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=8 if ratio == 4 else 0,
        )
    )
    return md, sparse, base, core


def _consumer_grads(consumer, sparse, md, base):
    """One differentiated pass; each consumer gets its own differentiable copy."""
    q, swa_k, cmp_k, attn_sink = (t.detach().clone().requires_grad_() for t in base)
    out = consumer(q, swa_k, cmp_k, attn_sink=attn_sink, sparse_indices=sparse, attention_metadata=md)
    upstream = torch.linspace(-1, 1, out.numel()).reshape_as(out)
    (out * upstream).sum().backward()
    return out, (q.grad, swa_k.grad, cmp_k.grad, attn_sink.grad)


def test_unaligned_documents_are_refused(dsv4):
    """A stream off the compression grid is an input bug, not a partial-block plan."""
    cu = torch.tensor([0, 10, 27], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=17, max_k=17)
    with pytest.raises(ValueError, match="per_doc_alignment"):
        dsv4.metadata.build_compressed_varlen_metadata(varlen, 4)
