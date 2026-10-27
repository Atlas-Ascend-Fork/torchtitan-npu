# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_gym.sparse import lightning_indexer
from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.models.common.document_intervals import document_intervals


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class CompressorImplementation(Configurable):
    """Select the compressor execution while keeping child overrides composable."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config

    def forward(
        self,
        compressor: "Compressor",
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compressor._forward(x, positions, cu_seqlens)


class Compressor(Module):
    """Document-packed key compression -- the **unsharded, CP-agnostic** implementation.

    The compressor sees one local token stream, that stream's own document-relative positions and
    (for the ratio-4 overlap only) that stream's document boundaries.  It knows nothing about
    context parallelism: pooling is a flat fold of ``ratio`` consecutive tokens, and the overlap's
    crossing-block state is masked at the first block of every ``cu_seqlens`` segment.  The
    context-parallel carry -- exchanging the predecessor's last block, prepending it, compressing
    and trimming it back out -- lives outside, in
    :class:`~torchtitan_npu.models.common.carry_exchange.CarryExchange`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        head_dim: int
        rope_head_dim: int
        compress_ratio: int
        implementation: "CompressorImplementation.Config" = field(
            default_factory=lambda: CompressorImplementation.Config()
        )

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.implementation = cfg.implementation.build()
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.nope_head_dim = cfg.head_dim - cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()
        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        # ``ape`` is a plain score bias on the compression block, not a
        # projection; own it directly like the upstream implementation.  Its
        # param_init is declared on the Config (registry-side, ``_APE_INIT``).
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features, dtype=torch.float32))

    @staticmethod
    def _overlap_transform(
        state: torch.Tensor,
        first_indices: torch.Tensor,
        *,
        value: float,
    ) -> torch.Tensor:
        """The C4A overlap window of one projected state.

        The window's left half is the previous block's a-series, rolled
        one block; the right half is the current block's b-series (the
        paper's C^a/C^b halves).  The first-block rows of the left half are
        filled with ``value``: the score state passes ``-inf`` (exactly zero
        softmax weight -- which also annihilates the rolled-in block-0 rows),
        the kv state passes ``0`` (defensive, like the reference
        implementations: its masked rows are multiplied by those exact-zero
        weights anyway).  The first block of every ``cu_seqlens`` segment is
        such a first block, which is what keeps a document -- or the CP
        wrapper's carry segment -- from reading across its own start.
        """
        assert state.size(-1) % 2 == 0, "the overlap window needs the 2*head_dim split"
        head_dim = state.size(-1) // 2
        n_blocks = state.size(1)
        # Functional roll/fill (gather + ``torch.where``) instead of
        # ``torch.roll`` + in-place scatter: the in-place write is dropped by
        # aot_autograd recompute and its data-dependent guard trips make_fx.
        prev_idx = (torch.arange(n_blocks, device=state.device) - 1).clamp_min(0)
        state_a = state[:, prev_idx, :, :head_dim]
        first_mask = torch.zeros(n_blocks, dtype=torch.bool, device=state.device).scatter(0, first_indices, True)
        state_a = torch.where(first_mask.view(1, -1, 1, 1), state.new_full((1,), value), state_a)
        state_b = state[:, :, :, head_dim:]
        return torch.cat([state_a, state_b], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress one local stream.

        ``cu_seqlens`` is keyword-only: a module with a sharding config is called through the
        framework's positional contract, which requires every positional-or-keyword parameter to
        be passed positionally.

        Args:
            x: The local stream, ``[B, L, D]`` (the row, one rank's slab, or the slab with the CP
                exchange's carry block prepended).
            positions: The stream's own document-relative positions, ``[B, L]``.  They reset at
                every document start, and the pooled blocks rotate at ``positions[..., ::ratio]``.
                With ``B > 1``, each row must start at position zero: each row boundary is a
                document boundary in the flattened stream. The upstream
                ``create_varlen_metadata_for_document`` includes these row starts in
                ``cu_seqlens``, so ratio-4 overlap never borrows tokens from the preceding row.
            cu_seqlens: The stream's document boundaries (``[n + 1]``), required by the ratio-4
                overlap: the first block of every segment has no predecessor block.  The
                non-overlap pooling ignores it -- a fold of ``ratio`` tokens needs no boundaries.

        Returns:
            The pooled key stream ``[B, L // ratio, D]``.
        """
        batch_size = x.size(0)
        if x.size(1) % self.compress_ratio:
            raise ValueError(
                f"each batch row ({x.size(1)} tokens) must contain whole compression blocks "
                f"of ratio {self.compress_ratio}."
            )
        # Implementations consume one packed stream whose boundaries already include every
        # batch row. Restore the container shape only after pooling, including RoPE.
        pooled = self.implementation.forward(
            self,
            x.flatten(0, 1).unsqueeze(0),
            positions.reshape(1, -1),
            cu_seqlens=cu_seqlens,
        )
        return pooled.reshape(batch_size, -1, pooled.size(-1))

    def _forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project, fold, softmax-gate, normalize and rotate the local stream.

        This is the **unsharded** reference implementation, and it is also the CP one: the carry
        that makes a slab compress like the row it came from is applied outside
        (:class:`~torchtitan_npu.models.common.carry_exchange.CarryExchange`), so the stream
        handed here is already the one to pool.  Returns the pooled key stream ``[1, L // ratio, D]``.
        """
        ratio = self.compress_ratio
        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(f"the compressor addresses one packed row, got shape {tuple(x.shape)}.")
        length = x.size(1)
        if length % ratio:
            raise ValueError(
                f"the local stream ({length} tokens) is not a whole number of compression blocks "
                f"of ratio {ratio}: documents are aligned by `dataloader.per_doc_alignment`, and "
                "the CP wrapper's augmented stream adds whole blocks."
            )
        n_blocks = length // ratio
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x).unflatten(1, (n_blocks, ratio))
            score = self.wgate(x).unflatten(1, (n_blocks, ratio)) + self.ape

        # -- overlap (ratio=4 only): the crossing block's a-series, masked at every segment start --
        if self.overlap:
            if cu_seqlens is None:
                raise ValueError(
                    "the ratio-4 overlap needs the local stream's cu_seqlens: the first block of "
                    "every document has no predecessor block to borrow."
                )
            first_indices = (cu_seqlens[:-1] // ratio).to(torch.int64)
            score = self._overlap_transform(score, first_indices, value=float("-inf"))
            kv = self._overlap_transform(kv, first_indices, value=0.0)

        # -- softmax pool + norm + RoPE (RoPE owns the [nope | rope] split) --
        # The gate is a softmax over the block's pooled rows: ``ratio`` of them, or ``2 * ratio``
        # once the overlap prepended the previous block's a-series.
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(x.dtype))
        return self.rope(kv.unsqueeze(2), positions=positions[..., ::ratio]).squeeze(2)


class Indexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: "Compressor.Config"
        # The ratio-4 overlap's cross-rank block, beside the compressor it feeds.
        carry_exchange: CarryExchange.Config
        num_index_heads: int
        index_head_dim: int
        rope_head_dim: int

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()
        # The exchange wires its own CP group in ``parallelize``; the compressor never sees it.
        self.carry_exchange = cfg.carry_exchange.build()

    def forward(self, x, qr, *, positions, attention_metadata, x_keys=None):
        """Project raw indexer queries, keys, and per-head weights.

        ``x`` is the query frame (this rank's chunks under CP) and ``qr`` its latent, so
        ``idx_q`` / ``idx_w`` derive from the same rows the queries do.  ``x_keys`` is the
        local slab the compressor projects its keys from (``x`` when it is not given): under CP
        the keys stay the slab, exactly as the attention's KV streams do.

        Returns:
            idx_q: Indexer queries ``[B, L, num_index_heads, index_head_dim]``
                with RoPE applied and Hadamard-rotated.
            idx_k: Indexer compressed keys in the container grid
                ``[B, L // ratio, index_head_dim]``, Hadamard-rotated.
            idx_w: Per-head indexer weights ``[B, L, num_index_heads]``
                (the local rows).
        """
        bsz, seqlen, _ = qr.size()
        positions_q = attention_metadata.positions_q
        idx_q = self.wq_b(qr)
        idx_q = idx_q.view(bsz, seqlen, self.num_index_heads, self.head_dim)
        idx_q = self.rope(idx_q, positions=positions_q)
        idx_q = self._rotate_activation(idx_q)
        # The carry is a stream transformation, not part of the compressor: the exchange
        # prepends the predecessor's block (a no-op at one degree) and the same compressor
        # instance then pools the augmented stream; its carry block is trimmed back off.
        carry = self.carry_exchange(
            x if x_keys is None else x_keys,
            positions,
            attention_metadata.cu_seq_q,
        )
        idx_k = self.compressor(
            carry.x,
            carry.positions,
            cu_seqlens=carry.cu_seqlens,
        )
        if carry.trim:
            idx_k = idx_k.narrow(1, carry.trim, idx_k.size(1) - carry.trim)
        idx_k = self._rotate_activation(idx_k)
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w

    @staticmethod
    def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        H = _hadamard(d, dtype=x.dtype, device=x.device)
        return F.linear(x, H) * (d**-0.5)


def select_compressed_slots(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """The reference selection: Attention Gym's packed ``lightning_indexer``.

    The operator applies each query's own document's causal limit itself and numbers the picks
    inside that document -- the numbering ``gather_attn`` reads -- so nothing rebases them here.
    Every row it cannot fill comes back as ``-1``, and the packed gather only reads entries
    inside the query's own document pool, so an empty pool can never name its neighbour's row.
    Its score is
    ``sum_h w * relu(q . k) / sqrt(H * D)``, and the caller's weights already carry the model's
    own scale, so multiplying them by ``sqrt(H * D)`` cancels that division exactly.  ``topk``
    is clamped to the pool's width, as this model's selection has always been.
    """
    bsz, _, heads, head_dim = idx_q.shape
    if bsz != 1:
        raise ValueError(
            "MBS>1 is currently supported only by the fused path; "
            f"the reference selector requires batch size 1 (got {bsz})."
        )
    return lightning_indexer(
        idx_q.float(),
        idx_k.float(),
        idx_w.float() * math.sqrt(heads * head_dim),
        min(topk, idx_k.size(1)),
        causal=True,
        compress_ratio=4,
        cu_seqlens=cu_seqlens.contiguous(),
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        impl="reference",
    )


def student_logits_at_picks(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> torch.Tensor:
    """The per-head scores at a document-local selection: the pass that carries gradient.

    The selection itself is discrete, so only the picks can be scored differentiably -- and a
    selection that returns indices alone leaves nothing to gather them from, which is why this
    is a pass of its own rather than a by-product of the selection.  ``topk_indices`` is
    renumbered onto the row-wide entries the pooled keys are stored in; a ``-1`` slot is
    gathered harmlessly and marked ``-inf``, the value a softmax drops. The reference path
    requires ``B == 1``, so the key axis is squeezed. Returns ``[1, L, K]``, head-summed and in
    the model's own score scale (``select_compressed_slots`` cancels the operator's internal
    division; this pass never divides).

    This is the pass ``LightningIndexer`` returns beside the selection, and the fused
    selector's backward consumes the same quantity: at the picks, the student logits are the
    only differentiable edge of the selection.
    """
    if idx_q.size(0) != 1:
        raise ValueError(
            "MBS>1 is currently supported only by the fused path; "
            f"the reference student logits require batch size 1 (got {idx_q.size(0)})."
        )
    pool_start_L1, _, _ = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=idx_q.size(1))
    picked_BLKDi = idx_k.squeeze(0)[(topk_indices + pool_start_L1).squeeze(0).clamp_min(0)].unsqueeze(0)
    scores_BLHK = torch.einsum("blhd,blkd->blhk", idx_q, picked_BLKDi)
    weighted_BLHK = scores_BLHK.relu() * idx_w.unsqueeze(-1)
    return weighted_BLHK.sum(dim=2).masked_fill(topk_indices < 0, -torch.inf)


class LightningIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int

    def __init__(self, config: Config):
        super().__init__()
        self.index_topk = config.index_topk

    def forward(self, idx_q, idx_k, idx_w, *, attention_metadata):
        """Select the compressed slots and score them differentiably.

        Returns ``(topk_indices, topk_scores)``:

        - ``topk_indices`` ``[B, L, K]`` document-local with ``-1`` for an unused slot.  The
          reference path keeps the selection operator's order: ``gather_attn`` sums the
          selected rows in that order, and re-sorting here would perturb the reference
          numerics (the checked-in golden is exact).  The fused selector canonicalises to
          descending order inside its Function, because there the saved indices must pair
          slot-for-slot with the teacher;
        - ``topk_scores`` ``[B, L, K]``: the student logits at those picks, differentiable in
          ``idx_q``/``idx_k``/``idx_w`` and ``-inf`` on every ``-1`` slot.  This is the value
          contract the reference path exposes; the fused override returns an unread carrier
          on the same edge because its teacher arrives on the gradient.
        """
        # The selector runs on a layer of its own ratio and reads that layer's entry; a
        # geometry without a compressed pool is the only shape it refuses.
        if attention_metadata.ratio <= 1:
            raise ValueError("LightningIndexer requires a compressed geometry (ratio > 1).")
        cu_seqlens = attention_metadata.q.cu_seqlens
        cmp_frame = attention_metadata.cmp_k
        if cmp_frame is None:
            raise ValueError("a compressed geometry needs the entry's compressed frame")
        cu_seqlens_k = cmp_frame.cu_seqlens
        # Selection is discrete: it carries no gradient.  The operator's order is kept as-is
        # for the reference path; the fused selector emits the canonical descending order
        # (see the module docstring), so the two paths agree on the *set* and on ``-1``.
        with torch.no_grad():
            topk_indices = select_compressed_slots(
                idx_q,
                idx_k,
                idx_w,
                cu_seqlens,
                cu_seqlens_k,
                self.index_topk,
            )
        topk_scores = student_logits_at_picks(
            idx_q,
            idx_k,
            idx_w,
            topk_indices,
            cu_seqlens,
            cu_seqlens_k,
        )
        return topk_indices, topk_scores
