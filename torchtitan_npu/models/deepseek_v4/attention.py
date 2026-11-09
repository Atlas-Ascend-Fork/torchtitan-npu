# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from attn_gym.sparse import AuxRequest, gather_attn
from torchtitan.models.common.attention import BaseAttention, VarlenAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.deepseek_v4.cp_attention import restore_key_stream
from torchtitan_npu.models.deepseek_v4.document_intervals import document_intervals
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer
from .indexer_loss import IndexerLoss
from .metadata import DeepSeekV4Metadata


class UnitScaleRMSNorm(RMSNorm):
    """RMSNorm with a fixed unit scale excluded from checkpoints."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        elementwise_affine: bool = field(default=False, init=False)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        del self.weight
        self.register_buffer("weight", torch.ones(config.normalized_shape), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        del buffer_device
        self.weight.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the original query normalization's input-dtype rounding.
        assert self.eps is not None
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


class CompressedSparseInnerAttention(Module):
    """DeepSeek sparse attention core for DeepSeek-V4.

    The core is Attention Gym's ``gather_attn`` over the packed offsets the metadata already
    carries, so no document-aware block mask is built per layer:

    - ``compress_ratio == 0``: sliding window plus sink, no compressed stream;
    - ``compress_ratio == 4`` (CSA): the indexer's picks, already numbered inside each query's
      own document -- the numbering ``gather_attn`` addresses a packed pool by;
    - a higher ratio (the model's HCA layers): every causally reachable compressed slot of the
      query's own document, which those layers reach without an indexer.

    NPU overrides replace the whole class; they answer the same contract.  Two limits of this
    default are worth knowing before it is used for training:

    - it does **not** chunk the query axis: the operator materialises intermediates that grow
      with the sequence length, the window/selection width and the head dimension, so it is a
      debugging and numerics reference rather than the path to train on -- the fused
      ``sparse_attn.asc`` / ``sparse_attn.pypto`` recipes are the ones the model ships (both
      replace this class and keep working under context parallel, which this core refuses);
    - it carries the indexer's distillation objective when one is configured (``aux_loss``):
      the teacher is recomputed from this operator's own log-sum-exp, so the core trains the
      indexer by itself.  Left unconfigured, the indexer gets no gradient here.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        # The int DSA window, replacing the varlen attention's ``(left, right)`` tuple: the
        # model's window is a count of preceding tokens.  The varlen base is also the type the
        # pinned decoder's mask dispatch recognises on the stacks that still build a single
        # mask object; this stack builds its metadata through the seam instead.
        window_size: int  # pyrefly: ignore [bad-override]
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        aux_loss: IndexerLoss.Config | None = None
        """The indexer's distillation objective, on the geometries that consume a selection.

        ``None`` on a layer whose core reaches its pool without an indexer -- a window-only or a
        heavily-compressed geometry: nothing is selected there, so there is no teacher and no
        student to compare.
        """

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the core's class plus its compression ratio.

            One entry per geometry is what lets the builder derive exactly the grid this layer
            addresses -- no stack-wide ratio table on any config, and no entry shared by layers
            whose kernels address different pools.
            """
            return (type(self), self.compress_ratio)

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> DeepSeekV4Metadata:
            """This geometry's per-microbatch contract (upstream #4897).

            Built from the pre-shard ``positions`` alone: the entry carries the row's boundaries
            and its own ratio's fold of them, so one build serves every layer of this geometry --
            the decoder keys the map by this config's :attr:`attention_metadata_key`.  This
            builder's contract refuses context parallel; the fused core's config builds an entry
            whose hook rebuilds it for this rank.
            """
            del padding_mask, max_num_documents, max_context_length
            # A document starts where ``positions`` resets to 0; this layer's ratio folds the row
            # into the pool grid its kernels address.  The selector's records are the selector
            # port's business: this builder produces the row's frames only, and a fused selector
            # backend attaches its own records.
            return DeepSeekV4Metadata.from_positions(positions, self.compress_ratio)

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Subclasses read ``self.window_size`` as an int.
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def make_hca_indices(
        self, q: torch.Tensor, cmp_k: torch.Tensor, cu_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor
    ):
        """A geometry without an indexer reaches every causally complete group of its document.

        Document-local numbering with ``-1`` for an unreachable slot -- the coordinate system
        ``gather_attn`` reads.  Built here rather than carried on the metadata: only this core
        reads it, and the matrix is ``S x (S / ratio)``.  ``q`` and ``cmp_k`` supply the query rows
        and the pool width as shapes, so nothing reads a device value on the host.
        """
        seqlen, n_cmp = q.size(1), cmp_k.size(1)
        pool_start, pool_end, local = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=seqlen)
        ends = torch.minimum(pool_end, pool_start + (local + 1) // self.compress_ratio)
        slots = torch.arange(n_cmp, device=cu_seqlens.device).unsqueeze(0)
        visible = (slots >= pool_start) & (slots < ends)
        return torch.where(visible, slots.expand(seqlen, -1) - pool_start, -1).unsqueeze(0)

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        attn_sink: torch.Tensor,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        attention_metadata: DeepSeekV4Metadata,
    ) -> torch.Tensor:
        # The indexer's scores at the picks; the distillation objective is their only reader.
        if not isinstance(attention_metadata, DeepSeekV4Metadata):
            raise TypeError(
                "CompressedSparseInnerAttention requires DeepSeekV4Metadata "
                f"attention metadata, got {type(attention_metadata).__name__}."
            )
        if self.compress_ratio == 4 and (cmp_k is None) != (topk_indices is None):
            raise ValueError("ratio-4 attention requires cmp_k and topk_indices together.")
        bsz, seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(1)
        # The frames the kernels grid: the query's and, for a pooling geometry, the compressed
        # pool's.  The reference path and a fused port read the same ones.
        cu_seqlens = attention_metadata.q.cu_seqlens
        if n_cmp == 0:
            # A window-only layer: the pool is empty, and the operator wants both offsets.
            cu_seqlens_k = cu_seqlens
            kv_indices = q.new_empty((bsz, seqlen, 0), dtype=torch.long)
            sparse_kv = q.new_zeros((bsz, 1, 0, head_dim))
        else:
            # ``n_cmp`` is this tensor's width, so the frame is present here.
            assert cmp_k is not None
            cmp_frame = attention_metadata.cmp_k
            if cmp_frame is None:
                raise ValueError("a compressed KV needs the entry's compressed frame")
            cu_seqlens_k = cmp_frame.cu_seqlens
            # The indexer's picks at ratio four, or, without one, every complete document group.
            if self.compress_ratio == 4:
                kv_indices = topk_indices
            else:
                kv_indices = self.make_hca_indices(q, cmp_k, cu_seqlens, cu_seqlens_k)
            sparse_kv = cmp_k.reshape(bsz, 1, n_cmp, head_dim)

        # Bound to a local so the guard below can narrow it.
        aux_loss = self.aux_loss
        aux = AuxRequest(lse=True) if aux_loss is not None and self.training else None
        result = gather_attn(
            q.transpose(1, 2),
            swa_k.reshape(bsz, 1, seqlen, head_dim),
            sparse_kv,
            kv_indices,
            attention_sink=attn_sink,
            sliding_window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            scale=self.softmax_scale,
            impl="reference",
            return_aux=aux,
        )
        if aux is None:
            assert isinstance(result, torch.Tensor)
            return result.transpose(1, 2)
        output, gathered = result
        out_BLHD = output.transpose(1, 2)
        if aux_loss is None or topk_indices is None or topk_scores is None or cmp_k is None:
            return out_BLHD
        # This layer's own distribution over the slots it read: the log-sum-exp already carries
        # the window, the selection and the sink, so the compressed
        # slots' mass falls out of it without a second pass over the row.  A packed row is one
        # batch element -- the indexer's picks are numbered against it -- so the objective takes
        # the row, not a batch.
        assert bsz == 1, "the reference indexer objective takes one packed row"
        return aux_loss(
            q.squeeze(0),
            cmp_k.squeeze(0),
            topk_indices.squeeze(0),
            topk_scores.squeeze(0),
            gathered.lse.squeeze(0).transpose(0, 1),
            cu_seqlens,
            cu_seqlens_k,
            softmax_scale=self.softmax_scale,
            carrier=out_BLHD,
        )


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        rope: RoPE.Config
        head_dim: int
        rope_head_dim: int
        q_lora_rank: int
        n_groups: int
        compress_ratio: int
        norm_eps: float

        # Declare submodule configs as fields so sharding can be assigned before
        # the modules are built.
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        q_head_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

        # Built only for ``compress_ratio > 1`` layers (``indexer`` only for ratio-4 CSA
        # layers, and it carries the selection node inside); the registry passes ``None``
        # otherwise.
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        # The sparse core the decoder replaces wholesale with ``sparse_attn.asc``.  It is the
        # layer's only metadata backend and the CP boundary for its KV streams; the annotation
        # stays the base type because a port's config is a replacement of this node.
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.rope = cfg.rope.build()

        # The CP mesh this layer restores its key side on, wired by ``parallelize``; ``None`` at
        # one degree, where the metadata carries no plan and nothing moves.
        self.cp_group = None

        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.q_head_norm = cfg.q_head_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # Bare head-wise sink parameter (fp32), matching the inference
        # reference and the kernels' ``[N1]`` sink contract.
        self.attn_sink = torch.nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))

        self.compressor = cfg.compressor.build() if cfg.compressor is not None else None
        self.indexer = cfg.indexer.build() if cfg.indexer is not None else None
        self.inner_attention = cfg.inner_attention.build()

    def parallelize(self, parallel_dims) -> None:
        """Take the CP mesh the layer's key stream is restored on, then recurse.

        The compressor and its carry exchange take the same mesh through their own
        ``parallelize`` (the framework recurses into every child module), so nothing here wraps
        them: their module paths, sharding and checkpoint entries stay as they are.
        """
        self.cp_group = parallel_dims.get_optional_mesh("cp")
        super().parallelize(parallel_dims)

    def forward(self, x, attention_metadata, positions):
        """The unified attention forward (CP and non-CP).

        The input layout already put these rows on **chunk ``r`` of every document**, so the
        query side projects them as they arrive and never moves.  The key side projects, rotates
        and pools the same rows -- every step is per token, so it commutes with the layout -- and
        each consumer restores what it reads on its own: the compressor its pooling window, the
        core this rank's swa window, whose sharding then gathers the whole row at its boundary.
        """
        bsz, seqlen, _ = x.size()
        metadata = attention_metadata

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, -1, self.head_dim)
        q = self.q_head_norm(q)
        # RoPE owns the [nope | rope] split: it receives the full-width
        # tensor and rotates only the trailing rope_head_dim channels.  The
        # positions are the ones the input layout carried with these rows.
        q = self.rope(q, positions=positions)

        # The key stream's own positions are the entry's: the input's at one degree, this rank's
        # window under CP.  The queries rotate at the positions the input carried.
        position_k = metadata.position_k

        # The swa projection + RoPE run on the rows the layout produced: RoPE is per token, so it
        # commutes with the movement.  The restore then hands the core this rank's window, whose
        # sharding gathers the rows to the whole row at the core's own boundary.
        swa_k = self.kv_norm(self.wkv(x))
        swa_k = self.rope(swa_k.unsqueeze(2), positions=positions).squeeze(2)
        swa_k = restore_key_stream(metadata.xchg, swa_k, group=self.cp_group)

        cmp_k = None
        topk_indices = topk_scores = None

        if self.compress_ratio > 1 and self.indexer is not None:
            # The indexer projects the selection's queries, keys and weights -- they ride the
            # doc-chunk rows with the attention's -- and selects from the compressed pool.
            # Both steps live behind this one call, so a fused kernel replaces only the inner
            # node's forward.
            topk_indices, topk_scores = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_metadata=metadata,
            )
        if self.compress_ratio > 1:
            assert self.compressor is not None, "compress_ratio > 1 requires the compressor submodule."
            # The compressor owns the restore and the ratio-4 carry: it pools the window this rank
            # holds and hands back the row's blocks, so the layer never sees either step.
            cmp_k = self.compressor(
                x,
                position_k,
                cu_seqlens=metadata.swa_k.cu_seqlens,
                attention_metadata=metadata,
            )

        # Inner-attention positional contract: sink + swa_k always; the compressed pool and
        # the indexer's ``(topk_indices, topk_scores)`` pair only on a pooled layer.
        o = self.inner_attention(
            q,
            swa_k,
            self.attn_sink,
            cmp_k=cmp_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            attention_metadata=metadata,
        )

        o = self.rope(o, positions=positions, inverse=True)

        # ``wo_a`` is a BatchedLinear over the head groups; group the heads
        # before the per-group matmul.
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        o = self.wo_a(o)
        o = o.reshape(bsz, seqlen, -1)
        o = self.wo_b(o)
        # The projection is per token, so ``wo_b``'s output stays in the frame the input
        # arrived in: the residual and every other consumer are token-wise, and the loss is
        # taken against labels the same layout moved.
        return o
