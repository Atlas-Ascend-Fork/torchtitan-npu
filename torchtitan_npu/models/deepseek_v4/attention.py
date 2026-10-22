# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from attn_gym.sparse import AuxRequest, gather_attn
from torchtitan.models.common.attention import (
    BaseAttention,
    VarlenAttention,
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.document_intervals import document_intervals
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor, Indexer, LightningIndexer
from .metadata import DeepSeekV4Metadata, build_compressed_varlen_metadata
from .token_dispatcher import CPTokenDispatcher


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

    - ``compress_ratio == 1``: sliding window plus sink, no compressed stream;
    - ``compress_ratio == 4`` (CSA): the selector's picks, already numbered inside each query's
      own document -- the numbering ``gather_attn`` addresses a packed pool by;
    - a higher ratio (the model's HCA layers): every causally reachable compressed slot of the
      query's own document, which those layers reach without a selector.

    NPU overrides replace the whole class; they answer the same contract.  Two limits of this
    default are worth knowing before it is used for training:

    - it does **not** chunk the query axis: the operator materialises intermediates that grow
      with the sequence length, the window/selection width and the head dimension, so it is a
      debugging and numerics reference rather than the path to train on -- the fused
      ``sparse_attn.asc`` / ``sparse_attn.pypto`` recipes are the ones the model ships (both
      replace this class and keep working under context parallel, which this core refuses);
    - it carries no indexer loss of its own: the LI objective is a property of a fused port (its
      SMLAG backward hands the fused selector its teacher), so a recipe that runs this core
      trains no gradient for the indexer's compressor.
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
        return_lse: bool = False
        """Hand back the per-head log-sum-exp beside the output.  A test hook: nothing in the
        model consumes it, and a fused core takes its own from the kernel."""

        @property
        def attention_metadata_key(self):
            """This layer's own entry: the core's class plus its compression ratio.

            One entry per geometry is what lets the builder derive exactly the plan this layer
            grids -- no stack-wide ratio table on any config, and no entry shared by layers whose
            kernels address different pools.
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

            Built from the pre-shard ``positions`` alone: the entry carries its own ratio's plan
            over the row this builder derives, so one build serves every layer of this geometry --
            the decoder keys the map by this config's :attr:`attention_metadata_key`.  This
            builder's contract refuses context parallel; the fused core's config builds an entry
            whose hook rebuilds it for this rank.
            """
            del padding_mask, max_num_documents, max_context_length
            # A document starts where ``positions`` resets to 0; the window size is the core's
            # own field, so it is part of what this entry's plan was derived for.
            varlen = create_varlen_metadata_for_document(positions)
            # The selector's record is the selector port's business: this builder produces the
            # packed contract only, and a fused selector backend attaches its own record.
            return build_compressed_varlen_metadata(varlen, self.compress_ratio, window_size=self.window_size)

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Subclasses read ``self.window_size`` as an int.
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.return_lse = config.return_lse

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        *,
        attn_sink: torch.Tensor | None = None,
        sparse_indices=None,
        attention_metadata: DeepSeekV4Metadata | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The indexer's projections are the selector's business, not the core's.
        del idx_q, idx_k, idx_w
        if attn_sink is None:
            raise ValueError("CompressedSparseInnerAttention requires attn_sink")
        if not isinstance(attention_metadata, DeepSeekV4Metadata):
            raise TypeError(
                "CompressedSparseInnerAttention requires DeepSeekV4Metadata "
                f"attention metadata, got {type(attention_metadata).__name__}."
            )
        bsz, seqlen, _, head_dim = q.size()
        n_cmp = 0 if cmp_k is None else cmp_k.size(1)
        cu_seqlens = attention_metadata.varlen.cu_seq_q
        if n_cmp == 0:
            # A window-only layer: the pool is empty, and the operator wants both offsets.
            cu_seqlens_k = cu_seqlens
            kv_indices = q.new_empty((bsz, seqlen, 0), dtype=torch.long)
            sparse_kv = q.new_zeros((bsz, 1, 0, head_dim))
        else:
            # ``n_cmp`` is this tensor's width, so it is present here.
            assert cmp_k is not None
            cu_seqlens_k = attention_metadata.plan.cu_seqlens_cmp_k
            if cu_seqlens_k is None:
                raise ValueError(f"No compressed pool layout for ratio={self.compress_ratio}.")
            kv_indices = self._slots(sparse_indices, cu_seqlens, cu_seqlens_k, seqlen, n_cmp)
            sparse_kv = cmp_k.reshape(bsz, 1, n_cmp, head_dim)

        aux = AuxRequest(lse=True) if self.return_lse else None
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
        return output.transpose(1, 2), gathered.lse

    def _slots(
        self,
        sparse_indices: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqlen: int,
        n_cmp: int,
    ) -> torch.Tensor:
        """The selection, in the per-document numbering ``gather_attn`` reads."""
        if self.compress_ratio == 4:
            if sparse_indices is None:
                raise ValueError("CompressedSparseInnerAttention requires sparse_indices at ratio 4")
            if sparse_indices.ndim == 4 and sparse_indices.shape[2] == 1:
                sparse_indices = sparse_indices.squeeze(2)
            if sparse_indices.ndim != 3:
                raise ValueError("sparse_indices must be [B, L, K] or [B, L, 1, K]")
            return sparse_indices
        # HCA: every causally reachable slot, numbered inside the query's own document.
        pool_start_L1, pool_end_L1, local_L1 = document_intervals(cu_seqlens, cu_seqlens_k, num_tokens=seqlen)
        ends_L1 = torch.minimum(pool_end_L1, pool_start_L1 + (local_L1 + 1) // self.compress_ratio)
        slots_1N = torch.arange(n_cmp, device=cu_seqlens.device).unsqueeze(0)
        local_LN = slots_1N.expand(seqlen, -1) - pool_start_L1
        visible_LN = (slots_1N >= pool_start_L1) & (slots_1N < ends_L1)
        return torch.where(visible_LN, local_LN, -1).unsqueeze(0)


class CompressedSparseAttention(Module):
    """Thin CP boundary containing LI and the sparse attention core."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lightning_indexer: LightningIndexer.Config | None
        inner_attention: CompressedSparseInnerAttention.Config

    def __init__(self, config: Config):
        super().__init__()
        self.lightning_indexer = config.lightning_indexer.build() if config.lightning_indexer is not None else None
        self.inner_attention = config.inner_attention.build()

    def forward(
        self, q, swa_k, cmp_k=None, *, idx_q=None, idx_k=None, idx_w=None, attn_sink=None, attention_metadata=None
    ):
        sparse_indices = None
        if self.lightning_indexer is not None:
            sparse_indices = self.lightning_indexer(idx_q, idx_k, idx_w, attention_metadata=attention_metadata)
        return self.inner_attention(
            q,
            swa_k,
            cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            sparse_indices=sparse_indices,
            attn_sink=attn_sink,
            attention_metadata=attention_metadata,
        )


class Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        inner_attention: Module.Config
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

        # Built only for ``compress_ratio > 1`` layers (``indexer`` only for
        # ratio-4 CSA layers); the registry passes ``None`` otherwise.
        compressor: Compressor.Config | None
        indexer: Indexer.Config | None
        compressed_sparse_attention: CompressedSparseAttention.Config

        # The CP token dispatcher (the RoutedExperts mirror): a submodule of
        # the attention, wired once by ``Attention.parallelize``.
        token_dispatcher: CPTokenDispatcher.Config = field(default_factory=CPTokenDispatcher.Config)

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

        self.token_dispatcher = cfg.token_dispatcher.build()

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
        self.compressed_sparse_attention = cfg.compressed_sparse_attention.build()

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the attention, then wire the CP mesh on the
        attention's own token dispatcher (the ``RoutedExperts.parallelize``
        mirror).  The compressors' dispatchers are wired by their owners'
        ``parallelize`` through the framework's ``Module.parallelize``
        recursion."""
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(cp_mesh=parallel_dims.get_optional_mesh("cp"))

    @property
    def attention_metadata_key(self):
        """The map key of the metadata this layer reads (upstream #4897).

        ``BaseAttention``'s default forwards to the built ``inner_attention``;
        this wrapper builds a ``compressed_sparse_attention`` around the DSA
        core instead (the same config node sits at ``inner_attention`` for the
        decoder's traversal), so the key comes from the core one level down.
        """
        return self.compressed_sparse_attention.inner_attention.attention_metadata_key

    def forward(self, x, attention_metadata, positions):
        """The unified attention forward (CP and non-CP).

        The Q side and the swa projection run on the local stream; the
        token dispatcher's ops serve every consumer with no context-
        parallel special-casing: ``gather`` exchanges the post-RoPE
        ``swa_k`` rows (the window plan) into the packed ori stream, the
        compressors gather their own block rows internally, and ``select``
        packs the pooled streams into the padded containers.  The
        containers' all-gather is declarative — the wrapper's
        ``ShardingConfig`` (``cp: S(1) -> R``) emits it at the wrapper
        boundary.
        """
        window = attention_metadata.window
        bsz, seqlen, _ = x.size()

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, -1, self.head_dim)
        q = self.q_head_norm(q)
        # RoPE owns the [nope | rope] split: it receives the full-width
        # tensor and rotates only the trailing rope_head_dim channels.
        q = self.rope(q, positions=positions)

        # The swa projection + RoPE run on the local rows (the sender's own
        # doc-relative positions — the attention's positions convention
        # resets per document); the window gather exchanges the post-RoPE
        # rows into the packed ori stream.
        swa_k = self.kv_norm(self.wkv(x))
        # Flatten only for a singleton batch; otherwise preserve per-batch
        # positions so the RoPE cache matches the query's sequence length.
        rope_positions = positions.reshape(1, -1) if positions is not None and bsz == 1 else positions
        swa_k = self.rope(
            swa_k.unsqueeze(2),
            positions=rope_positions,
        ).squeeze(2)
        swa_k = self.token_dispatcher.gather(swa_k, window)

        cmp_k = None
        idx_q = idx_k = idx_w = None

        if self.compress_ratio > 1 and self.indexer is not None:
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(),
                qr.detach(),
                positions=positions,
                attention_metadata=attention_metadata,
            )
            # The indexer's outputs: idx_q / idx_w (local), idx_k (the
            # pooled stream — packed into the container).
            idx_k = self.token_dispatcher.select(idx_k, attention_metadata.plan)

        if self.compress_ratio > 1:
            assert self.compressor is not None, "compress_ratio > 1 requires the compressor submodule."
            plan = attention_metadata.plan
            pooled = self.compressor(x, attention_metadata)
            cmp_k = self.token_dispatcher.select(pooled, plan)

        # Inner-attention positional contract: absent components are None.
        #   sink + swa_k always; + cmp_k when compress_ratio > 1;
        #   + idx_q/idx_k/idx_w when compress_ratio == 4 (indexer layer).
        o = self.compressed_sparse_attention(
            q,
            swa_k,
            cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            attn_sink=self.attn_sink,
            attention_metadata=attention_metadata,
        )

        o = self.rope(o, positions=positions, inverse=True)

        # ``wo_a`` is a BatchedLinear over the head groups; group the heads
        # before the per-group matmul.
        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        o = self.wo_a(o)
        o = o.reshape(bsz, seqlen, -1)
        return self.wo_b(o)
