"""DeepSeek-V4 sparse-attention override (registry-facing module).

The fused implementation (record tier + kernel) lives in ``ascendc.py``; the registration
is defined here so the override path stays ``override.deepseek_v4.sparse_attn.asc``.
The selector is not part of this bridge: it is the separate
``override.deepseek_v4.lightning_indexer.asc`` entry, and a training recipe must select the
two together (SMLAG emits the teacher, SLIKG consumes it).
Each fused core's own config builds the metadata it consumes (upstream #4897):
``AscCompressedSparseInnerAttention.Config.build_attention_metadata`` fills the
AscendC records and the LI record contract its config node names, so a recipe
no longer selects a metadata extension beside the kernel — ``sparse_attn.asc``
itself upgrades the default (identity) LI record provider to the AscendC one,
and a recipe that replaced that node first (torchao's quantized LI) keeps its
own contract.

The record geometry is the one thing the seam's builder cannot read: it is
handed the positions and nothing else, while a record is sized from the heads
and the head dimension.  So each fused entry states it once, defaulting to the
released V4-Flash shape -- which is what makes the shipped Flash recipes need no
arguments -- and a recipe that runs other widths states its own on the same
entry.  Nothing falls back: the port config declares both numbers without a
default, so an unstated geometry cannot size a record.
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention

if TYPE_CHECKING:
    from .ascendc import AscCompressedSparseInnerAttention


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description="Use cann_ops_transformer TND SparseFlashMLA with fused SMLAG backward",
)
def asc(
    cfg: CompressedSparseInnerAttention.Config,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_num_heads: int = 64,
    index_head_dim: int = 128,
) -> "AscCompressedSparseInnerAttention.Config":
    """The fused core, with the record geometry stated on the entry that selects it.

    The four widths default to the released V4-Flash shape, so the shipped Flash recipes need no
    arguments; a recipe that runs other widths states them here
    (``asc={"num_heads":16,"index_num_heads":8}``).  The reference core carries none of them:
    only a fused port's records are sized by them, and the port config has no default for them,
    so every record is sized from a value some entry stated.  The window and the selection width
    are not stated here -- they are the layer's own config fields (``window_size`` /
    ``index_topk``), which the model states once.

    The teacher this core emits on ``topk_scores``' gradient is the raw SMLAG marginal scaled by
    ``softmax_scale``; the selector's own entry (``lightning_indexer.asc``) consumes that edge
    with SLIKG.  Select both entries together when training the indexer.
    """
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .ascendc import AscCompressedSparseInnerAttention

    # ``sharding_config`` is copied from the source by ``derive``; the
    # sharding pass already ran before the overrides (update_from_config),
    # so the derived config keeps the sharding-pass values as-is.
    result = derive(
        cfg,
        AscCompressedSparseInnerAttention.Config,
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )
    return result
