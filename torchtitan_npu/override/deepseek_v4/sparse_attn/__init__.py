"""DeepSeek-V4 sparse-attention overrides (registry-facing module).

The AscendC fused implementation (record tier + kernel) lives in
``ascendc.py``, the PyPTO variant in ``pypto.py``; the registrations are
defined here so the override paths stay
``override.deepseek_v4.sparse_attn.{asc_li, asc, pypto, pypto_li}``.
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

import spmd_types as spmd
from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import dense_param_placement

from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer

if TYPE_CHECKING:
    from .ascendc import AscCompressedSparseInnerAttention
    from .pypto import PyPTOCompressedSparseInnerAttention


@override(
    target=LightningIndexer.Config,
    exact=True,
    description="Use AscendC TND LightningIndexer",
)
def asc_li(cfg: LightningIndexer.Config):
    from .ascendc import AscLightningIndexer

    return derive(cfg, AscLightningIndexer.Config)


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description=("Use cann_ops_transformer TND LightningIndexer and SparseFlashMLA with fused SASG/SLIG backward"),
)
def asc(
    cfg: CompressedSparseInnerAttention.Config,
    indexer_loss_coeff: float = 1.0,
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
    """
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .ascendc import AscCompressedSparseInnerAttention

    # ``sharding_config`` is copied from the source by ``derive``; the
    # sharding pass already ran before the overrides (update_from_config),
    # so the derived config keeps the sharding-pass values as-is.
    result = derive(
        cfg,
        AscCompressedSparseInnerAttention.Config,
        indexer_loss_coeff=indexer_loss_coeff,
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )
    # The AscendC core's own ``_indexer_loss_acc`` accumulator buffer needs its
    # placement declared (a replicated scalar) for the state-distribution
    # pass.
    sharding_config = result.sharding_config
    assert sharding_config is not None, "the asc override requires the sharding config"
    sharding_config.state_shardings["_indexer_loss_acc"] = dense_param_placement(tp=spmd.R)
    return result


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description="Use PyPTO LI/LIG with AscendC TND SMLA/SMLAG",
)
def pypto(
    cfg: CompressedSparseInnerAttention.Config,
    indexer_loss_coeff: float = 1.0,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_num_heads: int = 64,
    index_head_dim: int = 128,
) -> "PyPTOCompressedSparseInnerAttention.Config":
    """PyPTO LI/LIG with the AscendC SMLA/SMLAG kernels and their records.

    It reuses the AscendC record tier, so it states the same geometry, with the same released
    defaults.
    """
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .pypto import PyPTOCompressedSparseInnerAttention

    return derive(
        cfg,
        PyPTOCompressedSparseInnerAttention.Config,
        indexer_loss_coeff=indexer_loss_coeff,
        num_heads=num_heads,
        head_dim=head_dim,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )


@override(
    target=LightningIndexer.Config,
    exact=True,
    description="Use PyPTO LightningIndexer",
)
def pypto_li(cfg: LightningIndexer.Config):
    from .pypto import PyPTOLightningIndexer

    return derive(cfg, PyPTOLightningIndexer.Config)
