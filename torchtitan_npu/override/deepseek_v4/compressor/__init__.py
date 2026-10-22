"""DeepSeek-V4 compressor overrides."""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.compressor import CompressorImplementation

if TYPE_CHECKING:
    from .ascendc import AscCompressor


@override(
    target=CompressorImplementation.Config,
    exact=True,
    description="Use the CANN fused DeepSeek-V4 compressor",
)
def asc(cfg: CompressorImplementation.Config) -> "AscCompressor.Config":
    from .ascendc import AscCompressor

    return derive(cfg, AscCompressor.Config)
