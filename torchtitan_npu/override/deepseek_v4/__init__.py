"""DeepSeek-V4 AscendC overrides.

Importing this package registers the ``compressor``, ``lightning_indexer`` and
``sparse_attn`` override factories. Fused implementations are loaded only when
selected.
"""

from . import compressor, lightning_indexer, sparse_attn  # noqa: F401
