"""Document embedding backends.

Query embedding deliberately remains in :mod:`axp_daemon.embeddings`; these
backends are an indexing-only acceleration boundary.
"""

from .factory import (BackendFailure, EmbeddingBackend, IntelEmbeddingUnavailable,
                      create_indexing_embedder)

__all__ = ["BackendFailure", "EmbeddingBackend", "IntelEmbeddingUnavailable", "create_indexing_embedder"]
