"""Experimental, opt-in ANN adapter. Never replaces the production SQLite index."""
from __future__ import annotations


class UsearchHNSW:
    """Thin HNSW wrapper, isolating USearch from the live retrieval stack."""

    def __init__(self, dimension, *, dtype="f32", connectivity=24,
                 expansion_add=128, expansion_search=64):
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise RuntimeError("USearch not installed: pip install -r requirements-ann-benchmark.txt") from exc
        self.index = Index(
            ndim=int(dimension), metric="cos", dtype=dtype,
            connectivity=int(connectivity), expansion_add=int(expansion_add),
            expansion_search=int(expansion_search),
        )

    def add(self, keys, vectors):
        """Keys must correspond to stable chunk rowids."""
        self.index.add(keys, vectors)

    def search(self, vector, k):
        found = self.index.search(vector, int(k))
        return [(int(key), float(distance))
                for key, distance in zip(found.keys, found.distances)]

    def save(self, path):
        self.index.save(str(path))

    @property
    def memory_usage(self):
        return int(self.index.memory_usage)

    @property
    def size(self):
        return int(self.index.size)
