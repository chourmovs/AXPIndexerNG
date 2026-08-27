import time

from axp_core.hybrid import SearchConfig, search as hybrid_search
from axp_core.metadata import validate_index_signature


def search(con, embedder, query, limit=20, *, profile="hybrid", explain=False, reranker=None, config=None):
    validate_index_signature(con, embedder.model_id, embedder.dimension, getattr(embedder, "distance_metric", "cosine"))
    started = time.perf_counter()
    vector = embedder.embed_query(query)
    embedding_ms = (time.perf_counter() - started) * 1000
    result = hybrid_search(
        con, query, vector, limit, config=config or SearchConfig(), profile=profile, reranker=reranker, explain=explain
    )
    if explain:
        result["timings"]["query_embedding_ms"] = embedding_ms
        result["timings"]["total_ms"] += embedding_ms
    return result
