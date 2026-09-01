import time

from axp_core.hybrid import SearchConfig
from axp_core.hybrid import diversify
from axp_core.hybrid import search as hybrid_search
from axp_core.metadata import validate_index_signature

from .rag.retrieval import rank_documents, retrieve_document_passages


def search(con, embedder, query, limit=20, *, profile="hybrid", explain=False, reranker=None, config=None):
    validate_index_signature(con, embedder.model_id, embedder.dimension, getattr(embedder, "distance_metric", "cosine"))
    started = time.perf_counter()
    vector = embedder.embed_query(query)
    embedding_ms = (time.perf_counter() - started) * 1000
    active_config = config or SearchConfig()
    result = hybrid_search(
        con, query, vector, limit, config=active_config, profile=profile, reranker=reranker, explain=True
    )
    documents = rank_documents(result["results"])[:2]
    drilldown = retrieve_document_passages(con, embedder, query,
        [document["document_id"] for document in documents], query_vector=vector, config=active_config)
    # The drill-down result is already grouped by authoritative document rank.
    ranked = list(drilldown.passages)
    seen = {int(row["chunk_id"]) for row in ranked}
    ranked.extend(row for row in result["results"] if int(row["chunk_id"]) not in seen)
    result["results"] = diversify(ranked, limit, active_config.max_chunks_per_document)
    for rank, row in enumerate(result["results"], 1):
        row["final_rank"] = rank
    result["timings"].update(drilldown.timings)
    result["timings"]["query_embedding_ms"] = embedding_ms
    result["timings"]["total_ms"] += embedding_ms + drilldown.timings["drilldown_total_ms"]
    return result if explain else result["results"]
