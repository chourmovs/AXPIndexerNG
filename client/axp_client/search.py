import logging
import time

from axp_core.hybrid import SearchConfig
from axp_core.hybrid import search as hybrid_search
from axp_core.metadata import validate_index_signature

from .rag.retrieval import (classify_passages, classify_query_evidence_intent,
                            rank_documents, retrieve_document_passages)

LOGGER = logging.getLogger("axp_client")


def raw_search(con, embedder, query, limit=20, *, profile="hybrid", explain=False,
               reranker=None, config=None):
    """Return the unpresented hybrid candidate pool (the RAG retrieval boundary)."""
    validate_index_signature(con, embedder.model_id, embedder.dimension,
                             getattr(embedder, "distance_metric", "cosine"))
    started = time.perf_counter()
    vector = embedder.embed_query(query)
    embedding_ms = (time.perf_counter() - started) * 1000
    result = hybrid_search(con, query, vector, limit, config=config or SearchConfig(),
                           profile=profile, reranker=reranker, explain=True)
    result["timings"]["query_embedding_ms"] = embedding_ms
    result["timings"]["total_ms"] += embedding_ms
    return result if explain else result["results"]


def search(con, embedder, query, limit=20, *, profile="hybrid", explain=False, reranker=None, config=None):
    search_started = time.perf_counter()
    active_config = config or SearchConfig()
    result = raw_search(con, embedder, query, limit, profile=profile, reranker=reranker,
                        config=active_config, explain=True)
    raw_candidate_count = len(result["results"])
    presentation_started = time.perf_counter()
    intent = classify_query_evidence_intent(query)
    classify_passages(result["results"], intent)
    ranking_started = time.perf_counter()
    ranked_documents = rank_documents(result["results"], intent=intent)
    document_ranking_ms = (time.perf_counter() - ranking_started) * 1000
    documents = ranked_documents[:2]
    drilldown = retrieve_document_passages(con, embedder, query,
        [document["document_id"] for document in documents], config=active_config, intent=intent)
    # The drill-down result is already grouped by authoritative document rank.
    ranked = list(drilldown.passages)
    seen = {int(row["chunk_id"]) for row in ranked}
    ranked.extend(row for row in result["results"] if int(row["chunk_id"]) not in seen)
    classify_passages(ranked, intent)
    tier_counts = {tier: sum(row.get("evidence_tier") == tier for row in ranked)
                   for tier in ("DIRECT_ANSWER", "STRONG_SUPPORT", "TOPICAL_ONLY")}
    if intent.kind == "scalar_fact":
        ranked = [row for row in ranked if row["evidence_tier"] != "TOPICAL_ONLY"]
        ranked.sort(key=lambda row: ({"DIRECT_ANSWER": 0, "STRONG_SUPPORT": 1}[row["evidence_tier"]],
                                     -float(row.get("passage_score") or 0), row["chunk_id"]))
    # Standalone Search represents documents, unlike RAG's multi-passage evidence path.
    representatives = []
    for document in ranked_documents:
        representative = next((row for row in ranked
                               if int(row["document_id"]) == document["document_id"]), None)
        if representative is None:
            continue
        representative.update({key: document[key] for key in (
            "document_score", "document_query_coverage", "document_metadata_coverage",
            "complete_query_match",
        )})
        representatives.append(representative)
    result["results"] = representatives[:limit]
    for rank, row in enumerate(result["results"], 1):
        row["final_rank"] = rank
    result["timings"].update(drilldown.timings)
    result["timings"]["document_ranking_ms"] = document_ranking_ms
    result["query_intent"] = intent.kind
    result["identity_terms"] = sorted(intent.identity_terms)
    result["target_terms"] = sorted(intent.target_terms)
    if intent.kind == "scalar_fact":
        LOGGER.info("Search evidence query_intent=%s identity_terms=%s target_terms=%s "
                    "raw_candidates=%s direct_answer=%s strong_support=%s topical_only=%s displayed=%s",
                    intent.kind, sorted(intent.identity_terms), sorted(intent.target_terms),
                    raw_candidate_count, tier_counts["DIRECT_ANSWER"], tier_counts["STRONG_SUPPORT"],
                    tier_counts["TOPICAL_ONLY"], len(result["results"]))
    result["timings"]["presentation_ms"] = (time.perf_counter() - presentation_started) * 1000
    result["timings"]["search_total_ms"] = (time.perf_counter() - search_started) * 1000
    # Preserve the historical aggregate while adding unambiguous end-to-end naming.
    result["timings"]["total_ms"] = result["timings"]["search_total_ms"]
    return result if explain else result["results"]


# Lets dependency-injected callers discover the raw boundary without importing this module again.
search.raw_search = raw_search
