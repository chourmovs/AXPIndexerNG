import logging
import time

from axp_core.hybrid import SearchConfig
from axp_core.hybrid import search as hybrid_search
from axp_core.metadata import validate_index_signature

from .rag.retrieval import (classify_passages, classify_query_evidence_intent,
                            rank_documents, retrieve_document_passages)
from .rag.spiral import resolve_identity_documents

LOGGER = logging.getLogger("axp_client")
METADATA_IDENTITY_CANDIDATES = 8


def _metadata_identity_rows(con, document_ids):
    """Materialize one bounded presentation candidate without requiring a body hit."""
    ids = list(dict.fromkeys(int(value) for value in document_ids))
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(f"""SELECT d.id document_id,c.id chunk_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,d.source_id,
 s.label source_label,s.path source_path,c.text snippet,c.identifiers
 FROM documents d JOIN sources s ON s.id=d.source_id
 LEFT JOIN chunks c ON c.id=(SELECT min(c0.id) FROM chunks c0 WHERE c0.document_id=d.id)
 WHERE d.id IN ({placeholders})""", ids).fetchall()
    by_id = {int(row["document_id"]): dict(row) for row in rows}
    materialized = []
    for document_id in ids:
        row = by_id.get(document_id)
        if row is None:
            continue
        row["chunk_id"] = int(row["chunk_id"]) if row["chunk_id"] is not None else -document_id
        row["chunk_no"] = int(row["chunk_no"] or 0)
        row["snippet"] = row.get("snippet") or "Content extraction unavailable"
        row["identifiers"] = row.get("identifiers") or ""
        row["metadata_candidate"] = True
        materialized.append(row)
    return materialized


def raw_search(con, embedder, query, limit=20, *, profile="hybrid", explain=False,
               reranker=None, config=None, query_vector=None):
    """Return the unpresented hybrid candidate pool (the RAG retrieval boundary)."""
    validate_index_signature(con, embedder.model_id, embedder.dimension,
                             getattr(embedder, "distance_metric", "cosine"))
    started = time.perf_counter()
    vector = query_vector if query_vector is not None else embedder.embed_query(query)
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
    identity_started = time.perf_counter()
    identity_ids = (resolve_identity_documents(con, query, limit=METADATA_IDENTITY_CANDIDATES)
                    if con is not None else [])
    metadata_rows = _metadata_identity_rows(con, identity_ids) if con is not None else []
    metadata_identity_ms = (time.perf_counter() - identity_started) * 1000
    existing_documents = {int(row["document_id"]) for row in result["results"]}
    result["results"].extend(row for row in metadata_rows
                             if int(row["document_id"]) not in existing_documents)
    metadata_ids = set(identity_ids)
    for row in result["results"]:
        row["metadata_candidate"] = int(row["document_id"]) in metadata_ids
    raw_candidate_count = len(result["results"])
    presentation_started = time.perf_counter()
    intent = classify_query_evidence_intent(query)
    classify_passages(result["results"], intent)
    ranking_started = time.perf_counter()
    ranked_documents = rank_documents(result["results"], intent=intent, query=query)
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
        ranked = [row for row in ranked if row["evidence_tier"] != "TOPICAL_ONLY"
                  or row.get("ingestion_mode") == "metadata"]
        ranked.sort(key=lambda row: ({"DIRECT_ANSWER": 0, "STRONG_SUPPORT": 1, "TOPICAL_ONLY": 2}[row["evidence_tier"]],
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
            "complete_query_match", "filename_identity_match", "title_identity_match",
            "metadata_identity_coverage", "collection_coverage", "identity_terms",
            "collection_terms", "document_identity_priority",
        )})
        representative["metadata_candidate"] = document["document_id"] in metadata_ids
        representatives.append(representative)
    result["results"] = representatives[:limit]
    for rank, row in enumerate(result["results"], 1):
        row["final_rank"] = rank
    result["timings"].update(drilldown.timings)
    result["timings"]["document_ranking_ms"] = document_ranking_ms
    result["timings"]["metadata_identity_ms"] = metadata_identity_ms
    result["metadata_identity_candidates"] = len(identity_ids)
    result["query_intent"] = intent.kind
    result["identity_terms"] = sorted(intent.identity_terms)
    result["target_terms"] = sorted(intent.target_terms)
    for document in ranked_documents[:min(5, len(ranked_documents))]:
        LOGGER.info("Search document rank document_id=%s filename=%s filename_identity=%s "
                    "title_identity=%s identity_coverage=%.2f collection_coverage=%.2f document_score=%.3f",
                    document["document_id"], document["filename"], document["filename_identity_match"],
                    document["title_identity_match"], document["metadata_identity_coverage"],
                    document["collection_coverage"], document["document_score"])
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
