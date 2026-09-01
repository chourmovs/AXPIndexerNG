"""Authoritative retrieval and evidence classification for local RAG."""
from __future__ import annotations

import time
from dataclasses import dataclass

from axp_core.fts import search_documents as lexical_search_documents
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance
from axp_core.identifiers import extract_identifiers
from axp_core.vectors import search_documents as vector_search_documents


@dataclass(frozen=True)
class RagRetrievalResult:
    candidates: list[dict]
    content_evidence: list[dict]
    metadata_related: list[dict]
    timings: dict
    ranked_documents: list[dict]

    @property
    def diagnostics(self):
        return {
            "retrieval_candidates": len(self.candidates),
            "content_candidates": len(self.content_evidence),
            "metadata_candidates": len(self.candidates) - len(self.content_evidence),
            "ranked_documents": self.ranked_documents,
        }


@dataclass(frozen=True)
class DocumentDrilldownResult:
    passages: list[dict]
    timings: dict
    documents: list[dict]


def retrieve_document_passages(con, embedder, query, document_ids, *, query_vector=None,
                                search_depth=0, config=None):
    """Rank all existing chunks in a small selected-document working set."""
    started = time.perf_counter()
    ids = sorted({int(value) for value in document_ids})
    config = config or SearchConfig()
    if not ids:
        return DocumentDrilldownResult([], {"drilldown_total_ms": 0.0, "drilldown_fts_ms": 0.0,
                                            "drilldown_vector_ms": 0.0, "drilldown_scoring_ms": 0.0,
                                            "query_embedding_ms": 0.0}, [])
    embedding_ms = 0.0
    if query_vector is None and embedder is not None:
        tick = time.perf_counter()
        query_vector = embedder.embed_query(query)
        embedding_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    try:
        lexical = lexical_search_documents(con, query, ids)
    except Exception as exc:
        if "chunks_fts" not in str(exc) and "sources" not in str(exc):
            raise
        lexical = []
    fts_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    try:
        vectors = vector_search_documents(con, query_vector, ids) if query_vector is not None else []
    except Exception as exc:  # sqlite test databases and legacy vector-less databases remain readable.
        if "chunk_vectors" not in str(exc) and "sources" not in str(exc):
            raise
        vectors = []
    vector_ms = (time.perf_counter() - tick) * 1000
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(f"""SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,
 c.text snippet FROM chunks c JOIN documents d ON d.id=c.document_id
 WHERE c.document_id IN ({placeholders}) ORDER BY c.document_id,c.chunk_no""", ids).fetchall()
    merged = {int(row["chunk_id"]): dict(row) for row in rows}
    for item in merged.values():
        item.setdefault("identifiers", "")
        item.setdefault("source_id", None)
    for kind, candidates in (("lexical", lexical), ("vector", vectors)):
        for rank, candidate in enumerate(candidates, 1):
            item = merged[int(candidate["chunk_id"])]
            item.update({key: value for key, value in candidate.items() if value is not None})
            item[f"{kind}_rank"] = rank
    tick = time.perf_counter()
    query_ids = {value for value, _ in extract_identifiers(query)}
    quoted = [part for index, part in enumerate(query.split('"')) if index % 2 and part.strip()]
    query_terms = _meaningful_terms(query)
    document_terms = {}
    for item in merged.values():
        document_terms.setdefault(int(item["document_id"]), _meaningful_terms(
            f'{item.get("title") or ""} {item.get("filename") or ""}'))
    for item in merged.values():
        item.setdefault("lexical_rank", None)
        item.setdefault("vector_rank", None)
        item.setdefault("vector_distance", None)
        content_ids = {value for value, _ in extract_identifiers(item.get("snippet") or "")}
        item["exact_identifier_match"] = bool(query_ids & content_ids)
        item["exact_phrase_match"] = any(value.casefold() in (item.get("snippet") or "").casefold()
                                                 for value in quoted)
        item["exact_filename_match"] = False
        # Terms strongly satisfied by document identity no longer penalize its individual passages.
        passage_terms = query_terms - (query_terms & document_terms[int(item["document_id"])])
        _relevance(item, passage_terms or query_terms, config)
    passages = sorted(merged.values(), key=lambda item: (-item["passage_score"], item["chunk_id"]))
    scoring_ms = (time.perf_counter() - tick) * 1000
    diagnostics = []
    for document_id in ids:
        ranked = [item for item in passages if int(item["document_id"]) == document_id]
        best = ranked[0] if ranked else {}
        diagnostics.append({"document_id": document_id, "filename": best.get("filename"),
            "chunks_examined": len(ranked), "matching_chunks": sum(row.get("lexical_rank") is not None for row in ranked),
            "best_chunk_no": best.get("chunk_no"), "best_page_no": best.get("page_no"),
            "best_passage_score": best.get("passage_score", 0.0),
            "best_vector_similarity": best.get("vector_similarity"),
            "best_content_coverage": best.get("content_lexical_coverage", 0.0)})
    timings = {"drilldown_total_ms": (time.perf_counter() - started) * 1000,
               "drilldown_fts_ms": fts_ms, "drilldown_vector_ms": vector_ms,
               "drilldown_scoring_ms": scoring_ms, "query_embedding_ms": embedding_ms}
    return DocumentDrilldownResult(passages, timings, diagnostics)


def rank_documents(hits):
    grouped = {}
    for hit in hits:
        grouped.setdefault(int(hit["document_id"]), []).append(hit)
    ranked = []
    for document_id, rows in grouped.items():
        rows.sort(key=lambda row: (-float(row.get("passage_score", row.get("evidence_score", row.get("relevance_score"))) or 0),
                                   int(row.get("chunk_id") or 0)))
        scores = [float(row.get("passage_score", row.get("evidence_score", row.get("relevance_score"))) or 0)
                  for row in rows[:3]]
        scores += [0.0] * (3 - len(scores))
        strong = sum(bool(float(row.get("evidence_score", row.get("relevance_score")) or 0) >= .55 or
                          row.get("exact_content_identifier_match") or row.get("exact_content_phrase_match"))
                     for row in rows)
        first = rows[0]
        ranked.append({"document_id": document_id, "filename": first.get("filename"), "title": first.get("title"),
            "best_evidence_score": scores[0], "second_evidence_score": scores[1], "third_evidence_score": scores[2],
            "strong_hit_count": strong, "title_coverage": max(float(r.get("title_coverage") or 0) for r in rows),
            "exact_identifier_present": any(r.get("exact_content_identifier_match") for r in rows),
            "exact_phrase_present": any(r.get("exact_content_phrase_match") for r in rows),
            # Passage density and document metadata are combined only here.
            "document_score": scores[0] + .20*scores[1] + .10*scores[2] + .05*min(strong, 3)
                              + .10*max(float(r.get("title_coverage") or 0) for r in rows)
                              + .03*max(float(r.get("filename_coverage") or 0) for r in rows),
            "ranked_hits": rows})
    ranked.sort(key=lambda doc: (-doc["document_score"], -int(doc["exact_identifier_present"]),
                                 -int(doc["exact_phrase_present"]), doc["document_id"]))
    return ranked


def retrieve_rag_candidates(con, embedder, question, *, search_fn, limit=24, search_config=None):
    """Search once, then classify and enrich candidates without changing Search output."""
    started = time.perf_counter()
    found = search_fn(con, embedder, question, limit=limit, profile="hybrid", explain=True, config=search_config)
    candidates = [dict(row) for row in found.get("results", found)]
    document_ids = sorted({int(row["document_id"]) for row in candidates})
    documents = {}
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        documents = {
            int(row["id"]): dict(row)
            for row in con.execute(
                f"SELECT id,ingestion_mode,filename FROM documents WHERE id IN ({placeholders})",
                document_ids,
            )
        }

    content = [
        row for row in candidates
        if documents.get(int(row["document_id"]), {}).get("ingestion_mode", "content") == "content"
    ]
    chunk_ids = sorted({int(row["chunk_id"]) for row in content if row.get("chunk_id") is not None})
    chunk_text_by_id = {}
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        chunk_text_by_id = {
            int(row["id"]): row["text"]
            for row in con.execute(f"SELECT id,text FROM chunks WHERE id IN ({placeholders})", chunk_ids)
        }
    query_identifiers = {identifier for identifier, _ in extract_identifiers(question)}
    quoted = [part.strip().casefold() for index, part in enumerate(question.split('"')) if index % 2 and part.strip()]
    for row in content:
        text = chunk_text_by_id.get(int(row["chunk_id"])) if row.get("chunk_id") is not None else ""
        content_identifiers = {identifier for identifier, _ in extract_identifiers(text)}
        row["exact_content_identifier_match"] = bool(query_identifiers & content_identifiers)
        # Search's phrase signal currently uses a content snippet. Recomputing from full stored text makes
        # the RAG meaning explicit and prevents future snippet/metadata changes from weakening the boundary.
        row["exact_content_phrase_match"] = any(phrase in (text or "").casefold() for phrase in quoted)

    related_by_document = {}
    for row in candidates:
        document_id = int(row["document_id"])
        document = documents.get(document_id, {})
        if document.get("ingestion_mode") == "metadata":
            related_by_document.setdefault(
                document_id, {"document_id": document_id, "filename": document.get("filename")}
            )
    ranked_documents = rank_documents(content)
    return RagRetrievalResult(
        candidates=candidates,
        content_evidence=content,
        metadata_related=list(related_by_document.values()),
        timings={"retrieval_ms": (time.perf_counter() - started) * 1000},
        ranked_documents=ranked_documents,
    )
