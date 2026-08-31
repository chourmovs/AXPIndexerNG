"""Authoritative retrieval and evidence classification for local RAG."""
from __future__ import annotations

import time
from dataclasses import dataclass

from axp_core.identifiers import extract_identifiers


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


def rank_documents(hits):
    grouped = {}
    for hit in hits:
        grouped.setdefault(int(hit["document_id"]), []).append(hit)
    ranked = []
    for document_id, rows in grouped.items():
        rows.sort(key=lambda row: (-float(row.get("evidence_score", row.get("relevance_score")) or 0),
                                   int(row.get("chunk_id") or 0)))
        scores = [float(row.get("evidence_score", row.get("relevance_score")) or 0) for row in rows[:3]]
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
            "document_score": scores[0] + .20*scores[1] + .10*scores[2] + .05*min(strong, 3),
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
