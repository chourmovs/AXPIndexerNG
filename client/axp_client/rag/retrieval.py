"""Authoritative retrieval and evidence classification for local RAG."""
from __future__ import annotations

import time
import re
from dataclasses import dataclass

from axp_core.fts import search_documents as lexical_search_documents
from axp_core.fts import TOKEN_RE
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance, fold_search_text
from axp_core.identifiers import extract_identifiers
from axp_core.vectors import search_documents as vector_search_documents

SCALAR_TARGETS = ("density", "relative density", "boiling point", "flash point", "molecular weight",
                  "molar mass", "viscosity", "melting point", "ph", "concentration", "assay", "titre")
_VALUE_RE = re.compile(r"(?<![A-Za-z])(?:[<>≤≥~=]\s*)?[+-]?\d+(?:[.,]\d+)?(?:\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?)?(?:\s*[eE][+-]?\d+)?\s*(?:%|°\s*[CFK]|g\s*/\s*(?:cm[³3]|m[lL]|mol)|kg\s*/\s*(?:m[³3]|[lL])|mPa\.s|cP|kPa|Pa|bar|mol\s*/\s*[lL])?", re.I)
_ADMIN_NUMBER_RE = re.compile(r"\b(?:batch|date|year|page|revision)\s*[:#]?\s*\d+", re.I)


@dataclass(frozen=True)
class QueryEvidenceIntent:
    kind: str
    identity_terms: frozenset[str]
    target_terms: frozenset[str]


def classify_query_evidence_intent(query):
    """Conservatively split scalar-property targets from material identity."""
    folded = query.casefold()
    targets = {target for target in SCALAR_TARGETS if re.search(rf"\b{re.escape(target)}\b", folded)}
    meaningful = _meaningful_terms(query)
    target_words = {word for target in targets for word in target.split()}
    identity = (meaningful - target_words) | {match.group().replace(" ", "") for match in
                                              re.finditer(r"\b\d+(?:[.,]\d+)?\s*%", query)}
    return QueryEvidenceIntent("scalar_fact" if targets else "general_semantic",
                               frozenset(identity), frozenset(targets))


def document_identity_strength(document_name, identity_terms):
    terms = _meaningful_terms(document_name) | {match.group().replace(" ", "") for match in
                                                re.finditer(r"\b\d+(?:[.,]\d+)?\s*%", document_name)}
    wanted = set(identity_terms)
    def satisfied(term):
        components = {part for part in re.split(r"[-._/]", term) if len(part) >= 3}
        return term in terms or bool(components and components <= terms)
    matched = {term for term in wanted if satisfied(term)}
    if not wanted or not matched:
        return "none"
    coverage = len(matched) / len(wanted)
    # Central product names in SDS/MSDS/FDS/specification filenames are authoritative identity.
    marker = bool(terms & {"sds", "msds", "fds", "specification", "spec"})
    if coverage == 1:
        return "exact"
    if marker and coverage >= .5:
        return "strong"
    return "weak"


def detect_scalar_evidence(text, target_terms, *, proximity=220):
    """Detect a locally related property/value shape, excluding administrative numbers."""
    folded = (text or "").casefold()
    target_matches = [match for target in target_terms
                      for match in re.finditer(rf"\b{re.escape(target)}\b", folded)]
    values = [match for match in _VALUE_RE.finditer(text or "")
              if not _ADMIN_NUMBER_RE.search((text or "")[max(0, match.start()-12):match.end()])]
    # A temperature following "at" is commonly the test condition, not a density/viscosity value.
    if set(target_terms) & {"density", "relative density", "viscosity"}:
        values = [match for match in values if not re.search(r"°\s*[CFK]", match.group(), re.I)]
    distances = [max(0, max(target.start(), value.start()) - min(target.end(), value.end()))
                 for target in target_matches for value in values]
    nearest = min(distances, default=None)
    return {"answer_shape_detected": nearest is not None and nearest <= proximity,
            "target_value_proximity": nearest}


def classify_passages(passages, intent, *, neighbors=True):
    """Annotate candidate passages with factual evidence diagnostics in-place."""
    by_position = {(int(row.get("document_id", -1)), int(row.get("chunk_no", -999))): row for row in passages}
    for row in passages:
        name = f'{row.get("title") or ""} {row.get("filename") or row.get("path") or ""}'
        strength = document_identity_strength(name, intent.identity_terms)
        text = row.get("snippet") or row.get("text") or ""
        target_present = any(re.search(rf"\b{re.escape(term)}\b", text, re.I) for term in intent.target_terms)
        shape = detect_scalar_evidence(text, intent.target_terms)
        if neighbors and target_present and not shape["answer_shape_detected"]:
            doc, no = int(row.get("document_id", -1)), int(row.get("chunk_no", -999))
            for neighbor_no in (no - 1, no + 1):
                neighbor = by_position.get((doc, neighbor_no))
                if neighbor:
                    joined = text + "\n" + (neighbor.get("snippet") or neighbor.get("text") or "")
                    candidate = detect_scalar_evidence(joined, intent.target_terms)
                    if candidate["answer_shape_detected"]:
                        shape = candidate
                        break
        row.update(query_intent=intent.kind, identity_terms=sorted(intent.identity_terms),
                   target_terms=sorted(intent.target_terms), document_identity_strength=strength, **shape)
        identity_ok = strength in {"strong", "exact"} or bool(_meaningful_terms(text) & set(intent.identity_terms))
        if intent.kind != "scalar_fact":
            row["evidence_tier"] = "STRONG_SUPPORT"
        elif identity_ok and target_present and shape["answer_shape_detected"]:
            row["evidence_tier"] = "DIRECT_ANSWER"
        elif (identity_ok and target_present and row.get("scoped_lexical_rank") is not None and
              float(row.get("vector_similarity") or 0) >= .55):
            row["evidence_tier"] = "STRONG_SUPPORT"
        else:
            row["evidence_tier"] = "TOPICAL_ONLY"
    return passages


def build_scoped_passage_query(query, document_name=""):
    """Return factual terms not already satisfied by the selected document's identity."""
    meaningful = _meaningful_terms(query)
    identity = _meaningful_terms(document_name)
    def identity_satisfied(term):
        components = {part for part in re.split(r"[-._/]", term) if len(part) >= 3}
        return term in identity or bool(components and components <= identity)
    factual = {term for term in meaningful if not identity_satisfied(term)}
    original_terms = {}
    for match in TOKEN_RE.finditer(query or ""):
        token = match.group(0)
        folded = fold_search_text(token)
        variants = [(folded, token.casefold())]
        variants.extend(zip(re.split(r"[-._/]", folded), re.split(r"[-._/]", token.casefold())))
        for variant, original in variants:
            original_terms.setdefault(variant, original)
    # Never issue an empty MATCH expression. This fallback remains stopword-filtered.
    return " ".join(original_terms.get(term, term) for term in sorted(factual or meaningful))


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
                                search_depth=0, config=None, intent=None):
    """Rank all existing chunks in a small selected-document working set."""
    started = time.perf_counter()
    ids = list(dict.fromkeys(int(value) for value in document_ids))
    config = config or SearchConfig()
    intent = intent or classify_query_evidence_intent(query)
    if not ids:
        return DocumentDrilldownResult([], {"drilldown_total_ms": 0.0, "drilldown_fts_ms": 0.0,
                                            "drilldown_vector_ms": 0.0, "drilldown_scoring_ms": 0.0,
                                            "query_embedding_ms": 0.0}, [])
    embedding_ms = 0.0
    tick = time.perf_counter()
    lexical = []
    scoped_queries = {}
    for document_id in ids:
        identity_row = con.execute("SELECT title,filename FROM documents WHERE id=?", (document_id,)).fetchone()
        identity = "" if identity_row is None else f'{identity_row["title"] or ""} {identity_row["filename"] or ""}'
        scoped_query = build_scoped_passage_query(query, identity)
        scoped_queries[document_id] = scoped_query
        try:
            lexical.extend(lexical_search_documents(con, scoped_query, [document_id]))
        except Exception as exc:
            if "chunks_fts" not in str(exc) and "sources" not in str(exc):
                raise
    fts_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    try:
        vectors = []
        if intent.kind == "scalar_fact" and intent.target_terms:
            factual_query = " ".join(sorted(intent.target_terms))
            scoped_queries = dict.fromkeys(scoped_queries, factual_query)
        for scoped_query in dict.fromkeys(scoped_queries.values()):
            scoped_ids = [doc_id for doc_id, value in scoped_queries.items() if value == scoped_query]
            vector = query_vector if scoped_query == query and query_vector is not None else None
            if vector is None and embedder is not None:
                embed_started = time.perf_counter()
                vector = embedder.embed_query(scoped_query)
                embedding_ms += (time.perf_counter() - embed_started) * 1000
            if vector is not None:
                vectors.extend(vector_search_documents(con, vector, scoped_ids))
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
        ranks_by_document = {}
        for candidate in candidates:
            document_id = int(candidate["document_id"])
            rank = ranks_by_document.get(document_id, 0) + 1
            ranks_by_document[document_id] = rank
            item = merged[int(candidate["chunk_id"])]
            item.update({key: value for key, value in candidate.items() if value is not None})
            item[f"{kind}_rank"] = rank
            if kind == "lexical":
                item["scoped_lexical_rank"] = rank
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
        item.setdefault("scoped_lexical_rank", None)
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
    classify_passages(list(merged.values()), intent)
    passages = []
    for document_rank, document_id in enumerate(ids, 1):
        ranked = [item for item in merged.values() if int(item["document_id"]) == document_id]
        # Scoped factual lexical evidence is a bounded drill-down-only priority, not a global score change.
        ranked.sort(key=lambda item: (-{"DIRECT_ANSWER": 2, "STRONG_SUPPORT": 1,
                                       "TOPICAL_ONLY": 0}.get(item.get("evidence_tier"), 0),
                                      -int(bool(item.get("exact_phrase_match"))),
                                      -int(item.get("scoped_lexical_rank") is not None),
                                      item.get("scoped_lexical_rank") or 10**9,
                                      -item["passage_score"],
                                      -(item.get("vector_similarity") or 0), item["chunk_id"]))
        for passage_rank, item in enumerate(ranked, 1):
            item["document_rank"] = document_rank
            item["passage_rank"] = passage_rank
            item["scoped_passage_query"] = scoped_queries[document_id]
        passages.extend(ranked)
    scoring_ms = (time.perf_counter() - tick) * 1000
    diagnostics = []
    for document_rank, document_id in enumerate(ids, 1):
        ranked = [item for item in passages if int(item["document_id"]) == document_id]
        best = ranked[0] if ranked else {}
        diagnostics.append({"document_rank": document_rank, "document_id": document_id, "filename": best.get("filename"),
            "scoped_passage_query": scoped_queries[document_id],
            "chunks_examined": len(ranked), "matching_chunks": sum(row.get("lexical_rank") is not None for row in ranked),
            "best_chunk_no": best.get("chunk_no"), "best_page_no": best.get("page_no"),
            "best_passage_score": best.get("passage_score", 0.0),
            "scoped_lexical_rank": best.get("scoped_lexical_rank"),
            "best_vector_similarity": best.get("vector_similarity"),
            "best_content_coverage": best.get("content_lexical_coverage", 0.0)})
    timings = {"drilldown_total_ms": (time.perf_counter() - started) * 1000,
               "drilldown_fts_ms": fts_ms, "drilldown_vector_ms": vector_ms,
               "drilldown_scoring_ms": scoring_ms, "query_embedding_ms": embedding_ms}
    return DocumentDrilldownResult(passages, timings, diagnostics)


def rank_documents(hits, *, intent=None):
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
        identity_strength = first.get("document_identity_strength", "none")
        identity_bonus = {"none": 0, "weak": .01, "strong": .14, "exact": .22}[identity_strength]
        query_terms = set(intent.identity_terms) if intent and intent.kind == "general_semantic" else set()
        def coverage(fields):
            if not query_terms:
                return 0.0
            terms = _meaningful_terms(" ".join(str(row.get(field) or "") for field in fields for row in rows))
            return len(query_terms & terms) / len(query_terms)
        document_query_coverage = coverage(("title", "filename", "heading", "snippet", "text", "identifiers"))
        document_metadata_coverage = coverage(("title", "filename", "heading"))
        complete_query_match = bool(2 <= len(query_terms) <= 4 and document_query_coverage == 1.0
                                    and document_metadata_coverage > 0)
        ranked.append({"document_id": document_id, "filename": first.get("filename"), "title": first.get("title"),
            "best_evidence_score": scores[0], "second_evidence_score": scores[1], "third_evidence_score": scores[2],
            "strong_hit_count": strong, "title_coverage": max(float(r.get("title_coverage") or 0) for r in rows),
            "exact_identifier_present": any(r.get("exact_content_identifier_match") for r in rows),
            "exact_phrase_present": any(r.get("exact_content_phrase_match") for r in rows),
            "document_identity_strength": identity_strength,
            "document_query_coverage": document_query_coverage,
            "document_metadata_coverage": document_metadata_coverage,
            "complete_query_match": complete_query_match,
            # Passage density and document metadata are combined only here.
            "document_score": scores[0] + .20*scores[1] + .10*scores[2] + .05*min(strong, 3)
                              + .10*max(float(r.get("title_coverage") or 0) for r in rows)
                              + .03*max(float(r.get("filename_coverage") or 0) for r in rows) + identity_bonus,
            "ranked_hits": rows})
    if intent and intent.kind == "general_semantic" and 2 <= len(intent.identity_terms) <= 4:
        ranked.sort(key=lambda doc: (-int(doc["complete_query_match"]), -doc["document_score"],
                                     -int(doc["exact_identifier_present"]),
                                     -int(doc["exact_phrase_present"]), doc["document_id"]))
    else:
        ranked.sort(key=lambda doc: (-doc["document_score"], -int(doc["exact_identifier_present"]),
                                     -int(doc["exact_phrase_present"]), doc["document_id"]))
    return ranked


def retrieve_rag_candidates(con, embedder, question, *, search_fn, raw_search_fn=None,
                            limit=24, search_config=None):
    """Search once, then classify and enrich candidates without changing Search output."""
    started = time.perf_counter()
    retrieval_fn = raw_search_fn or getattr(search_fn, "raw_search", search_fn)
    found = retrieval_fn(con, embedder, question, limit=limit, profile="hybrid", explain=True,
                         config=search_config)
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
        row.setdefault("snippet", text or "")
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
    intent = classify_query_evidence_intent(question)
    classify_passages(content, intent)
    ranked_documents = rank_documents(content, intent=intent)
    return RagRetrievalResult(
        candidates=candidates,
        content_evidence=content,
        metadata_related=list(related_by_document.values()),
        timings={"retrieval_ms": (time.perf_counter() - started) * 1000},
        ranked_documents=ranked_documents,
    )
