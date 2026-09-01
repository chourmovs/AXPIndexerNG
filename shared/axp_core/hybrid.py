import time
import re
from dataclasses import dataclass
from pathlib import Path

from .fts import TOKEN_RE
from .fts import search as lexical_search
from .identifiers import extract_identifiers, normalize_identifier
from .vectors import search as vector_search

QUERY_STOPWORDS = {
    "avec", "dans", "des", "est", "les", "par", "pour", "que", "quel", "quelle", "quels",
    "quelles", "qui", "sont", "sur", "trouve", "donne", "une", "un", "de", "du", "la", "le",
    "and", "for", "from", "the", "this", "with", "what", "which", "where", "is", "are", "of",
    "a", "an", "in", "at", "give", "find",
}


@dataclass(frozen=True)
class SearchConfig:
    lexical_candidates: int = 100
    vector_candidates: int = 100
    rerank_candidates: int = 30
    rrf_k: int = 60
    max_chunks_per_document: int = 3
    vector_warning_threshold: int = 100_000
    min_vector_similarity: float = 0.35
    min_lexical_coverage: float = 0.5

    def __post_init__(self):
        if not 1 <= self.lexical_candidates <= 500 or not 1 <= self.vector_candidates <= 500:
            raise ValueError("candidate depths must be between 1 and 500")
        if not 10 <= self.rerank_candidates <= 100:
            raise ValueError("rerank_candidates must be between 10 and 100")
        if not -1.0 <= self.min_vector_similarity <= 1.0:
            raise ValueError("min_vector_similarity must be between -1 and 1")
        if not 0.0 <= self.min_lexical_coverage <= 1.0:
            raise ValueError("min_lexical_coverage must be between 0 and 1")


def diversify(rows, limit, maximum=3):
    """Apply a stable document cap without disturbing global relevance order."""
    counts = {}
    result = []
    for row in rows:
        doc = row["document_id"]
        if counts.get(doc, 0) >= maximum:
            continue
        result.append(row)
        counts[doc] = counts.get(doc, 0) + 1
        if len(result) >= limit:
            break
    return result


def _meaningful_terms(value):
    terms = set()
    for match in TOKEN_RE.finditer(value or ""):
        token = match.group(0).casefold()
        variants = (token, *re.split(r"[-._/]", token))
        terms.update(part for part in variants if len(part) >= 3 and part not in QUERY_STOPWORDS)
    return terms


def _coverage(query_terms, value):
    return len(query_terms & _meaningful_terms(value)) / len(query_terms) if query_terms else 0.0


def _relevance(item, query_terms, config):
    content = " ".join(str(item.get(field) or "") for field in ("snippet", "identifiers"))
    content_coverage = _coverage(query_terms, content)
    title_coverage = _coverage(query_terms, item.get("title"))
    filename_coverage = _coverage(query_terms, item.get("filename"))
    heading_coverage = _coverage(query_terms, item.get("heading"))
    aggregate = " ".join(str(item.get(field) or "") for field in
                         ("title", "filename", "heading", "snippet", "identifiers"))
    coverage = _coverage(query_terms, aggregate)
    distance = item.get("vector_distance")
    similarity = None if distance is None else max(-1.0, min(1.0, 1.0 - float(distance)))
    meaningful_exact = bool(item["exact_identifier_match"] or item["exact_phrase_match"])
    accepted = bool(
        (meaningful_exact and (similarity or 0) >= config.min_vector_similarity)
        or (similarity is not None and similarity >= config.min_vector_similarity)
        or (item.get("lexical_rank") is not None and content_coverage >= config.min_lexical_coverage)
    )
    v = max(0.0, min(1.0, similarity or 0.0))
    convergence = min(v, content_coverage)
    # Passage relevance is deliberately content-only.  Metadata remains useful to
    # the separate RAG document ranker, but must not make every chunk in a well
    # named document look like equally good evidence.
    passage_score = (0.58 * v + 0.24 * content_coverage + 0.18 * convergence)
    score = (0.50 * v + 0.20 * content_coverage + 0.15 * convergence + 0.10 * title_coverage
             + 0.03 * filename_coverage + 0.02 * heading_coverage)
    if accepted:
        passage_score += 0.10 * item["exact_identifier_match"] + 0.08 * item["exact_phrase_match"]
        score += 0.10 * item["exact_identifier_match"] + 0.08 * item["exact_phrase_match"]
        score += 0.05 * item["exact_filename_match"]
    item["vector_similarity"] = similarity
    item["lexical_coverage"] = float(coverage)
    item["content_lexical_coverage"] = float(content_coverage)
    item["title_coverage"] = float(title_coverage)
    item["filename_coverage"] = float(filename_coverage)
    item["heading_coverage"] = float(heading_coverage)
    item["convergence_score"] = float(convergence)
    item["passage_score"] = float(max(0.0, min(1.0, passage_score)))
    item["evidence_score"] = float(max(0.0, min(1.0, score)))
    item["relevance_score"] = item["evidence_score"]
    item["meaningful_exact_priority"] = int(meaningful_exact and accepted)
    return accepted


def search(con, query, query_vector, limit=20, rrf_k=60, *, config=None, profile="hybrid", reranker=None, explain=False):
    started = time.perf_counter()
    config = config or SearchConfig(rrf_k=rrf_k)
    timings = {"query_embedding_ms": 0.0}
    t = time.perf_counter()
    lexical = [] if profile == "fast" else lexical_search(con, query, config.lexical_candidates)
    timings["fts_retrieval_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    vector = vector_search(con, query_vector, config.vector_candidates)
    timings["vector_retrieval_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    merged = {}
    for kind, rows in (("lexical", lexical), ("vector", vector)):
        for rank, row in enumerate(rows, 1):
            item = merged.setdefault(row["chunk_id"], dict(row))
            item.setdefault("lexical_rank", None)
            item.setdefault("vector_rank", None)
            item.setdefault("bm25_score", None)
            item.setdefault("vector_distance", None)
            item.setdefault("rrf_score", 0.0)
            item[f"{kind}_rank"] = rank
            item["rrf_score"] += 1 / (config.rrf_k + rank)
            item.update({k: v for k, v in row.items() if v is not None})
    query_ids = {x for x, _ in extract_identifiers(query)}
    quoted = [part for i, part in enumerate(query.split('"')) if i % 2 and part.strip()]
    for item in merged.values():
        item_ids = {normalize_identifier(x) for x in item.get("identifiers", "").split()}
        stem = Path(item.get("filename") or item["path"]).stem.casefold()
        item["exact_identifier_match"] = bool(query_ids & item_ids)
        item["exact_filename_match"] = bool(query.strip() and query.strip().casefold() == stem)
        item["exact_phrase_match"] = any(x.casefold() in item["snippet"].casefold() for x in quoted)
        item["exact_priority"] = sum(
            (item["exact_identifier_match"], item["exact_filename_match"], item["exact_phrase_match"])
        )
    ranked = sorted(merged.values(), key=lambda x: (-x["exact_priority"], -x["rrf_score"], x["chunk_id"]))
    for rank, item in enumerate(ranked, 1):
        item["rrf_rank"] = rank
    timings["rrf_ms"] = (time.perf_counter() - t) * 1000
    timings["rerank_ms"] = 0.0
    if profile == "quality":
        if reranker is None:
            raise RuntimeError("Quality reranker model is not provisioned.")
        t = time.perf_counter()
        head = ranked[: config.rerank_candidates]
        scores = reranker.score(query, head)
        for item, score in zip(head, scores):
            item["reranker_score"] = float(score)
        head.sort(key=lambda x: (-x["exact_priority"], -x["reranker_score"], x["rrf_rank"]))
        ranked = head + ranked[config.rerank_candidates :]
        for rank, item in enumerate(head, 1):
            item["rerank_rank"] = rank
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    for item in ranked:
        item.setdefault("reranker_score", None)
        item.setdefault("rerank_rank", None)
    unfiltered_count = len(ranked)
    query_terms = _meaningful_terms(query)
    ranked = [item for item in ranked if _relevance(item, query_terms, config)]
    if profile == "quality":
        ranked.sort(key=lambda x: (-x["meaningful_exact_priority"],
                                   -(x["reranker_score"] if x["reranker_score"] is not None else float("-inf")),
                                   -x["evidence_score"], -x["convergence_score"], -x["rrf_score"], x["chunk_id"]))
    else:
        # Preserve the PR40 hybrid ordering exactly; reranking is quality-only.
        ranked.sort(key=lambda x: (-x["meaningful_exact_priority"], -x["passage_score"],
                                   -x["convergence_score"], -x["rrf_score"], x["chunk_id"]))
    final = diversify(ranked, limit, config.max_chunks_per_document)
    for rank, item in enumerate(final, 1):
        item["final_rank"] = rank
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    timings["fusion_ms"] = timings.pop("rrf_ms")
    timings["passage_scoring_ms"] = max(0.0, timings["total_ms"] - sum(
        timings.get(key, 0.0) for key in ("fts_retrieval_ms", "vector_retrieval_ms", "fusion_ms", "rerank_ms")))
    diagnostics = {
        "timings": timings,
        "candidate_counts": {
            "lexical": len(lexical), "vector": len(vector), "union": len(merged),
            "relevant": len(ranked), "filtered_out": unfiltered_count - len(ranked),
        },
        "relevance_thresholds": {
            "min_vector_similarity": config.min_vector_similarity,
            "min_lexical_coverage": config.min_lexical_coverage,
        },
        "total_chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "total_vectors": con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0],
    }
    return {"results": final, **diagnostics} if explain else final
