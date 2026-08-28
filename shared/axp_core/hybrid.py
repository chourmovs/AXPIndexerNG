import time
from dataclasses import dataclass
from pathlib import Path

from .fts import search as lexical_search
from .identifiers import extract_identifiers, normalize_identifier
from .vectors import search as vector_search


@dataclass(frozen=True)
class SearchConfig:
    lexical_candidates: int = 100
    vector_candidates: int = 100
    rerank_candidates: int = 30
    rrf_k: int = 60
    max_chunks_per_document: int = 3
    vector_warning_threshold: int = 100_000

    def __post_init__(self):
        if not 1 <= self.lexical_candidates <= 500 or not 1 <= self.vector_candidates <= 500:
            raise ValueError("candidate depths must be between 1 and 500")
        if not 10 <= self.rerank_candidates <= 100:
            raise ValueError("rerank_candidates must be between 10 and 100")


def diversify(rows, limit, maximum=3):
    """Round-robin documents, then second/third hits, without losing diagnostic rows."""
    groups, order = {}, []
    for row in rows:
        doc = row["document_id"]
        if doc not in groups:
            groups[doc] = []
            order.append(doc)
        groups[doc].append(row)
    result = []
    for depth in range(maximum):
        for doc in order:
            if depth < len(groups[doc]):
                result.append(groups[doc][depth])
                if len(result) == limit:
                    return result
    return result


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
        # A small rank-only safeguard; raw retriever scores remain untouched.
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
            # Model libraries commonly return numpy scalar types.  Search results are
            # a public boundary (CLI and HTTP), so never leak model-specific values.
            item["reranker_score"] = float(score)
        # Exact identifiers are protected; otherwise MaxSim is final and RRF breaks ties.
        head.sort(key=lambda x: (-x["exact_priority"], -x["reranker_score"], x["rrf_rank"]))
        ranked = head + ranked[config.rerank_candidates :]
        for rank, item in enumerate(head, 1):
            item["rerank_rank"] = rank
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    for item in ranked:
        item.setdefault("reranker_score", None)
        item.setdefault("rerank_rank", None)
    final = diversify(ranked, limit, config.max_chunks_per_document)
    for rank, item in enumerate(final, 1):
        item["final_rank"] = rank
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    diagnostics = {
        "timings": timings,
        "candidate_counts": {"lexical": len(lexical), "vector": len(vector), "union": len(merged)},
        "total_chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "total_vectors": con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0],
    }
    return {"results": final, **diagnostics} if explain else final
