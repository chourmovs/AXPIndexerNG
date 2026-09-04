"""Progressive, scope-aware retrieval with the legacy hybrid path as fallback."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from axp_core.fts import search_scoped
from axp_core.hybrid import SearchConfig, _meaningful_terms, fold_search_text
from axp_core.identifiers import extract_identifiers

from .answerability import decide_answerability, is_supporting_evidence
from .retrieval import (RagRetrievalResult, classify_query_evidence_intent,
                        rank_documents, retrieve_document_passages,
                        retrieve_rag_candidates)

LOGGER = logging.getLogger("axp_client")
IDENTITY_DOCUMENTS = 20
HOT_DOCUMENTS = 24
WARM_DOCUMENTS = 32
SEARCH_MORE_WARM_DOCUMENTS = 48
SCOPED_VECTOR_MAX_DOCUMENTS = 48
HOT_YEARS = 2
WARM_YEARS = 5


@dataclass(frozen=True)
class RetrievalScope:
    source_ids: tuple[int, ...] = ()
    path_prefixes: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    modified_after_ms: int | None = None
    modified_before_ms: int | None = None


@dataclass(frozen=True)
class SpiralStage:
    name: str
    strategy: str
    scope: RetrievalScope = field(default_factory=RetrievalScope)
    max_documents: int = IDENTITY_DOCUMENTS
    allow_early_stop: bool = True


@dataclass(frozen=True)
class SpiralPlan:
    stages: tuple[SpiralStage, ...]
    allow_global_fallback: bool = True


@dataclass(frozen=True)
class SpiralResult:
    retrieval: RagRetrievalResult
    decision: object
    stage: str
    trace: tuple[dict, ...]
    global_fallback_used: bool
    query_embedding_ms: float
    query_vector: object = None


def _years_ago_ms(years, now_ms=None):
    now = datetime.now(timezone.utc) if now_ms is None else datetime.fromtimestamp(now_ms / 1000, timezone.utc)
    try:
        then = now.replace(year=now.year - years)
    except ValueError:
        then = now.replace(year=now.year - years, day=28)
    return int(then.timestamp() * 1000)


def default_spiral_plan(*, search_depth=0, now_ms=None):
    if search_depth:
        return SpiralPlan((SpiralStage("warm_expanded", "scoped_lexical",
            RetrievalScope(modified_after_ms=_years_ago_ms(WARM_YEARS, now_ms)),
            SEARCH_MORE_WARM_DOCUMENTS), SpiralStage("global", "global_hybrid",
            max_documents=SEARCH_MORE_WARM_DOCUMENTS, allow_early_stop=False)))
    return SpiralPlan((
        SpiralStage("identity", "metadata_routed", max_documents=IDENTITY_DOCUMENTS),
        SpiralStage("hot", "scoped_lexical",
                    RetrievalScope(modified_after_ms=_years_ago_ms(HOT_YEARS, now_ms)), HOT_DOCUMENTS),
        SpiralStage("warm", "scoped_lexical",
                    RetrievalScope(modified_after_ms=_years_ago_ms(WARM_YEARS, now_ms)), WARM_DOCUMENTS),
        SpiralStage("global", "global_hybrid", max_documents=WARM_DOCUMENTS, allow_early_stop=False),
    ))


def resolve_identity_documents(con, question, *, limit=IDENTITY_DOCUMENTS, scope=None):
    """Route explicit identities using metadata only and deterministic tuple ranking."""
    intent = classify_query_evidence_intent(question)
    terms = set(intent.identity_terms)
    identifiers = {value for value, _ in extract_identifiers(question)}
    scope = scope or RetrievalScope()
    clauses, values = ["1=1"], []
    if scope.source_ids:
        clauses.append(f"source_id IN ({','.join('?' for _ in scope.source_ids)})")
        values.extend(scope.source_ids)
    if scope.extensions:
        clauses.append(f"lower(extension) IN ({','.join('?' for _ in scope.extensions)})")
        values.extend(value.casefold() for value in scope.extensions)
    if scope.path_prefixes:
        clauses.append("(" + " OR ".join("lower(path_key) LIKE ? ESCAPE '\\'"
                                           for _ in scope.path_prefixes) + ")")
        values.extend(_path_prefix(value) for value in scope.path_prefixes)
    rows = con.execute("SELECT id,title,filename,path,path_key,modified_unix_ms FROM documents WHERE " +
                       " AND ".join(clauses), values).fetchall()
    ranked = []
    for row in rows:
        item = dict(row)
        metadata = " ".join(str(item.get(key) or "") for key in ("title", "filename", "path", "path_key"))
        folded = fold_search_text(metadata)
        metadata_terms = _meaningful_terms(metadata)
        stem = re.sub(r"\.[^.]+$", "", item.get("filename") or "")
        normalized_stem = fold_search_text(stem)
        id_exact = any(value.casefold() in folded for value in identifiers)
        exact_stem = bool(normalized_stem and normalized_stem == fold_search_text(question))
        covered = terms & metadata_terms
        complete = bool(terms and covered == terms)
        if not (id_exact or exact_stem or covered):
            continue
        key = (-int(id_exact), -int(exact_stem), -int(complete), -len(covered),
               -int(item.get("modified_unix_ms") or 0), int(item["id"]))
        ranked.append((key, int(item["id"])))
    return [document_id for _, document_id in sorted(ranked)[:max(0, min(limit, SCOPED_VECTOR_MAX_DOCUMENTS))]]


class SpiralRetriever:
    def __init__(self, *, search_fn):
        self.search_fn = search_fn

    def retrieve(self, con, embedder, question, *, plan=None, search_depth=0, limit=24,
                 search_config=None, request_id="-"):
        plan = plan or default_spiral_plan(search_depth=search_depth)
        config = search_config or SearchConfig()
        started = time.perf_counter()
        tick = time.perf_counter()
        query_vector = embedder.embed_query(question) if embedder is not None else None
        embedding_ms = (time.perf_counter() - tick) * 1000
        carried, trace, last = [], [], None
        intent = classify_query_evidence_intent(question)
        stages = list(plan.stages)
        if not plan.allow_global_fallback:
            stages = [stage for stage in stages if stage.strategy != "global_hybrid"]
        document_columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        if not {"path_key", "modified_unix_ms", "extension"} <= document_columns:
            # Small unit/legacy databases predate schema v4; retain their historical global behavior.
            stages = [SpiralStage("global", "global_hybrid", allow_early_stop=False)]
        if plan.allow_global_fallback and not any(stage.strategy == "global_hybrid" for stage in stages):
            stages.append(SpiralStage("global", "global_hybrid", allow_early_stop=False))
        for stage in stages:
            stage_started = time.perf_counter()
            if stage.strategy == "global_hybrid":
                LOGGER.info("RAG spiral fallback request_id=%s stage=global", request_id)
                result = retrieve_rag_candidates(con, embedder, question, search_fn=self.search_fn,
                    limit=limit, search_config=config, query_vector=query_vector)
                global_used = True
            else:
                if stage.strategy == "metadata_routed":
                    ids = resolve_identity_documents(con, question, limit=stage.max_documents, scope=stage.scope)
                else:
                    lexical = search_scoped(con, question, source_ids=stage.scope.source_ids,
                        path_prefixes=stage.scope.path_prefixes, extensions=stage.scope.extensions,
                        modified_after_ms=stage.scope.modified_after_ms,
                        modified_before_ms=stage.scope.modified_before_ms,
                        limit=max(stage.max_documents * 4, stage.max_documents))
                    ids = list(dict.fromkeys(int(row["document_id"]) for row in lexical))[:stage.max_documents]
                ids = list(dict.fromkeys(carried + ids))[:SCOPED_VECTOR_MAX_DOCUMENTS]
                carried = ids
                drill = retrieve_document_passages(con, embedder, question, ids, query_vector=query_vector,
                                                    search_depth=search_depth, config=config, intent=intent)
                content = drill.passages
                result = RagRetrievalResult(content, content, [],
                    {"retrieval_ms": (time.perf_counter() - stage_started) * 1000},
                    rank_documents(content, intent=intent))
                global_used = False
            decision = decide_answerability(result.content_evidence)
            supporting = any(is_supporting_evidence(row) for row in result.content_evidence)
            scalar_direct = intent.kind != "scalar_fact" or any(
                row.get("evidence_tier") == "DIRECT_ANSWER" for row in result.content_evidence)
            sufficient = bool(decision.answerable and supporting and result.ranked_documents and scalar_direct)
            elapsed = (time.perf_counter() - stage_started) * 1000
            trace.append({"stage": stage.name, "strategy": stage.strategy,
                          "examined_documents": len(result.ranked_documents), "answerable": sufficient,
                          "elapsed_ms": elapsed})
            LOGGER.info("RAG spiral request_id=%s stage=%s documents=%s answerable=%s elapsed_ms=%.1f",
                        request_id, stage.name, len(result.ranked_documents), sufficient, elapsed)
            last = (result, decision, stage.name, global_used)
            if sufficient and stage.allow_early_stop:
                LOGGER.info("RAG spiral stop request_id=%s stage=%s reason=sufficient_evidence", request_id, stage.name)
                break
            if stage.strategy == "global_hybrid":
                break
        if last is None:
            empty = RagRetrievalResult([], [], [], {"retrieval_ms": 0.0}, [])
            last = (empty, decide_answerability([]), "none", False)
        result, decision, stage_name, global_used = last
        total_ms = (time.perf_counter() - started) * 1000
        result.timings.update({"retrieval_ms": total_ms, "query_embedding_ms": embedding_ms,
            "spiral_total_ms": total_ms, "spiral_stage_count": len(trace),
            "global_fallback_used": global_used, "spiral_trace": tuple(trace),
            **{f"spiral_{row['stage']}_ms": row["elapsed_ms"] for row in trace}})
        return SpiralResult(result, decision, stage_name, tuple(trace), global_used, embedding_ms, query_vector)


def _path_prefix(value):
    normalized = str(value).replace("\\", "/").casefold().rstrip("/")
    return normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
