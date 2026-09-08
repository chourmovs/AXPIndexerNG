"""Fast-path-first, centrally governed Ask retrieval."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from axp_core.document_identity import analyze_document_identity
from axp_core.fts import search_scoped
from axp_core.hybrid import SearchConfig
from .answerability import decide_answerability, is_supporting_evidence
from .retrieval import (RagRetrievalResult, classify_query_evidence_intent, rank_documents,
                        retrieve_document_passages, retrieve_rag_candidates)
from .spiral import RetrievalScope, SpiralResult, resolve_identity_documents

LOGGER = logging.getLogger("axp_client")
RAG_FAST_IDENTITY_MAX_DOCUMENTS = 3
RAG_PRIMARY_DOCUMENTS = 3
RAG_EXPANSION_MAX_NEW_DOCUMENTS = 3
RAG_PRIMARY_RETRIEVAL_BUDGET_S = 8.0
RAG_TOTAL_RETRIEVAL_BUDGET_S = 12.0

@dataclass(frozen=True)
class RetrievalPolicy:
    scope: RetrievalScope = field(default_factory=RetrievalScope)
    mode: str = "default"
    temporal_policy: str = "all_history"
    max_documents: int = RAG_PRIMARY_DOCUMENTS

class RetrievalCoordinator:
    def __init__(self, *, search_fn, primary_budget_s=RAG_PRIMARY_RETRIEVAL_BUDGET_S,
                 total_budget_s=RAG_TOTAL_RETRIEVAL_BUDGET_S):
        self.search_fn, self.primary_budget_s, self.total_budget_s = search_fn, primary_budget_s, total_budget_s

    def _identity(self, con, question, policy):
        try:
            ids = resolve_identity_documents(con, question, limit=RAG_FAST_IDENTITY_MAX_DOCUMENTS,
                                             scope=policy.scope)
        except Exception as exc:
            # Unit/legacy databases predating source and identity columns retain
            # the shared hybrid path.
            if "no such table" not in str(exc) and "no such column" not in str(exc):
                raise
            return [], None
        strong = []
        for document_id in ids:
            row = con.execute("SELECT d.filename,d.title,s.label source_label,s.path source_path FROM documents d JOIN sources s ON s.id=d.source_id WHERE d.id=?", (document_id,)).fetchone()
            match = analyze_document_identity(question, filename=row["filename"], title=row["title"], source_label=row["source_label"], source_path=row["source_path"])
            if any(match.priority): strong.append((tuple(match.priority), document_id))
        if not strong: return [], None
        best = max(priority for priority, _ in strong)
        ids = [document_id for priority, document_id in strong if priority == best][:3]
        return ids, "identity_fast_path" if len(ids) == 1 else "identity_ambiguous"

    def retrieve(self, con, embedder, question, *, policy=None, search_depth=0, limit=24, search_config=None, request_id="-"):
        policy, config, started = policy or RetrievalPolicy(), search_config or SearchConfig(), time.perf_counter()
        # Existing Skill schema/compiler remains wire-compatible; its old plan is
        # consumed as constraints, never as an execution-stage recipe.
        if hasattr(policy, "stages"):
            scoped = next((stage.scope for stage in policy.stages
                           if stage.scope.path_prefixes or stage.scope.source_ids or stage.scope.extensions),
                          RetrievalScope())
            policy = RetrievalPolicy(scope=scoped,
                                     mode="prefer" if policy.allow_global_fallback else "strict")
        tick = time.perf_counter(); vector = embedder.embed_query(question) if embedder else None
        embedding_ms = (time.perf_counter()-tick)*1000
        intent = classify_query_evidence_intent(question)
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        if "sources" not in tables or not {"path_key", "extension"} <= columns:
            result = retrieve_rag_candidates(con, embedder, question, search_fn=self.search_fn,
                                             limit=limit, search_config=config, query_vector=vector)
            decision = decide_answerability(result.content_evidence)
            total_ms = (time.perf_counter() - started) * 1000
            trace = ({"route": "hybrid_discovery", "global_hybrid_calls": 1,
                      "query_embeddings": int(vector is not None), "retrieval_total_ms": total_ms},)
            result.timings.update(retrieval_ms=total_ms, spiral_total_ms=total_ms,
                                  query_embedding_ms=embedding_ms, route="hybrid_discovery",
                                  global_hybrid_calls=1, query_embeddings=int(vector is not None),
                                  spiral_trace=trace)
            return SpiralResult(result, decision, "hybrid_discovery", trace, True,
                                embedding_ms, vector)
        ids, route = self._identity(con, question, policy); hybrid_calls = 0; related = []
        if not ids:
            route = "hybrid_discovery"
            scoped = policy.scope.path_prefixes or policy.scope.source_ids or policy.scope.extensions
            if scoped:
                rows = search_scoped(con, question, source_ids=policy.scope.source_ids, path_prefixes=policy.scope.path_prefixes, extensions=policy.scope.extensions, limit=max(24, limit))
                ids = list(dict.fromkeys(int(row["document_id"]) for row in rows))
            else:
                seed = retrieve_rag_candidates(con, embedder, question, search_fn=self.search_fn, limit=limit, search_config=config, query_vector=vector)
                hybrid_calls, related = 1, seed.metadata_related
                ids = [d["document_id"] for d in seed.ranked_documents]
        ids = list(dict.fromkeys(ids))[:RAG_PRIMARY_DOCUMENTS]
        drill = retrieve_document_passages(con, embedder, question, ids, query_vector=vector, config=config, intent=intent)
        passages = list(drill.passages)
        def make_result(): return RagRetrievalResult(passages, passages, related, {}, rank_documents(passages, intent=intent, query=question))
        result = make_result(); decision = decide_answerability(passages)
        sufficient = decision.answerable and any(is_supporting_evidence(row) for row in passages)
        primary_ms = (time.perf_counter()-started)*1000; expanded = False
        if not sufficient and policy.mode != "strict" and not search_depth and primary_ms < self.primary_budget_s*1000:
            seed = retrieve_rag_candidates(con, embedder, question, search_fn=self.search_fn, limit=max(48, limit*2), search_config=config, query_vector=vector)
            hybrid_calls += 1
            new_ids = [d["document_id"] for d in seed.ranked_documents if d["document_id"] not in ids][:3]
            if new_ids and time.perf_counter()-started < self.total_budget_s:
                passages += retrieve_document_passages(con, embedder, question, new_ids, query_vector=vector, config=config, intent=intent).passages
                related, expanded = seed.metadata_related, True; result = make_result(); decision = decide_answerability(passages)
        total_ms = (time.perf_counter()-started)*1000
        trace = ({"route": route, "primary_documents": len(ids), "drilldown_documents": len(ids), "expanded": expanded, "global_hybrid_calls": hybrid_calls, "query_embeddings": int(vector is not None), "retrieval_total_ms": total_ms},)
        result.timings.update(drill.timings); result.timings.update(retrieval_ms=total_ms, retrieval_total_ms=total_ms, spiral_total_ms=total_ms, route=route, expanded=expanded, global_hybrid_calls=hybrid_calls, query_embeddings=int(vector is not None), query_embedding_ms=embedding_ms, spiral_trace=trace)
        LOGGER.info("RAG retrieval request_id=%s route=%s documents=%s expanded=%s total_ms=%.1f", request_id, route, len(ids), expanded, total_ms)
        return SpiralResult(result, decision, route, trace, hybrid_calls > 0, embedding_ms, vector)
