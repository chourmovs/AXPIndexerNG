"""Fast-path-first, centrally governed Ask retrieval."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from contextlib import nullcontext
from axp_core.tracing import SearchCancelled, Trace, current_trace, span, traced
from axp_core.document_identity import _meaningful_terms, analyze_document_identity
from axp_core.fts import search_scoped
from axp_core.hybrid import SearchConfig
from .answerability import decide_answerability, is_supporting_evidence
from .retrieval import (RagRetrievalResult, classify_query_evidence_intent, rank_documents,
                        retrieve_document_passages, retrieve_rag_candidates, build_scoped_passage_query)
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

    @traced("RAG : routage identité")
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
            if row is None:  # A writer may delete a routed document before hydration.
                continue
            match = analyze_document_identity(question, filename=row["filename"], title=row["title"], source_label=row["source_label"], source_path=row["source_path"])
            if any(match.priority): strong.append((tuple(match.priority), document_id))
        if not strong: return [], None
        best = max(priority for priority, _ in strong)
        ids = [document_id for priority, document_id in strong if priority == best][:3]
        return ids, "identity_fast_path" if len(ids) == 1 else "identity_ambiguous"

    def retrieve(self, con, embedder, question, *, policy=None, search_depth=0, limit=24, search_config=None, request_id="-"):
        existing = current_trace()
        trace = existing or Trace(None if request_id == "-" else request_id)
        try:
            with nullcontext() if existing else trace.bind():
                result = self._retrieve(con, embedder, question, policy=policy, search_depth=search_depth,
                                        limit=limit, search_config=search_config, request_id=trace.request_id)
        except BaseException as exc:
            if not existing:
                trace.finish("cancelled" if trace.cancelled.is_set() or isinstance(exc, (SearchCancelled, KeyboardInterrupt)) else "error")
            raise
        if not existing:
            trace.finish()
        result.retrieval.timings["request_trace"] = trace.snapshot()
        return result

    @traced("Récupération RAG et budgets")
    def _retrieve(self, con, embedder, question, *, policy=None, search_depth=0, limit=24, search_config=None, request_id="-"):
        policy, config, started = policy or RetrievalPolicy(), search_config or SearchConfig(), time.perf_counter()
        # Existing Skill schema/compiler remains wire-compatible; its old plan is
        # consumed as constraints, never as an execution-stage recipe.
        if hasattr(policy, "stages"):
            scoped = next((stage.scope for stage in policy.stages
                           if stage.scope.path_prefixes or stage.scope.source_ids or stage.scope.extensions),
                          RetrievalScope())
            policy = RetrievalPolicy(scope=scoped,
                                     mode="prefer" if policy.allow_global_fallback else "strict")
        tick = time.perf_counter()
        with span("Embedding requête RAG", "embedder.embed_query") as measured:
            vector = embedder.embed_query(question) if embedder else None
            if embedder is None:
                measured.skip("no_embedder")
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
        tick = time.perf_counter()
        ids, route = self._identity(con, question, policy)
        identity_ms = (time.perf_counter() - tick) * 1000
        hybrid_calls = 0; related = []; seed = None
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
        drill = retrieve_document_passages(con, embedder, question, ids, query_vector=vector, search_depth=search_depth, config=config, intent=intent)
        passages = list(drill.passages)
        def make_result(): return RagRetrievalResult(passages, passages, related, {}, rank_documents(passages, intent=intent, query=question))
        result = make_result(); decision = decide_answerability(passages)
        def sufficient_evidence():
            return (decision.answerable and any(is_supporting_evidence(row) for row in passages)
                    and (intent.kind != "scalar_fact" or
                         any(row.get("evidence_tier") == "DIRECT_ANSWER" for row in passages)))
        initial_route = route
        primary_ms = (time.perf_counter()-started)*1000
        expansion_reason = "search_depth" if search_depth else (
            None if sufficient_evidence() else decision.reason.value)
        expanded = False; local_ms = global_ms = 0.0; local_embeddings = 0
        examined = len(passages); new_ids = []
        expansion_fts_ms = expansion_vector_ms = alternate_embedding_ms = 0.0
        # Primary drilldown already examines every passage of each identity doc.
        # Weak results therefore need a different content query, not a repeated
        # scan with the same vector. At most one alternate query per document.
        if expansion_reason and embedder and time.perf_counter()-started < self.primary_budget_s:
            tick = time.perf_counter()
            for document in drill.documents:
                if time.perf_counter()-started >= self.primary_budget_s:
                    break
                row = con.execute("SELECT filename,title FROM documents WHERE id=?",
                                  (document["document_id"],)).fetchone()
                factual = build_scoped_passage_query(question, f'{row["filename"]} {row["title"]}')
                if _meaningful_terms(factual) == _meaningful_terms(question):
                    continue
                alternate = retrieve_document_passages(con, embedder, question,
                    [document["document_id"]], config=config, intent=intent, search_depth=search_depth)
                local_embeddings += 1
                expansion_fts_ms += alternate.timings.get("drilldown_fts_ms", 0.0)
                expansion_vector_ms += alternate.timings.get("drilldown_vector_ms", 0.0)
                alternate_embedding_ms += alternate.timings.get("query_embedding_ms", 0.0)
                examined += len(alternate.passages)
                by_chunk = {row["chunk_id"]: row for row in passages}
                for row in alternate.passages:
                    old = by_chunk.get(row["chunk_id"])
                    if old is None or row.get("passage_score", 0) > old.get("passage_score", 0):
                        by_chunk[row["chunk_id"]] = row
                passages = list(by_chunk.values())
                expanded = True
            local_ms = (time.perf_counter()-tick)*1000
            result = make_result(); decision = decide_answerability(passages)
        if ((not sufficient_evidence() or search_depth) and policy.mode != "strict"
                and time.perf_counter()-started < self.total_budget_s):
            tick = time.perf_counter()
            # Reuse the discovery seed when one exists. Identity routing may
            # invoke exactly one global hybrid fallback, with the original vector.
            if seed is None:
                seed = retrieve_rag_candidates(con, embedder, question, search_fn=self.search_fn,
                    limit=max(48, limit*2), search_config=config, query_vector=vector)
                hybrid_calls += 1
            new_ids = [d["document_id"] for d in seed.ranked_documents
                       if d["document_id"] not in ids][:RAG_EXPANSION_MAX_NEW_DOCUMENTS]
            if new_ids and time.perf_counter()-started < self.total_budget_s:
                extra = retrieve_document_passages(con, embedder, question, new_ids,
                    query_vector=vector, config=config, intent=intent, search_depth=search_depth)
                expansion_fts_ms += extra.timings.get("drilldown_fts_ms", 0.0)
                expansion_vector_ms += extra.timings.get("drilldown_vector_ms", 0.0)
                examined += len(extra.passages)
                passages += extra.passages
                related, expanded = seed.metadata_related, True
                result = make_result(); decision = decide_answerability(passages)
            global_ms = (time.perf_counter()-tick)*1000
        if expanded or hybrid_calls and initial_route.startswith("identity"):
            route = "identity_expanded" if initial_route.startswith("identity") else "hybrid_expanded"
        total_ms = (time.perf_counter()-started)*1000
        trace = ({"route": route, "initial_route": initial_route, "final_route": route,
                  "expansion_reason": expansion_reason, "search_depth": search_depth,
                  "expansion_status": ("performed" if expanded else "not_needed" if not expansion_reason else
                      "budget_exhausted" if total_ms >= self.total_budget_s * 1000 else
                      "strict_scope" if policy.mode == "strict" else "no_new_candidates_or_query"),
                  "primary_documents": len(ids), "drilldown_documents": len(ids)+len(new_ids),
                  "passages_examined": examined,
                  "evidence_retained": sum(is_supporting_evidence(row) for row in passages),
                  "expanded": expanded, "global_hybrid_calls": hybrid_calls,
                  "query_embeddings": int(vector is not None)+local_embeddings,
                  "metadata_identity_ms": identity_ms, "local_expansion_ms": local_ms,
                  "hybrid_expansion_ms": global_ms, "primary_retrieval_ms": primary_ms,
                  "retrieval_total_ms": total_ms},)
        seed_timings = seed.timings if seed is not None else {}
        result.timings.update(seed_timings)
        result.timings.update(drill.timings)
        result.timings.update(
            fts_retrieval_ms=seed_timings.get("fts_retrieval_ms", 0.0) +
                drill.timings.get("drilldown_fts_ms", 0.0) + expansion_fts_ms,
            vector_retrieval_ms=seed_timings.get("vector_retrieval_ms", 0.0) +
                drill.timings.get("drilldown_vector_ms", 0.0) + expansion_vector_ms,
            alternate_embedding_ms=alternate_embedding_ms)
        result.timings.update(retrieval_ms=total_ms, spiral_total_ms=total_ms,
                              query_embedding_ms=embedding_ms, spiral_trace=trace, **trace[0])
        LOGGER.info("RAG retrieval request_id=%s trace=%s", request_id, trace[0])
        return SpiralResult(result, decision, route, trace, hybrid_calls > 0, embedding_ms, vector)
