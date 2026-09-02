import logging
import re
import threading
import time
import uuid

from axp_core.hybrid import SearchConfig

from .answerability import decide_answerability, is_supporting_evidence
from .citations import classify_citations
from .context import build_context
from .depth import depth_policy
from .llama_cpp_backend import GenerationCancelled, GenerationConfig
from .latency import PERFORMANCE_ESTIMATES, POLICY, estimate_prefill_seconds
from .operations import NativeOperationSupervisor
from .prompts import SYSTEM_PROMPT, system_prompt, user_prompt
from .response_policy import classify_response_plan
from .retrieval import (classify_query_evidence_intent, rank_documents,
                        retrieve_document_passages, retrieve_rag_candidates)

LOGGER = logging.getLogger("axp_client")
RETRIEVAL_LIMIT = 24
CITATION_REPAIR_MAX_TOKENS = 64
CITATION_REPAIR_MAX_ANSWER_CHARACTERS = 800
CITATION_REPAIR_MAX_SOURCES = 3
CITATION_REPAIR_PREFILL_BUDGET_SECONDS = 2.0


def select_supporting_documents(content, maximum=2):
    """Rank only documents that contain evidence accepted by the answerability boundary."""
    supporting = [row for row in content if is_supporting_evidence(row)]
    return supporting, rank_documents(supporting)[:maximum]


class ChatBusyError(Exception):
    pass


class ChatUnavailableError(Exception):
    pass


class GenerationFailedError(Exception):
    pass


class ModelLoadFailedError(Exception):
    pass


class ContextPreparationFailedError(Exception):
    pass


class ValidationFailedError(Exception):
    pass


class RagService:
    def __init__(self, *, backend, search_fn, connect_fn, db, embedder):
        self.backend, self.search_fn, self.connect_fn = backend, search_fn, connect_fn
        self.db, self.embedder = db, embedder
        self._generation_lock = threading.Lock()
        self.operations = NativeOperationSupervisor()

    def health(self):
        return {**self.backend.health(), "generation_busy": self.operations.busy or self._generation_lock.locked()}

    def cancel_generation(self):
        cancel_load = getattr(self.backend, "request_load_cancel", None)
        if cancel_load and cancel_load(): return True
        cancel = getattr(self.backend, "request_cancel", None)
        return bool(cancel and cancel())

    def close(self):
        self.operations.close()
        if callable(getattr(self.backend, "close", None)):
            self.backend.close()

    @property
    def busy(self):
        return self.operations.busy or self._generation_lock.locked()

    def activate(self, settings, profile=None):
        if self.busy:
            raise ChatBusyError
        self.backend.activate(settings, profile)

    def run_when_idle(self, operation):
        """Serialize runtime administration against load and generation."""
        if not self._generation_lock.acquire(blocking=False):
            raise ChatBusyError
        try:
            if self.operations.busy:
                raise ChatBusyError
            return operation()
        finally:
            self._generation_lock.release()

    def retry_model(self):
        self.backend.retry_load()
        return {"status": "reset", "model_state": self.health().get("model_state", "unloaded")}

    def _run_blocking(self, operation, heartbeat, *, interval=1.0):
        return self.operations.run(operation, heartbeat, interval=interval)

    def retrieve(self, question, search_depth=0):
        config = getattr(self.backend, "config", GenerationConfig())
        policy = depth_policy(search_depth, evidence_tokens=config.max_evidence_tokens,
                              answer_tokens=config.max_answer_tokens)
        with self.connect_fn(self.db, readonly=True) as con:
            return retrieve_rag_candidates(con, self.embedder, question, search_fn=self.search_fn,
                limit=policy.retrieval_limit, search_config=SearchConfig(
                    lexical_candidates=policy.candidate_depth, vector_candidates=policy.candidate_depth))

    def evaluate_gate(self, question, *, retrieval=None, config=None):
        retrieval = retrieval or self.retrieve(question)
        return retrieval, decide_answerability(retrieval.content_evidence, config)

    @staticmethod
    def _log_gate(request_id, decision, result_count):
        signals = decision.signals
        LOGGER.info(
            "RAG request id=%s decision=%s reason=%s best_vector=%.4f best_lexical=%.4f "
            "strong_chunks=%s support_chunks=%s exact_matches=%s documents=%s results=%s",
            request_id, decision.reason.value, decision.reason.value,
            signals["best_vector_similarity"], signals["best_lexical_coverage"],
            signals["strong_chunks"], signals["support_chunks"], signals["exact_content_matches"],
            signals["documents"], result_count,
        )

    def ask(self, question, *, debug=False, retrieval=None, progress=None, search_depth=0):
        def emit(event, **details):
            if progress is not None:
                progress({"event": event, **details})

        request_id, started = uuid.uuid4().hex[:12], time.perf_counter()
        model_load_ms = 0.0
        generation_config = getattr(self.backend, "config", GenerationConfig())
        intent = classify_query_evidence_intent(question)
        response_plan = classify_response_plan(question, intent)
        reasoning_enabled = generation_config.reasoning_enabled
        depth = depth_policy(search_depth, evidence_tokens=generation_config.max_evidence_tokens,
                             answer_tokens=generation_config.max_answer_tokens)
        requested_answer_tokens = (response_plan.answer_tokens if reasoning_enabled
                                   else depth.target_answer_tokens)
        active_system_prompt = system_prompt(reasoning_enabled)
        response_instruction = response_plan.instruction if reasoning_enabled else None
        if self.operations.busy or not self._generation_lock.acquire(blocking=False):
            raise ChatBusyError
        self._generation_lock.release()
        if not self.health().get("available"):
            raise ChatUnavailableError
        emit("retrieval_started")
        with self.connect_fn(self.db, readonly=True) as con:
            retrieval = retrieval or retrieve_rag_candidates(
                con, self.embedder, question, search_fn=self.search_fn, limit=depth.retrieval_limit,
                search_config=SearchConfig(lexical_candidates=depth.candidate_depth,
                                           vector_candidates=depth.candidate_depth)
            )
            candidates, all_content = retrieval.candidates, retrieval.content_evidence
            LOGGER.info("RAG raw retrieval request_id=%s raw_candidates=%s query_intent=%s",
                        request_id, len(candidates), intent.kind)
            selected_limit = 2 if search_depth == 0 else 3
            supporting_content, eligible_documents = select_supporting_documents(all_content, selected_limit)
            if intent.kind == "scalar_fact":
                # Recall first: document identity may lead us to an answer that was outside the seed snippet.
                eligible_documents = retrieval.ranked_documents[:selected_limit]
            decision = decide_answerability(all_content)
            if decision.answerable and supporting_content and not eligible_documents:
                LOGGER.error("RAG invariant violation request_id=%s answerable=true supporting_chunks=%s "
                             "ranked_documents=%s", request_id, len(supporting_content),
                             len(retrieval.ranked_documents))
                supporting_ids = {int(row["document_id"]) for row in supporting_content}
                fallback = next((doc for doc in retrieval.ranked_documents
                                 if int(doc["document_id"]) in supporting_ids), None)
                if fallback is not None:
                    eligible_documents = [fallback]
            ranking_finished = time.perf_counter()
            eligible_documents = eligible_documents[:selected_limit]
            drilldown = retrieve_document_passages(
                con, self.embedder, question, [doc["document_id"] for doc in eligible_documents],
                search_depth=search_depth,
                config=SearchConfig(lexical_candidates=depth.candidate_depth,
                                    vector_candidates=depth.candidate_depth),
                intent=intent,
            ) if eligible_documents else None
            content = (drilldown.passages if drilldown else
                       [hit for doc in eligible_documents for hit in doc["ranked_hits"]])
            if intent.kind == "scalar_fact":
                content = [row for row in content if row.get("evidence_tier") in
                           {"DIRECT_ANSWER", "STRONG_SUPPORT"}]
            drilldown_decision = decide_answerability(content) if content else decision
            if intent.kind == "scalar_fact":
                decision = drilldown_decision
            LOGGER.info("RAG retrieval request_id=%s search_depth=%s query_length=%s candidates=%s "
                        "relevant_chunks=%s relevant_documents=%s", request_id, search_depth, len(question),
                        len(candidates), len(all_content), len(retrieval.ranked_documents))
            LOGGER.info("RAG document ranking request_id=%s top=%s", request_id, [{key: doc[key] for key in
                ("document_id", "filename", "document_score", "best_evidence_score", "second_evidence_score",
                 "strong_hit_count", "title_coverage")} for doc in retrieval.ranked_documents[:5]])
            LOGGER.info("RAG evidence selection request_id=%s answerable=%s supporting_chunks=%s "
                        "ranked_documents=%s selected_documents=%s", request_id, decision.answerable,
                        len(supporting_content), len(retrieval.ranked_documents), [{
                            "document_id": doc["document_id"],
                            "best_evidence_score": doc["best_evidence_score"],
                            "document_score": doc["document_score"],
                            "support_chunk_count": len(doc["ranked_hits"]),
                        } for doc in eligible_documents[:2]])
            if drilldown:
                LOGGER.info("RAG drilldown request_id=%s search_depth=%s global_candidates=%s "
                            "drilldown_documents=%s drilldown_chunks_examined=%s factual_vector_query=%s "
                            "drilldown_passages_ranked=%s documents=%s", request_id, search_depth,
                            len(candidates), len(drilldown.documents),
                            sum(doc["chunks_examined"] for doc in drilldown.documents),
                            sorted({doc["scoped_passage_query"] for doc in drilldown.documents}), len(content),
                            drilldown.documents)
                drill_signals = drilldown_decision.signals
                LOGGER.info("RAG drilldown answerability request_id=%s initial_answerability=%s "
                            "drilldown_supporting_chunks=%s drilldown_best_vector=%.4f "
                            "drilldown_best_lexical=%.4f drilldown_best_passage_score=%.4f",
                            request_id, decision.answerable, drill_signals["support_chunks"],
                            drill_signals["best_vector_similarity"], drill_signals["best_lexical_coverage"],
                            max((float(row.get("passage_score") or 0) for row in content), default=0.0))
            related, retrieval_ms = retrieval.metadata_related, retrieval.timings["retrieval_ms"]
            emit("retrieval_complete", candidates=len(candidates), content_candidates=len(content))
            self._log_gate(request_id, decision, len(candidates))
            emit("gate_complete", answerable=decision.answerable)
            base = {"status": "insufficient_evidence", "answerable": False, "answer": None, "sources": [],
                    "related_documents": related, "decision": decision.public(),
                    "context": {"search_depth": search_depth}}
            if not decision.answerable:
                base["timings"] = {"retrieval_ms": retrieval_ms, "total_ms": (time.perf_counter() - started) * 1000}
                if debug:
                    base["debug"] = self._debug(retrieval, decision, None)
                return base
            if not self._generation_lock.acquire(blocking=False):
                raise ChatBusyError
            try:
                state = self.health().get("model_state")
                if state != "loaded":
                    emit("model_load_started")
                    load_start = time.perf_counter()
                    try:
                        def load_poll(elapsed):
                            snapshot = getattr(self.backend, "model_load_progress", lambda: {})()
                            if snapshot:
                                keys = ("elapsed_s", "phase", "native_lines_seen", "last_native_activity_age_s",
                                        "health_status", "gpu_offload_confirmed", "offloaded_layers", "total_layers",
                                        "slow_warning", "suspected_stall")
                                emit("model_load_progress", backend="intel_sycl",
                                     **{key: snapshot.get(key) for key in keys})
                            else: emit("model_load_heartbeat", elapsed_s=elapsed)
                        self._run_blocking(self.backend.ensure_loaded, load_poll, interval=.25)
                    except Exception as exc:
                        elapsed = time.perf_counter() - load_start
                        failure = self.health().get("failure_type", "model_load_failed")
                        if failure == "intel_gpu_model_load_cancelled":
                            emit("cancelled"); raise GenerationCancelled from exc
                        emit("model_load_failed", elapsed_s=round(elapsed, 1), error=failure)
                        LOGGER.exception("Model load failed request id=%s type=%s model=%s", request_id,
                                         type(exc).__name__, self.health().get("model_name", "configured model"))
                        raise ModelLoadFailedError from exc
                    emit("model_load_complete", elapsed_s=round(time.perf_counter() - load_start, 1))
                    model_load_ms = (time.perf_counter() - load_start) * 1000
                emit("context_preparation_started")
                context_start = time.perf_counter()
                try:
                    system_tokens = self.backend.count_tokens(active_system_prompt)
                    question_tokens = self.backend.count_tokens(
                        user_prompt(question, "", response_instruction))
                    fixed_tokens = system_tokens + question_tokens
                    effective_answer_tokens = min(requested_answer_tokens, max(0,
                        self.backend.context_window() - fixed_tokens - generation_config.safety_tokens))
                    context = build_context(con, content, token_counter=self.backend.count_tokens,
                                            context_window=self.backend.context_window(), fixed_prompt_tokens=fixed_tokens,
                                            config=__import__("axp_client.rag.context", fromlist=["ContextConfig"]).ContextConfig(
                                                max_documents=selected_limit,
                                                max_seeds_per_document=depth.seed_limit,
                                                neighbor_radius=depth.neighbor_radius,
                                                max_blocks=generation_config.max_context_blocks,
                                                answer_reserve_tokens=effective_answer_tokens,
                                                safety_reserve_tokens=generation_config.safety_tokens,
                                                max_evidence_tokens=depth.target_evidence_tokens))
                    original_context = context
                    health = self.health()
                    model_id = health.get("active_model_id") or health.get("model_name")
                    effective_device = health.get("inference_device_effective")
                    estimate = PERFORMANCE_ESTIMATES.get(model_id, effective_device)
                    prompt_tps = (estimate or {}).get("prompt_eval_tokens_per_second")
                    if prompt_tps is None and effective_device == "cpu":
                        prompt_tps = POLICY.default_cpu_prompt_tps
                    estimated = estimate_prefill_seconds(
                        fixed_tokens + (context.diagnostics["evidence_tokens"] or 0), prompt_tps)
                    reduced = False
                    original_budget = context.diagnostics["evidence_budget_tokens"]
                    # Rebuild from ranked hits at complete evidence boundaries for each rung.
                    if estimated is not None and estimated > POLICY.preferred_seconds:
                        for target in POLICY.evidence_ladder:
                            if original_budget is not None and target >= original_budget:
                                continue
                            candidate = build_context(con, content, token_counter=self.backend.count_tokens,
                                context_window=self.backend.context_window(), fixed_prompt_tokens=fixed_tokens,
                                config=__import__("axp_client.rag.context", fromlist=["ContextConfig"]).ContextConfig(
                                    max_documents=selected_limit, max_seeds_per_document=depth.seed_limit,
                                    neighbor_radius=depth.neighbor_radius,
                                    max_blocks=generation_config.max_context_blocks,
                                    answer_reserve_tokens=effective_answer_tokens,
                                    safety_reserve_tokens=generation_config.safety_tokens,
                                    max_evidence_tokens=target))
                            if not candidate.blocks:
                                continue
                            context, reduced = candidate, True
                            estimated = estimate_prefill_seconds(
                                fixed_tokens + (context.diagnostics["evidence_tokens"] or 0), prompt_tps)
                            emit("context_reduced_for_latency", evidence_budget_tokens=target,
                                 estimated_prefill_seconds=estimated)
                            if estimated is not None and estimated <= POLICY.preferred_seconds:
                                break
                except Exception as exc:
                    LOGGER.exception("Context preparation failed request id=%s type=%s", request_id,
                                     type(exc).__name__)
                    raise ContextPreparationFailedError from exc
            except Exception:
                self._generation_lock.release()
                raise
        context_ms = (time.perf_counter() - context_start) * 1000
        if not context.blocks:
            self._generation_lock.release()
            base["decision"] = {"reason": "no_context_evidence", "best_relevance": decision.best_relevance}
            return base
        evidence_tokens = context.diagnostics["evidence_tokens"]
        context_telemetry = {
            "search_depth": search_depth, "input_multiplier": depth.input_multiplier,
            "output_multiplier": depth.output_multiplier,
            "target_evidence_tokens": depth.target_evidence_tokens,
            "target_answer_tokens": depth.target_answer_tokens,
            "effective_answer_tokens": effective_answer_tokens,
            "response_mode": response_plan.mode,
            "response_target_words": response_plan.target_words,
            "response_requested_answer_tokens": requested_answer_tokens,
            "response_effective_answer_tokens": effective_answer_tokens,
            "search_depth_expanded_output": False if reasoning_enabled else search_depth > 0,
            "retrieved_results": len(candidates),
            "selected_documents": context.diagnostics["selected_documents"],
            "selected_blocks": context.diagnostics["selected_blocks"],
            "raw_candidate_evidence_tokens": original_context.diagnostics["evidence_tokens"],
            "selected_evidence_tokens": evidence_tokens,
            "final_prompt_tokens": fixed_tokens + evidence_tokens,
            "original_evidence_budget": original_budget,
            "effective_evidence_budget": context.diagnostics["evidence_budget_tokens"],
            "latency_budget_seconds": POLICY.preferred_seconds,
            "latency_hard_limit_seconds": POLICY.hard_seconds,
            "estimated_prefill_seconds": estimated,
            "latency_budget_exceeded": estimated is not None and estimated > POLICY.preferred_seconds,
            "context_reduced_for_latency": reduced,
            "context_limited": (depth.target_evidence_tokens is not None and
                                context.diagnostics["evidence_budget_tokens"] < depth.target_evidence_tokens),
            "latency_limited": reduced,
        }
        base["context"] = context_telemetry
        if (estimated is not None and estimated > POLICY.hard_seconds and
                evidence_tokens <= POLICY.minimum_evidence_tokens):
            self._generation_lock.release()
            base.update(status="local_generation_skipped_latency_budget", answerable=False,
                        sources=[block.source() for block in context.blocks],
                        decision={"reason": "local_generation_skipped_latency_budget",
                                  "best_relevance": decision.best_relevance})
            base["timings"] = {"retrieval_ms": retrieval_ms, "context_ms": context_ms,
                               "total_ms": (time.perf_counter() - started) * 1000}
            emit("generation_skipped", reason="local_generation_skipped_latency_budget", terminal=True,
                 **context_telemetry)
            return base
        emit("context_ready", documents=context.diagnostics["selected_documents"],
             blocks=context.diagnostics["selected_blocks"], evidence_tokens=evidence_tokens,
             evidence_budget_tokens=context.diagnostics["evidence_budget_tokens"],
             estimated_total_prompt_tokens=fixed_tokens + evidence_tokens,
             context_window_tokens=self.backend.context_window(), max_answer_tokens=effective_answer_tokens)
        LOGGER.info("RAG context request_id=%s selected_documents=%s selected_seed_chunks=%s "
                    "selected_chunk_ranges=%s target_evidence_tokens=%s effective_evidence_tokens=%s "
                    "target_answer_tokens=%s context_limited=%s latency_limited=%s", request_id,
                    context.diagnostics["selected_documents"], context.diagnostics["selected_seed_chunks"],
                    context.diagnostics["selected_chunk_ranges"], depth.target_evidence_tokens,
                    context.diagnostics["evidence_budget_tokens"], depth.target_answer_tokens,
                    context_telemetry["context_limited"], reduced)
        primary_by_document = {}
        for block in context.blocks:
            primary_by_document.setdefault(block.document_id, block)
        LOGGER.info("RAG final evidence request_id=%s documents=%s context_documents=%s context_blocks=%s "
                    "evidence_tokens=%s", request_id, [{"rank": block.document_rank,
                        "document_id": block.document_id, "filename": block.filename,
                        "primary_seed_chunk": block.seed_chunk_no, "primary_page": block.page_no,
                        "passage_score": block.relevance_score,
                        "scoped_lexical_rank": block.relevance_signals.get("scoped_lexical_rank")}
                    for block in primary_by_document.values()], context.diagnostics["selected_documents"],
                    context.diagnostics["selected_blocks"], evidence_tokens)
        generation_start = time.perf_counter()
        try:
            try:
                emit("generation_started")
                last_sequence, last_waiting_second = -1, -1
                def generation_poll(elapsed):
                    nonlocal last_sequence, last_waiting_second
                    snapshot_fn = getattr(self.backend, "generation_progress", None)
                    snapshot = snapshot_fn() if snapshot_fn else {}
                    sequence = snapshot.get("sequence", 0)
                    if sequence > 0 and sequence != last_sequence:
                        last_sequence = sequence
                        emit("generation_progress", **{key: snapshot.get(key) for key in (
                            "sequence", "generated_fragments", "generated_characters", "elapsed_s",
                            "time_to_first_token_ms", "last_fragment_age_s")})
                    elif sequence == 0 and int(elapsed) != last_waiting_second:
                        last_waiting_second = int(elapsed)
                        emit("generation_waiting_first_token", elapsed_s=elapsed)
                answer = self._run_blocking(
                    lambda: self.backend.generate(system_prompt=active_system_prompt,
                                                  user_prompt=user_prompt(
                                                      question, context.prompt_text,
                                                      response_instruction),
                                                  max_tokens=effective_answer_tokens),
                    generation_poll, interval=.25)
            except GenerationCancelled:
                elapsed = (time.perf_counter() - generation_start) * 1000
                LOGGER.info("RAG generation cancelled request_id=%s elapsed_ms=%.1f", request_id, elapsed)
                emit("cancelled")
                raise
            except Exception as exc:
                LOGGER.exception("Local generation failed request id=%s type=%s", request_id, type(exc).__name__)
                raise GenerationFailedError from exc
        finally:
            self._generation_lock.release()
        generation_ms = (time.perf_counter() - generation_start) * 1000
        telemetry = getattr(self.backend, "last_telemetry", None) or {}
        answer_word_count = len(re.findall(r"\w+", answer or ""))
        citation_match = re.search(r"\[S\d+\]", answer or "")
        telemetry.update({"inference_device_requested": health.get("inference_device_requested"),
                          "inference_device_effective": effective_device,
                          "prompt_tokens": fixed_tokens + evidence_tokens,
                          "response_mode": response_plan.mode,
                          "response_target_words": response_plan.target_words,
                          "response_requested_answer_tokens": requested_answer_tokens,
                          "response_effective_answer_tokens": effective_answer_tokens,
                          "search_depth": search_depth,
                          "search_depth_expanded_output": False if reasoning_enabled else search_depth > 0,
                          "answer_word_count": answer_word_count,
                          "first_citation_char_index": citation_match.start() if citation_match else None,
                          "response_over_target_words": answer_word_count > response_plan.target_words,
                          "response_finished_by_length": telemetry.get("finish_reason") == "length"})
        LOGGER.info("RAG response policy request_id=%s model_id=%s response_mode=%s target_words=%s "
                    "requested_answer_tokens=%s effective_answer_tokens=%s search_depth=%s "
                    "search_depth_expanded_output=%s", request_id, model_id, response_plan.mode,
                    response_plan.target_words, requested_answer_tokens, effective_answer_tokens,
                    search_depth, telemetry["search_depth_expanded_output"])
        ttft = telemetry.get("time_to_first_token_ms")
        if telemetry.get("prompt_eval_ms") is None and ttft:
            telemetry["prompt_eval_ms"] = ttft
            telemetry["prompt_eval_timing_derived"] = True
        if telemetry.get("prompt_eval_tokens_per_second") is None and telemetry.get("prompt_eval_ms"):
            telemetry["prompt_eval_tokens_per_second"] = telemetry["prompt_tokens"] / (telemetry["prompt_eval_ms"] / 1000)
        PERFORMANCE_ESTIMATES.update(model_id, effective_device,
            prompt_tps=telemetry.get("prompt_eval_tokens_per_second"),
            decode_tps=telemetry.get("decode_tokens_per_second"))
        emit("generation_complete", elapsed_s=round(generation_ms / 1000, 1),
             **{key: telemetry[key] for key in ("time_to_first_token_ms", "generation_ms", "decode_ms",
                 "completion_tokens", "decode_tokens_per_second", "overall_tokens_per_second",
                 "generated_characters", "generated_fragments", "finish_reason")
                if telemetry.get(key) is not None})
        emit("validation_started")
        try:
            if (answer or "").strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                base["decision"] = {"reason": "model_declined", "best_relevance": decision.best_relevance}
            else:
                supplied_ids = [block.id for block in context.blocks]
                validation_start = time.perf_counter()
                validation_reason, cited = classify_citations(answer, supplied_ids)
                citation_validation_ms = (time.perf_counter() - validation_start) * 1000
                citation_repair_ms = 0.0
                citation_repair_attempted = False
                citation_repair_reason = "not_needed"
                LOGGER.info("RAG citation validation request_id=%s validation_reason=%s cited_ids=%s "
                            "supplied_ids=%s", request_id, validation_reason, sorted(cited), supplied_ids)
                if validation_reason == "missing_citation":
                    LOGGER.info("RAG missing citation request_id=%s response_mode=%s answer_tokens=%s "
                                "answer_word_count=%s finish_reason=%s first_citation_char_index=None",
                                request_id, response_plan.mode, effective_answer_tokens,
                                answer_word_count, telemetry.get("finish_reason"))
                meaningful_prose = len(re.findall(r"\w+", (answer or "").strip())) >= 4
                repair_allowed = (validation_reason == "missing_citation" and meaningful_prose and
                                  1 <= len(supplied_ids) <= CITATION_REPAIR_MAX_SOURCES and
                                  len(answer) <= CITATION_REPAIR_MAX_ANSWER_CHARACTERS)
                if repair_allowed:
                    repair_tokens = self.backend.count_tokens(answer) + context.diagnostics["evidence_tokens"]
                    repair_estimate = estimate_prefill_seconds(repair_tokens, prompt_tps)
                    if repair_estimate is None or repair_estimate > CITATION_REPAIR_PREFILL_BUDGET_SECONDS:
                        repair_allowed = False
                        citation_repair_reason = "prefill_latency_budget"
                if repair_allowed:
                    repair_start = time.perf_counter()
                    citation_repair_attempted = True
                    citation_repair_reason = "missing_citation_short_grounded_prose"
                    repair_prompt = ("Return the same factual answer ONLY if every claim is supported by the "
                        f"supplied sources {supplied_ids}. Add only allowed citations. Otherwise output exactly "
                        "INSUFFICIENT_EVIDENCE. Do not add claims.\n\n"
                        f"Original answer:\n{answer}\n\n{context.prompt_text}")
                    if not self._generation_lock.acquire(blocking=False):
                        raise ChatBusyError
                    try:
                        answer = self.backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=repair_prompt,
                                                       max_tokens=min(CITATION_REPAIR_MAX_TOKENS,
                                                                      effective_answer_tokens))
                    finally:
                        self._generation_lock.release()
                    citation_repair_ms = (time.perf_counter() - repair_start) * 1000
                    if (answer or "").strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                        validation_reason, cited = "model_declined", set()
                    else:
                        validation_reason, cited = classify_citations(answer, supplied_ids)
                    LOGGER.info("RAG citation repair request_id=%s validation_reason=%s cited_ids=%s "
                                "supplied_ids=%s", request_id, validation_reason, sorted(cited), supplied_ids)
                elif validation_reason != "valid":
                    citation_repair_reason = validation_reason
                LOGGER.info("RAG citation repair telemetry request_id=%s citation_repair_attempted=%s "
                            "citation_repair_reason=%s citation_repair_max_tokens=%s citation_repair_ms=%.1f",
                            request_id, citation_repair_attempted, citation_repair_reason,
                            CITATION_REPAIR_MAX_TOKENS if citation_repair_attempted else 0,
                            citation_repair_ms)
                if validation_reason == "valid":
                    base.update(status="answered", answerable=True, answer=answer,
                                sources=[block.source() for block in context.blocks if block.id in cited])
                elif validation_reason == "model_declined":
                    base["decision"] = {"reason": "model_declined", "best_relevance": decision.best_relevance}
                else:
                    base["status"] = "ungrounded_generation"
                    base["decision"] = {"reason": "invalid_citations", "validation_reason": validation_reason,
                                        "best_relevance": decision.best_relevance}
        except Exception as exc:
            LOGGER.exception("Validation failed request id=%s type=%s", request_id, type(exc).__name__)
            raise ValidationFailedError from exc
        drill_timings = drilldown.timings if drilldown else {}
        base["timings"] = {"retrieval_ms": retrieval_ms, "global_retrieval_ms": retrieval_ms,
                           "document_ranking_ms": max(0.0, (ranking_finished - started) * 1000 - retrieval_ms),
                           **drill_timings, "context_ms": context_ms, "model_load_ms": model_load_ms,
                           "generation_ms": generation_ms,
                           "citation_validation_ms": locals().get("citation_validation_ms", 0.0),
                           "citation_repair_ms": locals().get("citation_repair_ms", 0.0),
                           "total_ms": (time.perf_counter() - started) * 1000}
        if telemetry:
            base["generation"] = telemetry
        if debug:
            base["debug"] = self._debug(retrieval, decision, context)
            base["debug"]["tokens"] = {"context_window_tokens": self.backend.context_window(),
                "system_tokens": system_tokens, "question_tokens": question_tokens,
                "evidence_tokens": context.diagnostics["evidence_tokens"],
                "reserved_answer_tokens": effective_answer_tokens,
                "total_prompt_tokens": fixed_tokens + context.diagnostics["evidence_tokens"]}
        LOGGER.info("RAG generation request_id=%s active_model_id=%s active_model_name=%s device=%s "
                    "evidence_tokens=%s prompt_tokens=%s "
                    "prompt_eval_ms=%s prompt_eval_tps=%s ttft_ms=%s completion_tokens=%s decode_tps=%s "
                    "generation_ms=%s reasoning_tokens=%s answer_tokens=%s completion_tokens_total=%s "
                    "reasoning_budget_tokens=%s reasoning_ms=%s generation_result=%s "
                    "latency_budget_s=%s context_reduced=%s", request_id,
                    self.health().get("active_model_id"), self.health().get("model_name"), effective_device,
                    evidence_tokens,
                    fixed_tokens + evidence_tokens, telemetry.get("prompt_eval_ms"),
                    telemetry.get("prompt_eval_tokens_per_second"), telemetry.get("time_to_first_token_ms"),
                    telemetry.get("completion_tokens"), telemetry.get("decode_tokens_per_second"),
                    telemetry.get("generation_ms"), telemetry.get("reasoning_tokens", 0),
                    telemetry.get("answer_tokens"), telemetry.get("completion_tokens_total"),
                    telemetry.get("reasoning_budget_tokens"), telemetry.get("reasoning_phase_ms"),
                    telemetry.get("generation_result"), POLICY.preferred_seconds, reduced)
        LOGGER.info("RAG request id=%s decision=%s results=%s documents=%s generation_ms=%.1f", request_id,
                    base["decision"]["reason"], len(candidates), decision.content_documents, generation_ms)
        LOGGER.info("RAG latency request_id=%s global_retrieval_ms=%.1f drilldown_ms=%.1f model_load_ms=%.1f "
                    "context_ms=%.1f generation_ms=%.1f citation_repair_ms=%.1f total_ms=%.1f", request_id,
                    base["timings"]["global_retrieval_ms"], drill_timings.get("drilldown_total_ms", 0.0),
                    model_load_ms, context_ms, generation_ms, base["timings"]["citation_repair_ms"],
                    base["timings"]["total_ms"])
        return base

    @staticmethod
    def _debug(retrieval, decision, context):
        return {**retrieval.diagnostics, "gate": decision.public(),
                "selected_evidence_ids": [] if context is None else [x.id for x in context.blocks],
                "selected_chunk_ids": [] if context is None else [x.chunk_ids for x in context.blocks],
                "context_characters": 0 if context is None else len(context.prompt_text)}
