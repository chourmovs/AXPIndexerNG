"""The single final-answer protocol shared by production RAG and qualification.

Retrieval and context packing intentionally do not live here.  This module starts
at the prepared evidence boundary so deterministic qualification can exercise
exactly the same prompts, response budget and output processing as production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .citations import classify_citations
from .llama_cpp_backend import GenerationConfig
from .prompts import system_prompt, user_prompt
from .response_policy import (ResponsePlan, canonicalize_scalar_response,
                              classify_response_plan, cleanup_truncated_tail)
from .retrieval import classify_query_evidence_intent


@dataclass
class FinalAnswerResult:
    answer: str
    query_intent: str
    response_mode: str
    target_words: int
    requested_answer_tokens: int
    effective_answer_tokens: int
    allowed_citation_ids: list[str]
    citation_validation: dict
    canonicalized: bool
    truncated_tail_cleaned: bool
    generation_telemetry: dict
    system_prompt: str
    user_prompt: str


def generate_final_answer(*, backend, question: str, evidence: str,
                          allowed_citation_ids: list[str], search_depth: int = 0,
                          response_plan: ResponsePlan | None = None,
                          requested_answer_tokens: int | None = None,
                          effective_answer_tokens: int | None = None,
                          generate_call: Callable[..., str] | None = None) -> FinalAnswerResult:
    """Generate and validate an answer from already prepared evidence."""
    config = getattr(backend, "config", GenerationConfig())
    intent = classify_query_evidence_intent(question)
    plan = response_plan or classify_response_plan(question, intent)
    reasoning_enabled = bool(getattr(config, "reasoning_enabled", False))
    requested = requested_answer_tokens if requested_answer_tokens is not None else (
        plan.answer_tokens if reasoning_enabled else getattr(config, "max_answer_tokens", 384))
    active_system = system_prompt(reasoning_enabled)
    instruction = plan.instruction if reasoning_enabled else None
    active_user = user_prompt(question, evidence, instruction,
                              allowed_citation_ids if reasoning_enabled else None)
    if effective_answer_tokens is None:
        fixed = backend.count_tokens(active_system) + backend.count_tokens(active_user)
        window = getattr(backend, "context_window", lambda: getattr(config, "context_size", 6144))()
        effective = min(requested, max(0, window - fixed - getattr(config, "safety_tokens", 96)))
    else:
        effective = effective_answer_tokens
    call = generate_call or backend.generate
    answer = call(system_prompt=active_system, user_prompt=active_user, max_tokens=effective)
    answer = answer if isinstance(answer, str) else getattr(answer, "text", str(answer or ""))
    telemetry = dict(getattr(backend, "last_telemetry", None) or {})
    canonicalized = tail_cleaned = False
    canonicalization_reason = None
    pre_tokens = backend.count_tokens(answer) if answer else 0
    if reasoning_enabled and plan.mode == "scalar_lookup":
        answer, canonicalized, canonicalization_reason = canonicalize_scalar_response(
            answer, allowed_citation_ids)
    elif (reasoning_enabled and telemetry.get("finish_reason") == "length" and
          plan.mode in {"summary", "analytical", "direct_lookup"}):
        answer, tail_cleaned = cleanup_truncated_tail(answer, allowed_citation_ids)
    telemetry.update(
        response_canonicalized=canonicalized,
        response_canonicalization_reason=canonicalization_reason,
        pre_canonical_answer_tokens=pre_tokens,
        post_canonical_answer_tokens=backend.count_tokens(answer) if answer else 0,
        truncated_tail_cleanup_applied=tail_cleaned,
        response_mode=plan.mode,
        response_target_words=plan.target_words,
        response_requested_answer_tokens=requested,
        response_effective_answer_tokens=effective,
        search_depth=search_depth,
    )
    reason, cited = classify_citations(answer, allowed_citation_ids)
    if answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
        reason, cited = "model_declined", set()
    return FinalAnswerResult(
        answer=answer, query_intent=intent.kind, response_mode=plan.mode,
        target_words=plan.target_words, requested_answer_tokens=requested,
        effective_answer_tokens=effective, allowed_citation_ids=list(allowed_citation_ids),
        citation_validation={"status": reason, "citations": sorted(cited)},
        canonicalized=canonicalized, truncated_tail_cleaned=tail_cleaned,
        generation_telemetry=telemetry, system_prompt=active_system, user_prompt=active_user)
