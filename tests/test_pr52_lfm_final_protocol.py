import io

from axp_client.rag.citations import classify_citations
from axp_client.rag.intel_sycl_backend import IntelSyclBackend
from axp_client.rag.llama_cpp_backend import GenerationConfig
from axp_client.rag.model_catalog import catalog_model
from axp_client.rag.prompts import user_prompt
from axp_client.rag.response_policy import (canonicalize_scalar_response,
                                            classify_response_plan,
                                            cleanup_truncated_tail)
from axp_client.rag.retrieval import classify_query_evidence_intent


def plan(question):
    return classify_response_plan(question, classify_query_evidence_intent(question))


def test_exact_closed_citation_contract_and_legacy_isolation():
    legacy = user_prompt("Q", "E", "Do it")
    exact = user_prompt("Q", "E", "Do it", ["S1"])
    several = user_prompt("Q", "E", "Do it", ["S1", "S2", "S3"])
    assert "ALLOWED CITATIONS" not in legacy
    assert "ALLOWED CITATIONS\n[S1]\n" in exact
    assert "Any other [S<number>] is invalid." in exact
    assert "[S1] [S2] [S3]" in several


def test_response_budgets_and_search_more_invariance():
    expected = {
        "What is the density of MTBE?": ("scalar_lookup", 128, 40),
        "What packaging is possible?": ("direct_lookup", 192, 100),
        "Summarize the main properties.": ("summary", 288, 160),
        "Compare these materials.": ("analytical", 320, 200),
    }
    for question, values in expected.items():
        response = plan(question)
        assert (response.mode, response.answer_tokens, response.target_words) == values
        assert [response.answer_tokens for _depth in (0, 1)] == [values[1], values[1]]
    assert "maximum 10 concise bullets" in plan("Summarize the main properties.").instruction
    assert "no repetitive conclusion" in plan("Summarize the main properties.").instruction


def test_scalar_canonicalization_never_repairs_or_invents_citations():
    answer = "The density is 0.74 g/cm³ [S1]. Extra unrelated material [S9]."
    cleaned, applied, reason = canonicalize_scalar_response(answer, ["S1"])
    assert cleaned == "The density is 0.74 g/cm³ [S1]."
    assert (applied, reason) == (True, "scalar_first_allowed_cited_sentence")
    assert classify_citations(cleaned, ["S1"])[0] == "valid"
    for invalid, validation in (("The density is 0.74 [S9].", "unknown_citation"),
                                ("The density is 0.74.", "missing_citation")):
        assert canonicalize_scalar_response(invalid, ["S1"])[0] == invalid
        assert classify_citations(invalid, ["S1"])[0] == validation
    assert canonicalize_scalar_response("INSUFFICIENT_EVIDENCE", ["S1"])[1] is False


def test_conservative_truncated_tail_cleanup():
    answer = "- Property A [S1]\n- Property B [S1]\n\nThese properties indicate that the material"
    assert cleanup_truncated_tail(answer, ["S1"]) == ("- Property A [S1]\n- Property B [S1]", True)
    cited_tail = "- Property A [S1]\n- Incomplete property [S1] and"
    assert cleanup_truncated_tail(cited_tail, ["S1"]) == (cited_tail, False)


class Response:
    def __init__(self):
        self.body = (b'data: {"choices":[{"delta":{"reasoning_content":"why "}}]}\n\n'
                     b'data: {"choices":[{"delta":{"content":"answer [S1]"},"finish_reason":"stop"}],'
                     b'"timings":{"predicted_n":3}}\n\ndata: [DONE]\n\n')
    def __enter__(self): return io.BytesIO(self.body)
    def __exit__(self, *_args): return False


def test_reasoning_transition_payload_and_headroom(tmp_path, monkeypatch):
    profile = catalog_model("lfm25-2.6b-q4")
    config = GenerationConfig(max_answer_tokens=128, model_id=profile.id,
        reasoning_enabled=profile.reasoning_enabled, reasoning_budget_tokens=profile.reasoning_budget_tokens,
        reasoning_budget_message=profile.reasoning_budget_message, reasoning_format=profile.reasoning_format)
    backend = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path)
    payloads = []
    monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
    monkeypatch.setattr(backend, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(backend, "_post", lambda _path, payload: payloads.append(payload) or Response())
    assert backend.generate(system_prompt="s", user_prompt="u") == "answer [S1]"
    payload = payloads[0]
    assert payload["reasoning_budget_tokens"] == 48
    assert payload["reasoning_budget_message"] == "Now produce the final answer only."
    assert payload["reasoning_format"] == "deepseek"
    message_tokens = len(profile.reasoning_budget_message.split())
    assert payload["max_tokens"] == 128 + 48 + 4 + message_tokens
    assert backend.last_telemetry["reasoning_budget_message_tokens"] == message_tokens


def test_non_reasoning_payload_has_no_reasoning_fields(tmp_path, monkeypatch):
    for model_id in ("smollm3-3b-q4km", "lfm25-1.2b-qad-q4"):
        profile = catalog_model(model_id)
        backend = IntelSyclBackend(tmp_path / "model.gguf", GenerationConfig(), tmp_path)
        payloads = []
        monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
        monkeypatch.setattr(backend, "count_tokens", lambda text: len(text.split()))
        monkeypatch.setattr(backend, "_post", lambda _path, payload: payloads.append(payload) or Response())
        backend.generate(system_prompt="s", user_prompt="u")
        assert not ({"reasoning_budget_tokens", "reasoning_budget_message", "reasoning_format"} & payloads[0].keys())
        assert profile.reasoning_budget_message is None
