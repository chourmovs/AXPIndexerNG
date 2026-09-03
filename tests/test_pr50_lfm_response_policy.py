from axp_client.rag.depth import depth_policy
from axp_client.rag.prompts import SYSTEM_PROMPT, system_prompt, user_prompt
from axp_client.rag.response_policy import classify_response_plan
from axp_client.rag.retrieval import classify_query_evidence_intent


def plan(question):
    return classify_response_plan(question, classify_query_evidence_intent(question))


def test_scalar_lookup():
    response = plan("What is the density of MTBE?")
    assert (response.mode, response.answer_tokens, response.target_words) == ("scalar_lookup", 128, 40)


def test_french_scalar_uses_existing_intent_without_expanding_it():
    intent = classify_query_evidence_intent("Quelle est la densité du MTBE ?")
    assert plan("Quelle est la densité du MTBE ?").mode == (
        "scalar_lookup" if intent.kind == "scalar_fact" else "direct_lookup"
    )


def test_direct_lookups():
    assert plan("What is the DOMINO reaction sequence?").mode == "direct_lookup"
    packaging = plan("What packaging is possible for ammonia 20.5%?")
    assert (packaging.mode, packaging.answer_tokens) == ("direct_lookup", 192)


def test_english_and_french_summaries():
    for question in (
        "Summarize the main physical properties of n-Heptane 99%.",
        "Résume les principales propriétés physiques du n-heptane 99%.",
    ):
        response = plan(question)
        assert (response.mode, response.answer_tokens, response.target_words) == ("summary", 288, 160)


def test_analytical_request():
    response = plan("Compare process A and process B.")
    assert (response.mode, response.answer_tokens) == ("analytical", 320)


def test_lfm_budget_does_not_follow_depth_output_multiplier():
    for question, expected in (
        ("What is the DOMINO reaction sequence?", 192),
        ("Summarize the main properties.", 288),
    ):
        response = plan(question)
        assert [response.answer_tokens for _ in (0, 1)] == [expected, expected]


def test_search_more_still_expands_retrieval_and_legacy_output():
    normal = depth_policy(0, evidence_tokens=1000, answer_tokens=256)
    expanded = depth_policy(1, evidence_tokens=1000, answer_tokens=256)
    for field in ("retrieval_limit", "candidate_depth", "seed_limit", "neighbor_radius",
                  "target_evidence_tokens"):
        assert getattr(expanded, field) > getattr(normal, field)
    assert (normal.target_answer_tokens, expanded.target_answer_tokens) == (256, 384)


def test_reasoning_system_suffix_is_isolated():
    assert system_prompt(False) == SYSTEM_PROMPT
    assert "FINAL RESPONSE DISCIPLINE" not in system_prompt(False)
    assert "FINAL RESPONSE DISCIPLINE" in system_prompt(True)


def test_mode_contract_is_citation_first_and_summary_is_not_sentence_limited():
    scalar = user_prompt("Question?", "[S1] evidence", plan("What is density?").instruction)
    assert "first substantive factual sentence" in scalar
    assert "valid supplied citation" in scalar
    summary = user_prompt("Summarize this", "[S1] evidence", plan("Summarize this").instruction)
    assert "Answer in 1–3 sentences." not in summary


def test_legacy_prompt_is_unchanged():
    expected = ("QUESTION\nQ\n\nEVIDENCE\n--- BEGIN EVIDENCE ---\nE\n--- END EVIDENCE ---\n\n"
                "Answer in 1–3 sentences. Every answer MUST contain [Sx]. If no supported answer exists, "
                "output INSUFFICIENT_EVIDENCE.")
    assert user_prompt("Q", "E") == expected
