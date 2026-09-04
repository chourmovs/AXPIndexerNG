from axp_client.rag.prompts import SYSTEM_PROMPT, user_prompt


def test_business_context_is_user_guidance_and_recipe_gets_allowlist():
    prompt = user_prompt("Question", "[S1] Evidence", "SKILL RESPONSE RECIPE", ["S1"],
                         business_context="Focus on operations.")
    assert "BUSINESS CONTEXT\nFocus on operations." in prompt
    assert prompt.index("BUSINESS CONTEXT") < prompt.index("--- BEGIN EVIDENCE ---")
    assert "NOT factual evidence" in prompt
    assert "[S1]" in prompt and "SKILL RESPONSE RECIPE" in prompt
    assert "Focus on operations" not in SYSTEM_PROMPT
