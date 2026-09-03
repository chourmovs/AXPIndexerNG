from axp_client.rag.final_protocol import generate_final_answer
from axp_client.rag.llama_cpp_backend import GenerationConfig


class Backend:
    def __init__(self):
        self.config = GenerationConfig(max_answer_tokens=256, reasoning_enabled=True,
            reasoning_budget_tokens=48, min_visible_answer_tokens=96)
        self.calls = []
        self.last_telemetry = {"finish_reason": "stop", "reasoning_budget_tokens": 32}

    def count_tokens(self, text): return len(text.split())
    def context_window(self): return 4096
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "The density is 0.74 g/cm³ [S1]. Extra unsupported sentence."


def test_shared_protocol_has_production_lfm_budget_and_closed_contract():
    backend = Backend()
    result = generate_final_answer(backend=backend, question="What is the density of TEST-MTBE?",
        evidence="[S1]\nDensity: 0.74 g/cm³", allowed_citation_ids=["S1"])
    assert result.response_mode == "scalar_lookup"
    assert result.requested_answer_tokens == result.effective_answer_tokens == 128
    assert backend.calls[0]["max_tokens"] == 128
    assert "S1" in result.user_prompt
    assert result.citation_validation == {"status": "valid", "citations": ["S1"]}
    assert result.canonicalized
    assert result.answer == "The density is 0.74 g/cm³ [S1]."
    assert backend.config.max_answer_tokens == 256
