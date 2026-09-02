import json

from axp_client.rag.intel_sycl_backend import IntelSyclBackend, sanitize_reasoning_leak
from axp_client.rag.llama_cpp_backend import GenerationConfig


class SseResponse:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        event = json.dumps({"choices": [{"delta": {"content": self.content},
                                         "finish_reason": "stop"}]})
        return iter((f"data: {event}\n".encode(), b"\n", b"data: [DONE]\n", b"\n"))

    def __exit__(self, *_args):
        return False


def test_sanitizer_preserves_structured_visible_answer():
    assert sanitize_reasoning_leak("The density is 0.74 g/cm³ [S1].", True) == (
        "The density is 0.74 g/cm³ [S1].", False, None,
    )


def test_sanitizer_discards_orphan_prefix_through_last_marker():
    content = "first </think> second \\</think> The density is 0.74 g/cm³ [S1]."
    assert sanitize_reasoning_leak(content, True) == (
        "The density is 0.74 g/cm³ [S1].", True, "orphan_closing",
    )


def test_sanitizer_removes_complete_block():
    assert sanitize_reasoning_leak("<think>hidden</think>The answer is 0.74 [S1].", True) == (
        "The answer is 0.74 [S1].", True, "complete_block",
    )


def test_sanitizer_never_exposes_unterminated_block():
    assert sanitize_reasoning_leak("<think>I need to inspect...", True) == (
        "", True, "unterminated",
    )


def test_sanitizer_does_not_affect_non_reasoning_content():
    content = "something </think> literal"
    assert sanitize_reasoning_leak(content, False) == (content, False, None)


def test_generate_sanitizes_before_answer_accounting(tmp_path, monkeypatch):
    config = GenerationConfig(
        max_answer_tokens=256,
        model_id="lfm25-2.6b-q4",
        reasoning_enabled=True,
        reasoning_budget_tokens=48,
        reasoning_format="deepseek",
        min_visible_answer_tokens=96,
    )
    backend = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path)
    tokenized = []
    content = "thinking-like garbage mentioning S9 </think> Actual answer [S1]"
    monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
    monkeypatch.setattr(backend, "count_tokens", lambda text: tokenized.append(text) or len(text.split()))
    monkeypatch.setattr(backend, "_post", lambda _path, _payload: SseResponse(content))

    assert backend.generate(system_prompt="system", user_prompt="user") == "Actual answer [S1]"
    assert tokenized == ["Actual answer [S1]"]
    assert backend.last_telemetry["answer_tokens"] == 3
    assert backend.last_telemetry["reasoning_leak_detected"] is True
    assert backend.last_telemetry["reasoning_leak_type"] == "orphan_closing"
    assert backend.last_telemetry["visible_answer_budget_tokens"] == 256
    assert backend.last_telemetry["native_max_tokens"] == 308
    assert backend.last_telemetry["reasoning_control_headroom_tokens"] == 4


def test_generate_reports_unterminated_leak_as_empty_answer(tmp_path, monkeypatch):
    config = GenerationConfig(reasoning_enabled=True, reasoning_budget_tokens=48,
                              min_visible_answer_tokens=96)
    backend = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path)
    monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
    monkeypatch.setattr(backend, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(backend, "_post", lambda _path, _payload: SseResponse("<think>secret"))

    assert backend.generate(system_prompt="system", user_prompt="user") == ""
    assert backend.last_telemetry["reasoning_leak_detected"] is True
    assert backend.last_telemetry["reasoning_leak_type"] == "unterminated"
    assert backend.last_telemetry["generation_result"] == "reasoning_leak_unterminated"
