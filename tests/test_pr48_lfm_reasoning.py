import logging
import time

from axp_client.rag.intel_sycl_backend import IntelSyclBackend
from axp_client.rag.llama_cpp_backend import GenerationConfig
from axp_client.rag.model_catalog import catalog_model


class SseResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        def lines():
            for event in self.events:
                time.sleep(0.001)
                yield f"data: {event}\n".encode()
                yield b"\n"
        return lines()

    def __exit__(self, *_args):
        return False


def reasoning_backend(tmp_path, monkeypatch, events):
    profile = catalog_model("lfm25-2.6b-q4")
    config = GenerationConfig(
        max_answer_tokens=profile.max_answer_tokens,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        repeat_penalty=profile.repeat_penalty,
        model_id=profile.id,
        reasoning_enabled=profile.reasoning_enabled,
        reasoning_budget_tokens=profile.reasoning_budget_tokens,
        reasoning_format=profile.reasoning_format,
        min_visible_answer_tokens=profile.min_visible_answer_tokens,
    )
    backend = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path, sycl_device_id="SYCL0")
    payloads = []
    monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
    monkeypatch.setattr(backend, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(
        backend,
        "_post",
        lambda _path, payload: payloads.append(payload) or SseResponse([*events, "[DONE]"]),
    )
    return backend, payloads


def test_profiles_enable_reasoning_only_for_lfm26():
    lfm = catalog_model("lfm25-2.6b-q4")
    assert (lfm.reasoning_enabled, lfm.reasoning_budget_tokens, lfm.reasoning_format,
            lfm.min_visible_answer_tokens) == (True, 48, "deepseek", 96)
    for model_id in ("smollm3-3b-q4km", "lfm25-1.2b-qad-q4"):
        profile = catalog_model(model_id)
        assert profile.reasoning_enabled is False
        assert profile.reasoning_budget_tokens is None


def test_native_budget_is_fixed_for_normal_and_extended_and_zero_for_tiny(tmp_path, monkeypatch):
    event = '{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}'
    for max_tokens, expected in ((256, 48), (384, 48), (64, 0)):
        backend, payloads = reasoning_backend(tmp_path, monkeypatch, [event])
        assert backend.generate(system_prompt="system", user_prompt="user", max_tokens=max_tokens) == "answer"
        assert payloads[0]["max_tokens"] == max_tokens
        assert payloads[0]["reasoning_budget_tokens"] == expected
        assert payloads[0]["reasoning_format"] == "deepseek"


def test_reasoning_is_private_and_has_separate_telemetry(tmp_path, monkeypatch, caplog):
    secret = "We need check evidence"
    events = [
        f'{{"choices":[{{"delta":{{"reasoning_content":"{secret}"}}}}]}}',
        '{"choices":[{"delta":{"reasoning_content":" carefully"}}]}',
        '{"choices":[{"delta":{"content":"The density is "}}]}',
        '{"choices":[{"delta":{"content":"0.74 g/cm³ [S1]."},"finish_reason":"stop"}]}',
    ]
    backend, _ = reasoning_backend(tmp_path, monkeypatch, events)
    with caplog.at_level(logging.INFO, logger="axp_client"):
        answer = backend.generate(system_prompt="system", user_prompt="user")
    assert answer == "The density is 0.74 g/cm³ [S1]."
    assert secret not in answer and secret not in caplog.text
    telemetry = backend.last_telemetry
    assert telemetry["reasoning_tokens"] > 0 and telemetry["answer_tokens"] > 0
    assert telemetry["visible_ttft_ms"] > telemetry["reasoning_ttft_ms"]
    assert telemetry["completion_tokens"] == telemetry["completion_tokens_total"]
    assert telemetry["generation_result"] == "normal_answer"


def test_output_limit_diagnoses_before_and_after_visible_answer(tmp_path, monkeypatch):
    reasoning_only = [
        '{"choices":[{"delta":{"reasoning_content":"hidden work"},"finish_reason":"length"}]}'
    ]
    backend, _ = reasoning_backend(tmp_path, monkeypatch, reasoning_only)
    assert backend.generate(system_prompt="system", user_prompt="user") == ""
    assert backend.last_telemetry["answer_tokens"] == 0
    assert backend.last_telemetry["generation_result"] == "output_limit_before_visible_answer"

    with_answer = [
        '{"choices":[{"delta":{"reasoning_content":"hidden"}}]}',
        '{"choices":[{"delta":{"content":"Visible [S1]"},"finish_reason":"length"}]}',
    ]
    backend, _ = reasoning_backend(tmp_path, monkeypatch, with_answer)
    assert backend.generate(system_prompt="system", user_prompt="user") == "Visible [S1]"
    assert backend.last_telemetry["generation_result"] == "output_limit_after_visible_answer"


def test_non_reasoning_payload_has_no_reasoning_parameters(tmp_path, monkeypatch):
    config = GenerationConfig()
    backend = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path, sycl_device_id="SYCL0")
    payloads = []
    monkeypatch.setattr(backend, "ensure_loaded", lambda: backend)
    monkeypatch.setattr(backend, "count_tokens", lambda _text: 1)
    monkeypatch.setattr(
        backend,
        "_post",
        lambda _path, payload: payloads.append(payload) or SseResponse([
            '{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}', "[DONE]"
        ]),
    )
    assert backend.generate(system_prompt="system", user_prompt="user") == "ok"
    assert "reasoning_budget_tokens" not in payloads[0]
    assert "reasoning_format" not in payloads[0]
    assert backend.last_telemetry["reasoning_tokens"] == 0
