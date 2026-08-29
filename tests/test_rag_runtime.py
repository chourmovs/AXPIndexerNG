import hashlib
import struct

import pytest
from axp_client.rag.answerability import AnswerabilityConfig, decide_answerability
from axp_client.rag.evaluation import confusion, threshold_sweep
from axp_client.rag.factory import ChatBackendConfigurationError, create_chat_backend
from axp_client.rag.llama_cpp_backend import GenerationConfig, LlamaCppBackend
from axp_client.rag.model import import_model, model_status


def candidate(vector, lexical, **values):
    return {"document_id": 1, "chunk_no": values.pop("chunk_no", 0), "vector_similarity": vector,
            "lexical_coverage": lexical, **values}


def test_gate_does_not_confuse_lexical_coverage_with_probability():
    decision = decide_answerability([candidate(0.2, 1.0)])
    assert not decision.answerable
    assert decision.reason == "topic_match_without_answer_support"


def test_gate_vector_lexical_and_content_identifier_rules():
    assert decide_answerability([candidate(0.6, 0.3)]).answerable
    assert decide_answerability([candidate(0.46, 0.3, exact_identifier_match=True)]).answerable
    assert not decide_answerability([candidate(0.46, 0.3, exact_filename_match=True)]).answerable


def test_backend_factory_has_no_fallback(tmp_path):
    backend = create_chat_backend({"chat_backend": "llama_cpp", "chat_model_path": tmp_path / "x.gguf"})
    assert isinstance(backend, LlamaCppBackend)
    with pytest.raises(ChatBackendConfigurationError):
        create_chat_backend({"chat_backend": "cloud", "chat_model_path": "x"})


def test_model_import_is_atomic_hashed_and_retains_source(tmp_path):
    source = tmp_path / "fixture.gguf"
    source.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\0" * 32)
    destination = tmp_path / "cache" / "model.gguf"
    manifest = import_model(source, destination)
    assert source.exists() and destination.read_bytes() == source.read_bytes()
    assert manifest["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert model_status(destination)["manifest"] == manifest
    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(b"NOPE" + b"\0" * 40)
    with pytest.raises(ValueError, match="model_invalid"):
        import_model(invalid, destination)


def test_non_thinking_generation_and_reasoning_removal(monkeypatch, tmp_path):
    calls = {}
    class Model:
        def create_chat_completion(self, **kwargs):
            calls.update(kwargs)
            return {"choices": [{"message": {"content": "<think>secret</think>Final [S1]"}}], "usage": {}}
    backend = LlamaCppBackend(tmp_path / "unused", GenerationConfig())
    backend._model = Model()
    assert backend.generate(system_prompt="system", user_prompt="user") == "Final [S1]"
    assert calls["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls["temperature"] > 0


def test_evaluator_metrics_and_sweep_are_conservative_and_deterministic():
    outcomes = [{"expected_answerable": True, "actual_answerable": True},
                {"expected_answerable": False, "actual_answerable": False},
                {"expected_answerable": False, "actual_answerable": True},
                {"expected_answerable": True, "actual_answerable": False}]
    metrics = confusion(outcomes)
    assert (metrics["correct_accepts"], metrics["correct_refusals"], metrics["false_accepts"],
            metrics["false_refusals"]) == (1, 1, 1, 1)
    cases = [{"expected_answerable": True, "hits": [candidate(.6, .3)]},
             {"expected_answerable": False, "hits": [candidate(.2, 1)]}]
    configs = [AnswerabilityConfig(strong_vector_similarity=.55), AnswerabilityConfig(strong_vector_similarity=.65)]
    result = threshold_sweep(cases, configs)
    assert result == threshold_sweep(cases, configs)
    assert result[0]["metrics"]["false_accept_rate"] == 0
