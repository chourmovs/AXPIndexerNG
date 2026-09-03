import io
from types import SimpleNamespace

import pytest

from axp_client.rag.hardware import HardwareCapabilities
from axp_client.rag.intel_sycl_backend import (
    DEVICE_LOST_CODE,
    DEVICE_LOST_TYPE,
    IntelSyclBackend,
    IntelSyclError,
    classify_native_failure,
)
from axp_client.rag.llama_cpp_backend import GenerationCancelled, GenerationConfig, build_chat_invocation
from axp_client.rag.model_catalog import CATALOG_VERSION, catalog_model
from axp_client.rag.runtime_manager import InferenceRuntimeManager, generation_config_for_profile


def backend(tmp_path, **kwargs):
    return IntelSyclBackend(tmp_path / "model.gguf", GenerationConfig(), tmp_path,
                            sycl_device_id="SYCL0", **kwargs)


def test_native_marker_is_specific():
    failure = classify_native_failure(
        "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)")
    assert (failure["failure_type"], failure["failure_code"], failure["retryable"]) == (
        DEVICE_LOST_TYPE, DEVICE_LOST_CODE, True)
    assert classify_native_failure("SYCL warning: optional feature unavailable") is None


def test_sidecar_command_disables_prompt_cache_and_preserves_gpu_flags(tmp_path):
    value = backend(tmp_path)
    value._port = 1234
    value._auth_file = tmp_path / "key"
    command = value._sidecar_command()
    for pair in (("--cache-ram", "0"), ("--device", "SYCL0"),
                 ("--split-mode", "none"), ("--n-gpu-layers", "all"), ("--parallel", "1")):
        position = command.index(pair[0])
        assert command[position + 1] == pair[1]


class ResetResponse:
    def __enter__(self):
        raise ConnectionResetError(10054, "reset")

    def __exit__(self, *_args):
        return False


def test_native_cause_beats_connection_reset_and_cleans_up(tmp_path, monkeypatch):
    value = backend(tmp_path)
    value._process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout: None, pid=1)
    value._model_state = "loaded"; value.gpu_offload_confirmed = True
    monkeypatch.setattr(value, "ensure_loaded", lambda: value)
    monkeypatch.setattr(value, "_post", lambda *_args, **_kwargs: ResetResponse())
    value._record_native_failure("UR_RESULT_ERROR_DEVICE_LOST", "generating")
    with pytest.raises(IntelSyclError, match=DEVICE_LOST_TYPE):
        value.generate(system_prompt="system", user_prompt="user")
    health = value.health()
    assert not health["sidecar_running"] and not health["model_loaded"]
    assert health["model_state"] == "unloaded"
    assert health["terminal_result"] == "device_lost"
    assert health["failure_code"] == DEVICE_LOST_CODE


def test_pure_connection_reset_remains_transport_failure(tmp_path, monkeypatch):
    value = backend(tmp_path)
    value._process = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout: None, pid=1)
    value._model_state = "loaded"
    monkeypatch.setattr(value, "ensure_loaded", lambda: value)
    monkeypatch.setattr(value, "_post", lambda *_args, **_kwargs: ResetResponse())
    with pytest.raises(IntelSyclError, match="intel_gpu_generation_connection_lost"):
        value.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(("model_id", "expected"), [
    ("smollm3-3b-q4km", (0.2, 0.8, 20, 1.0)),
    ("lfm25-1.2b-qad-q4", (0.1, 0.1, 50, 1.05)),
    ("lfm25-2.6b-q4", (0.1, 1.0, 50, 1.1)),
])
def test_intel_payload_includes_profile_temperature(tmp_path, monkeypatch, model_id, expected):
    profile = catalog_model(model_id)
    config = GenerationConfig(temperature=profile.temperature, top_p=profile.top_p,
                              top_k=profile.top_k, repeat_penalty=profile.repeat_penalty)
    value = IntelSyclBackend(tmp_path / "model.gguf", config, tmp_path, sycl_device_id="SYCL0")
    payloads = []
    monkeypatch.setattr(value, "ensure_loaded", lambda: value)
    monkeypatch.setattr(value, "count_tokens", lambda _text: 1)
    class Response:
        def __enter__(self): return io.BytesIO(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
        def __exit__(self, *_args): return False
    monkeypatch.setattr(value, "_post", lambda _path, payload: payloads.append(payload) or Response())
    assert value.generate(system_prompt="system", user_prompt="user") == "ok"
    assert tuple(payloads[0][key] for key in ("temperature", "top_p", "top_k", "repeat_penalty")) == expected


def test_lfm26_catalog_and_cpu_sampler_contract():
    profile = catalog_model("lfm25-2.6b-q4")
    assert CATALOG_VERSION == 4
    assert (profile.repository, profile.revision, profile.filename) == (
        "LiquidAI/LFM2.5-2.6B-GGUF", "b22e29ebf6249a8c9fcdda36914743e9980595c4",
        "LFM2.5-2.6B-Q4_0.gguf")
    assert (profile.sha256, profile.size_bytes, profile.license, profile.quantization) == (
        "91ad0c3150fdfd0d66d1abc0fbb1491c1d4cc14ae74915ddc937345c697c3a2b",
        1_593_894_720, "LFM Open License v1.0", "Q4_0")
    config = GenerationConfig(temperature=profile.temperature, top_p=profile.top_p,
                              top_k=profile.top_k, repeat_penalty=profile.repeat_penalty)
    def completion(messages, max_tokens, temperature, top_p, top_k, repeat_penalty): pass
    invocation, *_ = build_chat_invocation(completion, system_prompt="s", user_prompt="u",
                                            config=config, template_kwargs={})
    assert tuple(invocation[key] for key in ("temperature", "top_p", "top_k", "repeat_penalty")) == (
        0.1, 1.0, 50, 1.1)


class RecoveryBackend(IntelSyclBackend):
    def __init__(self, outcome):
        self.outcome = outcome; self.attempts = 0; self.closed = False
        self.sycl_device_id = "SYCL0"; self.session_id = "session"; self.offloaded_layers = 30
        self.last_device_loss_at = "now"
    def generate(self, **_request):
        self.attempts += 1
        if self.outcome == "lost": raise IntelSyclError(DEVICE_LOST_TYPE)
        if self.outcome == "cancel": raise GenerationCancelled
        return "answer"
    def ensure_loaded(self): return self
    def close(self, *args, **kwargs): self.closed = True
    def health(self): return {"backend": "intel_sycl", "gpu_offload_confirmed": True}


def recovery_manager(outcomes):
    created = []
    def factory(_settings, _profile):
        item = RecoveryBackend(outcomes[len(created)])
        item.config = generation_config_for_profile(_profile)
        created.append(item)
        return item
    hardware = HardwareCapabilities("cpu", intel_gpu_available=True, sycl_device_id="SYCL0")
    manager = InferenceRuntimeManager({"chat_model_path": "model.gguf", "chat_active_model_id": "lfm25-2.6b-q4",
        "chat_inference_device": "intel_gpu"}, intel_backend_factory=factory, hardware=hardware)
    return manager, created


def test_one_intel_only_recovery_succeeds():
    manager, created = recovery_manager(["lost", "success"])
    assert manager.generate(system_prompt="s", user_prompt="u", max_tokens=7) == "answer"
    assert [item.attempts for item in created] == [1, 1]
    assert created[0].config == created[1].config
    assert manager.health()["device_recovery_succeeded"] is True


def test_second_device_loss_is_terminal_and_cancel_is_not_retried():
    manager, created = recovery_manager(["lost", "lost"])
    with pytest.raises(IntelSyclError, match=DEVICE_LOST_TYPE):
        manager.generate(system_prompt="s", user_prompt="u")
    assert [item.attempts for item in created] == [1, 1]
    cancelled, cancel_created = recovery_manager(["cancel", "success"])
    with pytest.raises(GenerationCancelled):
        cancelled.generate(system_prompt="s", user_prompt="u")
    assert len(cancel_created) == 1


def test_model_switch_does_not_resurrect_stale_backend():
    manager, created = recovery_manager(["lost", "success"])
    stale = created[0]
    replacement = RecoveryBackend("success")
    def switch_then_fail(**_request):
        manager.backend = replacement
        manager.settings["chat_active_model_id"] = "smollm3-3b-q4km"
        raise IntelSyclError(DEVICE_LOST_TYPE)
    stale.generate = switch_then_fail
    with pytest.raises(IntelSyclError, match="intel_gpu_recovery_model_changed"):
        manager.generate(system_prompt="s", user_prompt="u")
    assert manager.backend is replacement and replacement.attempts == 0
