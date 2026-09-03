from dataclasses import fields

from axp_client.rag.hardware import HardwareCapabilities
from axp_client.rag.llama_cpp_backend import GenerationConfig
from axp_client.rag.model_catalog import CATALOG_VERSION, catalog_model
from axp_client.rag.response_policy import classify_response_plan
from axp_client.rag.retrieval import classify_query_evidence_intent
from axp_client.rag.runtime_manager import InferenceRuntimeManager, generation_config_for_profile


LFM_ID = "lfm25-2.6b-q4"
MINISTRAL_ID = "ministral3-3b-2512-q4km"


def settings(model_id=LFM_ID, device="cpu"):
    return {"chat_model_path": "model.gguf", "chat_active_model_id": model_id,
            "chat_inference_device": device}


class Backend:
    def __init__(self, profile):
        self.config = generation_config_for_profile(profile)
        self.closed = False

    def health(self):
        return {"available": True, "backend": "intel_sycl", "gpu_offload_confirmed": True}

    def ensure_loaded(self):
        return self

    def count_tokens(self, text):
        return len(text.split())

    def close(self):
        self.closed = True


def cpu_runtime(configuration=None):
    configuration = configuration or settings()
    return InferenceRuntimeManager(configuration, backend_factory=lambda _settings, profile: Backend(profile),
                                   hardware=HardwareCapabilities("cpu"))


def assert_lfm(config):
    assert config.model_id == LFM_ID
    assert config.reasoning_enabled is True
    assert config.reasoning_budget_tokens == 48
    assert config.max_answer_tokens == 256
    assert config.max_evidence_tokens == 3072
    assert config.temperature == 0.1


def test_active_lfm_profile_is_applied_at_startup_without_activation():
    runtime = cpu_runtime()
    assert_lfm(runtime.backend.config)
    health = runtime.health()
    assert health["active_model_id"] == health["backend_profile_id"] == LFM_ID
    assert health["backend_reasoning_enabled"] is True
    assert health["backend_max_answer_tokens"] == 256
    assert health["backend_max_evidence_tokens"] == 3072
    assert health["backend_temperature"] == 0.1


def test_pending_intel_has_complete_lfm_profile_before_qualification():
    hardware = HardwareCapabilities("cpu", intel_gpu_detected=True,
                                    sycl_probe_error="intel_sycl_probe_timeout")
    runtime = InferenceRuntimeManager(settings(device="intel_gpu"), hardware=hardware)
    assert_lfm(runtime.backend.config)


def test_first_request_qualification_retains_lfm_response_plan_and_config():
    unavailable = HardwareCapabilities("cpu", intel_gpu_detected=True,
                                       sycl_probe_error="intel_sycl_probe_timeout")
    available = HardwareCapabilities("cpu", intel_gpu_available=True, sycl_device_id="SYCL0")
    created = []

    def factory(_settings, profile):
        backend = Backend(profile)
        created.append(backend)
        return backend

    runtime = InferenceRuntimeManager(settings(device="intel_gpu"), hardware=unavailable,
                                      hardware_probe=lambda *_args, **_kwargs: available,
                                      intel_backend_factory=factory)
    pending_config = runtime.backend.config
    plan = classify_response_plan("What is the density of MTBE?",
                                  classify_query_evidence_intent("What is the density of MTBE?"))
    assert plan.answer_tokens == 128
    assert pending_config.reasoning_budget_tokens == 48
    runtime.count_tokens("first request")
    assert_lfm(runtime.backend.config)
    for field in fields(GenerationConfig):
        assert getattr(pending_config, field.name) == getattr(runtime.backend.config, field.name)


def test_restart_resolves_the_persisted_active_profile():
    persisted = settings()
    first = cpu_runtime(persisted)
    first.activate(persisted, catalog_model(LFM_ID))
    restarted = cpu_runtime(dict(first.settings))
    assert first.backend.config == restarted.backend.config == generation_config_for_profile(catalog_model(LFM_ID))


def test_switching_from_lfm_to_ministral_clears_reasoning_configuration():
    runtime = cpu_runtime()
    switched = settings(MINISTRAL_ID)
    runtime.activate(switched, catalog_model(MINISTRAL_ID))
    config = runtime.backend.config
    assert (config.model_id, config.temperature, config.top_p, config.top_k, config.repeat_penalty) == (
        MINISTRAL_ID, 0.2, 0.8, 20, 1.0)
    assert config.reasoning_enabled is False
    assert config.reasoning_budget_tokens is None
    assert config.reasoning_budget_message is None
    assert config.reasoning_format is None


def test_ministral_catalog_contract_and_private_download_metadata():
    assert CATALOG_VERSION == 4
    profile = catalog_model(MINISTRAL_ID)
    assert (profile.repository, profile.revision, profile.filename) == (
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        "eb599d408350ea2bb60452cb86be7c7b2fc28227",
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf")
    assert (profile.sha256, profile.size_bytes, profile.license, profile.quantization,
            profile.profile, profile.context_size) == (
        "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8",
        2_147_023_008, "Apache-2.0", "Q4_K_M", "balanced", 6144)
    assert profile.experimental is True and profile.reasoning_enabled is False
    assert not {"revision", "filename", "sha256"} & profile.public().keys()
