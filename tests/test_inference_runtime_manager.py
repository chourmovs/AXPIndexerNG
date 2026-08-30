import pytest
from axp_client.rag.hardware import HardwareCapabilities
from axp_client.rag.runtime_manager import InferenceDeviceError, InferenceRuntimeManager


class Backend:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def health(self):
        return {"available": True, "backend": "fake", "model_state": "unloaded", "model_name": self.name}

    def close(self):
        self.closed = True


def settings(device="auto", path="old.gguf"):
    return {"chat_model_path": path, "chat_active_model_id": None, "chat_inference_device": device}


def test_activation_prepares_then_swaps_and_closes_old():
    created = []
    def factory(configuration, _profile):
        backend = Backend(configuration["chat_model_path"]); created.append(backend); return backend
    runtime = InferenceRuntimeManager(settings(), backend_factory=factory,
        hardware=HardwareCapabilities("cpu", accelerator_reason="no_intel_gpu"))
    old = runtime.backend
    replacement = runtime.prepare_activation(settings(path="new.gguf"))
    assert runtime.backend is old and not old.closed
    runtime.commit_activation(settings(path="new.gguf"), replacement)
    assert runtime.backend is replacement and old.closed


def test_failed_replacement_construction_keeps_working_backend():
    def factory(configuration, _profile):
        if configuration["chat_model_path"] == "bad.gguf": raise RuntimeError("bad backend")
        return Backend(configuration["chat_model_path"])
    runtime = InferenceRuntimeManager(settings(), backend_factory=factory,
        hardware=HardwareCapabilities("cpu", accelerator_reason="no_intel_gpu"))
    old = runtime.backend
    with pytest.raises(RuntimeError, match="bad backend"):
        runtime.prepare_activation(settings(path="bad.gguf"))
    assert runtime.backend is old and not old.closed


def test_cpu_only_device_policy_is_truthful():
    hardware = HardwareCapabilities("cpu", True, "Intel Iris Xe", False, "accelerator_not_installed")
    runtime = InferenceRuntimeManager(settings(), backend_factory=lambda *_args: Backend("model"), hardware=hardware)
    auto = runtime.health()
    assert auto["inference_device_requested"] == "auto"
    assert auto["inference_device_effective"] == "cpu"
    assert auto["fallback_reason"] == "accelerator_not_installed"
    runtime.set_device("cpu")
    assert runtime.health()["inference_device_requested"] == "cpu"
    with pytest.raises(InferenceDeviceError, match="intel_gpu_unavailable"):
        runtime.set_device("intel_gpu")
    forced = InferenceRuntimeManager(settings("intel_gpu"), backend_factory=lambda *_args: Backend("model"),
                                     hardware=hardware).health()
    assert forced["inference_device_effective"] == "cpu"
    assert forced["fallback_reason"] == "accelerator_not_installed"
