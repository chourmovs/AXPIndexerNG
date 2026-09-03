from types import SimpleNamespace

import pytest
from axp_client.rag import model_manager as module
from axp_client.rag.hardware import HardwareCapabilities
from axp_client.rag.model_manager import ModelManager, ModelManagerError


class Accelerators:
    runtime_root = "runtime"
    def server_path(self): return "llama-server"
    def manifest(self): return {"installed": True}


class Controller:
    def __init__(self, hardware):
        self.hardware = hardware
        self.accelerators = Accelerators()
    def health(self):
        return {**self.hardware.public(), "accelerator_available": self.hardware.intel_gpu_available}
    def close(self): pass
    def _make_backend(self, *_args): return object()


def pending():
    return HardwareCapabilities("CPU", intel_gpu_detected=True, intel_gpu_name="Intel Iris Xe",
        intel_gpu_available=False, accelerator_reason="intel_sycl_probe_timeout",
        sycl_runtime_installed=True, sycl_probe_ok=False, sycl_probe_error="intel_sycl_probe_timeout")


def available():
    return HardwareCapabilities("CPU", intel_gpu_detected=True, intel_gpu_name="Intel Iris Xe",
        intel_gpu_available=True, accelerator_reason=None, sycl_runtime_installed=True,
        sycl_probe_ok=True, sycl_device_name="Intel Iris Xe")


class Runner:
    def __init__(self, *_args, **_kwargs):
        self.job = SimpleNamespace(state="idle", public=lambda: {"state": "idle"})
    def start(self, profile, model): return {"state": "preparing", "profile": profile, "model": model}


def prepare(monkeypatch, tmp_path, hardware):
    monkeypatch.setattr(module, "load_settings", lambda: {"chat_active_model_id": None})
    monkeypatch.setattr(module, "ModelQualificationRunner", Runner)
    return ModelManager(tmp_path, runtime=Controller(hardware))


def test_pending_probe_still_exposes_qualification_capability(monkeypatch, tmp_path):
    manager = prepare(monkeypatch, tmp_path, pending())
    assert manager.catalog()["hardware"]["qualification_supported"] is True
    assert manager.catalog()["hardware"]["accelerator_available"] is False


def test_qualification_reprobes_once_and_starts(monkeypatch, tmp_path):
    manager = prepare(monkeypatch, tmp_path, pending())
    probes = []
    monkeypatch.setattr(module, "detect_hardware", lambda *_args: probes.append(True) or available())
    assert manager.start_qualification()["state"] == "preparing"
    assert len(probes) == 1


def test_failed_reprobe_does_not_fallback_to_cpu(monkeypatch, tmp_path):
    manager = prepare(monkeypatch, tmp_path, pending())
    probes = []
    monkeypatch.setattr(module, "detect_hardware", lambda *_args: probes.append(True) or pending())
    with pytest.raises(ModelManagerError) as caught:
        manager.start_qualification()
    assert caught.value.code == "intel_gpu_unavailable"
    assert caught.value.details["probe_error"] == "intel_sycl_probe_timeout"
    assert len(probes) == 1


def test_available_intel_starts_without_reprobe(monkeypatch, tmp_path):
    manager = prepare(monkeypatch, tmp_path, available())
    monkeypatch.setattr(module, "detect_hardware", lambda *_args: pytest.fail("unexpected reprobe"))
    assert manager.start_qualification()["state"] == "preparing"
