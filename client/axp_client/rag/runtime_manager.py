"""Hot-switchable local inference runtime."""
import logging
import threading

from .hardware import detect_hardware
from .llama_cpp_backend import GenerationConfig, LlamaCppBackend
from .model_catalog import catalog_model

ALLOWED_DEVICES = {"auto", "cpu", "intel_gpu"}
LOGGER = logging.getLogger("axp_client")


class InferenceDeviceError(ValueError):
    pass


class InferenceRuntimeManager:
    def __init__(self, settings, backend_factory=None, hardware=None):
        self._lock = threading.RLock()
        self._factory = backend_factory or self._cpu_backend
        self.settings = dict(settings)
        self.hardware = hardware or detect_hardware()
        self.backend = self._factory(self.settings, None)

    @staticmethod
    def _cpu_backend(settings, profile):
        config = GenerationConfig(context_size=getattr(profile, "context_size", 6144),
            max_answer_tokens=getattr(profile, "max_answer_tokens", 384),
            max_evidence_tokens=getattr(profile, "max_evidence_tokens", None),
            max_context_documents=getattr(profile, "max_context_documents", 6),
            max_context_blocks=getattr(profile, "max_context_blocks", 12),
            max_seeds_per_document=getattr(profile, "max_seeds_per_document", 3),
            temperature=getattr(profile, "temperature", .2))
        return LlamaCppBackend(settings["chat_model_path"], config,
            getattr(profile, "chat_template_kwargs", None))

    def prepare_activation(self, settings, profile=None):
        """Construct a replacement without disturbing the working backend."""
        self._validate_device(settings.get("chat_inference_device", "auto"))
        return self._factory(settings, profile)

    def commit_activation(self, settings, replacement):
        with self._lock:
            old = self.backend
            self.settings, self.backend = dict(settings), replacement
        if callable(getattr(old, "close", None)):
            try:
                old.close()
            except Exception:
                LOGGER.exception("Previous local inference backend could not be closed cleanly")

    def activate(self, settings, profile=None):
        replacement = self.prepare_activation(settings, profile)
        self.commit_activation(settings, replacement)

    def _validate_device(self, device):
        if device not in ALLOWED_DEVICES:
            raise InferenceDeviceError("invalid_inference_device")
        # PR22 deliberately ships no Intel inference backend. Adapter/probe
        # discovery must never make the CPU backend masquerade as GPU inference.
        if device == "intel_gpu":
            raise InferenceDeviceError("intel_gpu_unavailable")

    def set_device(self, device):
        self._validate_device(device)
        with self._lock:
            self.settings["chat_inference_device"] = device

    def close(self):
        with self._lock:
            backend = self.backend
        if callable(getattr(backend, "close", None)):
            backend.close()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def health(self):
        value = self.backend.health()
        requested = self.settings.get("chat_inference_device", "auto")
        profile = catalog_model(self.settings.get("chat_active_model_id"))
        accelerator_reason = self.hardware.accelerator_reason or "sycl_runtime_unavailable"
        fallback = accelerator_reason if requested != "cpu" else None
        value.update(active_model_id=self.settings.get("chat_active_model_id"),
                     active_model_name=profile.name if profile else value.get("model_name"),
                     inference_device_requested=requested,
                     inference_device_effective="cpu", fallback_reason=fallback,
                     intel_gpu_detected=self.hardware.intel_gpu_detected,
                     intel_gpu_name=self.hardware.intel_gpu_name,
                     accelerator_available=False,
                     accelerator_reason=accelerator_reason)
        return value
