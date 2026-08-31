"""Hot-switchable local inference runtime."""
import logging
import threading

from .hardware import detect_hardware
from .accelerator_manager import AcceleratorManager
from .intel_sycl_backend import IntelSyclBackend
from .llama_cpp_backend import GenerationConfig, LlamaCppBackend
from .model_catalog import catalog_model

ALLOWED_DEVICES = {"auto", "cpu", "intel_gpu"}
LOGGER = logging.getLogger("axp_client")


class InferenceDeviceError(ValueError):
    pass


class InferenceRuntimeManager:
    def __init__(self, settings, backend_factory=None, hardware=None, intel_backend_factory=None,
                 accelerator_manager=None):
        self._lock = threading.RLock()
        self._factory = backend_factory or self._cpu_backend
        self._intel_factory = intel_backend_factory
        self.settings = dict(settings)
        self.accelerators = accelerator_manager or AcceleratorManager(__import__("axp_core.runtime", fromlist=["data_dir"]).data_dir())
        server = self.accelerators.server_path()
        self.hardware = hardware or detect_hardware(server)
        self.backend = self._make_backend(self.settings, None)

    def _make_backend(self, settings, profile):
        requested = settings.get("chat_inference_device", "auto")
        if requested == "intel_gpu" and self.hardware.intel_gpu_available:
            if self._intel_factory:
                return self._intel_factory(settings, profile)
            config = GenerationConfig(context_size=getattr(profile, "context_size", 6144),
                max_answer_tokens=getattr(profile, "max_answer_tokens", 384),
                max_evidence_tokens=getattr(profile, "max_evidence_tokens", None),
                max_context_documents=getattr(profile, "max_context_documents", 6),
                max_context_blocks=getattr(profile, "max_context_blocks", 12),
                max_seeds_per_document=getattr(profile, "max_seeds_per_document", 3),
                temperature=getattr(profile, "temperature", .2),
                top_p=getattr(profile, "top_p", .8), top_k=getattr(profile, "top_k", 20),
                repeat_penalty=getattr(profile, "repeat_penalty", 1.0))
            return IntelSyclBackend(settings["chat_model_path"], config, self.accelerators.runtime_root,
                self.accelerators.server_path(), getattr(profile, "chat_template_kwargs", None))
        return self._factory(settings, profile)

    @staticmethod
    def _cpu_backend(settings, profile):
        config = GenerationConfig(context_size=getattr(profile, "context_size", 6144),
            max_answer_tokens=getattr(profile, "max_answer_tokens", 384),
            max_evidence_tokens=getattr(profile, "max_evidence_tokens", None),
            max_context_documents=getattr(profile, "max_context_documents", 6),
            max_context_blocks=getattr(profile, "max_context_blocks", 12),
            max_seeds_per_document=getattr(profile, "max_seeds_per_document", 3),
            temperature=getattr(profile, "temperature", .2),
            top_p=getattr(profile, "top_p", .8), top_k=getattr(profile, "top_k", 20),
            repeat_penalty=getattr(profile, "repeat_penalty", 1.0))
        return LlamaCppBackend(settings["chat_model_path"], config,
            getattr(profile, "chat_template_kwargs", None))

    def prepare_activation(self, settings, profile=None):
        """Construct a replacement without disturbing the working backend."""
        self._validate_device(settings.get("chat_inference_device", "auto"))
        return self._make_backend(settings, profile)

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
        if device == "intel_gpu" and not self.hardware.intel_gpu_available:
            raise InferenceDeviceError("intel_gpu_unavailable")

    def set_device(self, device):
        self._validate_device(device)
        settings = dict(self.settings); settings["chat_inference_device"] = device
        replacement = self._make_backend(settings, catalog_model(settings.get("chat_active_model_id")))
        self.commit_activation(settings, replacement)

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
        accelerator_reason = self.hardware.accelerator_reason or None
        fallback = accelerator_reason if requested != "cpu" else None
        effective = "intel_gpu" if requested == "intel_gpu" and value.get("gpu_offload_confirmed") else "cpu"
        value.update(active_model_id=self.settings.get("chat_active_model_id"),
                     active_model_name=profile.name if profile else value.get("model_name"),
                     inference_device_requested=requested,
                     inference_device_effective=effective, fallback_reason=(None if effective == "intel_gpu" else fallback),
                     intel_gpu_detected=self.hardware.intel_gpu_detected,
                     intel_gpu_name=self.hardware.intel_gpu_name,
                     intel_gpu_vendor_id=self.hardware.intel_gpu_vendor_id,
                     intel_gpu_device_id=self.hardware.intel_gpu_device_id,
                     sycl_runtime_installed=self.hardware.sycl_runtime_installed,
                     sycl_probe_ok=self.hardware.sycl_probe_ok,
                     sycl_device_name=value.get("sycl_device_name") or self.hardware.sycl_device_name,
                     sycl_device_count=self.hardware.sycl_device_count,
                     sycl_probe_error=self.hardware.sycl_probe_error,
                     intel_gpu_available=self.hardware.intel_gpu_available,
                     accelerator_available=self.hardware.intel_gpu_available,
                     accelerator_reason=accelerator_reason)
        return value
