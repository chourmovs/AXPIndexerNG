"""Hot-switchable local inference runtime."""
import logging
import threading
import time

from .hardware import detect_hardware
from .accelerator_manager import AcceleratorManager
from .intel_sycl_backend import DEVICE_LOST_TYPE, IntelSyclBackend, IntelSyclError
from .llama_cpp_backend import GenerationCancelled, GenerationConfig, LlamaCppBackend
from .model_catalog import catalog_model

ALLOWED_DEVICES = {"auto", "cpu", "intel_gpu"}
LOGGER = logging.getLogger("axp_client")


class InferenceDeviceError(ValueError):
    pass


class _PendingIntelBackend:
    """Non-CPU placeholder used until bounded, on-demand qualification."""
    def __init__(self, profile, reason):
        self.config = GenerationConfig(context_size=getattr(profile, "context_size", 6144),
            max_answer_tokens=getattr(profile, "max_answer_tokens", 384),
            max_evidence_tokens=getattr(profile, "max_evidence_tokens", None),
            max_context_documents=getattr(profile, "max_context_documents", 6),
            max_context_blocks=getattr(profile, "max_context_blocks", 12),
            max_seeds_per_document=getattr(profile, "max_seeds_per_document", 3))
        self.reason = reason

    def health(self):
        return {"available": True, "backend": "intel_sycl", "model_state": "unloaded",
                "accelerator_state": "probe_timeout" if self.reason == "intel_sycl_probe_timeout" else "unverified",
                "failure_type": self.reason, "retryable": True, "gpu_offload_confirmed": False}

    def close(self):
        return None


class InferenceRuntimeManager:
    def __init__(self, settings, backend_factory=None, hardware=None, intel_backend_factory=None,
                 accelerator_manager=None, hardware_probe=None):
        self._lock = threading.RLock()
        self._factory = backend_factory or self._cpu_backend
        self._intel_factory = intel_backend_factory
        self._hardware_probe = hardware_probe or detect_hardware
        self.settings = dict(settings)
        self._device_loss_count = 0; self._last_device_loss_at = None
        self._device_recovery_attempted = False; self._device_recovery_succeeded = False
        self.accelerators = accelerator_manager or AcceleratorManager(__import__("axp_core.runtime", fromlist=["data_dir"]).data_dir())
        server = self.accelerators.server_path()
        self.hardware = hardware or detect_hardware(server, self.accelerators.runtime_root)
        LOGGER.info("Intel probe completed runtime=%s device_id=%s device_name=%s returncode=%s duration_ms=%s failure_type=%s",
                    "b10516", self.hardware.sycl_device_id, self.hardware.sycl_device_name,
                    self.hardware.sycl_probe_returncode, self.hardware.sycl_probe_duration_ms,
                    self.hardware.sycl_probe_error)
        self.backend = self._make_backend(self.settings, None)

    def _make_backend(self, settings, profile):
        requested = settings.get("chat_inference_device", "auto")
        if requested == "intel_gpu" and not self.hardware.intel_gpu_available:
            return _PendingIntelBackend(profile, self.hardware.sycl_probe_error or self.hardware.accelerator_reason)
        if requested in {"intel_gpu", "auto"} and self.hardware.intel_gpu_available:
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
                self.accelerators.server_path(), getattr(profile, "chat_template_kwargs", None),
                sycl_device_id=self.hardware.sycl_device_id,
                sycl_device_name=self.hardware.sycl_device_name,
                diagnostic=bool(settings.get("intel_diagnostic")))
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
        # Explicit Intel selection is retained even while qualification is
        # inconclusive; generation will either qualify Intel or fail explicitly.

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

    def _qualify_intel(self):
        with self._lock:
            if not isinstance(self.backend, _PendingIntelBackend):
                return
            self.hardware = self._hardware_probe(self.accelerators.server_path(), self.accelerators.runtime_root,
                                                 probe_timeout=30)
            if not self.hardware.intel_gpu_available:
                reason = self.hardware.sycl_probe_error or self.hardware.accelerator_reason or "intel_gpu_unavailable"
                raise InferenceDeviceError(reason)
            self.backend = self._make_backend(self.settings, catalog_model(self.settings.get("chat_active_model_id")))

    def ensure_loaded(self):
        self._qualify_intel()
        return self.backend.ensure_loaded()

    def count_tokens(self, text):
        self._qualify_intel()
        return self.backend.count_tokens(text)

    def generate(self, *, system_prompt, user_prompt, max_tokens=None):
        """Generate with at most one fresh, Intel-only recovery after DEVICE_LOST."""
        self._qualify_intel()
        with self._lock:
            backend = self.backend; model_id = self.settings.get("chat_active_model_id")
        try:
            return backend.generate(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
        except GenerationCancelled:
            raise
        except IntelSyclError as exc:
            if str(exc) != DEVICE_LOST_TYPE or not isinstance(backend, IntelSyclBackend):
                raise
            self._device_loss_count += 1
            self._last_device_loss_at = backend.last_device_loss_at
            self._device_recovery_attempted = True; self._device_recovery_succeeded = False
            LOGGER.warning("Intel DEVICE_LOST recovery attempt=1 model_id=%s device_id=%s",
                           model_id, backend.sycl_device_id)
            time.sleep(0.1)
            with self._lock:
                if self.backend is not backend or self.settings.get("chat_active_model_id") != model_id:
                    LOGGER.warning("Intel DEVICE_LOST recovery failed failure_type=model_runtime_changed")
                    raise IntelSyclError("intel_gpu_recovery_model_changed") from exc
                recovery_settings = dict(self.settings); recovery_settings["chat_inference_device"] = "intel_gpu"
                replacement = self._make_backend(recovery_settings, catalog_model(model_id))
                self.backend = replacement
            backend.close()
            try:
                replacement.ensure_loaded()
                answer = replacement.generate(system_prompt=system_prompt, user_prompt=user_prompt,
                                              max_tokens=max_tokens)
            except GenerationCancelled:
                raise
            except IntelSyclError as recovery_error:
                LOGGER.warning("Intel DEVICE_LOST recovery failed failure_type=%s", recovery_error)
                raise
            self._device_recovery_succeeded = True
            LOGGER.info("Intel DEVICE_LOST recovery succeeded new_session_id=%s offloaded_layers=%s",
                        replacement.session_id, replacement.offloaded_layers)
            return answer

    def health(self):
        value = self.backend.health()
        requested = self.settings.get("chat_inference_device", "auto")
        profile = catalog_model(self.settings.get("chat_active_model_id"))
        accelerator_reason = self.hardware.accelerator_reason or None
        fallback = accelerator_reason if requested != "cpu" else None
        if requested == "intel_gpu":
            effective = ("intel_gpu" if value.get("gpu_offload_confirmed") else
                         "intel_gpu" if value.get("backend") == "intel_sycl" else "none")
        elif requested == "auto" and value.get("backend") == "intel_sycl":
            effective = "intel_gpu" if value.get("gpu_offload_confirmed") else "intel_gpu"
        else:
            effective = "cpu"
        value.update(active_model_id=self.settings.get("chat_active_model_id"),
                     active_model_name=profile.name if profile else value.get("model_name"),
                     inference_device_requested=requested,
                     inference_device_effective=effective,
                     fallback_reason=(fallback if requested == "auto" and effective == "cpu" else None),
                     intel_gpu_detected=self.hardware.intel_gpu_detected,
                     intel_gpu_name=self.hardware.intel_gpu_name,
                     intel_gpu_vendor_id=self.hardware.intel_gpu_vendor_id,
                     intel_gpu_device_id=self.hardware.intel_gpu_device_id,
                     sycl_runtime_installed=self.hardware.sycl_runtime_installed,
                     sycl_probe_ok=self.hardware.sycl_probe_ok,
                     sycl_device_id=value.get("sycl_device_id") or self.hardware.sycl_device_id,
                     sycl_device_name=value.get("sycl_device_name") or self.hardware.sycl_device_name,
                     sycl_device_count=self.hardware.sycl_device_count,
                     sycl_probe_error=self.hardware.sycl_probe_error,
                     sycl_probe_returncode=self.hardware.sycl_probe_returncode,
                     sycl_probe_duration_ms=self.hardware.sycl_probe_duration_ms,
                     sycl_probe_stdout_excerpt=self.hardware.sycl_probe_stdout_excerpt,
                     sycl_probe_stderr_excerpt=self.hardware.sycl_probe_stderr_excerpt,
                     intel_gpu_available=self.hardware.intel_gpu_available,
                     accelerator_available=self.hardware.intel_gpu_available,
                     accelerator_reason=accelerator_reason)
        value.update(device_loss_count=self._device_loss_count,
                     last_device_loss_at=self._last_device_loss_at,
                     last_device_loss_code=("UR_RESULT_ERROR_DEVICE_LOST" if self._last_device_loss_at else None),
                     device_recovery_attempted=self._device_recovery_attempted,
                     device_recovery_succeeded=self._device_recovery_succeeded)
        return value
