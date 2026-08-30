"""Hot-switchable local inference runtime."""
import threading

from .llama_cpp_backend import GenerationConfig, LlamaCppBackend


class InferenceRuntimeManager:
    def __init__(self, settings, backend_factory=None):
        self._lock = threading.RLock()
        self._factory = backend_factory or self._cpu_backend
        self.settings = dict(settings)
        self.backend = self._factory(self.settings, None)

    @staticmethod
    def _cpu_backend(settings, profile):
        config = GenerationConfig(context_size=getattr(profile, "context_size", 6144),
            max_answer_tokens=getattr(profile, "max_answer_tokens", 384),
            temperature=getattr(profile, "temperature", .2))
        return LlamaCppBackend(settings["chat_model_path"], config,
            getattr(profile, "chat_template_kwargs", None))

    @property
    def busy(self):
        return False

    def activate(self, settings, profile=None):
        with self._lock:
            old = self.backend
            replacement = self._factory(settings, profile)
            if callable(getattr(old, "close", None)): old.close()
            self.settings, self.backend = dict(settings), replacement

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def health(self):
        value = self.backend.health()
        requested = self.settings.get("chat_inference_device", "auto")
        value.update(active_model_id=self.settings.get("chat_active_model_id"),
                     active_model_name=value.get("model_name"),
                     inference_device_requested=requested,
                     inference_device_effective="cpu", accelerator_available=False)
        return value
