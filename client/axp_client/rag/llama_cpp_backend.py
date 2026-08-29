import threading
from pathlib import Path

CONTEXT_SIZE = 8192
MAX_ANSWER_TOKENS = 512
TEMPERATURE = 0.1
GPU_LAYERS = 0


class LlamaCppBackend:
    """Lazy local GGUF backend. Importing AXP never imports llama_cpp."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self):
        return self._model is not None

    def health(self):
        configured = bool(str(self.model_path))
        if not configured or not self.model_path.is_file():
            return {"available": False, "reason": "model_missing", "backend": "llama_cpp", "model_configured": configured,
                    "model_loaded": False}
        try:
            import importlib.util

            installed = importlib.util.find_spec("llama_cpp") is not None
        except (ImportError, ValueError):
            installed = False
        return {"available": installed, "reason": None if installed else "backend_missing", "backend": "llama_cpp",
                "model_configured": True, "model_loaded": self.loaded}

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if not self.model_path.is_file():
                        raise FileNotFoundError("chat model is not provisioned")
                    from llama_cpp import Llama  # lazy optional, local-only dependency

                    self._model = Llama(model_path=str(self.model_path), n_ctx=CONTEXT_SIZE, n_gpu_layers=GPU_LAYERS)
        return self._model

    def generate(self, *, system_prompt, user_prompt):
        result = self._load().create_chat_completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=TEMPERATURE,
        )
        return result["choices"][0]["message"]["content"]
