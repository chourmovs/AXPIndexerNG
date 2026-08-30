from __future__ import annotations

import importlib.metadata
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .model import validate_gguf

RECOMMENDED_MODEL = "Qwen3-4B-Q4_K_M"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class GenerationConfig:
    context_size: int = 8192
    max_answer_tokens: int = 512
    safety_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    n_gpu_layers: int = 0


class LlamaCppBackend:
    """Single-instance, lazy, quiet, CPU-only GGUF backend."""

    def __init__(self, model_path, config=None):
        self.model_path = Path(model_path)
        self.config = config or GenerationConfig()
        self._model = None
        self._load_lock = threading.Lock()
        self._load_ms = None
        self._model_state = "unloaded"
        self._failure = {}
        self.last_telemetry = {}

    @property
    def loaded(self):
        return self._model is not None

    def health(self):
        valid, reason = validate_gguf(self.model_path)
        try:
            version = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if self._model_state == "failed":
            reason = "model_load_failed"
        return {"available": bool(valid and version and not self._load_failed), "reason": reason or (None if version else "backend_missing"),
                "backend": "llama_cpp", "backend_version": version, "model_configured": self.model_path.is_file(),
                "model_valid": valid, "model_loaded": self.loaded, "model_state": self._model_state,
                "model_load_ms": self._load_ms, "last_model_load_ms": self._load_ms,
                "model_name": self.model_path.stem, **self._failure,
                "context_size": self.config.context_size, "recommended_model": RECOMMENDED_MODEL}

    @property
    def _load_failed(self):
        return self._model_state == "failed"

    def ensure_loaded(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if self._model_state == "failed":
                        raise RuntimeError("model_load_failed")
                    started = time.perf_counter()
                    self._model_state = "loading"
                    try:
                        valid, reason = validate_gguf(self.model_path)
                        if not valid:
                            raise ValueError(reason)
                        from llama_cpp import Llama
                        self._model = Llama(model_path=str(self.model_path), n_ctx=self.config.context_size,
                                            n_gpu_layers=self.config.n_gpu_layers, verbose=False)
                    except Exception as exc:
                        self._model_state = "failed"
                        self._failure = {"failure_type": type(exc).__name__,
                                         "failure_message": "Failed to load GGUF model",
                                         "failed_at_ms": int(time.time() * 1000)}
                        raise
                    self._load_ms = (time.perf_counter() - started) * 1000
                    self._model_state = "loaded"
                    self._failure = {}
        return self._model

    def retry_load(self):
        with self._load_lock:
            if self._model is not None:
                return
            self._model_state = "unloaded"
            self._failure = {}
            self._load_ms = None

    def count_tokens(self, text):
        return len(self.ensure_loaded().tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def context_window(self):
        return self.config.context_size

    def generate(self, *, system_prompt, user_prompt):
        started = time.perf_counter()
        model = self.ensure_loaded()
        result = model.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=self.config.max_answer_tokens, temperature=self.config.temperature,
            top_p=self.config.top_p, top_k=self.config.top_k,
            chat_template_kwargs={"enable_thinking": False},
        )
        elapsed = (time.perf_counter() - started) * 1000
        usage = result.get("usage", {})
        completion = int(usage.get("completion_tokens") or 0)
        self.last_telemetry = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                               "generation_ms": elapsed, "tokens_per_second": completion / (elapsed / 1000) if elapsed else None}
        return THINK_RE.sub("", result["choices"][0]["message"]["content"] or "").strip()
