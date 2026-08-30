from __future__ import annotations

import importlib.metadata
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .model import validate_gguf
from .cpu import detect_cpu

RECOMMENDED_MODEL = "Qwen3-1.7B-Q4_K_M"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def classify_load_failure(exc, model_path=None):
    """Map native failures to stable, safe diagnostics (never a traceback/path)."""
    text = f"{exc!r} {exc}".lower()
    winerror = getattr(exc, "winerror", None)
    if winerror == -1073741795 or "-1073741795" in text or "0xc000001d" in text or "illegal instruction" in text:
        return {"failure_type": "backend_cpu_incompatible", "failure_code": "0xc000001d",
                "failure_reason": "Local AI runtime is not compatible with this CPU.", "retryable": False}
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return {"failure_type": "backend_missing", "failure_code": None,
                "failure_reason": "The local AI backend is not installed.", "retryable": False}
    if model_path is not None and not Path(model_path).is_file():
        kind, retryable = "model_missing", False
    elif isinstance(exc, ValueError):
        kind, retryable = "model_invalid", False
    else:
        kind, retryable = "model_load_failed", True
    return {"failure_type": kind, "failure_code": None,
            "failure_reason": "The local answer model could not be loaded.", "retryable": retryable}


@dataclass(frozen=True)
class GenerationConfig:
    context_size: int = 6144
    max_answer_tokens: int = 384
    safety_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    n_gpu_layers: int = 0


class LlamaCppBackend:
    """Single-instance, lazy, quiet, CPU-only GGUF backend."""

    def __init__(self, model_path, config=None, chat_template_kwargs=None):
        self.model_path = Path(model_path)
        self.config = config or GenerationConfig()
        self._model = None
        self._load_lock = threading.Lock()
        self._load_ms = None
        self._model_state = "unloaded"
        self._failure = {}
        self.last_telemetry = {}
        self.chat_template_kwargs = chat_template_kwargs or {"enable_thinking": False}
        self.cpu = detect_cpu()

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
            reason = self._failure.get("failure_type", "model_load_failed")
        elif not self.cpu.runtime_cpu_compatible:
            reason = "backend_cpu_incompatible"
        elif not valid:
            reason = "model_missing" if reason == "model_missing" else "model_invalid"
        return {"available": bool(valid and version and self.cpu.runtime_cpu_compatible and not self._load_failed), "reason": reason or (None if version else "backend_missing"),
                "backend": "llama_cpp", "backend_version": version, "model_configured": self.model_path.is_file(),
                "model_valid": valid, "model_installed": valid, "model_selected": True,
                "model_loaded": self.loaded, "model_state": self._model_state,
                "model_load_ms": self._load_ms, "last_model_load_ms": self._load_ms,
                "model_name": self.model_path.stem, **self.cpu.public(), **self._failure,
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
                        if not self.cpu.runtime_cpu_compatible:
                            raise OSError("CPU does not provide the AVX state required by this AXP runtime")
                        valid, reason = validate_gguf(self.model_path)
                        if not valid:
                            raise ValueError(reason)
                        from llama_cpp import Llama
                        self._model = Llama(model_path=str(self.model_path), n_ctx=self.config.context_size,
                                            n_gpu_layers=self.config.n_gpu_layers, verbose=False)
                    except Exception as exc:
                        self._model_state = "failed"
                        self._failure = classify_load_failure(exc, self.model_path)
                        self._failure.update(
                                        {"failure_message": "Failed to load GGUF model",
                                         "failed_at_ms": int(time.time() * 1000)}
                        )
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
            chat_template_kwargs=self.chat_template_kwargs,
        )
        elapsed = (time.perf_counter() - started) * 1000
        usage = result.get("usage", {})
        completion = int(usage.get("completion_tokens") or 0)
        self.last_telemetry = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
                               "generation_ms": elapsed, "tokens_per_second": completion / (elapsed / 1000) if elapsed else None}
        return THINK_RE.sub("", result["choices"][0]["message"]["content"] or "").strip()

    def close(self):
        with self._load_lock:
            model, self._model = self._model, None
            if model is not None and callable(getattr(model, "close", None)):
                model.close()
            self._model_state = "unloaded"
