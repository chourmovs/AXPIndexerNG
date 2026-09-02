from __future__ import annotations

import importlib.metadata
import inspect
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .cpu import detect_cpu
from .model import validate_gguf

RECOMMENDED_MODEL = "Qwen3-1.7B-Q4_K_M"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
NO_THINK_DIRECTIVE = "/no_think\n"


def build_chat_invocation(create_chat_completion, *, system_prompt, user_prompt, config, template_kwargs,
                          max_tokens=None):
    """Build an invocation compatible with the installed llama-cpp API before calling it."""
    parameters = inspect.signature(create_chat_completion).parameters
    supports_template_kwargs = "chat_template_kwargs" in parameters
    supports_kwargs = any(value.kind is inspect.Parameter.VAR_KEYWORD for value in parameters.values())
    non_thinking = bool(template_kwargs) and template_kwargs.get("enable_thinking") is False
    system_content = NO_THINK_DIRECTIVE + system_prompt if non_thinking and not supports_template_kwargs else system_prompt
    invocation = {
        "messages": [{"role": "system", "content": system_content},
                     {"role": "user", "content": user_prompt}],
        "max_tokens": config.max_answer_tokens if max_tokens is None else max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }
    if supports_template_kwargs and template_kwargs:
        invocation["chat_template_kwargs"] = template_kwargs
    if "repeat_penalty" in parameters or supports_kwargs:
        invocation["repeat_penalty"] = config.repeat_penalty
    return invocation, supports_template_kwargs, non_thinking and not supports_template_kwargs


class CpuIncompatibleError(OSError):
    """The packaged backend's declared ISA is unavailable to this process."""


class GenerationCancelled(Exception):
    """Native streamed generation exited after a cooperative cancel request."""


def classify_load_failure(exc, model_path=None):
    """Map native failures to stable, safe diagnostics (never a traceback/path)."""
    text = f"{exc!r} {exc}".lower()
    winerror = getattr(exc, "winerror", None)
    if isinstance(exc, CpuIncompatibleError):
        return {"failure_type": "backend_cpu_incompatible", "failure_code": "avx_unavailable",
                "failure_reason": "Local AI runtime is not compatible with this CPU.", "retryable": False}
    if winerror == -1073741795 or "-1073741795" in text or "0xc000001d" in text or "illegal instruction" in text:
        return {"failure_type": "backend_cpu_incompatible", "failure_code": "0xc000001d",
                "failure_reason": "Local AI runtime is not compatible with this CPU.", "retryable": False}
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return {"failure_type": "backend_missing", "failure_code": None,
                "failure_reason": "The local AI backend is not installed.", "retryable": False}
    if type(exc).__name__ == "TemplateSyntaxError" and type(exc).__module__.startswith("jinja2"):
        return {"failure_type": "model_template_incompatible", "failure_code": None,
                "failure_reason": "The selected model uses a chat template that is incompatible with this AXP runtime.",
                "retryable": False}
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
    max_evidence_tokens: int | None = None
    max_context_documents: int = 6
    max_context_blocks: int = 12
    max_seeds_per_document: int = 3
    safety_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float = 1.0
    n_gpu_layers: int = 0
    model_id: str | None = None
    reasoning_enabled: bool = False
    reasoning_budget_tokens: int | None = None
    reasoning_format: str | None = None
    min_visible_answer_tokens: int = 0


class LlamaCppBackend:
    """Single-instance, lazy, quiet, CPU-only GGUF backend."""

    def __init__(self, model_path, config=None, chat_template_kwargs=None):
        self.model_path = Path(model_path)
        self.config = config or GenerationConfig()
        self._model = None
        self._load_lock = threading.Lock()
        self._load_ms = None
        self._cpu_settings = {"n_threads": None, "n_threads_batch": None, "n_batch": None}
        self._model_state = "unloaded"
        self._failure = {}
        self.last_telemetry = {}
        self._chat_template_kwargs_supported = None
        self._no_think_compatibility = None
        self.chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs is not None else {}
        self.cpu = detect_cpu()
        self._progress_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._progress = self._new_progress()

    @staticmethod
    def _new_progress():
        return {"active": False, "phase": "idle", "sequence": 0, "started_monotonic": None,
                "elapsed_s": None, "time_to_first_token_ms": None, "generated_fragments": 0,
                "generated_characters": 0, "last_fragment_age_s": None, "finish_reason": None,
                "cancel_requested": False}

    def generation_progress(self):
        with self._progress_lock:
            value = dict(self._progress)
            now = time.perf_counter()
            if value["started_monotonic"] is not None:
                value["elapsed_s"] = max(0.0, now - value["started_monotonic"])
            last = value.pop("last_fragment_monotonic", None)
            value["last_fragment_age_s"] = None if last is None else max(0.0, now - last)
            return value

    def generation_active(self):
        with self._progress_lock:
            return bool(self._progress["active"])

    def request_cancel(self):
        with self._progress_lock:
            if not self._progress["active"]:
                return False
            self._cancel_event.set()
            self._progress.update(phase="cancel_requested", cancel_requested=True)
            return True

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
                "context_size": self.config.context_size, "recommended_model": RECOMMENDED_MODEL,
                "chat_template_kwargs_supported": self._chat_template_kwargs_supported,
                "no_think_compatibility": self._no_think_compatibility, **self._cpu_settings,
                **self._public_generation_health()}

    def _public_generation_health(self):
        progress = self.generation_progress()
        return {"generation_phase": progress["phase"], "generation_elapsed_s": progress["elapsed_s"],
                "time_to_first_token_ms": progress["time_to_first_token_ms"],
                "generated_fragments": progress["generated_fragments"],
                "generated_characters": progress["generated_characters"],
                "last_fragment_age_s": progress["last_fragment_age_s"]}

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
                            raise CpuIncompatibleError("CPU does not provide the AVX state required by this AXP runtime")
                        valid, reason = validate_gguf(self.model_path)
                        if not valid:
                            raise ValueError(reason)
                        from llama_cpp import Llama
                        self._model = Llama(model_path=str(self.model_path), n_ctx=self.config.context_size,
                                            n_gpu_layers=self.config.n_gpu_layers, verbose=False)
                        self._cpu_settings = {name: getattr(self._model, name, None)
                                              for name in ("n_threads", "n_threads_batch", "n_batch")}
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

    def generate(self, *, system_prompt, user_prompt, max_tokens=None):
        model = self.ensure_loaded()
        invocation, supported, compatibility = build_chat_invocation(
            model.create_chat_completion, system_prompt=system_prompt, user_prompt=user_prompt,
            config=self.config, template_kwargs=self.chat_template_kwargs, max_tokens=max_tokens,
        )
        self._chat_template_kwargs_supported = supported
        self._no_think_compatibility = compatibility
        # The qualified 0.3.24 runtime always takes this branch. The signature
        # guard only keeps lightweight legacy test doubles usable.
        supports_stream = "stream" in inspect.signature(model.create_chat_completion).parameters
        if supports_stream:
            invocation["stream"] = True
        self._cancel_event.clear()
        started = time.perf_counter()
        with self._progress_lock:
            self._progress = self._new_progress()
            self._progress.update(active=True, phase="waiting_first_token", started_monotonic=started)
        stream = None
        fragments, first_token, finish_reason = [], None, None
        try:
            stream = model.create_chat_completion(**invocation)
            if not supports_stream:
                content = (stream.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                stream = iter([{"choices": [{"delta": {"content": content}, "finish_reason":
                                                     (stream.get("choices") or [{}])[0].get("finish_reason")}]}])
            for chunk in stream:
                if self._cancel_event.is_set():
                    raise GenerationCancelled
                choice = (chunk.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    if first_token is None:
                        first_token = now
                    fragments.append(content)
                    with self._progress_lock:
                        self._progress.update(phase="generating", sequence=self._progress["sequence"] + 1,
                                              generated_fragments=self._progress["generated_fragments"] + 1,
                                              generated_characters=self._progress["generated_characters"] + len(content),
                                              time_to_first_token_ms=(first_token - started) * 1000,
                                              last_fragment_monotonic=now)
                if self._cancel_event.is_set():
                    raise GenerationCancelled
            ended = time.perf_counter()
            answer = THINK_RE.sub("", "".join(fragments)).strip()
            tokenizer = getattr(model, "tokenize", None)
            completion = (len(tokenizer(answer.encode("utf-8"), add_bos=False, special=True))
                          if answer and callable(tokenizer) else (len(answer.split()) if answer else None))
            generation_ms = (ended - started) * 1000
            decode_ms = (ended - first_token) * 1000 if first_token is not None else None
            self.last_telemetry = {
                "time_to_first_token_ms": None if first_token is None else (first_token - started) * 1000,
                "generation_ms": generation_ms, "decode_ms": decode_ms, "completion_tokens": completion,
                "decode_tokens_per_second": completion / (decode_ms / 1000) if completion is not None and decode_ms else None,
                "overall_tokens_per_second": completion / (generation_ms / 1000) if completion is not None and generation_ms else None,
                "generated_characters": sum(map(len, fragments)), "generated_fragments": len(fragments),
                "finish_reason": finish_reason, **self._cpu_settings}
            with self._progress_lock:
                self._progress.update(active=False, phase="completed", finish_reason=finish_reason)
            return answer
        except GenerationCancelled:
            if stream is not None and callable(getattr(stream, "close", None)):
                stream.close()
            with self._progress_lock:
                self._progress.update(active=False, phase="cancelled", finish_reason="cancelled")
            raise
        except Exception:
            with self._progress_lock:
                self._progress.update(active=False, phase="failed")
            raise

    def close(self):
        with self._load_lock:
            model, self._model = self._model, None
            if model is not None and callable(getattr(model, "close", None)):
                model.close()
            self._model_state = "unloaded"
