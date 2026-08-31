"""Isolated localhost transport for the pinned llama.cpp Intel SYCL server."""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import tempfile
import logging
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from axp_core.runtime import runtime_paths

from .accelerator_catalog import INTEL_SYCL
from .llama_cpp_backend import GenerationCancelled, NO_THINK_DIRECTIVE, THINK_RE
from .model import validate_gguf
from .sycl_probe import DEVICE_SELECTOR, INTEL_GPU_RE, child_environment

LOOPBACK = "127.0.0.1"
INTEL_LOAD_WARN_AFTER_S = 120
INTEL_LOAD_STALL_WARN_AFTER_S = 240
INTEL_LOAD_HARD_TIMEOUT_S = 600
INTEL_LOAD_ACTIVITY_STALE_S = 120
OFFLOAD_RE = re.compile(r"offload(?:ed|ing)?\s+(\d+)(?:\s*/\s*|\s+of\s+)(\d+)\s+layers?", re.I)
BUFFER_RE = re.compile(r"(SYCL\d*|GPU|CPU|model).*?buffer size\s*=\s*([\d.]+)\s*(KiB|MiB|GiB|KB|MB|GB|B)", re.I)
MAX_GPU_MARKERS = 32
LOGGER = logging.getLogger("axp_client")


class IntelSyclError(RuntimeError):
    pass


def parse_sse(lines):
    """Parse SSE events from byte or text lines, including a final unterminated event."""
    event = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line:
            if line.startswith("data:"):
                event.append(line[5:].lstrip())
            continue
        if event:
            yield "\n".join(event)
            event.clear()
    if event:
        yield "\n".join(event)


class IntelSyclBackend:
    def __init__(self, model_path, config, runtime_dir, server_path=None, chat_template_kwargs=None,
                 popen=subprocess.Popen, urlopen=urllib.request.urlopen, load_timeout=INTEL_LOAD_HARD_TIMEOUT_S,
                 monotonic=time.monotonic, sleep=time.sleep, sycl_device_id=None, sycl_device_name=None,
                 diagnostic=False):
        self.model_path, self.config = Path(model_path), config
        self.runtime_dir = Path(runtime_dir).resolve()
        self.server_path = Path(server_path or self.runtime_dir / "llama-server.exe").resolve()
        self.chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs is not None else {}
        self._popen, self._urlopen, self.load_timeout = popen, urlopen, load_timeout
        self._monotonic, self._sleep = monotonic, sleep
        self._process = None; self._port = None; self._model_state = "unloaded"; self._load_ms = None
        self._load_lock = threading.Lock(); self._cancel_event = threading.Event(); self._progress_lock = threading.Lock()
        self._progress = self._new_progress(); self._failure = {}; self.last_telemetry = {}
        self.gpu_offload_requested = True; self.gpu_offload_confirmed = False
        self.offloaded_layers = self.total_layers = None; self._diagnostic = []
        self.sycl_device_id, self.sycl_device_name = sycl_device_id, sycl_device_name
        self.diagnostic = diagnostic; self.verbosity = 5 if diagnostic else 4
        self.session_id = None; self._session_result = "not_started"
        self.gpu_buffer_bytes = self.cpu_buffer_bytes = None
        self.native_gpu_markers = []
        self._load_cancel = threading.Event(); self._load_progress_lock = threading.Lock()
        self._load_progress = self._new_load_progress(); self._api_key = None
        self._auth_dir = self._auth_file = None; self._reader_thread = None

    @staticmethod
    def _new_progress():
        return {"active": False, "phase": "idle", "sequence": 0, "started_monotonic": None,
                "elapsed_s": None, "time_to_first_token_ms": None, "generated_fragments": 0,
                "generated_characters": 0, "last_fragment_age_s": None, "finish_reason": None,
                "cancel_requested": False, "prompt_total": None, "prompt_cached": None,
                "prompt_processed": None, "prompt_progress_time_ms": None}

    @staticmethod
    def _new_load_progress():
        return {"active": False, "phase": "idle", "started_monotonic": None,
                "last_native_activity_monotonic": None, "last_health_activity_monotonic": None,
                "native_lines_seen": 0, "model_path_configured": False, "sidecar_pid": None,
                "health_reachable": False, "health_status": "unreachable", "health_http_status": None,
                "slow_warning": False, "suspected_stall": False, "cancel_requested": False,
                "failure_type": None, "failure_reason": None}

    def model_load_active(self):
        with self._load_progress_lock: return self._load_progress["active"]

    def model_load_progress(self):
        with self._load_progress_lock: value = dict(self._load_progress)
        now = self._monotonic(); start = value.get("started_monotonic")
        value["elapsed_s"] = None if start is None else now - start
        for name in ("native", "health"):
            stamp = value.get(f"last_{name}_activity_monotonic")
            value[f"last_{name}_activity_age_s"] = None if stamp is None else now - stamp
        value.update(gpu_offload_requested=True, gpu_offload_confirmed=self.gpu_offload_confirmed,
                     offloaded_layers=self.offloaded_layers, total_layers=self.total_layers,
                     gpu_buffer_bytes=self.gpu_buffer_bytes, cpu_buffer_bytes=self.cpu_buffer_bytes,
                     sycl_device_id=self.sycl_device_id, sycl_device_name=self.sycl_device_name)
        return value

    def request_load_cancel(self):
        if not self.model_load_active(): return False
        self._load_cancel.set()
        with self._load_progress_lock: self._load_progress.update(cancel_requested=True)
        return True

    @property
    def loaded(self): return self._process is not None and self._process.poll() is None and self._model_state == "loaded"

    def generation_active(self):
        with self._progress_lock: return self._progress["active"]

    def generation_progress(self):
        with self._progress_lock:
            value = dict(self._progress); now = time.perf_counter()
            if value["started_monotonic"] is not None: value["elapsed_s"] = now - value["started_monotonic"]
            last = value.pop("last_fragment_monotonic", None)
            value["last_fragment_age_s"] = None if last is None else now - last
            return value

    def request_cancel(self):
        if self.request_load_cancel(): return True
        with self._progress_lock:
            if not self._progress["active"]: return False
            self._cancel_event.set(); self._progress.update(phase="cancel_requested", cancel_requested=True); return True

    def health(self):
        valid, reason = validate_gguf(self.model_path)
        if self._model_state == "failed": reason = self._failure.get("failure_type")
        progress = self.generation_progress()
        load = self.model_load_progress()
        return {"available": valid, "reason": reason, "backend": "intel_sycl",
                "backend_version": INTEL_SYCL.tag, "accelerator_backend": "sycl", "accelerator_runtime": INTEL_SYCL.tag,
                "model_configured": self.model_path.is_file(), "model_valid": valid, "model_loaded": self.loaded,
                "model_state": self._model_state, "model_load_ms": self._load_ms, "context_size": self.config.context_size,
                "sidecar_pid": getattr(self._process, "pid", None), "sidecar_host": LOOPBACK, "sidecar_port": self._port,
                "sycl_device_name": self.sycl_device_name, "sycl_device_selector": DEVICE_SELECTOR,
                "sycl_device_id": self.sycl_device_id, "sidecar_session_id": self.session_id,
                "native_verbosity": self.verbosity,
                "gpu_offload_requested": True, "gpu_offload_confirmed": self.gpu_offload_confirmed,
                "offloaded_layers": self.offloaded_layers, "total_layers": self.total_layers,
                "gpu_buffer_bytes": self.gpu_buffer_bytes, "cpu_buffer_bytes": self.cpu_buffer_bytes,
                "native_gpu_markers": list(self.native_gpu_markers),
                "accelerator_state": "confirmed" if self.gpu_offload_confirmed else "unconfirmed",
                "generation_phase": progress["phase"], "generation_elapsed_s": progress["elapsed_s"],
                "time_to_first_token_ms": progress["time_to_first_token_ms"],
                "generated_fragments": progress["generated_fragments"],
                "generated_characters": progress["generated_characters"],
                "model_load_active": load["active"], "model_load_phase": load["phase"],
                "model_load_elapsed_s": load["elapsed_s"],
                "model_load_last_activity_age_s": load["last_native_activity_age_s"],
                "model_load_slow": load["slow_warning"],
                "model_load_suspected_stall": load["suspected_stall"],
                "sidecar_running": self._process is not None and self._process.poll() is None, **self._failure}

    def _record_native_evidence(self, line):
        """Consume one native line. Positive proof is sticky for this model session."""
        text, lower = line.strip(), line.lower()
        strong = False
        match = OFFLOAD_RE.search(line)
        if match:
            self.offloaded_layers, self.total_layers = map(int, match.groups())
            strong = self.offloaded_layers > 0
        if INTEL_GPU_RE.search(line) and re.search(r"(?:SYCL|Level.Zero|device)", line, re.I):
            self.sycl_device_name = text[:300]
        buffer = BUFFER_RE.search(line)
        if buffer:
            scale = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024**2, "MIB": 1024**2,
                     "GB": 1024**3, "GIB": 1024**3}[buffer.group(3).upper()]
            size = int(float(buffer.group(2)) * scale)
            if re.search(r"SYCL|GPU", buffer.group(1), re.I):
                self.gpu_buffer_bytes = size
                strong = strong or size > 0
            else:
                self.cpu_buffer_bytes = size
        explicit = bool(re.search(r"offloading (?:\d+ repeating layers|output layer) to gpu", line, re.I))
        strong = strong or explicit
        if match or buffer or explicit or ("selected" in lower and "sycl" in lower and "device" in lower):
            self.native_gpu_markers.append(text[:500])
            self.native_gpu_markers = self.native_gpu_markers[-MAX_GPU_MARKERS:]
        # Never assign False here: absence on a subsequent line is not evidence.
        if strong:
            self.gpu_offload_confirmed = True

    def _endpoint(self, path): return f"http://{LOOPBACK}:{self._port}{path}"

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind((LOOPBACK, 0)); return sock.getsockname()[1]

    def _log_reader(self, stream):
        log = runtime_paths()["logs"] / "intel-sycl.log"
        handler = RotatingFileHandler(log, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        try:
            stamp = datetime.now(timezone.utc).isoformat()
            header = ("=== AXP INTEL SESSION START ===\n" f"timestamp={stamp}\n"
                f"session_id={self.session_id}\nruntime={INTEL_SYCL.tag}\nmodel={self.model_path.stem}\n"
                f"pid={getattr(self._process, 'pid', None)}\nprobe_device_id={self.sycl_device_id}\n"
                f"probe_device_name={self.sycl_device_name}\nverbosity={self.verbosity}\n"
                "gpu_layers=all\nsplit_mode=none\n=== NATIVE LLAMA OUTPUT ===\n")
            handler.stream.write(header); handler.flush()
            for line in iter(stream.readline, ""):
                handler.stream.write(line); handler.flush(); self._diagnostic.append(line[-1000:])
                self._diagnostic = self._diagnostic[-200:]
                with self._load_progress_lock:
                    self._load_progress["last_native_activity_monotonic"] = self._monotonic()
                    self._load_progress["native_lines_seen"] += 1
                lower = line.lower()
                phase = ("tensor_loading" if "load_tensors" in lower or "tensor loading" in lower else
                         "model_opening" if "loading model" in lower or "model loader" in lower else
                         "gpu_offloading" if "offload" in lower else
                         "gpu_allocating" if "buffer size" in lower or "allocating buffer" in lower else
                         "server_initializing" if "context initialization" in lower else
                         "waiting_health" if "server ready" in lower or "listening" in lower else
                         "runtime_initializing" if "sycl" in lower or "level zero" in lower else None)
                if phase:
                    with self._load_progress_lock: self._load_progress["phase"] = phase
                self._record_native_evidence(line)
        finally:
            footer = ("=== AXP INTEL SESSION END ===\n"
                f"timestamp={datetime.now(timezone.utc).isoformat()}\nresult={self._session_result}\n"
                f"offloaded_layers={self.offloaded_layers}\ngpu_buffer_bytes={self.gpu_buffer_bytes}\n"
                f"cpu_buffer_bytes={self.cpu_buffer_bytes}\n"
                f"native_lines_seen={self._load_progress.get('native_lines_seen', 0)}\n")
            handler.stream.write(footer); handler.flush()
            handler.close()

    def _create_auth(self):
        root = runtime_paths()["runtime"] / "sidecars"; root.mkdir(parents=True, exist_ok=True)
        self._auth_dir = Path(tempfile.mkdtemp(prefix="intel-", dir=root)); os.chmod(self._auth_dir, 0o700)
        self._auth_file = self._auth_dir / "api-key.txt"; self._api_key = secrets.token_urlsafe(32)
        fd = os.open(self._auth_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(self._api_key); handle.flush(); os.fsync(handle.fileno())

    def _sidecar_command(self):
        """Build the pinned runtime command without ever embedding key material."""
        return [str(self.server_path), "--model", str(self.model_path), "--host", LOOPBACK,
                "--port", str(self._port), "--ctx-size", str(self.config.context_size),
                "--parallel", "1", "--device", self.sycl_device_id, "--split-mode", "none",
                "--n-gpu-layers", "all", "--api-key-file", str(self._auth_file),
                "-lv", str(self.verbosity)]

    def _request(self, path, payload=None, timeout=None):
        headers = {"Authorization": f"Bearer {self._api_key}"}; data = None
        if payload is not None: data = json.dumps(payload).encode(); headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._endpoint(path), data=data, headers=headers)
        return self._urlopen(request, timeout=timeout or self.load_timeout)

    def _health_state(self):
        try:
            with self._request("/health", timeout=1) as response:
                status, raw = response.status, response.read(65536)
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read(65536)
        except (OSError, urllib.error.URLError): return "unreachable"
        with self._load_progress_lock:
            self._load_progress.update(last_health_activity_monotonic=self._monotonic(),
                                       health_reachable=True, health_http_status=status)
        try: body = json.loads(raw or b"{}")
        except (ValueError, TypeError): body = None
        marker = str(body.get("status", "")).lower() if isinstance(body, dict) else ""
        state = ("auth_failed" if status in (401, 403) else "ready" if status == 200 and marker in ("ok", "ready")
                 else "loading" if status in (202, 503) or marker in ("loading", "starting") else "invalid")
        with self._load_progress_lock: self._load_progress["health_status"] = state
        return state

    def ensure_loaded(self):
        if self.loaded: return self
        with self._load_lock:
            if self.loaded: return self
            valid, reason = validate_gguf(self.model_path)
            if not valid: raise IntelSyclError(reason)
            if self.runtime_dir not in self.server_path.parents or not self.server_path.is_file():
                raise IntelSyclError("intel_sycl_runtime_invalid")
            if not self.sycl_device_id:
                raise IntelSyclError("intel_sycl_device_id_unresolved")
            self.close(); started = self._monotonic(); self._model_state = "loading"; self._port = self._free_port()
            self.gpu_offload_confirmed = False
            self.offloaded_layers = self.total_layers = None
            self.gpu_buffer_bytes = self.cpu_buffer_bytes = None
            self.native_gpu_markers = []
            self.session_id = uuid.uuid4().hex; self._session_result = "starting"
            self._load_cancel.clear(); self._create_auth(); self._load_progress = self._new_load_progress()
            with self._load_progress_lock:
                self._load_progress.update(active=True, phase="spawning", started_monotonic=started,
                    last_native_activity_monotonic=started, model_path_configured=True)
            command = self._sidecar_command()
            try:
                env = child_environment(self.runtime_dir)
                if self.diagnostic: env["GGML_SYCL_DEBUG"] = "1"
                LOGGER.info("Intel sidecar starting session_id=%s runtime=%s model_id=%s device_id=%s device_name=%s verbosity=%s",
                            self.session_id, INTEL_SYCL.tag, self.model_path.stem, self.sycl_device_id,
                            self.sycl_device_name, self.verbosity)
                self._process = self._popen(command, cwd=str(self.runtime_dir), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                with self._load_progress_lock:
                    self._load_progress.update(phase="runtime_initializing", sidecar_pid=self._process.pid)
                self._reader_thread = threading.Thread(target=self._log_reader, args=(self._process.stdout,), daemon=False,
                    name="axp-intel-sycl-log"); self._reader_thread.start()
                while True:
                    elapsed = self._monotonic() - started
                    if self._load_cancel.is_set(): raise IntelSyclError("intel_gpu_model_load_cancelled")
                    if self._process.poll() is not None:
                        text = "".join(self._diagnostic).lower()
                        code = ("intel_gpu_out_of_memory" if re.search(r"out of memory|bad_alloc|allocation failed", text)
                                else "intel_gpu_device_lost" if "device lost" in text
                                else "intel_gpu_model_unsupported" if "unsupported model" in text
                                else "intel_gpu_process_exited")
                        raise IntelSyclError(code)
                    state = self._health_state()
                    if state == "auth_failed": raise IntelSyclError("intel_gpu_auth_failed")
                    if state == "ready": break
                    snapshot = self.model_load_progress(); native_age = snapshot["last_native_activity_age_s"]
                    health_age = snapshot["last_health_activity_age_s"]
                    with self._load_progress_lock:
                        self._load_progress["slow_warning"] = elapsed >= INTEL_LOAD_WARN_AFTER_S
                        self._load_progress["suspected_stall"] = (elapsed >= INTEL_LOAD_STALL_WARN_AFTER_S and
                            native_age >= INTEL_LOAD_ACTIVITY_STALE_S and
                            (health_age is None or health_age >= INTEL_LOAD_ACTIVITY_STALE_S))
                    if elapsed >= self.load_timeout: raise IntelSyclError("intel_gpu_model_load_timeout")
                    self._sleep(.2)
                LOGGER.info("Intel model server ready session_id=%s pid=%s native_lines_seen=%s",
                            self.session_id, self._process.pid, self._load_progress["native_lines_seen"])
                if not self.gpu_offload_confirmed:
                    self._session_result = "offload_not_confirmed"
                    LOGGER.info("Intel offload NOT confirmed session_id=%s device_id=%s native_lines_seen=%s failure_type=intel_gpu_offload_not_confirmed",
                                self.session_id, self.sycl_device_id, self._load_progress["native_lines_seen"])
                    raise IntelSyclError("intel_gpu_offload_not_confirmed")
                self._model_state = "loaded"; self._load_ms = (self._monotonic() - started) * 1000; self._failure = {}
                self._session_result = "offload_confirmed"
                LOGGER.info("Intel offload confirmed session_id=%s pid=%s load_ms=%.1f offloaded_layers=%s total_layers=%s gpu_buffer_bytes=%s cpu_buffer_bytes=%s",
                            self.session_id, self._process.pid, self._load_ms, self.offloaded_layers,
                            self.total_layers, self.gpu_buffer_bytes, self.cpu_buffer_bytes)
                with self._load_progress_lock: self._load_progress.update(active=False, phase="ready")
            except Exception as exc:
                self._failure = {"failure_type": str(exc) if isinstance(exc, IntelSyclError) else "intel_gpu_backend_failed",
                                 "failure_reason": "Intel GPU inference could not be started.", "retryable": True}
                code = self._failure["failure_type"]
                self._model_state = "unloaded" if code == "intel_gpu_model_load_cancelled" else "failed"
                with self._load_progress_lock:
                    self._load_progress.update(active=False, phase="cancelled" if code.endswith("cancelled") else
                        "timed_out" if code.endswith("timeout") else "failed", failure_type=code,
                        failure_reason=self._failure["failure_reason"])
                self.close(failed=True); raise
        return self

    def retry_load(self):
        self.close(); self._model_state = "unloaded"; self._failure = {}

    def context_window(self): return self.config.context_size

    def _post(self, path, payload, timeout=None):
        return self._request(path, payload, timeout)

    def count_tokens(self, text):
        self.ensure_loaded()
        with self._post("/tokenize", {"content": text, "add_special": False}) as response:
            value = json.loads(response.read())
        tokens = value.get("tokens")
        if not isinstance(tokens, list): raise IntelSyclError("intel_gpu_tokenization_failed")
        return len(tokens)

    def generate(self, *, system_prompt, user_prompt):
        self.ensure_loaded(); self._cancel_event.clear(); started = time.perf_counter()
        system = (NO_THINK_DIRECTIVE + system_prompt if self.chat_template_kwargs.get("enable_thinking") is False
                  else system_prompt)
        payload = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                   "stream": True, "max_tokens": self.config.max_answer_tokens, "temperature": self.config.temperature,
                   "return_progress": True,
                   "top_p": self.config.top_p, "top_k": self.config.top_k,
                   "repeat_penalty": self.config.repeat_penalty}
        with self._progress_lock:
            self._progress = self._new_progress(); self._progress.update(active=True, phase="waiting_first_token",
                                                                          started_monotonic=started)
        fragments, first, finish, native_timings = [], None, None, None
        try:
            with self._post("/v1/chat/completions", payload) as response:
                for data in parse_sse(response):
                    if self._cancel_event.is_set(): raise GenerationCancelled
                    if data == "[DONE]": break
                    try: chunk = json.loads(data)
                    except (ValueError, TypeError) as exc: raise IntelSyclError("intel_gpu_generation_failed") from exc
                    progress = chunk.get("prompt_progress")
                    if progress is None and chunk.get("type") == "prompt_progress": progress = chunk.get("data", chunk)
                    if isinstance(progress, dict):
                        with self._progress_lock:
                            self._progress.update(phase="prompt_evaluating",
                                prompt_total=progress.get("total", progress.get("prompt_total")),
                                prompt_cached=progress.get("cached", progress.get("prompt_cached")),
                                prompt_processed=progress.get("processed", progress.get("prompt_processed")),
                                prompt_progress_time_ms=progress.get("time_ms", progress.get("prompt_progress_time_ms")))
                    if isinstance(chunk.get("timings"), dict): native_timings = chunk["timings"]
                    choice = (chunk.get("choices") or [{}])[0]; delta = choice.get("delta") or {}
                    finish = choice.get("finish_reason") or finish
                    content = delta.get("content")  # reasoning_content is intentionally ignored
                    if isinstance(content, str) and content:
                        now = time.perf_counter(); first = first or now; fragments.append(content)
                        with self._progress_lock:
                            self._progress.update(phase="generating", sequence=self._progress["sequence"] + 1,
                                generated_fragments=self._progress["generated_fragments"] + 1,
                                generated_characters=self._progress["generated_characters"] + len(content),
                                time_to_first_token_ms=(first-started)*1000, last_fragment_monotonic=now)
            ended = time.perf_counter(); answer = THINK_RE.sub("", "".join(fragments)).strip()
            completion = self.count_tokens(answer) if answer else 0; generation_ms = (ended-started)*1000
            decode_ms = (ended-first)*1000 if first else None
            timing = native_timings or {}
            prompt_ms = timing.get("prompt_ms", timing.get("prompt_eval_ms"))
            prompt_n = timing.get("prompt_n", timing.get("prompt_tokens"))
            predicted_ms = timing.get("predicted_ms", timing.get("decode_ms"))
            predicted_n = timing.get("predicted_n", timing.get("completion_tokens"))
            self.last_telemetry = {"time_to_first_token_ms": (first-started)*1000 if first else None,
                "generation_ms": generation_ms, "prompt_tokens": prompt_n, "prompt_eval_ms": prompt_ms,
                "prompt_eval_tokens_per_second": timing.get("prompt_per_second") or
                    (prompt_n/(prompt_ms/1000) if prompt_n is not None and prompt_ms else None),
                "prompt_eval_timing_derived": prompt_ms is None,
                "decode_ms": predicted_ms or decode_ms, "completion_tokens": predicted_n or completion,
                "decode_tokens_per_second": timing.get("predicted_per_second") or
                    ((predicted_n or completion)/((predicted_ms or decode_ms)/1000) if (predicted_ms or decode_ms) else None),
                "generated_characters": sum(map(len, fragments)), "generated_fragments": len(fragments),
                "finish_reason": finish, "backend": "intel_sycl", "runtime": INTEL_SYCL.tag,
                "model_load_ms": self._load_ms, "gpu_offload_confirmed": self.gpu_offload_confirmed,
                "offloaded_layers": self.offloaded_layers, "total_layers": self.total_layers,
                "inference_device_requested": "intel_gpu", "inference_device_effective": "intel_gpu",
                "gpu_buffer_bytes": self.gpu_buffer_bytes}
            with self._progress_lock: self._progress.update(active=False, phase="completed", finish_reason=finish)
            return answer
        except GenerationCancelled:
            with self._progress_lock: self._progress.update(active=False, phase="cancelled", finish_reason="cancelled")
            raise
        except Exception as exc:
            with self._progress_lock: self._progress.update(active=False, phase="failed")
            if isinstance(exc, IntelSyclError): raise
            raise IntelSyclError("intel_gpu_generation_failed") from exc

    def close(self, failed=False):
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        if self._reader_thread and self._reader_thread is not threading.current_thread():
            self._reader_thread.join(timeout=5)
        self._reader_thread = None
        if self._auth_dir: shutil.rmtree(self._auth_dir, ignore_errors=True)
        self._api_key = None; self._auth_file = self._auth_dir = None; self._port = None
        if not failed: self._model_state = "unloaded"
        if process is not None:
            LOGGER.info("Intel sidecar stopped session_id=%s pid=%s result=%s native_lines_seen=%s",
                        self.session_id, getattr(process, "pid", None), self._session_result,
                        self._load_progress.get("native_lines_seen", 0))
