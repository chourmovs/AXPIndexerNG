"""Isolated localhost transport for the pinned llama.cpp Intel SYCL server."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path

from axp_core.runtime import runtime_paths

from .accelerator_catalog import INTEL_SYCL
from .llama_cpp_backend import GenerationCancelled, NO_THINK_DIRECTIVE, THINK_RE
from .model import validate_gguf

LOOPBACK = "127.0.0.1"
DEVICE_SELECTOR = "level_zero:gpu"
OFFLOAD_RE = re.compile(r"offload(?:ed|ing)?\s+(\d+)(?:\s*/\s*|\s+of\s+)(\d+)\s+layers?", re.I)
INTEL_GPU_RE = re.compile(r"(?=.*intel)(?=.*(?:gpu|graphics|arc|iris|uhd|xe)).+", re.I)


class IntelSyclError(RuntimeError):
    pass


def child_environment(runtime_dir):
    env = dict(os.environ)
    for key in tuple(env):
        if key.upper().startswith(("ONEAPI_", "SYCL_")) or key.upper() in ("ZE_AFFINITY_MASK",):
            env.pop(key, None)
    env["ONEAPI_DEVICE_SELECTOR"] = DEVICE_SELECTOR
    env["PATH"] = str(runtime_dir) + os.pathsep + env.get("PATH", "")
    return env


def parse_device_list(output):
    """Return recognizably Intel GPU device lines, never CPU-only matches."""
    devices = []
    for raw in output.splitlines():
        line = raw.strip()
        if INTEL_GPU_RE.search(line) and not re.search(r"\bcpu\b", line, re.I):
            devices.append(line[:300])
    return devices


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


def probe_sycl(server_path, timeout=15, runner=subprocess.run):
    server_path = Path(server_path)
    if not server_path.is_file() or not server_path.stat().st_size:
        return {"sycl_probe_ok": False, "sycl_device_name": None, "sycl_device_count": 0,
                "sycl_probe_error": "intel_sycl_runtime_invalid"}
    try:
        result = runner([str(server_path), "--list-devices"], cwd=str(server_path.parent),
            env=child_environment(server_path.parent), capture_output=True, text=True, timeout=timeout,
            check=False, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"sycl_probe_ok": False, "sycl_device_name": None, "sycl_device_count": 0,
                "sycl_probe_error": "intel_gpu_driver_or_level_zero_unavailable"}
    except OSError:
        return {"sycl_probe_ok": False, "sycl_device_name": None, "sycl_device_count": 0,
                "sycl_probe_error": "intel_sycl_runtime_invalid"}
    devices = parse_device_list((result.stdout or "") + "\n" + (result.stderr or ""))
    error = None if result.returncode == 0 and devices else (
        "intel_sycl_device_not_found" if result.returncode == 0 else "intel_gpu_driver_or_level_zero_unavailable")
    return {"sycl_probe_ok": error is None, "sycl_device_name": devices[0] if devices else None,
            "sycl_device_count": len(devices), "sycl_probe_error": error}


class IntelSyclBackend:
    def __init__(self, model_path, config, runtime_dir, server_path=None, chat_template_kwargs=None,
                 popen=subprocess.Popen, urlopen=urllib.request.urlopen, load_timeout=120):
        self.model_path, self.config = Path(model_path), config
        self.runtime_dir = Path(runtime_dir).resolve()
        self.server_path = Path(server_path or self.runtime_dir / "llama-server.exe").resolve()
        self.chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs is not None else {}
        self._popen, self._urlopen, self.load_timeout = popen, urlopen, load_timeout
        self._process = None; self._port = None; self._model_state = "unloaded"; self._load_ms = None
        self._load_lock = threading.Lock(); self._cancel_event = threading.Event(); self._progress_lock = threading.Lock()
        self._progress = self._new_progress(); self._failure = {}; self.last_telemetry = {}
        self.gpu_offload_requested = True; self.gpu_offload_confirmed = False
        self.offloaded_layers = self.total_layers = None; self._diagnostic = []; self.sycl_device_name = None

    @staticmethod
    def _new_progress():
        return {"active": False, "phase": "idle", "sequence": 0, "started_monotonic": None,
                "elapsed_s": None, "time_to_first_token_ms": None, "generated_fragments": 0,
                "generated_characters": 0, "last_fragment_age_s": None, "finish_reason": None,
                "cancel_requested": False}

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
        with self._progress_lock:
            if not self._progress["active"]: return False
            self._cancel_event.set(); self._progress.update(phase="cancel_requested", cancel_requested=True); return True

    def health(self):
        valid, reason = validate_gguf(self.model_path)
        if self._model_state == "failed": reason = self._failure.get("failure_type")
        progress = self.generation_progress()
        return {"available": valid and self._model_state != "failed", "reason": reason, "backend": "intel_sycl",
                "backend_version": INTEL_SYCL.tag, "accelerator_backend": "sycl", "accelerator_runtime": INTEL_SYCL.tag,
                "model_configured": self.model_path.is_file(), "model_valid": valid, "model_loaded": self.loaded,
                "model_state": self._model_state, "model_load_ms": self._load_ms, "context_size": self.config.context_size,
                "sidecar_pid": getattr(self._process, "pid", None), "sidecar_host": LOOPBACK, "sidecar_port": self._port,
                "sycl_device_name": self.sycl_device_name, "sycl_device_selector": DEVICE_SELECTOR,
                "gpu_offload_requested": True, "gpu_offload_confirmed": self.gpu_offload_confirmed,
                "offloaded_layers": self.offloaded_layers, "total_layers": self.total_layers,
                "generation_phase": progress["phase"], "generation_elapsed_s": progress["elapsed_s"],
                "time_to_first_token_ms": progress["time_to_first_token_ms"],
                "generated_fragments": progress["generated_fragments"],
                "generated_characters": progress["generated_characters"], **self._failure}

    def _endpoint(self, path): return f"http://{LOOPBACK}:{self._port}{path}"

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind((LOOPBACK, 0)); return sock.getsockname()[1]

    def _log_reader(self, stream):
        log = runtime_paths()["logs"] / "intel-sycl.log"
        handler = RotatingFileHandler(log, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        try:
            for line in iter(stream.readline, ""):
                handler.stream.write(line); handler.flush(); self._diagnostic.append(line[-1000:])
                self._diagnostic = self._diagnostic[-200:]
                match = OFFLOAD_RE.search(line)
                if match:
                    self.offloaded_layers, self.total_layers = map(int, match.groups())
                if INTEL_GPU_RE.search(line) and re.search(r"(?:SYCL|Level.Zero|device|buffer)", line, re.I):
                    self.sycl_device_name = line.strip()[:300]
                text = "".join(self._diagnostic)
                self.gpu_offload_confirmed = bool(self.sycl_device_name and
                    (self.offloaded_layers and self.offloaded_layers > 0 or
                     re.search(r"(?:SYCL|GPU).*(?:buffer|offload)", text, re.I)))
        finally:
            handler.close()

    def ensure_loaded(self):
        if self.loaded: return self
        with self._load_lock:
            if self.loaded: return self
            valid, reason = validate_gguf(self.model_path)
            if not valid: raise IntelSyclError(reason)
            if self.runtime_dir not in self.server_path.parents or not self.server_path.is_file():
                raise IntelSyclError("intel_sycl_runtime_invalid")
            started = time.perf_counter(); self._model_state = "loading"; self._port = self._free_port()
            command = [str(self.server_path), "--model", str(self.model_path), "--host", LOOPBACK,
                       "--port", str(self._port), "--ctx-size", str(self.config.context_size),
                       "--parallel", "1", "--n-gpu-layers", "999"]
            try:
                self._process = self._popen(command, cwd=str(self.runtime_dir), env=child_environment(self.runtime_dir),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                threading.Thread(target=self._log_reader, args=(self._process.stdout,), daemon=True,
                                 name="axp-intel-sycl-log").start()
                deadline = time.monotonic() + self.load_timeout
                while time.monotonic() < deadline:
                    if self._process.poll() is not None: raise IntelSyclError("intel_gpu_model_load_failed")
                    try:
                        with self._urlopen(self._endpoint("/health"), timeout=1) as response:
                            body = json.loads(response.read() or b"{}")
                            if response.status == 200 and body.get("status") in ("ok", "ready", None): break
                    except (OSError, ValueError, urllib.error.URLError): pass
                    time.sleep(.1)
                else: raise IntelSyclError("intel_gpu_model_load_failed")
                if not self.gpu_offload_confirmed:
                    raise IntelSyclError("intel_gpu_backend_failed")
                self._model_state = "loaded"; self._load_ms = (time.perf_counter() - started) * 1000; self._failure = {}
            except Exception as exc:
                self._failure = {"failure_type": str(exc) if isinstance(exc, IntelSyclError) else "intel_gpu_backend_failed",
                                 "failure_reason": "Intel GPU inference could not be started.", "retryable": True}
                self._model_state = "failed"; self.close(failed=True); raise
        return self

    def retry_load(self):
        self.close(); self._model_state = "unloaded"; self._failure = {}

    def context_window(self): return self.config.context_size

    def _post(self, path, payload, timeout=None):
        request = urllib.request.Request(self._endpoint(path), json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        return self._urlopen(request, timeout=timeout or self.load_timeout)

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
                   "top_p": self.config.top_p, "top_k": self.config.top_k,
                   "repeat_penalty": self.config.repeat_penalty}
        with self._progress_lock:
            self._progress = self._new_progress(); self._progress.update(active=True, phase="waiting_first_token",
                                                                          started_monotonic=started)
        fragments, first, finish = [], None, None
        try:
            with self._post("/v1/chat/completions", payload) as response:
                for data in parse_sse(response):
                    if self._cancel_event.is_set(): raise GenerationCancelled
                    if data == "[DONE]": break
                    try: chunk = json.loads(data)
                    except (ValueError, TypeError) as exc: raise IntelSyclError("intel_gpu_generation_failed") from exc
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
            self.last_telemetry = {"time_to_first_token_ms": (first-started)*1000 if first else None,
                "generation_ms": generation_ms, "decode_ms": decode_ms, "completion_tokens": completion,
                "decode_tokens_per_second": completion/(decode_ms/1000) if decode_ms else None,
                "generated_characters": sum(map(len, fragments)), "generated_fragments": len(fragments),
                "finish_reason": finish}
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
        if not failed: self._model_state = "unloaded"
