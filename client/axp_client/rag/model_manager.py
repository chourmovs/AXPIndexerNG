"""Secure, release-catalog model installation service."""
import hashlib
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from axp_core.runtime import atomic_write_json, load_settings, save_settings

from .model_catalog import CATALOG_VERSION, MODELS, catalog_model
from .accelerator_catalog import INTEL_SYCL
from .accelerator_manager import AcceleratorError, AcceleratorManager
from .hardware import detect_hardware
from .benchmark import BenchmarkRunner
from .runtime_manager import ALLOWED_DEVICES, InferenceDeviceError

ACTIVE_DOWNLOAD_STATES = {"queued", "connecting", "downloading", "verifying", "installing"}
APPROVED_HOSTS = ("huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co",
                  "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co")


class ModelManagerError(RuntimeError):
    def __init__(self, code, details=None):
        super().__init__(code); self.code, self.details = code, details or {}


class TrustedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(host == allowed or host.endswith("." + allowed)
                                                for allowed in APPROVED_HOSTS):
            raise urllib.error.HTTPError(req.full_url, 403, "unapproved_redirect", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class DownloadJob:
    job_id: str
    model_id: str
    state: str = "queued"
    bytes_downloaded: int = 0
    bytes_total: int = 0
    percentage: float = 0
    bytes_per_second: float = 0
    eta_seconds: float | None = None
    started_at: float = 0
    updated_at: float = 0
    error: str | None = None
    activate: bool = False

    def public(self): return asdict(self)


class ModelManager:
    def __init__(self, cache_root, runtime=None, opener=None):
        self.root = Path(cache_root) / "chat"
        self.models_dir, self.downloads_dir = self.root / "models", self.root / "downloads"
        self.models_dir.mkdir(parents=True, exist_ok=True); self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.accelerators = getattr(getattr(runtime, "backend", runtime), "accelerators", None) or AcceleratorManager(cache_root)
        self._accelerator_job = None
        self._benchmark = None
        self.opener = opener or urllib.request.build_opener(TrustedRedirectHandler())
        self._lock, self._job, self._cancel = threading.Lock(), None, threading.Event()

    def model_path(self, model_id): return self.models_dir / model_id / "model.gguf"
    def manifest_path(self, model_id): return self.models_dir / model_id / "manifest.json"

    def catalog(self):
        settings = load_settings(); active = settings.get("chat_active_model_id")
        health = self.runtime.health() if self.runtime else {}
        result = []
        for model in MODELS:
            partial = self.downloads_dir / f"{model.id}.gguf.part"
            selected = active == model.id
            result.append({**model.public(), "installed": self.model_path(model.id).is_file(),
                           "active": selected, "selected": selected,
                           "model_loaded": bool(selected and health.get("model_loaded")),
                           "model_state": health.get("model_state") if selected else "installed",
                           "failure_type": health.get("failure_type") if selected else None,
                           "failure_reason": health.get("failure_reason") if selected else None,
                           "retryable": health.get("retryable") if selected else None,
                           "partial_bytes": partial.stat().st_size if partial.exists() else 0,
                           "download": self._job.public() if self._job and self._job.model_id == model.id else None})
        custom = settings.get("chat_model_path") if not active else None
        return {"catalog_version": CATALOG_VERSION, "active_model_id": active,
                "models": result, "custom_model": {"name": "Custom local model",
                "filename": Path(custom).name, "installed": Path(custom).is_file(), "active": True} if custom else None,
                "device": {key: health.get(key) for key in ("inference_device_requested",
                    "inference_device_effective", "fallback_reason")},
                "hardware": {**{key: health.get(key) for key in ("intel_gpu_detected", "intel_gpu_name",
                    "intel_gpu_vendor_id", "intel_gpu_device_id", "accelerator_available", "accelerator_reason",
                    "sycl_runtime_installed", "sycl_probe_ok", "sycl_device_name", "sycl_probe_error")},
                    "accelerator": {**INTEL_SYCL.public(), "installed": bool(self.accelerators.manifest()),
                                    "download": dict(self._accelerator_job) if self._accelerator_job else None}},
                "benchmark": self._benchmark.job.public() if self._benchmark else {"state": "idle"}}

    def start_benchmark(self, profile_name="quick"):
        settings = load_settings(); profile = catalog_model(settings.get("chat_active_model_id"))
        if profile is None or not Path(settings["chat_model_path"]).is_file():
            raise ModelManagerError("benchmark_model_required")
        controller = getattr(self.runtime, "backend", self.runtime)
        if not controller or not controller.hardware.intel_gpu_available:
            raise ModelManagerError("intel_gpu_unavailable")
        def configured(max_tokens):
            values = {key: getattr(profile, key) for key in profile.__dataclass_fields__}
            values["max_answer_tokens"] = max_tokens
            return type(profile)(**values)
        runner = BenchmarkRunner(lambda limit: controller._cpu_backend(settings, configured(limit)),
            lambda limit: controller._make_backend({**settings, "chat_inference_device": "intel_gpu"}, configured(limit)),
            profile.id, {"cpu": controller.hardware.cpu_name, "intel_gpu": controller.hardware.intel_gpu_name,
                           "intel_device_id": controller.hardware.intel_gpu_device_id,
                           "sycl_device": controller.hardware.sycl_device_name})
        def transaction():
            controller.close(); self._benchmark = runner; return runner.start(profile_name)
        try:
            result = self.runtime.run_when_idle(transaction) if callable(getattr(self.runtime, "run_when_idle", None)) else transaction()
        except Exception as exc:
            if type(exc).__name__ == "ChatBusyError": raise ModelManagerError("chat_busy") from exc
            if isinstance(exc, (ValueError, RuntimeError)): raise ModelManagerError(str(exc)) from exc
            raise
        def restore():
            while runner.job.state not in ("complete", "failed", "cancelled"): time.sleep(.2)
            try: controller.activate(settings, profile)
            except Exception: pass
        threading.Thread(target=restore, daemon=True, name="axp-benchmark-restore").start()
        return result

    def cancel_benchmark(self):
        if not self._benchmark: raise ModelManagerError("benchmark_not_active")
        try: return self._benchmark.cancel()
        except RuntimeError as exc: raise ModelManagerError(str(exc)) from exc

    def start_accelerator_download(self):
        with self._lock:
            if self._accelerator_job and self._accelerator_job["state"] in ACTIVE_DOWNLOAD_STATES:
                raise ModelManagerError("accelerator_download_busy")
            self._accelerator_job = {"state": "queued", "bytes_downloaded": 0,
                                     "bytes_total": INTEL_SYCL.exact_size, "percentage": 0,
                                     "error": None}
        def work():
            try:
                self._accelerator_job["state"] = "downloading"
                def progress(done, total):
                    self._accelerator_job.update(bytes_downloaded=done, percentage=done * 100 / total)
                self.accelerators.download_and_install(progress)
                self._accelerator_job.update(state="ready", bytes_downloaded=INTEL_SYCL.exact_size, percentage=100)
                controller = getattr(self.runtime, "backend", self.runtime)
                if controller:
                    controller.hardware = detect_hardware(self.accelerators.server_path())
            except AcceleratorError as exc:
                self._accelerator_job.update(state="failed", error=exc.code)
            except Exception:
                self._accelerator_job.update(state="failed", error="accelerator_download_failed")
        threading.Thread(target=work, daemon=True, name="axp-accelerator-download").start()
        return dict(self._accelerator_job)

    def remove_accelerator(self):
        controller = getattr(self.runtime, "backend", self.runtime)
        if controller and (controller.health().get("inference_device_effective") == "intel_gpu" or
                           controller.health().get("sidecar_pid")):
            raise ModelManagerError("accelerator_in_use")
        self.accelerators.remove()
        if controller: controller.hardware = detect_hardware()
        return self.catalog()

    def start_download(self, model_id, *, activate=False):
        model = catalog_model(model_id)
        if model is None: raise ModelManagerError("model_not_found")
        with self._lock:
            if self._job and self._job.state in ACTIVE_DOWNLOAD_STATES: raise ModelManagerError("model_download_busy")
            self._cancel.clear(); now = time.time()
            self._job = DownloadJob(os.urandom(12).hex(), model_id, bytes_total=model.size_bytes,
                                    started_at=now, updated_at=now, activate=activate)
            threading.Thread(target=self._download, args=(model, self._job), daemon=True,
                             name="axp-model-download").start()
            return self._job.public()

    def cancel(self, model_id):
        if not self._job or self._job.model_id != model_id or self._job.state not in ACTIVE_DOWNLOAD_STATES:
            raise ModelManagerError("download_not_active")
        self._cancel.set(); return self._job.public()

    def _state(self, job, state, error=None):
        job.state, job.error, job.updated_at = state, error, time.time()

    def _download(self, model, job):
        part = self.downloads_dir / f"{model.id}.gguf.part"
        try:
            margin = max(512 * 1024**2, model.size_bytes // 10)
            free = shutil.disk_usage(self.downloads_dir).free
            existing = part.stat().st_size if part.exists() else 0
            if free < model.size_bytes - existing + margin:
                raise ModelManagerError("insufficient_disk", {"required_bytes": model.size_bytes + margin, "free_bytes": free})
            digest = hashlib.sha256(); offset = existing if existing <= model.size_bytes else 0
            if offset:
                with part.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""): digest.update(chunk)
            self._state(job, "connecting")
            request = urllib.request.Request(model.url, headers={"Range": f"bytes={offset}-"} if offset else {"Accept-Encoding": "identity"})
            response = self.opener.open(request, timeout=60)
            if offset and (getattr(response, "status", response.getcode()) != 206 or
                           not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                response.close(); offset = 0; digest = hashlib.sha256(); part.unlink(missing_ok=True)
                response = self.opener.open(urllib.request.Request(model.url, headers={"Accept-Encoding": "identity"}), timeout=60)
            job.bytes_downloaded = offset; started = time.monotonic(); self._state(job, "downloading")
            with response, part.open("ab" if offset else "wb") as target:
                while True:
                    if self._cancel.is_set(): raise ModelManagerError("download_cancelled")
                    chunk = response.read(1024 * 1024)
                    if not chunk: break
                    target.write(chunk); digest.update(chunk); job.bytes_downloaded += len(chunk)
                    elapsed = max(time.monotonic() - started, .001)
                    job.percentage = min(100, job.bytes_downloaded * 100 / model.size_bytes)
                    job.bytes_per_second = max(0, (job.bytes_downloaded - offset) / elapsed)
                    remaining = max(0, model.size_bytes - job.bytes_downloaded)
                    job.eta_seconds = remaining / job.bytes_per_second if job.bytes_per_second else None
                    job.updated_at = time.time()
            self._state(job, "verifying")
            if job.bytes_downloaded != model.size_bytes: raise ModelManagerError("integrity_mismatch")
            if digest.hexdigest() != model.sha256: raise ModelManagerError("integrity_mismatch")
            with part.open("rb") as source:
                if source.read(4) != b"GGUF": raise ModelManagerError("invalid_gguf")
            self._state(job, "installing"); destination = self.model_path(model.id)
            destination.parent.mkdir(parents=True, exist_ok=True); os.replace(part, destination)
            atomic_write_json(self.manifest_path(model.id), {"model_id": model.id, "display_name": model.name,
                "source_repository": model.repository, "source_revision": model.revision,
                "source_filename": model.filename, "expected_sha256": model.sha256,
                "actual_sha256": model.sha256, "size_bytes": model.size_bytes, "license": model.license,
                "installed_at_ms": int(time.time() * 1000), "catalog_version": CATALOG_VERSION})
            if job.activate: self.activate(model.id)
            job.percentage = 100; self._state(job, "ready")
        except ModelManagerError as exc:
            self._state(job, "cancelled" if exc.code == "download_cancelled" else "failed", exc.code)
            if exc.code in ("integrity_mismatch", "invalid_gguf"): part.unlink(missing_ok=True)
        except ssl.SSLError:
            self._state(job, "failed", "tls_error")
        except urllib.error.URLError as exc:
            self._state(job, "failed", "tls_error" if isinstance(exc.reason, ssl.SSLError) else "network_error")
        except OSError:
            self._state(job, "failed", "network_error")

    def activate(self, model_id):
        model = catalog_model(model_id); path = self.model_path(model_id)
        if model is None: raise ModelManagerError("model_not_found")
        if not path.is_file() or not self.manifest_path(model_id).is_file(): raise ModelManagerError("model_not_installed")
        def transaction():
            settings = load_settings(); previous = dict(settings)
            settings.update(chat_active_model_id=model_id, chat_model_path=str(path))
            replacement = None
            controller = getattr(self.runtime, "backend", self.runtime)
            try:
                if controller and callable(getattr(controller, "prepare_activation", None)):
                    replacement = controller.prepare_activation(settings, model)
                save_settings(settings)
                if replacement is not None:
                    controller.commit_activation(settings, replacement)
                elif self.runtime:
                    self.runtime.activate(settings, model)
            except Exception:
                if replacement is not None and callable(getattr(replacement, "close", None)):
                    replacement.close()
                save_settings(previous); raise
            return self.catalog()
        try:
            return self.runtime.run_when_idle(transaction) if callable(getattr(self.runtime, "run_when_idle", None)) else transaction()
        except Exception as exc:
            if type(exc).__name__ == "ChatBusyError": raise ModelManagerError("chat_busy") from exc
            raise

    def set_device(self, device):
        if device not in ALLOWED_DEVICES:
            raise ModelManagerError("invalid_inference_device")
        def transaction():
            settings = load_settings(); previous = dict(settings)
            settings["chat_inference_device"] = device
            controller = getattr(self.runtime, "backend", self.runtime)
            try:
                if controller: controller.set_device(device)
                save_settings(settings)
            except InferenceDeviceError as exc:
                raise ModelManagerError(str(exc)) from exc
            except Exception:
                if controller: controller.set_device(previous.get("chat_inference_device", "auto"))
                raise
            return self.catalog()
        try:
            return self.runtime.run_when_idle(transaction) if callable(getattr(self.runtime, "run_when_idle", None)) else transaction()
        except Exception as exc:
            if type(exc).__name__ == "ChatBusyError": raise ModelManagerError("chat_busy") from exc
            raise

    def remove(self, model_id):
        if load_settings().get("chat_active_model_id") == model_id: raise ModelManagerError("model_active")
        if catalog_model(model_id) is None: raise ModelManagerError("model_not_found")
        shutil.rmtree(self.models_dir / model_id, ignore_errors=True); return self.catalog()
