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
        self.opener = opener or urllib.request.build_opener(TrustedRedirectHandler())
        self._lock, self._job, self._cancel = threading.Lock(), None, threading.Event()

    def model_path(self, model_id): return self.models_dir / model_id / "model.gguf"
    def manifest_path(self, model_id): return self.models_dir / model_id / "manifest.json"

    def catalog(self):
        settings = load_settings(); active = settings.get("chat_active_model_id")
        result = []
        for model in MODELS:
            partial = self.downloads_dir / f"{model.id}.gguf.part"
            result.append({**model.public(), "installed": self.model_path(model.id).is_file(),
                           "active": active == model.id, "partial_bytes": partial.stat().st_size if partial.exists() else 0,
                           "download": self._job.public() if self._job and self._job.model_id == model.id else None})
        custom = settings.get("chat_model_path")
        return {"catalog_version": CATALOG_VERSION, "active_model_id": active,
                "models": result, "custom_model": {"name": "Custom local model", "installed": Path(custom).is_file(),
                "active": not active} if custom else None}

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
        except (urllib.error.URLError, OSError):
            self._state(job, "failed", "network_error")

    def activate(self, model_id):
        model = catalog_model(model_id); path = self.model_path(model_id)
        if model is None: raise ModelManagerError("model_not_found")
        if not path.is_file() or not self.manifest_path(model_id).is_file(): raise ModelManagerError("model_not_installed")
        if self.runtime and self.runtime.busy: raise ModelManagerError("chat_busy")
        settings = load_settings(); previous = dict(settings)
        settings.update(chat_active_model_id=model_id, chat_model_path=str(path))
        try:
            save_settings(settings)
            if self.runtime: self.runtime.activate(settings, model)
        except Exception:
            save_settings(previous); raise
        return self.catalog()

    def remove(self, model_id):
        if load_settings().get("chat_active_model_id") == model_id: raise ModelManagerError("model_active")
        if catalog_model(model_id) is None: raise ModelManagerError("model_not_found")
        shutil.rmtree(self.models_dir / model_id, ignore_errors=True); return self.catalog()
