"""Secure installation and qualification of the optional Intel SYCL bundle."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import time
import uuid
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from axp_core.runtime import atomic_write_json

from .accelerator_catalog import INTEL_SYCL

APPROVED_HOSTS = ("github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com")
LOGGER = logging.getLogger("axp_client")


class AcceleratorError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(host == item or host.endswith("." + item) for item in APPROVED_HOSTS):
            raise urllib.error.HTTPError(req.full_url, 403, "unapproved_redirect", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _inside(root, member):
    root = root.resolve()
    candidate = (root / member).resolve()
    return candidate != root and root in candidate.parents


def safe_extract(archive, destination):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not _inside(destination, info.filename):
                raise AcceleratorError("accelerator_zip_traversal")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise AcceleratorError("accelerator_zip_symlink")
        bundle.extractall(destination)


def discover_binaries(root):
    root = Path(root).resolve()
    found = {}
    for name, key in (("llama-server.exe", "server_path"), ("llama-bench.exe", "bench_path"),
                      ("llama-cli.exe", "cli_path")):
        matches = [item.resolve() for item in root.rglob(name) if item.is_file() and item.stat().st_size > 0]
        if any(root not in item.parents for item in matches):
            raise AcceleratorError("accelerator_runtime_invalid")
        if name == "llama-server.exe" and len(matches) != 1:
            raise AcceleratorError("accelerator_server_missing" if not matches else "accelerator_server_ambiguous")
        if len(matches) > 1:
            raise AcceleratorError("accelerator_binary_ambiguous")
        found[key] = str(matches[0].relative_to(root)) if matches else None
    return found


def validate_archive(path, release=INTEL_SYCL):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != release.exact_size:
        raise AcceleratorError("accelerator_size_mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != release.sha256:
        raise AcceleratorError("accelerator_sha256_mismatch")


class AcceleratorManager:
    def __init__(self, data_root, opener=None):
        self.data_root = Path(data_root)
        self.runtime_root = self.data_root / "runtime" / "accelerators" / "intel-sycl" / INTEL_SYCL.tag
        self.download_root = self.data_root / "runtime" / "downloads"
        self.opener = opener or urllib.request.build_opener(GitHubRedirectHandler())

    @property
    def manifest_path(self):
        return self.runtime_root / "manifest.json"

    def manifest(self):
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            server = (self.runtime_root / value["server_path"]).resolve()
            if self.runtime_root.resolve() not in server.parents or not server.is_file() or not server.stat().st_size:
                return None
            return value
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def server_path(self):
        value = self.manifest()
        return self.runtime_root / value["server_path"] if value else None

    def install_archive(self, archive):
        validate_archive(archive)
        parent = self.runtime_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{INTEL_SYCL.tag}-", dir=parent))
        backup = parent / f"{self.runtime_root.name}.backup-{uuid.uuid4().hex}"
        try:
            safe_extract(archive, staging)
            binaries = discover_binaries(staging)
            manifest = {"runtime_id": INTEL_SYCL.id, "upstream_tag": INTEL_SYCL.tag,
                        "upstream_commit": INTEL_SYCL.commit, "asset_sha256": INTEL_SYCL.sha256,
                        "installed_at": int(time.time()), **binaries}
            atomic_write_json(staging / "manifest.json", manifest)
            had_runtime = self.runtime_root.exists()
            if had_runtime:
                self._replace_with_retry(self.runtime_root, backup)
            try:
                self._replace_with_retry(staging, self.runtime_root)
            except Exception:
                if had_runtime and backup.exists() and not self.runtime_root.exists():
                    self._replace_with_retry(backup, self.runtime_root)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _replace_with_retry(source, destination):
        delays = (0.05, 0.1, 0.2, 0.4)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(source, destination)
                return
            except (PermissionError, OSError):
                if attempt == len(delays):
                    raise
                time.sleep(delays[attempt])

    def download_and_install(self, progress=None, cancel=None):
        """Download only after this explicit method is called; URL is catalog-owned."""
        self.download_root.mkdir(parents=True, exist_ok=True)
        part = self.download_root / f"{INTEL_SYCL.id}.zip.part"
        offset = part.stat().st_size if part.exists() and part.stat().st_size <= INTEL_SYCL.exact_size else 0
        if progress:
            progress("connecting", offset, INTEL_SYCL.exact_size, offset)
        digest = hashlib.sha256()
        if offset:
            with part.open("rb") as old:
                for chunk in iter(lambda: old.read(1024 * 1024), b""):
                    digest.update(chunk)
        request = urllib.request.Request(INTEL_SYCL.url, headers={"Range": f"bytes={offset}-"} if offset else
                                         {"Accept-Encoding": "identity"})
        response = self.opener.open(request, timeout=60)
        if offset and getattr(response, "status", response.getcode()) != 206:
            response.close(); part.unlink(missing_ok=True); offset = 0; digest = hashlib.sha256()
            response = self.opener.open(urllib.request.Request(INTEL_SYCL.url,
                headers={"Accept-Encoding": "identity"}), timeout=60)
        downloaded = offset
        if progress:
            progress("downloading", downloaded, INTEL_SYCL.exact_size, offset)
        with response, part.open("ab" if offset else "wb") as target:
            while True:
                if cancel and cancel.is_set():
                    raise AcceleratorError("accelerator_download_cancelled")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk); digest.update(chunk); downloaded += len(chunk)
                if downloaded > INTEL_SYCL.exact_size:
                    raise AcceleratorError("accelerator_size_mismatch")
                if progress:
                    progress("downloading", downloaded, INTEL_SYCL.exact_size, offset)
        if downloaded != INTEL_SYCL.exact_size:
            raise AcceleratorError("accelerator_size_mismatch")
        if digest.hexdigest() != INTEL_SYCL.sha256:
            part.unlink(missing_ok=True)
            raise AcceleratorError("accelerator_sha256_mismatch")
        if progress:
            progress("verifying", downloaded, INTEL_SYCL.exact_size, offset)
            progress("installing", downloaded, INTEL_SYCL.exact_size, offset)
        result = self.install_archive(part)
        part.unlink(missing_ok=True)
        return result

    def remove(self):
        try:
            shutil.rmtree(self.runtime_root)
        except FileNotFoundError:
            return
        except OSError as exc:
            LOGGER.exception("Intel accelerator removal failed")
            raise AcceleratorError("accelerator_remove_failed") from exc
        if self.runtime_root.exists():
            LOGGER.error("Intel accelerator removal left runtime directory present")
            raise AcceleratorError("accelerator_remove_failed")
