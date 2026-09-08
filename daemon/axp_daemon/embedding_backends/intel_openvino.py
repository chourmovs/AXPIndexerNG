from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from ..embeddings import model_spec


RUNTIME_SCHEMA = 1


class IntelEmbeddingUnavailable(RuntimeError):
    """The optional, qualified Intel embedding accelerator cannot be used."""

    code = "intel_embedding_runtime_missing"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(root, spec):
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntelEmbeddingUnavailable("intel_embedding_runtime_missing: manifest is absent or invalid") from exc
    required = {
        "runtime_schema": RUNTIME_SCHEMA, "backend": "openvino_genai", "model_id": spec.model_id,
        "embedding_dimension": spec.dimension, "distance_metric": spec.distance_metric,
        "pooling": "MEAN", "normalize": True,
        "maximum_sequence_length": 512, "query_prefix": spec.query_prefix,
        "document_prefix": spec.document_prefix,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise IntelEmbeddingUnavailable("intel_embedding_manifest_incompatible")
    if not value.get("source_revision") or not isinstance(value.get("export_tools"), dict):
        raise IntelEmbeddingUnavailable("intel_embedding_manifest_unpinned")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise IntelEmbeddingUnavailable("intel_embedding_manifest_unverified")
    for relative, expected in files.items():
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents or not candidate.is_file() or _sha256(candidate) != expected:
            raise IntelEmbeddingUnavailable(f"intel_embedding_asset_verification_failed: {relative}")
    return value


class IntelOpenVINOBackend:
    backend = "openvino_genai"
    effective_device = "intel_gpu"
    fallback_reason = None

    def __init__(self, model, *, runtime_dir, cache_dir=None, batch_size=64, require_qualification=True):
        self.spec = model_spec(model)
        if not self.spec.dimension:
            raise IntelEmbeddingUnavailable("intel_embedding_model_not_supported")
        self.model_id = self.spec.model_id
        self.dimension = self.spec.dimension
        self.distance_metric = self.spec.distance_metric
        self.runtime_dir = Path(runtime_dir)
        manifest = _load_manifest(self.runtime_dir, self.spec)
        self.batch_size = max(1, min(int(batch_size), 256))
        if os.name == "nt":
            for folder in (self.runtime_dir / "bin", self.runtime_dir / "python" / "Library" / "bin"):
                if folder.is_dir():
                    os.add_dll_directory(str(folder))
        try:
            import openvino
            import openvino_genai
        except ImportError as exc:
            raise IntelEmbeddingUnavailable("intel_embedding_runtime_missing") from exc
        core = openvino.Core()
        devices = list(core.available_devices)
        if "GPU" not in devices:
            raise IntelEmbeddingUnavailable("intel_embedding_gpu_not_found")
        try:
            self.device_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        except Exception:  # device name is diagnostic only
            self.device_name = "OpenVINO GPU"
        self.runtime_version = getattr(openvino, "__version__", None) or importlib.metadata.version("openvino")
        self.backend_version = getattr(openvino_genai, "__version__", "unknown")
        qualification = self._qualification()
        self.intel_gpu_qualified = self._qualification_matches(qualification)
        if require_qualification and not self.intel_gpu_qualified:
            raise IntelEmbeddingUnavailable("intel_embedding_not_qualified")
        model_path = self.runtime_dir / manifest.get("model_path", ".")
        config = {"CACHE_DIR": str(Path(cache_dir))} if cache_dir else {}
        try:
            # Never use AUTO: successful construction is the proof that this model compiled for GPU.
            self._pipeline = openvino_genai.TextEmbeddingPipeline(str(model_path), "GPU", config)
        except Exception as exc:
            raise IntelEmbeddingUnavailable(f"intel_embedding_gpu_compile_failed: {exc}") from exc

    def _qualification(self):
        try:
            return json.loads((self.runtime_dir / "qualification.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _qualification_matches(self, value):
        return bool(value.get("qualified") and value.get("model_id") == self.model_id
                    and value.get("device_name") == self.device_name
                    and value.get("openvino_version") == self.runtime_version
                    and value.get("backend") == self.backend)

    def embed_documents(self, texts):
        try:
            result = self._pipeline.embed_documents(list(texts))
        except AttributeError:
            result = self._pipeline.embed(list(texts))
        return [list(vector) for vector in result]

    def embed_query(self, text):
        # Provided for qualification only. Production query creation stays FastEmbed CPU.
        try:
            value = self._pipeline.embed_query(text)
        except AttributeError:
            value = self._pipeline.embed([text])[0]
        return list(value)

    def health(self):
        return {"backend": self.backend, "effective_device": self.effective_device, "healthy": True,
                "intel_gpu_qualified": self.intel_gpu_qualified, "intel_gpu_device_name": self.device_name,
                "openvino_version": self.runtime_version, "backend_version": self.backend_version}
