from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from axp_core.runtime import data_dir

from .cpu import CPUEmbeddingBackend
from .intel_openvino import IntelEmbeddingUnavailable, IntelOpenVINOBackend


class BackendFailure(RuntimeError):
    """A backend-wide failure which must not be treated as a bad document."""


class EmbeddingBackend(ABC):
    model_id: str
    dimension: int
    distance_metric: str

    @abstractmethod
    def embed_documents(self, texts): ...

    @abstractmethod
    def embed_query(self, text): ...

    @abstractmethod
    def health(self): ...


class AutoFallbackBackend:
    """Use qualified GPU until its first failure, then retry that whole batch on CPU."""

    def __init__(self, candidate, cpu_factory, requested="auto", fallback_reason=None):
        self._active = candidate
        self._cpu_factory = cpu_factory
        self.requested_device = requested
        self.fallback_reason = fallback_reason
        self.model_id = candidate.model_id
        self.dimension = candidate.dimension
        self.distance_metric = candidate.distance_metric

    @property
    def effective_device(self):
        return self._active.effective_device

    @property
    def backend(self):
        return self._active.backend

    @property
    def intel_gpu_qualified(self):
        return bool(getattr(self._active, "intel_gpu_qualified", False))

    @property
    def intel_gpu_device_name(self):
        return getattr(self._active, "device_name", None)

    def embed_documents(self, texts):
        values = list(texts)
        try:
            return self._active.embed_documents(values)
        except Exception as exc:  # backend failure: one transition for the rest of this scan
            if self.effective_device != "intel_gpu":
                raise
            self.fallback_reason = f"intel_gpu_batch_failed: {type(exc).__name__}: {exc}"
            self._active = self._cpu_factory()
            return self._active.embed_documents(values)

    def embed_query(self, text):
        return self._active.embed_query(text)

    def health(self):
        value = self._active.health()
        value.update(requested_device=self.requested_device, fallback_reason=self.fallback_reason)
        return value


class StrictBackend:
    """Prevent indexer's document-error bisection from hiding accelerator failure."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def embed_documents(self, texts):
        try:
            return self._backend.embed_documents(texts)
        except Exception as exc:
            raise BackendFailure(f"intel_embedding_backend_failed: {exc}") from exc


def create_indexing_embedder(model, *, device="auto", cache_dir=None, batch_size=64, local_only=True,
                             runtime_dir=None, cpu_factory=CPUEmbeddingBackend,
                             gpu_factory=IntelOpenVINOBackend):
    if device not in ("auto", "cpu", "intel_gpu"):
        raise ValueError(f"unsupported embedding device: {device}")
    runtime_dir = Path(runtime_dir or data_dir() / "runtime" / "intel-embedding")
    def cpu():
        return cpu_factory(model, cache_dir=cache_dir, batch_size=batch_size, local_only=local_only)
    if device == "cpu":
        backend = cpu()
        backend.requested_device = device
        return backend
    try:
        gpu = gpu_factory(model, runtime_dir=runtime_dir, cache_dir=data_dir() / "cache" / "openvino",
                          batch_size=batch_size, require_qualification=True)
    except Exception as exc:
        if device == "intel_gpu":
            if isinstance(exc, IntelEmbeddingUnavailable):
                raise
            raise IntelEmbeddingUnavailable(f"intel_embedding_backend_failed: {exc}") from exc
        backend = AutoFallbackBackend(cpu(), cpu, fallback_reason=str(exc))
        return backend
    if device == "intel_gpu":
        gpu.requested_device = device
        return StrictBackend(gpu)
    return AutoFallbackBackend(gpu, cpu)
