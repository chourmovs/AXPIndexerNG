import pytest

from axp_daemon.embedding_backends import BackendFailure, IntelEmbeddingUnavailable, create_indexing_embedder


class FakeCPU:
    backend = "fastembed"
    effective_device = "cpu"
    intel_gpu_qualified = False
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension = 384
    distance_metric = "cosine"

    def __init__(self, model, **kwargs):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dimension

    def health(self):
        return {"backend": self.backend, "healthy": True}


class FakeGPU(FakeCPU):
    backend = "openvino_genai"
    effective_device = "intel_gpu"
    intel_gpu_qualified = True
    device_name = "Fake Intel GPU"


def test_cpu_never_constructs_gpu():
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU factory called")

    value = create_indexing_embedder("balanced", device="cpu", cpu_factory=FakeCPU, gpu_factory=forbidden)
    assert value.effective_device == "cpu"
    assert value.dimension == 384


def test_auto_selects_qualified_gpu():
    value = create_indexing_embedder("balanced", device="auto", cpu_factory=FakeCPU, gpu_factory=FakeGPU)
    assert value.effective_device == "intel_gpu"
    assert value.model_id == FakeCPU.model_id


def test_forced_gpu_requires_accelerator():
    def unavailable(*args, **kwargs):
        raise IntelEmbeddingUnavailable("intel_embedding_runtime_missing")

    with pytest.raises(IntelEmbeddingUnavailable, match="runtime_missing"):
        create_indexing_embedder("balanced", device="intel_gpu", cpu_factory=FakeCPU, gpu_factory=unavailable)


def test_auto_retries_failed_batch_once_on_cpu_and_stays_cpu():
    class FailingGPU(FakeGPU):
        def embed_documents(self, texts):
            raise RuntimeError("device out of memory")

    value = create_indexing_embedder("balanced", device="auto", cpu_factory=FakeCPU, gpu_factory=FailingGPU)
    failed_batch = ["one", "two"]
    assert len(value.embed_documents(failed_batch)) == 2
    assert value.effective_device == "cpu"
    assert "out of memory" in value.fallback_reason
    value.embed_documents(["three"])
    assert value._active.calls == [failed_batch, ["three"]]


def test_strict_gpu_converts_runtime_error_to_backend_failure():
    class FailingGPU(FakeGPU):
        def embed_documents(self, texts):
            raise RuntimeError("device lost")

    value = create_indexing_embedder("balanced", device="intel_gpu", cpu_factory=FakeCPU,
                                     gpu_factory=FailingGPU)
    with pytest.raises(BackendFailure, match="device lost"):
        value.embed_documents(["document"])
