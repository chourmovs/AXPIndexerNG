from pathlib import Path

import pytest

from axp_client.rag.hardware import HardwareCapabilities
from axp_client.rag.runtime_manager import InferenceDeviceError, InferenceRuntimeManager
from axp_client.rag.retrieval import rank_documents
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance, diversify


class Backend:
    def __init__(self, name): self.name = name
    def health(self):
        return {"available": True, "backend": self.name, "model_state": "unloaded",
                "gpu_offload_confirmed": self.name == "intel_sycl"}
    def ensure_loaded(self): return None
    def close(self): return None


def hardware(ok, error=None):
    return HardwareCapabilities("cpu", True, "Intel Iris Xe", ok,
        None if ok else error, sycl_runtime_installed=True, sycl_probe_ok=ok,
        sycl_device_id="SYCL0" if ok else None, sycl_device_name="Intel Iris Xe" if ok else None,
        sycl_probe_error=error)


def settings(device):
    return {"chat_model_path": "model.gguf", "chat_active_model_id": None,
            "chat_inference_device": device}


def test_explicit_intel_timeout_is_pending_then_qualifies_without_cpu():
    cpu = []
    runtime = InferenceRuntimeManager(settings("intel_gpu"), hardware=hardware(False, "intel_sycl_probe_timeout"),
        backend_factory=lambda *_: cpu.append(True) or Backend("cpu"),
        intel_backend_factory=lambda *_: Backend("intel_sycl"),
        hardware_probe=lambda *_, **__: hardware(True))
    assert runtime.health()["accelerator_state"] == "probe_timeout"
    assert runtime.health()["inference_device_effective"] == "intel_gpu" and not cpu
    runtime.ensure_loaded()
    assert runtime.health()["backend"] == "intel_sycl" and not cpu


def test_explicit_intel_second_timeout_is_stable_error_without_cpu():
    cpu = []
    runtime = InferenceRuntimeManager(settings("intel_gpu"), hardware=hardware(False, "intel_sycl_probe_timeout"),
        backend_factory=lambda *_: cpu.append(True) or Backend("cpu"),
        hardware_probe=lambda *_, **__: hardware(False, "intel_sycl_probe_timeout"))
    with pytest.raises(InferenceDeviceError, match="intel_sycl_probe_timeout"):
        runtime.ensure_loaded()
    assert not cpu and runtime.health()["retryable"]


def test_auto_fallback_and_explicit_cpu_are_authoritative():
    made = []
    for device in ("auto", "cpu"):
        runtime = InferenceRuntimeManager(settings(device), hardware=hardware(False, "intel_sycl_probe_timeout"),
            backend_factory=lambda *_: made.append("cpu") or Backend("cpu"),
            intel_backend_factory=lambda *_: made.append("intel") or Backend("intel_sycl"))
        assert runtime.health()["inference_device_effective"] == "cpu"
    assert made == ["cpu", "cpu"]


def scored(text, *, title="MSDS MTBE SIMFEX", filename="MSDS MTBE SIMFEX.pdf", vector=.65):
    row = {"snippet": text, "identifiers": "", "title": title, "filename": filename, "heading": "",
           "vector_distance": 1-vector, "lexical_rank": 1, "exact_identifier_match": False,
           "exact_phrase_match": False, "exact_filename_match": False, "exact_priority": 0}
    _relevance(row, _meaningful_terms("density of MTBE"), SearchConfig(min_lexical_coverage=.1))
    return row


def test_mtbe_document_and_content_passage_are_ranked_separately():
    passages = [scored("General MTBE product identification."),
                scored("Storage and handling recommendations."),
                scored("Physical properties. Density at 20°C: 0.74 g/cm³.", vector=.72)]
    for index, row in enumerate(passages, 1):
        row.update(document_id=1, chunk_id=index)
    other = scored("A density table for another solvent", title="Other", filename="other.pdf", vector=.61)
    other.update(document_id=2, chunk_id=4)
    ranked = rank_documents([*passages, other])
    assert ranked[0]["filename"] == "MSDS MTBE SIMFEX.pdf"
    assert ranked[0]["ranked_hits"][0]["chunk_id"] == 3
    assert passages[2]["passage_score"] - passages[1]["passage_score"] > .15


def test_stable_cap_preserves_global_passage_order():
    rows = [{"document_id": 1, "passage_score": score} for score in (.9, .8, .7, .6)]
    rows += [{"document_id": 2, "passage_score": .65}]
    assert [row["passage_score"] for row in diversify(rows, 5, 3)] == [.9, .8, .7, .65]


def test_search_more_includes_latency_skip_and_depth_guard():
    source = (Path(__file__).parents[1] / "client/axp_client/web/ask.js").read_text(encoding="utf-8")
    assert "local_generation_skipped_latency_budget'].includes(response.status)" in source
    assert "response.context?.search_depth !== 1" in source
