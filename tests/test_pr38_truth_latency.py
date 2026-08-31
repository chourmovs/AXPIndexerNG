import pytest

from axp_client.rag.intel_sycl_backend import IntelSyclBackend
from axp_client.rag.latency import InteractiveLatencyPolicy, PerformanceEstimates, estimate_prefill_seconds
from axp_client.rag.llama_cpp_backend import GenerationConfig


def backend(tmp_path):
    return IntelSyclBackend(tmp_path / "model.gguf", GenerationConfig(), tmp_path)


def test_intel_native_proof_is_structured_and_sticky(tmp_path):
    value = backend(tmp_path)
    for line in ("Intel GPU discovered\n", "offloading 30 repeating layers to GPU\n",
                 "offloading output layer to GPU\n", "offloaded 31/31 layers to GPU\n",
                 "SYCL0 model buffer size = 1024 MiB\n", "server listening\n",
                 "slot available\n", "request complete\n"):
        value._record_native_evidence(line)
    assert value.gpu_offload_confirmed is True
    assert (value.offloaded_layers, value.total_layers) == (31, 31)
    assert value.gpu_buffer_bytes == 1024 * 1024 * 1024
    assert len(value.native_gpu_markers) <= 32


def test_generic_intel_runtime_lines_are_not_offload_proof(tmp_path):
    value = backend(tmp_path)
    for line in ("SYCL runtime initialized", "Intel GPU detected", "Level Zero available",
                 "llama_server: model loaded", "server listening"):
        value._record_native_evidence(line)
    assert value.gpu_offload_confirmed is False
    assert value.offloaded_layers is None
    assert value.gpu_buffer_bytes is None


def test_latency_estimates_match_field_and_gpu_cases():
    assert estimate_prefill_seconds(2800, 9) == pytest.approx(311.111, rel=1e-3)
    assert estimate_prefill_seconds(2800, 9) > InteractiveLatencyPolicy().hard_seconds
    assert estimate_prefill_seconds(800, 150) == pytest.approx(5.333, rel=1e-3)
    assert estimate_prefill_seconds(800, 150) < InteractiveLatencyPolicy().preferred_seconds


def test_performance_estimator_dampens_anomalous_fast_sample():
    estimates = PerformanceEstimates()
    estimates.update("model", "cpu", prompt_tps=10, decode_tps=3)
    result = estimates.update("model", "cpu", prompt_tps=1000, decode_tps=300)
    assert result["prompt_eval_tokens_per_second"] == pytest.approx(109)
    assert result["sample_count"] == 2
