"""Sequential, local-only CPU versus Intel GPU inference benchmark."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

PROFILES = {
    "quick": {"words": 256, "max_tokens": 32},
    "rag": {"words": 1536, "max_tokens": 64},
}
BENCHMARK_STATES = {"idle", "preparing", "cpu_loading", "cpu_cold", "cpu_warm", "intel_loading",
                    "intel_cold", "intel_warm", "comparing", "complete", "complete_with_errors", "failed", "cancelled"}


def benchmark_prompt(profile="quick"):
    spec = PROFILES[profile]
    seed = "local deterministic benchmark evidence measures inference latency without company documents"
    words = (seed.split() * ((spec["words"] // len(seed.split())) + 1))[:spec["words"]]
    return " ".join(words) + "\nQuestion: Summarize the benchmark evidence briefly."


def safe_ratio(numerator, denominator):
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or denominator <= 0:
        return None
    return numerator / denominator


def compare_results(cpu, intel):
    cpu_warm, intel_warm = cpu.get("warm", {}), intel.get("warm", {})
    speedup = {"warm_ttft": safe_ratio(cpu_warm.get("ttft_ms"), intel_warm.get("ttft_ms")),
               "warm_generation": safe_ratio(cpu_warm.get("generation_ms"), intel_warm.get("generation_ms")),
               "warm_decode": safe_ratio(intel_warm.get("decode_tps"), cpu_warm.get("decode_tps"))}
    generation = speedup["warm_generation"]
    assessment = "mixed" if generation is None else (
        "intel_gpu_promising" if generation >= 1.2 else "cpu_faster" if generation <= 1 / 1.1 else "mixed")
    return speedup, assessment


@dataclass
class BenchmarkJob:
    state: str = "idle"
    profile: str = "quick"
    result: dict | None = None
    error: str | None = None
    started_at: float | None = None
    updated_at: float | None = None
    cancel_requested: bool = False
    def public(self): return asdict(self)


class BenchmarkRunner:
    """Background coordinator. Factories must create independent benchmark-only backends."""
    def __init__(self, cpu_factory, intel_factory, model_name, hardware=None):
        self.cpu_factory, self.intel_factory = cpu_factory, intel_factory
        self.model_name, self.hardware = model_name, hardware or {}
        self._lock = threading.Lock(); self._cancel = threading.Event(); self.job = BenchmarkJob(); self._current_backend = None

    def start(self, profile="quick"):
        if profile not in PROFILES: raise ValueError("invalid_benchmark_profile")
        with self._lock:
            if self.job.state not in ("idle", "complete", "complete_with_errors", "failed", "cancelled"):
                raise RuntimeError("benchmark_busy")
            now = time.time(); self.job = BenchmarkJob("preparing", profile, started_at=now, updated_at=now)
            self._cancel.clear()
        threading.Thread(target=self._run, daemon=True, name="axp-inference-benchmark").start()
        return self.job.public()

    def cancel(self):
        if self.job.state not in BENCHMARK_STATES - {"idle", "complete", "failed", "cancelled"}:
            raise RuntimeError("benchmark_not_active")
        self.job.cancel_requested = True; self._cancel.set()
        backend = self._current_backend
        if backend:
            if not getattr(backend, "request_load_cancel", lambda: False)():
                getattr(backend, "request_cancel", lambda: False)()
        return self.job.public()

    def _state(self, state): self.job.state = state; self.job.updated_at = time.time()

    def _measure(self, backend, state):
        self._state(state)
        prompt = benchmark_prompt(self.job.profile)
        prompt_tokens = backend.count_tokens(prompt)
        backend.generate(system_prompt="Answer only from the supplied synthetic benchmark text.",
                         user_prompt=prompt)
        if self._cancel.is_set(): raise InterruptedError
        telemetry = backend.last_telemetry
        ttft = telemetry.get("time_to_first_token_ms")
        prompt_eval_ms = telemetry.get("prompt_eval_ms") or ttft
        return {"prompt_tokens": telemetry.get("prompt_tokens") or prompt_tokens,
                "prompt_eval_ms": prompt_eval_ms,
                "prompt_eval_timing_derived": telemetry.get("prompt_eval_ms") is None,
                "prompt_eval_tokens_per_second": (telemetry.get("prompt_eval_tokens_per_second") or
                    (prompt_tokens / (prompt_eval_ms / 1000) if prompt_eval_ms else None)),
                "ttft_ms": ttft,
                "generation_ms": telemetry.get("generation_ms"),
                "completion_tokens": telemetry.get("completion_tokens"),
                "decode_ms": telemetry.get("decode_ms"),
                "decode_tps": telemetry.get("decode_tokens_per_second"),
                "decode_tokens_per_second": telemetry.get("decode_tokens_per_second")}

    def _backend_result(self, factory, prefix, label):
        backend = factory(PROFILES[self.job.profile]["max_tokens"])
        try:
            self._current_backend = backend
            self._state(f"{prefix}_loading"); started = time.perf_counter(); backend.ensure_loaded()
            load_ms = (time.perf_counter()-started)*1000
            cold = self._measure(backend, f"{prefix}_cold"); warm = self._measure(backend, f"{prefix}_warm")
            health = backend.health()
            return {"backend": label, "model_load_ms": load_ms, "cold": cold, "warm": warm,
                    **({key: health.get(key) for key in ("gpu_offload_confirmed", "offloaded_layers",
                        "total_layers", "gpu_buffer_bytes", "sycl_device_name")} if prefix == "intel" else {})}
        finally: backend.close(); self._current_backend = None

    def _run(self):
        try:
            cpu = self._backend_result(self.cpu_factory, "cpu", "llama-cpp-python 0.3.24 AVX")
            try: intel = self._backend_result(self.intel_factory, "intel", "llama.cpp b10516 SYCL")
            except InterruptedError: raise
            except Exception as exc:
                intel = {"status": "failed", "phase": "loading", "error": str(exc),
                         "gpu_offload_confirmed": False}
                self.job.result = {"model": self.model_name, "profile": self.job.profile,
                    "hardware": self.hardware, "cpu": cpu, "intel_gpu": intel,
                    "speedup": {}, "assessment": "intel_gpu_failed"}
                self._state("complete_with_errors"); return
            self._state("comparing"); speedup, assessment = compare_results(cpu, intel)
            self.job.result = {"model": self.model_name, "profile": self.job.profile, "hardware": self.hardware,
                               "cpu": cpu, "intel_gpu": intel, "speedup": speedup, "assessment": assessment}
            self._state("complete")
        except InterruptedError: self._state("cancelled")
        except Exception: self.job.error = "benchmark_failed"; self._state("failed")
