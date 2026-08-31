"""Central policy and conservative performance estimates for interactive local RAG."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class InteractiveLatencyPolicy:
    preferred_seconds: float = 30.0
    hard_seconds: float = 60.0
    evidence_ladder: tuple[int, ...] = (1200, 800, 500, 300)
    minimum_evidence_tokens: int = 300
    # A deliberately conservative cold-start estimate, based on field measurements.
    default_cpu_prompt_tps: float = 9.0


POLICY = InteractiveLatencyPolicy()


def estimate_prefill_seconds(prompt_tokens, prompt_eval_tokens_per_second):
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens < 0:
        return None
    if not isinstance(prompt_eval_tokens_per_second, (int, float)) or prompt_eval_tokens_per_second <= 0:
        return None
    return float(prompt_tokens) / float(prompt_eval_tokens_per_second)


class PerformanceEstimates:
    """Process-local, conservative EWMA keyed by model and effective device.

    Upward changes are intentionally damped so one unusually fast run cannot make
    context selection suddenly optimistic. Downward changes are learned promptly.
    """
    def __init__(self):
        self._values = {}
        self._lock = threading.Lock()

    def get(self, model_id, device):
        with self._lock:
            value = self._values.get((model_id, device))
            return dict(value) if value else None

    def update(self, model_id, device, *, prompt_tps=None, decode_tps=None):
        if not model_id or device not in ("cpu", "intel_gpu"):
            return None
        with self._lock:
            old = self._values.get((model_id, device), {})
            result = dict(old)
            for key, sample in (("prompt_eval_tokens_per_second", prompt_tps),
                                ("decode_tokens_per_second", decode_tps)):
                if not isinstance(sample, (int, float)) or sample <= 0:
                    continue
                previous = old.get(key)
                # Learn regressions faster (50%) than improvements (10%).
                weight = .5 if previous is not None and sample < previous else .1
                result[key] = float(sample) if previous is None else previous * (1 - weight) + sample * weight
            result["sample_count"] = int(old.get("sample_count", 0)) + 1
            result["last_updated"] = time.time()
            self._values[(model_id, device)] = result
            return dict(result)


PERFORMANCE_ESTIMATES = PerformanceEstimates()
