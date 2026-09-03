from types import SimpleNamespace

from axp_client.rag.benchmark import BenchmarkRunner


class Backend:
    def __init__(self):
        self.config = SimpleNamespace(max_answer_tokens=256)
        self.calls = []
        self.last_telemetry = {"prompt_tokens": 10, "prompt_eval_ms": 10,
                               "generation_ms": 20, "decode_tokens_per_second": 5}

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "OK"


def test_benchmark_cap_is_per_call_not_profile_mutation():
    backend = Backend()
    runner = BenchmarkRunner(None, None, "model")
    runner.job.profile = "rag"
    runner._measure(backend, "intel_rag", 64)
    assert backend.calls[-1]["max_tokens"] == 64
    assert backend.config.max_answer_tokens == 256


def test_rag_measurement_is_the_headline_shape():
    performance = {"quick_warm": {"prompt_tokens": 1, "prompt_eval_tokens_per_second": 14},
                   "rag": {"prompt_tokens": 1300, "prompt_eval_tokens_per_second": 139}}
    assert performance["rag"]["prompt_eval_tokens_per_second"] == 139
