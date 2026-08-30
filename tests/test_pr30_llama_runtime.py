import importlib.util
import sys
from pathlib import Path
from types import ModuleType
VERIFIER_PATH = Path(__file__).parents[1] / "scripts/verify_llama_runtime.py"
SPEC = importlib.util.spec_from_file_location("pr30_verify_llama_runtime", VERIFIER_PATH)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_runtime_pin_is_exactly_0324():
    requirements = (Path(__file__).parents[1] / "requirements-runtime.txt").read_text(encoding="utf-8")
    assert "llama-cpp-python==0.3.24" in requirements
    assert "llama-cpp-python==0.3.23" not in requirements
    assert VERIFIER.EXPECTED_BACKEND_VERSION == "0.3.24"


def test_forbidden_isa_policy_remains_avx_only():
    assert VERIFIER.verify_features("AVX = 1 | AVX2 = 0 | BMI2 = 0 | FMA = 0 | F16C = 0") == {
        "AVX": True, "AVX2": False, "BMI2": False, "FMA": False, "F16C": False,
    }
    assert {"AVX2", "AVX_VNNI", "BMI2", "FMA", "F16C", "LLAMAFILE"} <= VERIFIER.FORBIDDEN_FEATURES
    assert all(feature.startswith("AVX512") for feature in VERIFIER.FORBIDDEN_FEATURES
               if feature.startswith("AVX512"))


def test_generation_tag_probe_compiles_and_checks_rendered_body(monkeypatch):
    calls = {}

    class Formatter:
        def __init__(self, *, template, eos_token, bos_token):
            calls.update(template=template, eos_token=eos_token, bos_token=bos_token)

        def __call__(self, *, messages):
            calls["messages"] = messages
            return type("Rendered", (), {"prompt": "assistant text"})()

    package = ModuleType("llama_cpp")
    chat_format = ModuleType("llama_cpp.llama_chat_format")
    chat_format.Jinja2ChatFormatter = Formatter
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat_format)

    assert VERIFIER.verify_hf_generation_tag() is True
    assert calls["template"] == "{% generation %}assistant text{% endgeneration %}"
    assert calls["messages"] == []
