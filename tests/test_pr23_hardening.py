import re
from pathlib import Path
from urllib.parse import urlparse

from axp_client.rag.llama_cpp_backend import classify_load_failure
from axp_client.rag.model_catalog import MODELS, catalog_model


def test_curated_catalog_is_exact_and_immutable():
    for model in MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", model.sha256)
        assert model.repository and model.filename.endswith(".gguf") and model.size_bytes > 0
        assert urlparse(model.url).scheme == "https" and model.revision in model.url
        assert not any(part in model.url.lower().split("/") for part in ("main", "master", "latest"))
    qwen = catalog_model("qwen3-1.7b-q4km")
    assert (qwen.size_bytes, qwen.revision, qwen.sha256) == (
        1_282_439_264, "daeb8e2d528a760970442092f6bf1e55c3b659eb",
        "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5")
    smol = catalog_model("smollm3-3b-q4km")
    assert (smol.size_bytes, smol.revision, smol.sha256) == (
        1_915_305_312, "4965cb60b150737b68a0408c36aeefb65078f894",
        "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e")


def test_windows_illegal_instruction_is_non_retryable(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    for exc in (OSError("Windows Error 0xc000001d"), OSError("[WinError -1073741795]")):
        failure = classify_load_failure(exc, model)
        assert failure["failure_type"] == "backend_cpu_incompatible"
        assert failure["retryable"] is False


def test_frontend_selected_ready_and_retry_contract():
    source = (Path(__file__).parents[1] / "client/axp_client/web/ask.js").read_text()
    assert "model.model_loaded ? ' · ACTIVE · READY'" in source
    assert "model.model_state === 'loading' ? ' · SELECTED · LOADING'" in source
    assert "model.model_state === 'failed' ? ' · SELECTED · LOAD FAILED'" in source
    assert "state.retryable === true" in source


def test_static_assets_require_revalidation():
    source = (Path(__file__).parents[1] / "client/axp_client/server.py").read_text()
    assert 'self.send_header("Cache-Control", "no-cache")' in source


def test_ci_wheel_index_and_release_source_build_are_distinct():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements-runtime.txt").read_text()
    release = (root / ".github/workflows/release.yml").read_text()
    assert "https://abetlen.github.io/llama-cpp-python/whl/cpu" in requirements
    assert "llama-cpp-python==0.3.23" in requirements
    assert "--no-binary llama-cpp-python" in release
    assert "-DGGML_NATIVE=OFF" in release
