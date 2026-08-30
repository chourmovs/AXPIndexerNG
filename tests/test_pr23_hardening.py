import importlib.util
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from axp_client.rag.llama_cpp_backend import LlamaCppBackend, classify_load_failure
from axp_client.rag.model_catalog import MODELS, catalog_model

VERIFIER_PATH = Path(__file__).parents[1] / "scripts/verify_chat_model_catalog.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location("verify_chat_model_catalog", VERIFIER_PATH)
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER)
verify_entry = VERIFIER.verify_entry


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


@pytest.mark.parametrize(("model_id", "xet_hash"), (
    ("qwen3-1.7b-q4km", "0a8e661bad7f1ea5accdd078b6a2aca20ff0201100bbf128aa1cc22c643d7221"),
    ("smollm3-3b-q4km", "777b1c9982e98ca62b4a6a16914bb6bfb7d07585714aea681276e96c90aa0f04"),
))
def test_catalog_verifier_accepts_distinct_xet_and_lfs_identities(model_id, xet_hash):
    model = catalog_model(model_id)
    repo_file = SimpleNamespace(path=model.filename, size=model.size_bytes,
                                lfs=SimpleNamespace(oid=model.sha256), xet_hash=xet_hash)

    class Api:
        def get_paths_info(self, **kwargs):
            assert kwargs == {"repo_id": model.repository, "paths": [model.filename],
                              "revision": model.revision, "repo_type": "model"}
            return [repo_file]

    result = verify_entry(Api(), model)
    assert result == {"size": model.size_bytes, "sha256": model.sha256, "xet_hash": xet_hash}
    assert result["xet_hash"] != result["sha256"]


@pytest.mark.parametrize("failure", ("size", "sha256", "mutable", "missing", "lfs"))
def test_catalog_verifier_rejects_invalid_canonical_metadata(failure):
    model = catalog_model("qwen3-1.7b-q4km")
    checked_model = replace(model, revision="main") if failure == "mutable" else model
    lfs = None if failure == "lfs" else {"oid": "0" * 64 if failure == "sha256" else model.sha256}
    repo_file = {"path": model.filename, "size": model.size_bytes + (1 if failure == "size" else 0),
                 "lfs": lfs, "xet_hash": "0a8e661bad7f1ea5accdd078b6a2aca20ff0201100bbf128aa1cc22c643d7221"}

    class Api:
        def get_paths_info(self, **_kwargs):
            return [] if failure == "missing" else [repo_file]

    with pytest.raises(ValueError):
        verify_entry(Api(), checked_model)


def test_windows_illegal_instruction_is_non_retryable(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    for exc in (OSError("Windows Error 0xc000001d"), OSError("[WinError -1073741795]")):
        failure = classify_load_failure(exc, model)
        assert failure["failure_type"] == "backend_cpu_incompatible"
        assert failure["retryable"] is False


def test_avx_preflight_is_cpu_incompatible_and_non_retryable(tmp_path):
    backend = LlamaCppBackend(tmp_path / "unused.gguf")
    backend.cpu = SimpleNamespace(runtime_cpu_compatible=False,
                                  public=lambda: {"runtime_cpu_compatible": False})
    with pytest.raises(OSError, match="AVX state"):
        backend.ensure_loaded()
    health = backend.health()
    assert health["failure_type"] == health["reason"] == "backend_cpu_incompatible"
    assert health["failure_code"] == "avx_unavailable"
    assert health["retryable"] is False


def test_frontend_selected_ready_and_retry_contract():
    source = (Path(__file__).parents[1] / "client/axp_client/web/ask.js").read_text(encoding="utf-8")
    assert "model.model_loaded ? ' · ACTIVE · READY'" in source
    assert "model.model_state === 'loading' ? ' · SELECTED · LOADING'" in source
    assert "model.model_state === 'failed' ? ' · SELECTED · LOAD FAILED'" in source
    assert "state.retryable === true" in source
    assert "backend_cpu_incompatible: 'This AXP build requires CPU instructions unavailable on this PC.'" in source
    assert "state.reason === 'backend_cpu_incompatible'" in source


def test_static_assets_require_revalidation():
    source = (Path(__file__).parents[1] / "client/axp_client/server.py").read_text(encoding="utf-8")
    assert 'self.send_header("Cache-Control", "no-cache")' in source


def test_ci_wheel_index_and_release_source_build_are_distinct():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements-runtime.txt").read_text(encoding="utf-8")
    release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "https://abetlen.github.io/llama-cpp-python/whl/cpu" in requirements
    assert "llama-cpp-python==0.3.23" in requirements
    assert "--no-binary llama-cpp-python" in release
    assert "-DGGML_NATIVE=OFF" in release
    assert "verify_llama_runtime.py" in release
