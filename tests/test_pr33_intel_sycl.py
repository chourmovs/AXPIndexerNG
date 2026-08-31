import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from axp_client.rag.accelerator_catalog import INTEL_SYCL
from axp_client.rag.accelerator_manager import AcceleratorError, discover_binaries, safe_extract
from axp_client.rag.benchmark import benchmark_prompt, compare_results, safe_ratio
from axp_client.rag.intel_sycl_backend import LOOPBACK, parse_sse
from axp_client.rag.sycl_probe import parse_device_list, probe_sycl


def test_catalog_is_exactly_release_pinned():
    assert (INTEL_SYCL.id, INTEL_SYCL.upstream_repository, INTEL_SYCL.tag, INTEL_SYCL.commit) == (
        "intel-sycl-b10516", "ggml-org/llama.cpp", "b10516", "b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9")
    assert INTEL_SYCL.asset == "llama-b10516-bin-win-sycl-x64.zip"
    assert INTEL_SYCL.exact_size == 119_741_566
    assert INTEL_SYCL.sha256 == "b9a5a42ddc4033f05003b127d3fd18583565b33971eb01723c6711e95ece42b4"
    assert INTEL_SYCL.url.endswith(f"/{INTEL_SYCL.tag}/{INTEL_SYCL.asset}")


def test_secure_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle: bundle.writestr("../escape.exe", b"bad")
    with pytest.raises(AcceleratorError, match="accelerator_zip_traversal"):
        safe_extract(archive, tmp_path / "stage")


def test_binary_discovery_requires_one_nonempty_server(tmp_path):
    with pytest.raises(AcceleratorError, match="accelerator_server_missing"): discover_binaries(tmp_path)
    for folder in (tmp_path / "a", tmp_path / "b"):
        folder.mkdir(); (folder / "llama-server.exe").write_bytes(b"MZ")
    with pytest.raises(AcceleratorError, match="accelerator_server_ambiguous"): discover_binaries(tmp_path)


def test_device_parser_accepts_only_recognizable_intel_gpu():
    output = "0: Intel(R) Iris(R) Xe Graphics (GPU)\n1: Intel CPU\n2: NVIDIA GPU"
    assert parse_device_list(output) == [{"id": "0", "name": "Intel(R) Iris(R) Xe Graphics (GPU)",
                                          "raw": "0: Intel(R) Iris(R) Xe Graphics (GPU)"}]


def test_probe_contract(tmp_path):
    server = tmp_path / "llama-server.exe"; server.write_bytes(b"MZ")
    def runner(command, **kwargs):
        assert command == [str(server), "--list-devices"]
        assert kwargs["shell"] is False and kwargs["timeout"] == 15
        assert kwargs["cwd"] == str(server.parent)
        assert kwargs["env"]["ONEAPI_DEVICE_SELECTOR"] == "level_zero:gpu"
        return subprocess.CompletedProcess([], 0, "sycl0: Intel Arc Graphics GPU", "")
    result = probe_sycl(server, runner=runner)
    assert result["ok"] and result["device_count"] == 1


@pytest.mark.parametrize("line", ["Intel(R) Iris(R) Xe Graphics", "SYCL0: Intel(R) Iris(R) Xe Graphics",
                                  "Intel(R) Graphics [0x46A8]"])
def test_device_parser_accepts_realistic_gpu_names(line):
    assert parse_device_list(line)


@pytest.mark.parametrize("line", ["Intel(R) Core(TM) i7 CPU", "OpenCL CPU Intel", "Basic Display Adapter"])
def test_device_parser_rejects_non_gpu_devices(line):
    assert parse_device_list(line) == []


def test_probe_sanitizes_child_environment_and_prepends_both_paths(tmp_path, monkeypatch):
    root = tmp_path / "runtime"; server_dir = root / "bin"; server_dir.mkdir(parents=True)
    server = server_dir / "llama-server.exe"; server.write_bytes(b"MZ")
    monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "bad")
    monkeypatch.setenv("SYCL_DEVICE_FILTER", "bad")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "bad")
    def runner(_command, **kwargs):
        env = kwargs["env"]
        assert "SYCL_DEVICE_FILTER" not in env and "ZE_AFFINITY_MASK" not in env
        assert env["PATH"].split(__import__("os").pathsep)[:2] == [str(server_dir), str(root)]
        return subprocess.CompletedProcess([], 0, "Intel Graphics", "")
    assert probe_sycl(server, root, runner=runner)["ok"]


def test_only_authoritative_probe_owns_list_devices_contract():
    rag = Path(__file__).parents[1] / "client" / "axp_client" / "rag"
    owners = [path.name for path in rag.glob("*.py") if '"--list-devices"' in path.read_text(encoding="utf-8")]
    assert owners == ["sycl_probe.py"]


def test_sse_role_reasoning_and_done_are_transport_safe():
    lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n', b'\n',
             b'data: {"choices":[{"delta":{"reasoning_content":"secret","content":"answer"}}]}\n',
             b'\n', b'data: [DONE]\n']
    events = list(parse_sse(lines))
    assert len(events) == 3 and json.loads(events[1])["choices"][0]["delta"]["content"] == "answer"
    assert events[-1] == "[DONE]"


def test_benchmark_is_deterministic_and_ratios_are_truthful():
    assert benchmark_prompt("quick") == benchmark_prompt("quick")
    cpu = {"warm": {"ttft_ms": 100, "generation_ms": 120, "decode_tps": 2}}
    gpu = {"warm": {"ttft_ms": 50, "generation_ms": 60, "decode_tps": 6}}
    speedup, assessment = compare_results(cpu, gpu)
    assert speedup == {"warm_ttft": 2, "warm_generation": 2, "warm_decode": 3}
    assert assessment == "intel_gpu_promising" and safe_ratio(1, 0) is None


def test_sidecar_contract_is_loopback_only():
    assert LOOPBACK == "127.0.0.1"
