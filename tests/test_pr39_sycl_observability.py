import os
from axp_client.rag.hardware import detect_hardware
from axp_client.rag.intel_sycl_backend import IntelSyclBackend
from axp_client.rag.llama_cpp_backend import GenerationConfig
from axp_client.rag.sycl_probe import parse_device_list


def test_native_sycl_identity_is_preserved():
    assert parse_device_list("SYCL0: Intel(R) Iris(R) Xe Graphics") == [{
        "id": "SYCL0", "name": "Intel(R) Iris(R) Xe Graphics",
        "raw": "SYCL0: Intel(R) Iris(R) Xe Graphics"}]


def test_sidecar_command_binds_one_probed_device_at_trace_level(tmp_path):
    backend = IntelSyclBackend(tmp_path / "model.gguf", GenerationConfig(), tmp_path,
                               sycl_device_id="SYCL0", sycl_device_name="Intel Iris Xe")
    backend._port = 1234
    backend._auth_file = tmp_path / "secret-file"
    command = backend._sidecar_command()
    assert command[command.index("--device") + 1] == "SYCL0"
    assert command[command.index("--split-mode") + 1] == "none"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("-lv") + 1] == "4"
    assert "api-key-material" not in " ".join(command)


def test_qualification_uses_deeper_child_only_debug(tmp_path):
    backend = IntelSyclBackend(tmp_path / "model.gguf", GenerationConfig(), tmp_path,
                               sycl_device_id="SYCL0", diagnostic=True)
    assert backend.verbosity == 5
    assert "GGML_SYCL_DEBUG" not in os.environ


def test_probe_excerpts_survive_hardware_capabilities(monkeypatch, tmp_path):
    server = tmp_path / "llama-server.exe"; server.write_bytes(b"MZ")
    monkeypatch.setattr("axp_client.rag.hardware.platform.system", lambda: "Windows")
    monkeypatch.setattr("axp_client.rag.hardware.windows_display_adapters", lambda: [{
        "DeviceString": "Intel Iris Xe", "DeviceID": "PCI\\VEN_8086&DEV_1234", "StateFlags": 1}])
    monkeypatch.setattr("axp_client.rag.hardware.probe_sycl", lambda *_: {
        "installed": True, "ok": True, "device_id": "SYCL0", "device_name": "Intel Iris Xe",
        "device_count": 1, "error_code": None, "returncode": 0, "duration_ms": 7,
        "stdout_excerpt": "SYCL0: Intel Iris Xe", "stderr_excerpt": "Level Zero ready"})
    result = detect_hardware(server, tmp_path)
    assert result.sycl_device_id == "SYCL0"
    assert result.sycl_probe_stdout_excerpt == "SYCL0: Intel Iris Xe"
    assert result.sycl_probe_stderr_excerpt == "Level Zero ready"
