import json

from axp_core.build_info import build_info


def test_valid_release_build_info_is_safely_exposed(tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    path.write_text(json.dumps({"version": "v0.3.4-alpha5", "commit": "abcdef1234567890", "release": True}))
    assert build_info(path) == {"version": "v0.3.4-alpha5", "commit": "abcdef1", "release": True}


def test_valid_development_build_info(tmp_path):
    path = tmp_path / "BUILD_INFO.json"
    path.write_text(json.dumps({"version": "dev-abcdef1", "commit": "abcdef1234567890", "release": False}))
    assert build_info(path)["version"] == "dev-abcdef1"


def test_missing_or_malformed_build_info_never_crashes(tmp_path):
    fallback = {"version": "dev", "commit": None, "release": False}
    assert build_info(tmp_path / "missing.json") == fallback
    malformed = tmp_path / "BUILD_INFO.json"
    malformed.write_text("{")
    assert build_info(malformed) == fallback
    malformed.write_text(json.dumps({"version": "<script>", "commit": "nope", "release": "yes"}))
    assert build_info(malformed) == fallback


def test_environment_build_info_override(monkeypatch, tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"version": "v1.2.3", "commit": "1234567", "release": True}))
    monkeypatch.setenv("AXP_BUILD_INFO", str(path))
    assert build_info() == {"version": "v1.2.3", "commit": "1234567", "release": True}
