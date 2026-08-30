import zipfile
from pathlib import Path

import pytest

from scripts.prune_portable_runtime import audit, prune, verify, verify_zip


def test_sanitizer_audits_removes_and_rejects_cache_artifacts(tmp_path):
    cache = tmp_path / "package" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"cache")
    (tmp_path / ".ruff_cache").mkdir()
    before = audit(tmp_path)
    assert before.pycache_directories == 1 and before.pyc_files == 1
    with pytest.raises(RuntimeError, match="Forbidden cache"):
        verify(tmp_path)
    prune(tmp_path)
    assert verify(tmp_path)
    assert audit(tmp_path).pycache_directories == 0


def test_zip_verifier_requires_store_and_rejects_cache(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("AXPIndexerNG/AXPIndexerTray.pyw", "")
        output.writestr("AXPIndexerNG/python/pythonw.exe", "")
    result = verify_zip(archive)
    assert result["files"] == 2 and result["cache_artifacts"] == 0
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("AXPIndexerNG/__pycache__/bad.pyc", "bad")
    with pytest.raises(RuntimeError, match="forbidden cache"):
        verify_zip(archive)


@pytest.mark.parametrize(
    "artifact",
    (
        "AXPIndexerNG/package/__pycache__/",
        "AXPIndexerNG/package/module.pyc",
        "AXPIndexerNG/package/module.pyo",
    ),
)
def test_zip_verifier_rejects_every_bytecode_artifact(tmp_path, artifact):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("AXPIndexerNG/AXPIndexerTray.pyw", "")
        output.writestr("AXPIndexerNG/python/pythonw.exe", "")
        output.writestr(artifact, b"")
    with pytest.raises(RuntimeError, match="forbidden cache"):
        verify_zip(archive)


def test_release_policy_guards(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "prune_portable_runtime.py prune" in workflow
    assert "prune_portable_runtime.py verify" in workflow
    assert "verify-zip" in workflow and "Post-prune" in workflow
    assert "CompressionLevel NoCompression" in workflow
    unsafe = "taskkill /" + "IM pythonw.exe"
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            try:
                assert unsafe.casefold() not in path.read_text(encoding="utf-8").casefold()
            except UnicodeDecodeError:
                pass
