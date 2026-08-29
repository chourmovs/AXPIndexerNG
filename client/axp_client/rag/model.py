import hashlib
import json
import os
import struct
import tempfile
import time
from pathlib import Path

GGUF_MAGIC = b"GGUF"
MIN_GGUF_SIZE = 24
MANIFEST_NAME = "model.manifest.json"


def validate_gguf(path, *, minimum_size=MIN_GGUF_SIZE):
    path = Path(path)
    if not path.exists():
        return False, "model_missing"
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return False, "model_invalid"
    try:
        if path.stat().st_size < minimum_size:
            return False, "model_invalid"
        with path.open("rb") as handle:
            magic, version = handle.read(4), struct.unpack("<I", handle.read(4))[0]
        if magic != GGUF_MAGIC or version not in (2, 3):
            return False, "model_invalid"
    except (OSError, struct.error):
        return False, "model_invalid"
    return True, None


def manifest_path(model_path):
    return Path(model_path).with_name(MANIFEST_NAME)


def import_model(source, destination):
    source, destination = Path(source), Path(destination)
    valid, reason = validate_gguf(source)
    if not valid:
        raise ValueError(reason)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    descriptor, temporary_name = tempfile.mkstemp(prefix="model.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while data := reader.read(1024 * 1024):
                digest.update(data)
                writer.write(data)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        manifest = {"filename": source.name, "size_bytes": destination.stat().st_size,
                    "sha256": digest.hexdigest(), "imported_ms": int(time.time() * 1000)}
        manifest_tmp = manifest_path(destination).with_suffix(".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path(destination))
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def model_status(model_path):
    path = Path(model_path)
    valid, reason = validate_gguf(path)
    manifest = None
    try:
        manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    manifest_match = bool(valid and manifest and path.stat().st_size == manifest.get("size_bytes"))
    if valid and manifest and not manifest_match:
        reason = "model_changed"
    return {"configured": path.is_file(), "valid": valid, "reason": reason, "manifest": manifest,
            "manifest_match": manifest_match}


def verify_model(model_path):
    """Explicit local verification; unlike health/status, this hashes the whole model."""
    path = Path(model_path)
    status = model_status(path)
    actual_sha256 = None
    if status["valid"]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1024 * 1024):
                digest.update(data)
        actual_sha256 = digest.hexdigest()
    expected = (status.get("manifest") or {}).get("sha256")
    sha256_match = bool(actual_sha256 and expected and actual_sha256 == expected)
    return {**status, "manifest_match": bool(status["manifest_match"] and sha256_match),
            "sha256_match": sha256_match}


def remove_model(model_path):
    Path(model_path).unlink(missing_ok=True)
    manifest_path(model_path).unlink(missing_ok=True)
