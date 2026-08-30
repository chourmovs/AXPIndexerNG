"""Verify curated Hugging Face metadata without downloading GGUF files."""
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT / "client", ROOT / "shared"):
    sys.path.insert(0, str(package_root))


def _value(value, *names):
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def verify_entry(api, model):
    """Return canonical repository metadata or raise on any catalog mismatch."""
    parsed = urlparse(model.url)
    immutable = bool(re.fullmatch(r"[0-9a-f]{40}", model.revision))
    path_parts = {part.lower() for part in parsed.path.split("/")}
    if (parsed.scheme != "https" or not immutable
            or f"/resolve/{model.revision}/" not in parsed.path
            or path_parts.intersection({"main", "master", "latest"})):
        raise ValueError("catalog URL is not immutable HTTPS")

    files = api.get_paths_info(repo_id=model.repository, paths=[model.filename],
                               revision=model.revision, repo_type="model")
    if len(files) != 1 or _value(files[0], "path", "rfilename") != model.filename:
        raise ValueError("catalog file is missing at the immutable revision")
    repo_file = files[0]
    remote_size = _value(repo_file, "size")
    if remote_size != model.size_bytes:
        raise ValueError(f"expected {model.size_bytes} bytes, remote reports {remote_size}")
    lfs = _value(repo_file, "lfs")
    if lfs is None:
        raise ValueError("canonical LFS metadata is missing")
    # The Hub JSON calls this `oid`; BlobLfsInfo exposes the same value as
    # `sha256` in currently supported huggingface_hub releases.
    content_sha256 = _value(lfs, "oid", "sha256")
    if content_sha256 != model.sha256:
        raise ValueError("remote LFS SHA-256 identity does not match the catalog")
    xet = _value(repo_file, "xet_hash")
    if xet is None:
        xet_data = _value(repo_file, "xet_file_data")
        xet = _value(xet_data, "file_hash", "hash") if xet_data is not None else None
    return {"size": remote_size, "sha256": content_sha256, "xet_hash": xet}


def main():
    from huggingface_hub import HfApi
    from axp_client.rag.model_catalog import MODELS

    api = HfApi()
    failures = []
    for model in MODELS:
        try:
            metadata = verify_entry(api, model)
            lines = [f"{model.id}:", "  revision resolves", f"  size {metadata['size']}",
                     f"  SHA-256 {metadata['sha256']}"]
            if metadata["xet_hash"]:
                lines.append(f"  Xet {metadata['xet_hash']}")
            print("\n".join(lines))
        except Exception as exc:
            failures.append(f"{model.id}: metadata verification failed ({exc})")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
