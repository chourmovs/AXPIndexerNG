"""Manually verify curated Hugging Face metadata without downloading GGUF files."""
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT / "client", ROOT / "shared"):
    sys.path.insert(0, str(package_root))

from axp_client.rag.model_catalog import MODELS  # noqa: E402
from axp_client.rag.model_manager import TrustedRedirectHandler  # noqa: E402


def main():
    opener = urllib.request.build_opener(TrustedRedirectHandler())
    failures = []
    for model in MODELS:
        try:
            parsed = urlparse(model.url)
            if parsed.scheme != "https" or f"/resolve/{model.revision}/" not in parsed.path:
                raise ValueError("catalog URL is not immutable HTTPS")
            with opener.open(urllib.request.Request(model.url, method="HEAD"), timeout=30) as response:
                linked_size = int(response.headers.get("X-Linked-Size") or response.headers["Content-Length"])
                linked_hash = (response.headers.get("X-Linked-Etag") or response.headers.get("ETag", "")).strip('"')
            if linked_size != model.size_bytes:
                failures.append(f"{model.id}: expected {model.size_bytes} bytes, remote reports {linked_size}")
            if linked_hash != model.sha256:
                failures.append(f"{model.id}: remote SHA-256 identity does not match the catalog")
            print(f"{model.id}: revision resolves; {linked_size} bytes; SHA-256 {linked_hash}")
        except (KeyError, OSError, ValueError, urllib.error.URLError) as exc:
            failures.append(f"{model.id}: metadata verification failed ({type(exc).__name__})")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
