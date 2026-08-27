import hashlib
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx"}


def path_key(path):
    return str(path.resolve()).casefold()


def discover(root):
    return sorted(
        (p for p in Path(root).resolve().rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED), key=path_key
    )


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
