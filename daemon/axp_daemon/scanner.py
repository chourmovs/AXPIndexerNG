import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx", ".pptx"}
DRIVE_IGNORES = {"$recycle.bin", "system volume information"}
TEMPORARY_PREFIXES = ("~$", ".~lock.")


class SourceUnavailable(OSError):
    pass


def is_supported_document(name):
    folded = name.casefold()
    return not folded.startswith(TEMPORARY_PREFIXES) and Path(name).suffix.casefold() in SUPPORTED


@dataclass
class Discovery:
    root: Path
    recursive: bool = True
    complete: bool = True
    errors: list[str] = field(default_factory=list)
    discovered: int = 0

    def __iter__(self):
        if not self.root.exists() or not self.root.is_dir():
            raise SourceUnavailable(f"Source root is unavailable: {self.root}")
        yield from self._walk(self.root)

    def _walk(self, directory):
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            self.complete = False
            self.errors.append(f"{directory}: {exc}")
            return
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if self.recursive and entry.name.casefold() not in DRIVE_IGNORES:
                            yield from self._walk(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and is_supported_document(entry.name):
                        self.discovered += 1
                        yield Path(entry.path)
                except OSError as exc:
                    self.complete = False
                    self.errors.append(f"{entry.path}: {exc}")


def path_key(path):
    return os.path.normcase(os.path.abspath(os.fspath(path))).casefold()


def discover(root, recursive=True):
    return Discovery(Path(root), recursive=recursive)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
