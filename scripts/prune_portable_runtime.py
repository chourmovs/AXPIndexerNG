"""Audit, conservatively prune, and verify the portable Windows runtime."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

CACHE_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})
PRUNE_DIRECTORIES = (
    "python/Lib/test",
    "python/Lib/idlelib",
    "python/Lib/turtledemo",
    "python/Lib/ensurepip",
)


@dataclass(frozen=True)
class Inventory:
    total_bytes: int
    total_files: int
    total_directories: int
    files_under_4k: int
    files_under_64k: int
    pycache_directories: int
    pyc_files: int
    pyo_files: int
    top_directories_by_files: list[tuple[str, int]]
    top_directories_by_bytes: list[tuple[str, int]]
    top_extensions: list[tuple[str, int]]


def audit(root: Path, top=15) -> Inventory:
    root = Path(root).resolve()
    files = [path for path in root.rglob("*") if path.is_file()]
    directories = [path for path in root.rglob("*") if path.is_dir()]
    sizes, directory_files, directory_bytes, extensions = {}, Counter(), Counter(), Counter()
    for path in files:
        size = path.stat().st_size
        sizes[path] = size
        relative = path.relative_to(root)
        extensions[path.suffix.casefold() or "<none>"] += 1
        for parent in (relative.parent, *relative.parents[1:]):
            name = parent.as_posix()
            if name == ".":
                continue
            directory_files[name] += 1
            directory_bytes[name] += size
    return Inventory(
        total_bytes=sum(sizes.values()), total_files=len(files), total_directories=len(directories),
        files_under_4k=sum(size < 4096 for size in sizes.values()),
        files_under_64k=sum(size < 65536 for size in sizes.values()),
        pycache_directories=sum(path.name == "__pycache__" for path in directories),
        pyc_files=sum(path.suffix.casefold() == ".pyc" for path in files),
        pyo_files=sum(path.suffix.casefold() == ".pyo" for path in files),
        top_directories_by_files=directory_files.most_common(top),
        top_directories_by_bytes=directory_bytes.most_common(top),
        top_extensions=extensions.most_common(top),
    )


def forbidden_paths(root: Path):
    return sorted(
        path for path in Path(root).rglob("*")
        if path.name in CACHE_DIRECTORIES or (path.is_file() and path.suffix.casefold() in CACHE_SUFFIXES)
    )


def _remove(path: Path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def prune(root: Path):
    """Remove caches and conservative standard-library/build-only payload."""
    root = Path(root)
    removed = []
    for path in forbidden_paths(root):
        if path.exists():
            _remove(path)
            removed.append(str(path.relative_to(root)))
    for relative in PRUNE_DIRECTORIES:
        path = root / relative
        if path.exists():
            _remove(path)
            removed.append(relative)
    site_packages = root / "python" / "Lib" / "site-packages"
    for pattern in ("pip", "pip-*.dist-info", "wheel", "wheel-*.dist-info"):
        for path in site_packages.glob(pattern):
            _remove(path)
            removed.append(str(path.relative_to(root)))
    scripts = root / "python" / "Scripts"
    for pattern in ("pip*.exe", "wheel.exe"):
        for path in scripts.glob(pattern):
            _remove(path)
            removed.append(str(path.relative_to(root)))
    return sorted(set(removed))


def verify(root: Path):
    forbidden = forbidden_paths(root)
    if forbidden:
        sample = ", ".join(str(path) for path in forbidden[:10])
        raise RuntimeError(f"Forbidden cache artifacts remain ({len(forbidden)}): {sample}")
    return True


def verify_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        bad_compression = [item.filename for item in entries if item.compress_type != zipfile.ZIP_STORED]
        forbidden = []
        files = []
        for item in entries:
            parts = PurePosixPath(item.filename).parts
            if any(part in CACHE_DIRECTORIES for part in parts) or PurePosixPath(item.filename).suffix.casefold() in CACHE_SUFFIXES:
                forbidden.append(item.filename)
            if not item.is_dir():
                files.append(item)
        if bad_compression:
            raise RuntimeError(f"ZIP contains {len(bad_compression)} entries that are not STORE")
        if forbidden:
            raise RuntimeError(f"ZIP contains {len(forbidden)} forbidden cache artifacts")
        names = {item.filename.replace("\\", "/").rstrip("/") for item in files}
        for suffix in ("AXPIndexerNG/AXPIndexerTray.pyw", "AXPIndexerNG/python/pythonw.exe"):
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(f"ZIP required entry missing: {suffix}")
        return {
            "entries": len(entries), "files": len(files),
            "uncompressed_bytes": sum(item.file_size for item in files),
            "files_under_4k": sum(item.file_size < 4096 for item in files),
            "files_under_64k": sum(item.file_size < 65536 for item in files),
            "cache_artifacts": 0,
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prune", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("root", type=Path)
    archive = subparsers.add_parser("verify-zip")
    archive.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = asdict(audit(args.root))
    elif args.command == "prune":
        before = asdict(audit(args.root))
        removed = prune(args.root)
        verify(args.root)
        result = {"before": before, "after": asdict(audit(args.root)), "removed": removed}
    elif args.command == "verify":
        verify(args.root)
        result = {"verified": True, **asdict(audit(args.root))}
    else:
        result = verify_zip(args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
