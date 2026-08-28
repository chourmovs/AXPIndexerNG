import os

import pytest
from axp_core.database import connect
from axp_daemon.indexer import scan
from axp_daemon.scanner import discover
from conftest import FakeEmbedder


def test_multi_root(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "a.txt").write_text("alpha")
    (b / "b.txt").write_text("beta")
    c = connect(tmp_path / "x.db", dimension=3)
    scan(c, a, FakeEmbedder())
    scan(c, b, FakeEmbedder())
    (a / "a.txt").unlink()
    scan(c, a, FakeEmbedder())
    assert [r[0] for r in c.execute("select path from documents")] == [str(b / "b.txt")]


def test_discover_recurses_through_every_level(tmp_path):
    root = tmp_path / "root"
    level1 = root / "level1"
    level2 = level1 / "level2"
    level3 = level2 / "level3"
    level3.mkdir(parents=True)
    expected = {root / "root.txt", level1 / "one.txt", level2 / "two.md", level3 / "three.pdf"}
    for path in expected:
        path.write_text("discovery only")

    traversal = discover(root, recursive=True)

    assert set(traversal) == expected
    assert traversal.complete
    assert traversal.discovered == len(expected)


def test_discover_non_recursive_only_returns_root_files(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    direct = root / "root.txt"
    direct.write_text("direct")
    (nested / "nested.txt").write_text("nested")

    assert list(discover(root, recursive=False)) == [direct]


def test_discover_does_not_follow_directory_symlinks(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "external.txt").write_text("must not be discovered")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    assert list(discover(root, recursive=True)) == []


def test_inaccessible_subdirectory_marks_enumeration_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "root"
    blocked = root / "blocked"
    blocked.mkdir(parents=True)
    real_scandir = os.scandir

    def deny_blocked(path):
        if os.fspath(path) == os.fspath(blocked):
            raise PermissionError("access denied")
        return real_scandir(path)

    monkeypatch.setattr("axp_daemon.scanner.os.scandir", deny_blocked)
    traversal = discover(root, recursive=True)

    assert list(traversal) == []
    assert not traversal.complete
    assert any("access denied" in error for error in traversal.errors)
