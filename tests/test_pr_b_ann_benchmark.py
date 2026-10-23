"""PR-B: deterministic exact reference, sampling and adapter contract."""
import numpy as np
import pytest

from axp_core.ann_metrics import exact_cosine_topk, recall_at_k, select_by_position


def test_position_sampling_has_requested_cardinality_and_spread():
    for total, size in ((10, 3), (100, 50), (17, 17), (11, 1)):
        positions = [i for i in range(1, total + 1) if select_by_position(i, total, size)]
        assert len(positions) == size
        assert positions[-1] == total
    with pytest.raises(ValueError):
        select_by_position(0, 10, 3)


def test_cosine_topk_and_recall_are_based_on_same_cohort():
    matrix = np.array([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32)
    exact = exact_cosine_topk(matrix, np.array([3, 7, 13]), [1, 0], 2)
    assert exact == [3, 13]
    assert recall_at_k(exact, [13, 7], 2) == .5
    assert recall_at_k(exact, exact, 2) == 1.
    with pytest.raises(ValueError):
        exact_cosine_topk(matrix, np.array([3, 7]), [1, 0], 2)


def test_usearch_optional_integration_smoke(tmp_path):
    pytest.importorskip("usearch")
    from axp_core.ann import UsearchHNSW

    index = UsearchHNSW(2, connectivity=8, expansion_add=32, expansion_search=32)
    index.add(np.array([11, 22, 33], dtype=np.uint64),
              np.array([[1., 0.], [0., 1.], [0.8, 0.6]], dtype=np.float32))
    assert index.size == 3
    assert index.search(np.array([1, 0], dtype=np.float32), 1)[0][0] == 11
    path = tmp_path / "candidate.usearch"
    index.save(path)
    assert path.is_file() and path.stat().st_size > 0



def _benchmark_args(db, queries_json, tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        db=db, max_vectors=24, queries_json=queries_json, query_count=4, top_k=4,
        connectivity=8, expansion_add=32, expansion_search=32, dtype="f32",
        batch_size=8, seed=42, index_output=None, output=None,
    )


def _synthetic_ann_catalog(tmp_path):
    import math
    from axp_core.database import connect
    from axp_core.metadata import ensure_index_signature
    from axp_core.vectors import upsert

    db = tmp_path / "synthetic-ann.db"
    with connect(db, dimension=2) as con:
        ensure_index_signature(con, "synthetic-2d", 2)
        for chunk_id in range(1, 33):
            theta = 2 * math.pi * chunk_id / 32
            upsert(con, chunk_id, [math.cos(theta), math.sin(theta)])
        con.commit()
    return db


def test_full_benchmark_closes_windows_mmaps(tmp_path, monkeypatch):
    """Exercises the actual benchmark and temporary-directory cleanup on Windows."""
    pytest.importorskip("usearch")
    pytest.importorskip("sqlite_vec")
    from scripts import benchmark_ann

    db = _synthetic_ann_catalog(tmp_path)
    queries = tmp_path / "queries.json"
    queries.write_text('[{"query":"north","vector":[1.0,0.0]}]', encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(benchmark_ann.tempfile, "tempdir", str(workspace))
    report = benchmark_ann.run(_benchmark_args(db, queries, tmp_path))

    assert report["sample_vectors"] == 24
    assert report["persisted_reopen_ok"]
    assert report["queries"] == 1
    assert report["latency"]["ann_ms"]["n"] == 1
    assert not list(workspace.glob("axp-ann-benchmark-*"))


def test_failed_benchmark_releases_windows_mmaps(tmp_path, monkeypatch):
    """A primary exception must not be replaced by WinError 32 during cleanup."""
    pytest.importorskip("usearch")
    pytest.importorskip("sqlite_vec")
    from scripts import benchmark_ann

    db = _synthetic_ann_catalog(tmp_path)
    malformed = tmp_path / "bad-queries.json"
    malformed.write_text('[{"query":"wrong dimension","vector":[1.0]}]', encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(benchmark_ann.tempfile, "tempdir", str(workspace))
    with pytest.raises(ValueError, match="wrong dimension"):
        benchmark_ann.run(_benchmark_args(db, malformed, tmp_path))
    assert not list(workspace.glob("axp-ann-benchmark-*"))
