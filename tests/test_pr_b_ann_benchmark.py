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
