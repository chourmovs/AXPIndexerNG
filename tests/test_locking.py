import pytest
from axp_core.locking import AlreadyLocked, FileLock


def test_os_backed_lock_survives_stale_file(tmp_path):
    path = tmp_path / "instance.lock"
    first = FileLock(path).acquire()
    with pytest.raises(AlreadyLocked):
        FileLock(path).acquire()
    first.release()
    second = FileLock(path).acquire()
    second.release()
    assert path.exists()
