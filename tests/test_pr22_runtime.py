import threading
import time

import pytest
from axp_client.rag.operations import NativeOperationSupervisor


def test_disconnected_observer_does_not_release_native_operation():
    """A failed HTTP observer never cancels or releases native ownership."""
    supervisor = NativeOperationSupervisor()
    entered, release = threading.Event(), threading.Event()
    concurrent = calls = 0
    guard = threading.Lock()

    def native():
        nonlocal concurrent, calls
        with guard:
            concurrent += 1; calls += 1
            assert concurrent == 1
        entered.set(); release.wait(2)
        with guard: concurrent -= 1
        return "done"

    observer_left = threading.Event()
    def request():
        try:
            supervisor.run(native, lambda _elapsed: (_ for _ in ()).throw(BrokenPipeError()), interval=.01)
        except BrokenPipeError:
            observer_left.set()

    thread = threading.Thread(target=request); thread.start()
    assert entered.wait(1) and observer_left.wait(1)
    assert supervisor.busy
    with pytest.raises(RuntimeError, match="chat_busy"):
        supervisor.run(native)
    release.set(); thread.join(1)
    for _ in range(100):
        if not supervisor.busy: break
        time.sleep(.01)
    assert not supervisor.busy
    assert supervisor.run(native) == "done"
    assert calls == 2
    supervisor.close()
