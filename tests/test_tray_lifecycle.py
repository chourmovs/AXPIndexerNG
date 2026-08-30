from types import SimpleNamespace

from axp_core.runtime import read_json, runtime_paths
from axp_tray import app, process


def test_intentional_stop_and_next_launch_restore_desired(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(process, "send_control", lambda command: command == "stop")
    assert process.stop_daemon(intentional=True)
    paths = runtime_paths()
    assert read_json(paths["desired"])["state"] == "stopped"
    app.establish_startup_desired({"auto_start_daemon": True}, paths)
    assert read_json(paths["desired"])["state"] == "running"


def test_absent_client_is_harmless():
    assert process.stop_client({"web_port": 1}, timeout=0.01) is False


def test_tray_shutdown_requests_daemon_and_client(monkeypatch):
    calls = []
    application = app.TrayApplication.__new__(app.TrayApplication)
    application.settings = {"web_port": 8765}
    application.icon = SimpleNamespace(stop=lambda: calls.append("icon"))
    application.root = SimpleNamespace(quit=lambda: calls.append("quit"), destroy=lambda: calls.append("destroy"))
    application._on_tk = lambda callback: callback()
    monkeypatch.setattr(app, "stop_daemon", lambda intentional: calls.append(("daemon", intentional)))
    monkeypatch.setattr(app, "stop_client", lambda settings: calls.append(("client", settings)) or True)
    monkeypatch.setattr(app, "read_daemon_state", lambda: {"state": "stopped"})
    monkeypatch.setattr(app, "cleanup_owned_processes", lambda roles: calls.append(("cleanup", roles)))

    application._shutdown_worker()

    assert ("daemon", True) in calls
    assert ("client", application.settings) in calls
    assert ("cleanup", {"client", "daemon"}) in calls
    assert calls[-3:] == ["icon", "quit", "destroy"]


def test_scheduled_shutdown_preserves_daemon(monkeypatch):
    calls = []
    application = app.TrayApplication.__new__(app.TrayApplication)
    application.settings = {"web_port": 8765, "daemon_runtime_mode": "scheduled_task"}
    application.icon = SimpleNamespace(stop=lambda: None)
    application.root = SimpleNamespace(quit=lambda: None, destroy=lambda: None)
    application._on_tk = lambda callback: callback()
    monkeypatch.setattr(app, "stop_daemon", lambda **kwargs: calls.append("daemon"))
    monkeypatch.setattr(app, "stop_client", lambda settings: True)
    monkeypatch.setattr(app, "cleanup_owned_processes", lambda roles: calls.append(roles))

    application._shutdown_worker()

    assert "daemon" not in calls
    assert calls == [{"client"}]
