import os
from pathlib import Path
from types import SimpleNamespace

import psutil
from axp_tray import process


class FakeProcess:
    def __init__(self, pid, created, executable, command):
        self.pid, self._created, self._executable, self._command = pid, created, executable, command
        self.terminated = self.killed = False

    def create_time(self):
        return self._created

    def exe(self):
        return str(self._executable)

    def cmdline(self):
        return self._command

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def identity(root, fake, role="client", launch_mode="interactive"):
    return {
        "pid": fake.pid, "create_time": fake.create_time(), "role": role,
        "module": f"axp_{role}", "executable": str(fake._executable),
        "installation_root": str(root), "launch_mode": launch_mode,
    }


def test_owned_identity_rejects_pid_reuse_unrelated_and_other_install(tmp_path, monkeypatch):
    root = tmp_path / "AXP"
    executable = root / "python" / "pythonw.exe"
    monkeypatch.setattr(process, "installation_root", lambda: root)
    expected = FakeProcess(123, 10.0, executable, [str(executable), "-B", "-m", "axp_client"])
    item = identity(root, expected)
    assert process.is_owned_process(expected, item, {"client"})
    assert not process.is_owned_process(FakeProcess(123, 11.0, executable, expected.cmdline()), item, {"client"})
    assert not process.is_owned_process(FakeProcess(123, 10.0, tmp_path / "Other/pythonw.exe", expected.cmdline()), item)
    other = dict(item, installation_root=str(tmp_path / "Other"))
    assert not process.is_owned_process(expected, other)
    unrelated = FakeProcess(123, 10.0, executable, [str(executable), "unrelated.py"])
    assert not process.is_owned_process(unrelated, item)


def test_current_tray_is_never_owned(tmp_path, monkeypatch):
    root = tmp_path / "AXP"
    executable = root / "python" / "pythonw.exe"
    monkeypatch.setattr(process, "installation_root", lambda: root)
    fake = FakeProcess(os.getpid(), 10.0, executable, [str(executable), "-m", "axp_client"])
    assert not process.is_owned_process(fake, identity(root, fake))


def test_spawn_registers_role_create_time_and_bytecode_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path / "data"))
    executable = tmp_path / "pythonw.exe"
    child = SimpleNamespace(pid=4321, terminate=lambda: None)
    calls = {}
    monkeypatch.setattr(process, "pythonw", lambda: executable)
    monkeypatch.setattr(process.subprocess, "Popen", lambda command, **kwargs: calls.update(command=command) or child)
    monkeypatch.setattr(process.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42.5))
    monkeypatch.setattr(process, "installation_root", lambda: tmp_path)
    settings = {"model_cache": str(tmp_path / "models")}

    process.spawn("client", ["serve"], settings)

    registered = process.owned_processes()[0]
    assert (registered["role"], registered["pid"], registered["create_time"]) == ("client", 4321, 42.5)
    assert calls["command"][1:4] == ["-B", "-m", "axp_client"]
    assert process.process_environment(settings)["PYTHONDONTWRITEBYTECODE"] == "1"


def test_cleanup_escalates_only_verified_process(monkeypatch):
    owned = FakeProcess(10, 1, Path("pythonw.exe"), [])
    unrelated = FakeProcess(11, 1, Path("pythonw.exe"), [])
    owned_identity = {"pid": 10, "role": "client"}
    rounds = [[(owned, owned_identity)], [(owned, owned_identity)], [(owned, owned_identity)], []]
    monkeypatch.setattr(process, "verified_owned_processes", lambda roles: rounds.pop(0))
    monkeypatch.setattr(process, "is_owned_process", lambda candidate, item, roles: candidate is owned)
    monkeypatch.setattr(process.psutil, "wait_procs", lambda children, timeout: ([], children))

    process.cleanup_owned_processes({"client"}, graceful_timeout=0, terminate_timeout=0)

    assert owned.terminated and owned.killed
    assert not unrelated.terminated and not unrelated.killed


def test_identification_denies_unreadable_process(monkeypatch):
    fake = SimpleNamespace(pid=123, create_time=lambda: (_ for _ in ()).throw(psutil.AccessDenied(123)))
    assert not process.is_owned_process(fake, {"pid": 123, "role": "client"})
