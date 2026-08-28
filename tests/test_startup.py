import sys
from pathlib import PureWindowsPath
from types import SimpleNamespace

from axp_tray import startup


class Key:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def test_start_with_windows_enable_disable_and_spaces(monkeypatch):
    values = {}
    fake = SimpleNamespace(
        HKEY_CURRENT_USER=object(), REG_SZ=1,
        CreateKey=lambda *_: Key(), OpenKey=lambda *_: Key(),
        SetValueEx=lambda _key, name, _reserved, _type, value: values.__setitem__(name, value),
        QueryValueEx=lambda _key, name: (values[name], 1),
        DeleteValue=lambda _key, name: values.pop(name),
    )
    monkeypatch.setattr(startup.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "winreg", fake)
    monkeypatch.setattr(startup, "installation_root", lambda: PureWindowsPath(r"C:\Program Files\AXP Indexer"))
    startup.set_enabled(True)
    assert startup.is_enabled()
    assert "Program Files" in values[startup.VALUE_NAME] and "pythonw.exe" in values[startup.VALUE_NAME]
    values[startup.VALUE_NAME] = r'"C:\Old Location\python\pythonw.exe" "C:\Old Location\AXPIndexerTray.pyw"'
    assert startup.repair_registration()
    assert "Program Files" in values[startup.VALUE_NAME]
    startup.set_enabled(False)
    assert not values
