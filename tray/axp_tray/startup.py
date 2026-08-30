import os

from axp_core.runtime import installation_root

VALUE_NAME = "AXPIndexerNG"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def startup_command():
    root = installation_root()
    return f'"{root / "python" / "pythonw.exe"}" -B "{root / "AXPIndexerTray.pyw"}"'


def is_enabled():
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
        return value == startup_command()
    except OSError:
        return False


def repair_registration():
    """Rewrite our per-user Run value when a portable installation was moved."""
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
    except OSError:
        return False
    if value != startup_command():
        try:
            set_enabled(True)
        except OSError:
            return False
    return True


def set_enabled(enabled):
    if os.name != "nt":
        raise RuntimeError("Windows startup registration is only available on Windows")
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        if enabled:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, startup_command())
        else:
            try:
                winreg.DeleteValue(key, VALUE_NAME)
            except FileNotFoundError:
                pass
