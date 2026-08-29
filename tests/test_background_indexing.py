import re
from pathlib import Path

from axp_core.background import access_path_for, resolve_source_path
from axp_core.runtime import load_settings
from axp_tray.background_task import task_xml


def test_installation_id_is_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    first = load_settings()["installation_id"]
    assert load_settings()["installation_id"] == first
    assert re.fullmatch(r"[0-9a-f-]{36}", first)


def test_task_xml_security_and_policy():
    xml = task_xml(r"DOMAIN\User", r"C:\AXP\python\pythonw.exe", r"C:\AXP\AXPIndexerDaemon.pyw",
                   r"C:\AXP", "installation-id")
    for value in ("DOMAIN\\User", "Password", "LeastPrivilege", "IgnoreNew", "<Count>3</Count>",
                  "<Interval>PT1M</Interval>", "<StartWhenAvailable>true", "<DisallowStartIfOnBatteries>false",
                  "<StopIfGoingOnBatteries>false", "AXPIndexerDaemon.pyw", "<WorkingDirectory>"):
        assert value in xml
    folded = xml.casefold()
    assert "actual_password" not in folded
    assert "highestavailable" not in folded


def test_mapped_drive_access_keeps_logical_identity():
    mappings = {"K:": r"\\server\share"}
    access = access_path_for(r"K:\INDUS CSR\Rapports\étude finale.pdf", mappings)
    assert str(access) == r"\\server\share\INDUS CSR\Rapports\étude finale.pdf"
    resolved = resolve_source_path(r"K:\INDUS CSR", mappings)
    logical = resolved.logical_for(Path(str(resolved.access_root)) / "Study.xlsx")
    assert str(logical).endswith(r"K:\INDUS CSR\Study.xlsx")
