"""Secure, testable Windows Task Scheduler integration."""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from axp_core.runtime import installation_root

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"


def task_name(installation_id):
    return f"AXPIndexerNG-{installation_id.replace('-', '')[:8]}"


def current_windows_principal():
    if os.name != "nt":
        raise OSError("Windows Task Scheduler is only available on Windows")
    size = ctypes.c_ulong(0)
    ctypes.windll.secur32.GetUserNameExW(2, None, ctypes.byref(size))  # NameSamCompatible
    buffer = ctypes.create_unicode_buffer(size.value)
    if not ctypes.windll.secur32.GetUserNameExW(2, buffer, ctypes.byref(size)):
        raise ctypes.WinError()
    return buffer.value


def task_xml(principal, executable, launcher, working_directory, installation_id):
    values = (escape(str(Path(value).absolute())) for value in (executable, launcher, working_directory))
    executable, launcher, working_directory = values
    principal, installation_id = escape(principal), escape(installation_id)
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="{NS}"><RegistrationInfo><Description>AXPIndexerNG installation {installation_id}</Description></RegistrationInfo>
<Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{principal}</UserId></LogonTrigger></Triggers>
<Principals><Principal id="Author"><UserId>{principal}</UserId><LogonType>Password</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT0S</ExecutionTimeLimit><RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure></Settings>
<Actions Context="Author"><Exec><Command>{executable}</Command><Arguments>&quot;{launcher}&quot; --scheduled-task</Arguments><WorkingDirectory>{working_directory}</WorkingDirectory></Exec></Actions></Task>'''


@dataclass
class TaskStatus:
    state: str
    message: str = ""
    owned: bool = False


class TaskSchedulerBackend:
    """All schtasks calls live here; registration lets schtasks own its visible password prompt."""

    def __init__(self, settings, root=None, principal=None, runner=subprocess.run):
        self.settings = settings
        self.root = Path(root or installation_root()).resolve()
        self.principal = principal or current_windows_principal()
        self.runner = runner
        self.name = task_name(settings["installation_id"])
        bundled = self.root / "python" / "pythonw.exe"
        self.executable = bundled.resolve()
        self.launcher = (self.root / "AXPIndexerDaemon.pyw").resolve()

    def expected_xml(self):
        return task_xml(self.principal, self.executable, self.launcher, self.root,
                        self.settings["installation_id"])

    def _query_xml(self):
        result = self.runner(["schtasks.exe", "/Query", "/TN", self.name, "/XML"],
                             capture_output=True, text=True, check=False)
        return result.returncode, result.stdout

    def status(self):
        code, xml = self._query_xml()
        if code:
            return TaskStatus("not_configured")
        try:
            root = ET.fromstring(xml)
            def find(path):
                return root.findtext(path, namespaces={"t": NS}) or ""
            expected = (str(self.executable), f'"{self.launcher}" --scheduled-task', str(self.root),
                        self.principal, "Password", "LeastPrivilege")
            actual = (find(".//t:Command"), find(".//t:Arguments"), find(".//t:WorkingDirectory"),
                      find(".//t:Principal/t:UserId"), find(".//t:LogonType"), find(".//t:RunLevel"))
            description = find(".//t:RegistrationInfo/t:Description")
            owned = self.settings["installation_id"] in description
            if not owned or actual != expected:
                return TaskStatus("needs_repair", "Background task configuration differs from this installation. Repair required.",
                                  owned=owned)
            return TaskStatus("ready", owned=True)
        except ET.ParseError:
            return TaskStatus("unknown", "Windows returned an invalid scheduled task definition")

    def register(self):
        # No password argument is supplied. A new visible console is essential so schtasks prompts directly.
        status = self.status()
        if status.state != "not_configured":
            raise RuntimeError("A task already uses this installation's task name; explicit repair is required")
        self._register_xml()

    def _register_xml(self):
        path = None
        try:
            handle, name = tempfile.mkstemp(prefix="axp-task-", suffix=".xml")
            path = Path(name)
            with os.fdopen(handle, "w", encoding="utf-16") as stream:
                stream.write(self.expected_xml())
            flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            result = self.runner(["schtasks.exe", "/Create", "/TN", self.name, "/XML", str(path), "/F"],
                                 creationflags=flags, check=False)
            if result.returncode:
                raise RuntimeError("Windows or company security policy refused the scheduled task")
        finally:
            if path:
                path.unlink(missing_ok=True)

    def repair(self):
        status = self.status()
        if status.state != "not_configured" and not status.owned:
            raise RuntimeError("Refusing to repair scheduled task because ownership could not be verified")
        self._register_xml()

    def run(self):
        result = self.runner(["schtasks.exe", "/Run", "/TN", self.name], capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("Background daemon unavailable; task repair or credential refresh may be required")
        return result

    def delete(self):
        status = self.status()
        if status.state not in ("ready", "running") or not status.owned:
            raise RuntimeError("Refusing to delete scheduled task because ownership could not be verified")
        return self.runner(["schtasks.exe", "/Delete", "/TN", self.name, "/F"], check=True)


def background_task_status(settings, **kwargs):
    return TaskSchedulerBackend(settings, **kwargs).status()
