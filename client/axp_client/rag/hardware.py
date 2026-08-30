"""Unprivileged hardware discovery and accelerator probing."""
import ctypes
import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareCapabilities:
    cpu_name: str
    intel_gpu_detected: bool = False
    intel_gpu_name: str | None = None
    intel_gpu_available: bool = False
    accelerator_reason: str = "accelerator_not_installed"

    def public(self):
        return asdict(self)


def windows_display_adapters():
    if platform.system() != "Windows":
        return []
    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("DeviceName", ctypes.c_wchar * 32),
                    ("DeviceString", ctypes.c_wchar * 128), ("StateFlags", ctypes.c_ulong),
                    ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]
    devices, index = [], 0
    while True:
        device = DisplayDevice(); device.cb = ctypes.sizeof(device)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break
        if device.DeviceString:
            devices.append(device.DeviceString)
        index += 1
    return devices


def detect_hardware(accelerator_executable=None):
    adapters = windows_display_adapters()
    intel = next((name for name in adapters if "intel" in name.lower()
                  and "basic display" not in name.lower()), None)
    available, reason = False, "accelerator_not_installed"
    if accelerator_executable:
        try:
            probe = subprocess.run([str(accelerator_executable), "--list-devices"], capture_output=True,
                                   text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            available = probe.returncode == 0 and "intel" in (probe.stdout + probe.stderr).lower()
            reason = None if available else "intel_driver_or_runtime_unavailable"
        except (OSError, subprocess.TimeoutExpired):
            reason = "intel_driver_or_runtime_unavailable"
    return HardwareCapabilities(platform.processor() or platform.machine() or "Unknown CPU",
                                bool(intel), intel, available, reason)
