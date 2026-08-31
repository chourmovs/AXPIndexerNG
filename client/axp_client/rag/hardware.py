"""Unprivileged hardware discovery and accelerator probing."""
import ctypes
import platform
import re
from dataclasses import asdict, dataclass

from .sycl_probe import probe_sycl


@dataclass(frozen=True)
class HardwareCapabilities:
    cpu_name: str
    intel_gpu_detected: bool = False
    intel_gpu_name: str | None = None
    intel_gpu_available: bool = False
    accelerator_reason: str = "accelerator_not_installed"
    intel_gpu_vendor_id: str | None = None
    intel_gpu_device_id: str | None = None
    sycl_runtime_installed: bool = False
    sycl_probe_ok: bool = False
    sycl_device_name: str | None = None
    sycl_device_count: int = 0
    sycl_probe_error: str | None = None
    sycl_probe_returncode: int | None = None
    sycl_probe_duration_ms: int = 0

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
            devices.append({"DeviceString": device.DeviceString, "DeviceID": device.DeviceID,
                            "StateFlags": int(device.StateFlags)})
        index += 1
    return devices


def detect_hardware(accelerator_executable=None, accelerator_runtime_root=None):
    if platform.system() != "Windows":
        return HardwareCapabilities(platform.processor() or platform.machine() or "Unknown CPU",
                                    accelerator_reason="unsupported_platform")
    adapters = windows_display_adapters()
    normalized = [item if isinstance(item, dict) else {"DeviceString": item, "DeviceID": "", "StateFlags": 0}
                  for item in adapters]
    intel = next((item for item in normalized if "intel" in item["DeviceString"].lower()
                  and "basic display" not in item["DeviceString"].lower()
                  and (not re.search(r"VEN_([0-9A-F]{4})", item["DeviceID"], re.I)
                       or re.search(r"VEN_8086", item["DeviceID"], re.I))), None)
    available = False
    reason = "accelerator_not_installed" if intel else "no_intel_gpu"
    probe = probe_sycl(accelerator_executable, accelerator_runtime_root) if accelerator_executable else None
    available = bool(probe and probe["ok"])
    if probe:
        reason = None if available else probe["error_code"]
    identity = intel.get("DeviceID", "") if intel else ""
    vendor = re.search(r"VEN_([0-9A-F]{4})", identity, re.I)
    device = re.search(r"DEV_([0-9A-F]{4})", identity, re.I)
    return HardwareCapabilities(platform.processor() or platform.machine() or "Unknown CPU",
                                bool(intel), intel["DeviceString"] if intel else None, available, reason,
                                vendor.group(1).upper() if vendor else None,
                                device.group(1).upper() if device else None,
                                bool(probe and probe["installed"]), available,
                                probe["device_name"] if probe else None, probe["device_count"] if probe else 0,
                                probe["error_code"] if probe and not available else None,
                                probe["returncode"] if probe else None, probe["duration_ms"] if probe else 0)
