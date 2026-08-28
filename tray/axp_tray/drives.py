import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class Drive:
    root: str
    kind: str


DRIVE_TYPES = {2: "Removable", 3: "Fixed", 4: "Network", 5: "CD-ROM", 6: "RAM disk"}


def list_drives():
    if not hasattr(ctypes, "windll"):
        return []
    kernel = ctypes.windll.kernel32
    mask = kernel.GetLogicalDrives()
    result = []
    for index in range(26):
        if mask & (1 << index):
            root = f"{chr(65 + index)}:\\"
            result.append(Drive(root, DRIVE_TYPES.get(kernel.GetDriveTypeW(root), "Unknown")))
    return result
