"""CPU/OS instruction capability reporting for the portable native runtime."""
from __future__ import annotations

import ctypes
import platform
from dataclasses import asdict, dataclass

RUNTIME_REQUIRED_ISA = "x86-64 + AVX"


@dataclass(frozen=True)
class CpuCapabilities:
    cpu_name: str
    cpu_arch: str
    avx_available: bool
    avx2_available: bool
    fma_available: bool
    f16c_available: bool
    bmi2_available: bool
    runtime_required_isa: str
    runtime_cpu_compatible: bool

    def public(self):
        return asdict(self)


def _flags() -> set[str]:
    """Return OS-enabled flags; py-cpuinfo uses CPUID and OSXSAVE/XGETBV."""
    try:
        from cpuinfo import get_cpu_info

        return {str(flag).lower() for flag in get_cpu_info().get("flags", ())}
    except (ImportError, OSError, ValueError):
        return set()


def detect_cpu() -> CpuCapabilities:
    arch = platform.machine().lower() or "unknown"
    flags = _flags()
    # IsProcessorFeaturePresent is the OS authority when available. In
    # particular, AVX is false unless Windows enabled the extended register state.
    if platform.system() == "Windows":
        try:
            kernel = ctypes.windll.kernel32
            flags.update(name for feature, name in ((39, "avx"), (40, "avx2"))
                         if kernel.IsProcessorFeaturePresent(feature))
        except (AttributeError, OSError):
            pass
    avx = "avx" in flags
    compatible = avx if arch in {"amd64", "x86_64", "x64"} else False
    return CpuCapabilities(platform.processor() or platform.machine() or "Unknown CPU", arch,
                           avx, "avx2" in flags, "fma" in flags, "f16c" in flags,
                           "bmi2" in flags, RUNTIME_REQUIRED_ISA, compatible)
