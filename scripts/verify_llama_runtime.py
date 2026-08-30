"""Verify the packaged llama.cpp binary, not merely its Python import."""
import importlib.metadata
import json
import platform
import re

import llama_cpp
from axp_client.rag.cpu import detect_cpu


def main():
    raw = llama_cpp.llama_print_system_info()
    system_info = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    def feature(name):
        match = re.search(rf"(?:^|\|)\s*{re.escape(name)}\s*=\s*([01])(?:\s*\||$)", system_info)
        if match is None:
            raise RuntimeError(f"llama.cpp did not report the compiled {name} feature")
        return match.group(1) == "1"

    actual = {name.lower(): feature(name) for name in ("AVX", "AVX2", "AVX512")}
    if actual != {"avx": True, "avx2": False, "avx512": False}:
        raise RuntimeError(f"unexpected packaged llama.cpp ISA policy: {actual}")
    cpu = detect_cpu().public()
    print(json.dumps({"backend_version": importlib.metadata.version("llama-cpp-python"),
                      "platform": platform.platform(), "cpu": cpu,
                      "compiled_features": actual, "system_info": system_info}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
