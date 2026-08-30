"""Verify the packaged llama.cpp binary, not merely its Python import."""
import importlib.metadata
import json
import platform
import re
from axp_client.rag.llama_cpp_backend import GenerationConfig, build_chat_invocation

FORBIDDEN_FEATURES = {"AVX2", "AVX_VNNI", "AVX512", "AVX512_VBMI", "AVX512_VNNI",
                      "AVX512_BF16", "BMI2", "FMA", "F16C", "LLAMAFILE"}


def verify_features(system_info):
    """Validate llama.cpp's active-only CPU feature report."""
    reported = {match.group(1): match.group(2) == "1" for match in re.finditer(
        r"\b([A-Z][A-Z0-9_]*)\s*=\s*([01])\b", system_info
    )}
    if reported.get("AVX") is not True:
        raise RuntimeError("packaged llama.cpp does not report required AVX support")
    enabled_forbidden = sorted(feature for feature in FORBIDDEN_FEATURES if reported.get(feature, False))
    if enabled_forbidden:
        raise RuntimeError(f"packaged llama.cpp enables forbidden CPU features: {', '.join(enabled_forbidden)}")
    return reported


def main():
    import llama_cpp
    from axp_client.rag.cpu import detect_cpu

    raw = llama_cpp.llama_print_system_info()
    system_info = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    reported = verify_features(system_info)
    cpu = detect_cpu().public()
    invocation, supports_template_kwargs, no_think_compatibility = build_chat_invocation(
        llama_cpp.Llama.create_chat_completion, system_prompt="contract probe", user_prompt="contract probe",
        config=GenerationConfig(), template_kwargs={"enable_thinking": False},
    )
    print(json.dumps({"backend_version": importlib.metadata.version("llama-cpp-python"),
                      "platform": platform.platform(), "cpu": cpu,
                      "reported_features": reported, "system_info": system_info,
                      "chat_template_kwargs_supported": supports_template_kwargs,
                      "no_think_compatibility": no_think_compatibility,
                      "chat_invocation_keys": sorted(invocation)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
