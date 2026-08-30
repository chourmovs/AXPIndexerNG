"""Verify the packaged llama.cpp binary, not merely its Python import."""
import importlib.metadata
import json
import platform
import re
import inspect

from axp_client.rag.llama_cpp_backend import GenerationConfig, build_chat_invocation

FORBIDDEN_FEATURES = {"AVX2", "AVX_VNNI", "AVX512", "AVX512_VBMI", "AVX512_VNNI",
                      "AVX512_BF16", "BMI2", "FMA", "F16C", "LLAMAFILE"}
EXPECTED_BACKEND_VERSION = "0.3.24"


def verify_hf_generation_tag():
    """Compile and render the HF generation extension used by SmolLM3."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    formatter = Jinja2ChatFormatter(
        template="{% generation %}assistant text{% endgeneration %}",
        eos_token="",
        bos_token="",
    )
    rendered = formatter(messages=[]).prompt
    if "assistant text" not in rendered:
        raise RuntimeError("packaged llama-cpp-python did not render the HF generation-tag body")
    return True


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

    backend_version = importlib.metadata.version("llama-cpp-python")
    if backend_version != EXPECTED_BACKEND_VERSION:
        raise RuntimeError(f"expected llama-cpp-python {EXPECTED_BACKEND_VERSION}, found {backend_version}")
    reported = verify_features(system_info)
    hf_generation_tag_supported = verify_hf_generation_tag()
    cpu = detect_cpu().public()
    invocation, supports_template_kwargs, no_think_compatibility = build_chat_invocation(
        llama_cpp.Llama.create_chat_completion, system_prompt="contract probe", user_prompt="contract probe",
        config=GenerationConfig(), template_kwargs={"enable_thinking": False},
    )
    chat_stream_supported = "stream" in inspect.signature(llama_cpp.Llama.create_chat_completion).parameters
    if not chat_stream_supported:
        raise RuntimeError("packaged llama-cpp-python does not support streamed chat completion")
    print(json.dumps({"backend_version": backend_version,
                      "platform": platform.platform(), "cpu": cpu,
                      "reported_features": reported, "system_info": system_info,
                      "forbidden_features": {feature: reported.get(feature, False)
                                             for feature in sorted(FORBIDDEN_FEATURES)},
                      "hf_generation_tag_supported": hf_generation_tag_supported,
                      "chat_stream_supported": chat_stream_supported,
                      "chat_template_kwargs_supported": supports_template_kwargs,
                      "no_think_compatibility": no_think_compatibility,
                      "chat_invocation_keys": sorted(invocation)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
