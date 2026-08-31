from pathlib import Path

from axp_client.rag.llama_cpp_backend import GenerationConfig, build_chat_invocation
from axp_client.rag.model_catalog import CATALOG_VERSION, MODELS, catalog_model


def test_lfm_catalog_contract_and_existing_model_identity():
    assert CATALOG_VERSION == 2
    assert [model.id for model in MODELS] == [
        "qwen3-1.7b-q4km", "smollm3-3b-q4km", "lfm25-1.2b-qad-q4"]
    qwen, smol, lfm = MODELS
    assert (qwen.repository, qwen.revision, qwen.filename, qwen.sha256, qwen.size_bytes) == (
        "ggml-org/Qwen3-1.7B-GGUF", "daeb8e2d528a760970442092f6bf1e55c3b659eb",
        "Qwen3-1.7B-Q4_K_M.gguf", "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5",
        1_282_439_264)
    assert (smol.repository, smol.revision, smol.filename, smol.sha256, smol.size_bytes) == (
        "ggml-org/SmolLM3-3B-GGUF", "4965cb60b150737b68a0408c36aeefb65078f894",
        "SmolLM3-Q4_K_M.gguf", "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
        1_915_305_312)
    assert (lfm.repository, lfm.revision, lfm.filename, lfm.sha256, lfm.size_bytes) == (
        "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "6767265158422fb8a19c62ceb45f16f05363615b",
        "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf",
        "bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1", 695_755_488)
    assert (qwen.recommended, smol.recommended, lfm.recommended) == (True, False, False)
    assert (qwen.experimental, smol.experimental, lfm.experimental) == (False, False, True)
    assert lfm.license == "LFM Open License v1.0"


def test_public_quantization_and_maturity_are_model_owned():
    public = {model.id: model.public() for model in MODELS}
    assert [public[model.id]["quantization"] for model in MODELS] == ["Q4_K_M", "Q4_K_M", "QAD Q4_0"]
    assert public["lfm25-1.2b-qad-q4"]["experimental"] is True
    assert public["qwen3-1.7b-q4km"]["recommended"] is True


def invocation(profile, completion):
    config = GenerationConfig(max_answer_tokens=profile.max_answer_tokens,
        temperature=profile.temperature, top_p=profile.top_p, top_k=profile.top_k,
        repeat_penalty=profile.repeat_penalty)
    return build_chat_invocation(completion, system_prompt="system", user_prompt="user",
                                 config=config, template_kwargs=profile.chat_template_kwargs)[0]


def test_thinking_and_sampling_are_model_specific():
    def modern(messages, max_tokens, temperature, top_p, top_k, repeat_penalty,
               chat_template_kwargs=None):
        pass
    qwen, smol, lfm = MODELS
    for model in (qwen, smol):
        call = invocation(model, modern)
        assert call["chat_template_kwargs"] == {"enable_thinking": False}
        assert call["messages"][0]["content"] == "system"
        assert (call["temperature"], call["top_p"], call["top_k"], call["repeat_penalty"]) == (0.2, 0.8, 20, 1.0)
    call = invocation(lfm, modern)
    assert "chat_template_kwargs" not in call
    assert not call["messages"][0]["content"].startswith("/no_think")
    assert (call["temperature"], call["top_p"], call["top_k"], call["repeat_penalty"]) == (0.1, 0.1, 50, 1.05)


def test_legacy_qwen_no_think_fallback_but_lfm_has_no_directive():
    def legacy(messages, max_tokens, temperature, top_p, top_k):
        pass
    qwen = invocation(MODELS[0], legacy)
    lfm = invocation(MODELS[2], legacy)
    assert qwen["messages"][0]["content"] == "/no_think\nsystem"
    assert lfm["messages"][0]["content"] == "system"


def test_lfm_context_policy_is_low_latency_and_physically_bounded():
    lfm = catalog_model("lfm25-1.2b-qad-q4")
    assert (lfm.context_size, lfm.max_answer_tokens, lfm.max_evidence_tokens,
            lfm.max_context_documents, lfm.max_context_blocks,
            lfm.max_seeds_per_document) == (6144, 160, 1536, 3, 5, 2)


def test_cpu_runtime_pin_and_static_lfm2_architecture_contract():
    root = Path(__file__).parents[1]
    assert "llama-cpp-python==0.3.24" in (root / "requirements-runtime.txt").read_text()
    assert "LLM_ARCH_LFM2" in (root / "scripts/verify_llama_runtime.py").read_text()
    assert "verify_lfm2_architecture(llama_cpp.__file__)" in (
        root / "scripts/verify_llama_runtime.py").read_text()
