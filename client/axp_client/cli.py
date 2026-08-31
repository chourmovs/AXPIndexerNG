import argparse
import json
import os

from axp_core.database import capability_report, connect
from axp_core.hybrid import SearchConfig
from axp_core.runtime import configure_logging, load_settings
from axp_daemon.embeddings import embedder_for_index

from .rag.evaluation import evaluate, format_summary, load_cases, threshold_sweep
from .rag.factory import create_chat_backend
from .rag.benchmark import BenchmarkRunner
from .rag.runtime_manager import InferenceRuntimeManager
from .rag.model_catalog import catalog_model
from .rag.model import import_model, model_status, remove_model, verify_model
from .rag.service import ChatUnavailableError, GenerationFailedError, RagService
from .reranker import Reranker
from .search import search
from .server import serve

LOGGER = configure_logging("axp_client", "client.log")


def main(argv=None):
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    model = s.add_parser("chat-model")
    model_sub = model.add_subparsers(dest="model_cmd", required=True)
    model_import = model_sub.add_parser("import")
    model_import.add_argument("--file", required=True)
    model_sub.add_parser("status")
    model_sub.add_parser("verify")
    model_sub.add_parser("test")
    intel_test = model_sub.add_parser("intel-test")
    intel_test.add_argument("--json", action="store_true", dest="json_output")
    model_benchmark = model_sub.add_parser("benchmark")
    model_benchmark.add_argument("--profile", choices=("quick", "rag"), default="quick")
    model_remove = model_sub.add_parser("remove")
    model_remove.add_argument("--yes", action="store_true")
    evaluation = s.add_parser("rag-eval")
    evaluation.add_argument("--db", required=True)
    evaluation.add_argument("--cases", required=True)
    evaluation.add_argument("--full", action="store_true")
    evaluation.add_argument("--output")
    evaluation.add_argument("--sweep", action="store_true")
    for cmd in ("health", "search", "serve", "ask", "rag-diagnose"):
        q = s.add_parser(cmd)
        q.add_argument("--db", required=True)
        if cmd == "search":
            q.add_argument("--query", required=True)
            q.add_argument(
                "--profile",
                choices=("fast", "hybrid", "quality"),
                default="hybrid",
                help="retrieval profile; quality adds ColBERT reranking (independent of the index embedding profile)",
            )
            q.add_argument("--limit", type=int, default=20)
            q.add_argument("--explain", action="store_true")
            q.add_argument("--lexical-candidates", type=int, default=100)
            q.add_argument("--vector-candidates", type=int, default=100)
            q.add_argument("--rerank-candidates", type=int, default=30)
        if cmd == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8765)
        if cmd == "ask":
            q.add_argument("--question", required=True)
            q.add_argument("--debug", action="store_true")
        if cmd == "rag-diagnose":
            q.add_argument("--query", required=True)
    a = p.parse_args(argv)
    if a.cmd == "chat-model":
        path = load_settings()["chat_model_path"]
        if a.model_cmd == "import":
            print(json.dumps(import_model(a.file, path)))
        elif a.model_cmd == "status":
            print(json.dumps(model_status(path)))
        elif a.model_cmd == "verify":
            print(json.dumps(verify_model(path)))
        elif a.model_cmd == "test":
            backend = create_chat_backend(load_settings())
            started = __import__("time").perf_counter()
            result = {key: backend.health().get(key) for key in (
                "backend", "backend_version", "cpu_name", "cpu_arch", "avx_available", "avx2_available",
                "fma_available", "f16c_available", "bmi2_available", "runtime_required_isa",
                "runtime_cpu_compatible")}
            result.update(model_path=str(path), model_valid=backend.health().get("model_valid"),
                          model_load_ok=False, load_ms=None, tokenizer_ok=False, generation_ok=False,
                          failure_type=None, failure_code=None, failure_reason=None, retryable=None)
            try:
                backend.ensure_loaded()
                load_ms = (__import__("time").perf_counter() - started) * 1000
                tokenizer_ok = backend.count_tokens("AXP local model self test") > 0
                answer = backend.generate(system_prompt="Answer briefly.", user_prompt="Reply with OK.")
                health, telemetry = backend.health(), backend.last_telemetry
                result.update(model_load_ok=True, load_ms=load_ms, model_load_ms=load_ms,
                              model_name=health.get("model_name"), tokenizer_ok=tokenizer_ok,
                              generation_ok=bool(answer), **{key: telemetry.get(key) for key in (
                                  "time_to_first_token_ms", "completion_tokens", "generation_ms",
                                  "decode_tokens_per_second", "overall_tokens_per_second", "n_threads",
                                  "n_threads_batch", "n_batch")})
            except Exception as exc:
                LOGGER.exception("Chat model self-test failed type=%s", type(exc).__name__)
                failed = backend.health()
                result.update({key: failed.get(key) for key in
                               ("failure_type", "failure_code", "failure_reason", "retryable")})
            print(json.dumps(result))
        elif a.model_cmd == "intel-test":
            settings = load_settings(); settings.update(chat_inference_device="intel_gpu", intel_diagnostic=True)
            profile = catalog_model(settings.get("chat_active_model_id"))
            if profile is None: p.error("chat-model intel-test requires an active catalog model")
            runtime = InferenceRuntimeManager(settings); backend = runtime.backend
            result = {"runtime": "b10516", "probe_returncode": runtime.hardware.sycl_probe_returncode,
                "probe_device_id": runtime.hardware.sycl_device_id, "probe_device_name": runtime.hardware.sycl_device_name,
                "probe_stdout_excerpt": runtime.hardware.sycl_probe_stdout_excerpt,
                "probe_stderr_excerpt": runtime.hardware.sycl_probe_stderr_excerpt, "model_id": profile.id,
                "model_name": profile.name, "selected_device": runtime.hardware.sycl_device_id,
                "verbosity": 5, "offload_requested": True, "model_load_ok": False, "axp_ok": False}
            try:
                backend.ensure_loaded(); health = backend.health()
                answer = backend.generate(system_prompt="Follow the instruction exactly.",
                                          user_prompt="Reply with exactly: AXP_OK")
                telemetry = backend.last_telemetry
                result.update(model_load_ok=True, axp_ok=answer.strip() == "AXP_OK",
                    session_id=health.get("sidecar_session_id"), offloaded_layers=health.get("offloaded_layers"),
                    total_layers=health.get("total_layers"), gpu_buffer_bytes=health.get("gpu_buffer_bytes"),
                    cpu_buffer_bytes=health.get("cpu_buffer_bytes"), native_markers=health.get("native_gpu_markers"),
                    prompt_tokens=telemetry.get("prompt_tokens"), prompt_eval_ms=telemetry.get("prompt_eval_ms"),
                    prompt_eval_tps=telemetry.get("prompt_eval_tokens_per_second"),
                    completion_tokens=telemetry.get("completion_tokens"), decode_ms=telemetry.get("decode_ms"),
                    decode_tps=telemetry.get("decode_tokens_per_second"), qualification="pass" if answer.strip() == "AXP_OK" else "fail")
            except Exception as exc:
                failure = backend.health().get('failure_type', 'intel_gpu_backend_failed')
                health = backend.health()
                result.update(qualification="fail", failure_type=failure, failure_detail=str(exc),
                    session_id=health.get("sidecar_session_id"), offloaded_layers=health.get("offloaded_layers"),
                    total_layers=health.get("total_layers"), gpu_buffer_bytes=health.get("gpu_buffer_bytes"),
                    cpu_buffer_bytes=health.get("cpu_buffer_bytes"), native_markers=health.get("native_gpu_markers"))
            finally: runtime.close()
            if a.json_output: print(json.dumps(result, indent=2))
            else:
                for key, value in result.items(): print(f"{key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
        elif a.model_cmd == "benchmark":
            settings = load_settings(); profile = catalog_model(settings.get("chat_active_model_id"))
            if profile is None or not __import__("pathlib").Path(settings["chat_model_path"]).is_file():
                p.error("chat-model benchmark requires an installed active catalog model")
            runtime = InferenceRuntimeManager({**settings, "chat_inference_device": "cpu"})
            if not runtime.hardware.intel_gpu_available:
                p.error("chat-model benchmark requires a qualified Intel SYCL runtime")
            def configured(max_tokens):
                values = {key: getattr(profile, key) for key in profile.__dataclass_fields__}
                values["max_answer_tokens"] = max_tokens
                return type(profile)(**values)
            runner = BenchmarkRunner(lambda limit: runtime._cpu_backend(settings, configured(limit)),
                lambda limit: runtime._make_backend({**settings, "chat_inference_device": "intel_gpu"}, configured(limit)),
                profile.name, {"cpu": runtime.hardware.cpu_name, "intel_gpu": runtime.hardware.intel_gpu_name,
                               "intel_device_id": runtime.hardware.intel_gpu_device_id,
                               "sycl_device": runtime.hardware.sycl_device_name})
            runner.start(a.profile)
            while runner.job.state not in ("complete", "complete_with_errors", "failed", "cancelled"):
                __import__("time").sleep(.2)
            print(json.dumps(runner.job.public(), indent=2))
        elif not a.yes:
            p.error("chat-model remove requires --yes")
        else:
            remove_model(path)
            print(json.dumps({"removed": True}))
        return
    if a.cmd == "health":
        print(json.dumps(capability_report(connect(a.db))))
        return
    con = connect(a.db, readonly=True)
    e = embedder_for_index(con, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), local_only=True)
    if a.cmd == "rag-diagnose":
        service = RagService(backend=create_chat_backend(load_settings()), search_fn=search,
                             connect_fn=connect, db=a.db, embedder=e)
        retrieval, decision = service.evaluate_gate(a.query)
        scores = [{key: row.get(key) for key in
                   ("document_id", "chunk_id", "lexical_score", "vector_similarity", "hybrid_score")}
                  for row in retrieval.candidates]
        print(json.dumps({"query": a.query,
                          "top_lexical_score": max((row.get("lexical_score") or 0 for row in scores), default=0),
                          "top_vector_similarities": sorted(
                              (row.get("vector_similarity") or 0 for row in scores), reverse=True)[:10],
                          "scores": scores, "answerability": decision.public()}, ensure_ascii=False, indent=2))
        return
    if a.cmd == "rag-eval":
        cases = load_cases(a.cases)
        rag = RagService(backend=create_chat_backend(load_settings()), search_fn=search, connect_fn=connect,
                         db=a.db, embedder=e)
        captured = {}

        def runner(question, mode):
            if question not in captured:
                captured[question] = rag.retrieve(question)
            retrieval = captured[question]
            if mode == "full":
                return rag.ask(question, retrieval=retrieval)
            _, decision = rag.evaluate_gate(question, retrieval=retrieval)
            return {"answerable": decision.answerable, "decision": decision.public(),
                    "timings": retrieval.timings}
        result = evaluate(cases, runner, mode="full" if a.full else "gate-only")
        result["answerability_config"] = __import__("dataclasses").asdict(
            __import__("axp_client.rag.answerability", fromlist=["AnswerabilityConfig"]).AnswerabilityConfig())
        result["model_manifest"] = model_status(load_settings()["chat_model_path"]).get("manifest")
        result["backend_version"] = rag.health().get("backend_version")
        result["axp_commit"] = os.getenv("GITHUB_SHA")
        if a.sweep:
            if a.full:
                p.error("--sweep is gate-only")
            swept = [{**case, "hits": captured[case["question"]].content_evidence} for case in cases]
            result["threshold_sweep"] = threshold_sweep(swept)
        if a.output:
            __import__("pathlib").Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(format_summary(result))
        return
    if a.cmd == "search":
        config = SearchConfig(a.lexical_candidates, a.vector_candidates, a.rerank_candidates)
        reranker = Reranker(cache_dir=os.getenv("FASTEMBED_CACHE_PATH")) if a.profile == "quality" else None
        value = search(con, e, a.query, a.limit, profile=a.profile, explain=a.explain, reranker=reranker, config=config)
        print(json.dumps(value, ensure_ascii=False))
        return
    if a.cmd == "ask":
        service = RagService(backend=create_chat_backend(load_settings()), search_fn=search,
                             connect_fn=connect, db=a.db, embedder=e)
        try:
            value = service.ask(a.question, debug=a.debug)
        except ChatUnavailableError:
            value = {"error": "chat_model_unavailable"}
        except GenerationFailedError:
            value = {"status": "generation_unavailable", "answerable": False, "error": "local_generation_failed"}
        print(json.dumps(value, ensure_ascii=False))
        return
    serve(a.db, e, a.host, a.port)
