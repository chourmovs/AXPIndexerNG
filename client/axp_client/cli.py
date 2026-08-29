import argparse
import json
import os

from axp_core.database import capability_report, connect
from axp_core.hybrid import SearchConfig
from axp_core.runtime import configure_logging, load_settings
from axp_daemon.embeddings import embedder_for_index

from .rag.evaluation import evaluate, format_summary, load_cases, threshold_sweep
from .rag.factory import create_chat_backend
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
    model_remove = model_sub.add_parser("remove")
    model_remove.add_argument("--yes", action="store_true")
    evaluation = s.add_parser("rag-eval")
    evaluation.add_argument("--db", required=True)
    evaluation.add_argument("--cases", required=True)
    evaluation.add_argument("--full", action="store_true")
    evaluation.add_argument("--output")
    evaluation.add_argument("--sweep", action="store_true")
    for cmd in ("health", "search", "serve", "ask"):
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
    a = p.parse_args(argv)
    if a.cmd == "chat-model":
        path = load_settings()["chat_model_path"]
        if a.model_cmd == "import":
            print(json.dumps(import_model(a.file, path)))
        elif a.model_cmd == "status":
            print(json.dumps(model_status(path)))
        elif a.model_cmd == "verify":
            print(json.dumps(verify_model(path)))
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
