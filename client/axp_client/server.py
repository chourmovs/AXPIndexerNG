import ipaddress
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from axp_core.background import access_path_for
from axp_core.database import connect
from axp_core.runtime import configure_logging, load_settings

from .rag.model_manager import ModelManager, ModelManagerError
from .rag.runtime_manager import InferenceRuntimeManager
from .rag.llama_cpp_backend import GenerationCancelled
from .rag.service import (
    ChatBusyError,
    ChatUnavailableError,
    ContextPreparationFailedError,
    GenerationFailedError,
    ModelLoadFailedError,
    RagService,
    ValidationFailedError,
)
from .reranker import Reranker
from .search import search

WEB = Path(__file__).parent / "web"
LOGGER = configure_logging("axp_client", "client.log")


def open_with_default_application(path):
    """Open *path* using its Windows file association."""
    os.startfile(str(path))


def _is_loopback(address):
    if address.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


MAX_ASK_BODY = 64 * 1024
MAX_QUESTION_LENGTH = 4_000
CHAT_FAILURE_CODES = {"model_missing", "model_invalid", "backend_missing",
                      "backend_cpu_incompatible", "model_load_failed", "model_template_incompatible"}


def _chat_failure_code(rag_service, default="chat_model_unavailable"):
    health = rag_service.health()
    code = health.get("failure_type") or health.get("reason")
    return code if code in CHAT_FAILURE_CODES else default


class DocumentNotFoundError(Exception):
    pass


def resolve_document_access_path(db, document_id, *, directory=False):
    """Resolve a document action from its database identity, never browser input."""
    with connect(db, readonly=True) as con:
        row = con.execute("SELECT path FROM documents WHERE id=?", (document_id,)).fetchone()
    if row is None:
        raise DocumentNotFoundError
    logical = Path(row["path"])
    fallback = Path(access_path_for(row["path"], load_settings().get("background_drive_mappings", {})))
    candidates = (logical.parent, fallback.parent) if directory else (logical, fallback)
    return next((path for path in candidates if path.exists()), None)


def make_handler(db, embedder, open_file=open_with_default_application, rag_service=None, model_manager=None):
    quality_reranker = None

    class Handler(BaseHTTPRequestHandler):
        def local_action_allowed(self):
            if not _is_loopback(self.client_address[0]):
                return False
            if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                return False
            origin = self.headers.get("Origin")
            if not origin:
                return True
            parsed = urlparse(origin)
            default_port = 443 if parsed.scheme == "https" else 80
            try:
                return (parsed.scheme in ("http", "https") and _is_loopback(parsed.hostname or "")
                        and (parsed.port or default_port) == self.server.server_address[1])
            except ValueError:
                return False

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if urlparse(self.path).path.startswith(("/api/ask", "/api/models")):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self.send_json({"status": "ok", "pid": os.getpid()})
            if url.path == "/api/search":
                nonlocal quality_reranker
                q = parse_qs(url.query).get("q", [""])[0]
                explain = parse_qs(url.query).get("explain", ["0"])[0] == "1"
                profile = parse_qs(url.query).get("profile", ["hybrid"])[0]
                if profile == "quality" and quality_reranker is None:
                    quality_reranker = Reranker(cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
                with connect(db, readonly=True) as con:
                    return self.send_json(
                        search(
                            con,
                            embedder,
                            q,
                            profile=profile,
                            explain=explain,
                            reranker=quality_reranker if profile == "quality" else None,
                        )
                        if q
                        else []
                    )
            if url.path == "/api/ask/health":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "ask is only available locally"}, 403)
                if rag_service is None:
                    return self.send_json({"available": False, "reason": "not_configured"})
                return self.send_json(rag_service.health())
            if url.path == "/api/models":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json(model_manager.catalog() if model_manager else {"models": []})
            if url.path == "/api/models/benchmark":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json(model_manager.catalog().get("benchmark", {"state": "idle"}) if model_manager else
                                      {"state": "idle"})
            if url.path.startswith("/api/document/"):
                try:
                    doc_id = int(url.path.rsplit("/", 1)[1])
                except ValueError:
                    return self.send_json({"error": "not found"}, 404)
                with connect(db, readonly=True) as con:
                    doc = con.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
                    if not doc:
                        return self.send_json({"error": "not found"}, 404)
                    chunks = [
                        dict(x)
                        for x in con.execute("SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_no", (doc_id,))
                    ]
                    return self.send_json({"document": dict(doc), "chunks": chunks})
            names = {"/": "index.html", "/app.js": "app.js", "/api.js": "api.js", "/search.js": "search.js",
                     "/ask.js": "ask.js", "/documents.js": "documents.js", "/ui.js": "ui.js",
                     "/style.css": "style.css"}
            if url.path not in names:
                return self.send_json({"error": "not found"}, 404)
            data = (WEB / names[url.path]).read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/css" if names[url.path].endswith(".css") else
                "text/html" if names[url.path].endswith(".html") else "text/javascript",
            )
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            if url.path == "/api/models/device":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if model_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict) or not isinstance(body.get("device"), str):
                        return self.send_json({"error": "invalid_inference_device"}, 400)
                    return self.send_json(model_manager.set_device(body["device"]))
                except (ValueError, json.JSONDecodeError):
                    return self.send_json({"error": "invalid_request"}, 400)
                except ModelManagerError as exc:
                    status = 409 if exc.code in ("chat_busy", "intel_gpu_unavailable") else 400
                    return self.send_json({"error": exc.code, **exc.details}, status)
            if url.path in ("/api/models/accelerator/download", "/api/models/accelerator/remove",
                            "/api/models/accelerator/cancel", "/api/models/accelerator/probe"):
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if model_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    if url.path.endswith("/download"): result = model_manager.start_accelerator_download()
                    elif url.path.endswith("/cancel"): result = model_manager.cancel_accelerator_download()
                    elif url.path.endswith("/probe"): result = model_manager.retry_accelerator_probe()
                    else: result = model_manager.remove_accelerator()
                    return self.send_json(result, 202 if url.path.endswith(("/download", "/cancel")) else 200)
                except ModelManagerError as exc:
                    return self.send_json({"error": exc.code, **exc.details}, 409)
            if url.path in ("/api/models/benchmark", "/api/models/benchmark/cancel"):
                if not self.local_action_allowed(): return self.send_json({"error": "forbidden_origin"}, 403)
                if model_manager is None: return self.send_json({"error": "not_configured"}, 503)
                try:
                    if url.path.endswith("/cancel"): return self.send_json(model_manager.cancel_benchmark(), 202)
                    length = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(length) or b"{}")
                    return self.send_json(model_manager.start_benchmark(body.get("profile", "quick")), 202)
                except ModelManagerError as exc: return self.send_json({"error": exc.code, **exc.details}, 409)
            if len(parts) == 4 and parts[:2] == ["api", "models"] and parts[3] in (
                    "download", "cancel", "activate", "remove"):
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if model_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    if parts[3] == "download":
                        length = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(length) or b"{}")
                        return self.send_json(model_manager.start_download(parts[2], activate=bool(body.get("activate"))), 202)
                    result = getattr(model_manager, parts[3])(parts[2])
                    return self.send_json(result)
                except ModelManagerError as exc:
                    status = 404 if exc.code == "model_not_found" else 409
                    return self.send_json({"error": exc.code, **exc.details}, status)
            if url.path == "/api/ask/model/retry":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if rag_service is None:
                    return self.send_json({"error": "chat_model_unavailable"}, 503)
                return self.send_json(rag_service.retry_model())
            if url.path == "/api/ask/cancel":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if rag_service is None or not rag_service.cancel_generation():
                    return self.send_json({"error": "no_active_generation"}, 409)
                return self.send_json({"status": "cancel_requested"}, 202)
            if url.path in ("/api/ask", "/api/ask/stream"):
                def send(value, status=200):
                    return self.send_json(value, status)
                if not _is_loopback(self.client_address[0]):
                    return send({"error": "ask is only available locally"}, 403)
                if not self.local_action_allowed():
                    return send({"error": "forbidden_origin"}, 403)
                if self.headers.get_content_type() != "application/json":
                    return send({"error": "unsupported_media_type"}, 415)
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    return send({"error": "invalid_request"}, 400)
                try:
                    length = int(raw_length)
                except ValueError:
                    return send({"error": "invalid_request"}, 400)
                if length < 0:
                    return send({"error": "invalid_request"}, 400)
                if length > MAX_ASK_BODY:
                    return send({"error": "request_too_large"}, 413)
                try:
                    body = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return send({"error": "invalid_json"}, 400)
                if not isinstance(body, dict) or not isinstance(body.get("question"), str):
                    return send({"error": "invalid_question"}, 400)
                question = body["question"].strip()
                if not question:
                    return send({"error": "invalid_question"}, 400)
                if len(question) > MAX_QUESTION_LENGTH:
                    return send({"error": "question_too_large"}, 413)
                if not isinstance(body.get("debug", False), bool):
                    return send({"error": "invalid_debug"}, 400)
                search_depth = body.get("search_depth", 0)
                if type(search_depth) is not int or search_depth not in (0, 1):
                    return send({"error": "invalid_request", "code": "invalid_search_depth"}, 400)
                if rag_service is None:
                    return send({"error": "chat_model_unavailable"}, 503)
                if url.path == "/api/ask/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()

                    terminal_sent = False
                    disconnected = False

                    def progress(event):
                        nonlocal terminal_sent, disconnected
                        is_terminal = event.get("event") in ("final", "error", "cancelled")
                        if is_terminal:
                            if terminal_sent:
                                return
                            terminal_sent = True
                        try:
                            self.wfile.write(json.dumps(event).encode() + b"\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            disconnected = True
                            if not is_terminal:
                                raise

                    try:
                        ask_options = {"debug": body.get("debug", False), "progress": progress}
                        if search_depth:
                            ask_options["search_depth"] = search_depth
                        response = rag_service.ask(question, **ask_options)
                        progress({"event": "final", "response": response})
                    except ChatUnavailableError:
                        progress({"event": "error", "error": _chat_failure_code(rag_service)})
                    except ChatBusyError:
                        progress({"event": "error", "error": "chat_busy"})
                    except GenerationCancelled:
                        # RagService emitted the terminal cancellation after the native iterator exited.
                        pass
                    except GenerationFailedError:
                        progress({"event": "error", "error": "local_generation_failed"})
                    except ModelLoadFailedError:
                        progress({"event": "error", "error": _chat_failure_code(rag_service, "model_load_failed")})
                    except ContextPreparationFailedError:
                        progress({"event": "error", "error": "context_preparation_failed"})
                    except ValidationFailedError:
                        progress({"event": "error", "error": "validation_failed"})
                    except (BrokenPipeError, ConnectionResetError):
                        disconnected = True
                        LOGGER.info("Ask stream client disconnected")
                    except Exception:
                        LOGGER.exception("Unexpected Ask stream failure")
                        if not terminal_sent:
                            try:
                                progress({"event": "error", "error": "stream_internal_error"})
                            except (BrokenPipeError, ConnectionResetError):
                                disconnected = True
                    finally:
                        if not terminal_sent and not disconnected:
                            try:
                                progress({"event": "error", "error": "stream_internal_error"})
                            except (BrokenPipeError, ConnectionResetError):
                                LOGGER.info("Ask stream client disconnected before terminal event")
                    return
                try:
                    ask_options = {"debug": body.get("debug", False)}
                    if search_depth:
                        ask_options["search_depth"] = search_depth
                    return send(rag_service.ask(question, **ask_options))
                except ChatUnavailableError:
                    return send({"error": _chat_failure_code(rag_service)}, 503)
                except ChatBusyError:
                    return send({"error": "chat_busy"}, 429)
                except GenerationFailedError:
                    return send({"status": "generation_unavailable", "answerable": False,
                                           "error": "local_generation_failed"}, 503)
                except ModelLoadFailedError:
                    return send({"error": _chat_failure_code(rag_service, "model_load_failed")}, 503)
                except ContextPreparationFailedError:
                    return send({"error": "context_preparation_failed"}, 503)
                except ValidationFailedError:
                    return send({"error": "validation_failed"}, 503)
            if url.path == "/api/shutdown":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "shutdown is only available locally"}, 403)
                LOGGER.info("Web client shutdown requested")
                self.send_json({"status": "stopping", "pid": os.getpid()})
                threading.Thread(target=self.server.shutdown, name="axp-client-shutdown", daemon=True).start()
                return
            parts = url.path.strip("/").split("/")
            if (len(parts) == 4 and parts[:2] == ["api", "document"]
                    and parts[3] in ("open", "open-dir")):
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                try:
                    document_id = int(parts[2])
                except ValueError:
                    return self.send_json({"error": "document not found"}, 404)
                try:
                    path = resolve_document_access_path(db, document_id, directory=parts[3] == "open-dir")
                except DocumentNotFoundError:
                    return self.send_json({"error": "document not found", "document_id": document_id}, 404)
                if path is None:
                    return self.send_json(
                        {"error": "The indexed directory is no longer accessible." if parts[3] == "open-dir"
                         else "The indexed file no longer exists on disk.", "document_id": document_id}, 410
                    )
                try:
                    LOGGER.info("Opening indexed document action=%s id=%s path=%s", parts[3], document_id, path)
                    open_file(path)
                except (AttributeError, OSError):
                    LOGGER.exception("Could not open indexed document id=%s path=%s", document_id, path)
                    return self.send_json(
                        {"error": "The directory could not be opened." if parts[3] == "open-dir"
                         else "The file could not be opened.", "document_id": document_id}, 500
                    )
                return self.send_json({"status": "opened", "document_id": document_id})
            return self.send_json({"error": "not found"}, 404)

        def log_message(self, *args):
            status = str(args[1]) if len(args) > 1 else ""
            poll = self.command == "GET" and urlparse(self.path).path in {
                "/health", "/api/ask/health", "/api/models", "/api/models/benchmark"}
            (LOGGER.debug if poll and status.startswith("2") else LOGGER.info)(*args)

    return Handler


def serve(db, embedder, host="127.0.0.1", port=8765):
    settings = load_settings()
    backend = InferenceRuntimeManager(settings)
    rag_service = RagService(backend=backend, search_fn=search, connect_fn=connect, db=db, embedder=embedder)
    model_manager = ModelManager(settings["model_cache"], runtime=rag_service)
    server = ThreadingHTTPServer((host, port), make_handler(db, embedder, rag_service=rag_service,
                                                            model_manager=model_manager))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        rag_service.close()
        LOGGER.info("Web client stopped")
