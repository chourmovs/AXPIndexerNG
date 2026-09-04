import ipaddress
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from axp_core.background import access_path_for
from axp_core.build_info import build_info
from axp_core.database import SearchReaderPool, connect, search_reader_diagnostics
from axp_core.runtime import configure_logging, load_settings, validate_loopback_host
from axp_daemon.embeddings import embedder_for_index

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
from .startup_state import ClientStartupState

WEB = Path(__file__).parent / "web"
LOGGER = configure_logging("axp_client", "client.log")
_BUILD = build_info()
LOGGER.info("AXP build version=%s commit=%s release=%s", _BUILD["version"],
            _BUILD["commit"] or "unknown", _BUILD["release"])


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
MAX_ADMIN_BODY = 8 * 1024
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


def make_handler(db, embedder=None, open_file=open_with_default_application, rag_service=None, model_manager=None,
                 search_readers=None, runtime=None, startup_state=None):
    quality_reranker = None
    search_readers = search_readers or (None if runtime else SearchReaderPool(db))

    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
                             "frame-ancestors 'none'")
            super().end_headers()

        def read_json_body(self, max_bytes):
            if self.headers.get_content_type() != "application/json":
                return None, ("unsupported_media_type", 415)
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None, ("invalid_request", 400)
            try:
                length = int(raw_length)
            except ValueError:
                return None, ("invalid_request", 400)
            if length < 0:
                return None, ("invalid_request", 400)
            if length > max_bytes:
                return None, ("request_too_large", 413)
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None, ("invalid_json", 400)
            if not isinstance(body, dict):
                return None, ("invalid_request", 400)
            return body, None

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
            if urlparse(self.path).path.startswith(("/api/ask", "/api/models")):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self.send_json({"status": "ok", "pid": os.getpid()})
            if url.path == "/api/version":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json(build_info())
            if url.path == "/api/startup":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                return self.send_json(startup_state.snapshot() if startup_state else {
                    "phase": "ready", "server": {"state": "ready"},
                    "search": {"state": "ready"}, "local_ai": {"state": "ready"}})
            if url.path == "/api/search":
                http_started = time.perf_counter()
                nonlocal quality_reranker
                current_embedder = runtime.embedder if runtime else embedder
                current_readers = runtime.search_readers if runtime else search_readers
                if startup_state and (current_embedder is None or current_readers is None):
                    status = startup_state.snapshot()["search"]
                    code = "search_initializing" if status["state"] == "initializing" else "search_unavailable"
                    return self.send_json({"error": code, "retryable": status["state"] == "initializing"}, 503)
                q = parse_qs(url.query).get("q", [""])[0]
                explain = parse_qs(url.query).get("explain", ["0"])[0] == "1"
                profile = parse_qs(url.query).get("profile", ["hybrid"])[0]
                if profile == "quality" and quality_reranker is None:
                    quality_reranker = Reranker(cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
                acquire_started = time.perf_counter()
                with current_readers.acquire() as (con, reader_reused):
                    db_acquire_ms = (time.perf_counter() - acquire_started) * 1000
                    result = search(
                            con,
                            current_embedder,
                            q,
                            profile=profile,
                            explain=True,
                            reranker=quality_reranker if profile == "quality" else None,
                        ) if q else {"results": [], "timings": {}, "candidate_counts": {}}
                    result["timings"]["db_acquire_ms"] = db_acquire_ms
                    result["diagnostics"] = {
                        "reader_reused": reader_reused,
                        **search_reader_diagnostics(con),
                    }
                    LOGGER.info("Search complete query_length=%s timings=%s candidate_counts=%s diagnostics=%s",
                                len(q), result.get("timings"), result.get("candidate_counts"),
                                result.get("diagnostics"))
                    result["timings"]["http_search_total_ms"] = (time.perf_counter() - http_started) * 1000
                    return self.send_json(result if explain else result["results"])
            if url.path == "/api/ask/health":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "ask is only available locally"}, 403)
                current_rag = runtime.rag_service if runtime else rag_service
                if current_rag is None:
                    return self.send_json({"available": False, "reason": "not_configured"})
                return self.send_json(current_rag.health())
            if url.path == "/api/models":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                manager = runtime.model_manager if runtime else model_manager
                return self.send_json(manager.catalog() if manager else {"models": []})
            if url.path == "/api/models/benchmark":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                manager = runtime.model_manager if runtime else model_manager
                return self.send_json(manager.catalog().get("benchmark", {"state": "idle"}) if manager else
                                      {"state": "idle"})
            if url.path == "/api/models/qualification":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "forbidden"}, 403)
                manager = runtime.model_manager if runtime else model_manager
                return self.send_json(manager.qualification_status() if manager else {"state": "idle"})
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
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            url = urlparse(self.path)
            parts = url.path.strip("/").split("/")
            current_rag = runtime.rag_service if runtime else rag_service
            current_manager = runtime.model_manager if runtime else model_manager
            if url.path == "/api/models/device":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if current_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    body, error = self.read_json_body(MAX_ADMIN_BODY)
                    if error:
                        return self.send_json({"error": error[0]}, error[1])
                    if not isinstance(body, dict) or not isinstance(body.get("device"), str):
                        return self.send_json({"error": "invalid_inference_device"}, 400)
                    return self.send_json(current_manager.set_device(body["device"]))
                except ModelManagerError as exc:
                    status = 409 if exc.code in ("chat_busy", "intel_gpu_unavailable") else 400
                    return self.send_json({"error": exc.code, **exc.details}, status)
            if url.path in ("/api/models/accelerator/download", "/api/models/accelerator/remove",
                            "/api/models/accelerator/cancel", "/api/models/accelerator/probe"):
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if current_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    if url.path.endswith("/download"): result = current_manager.start_accelerator_download()
                    elif url.path.endswith("/cancel"): result = current_manager.cancel_accelerator_download()
                    elif url.path.endswith("/probe"): result = current_manager.retry_accelerator_probe()
                    else: result = current_manager.remove_accelerator()
                    return self.send_json(result, 202 if url.path.endswith(("/download", "/cancel")) else 200)
                except ModelManagerError as exc:
                    return self.send_json({"error": exc.code, **exc.details}, 409)
            if url.path in ("/api/models/benchmark", "/api/models/benchmark/cancel"):
                if not self.local_action_allowed(): return self.send_json({"error": "forbidden_origin"}, 403)
                if current_manager is None: return self.send_json({"error": "not_configured"}, 503)
                try:
                    if url.path.endswith("/cancel"): return self.send_json(current_manager.cancel_benchmark(), 202)
                    body, error = self.read_json_body(MAX_ADMIN_BODY)
                    if error: return self.send_json({"error": error[0]}, error[1])
                    return self.send_json(current_manager.start_benchmark(body.get("profile", "quick")), 202)
                except ModelManagerError as exc: return self.send_json({"error": exc.code, **exc.details}, 409)
            if url.path in ("/api/models/qualification/start", "/api/models/qualification/cancel"):
                if not self.local_action_allowed(): return self.send_json({"error": "forbidden_origin"}, 403)
                if current_manager is None: return self.send_json({"error": "not_configured"}, 503)
                try:
                    if url.path.endswith("/cancel"):
                        return self.send_json(current_manager.cancel_qualification(), 202)
                    body, error = self.read_json_body(MAX_ADMIN_BODY)
                    if error: return self.send_json({"error": error[0]}, error[1])
                    return self.send_json(current_manager.start_qualification(body.get("profile", "standard"),
                        body.get("model")), 202)
                except ModelManagerError as exc: return self.send_json({"error": exc.code, **exc.details}, 409)
            if len(parts) == 4 and parts[:2] == ["api", "models"] and parts[3] in (
                    "download", "cancel", "activate", "remove"):
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if current_manager is None:
                    return self.send_json({"error": "not_configured"}, 503)
                try:
                    if parts[3] == "download":
                        body, error = self.read_json_body(MAX_ADMIN_BODY)
                        if error: return self.send_json({"error": error[0]}, error[1])
                        return self.send_json(current_manager.start_download(parts[2], activate=bool(body.get("activate"))), 202)
                    result = getattr(current_manager, parts[3])(parts[2])
                    return self.send_json(result)
                except ModelManagerError as exc:
                    status = 404 if exc.code == "model_not_found" else 409
                    return self.send_json({"error": exc.code, **exc.details}, status)
            if url.path == "/api/ask/model/retry":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if current_rag is None:
                    return self.send_json({"error": "chat_model_unavailable"}, 503)
                return self.send_json(current_rag.retry_model())
            if url.path == "/api/ask/cancel":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
                if current_rag is None or not current_rag.cancel_generation():
                    return self.send_json({"error": "no_active_generation"}, 409)
                return self.send_json({"status": "cancel_requested"}, 202)
            if url.path in ("/api/ask", "/api/ask/stream"):
                if runtime:
                    runtime.user_activity.set()
                def send(value, status=200):
                    return self.send_json(value, status)
                if not _is_loopback(self.client_address[0]):
                    return send({"error": "ask is only available locally"}, 403)
                if not self.local_action_allowed():
                    return send({"error": "forbidden_origin"}, 403)
                body, error = self.read_json_body(MAX_ASK_BODY)
                if error:
                    return send({"error": error[0]}, error[1])
                if not isinstance(body.get("question"), str):
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
                ai_startup = startup_state.snapshot()["local_ai"] if startup_state else {"state": "ready"}
                if current_rag is None or ai_startup["state"] not in ("ready", "ready_with_warmup_warning"):
                    return send({"error": "local_ai_initializing", "retryable": True,
                                 "phase": ai_startup["state"]}, 503)
                if url.path == "/api/ask/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Cache-Control", "no-store")
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
                        response = current_rag.ask(question, **ask_options)
                        progress({"event": "final", "response": response})
                    except ChatUnavailableError:
                        progress({"event": "error", "error": _chat_failure_code(current_rag)})
                    except ChatBusyError:
                        progress({"event": "error", "error": "chat_busy"})
                    except GenerationCancelled:
                        # RagService emitted the terminal cancellation after the native iterator exited.
                        pass
                    except GenerationFailedError:
                        progress({"event": "error", "error": "local_generation_failed"})
                    except ModelLoadFailedError:
                        progress({"event": "error", "error": _chat_failure_code(current_rag, "model_load_failed")})
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
                    return send(current_rag.ask(question, **ask_options))
                except ChatUnavailableError:
                    return send({"error": _chat_failure_code(current_rag)}, 503)
                except ChatBusyError:
                    return send({"error": "chat_busy"}, 429)
                except GenerationFailedError:
                    return send({"status": "generation_unavailable", "answerable": False,
                                           "error": "local_generation_failed"}, 503)
                except ModelLoadFailedError:
                    return send({"error": _chat_failure_code(current_rag, "model_load_failed")}, 503)
                except ContextPreparationFailedError:
                    return send({"error": "context_preparation_failed"}, 503)
                except ValidationFailedError:
                    return send({"error": "validation_failed"}, 503)
            if url.path == "/api/shutdown":
                if not self.local_action_allowed():
                    return self.send_json({"error": "forbidden_origin"}, 403)
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


class ClientRuntime:
    """Own independently initialized search and AI capabilities."""
    def __init__(self, db, settings, state):
        self.db, self.settings, self.state = db, settings, state
        self.cancel = threading.Event()
        self.user_activity = threading.Event()
        self._lock = threading.Lock()
        self._search_ready = threading.Event()
        self._embedder = self._search_readers = self._rag_service = self._model_manager = None

    def _get(self, name):
        with self._lock:
            return getattr(self, "_" + name)

    @property
    def embedder(self): return self._get("embedder")
    @property
    def search_readers(self): return self._get("search_readers")
    @property
    def rag_service(self): return self._get("rag_service")
    @property
    def model_manager(self): return self._get("model_manager")

    def start(self):
        threading.Thread(target=self._initialize_search, name="axp-client-search-init", daemon=True).start()
        threading.Thread(target=self._initialize_ai, name="axp-client-ai-init", daemon=True).start()

    def _initialize_search(self):
        try:
            with connect(self.db, readonly=True) as con:
                embedder = embedder_for_index(con, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), local_only=True)
            self.state.timing("embedder_ready_ms")
            if self.cancel.is_set(): return
            readers = SearchReaderPool(self.db)
            with self._lock:
                self._embedder, self._search_readers = embedder, readers
            self.state.update("search", state="ready", phase=None)
            self.state.timing("search_ready_ms")
        except Exception as exc:
            LOGGER.exception("Background search initialization failed")
            self.state.update("search", state="failed", phase=None, error=str(exc))
        finally:
            self._search_ready.set()

    def _initialize_ai(self):
        probe_started = time.perf_counter()
        self.state.update("local_ai", state="probing_gpu", phase="probing_gpu")
        try:
            backend = InferenceRuntimeManager(self.settings)
            self.state.timing("intel_probe_ms", (time.perf_counter() - probe_started) * 1000)
            while not self.cancel.is_set() and not self._search_ready.wait(.1): pass
            if self.cancel.is_set():
                backend.close(); return
            embedder = self.embedder
            if embedder is None:
                raise RuntimeError("search dependencies unavailable")
            rag = RagService(backend=backend, search_fn=search, connect_fn=connect, db=self.db, embedder=embedder)
            manager = ModelManager(self.settings["model_cache"], runtime=rag)
            with self._lock:
                self._rag_service, self._model_manager = rag, manager
            health = rag.health()
            model_id = self.settings.get("chat_active_model_id")
            self.state.update("local_ai", model_id=model_id, model_name=health.get("model_name"),
                              device=health.get("device_name") or health.get("backend"))
            requested = self.settings.get("chat_inference_device", "auto")
            if not model_id:
                self.state.update("local_ai", state="unconfigured", phase=None); return
            if requested not in ("intel_gpu", "auto") or not backend.hardware.intel_gpu_available:
                self.state.update("local_ai", state="ready", phase=None); return
            self.state.update("local_ai", state="loading_model", phase="loading_model")
            loaded = time.perf_counter(); rag.run_when_idle(backend.ensure_loaded)
            self.state.timing("model_load_ms", (time.perf_counter() - loaded) * 1000)
            if self.cancel.is_set(): return
            # A short grace period makes an immediately submitted real request win.
            if self.user_activity.wait(.2):
                warmup = {"state": "skipped_user_activity"}
            else:
                self.state.update("local_ai", state="warming_model", phase="warming_model")
                filler = "Safety data property value unit method source section verified local evidence. " * 95
                LOGGER.info("AI warm-up started model_id=%s", model_id)
                try:
                    telemetry = rag.try_warmup("You are warming a local inference engine.",
                                               filler + "\nQuestion: acknowledge readiness.")
                    warmup = {"state": "completed", **telemetry} if telemetry is not None else {"state": "skipped_busy"}
                    if telemetry is not None:
                        LOGGER.info("Active model warm-up completed model_id=%s prompt_tokens=%s prompt_eval_ms=%s "
                                    "prompt_eval_tps=%s ttft_ms=%s generation_ms=%s warmup_ms=%s", model_id,
                                    telemetry.get("prompt_tokens"), telemetry.get("prompt_eval_ms"),
                                    telemetry.get("prompt_eval_tokens_per_second"), telemetry.get("time_to_first_token_ms"),
                                    telemetry.get("generation_ms"), telemetry.get("warmup_ms"))
                        self.state.timing("model_warmup_ms", telemetry.get("warmup_ms"))
                except Exception as exc:
                    LOGGER.warning("Active model warm-up failed model_id=%s error=%s", model_id, exc)
                    warmup = {"state": "failed", "error": str(exc)}
            health = rag.health()
            final_state = "ready_with_warmup_warning" if warmup["state"] == "failed" else "ready"
            self.state.update("local_ai", state=final_state, phase=None, warmup=warmup,
                              offloaded_layers=health.get("offloaded_layers"), total_layers=health.get("total_layers"))
            self.state.timing("ai_ready_ms")
        except Exception as exc:
            LOGGER.exception("Background local AI initialization failed")
            self.state.update("local_ai", state="failed", phase=None, error=str(exc))

    def close(self):
        self.cancel.set()
        rag, readers = self.rag_service, self.search_readers
        if rag: rag.close()
        if readers: readers.close()


def serve(db, embedder=None, host="127.0.0.1", port=8765):
    validate_loopback_host(host)
    settings = load_settings()
    state = ClientStartupState()
    runtime = ClientRuntime(db, settings, state)
    server = ThreadingHTTPServer((host, port), make_handler(db, runtime=runtime, startup_state=state))
    state.update("server", state="ready")
    state.timing("http_bind_ms")
    LOGGER.info("Startup context component=client pid=%s host=%s port=%s db_path=%s",
                os.getpid(), host, port, db)
    runtime.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()
        timings = state.snapshot()["timings"]
        LOGGER.info("Client startup timings %s", " ".join(f"{key}={value}" for key, value in timings.items()))
        LOGGER.info("Web client stopped")
