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

from .rag.factory import create_chat_backend
from .rag.service import ChatBusyError, ChatUnavailableError, GenerationFailedError, RagService
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


def make_handler(db, embedder, open_file=open_with_default_application, rag_service=None):
    quality_reranker = None

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if urlparse(self.path).path.startswith("/api/ask"):
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
            names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if url.path not in names:
                return self.send_json({"error": "not found"}, 404)
            data = (WEB / names[url.path]).read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[names[url.path]],
            )
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/api/ask":
                def send(value, status=200):
                    return self.send_json(value, status)
                if not _is_loopback(self.client_address[0]):
                    return send({"error": "ask is only available locally"}, 403)
                if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                    return send({"error": "forbidden_origin"}, 403)
                origin = self.headers.get("Origin")
                if origin:
                    parsed = urlparse(origin)
                    server_port = self.server.server_address[1]
                    default_port = 443 if parsed.scheme == "https" else 80
                    try:
                        origin_local = parsed.scheme in ("http", "https") and _is_loopback(parsed.hostname or "")
                        origin_port = parsed.port or default_port
                    except ValueError:
                        origin_local, origin_port = False, None
                    if not origin_local or origin_port != server_port:
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
                if rag_service is None:
                    return send({"error": "chat_model_unavailable"}, 503)
                try:
                    return send(rag_service.ask(question, debug=body.get("debug", False)))
                except ChatUnavailableError:
                    reason = rag_service.health().get("reason")
                    return send({"error": reason if reason in ("model_invalid", "model_load_failed")
                                 else "chat_model_unavailable"}, 503)
                except ChatBusyError:
                    return send({"error": "chat_busy"}, 429)
                except GenerationFailedError:
                    return send({"status": "generation_unavailable", "answerable": False,
                                           "error": "local_generation_failed"}, 503)
            if url.path == "/api/shutdown":
                if not _is_loopback(self.client_address[0]):
                    return self.send_json({"error": "shutdown is only available locally"}, 403)
                LOGGER.info("Web client shutdown requested")
                self.send_json({"status": "stopping", "pid": os.getpid()})
                threading.Thread(target=self.server.shutdown, name="axp-client-shutdown", daemon=True).start()
                return
            parts = url.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "document"] and parts[3] == "open":
                try:
                    document_id = int(parts[2])
                except ValueError:
                    return self.send_json({"error": "document not found"}, 404)
                with connect(db, readonly=True) as con:
                    row = con.execute("SELECT path FROM documents WHERE id=?", (document_id,)).fetchone()
                if row is None:
                    return self.send_json({"error": "document not found", "document_id": document_id}, 404)
                path = Path(row["path"])
                if not path.exists():
                    path = access_path_for(row["path"], load_settings().get("background_drive_mappings", {}))
                if not path.exists():
                    return self.send_json(
                        {"error": "The indexed file no longer exists on disk.", "document_id": document_id}, 410
                    )
                try:
                    LOGGER.info("Opening indexed document id=%s path=%s", document_id, path)
                    open_file(path)
                except (AttributeError, OSError):
                    LOGGER.exception("Could not open indexed document id=%s path=%s", document_id, path)
                    return self.send_json(
                        {"error": "The file could not be opened.", "document_id": document_id}, 500
                    )
                return self.send_json({"status": "opened", "document_id": document_id})
            return self.send_json({"error": "not found"}, 404)

        def log_message(self, *args):
            LOGGER.info(*args)

    return Handler


def serve(db, embedder, host="127.0.0.1", port=8765):
    settings = load_settings()
    backend = create_chat_backend(settings)
    rag_service = RagService(backend=backend, search_fn=search, connect_fn=connect, db=db, embedder=embedder)
    server = ThreadingHTTPServer((host, port), make_handler(db, embedder, rag_service=rag_service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        LOGGER.info("Web client stopped")
