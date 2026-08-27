import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from axp_core.database import connect

from .search import search

WEB = Path(__file__).parent / "web"


def make_handler(db, embedder):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self.send_json({"status": "ok"})
            if url.path == "/api/search":
                q = parse_qs(url.query).get("q", [""])[0]
                explain = parse_qs(url.query).get("explain", ["0"])[0] == "1"
                profile = parse_qs(url.query).get("profile", ["hybrid"])[0]
                with connect(db, readonly=True) as con:
                    return self.send_json(search(con, embedder, q, profile=profile, explain=explain) if q else [])
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

        def log_message(self, *args):
            pass

    return Handler


def serve(db, embedder, host="127.0.0.1", port=8765):
    ThreadingHTTPServer((host, port), make_handler(db, embedder)).serve_forever()
