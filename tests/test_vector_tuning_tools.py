import csv
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.benchmark_search_diagnostics import fixture
from scripts.benchmark_vectors import summary
from scripts.benchmark_search_load import request


def test_percentiles_use_observed_nearest_rank():
    result = summary(list(range(1, 21)))
    assert result["p50"] == 10.5
    assert result["p95"] == 19
    assert summary([7])["p95"] == 7


def test_http_probe_collects_timings_and_reports_errors():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if "failure" in self.path:
                self.send_error(503)
                return
            assert "q=pressure+test" in self.path and "explain=1" in self.path
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"timings":{"vector_retrieval_ms":12}}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        result = request(base, "pressure test", 2)
        assert result["ok"] and result["timings"]["vector_retrieval_ms"] == 12
        assert not request(base, "failure", 2)["ok"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_readonly_sweep_preserves_catalog_and_exports_stages(tmp_path):
    db = tmp_path / "catalog.db"
    fixture(db, 3)
    before = db.read_bytes()
    vectors = tmp_path / "vectors.json"
    vectors.write_text(json.dumps([{"query": "pressure", "vector": [1, 0, 0]}]))
    output = tmp_path / "measurements"
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vector_tuning.py"
    subprocess.run([sys.executable, str(script), "--db", str(db), "--vectors-json", str(vectors),
                    "--output-dir", str(output), "--rounds", "1", "--repeats", "2",
                    "--cache-mib", "8", "16", "--mmap-mib", "0", "--top-k", "2"], check=True)
    assert db.read_bytes() == before
    raw = json.loads((output / "raw.json").read_text())
    assert len(raw["records"]) == 2
    for record in raw["records"]:
        result = record["result"]
        assert result["sqlite_settings"]["cache_size"] == -record["config"]["cache_mib"] * 1024
        assert all(r["equivalent"] for r in result["rows"])
        assert "combined" in result["diagnostics"]["queries"]
    with (output / "stages.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {r["stage"] for r in rows} == {"combined", "split", "knn", "enrichment"}
    assert all(r["same_k_order_and_distances_unchanged"] == "True" for r in rows)
    assert (output / "report.md").is_file()
