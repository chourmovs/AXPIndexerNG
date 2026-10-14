"""Synthetic October recovery benchmark; never use this script on a user catalog.

Cold means a newly opened connection, not a flushed OS page cache. Embeddings
are deterministic fixtures; no model or generation performance is measured.
"""

import sys, json, time, argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for folder in ("shared", "daemon", "client", "tray"):
    sys.path.insert(0, str(root / folder))
import numpy as np  # noqa: E402
from axp_core.database import connect  # noqa: E402
from axp_core.metadata import ensure_index_signature  # noqa: E402
from axp_client.search import search, raw_search  # noqa: E402
from axp_client.rag.coordinator import RetrievalCoordinator  # noqa: E402
from axp_client.rag.spiral import resolve_identity_documents  # noqa: E402


class Embedder:
    model_id = "synthetic"
    dimension = 384
    distance_metric = "cosine"

    def embed_query(self, text):
        v = np.zeros(self.dimension, dtype=np.float32)
        v[0] = float("pressure" in text.lower())
        v[1] = float("storage" in text.lower())
        v[2] = float("DOMINO" in text)
        if not v.any():
            v[3] = 1
        return (v / np.linalg.norm(v)).tolist()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--db", type=Path, required=True, help="Dedicated synthetic catalog, not a user database")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
e = Embedder()
db = args.db
if db.exists():
    from axp_core.metadata import validate_index_signature

    existing = connect(db, readonly=True)
    validate_index_signature(existing, e.model_id, e.dimension, e.distance_metric)
    if existing.execute("SELECT count(*) FROM chunks").fetchone()[0] != 467255:
        raise ValueError("Not the expected synthetic fixture")
    existing.close()
else:
    db.parent.mkdir(parents=True, exist_ok=True)
if not db.exists():
    t = time.perf_counter()
    c = connect(db, dimension=e.dimension)
    c.execute(
        "INSERT INTO sources(id,path,path_key,label,kind,created_ms,updated_ms) VALUES(1,'/fixture','/fixture','Fixture','folder',0,0)"
    )
    ensure_index_signature(c, e.model_id, e.dimension, e.distance_metric)
    c.executemany(
        "INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(?,1,?,? ,'.txt',0,0,'',0,?,?)",
        [
            (
                i,
                f"/fixture/record-{i}.txt",
                f"/fixture/record-{i}.txt",
                "DOMINO" if i == 1 else f"Archive {i}",
                "DOMINO.txt" if i == 1 else f"record-{i}.txt",
            )
            for i in range(1, 23364)
        ],
    )
    blobs = [
        np.asarray(e.embed_query(x), dtype=np.float32).tobytes()
        for x in ("DOMINO pressure", "DOMINO storage", "irrelevant")
    ]
    for start in range(1, 467256, 5000):
        chunks = []
        vectors = []
        for i in range(start, min(start + 5000, 467256)):
            doc = (i - 1) // 20 + 1
            no = (i - 1) % 20
            text = (
                (
                    "Operating pressure is 12 bar in reactor DOMINO."
                    if no % 2 == 0
                    else "DOMINO storage temperature is 8 C."
                )
                if doc == 1
                else f"Archival unrelated notes entry {i}."
            )
            chunks.append((i, doc, no, text))
            vectors.append((i, blobs[no % 2 if doc == 1 else 2]))
        c.executemany("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(?,?,?,?)", chunks)
        c.executemany("INSERT INTO chunk_vectors(rowid,embedding) VALUES(?,?)", vectors)
        c.commit()
    print("build_s", time.perf_counter() - t, flush=True)
    c.close()
queries = [
    ("name", "DOMINO"),
    ("hybrid", "reactor pressure"),
    ("identified", "DOMINO pressure"),
    ("multiple", "DOMINO storage pressure"),
    ("absent", "unobtainium calibration"),
]
rows = []
for scenario, q in queries:
    for rep in range(args.repeats):
        c = connect(db, readonly=True)
        for temp in ("cold_connection", "warm"):
            tick = time.perf_counter()
            ids = resolve_identity_documents(c, q)
            identity = (time.perf_counter() - tick) * 1000
            r = search(c, e, q, explain=True)
            rag = RetrievalCoordinator(search_fn=raw_search).retrieve(c, e, q)
            rows.append(
                dict(
                    query=q,
                    scenario=scenario,
                    temperature=temp,
                    rep=rep,
                    identity_ms=identity,
                    search=r["timings"],
                    top=r["results"][0]["document_id"] if r["results"] else None,
                    rag=rag.retrieval.timings,
                    answerable=rag.decision.answerable,
                    passages=len(rag.retrieval.content_evidence),
                )
            )
        c.close()
    print(scenario, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(dict(documents=23363, chunks=467255, dimension=384, repetitions=args.repeats, rows=rows), indent=2)
)
