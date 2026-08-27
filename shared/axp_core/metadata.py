def ensure_model(con, model_id, dimension, revision="unknown"):
    wanted = {"model_id": model_id, "embedding_dimension": str(dimension), "model_revision": revision}
    existing = dict(con.execute("SELECT key,value FROM metadata").fetchall())
    conflicts = {k for k, v in wanted.items() if k in existing and existing[k] != v}
    if conflicts:
        raise RuntimeError("Embedding configuration changed; reindex required: " + ", ".join(sorted(conflicts)))
    con.executemany("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", wanted.items())
    con.commit()
