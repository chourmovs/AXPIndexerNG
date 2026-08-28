"""Manual synthetic 100k-document catalog benchmark (not part of normal CI)."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from axp_core.database import connect
from axp_core.fts import search as fts_search
from axp_core.sources import add_source, list_sources, remove_source
from axp_core.vectors import upsert


def timed(function):
    started = time.perf_counter()
    value = function()
    return value, round((time.perf_counter() - started) * 1000, 2)


def run(db, documents=100_000, vector_rows=10_000):
    con = connect(db, dimension=3)
    source_a = add_source(con, r"D:\SyntheticA")
    source_b = add_source(con, r"E:\SyntheticB")
    batch = []
    for index in range(documents):
        source_id = source_a["id"] if index < documents // 2 else source_b["id"]
        batch.append((source_id, f"file-{index}.txt", f"key-{index}", ".txt", index, index, str(index), index,
                      f"Synthetic {index}", f"file-{index}.txt"))
        if len(batch) == 2000:
            con.executemany("""INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,
                            sha256,indexed_unix_ms,title,filename) VALUES(?,?,?,?,?,?,?,?,?,?)""", batch)
            con.commit(); batch.clear()
    if batch:
        con.executemany("""INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,
                        sha256,indexed_unix_ms,title,filename) VALUES(?,?,?,?,?,?,?,?,?,?)""", batch)
        con.commit()
    for index in range(min(vector_rows, documents)):
        chunk_id = con.execute("INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,0,?)",
                               (index + 1, f"reactor synthetic content {index}")).lastrowid
        con.execute("INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
                    (chunk_id, f"reactor synthetic content {index}", "Synthetic", f"file-{index}.txt", "", ""))
        upsert(con, chunk_id, [1.0, float(index % 2), 0.0])
        if index and index % 1000 == 0:
            con.commit()
    con.commit()
    _, list_ms = timed(lambda: list_sources(con))
    _, lookup_ms = timed(lambda: con.execute("SELECT * FROM documents WHERE path_key=?", ("key-99999",)).fetchone())
    fts_rows, fts_ms = timed(lambda: fts_search(con, "reactor", 20))
    _, delete_ms = timed(lambda: remove_source(con, source_a["id"]))
    remaining = con.execute("SELECT count(*) FROM documents WHERE source_id=?", (source_b["id"],)).fetchone()[0]
    return {"documents": documents, "vectors": min(vector_rows, documents), "source_list_ms": list_ms,
            "document_lookup_ms": lookup_ms, "fts_ms": fts_ms, "fts_results": len(fts_rows),
            "scoped_source_delete_ms": delete_ms, "remaining_source_b_documents": remaining}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db")
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--vectors", type=int, default=10_000)
    args = parser.parse_args()
    if args.db:
        print(json.dumps(run(args.db, args.documents, args.vectors), indent=2))
    else:
        with tempfile.TemporaryDirectory() as directory:
            print(json.dumps(run(Path(directory) / "scale.db", args.documents, args.vectors), indent=2))


if __name__ == "__main__":
    main()
