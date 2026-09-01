"""PR44 adversarial coverage for selected-document passage recovery."""
import sqlite3

import pytest

from axp_client.rag.citations import classify_citations
from axp_client.rag.context import ContextConfig, build_context, select_distinct_seeds
from axp_client.rag.retrieval import rank_documents, retrieve_document_passages
from axp_core.hybrid import diversify


class Embedder:
    def embed_query(self, _query):
        return [1.0, 0.0, 0.0]


def mtbe_database(tmp_path):
    con = sqlite3.connect(tmp_path / "pr44.db")
    con.row_factory = sqlite3.Row
    con.executescript("""CREATE TABLE sources(id INTEGER PRIMARY KEY,label TEXT,path TEXT);
        CREATE TABLE documents(id INTEGER PRIMARY KEY,source_id INTEGER,path TEXT,filename TEXT,title TEXT,
            ingestion_mode TEXT DEFAULT 'content');
        CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,page_no INTEGER,
            section_heading TEXT,text TEXT,identifiers TEXT DEFAULT '');
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text,title,filename,heading,identifiers);""")
    con.execute("insert into sources values(1,'documents','/docs')")
    document = con.execute("insert into documents(source_id,path,filename,title) values(1,?,?,?)",
        ("/docs/mtbe.pdf", "MSDS MTBE SIMFEX.pdf", "MTBE safety data sheet")).lastrowid
    texts = ["Product identification", "Hazards", "Composition", "First aid", "Fire fighting",
             "Handling", "Exposure controls", "Physical properties introduction",
             "Density at 20°C: 0.74 g/cm³", "Stability"]
    chunk_ids = []
    for number, text in enumerate(texts, 1):
        chunk = con.execute("insert into chunks(document_id,chunk_no,page_no,text) values(?,?,?,?)",
                            (document, number, number, text)).lastrowid
        con.execute("insert into chunks_fts(rowid,text,title,filename,heading,identifiers) values(?,?,?,?,?,?)",
                    (chunk, text, "MTBE safety data sheet", "MSDS MTBE SIMFEX.pdf", "", ""))
        chunk_ids.append(chunk)
    con.commit()
    return con, document, chunk_ids


def test_global_cap_hides_density_but_drilldown_recovers_all_chunks(tmp_path):
    con, document, chunks = mtbe_database(tmp_path)
    global_rows = [{"document_id": document, "chunk_id": chunks[index], "chunk_no": index + 1,
                    "passage_score": 1 - index / 100, "filename": "MSDS MTBE SIMFEX.pdf",
                    "title": "MTBE safety data sheet", "evidence_score": .8,
                    "title_coverage": 1.0, "filename_coverage": 1.0}
                   for index in (0, 2, 7, 8)]
    capped = diversify(global_rows, 20, 3)
    assert chunks[8] not in {row["chunk_id"] for row in capped}
    assert rank_documents(capped)[0]["document_id"] == document

    result = retrieve_document_passages(con, Embedder(), "density of MTBE", [document])
    assert result.documents[0]["chunks_examined"] == 10
    assert result.passages[0]["chunk_id"] == chunks[8]
    assert result.passages[0]["content_lexical_coverage"] == pytest.approx(1.0)
    assert result.passages[0]["title_coverage"] == 0.0  # metadata is excluded from passage scoring

    context = build_context(con, result.passages, ContextConfig(max_documents=1,
                            max_seeds_per_document=1, neighbor_radius=1, character_budget=2000))
    assert "Density at 20°C: 0.74 g/cm³" in context.prompt_text
    assert set(context.blocks[0].chunk_nos) == {8, 9, 10}


def test_distinct_seed_policy_after_scoped_scan(tmp_path):
    con, document, _chunks = mtbe_database(tmp_path)
    result = retrieve_document_passages(con, None, "density of MTBE", [document])
    seeds = select_distinct_seeds(result.passages, 3)
    assert all(abs(a["chunk_no"] - b["chunk_no"]) > 1 for i, a in enumerate(seeds) for b in seeds[i + 1:])


def test_citation_validation_reasons_remain_strict():
    assert classify_citations("Density is 0.74 [S1]", ["S1"])[0] == "valid"
    assert classify_citations("Density is 0.74", ["S1"])[0] == "missing_citation"
    assert classify_citations("Density is 0.74 [S9]", ["S1"])[0] == "unknown_citation"
    assert classify_citations("[S1]", ["S1"])[0] == "citation_only_no_prose"
