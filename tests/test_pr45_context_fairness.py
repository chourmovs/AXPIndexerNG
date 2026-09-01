"""PR45 regressions for hierarchical drill-down and fair context packing."""
import sqlite3

from axp_client.rag.context import ContextConfig, build_context
from axp_client.rag.retrieval import build_scoped_passage_query, retrieve_document_passages


def database():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE sources(id INTEGER PRIMARY KEY,label TEXT,path TEXT);
      CREATE TABLE documents(id INTEGER PRIMARY KEY,source_id INTEGER,path TEXT,filename TEXT,title TEXT,
        ingestion_mode TEXT DEFAULT 'content');
      CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,page_no INTEGER,
        section_heading TEXT,text TEXT,identifiers TEXT DEFAULT '');
      CREATE VIRTUAL TABLE chunks_fts USING fts5(text,title,filename,heading,identifiers);
      INSERT INTO sources VALUES(1,'docs','/docs');
      INSERT INTO documents VALUES(1,1,'a.pdf','HEPTANE 99% FDS.pdf','n-Heptane 99%','content');
      INSERT INTO documents VALUES(2,1,'b.pdf','Second.pdf','Second','content');
      INSERT INTO documents VALUES(3,1,'c.pdf','Third.pdf','Third','content');
    """)
    return con


def add(con, chunk_id, document_id, chunk_no, text):
    con.execute("INSERT INTO chunks(id,document_id,chunk_no,page_no,section_heading,text) VALUES(?,?,?,?,?,?)",
                (chunk_id, document_id, chunk_no, chunk_no, "", text))
    doc = con.execute("SELECT title,filename FROM documents WHERE id=?", (document_id,)).fetchone()
    con.execute("INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
                (chunk_id, text, doc["title"], doc["filename"], "", ""))


def hit(document, chunk, number, score, document_rank, passage_rank=1):
    return {"document_id": document, "chunk_id": chunk, "chunk_no": number,
            "passage_score": score, "relevance_score": score, "document_rank": document_rank,
            "passage_rank": passage_rank}


def test_two_documents_survive_and_oversized_primary_is_shrunk():
    con = database()
    add(con, 1, 1, 1, "alpha " * 1000)
    add(con, 2, 2, 1, "small beta evidence")
    result = build_context(con, [hit(1, 1, 1, .9, 1), hit(2, 2, 1, .8, 2)],
        ContextConfig(neighbor_radius=0, max_evidence_tokens=90, answer_reserve_tokens=0,
                      safety_reserve_tokens=0), token_counter=lambda text: len(text.split()),
        context_window=1000)
    assert result.diagnostics["selected_documents"] == 2
    assert [block.document_id for block in result.blocks] == [1, 2]
    assert "small beta evidence" in result.prompt_text


def test_seed_is_preserved_before_neighbor_under_tight_budget():
    con = database()
    add(con, 1, 1, 9, "before " * 100)
    add(con, 2, 1, 10, "FACTUAL DENSITY 0.68")
    add(con, 3, 1, 11, "after " * 100)
    result = build_context(con, [hit(1, 2, 10, .9, 1)], ContextConfig(neighbor_radius=1,
        max_evidence_tokens=45, answer_reserve_tokens=0, safety_reserve_tokens=0),
        token_counter=lambda text: len(text.split()), context_window=1000)
    assert "FACTUAL DENSITY 0.68" in result.prompt_text
    assert result.blocks[0].seed_chunk_no == 10


def test_authoritative_document_and_round_robin_passage_order():
    con = database()
    rows = []
    chunk_id = 1
    for document, document_rank, score in ((1, 1, .75), (2, 2, .85), (3, 3, .95)):
        for passage_rank, number in enumerate((1, 10), 1):
            add(con, chunk_id, document, number, f"D{document} P{passage_rank}")
            rows.append(hit(document, chunk_id, number, score, document_rank, passage_rank))
            chunk_id += 1
    result = build_context(con, list(reversed(rows)), ContextConfig(neighbor_radius=0,
        max_seeds_per_document=2, character_budget=10000))
    assert [(block.document_rank, block.passage_rank) for block in result.blocks] == [
        (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]


def test_factual_scoped_query_removes_identity_and_question_stopwords():
    assert build_scoped_passage_query("What is the density of n-Heptane 99%?",
                                      "HEPTANE 99% FDS") == "density"
    assert build_scoped_passage_query("Quelle est la densité de l'ammoniaque 20.5% ?",
                                      "AMMONIAQUE 20.5%") == "densité"


def test_density_is_only_scoped_lexical_match_and_ranks_first():
    con = database()
    for number in range(1, 101):
        text = "General safety information about n-Heptane" if number != 73 else "Density at 20 C: 0.68 g/cm3"
        add(con, number, 1, number, text)
    result = retrieve_document_passages(con, None, "What is the density of n-Heptane 99%?", [1])
    assert result.documents[0]["matching_chunks"] == 1
    assert result.passages[0]["chunk_no"] == 73
    assert result.passages[0]["document_rank"] == result.passages[-1]["document_rank"] == 1
