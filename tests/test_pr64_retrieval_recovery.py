import sqlite3
from axp_client.rag.spiral import RetrievalScope, resolve_identity_documents


def test_scoped_identity_sql_qualifies_joined_columns():
    con = sqlite3.connect(":memory:"); con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE sources(id INTEGER PRIMARY KEY, label TEXT, path TEXT, path_key TEXT);
      CREATE TABLE documents(id INTEGER PRIMARY KEY, source_id INTEGER, title TEXT, filename TEXT,
        path TEXT, path_key TEXT, extension TEXT, modified_unix_ms INTEGER);
      INSERT INTO sources VALUES(1,'SEQUENCES','K:\\R&D\\R&D\\SEQUENCES','k:\\r&d\\r&d\\sequences');
      INSERT INTO documents VALUES(1,1,'DOMINO','DOMINO.docx','K:\\R&D\\R&D\\SEQUENCES\\DOMINO.docx',
        'k:\\r&d\\r&d\\sequences\\domino.docx','.docx',0);
    """)
    ids = resolve_identity_documents(con, "montre-moi la séquence DOMINO", scope=RetrievalScope(
        source_ids=(1,), path_prefixes=(r"K:\R&D\R&D\SEQUENCES",), extensions=(".docx",)), limit=3)
    assert ids == [1]
