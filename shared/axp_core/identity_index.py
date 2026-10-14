"""Additive document-only candidate index; passage/vector indexes are untouched."""

from __future__ import annotations

import re

from .document_identity import _ordered_terms


def ensure_identity_index(con):
    """Install once under the writer lock, with transactional backfill and triggers."""
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='document_identity_fts'").fetchone():
        return
    con.execute("SAVEPOINT identity_index")
    try:
        con.execute("""CREATE VIRTUAL TABLE document_identity_fts USING fts5(
            filename,title,path,path_key,unicode_compat,content='',contentless_delete=1,
            tokenize='unicode61 remove_diacritics 2')""")
        compatibility = "CASE WHEN (new.filename || new.title || new.path || new.path_key) GLOB '*[^ -~]*' THEN 'unicode' ELSE '' END"
        con.execute(f"""CREATE TRIGGER document_identity_insert AFTER INSERT ON documents BEGIN
            INSERT INTO document_identity_fts(rowid,filename,title,path,path_key,unicode_compat)
            VALUES(new.id,new.filename,new.title,new.path,new.path_key,{compatibility}); END""")
        con.execute("""CREATE TRIGGER document_identity_delete AFTER DELETE ON documents BEGIN
            DELETE FROM document_identity_fts WHERE rowid=old.id; END""")
        con.execute(f"""CREATE TRIGGER document_identity_update AFTER UPDATE OF filename,title,path,path_key ON documents BEGIN
            DELETE FROM document_identity_fts WHERE rowid=old.id;
            INSERT INTO document_identity_fts(rowid,filename,title,path,path_key,unicode_compat)
            VALUES(new.id,new.filename,new.title,new.path,new.path_key,{compatibility}); END""")
        con.execute("""INSERT INTO document_identity_fts(rowid,filename,title,path,path_key,unicode_compat)
            SELECT id,filename,title,path,path_key,
                CASE WHEN (filename || title || path || path_key) GLOB '*[^ -~]*' THEN 'unicode' ELSE '' END
            FROM documents""")
        con.execute("RELEASE identity_index")
    except Exception:
        con.execute("ROLLBACK TO identity_index")
        con.execute("RELEASE identity_index")
        raise


def identity_candidate_ids(con, question, identifiers):
    """Uncapped superset of matches, including plural stems and separator variants.

    Ranking still uses the historical Python matcher. Older read-only catalogs
    retain its exhaustive fallback until their next normal writer connection.
    """
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='document_identity_fts'").fetchone():
        return None
    # FTS token boundaries differ from compound identity boundaries. OR prefixes
    # intentionally overselect; only the authoritative matcher accepts a match.
    terms = {part for term in _ordered_terms(question) for part in re.findall(r"[^\W_]+", term, re.UNICODE)}
    ids = set()
    if terms:
        expression = " OR ".join('"' + term.replace('"', '""') + '"*' for term in sorted(terms))
        ids.update(
            row[0]
            for row in con.execute(
                "SELECT rowid FROM document_identity_fts WHERE document_identity_fts MATCH ?", (expression,)
            )
        )
    # unicode61 removes common accents but does not implement all of NFKD
    # (e.g. compatibility ligatures/fullwidth letters). Keep those documents in
    # the candidate superset so the historical Unicode matcher remains exact.
    ids.update(
        row[0]
        for row in con.execute(
            "SELECT rowid FROM document_identity_fts WHERE document_identity_fts MATCH 'unicode_compat:unicode'"
        )
    )
    if identifiers:
        # Normalized equipment IDs can have different separators in the query
        # and filename. SQLite performs this rare fallback without Python corpus
        # normalization. Path checks retain the legacy literal ID semantics.
        metadata = "lower(filename || ' ' || title)"
        for separator in ("-", ".", "_", "/", " "):
            metadata = f"replace({metadata},'{separator}','')"
        clauses, values = [], []
        for identifier in identifiers:
            clauses.append(f"({metadata} LIKE ? OR lower(path || ' ' || path_key) LIKE ?)")
            values.extend((f"%{identifier.lower()}%", f"%{identifier.lower()}%"))
        ids.update(row[0] for row in con.execute("SELECT id FROM documents WHERE " + " OR ".join(clauses), values))
    return ids
