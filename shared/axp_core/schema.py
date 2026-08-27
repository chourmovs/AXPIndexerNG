SCHEMA_VERSION = 1
BASE_SCHEMA = '''
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 id INTEGER PRIMARY KEY, source_root TEXT NOT NULL, path TEXT NOT NULL, path_key TEXT NOT NULL UNIQUE,
 extension TEXT NOT NULL, size_bytes INTEGER NOT NULL, modified_unix_ms INTEGER NOT NULL,
 sha256 TEXT NOT NULL, indexed_unix_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS documents_root ON documents(source_root);
CREATE TABLE IF NOT EXISTS chunks(
 id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 chunk_no INTEGER NOT NULL, text TEXT NOT NULL, page_no INTEGER, char_start INTEGER, char_end INTEGER,
 UNIQUE(document_id, chunk_no));
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text);
 INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text); END;
'''
