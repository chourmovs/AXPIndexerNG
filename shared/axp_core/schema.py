SCHEMA_VERSION = 4
PREVIOUS_SCHEMA_VERSION = 3
CHUNKER_VERSION = 2
EMBEDDING_INPUT_VERSION = 2
DISTANCE_METRIC = "cosine"

SOURCES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
 id INTEGER PRIMARY KEY,
 path TEXT NOT NULL,
 path_key TEXT NOT NULL UNIQUE,
 label TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('folder','drive','unc')),
 enabled INTEGER NOT NULL DEFAULT 1,
 recursive INTEGER NOT NULL DEFAULT 1,
 status TEXT NOT NULL DEFAULT 'idle'
   CHECK(status IN ('idle','scanning','paused','offline','error','disabled')),
 last_scan_started_ms INTEGER,
 last_scan_completed_ms INTEGER,
 last_success_ms INTEGER,
 last_error TEXT,
 last_file_count INTEGER NOT NULL DEFAULT 0,
 last_chunk_count INTEGER NOT NULL DEFAULT 0,
 last_seen_count INTEGER NOT NULL DEFAULT 0,
 last_content_count INTEGER NOT NULL DEFAULT 0,
 last_metadata_count INTEGER NOT NULL DEFAULT 0,
 last_ignored_count INTEGER NOT NULL DEFAULT 0,
 last_failed_count INTEGER NOT NULL DEFAULT 0,
 last_extension_breakdown TEXT NOT NULL DEFAULT '{}',
 created_ms INTEGER NOT NULL,
 updated_ms INTEGER NOT NULL
);
"""

BASE_SCHEMA = SOURCES_SCHEMA + """
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 id INTEGER PRIMARY KEY,
 source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
 path TEXT NOT NULL, path_key TEXT NOT NULL UNIQUE,
 extension TEXT NOT NULL, size_bytes INTEGER NOT NULL, modified_unix_ms INTEGER NOT NULL,
 sha256 TEXT NOT NULL, indexed_unix_ms INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '', filename TEXT NOT NULL DEFAULT '',
 ingestion_mode TEXT NOT NULL DEFAULT 'content' CHECK(ingestion_mode IN ('content','metadata')));
CREATE INDEX IF NOT EXISTS documents_source ON documents(source_id);
CREATE TABLE IF NOT EXISTS chunks(
 id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 chunk_no INTEGER NOT NULL, text TEXT NOT NULL, page_no INTEGER, char_start INTEGER, char_end INTEGER,
 section_heading TEXT NOT NULL DEFAULT '', identifiers TEXT NOT NULL DEFAULT '',
 UNIQUE(document_id, chunk_no));
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
 text, title, filename, heading, identifiers,
 content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers)
 VALUES(new.id,new.text,'','',new.section_heading,new.identifiers); END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
 DELETE FROM chunks_fts WHERE rowid=old.id; END;
"""
