use anyhow::{Context, Result, ensure};
use rusqlite::Connection;
use serde::Serialize;

pub const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Serialize)]
pub struct Health {
    pub sqlite: String,
    pub schema: i64,
    pub foreign_keys: bool,
    pub fts5: bool,
}

pub(crate) fn configure(conn: &Connection) -> Result<()> {
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch(
        "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
    )?;
    Ok(())
}

pub(crate) fn initialize(conn: &Connection) -> Result<()> {
    configure(conn)?;
    conn.execute_batch(r#"
      CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
      INSERT INTO schema_version(version) SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_version);
      CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY, source_root TEXT NOT NULL, path TEXT NOT NULL,
        path_key TEXT NOT NULL UNIQUE, extension TEXT, size_bytes INTEGER NOT NULL,
        modified_unix_ms INTEGER, sha256 TEXT NOT NULL, indexed_unix_ms INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_no INTEGER NOT NULL, text TEXT NOT NULL, UNIQUE(document_id, chunk_no));
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
      CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text); END;
      CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text); END;
      CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts,rowid,text) VALUES('delete',old.id,old.text);
        INSERT INTO chunks_fts(rowid,text) VALUES(new.id,new.text); END;
    "#).context("initialize schema")?;
    Ok(())
}

pub(crate) fn health(conn: &Connection) -> Result<Health> {
    let schema = conn.query_row("SELECT version FROM schema_version", [], |r| r.get(0))?;
    ensure!(
        schema == SCHEMA_VERSION,
        "unsupported schema version {schema}"
    );
    let foreign_keys = conn.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))? == 1;
    ensure!(foreign_keys, "foreign keys disabled");
    let fts5 = conn.query_row("SELECT sqlite_compileoption_used('ENABLE_FTS5')", [], |r| {
        r.get::<_, i64>(0)
    })? == 1;
    ensure!(fts5, "FTS5 unavailable");
    Ok(Health {
        sqlite: rusqlite::version().into(),
        schema,
        foreign_keys,
        fts5,
    })
}
