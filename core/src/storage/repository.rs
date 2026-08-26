use super::{schema, search};
use crate::model::{Chunk, Document, NewDocument};
use crate::storage::{Health, SearchQuery, SearchResult};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

pub struct Database {
    conn: Connection,
}
impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        schema::initialize(&conn)?;
        Ok(Self { conn })
    }
    pub fn health(&self) -> Result<Health> {
        schema::health(&self.conn)
    }
    pub fn search(&self, q: SearchQuery<'_>) -> Result<Vec<SearchResult>> {
        search::search(&self.conn, q)
    }
    pub fn find_by_key(&self, key: &str) -> Result<Option<Document>> {
        self.conn.query_row("SELECT id,source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms FROM documents WHERE path_key=?",[key],row_doc).optional().map_err(Into::into)
    }
    pub fn document(&self, id: i64) -> Result<Option<(Document, Vec<Chunk>)>> {
        let Some(d)=self.conn.query_row("SELECT id,source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms FROM documents WHERE id=?",[id],row_doc).optional()? else{return Ok(None)};
        let mut s = self.conn.prepare(
            "SELECT id,document_id,chunk_no,text FROM chunks WHERE document_id=? ORDER BY chunk_no",
        )?;
        let cs = s
            .query_map([id], |r| {
                Ok(Chunk {
                    id: r.get(0)?,
                    document_id: r.get(1)?,
                    chunk_no: r.get(2)?,
                    text: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Some((d, cs)))
    }
    pub fn upsert(&mut self, d: &NewDocument, chunks: Option<&[String]>) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(r#"INSERT INTO documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(path_key) DO UPDATE SET source_root=excluded.source_root,path=excluded.path,extension=excluded.extension,size_bytes=excluded.size_bytes,modified_unix_ms=excluded.modified_unix_ms,sha256=excluded.sha256,indexed_unix_ms=excluded.indexed_unix_ms"#,params![d.source_root,d.path,d.path_key,d.extension,d.size_bytes,d.modified_unix_ms,d.sha256,d.indexed_unix_ms])?;
        let id = tx.query_row(
            "SELECT id FROM documents WHERE path_key=?",
            [&d.path_key],
            |r| r.get(0),
        )?;
        if let Some(chunks) = chunks {
            tx.execute("DELETE FROM chunks WHERE document_id=?", [id])?;
            for (n, text) in chunks.iter().enumerate() {
                tx.execute(
                    "INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,?,?)",
                    params![id, n as i64, text],
                )?;
            }
        }
        tx.commit()?;
        Ok(id)
    }
    pub fn delete(&self, id: i64) -> Result<bool> {
        Ok(self
            .conn
            .execute("DELETE FROM documents WHERE id=?", [id])?
            > 0)
    }
    pub fn documents_for_root(&self, root: &str) -> Result<Vec<Document>> {
        let mut s=self.conn.prepare("SELECT id,source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms FROM documents WHERE source_root=?")?;
        Ok(s.query_map([root], row_doc)?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM documents", [], |r| r.get(0))?)
    }
}
fn row_doc(r: &rusqlite::Row<'_>) -> rusqlite::Result<Document> {
    Ok(Document {
        id: r.get(0)?,
        source_root: r.get(1)?,
        path: r.get(2)?,
        path_key: r.get(3)?,
        extension: r.get(4)?,
        size_bytes: r.get(5)?,
        modified_unix_ms: r.get(6)?,
        sha256: r.get(7)?,
        indexed_unix_ms: r.get(8)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    fn d(key: &str) -> NewDocument {
        NewDocument {
            source_root: "r".into(),
            path: key.into(),
            path_key: key.into(),
            extension: Some("txt".into()),
            size_bytes: 1,
            modified_unix_ms: Some(1),
            sha256: "x".into(),
            indexed_unix_ms: 1,
        }
    }
    #[test]
    fn lifecycle_and_concurrency() {
        let t = tempdir().unwrap();
        let p = t.path().join("x.db");
        let mut writer = Database::open(&p).unwrap();
        let reader = Database::open(&p).unwrap();
        writer
            .upsert(&d("a"), Some(&["reactor pressure".into()]))
            .unwrap();
        assert_eq!(
            reader
                .search(SearchQuery {
                    text: "reactor",
                    limit: 10
                })
                .unwrap()
                .len(),
            1
        );
        let id = writer.find_by_key("a").unwrap().unwrap().id;
        writer.upsert(&d("a"), Some(&["other".into()])).unwrap();
        assert!(
            reader
                .search(SearchQuery {
                    text: "reactor",
                    limit: 10
                })
                .unwrap()
                .is_empty()
        );
        writer.delete(id).unwrap();
        assert_eq!(writer.count().unwrap(), 0);
        Database::open(&p).unwrap().health().unwrap();
    }
}
