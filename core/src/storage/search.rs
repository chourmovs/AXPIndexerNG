use anyhow::Result;
use rusqlite::{Connection, params};
use serde::Serialize;

#[derive(Debug)]
pub struct SearchQuery<'a> {
    pub text: &'a str,
    pub limit: usize,
}
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub document_id: i64,
    pub path: String,
    pub chunk_id: i64,
    pub chunk_no: i64,
    pub score: f64,
    pub snippet: String,
}

pub(crate) fn search(conn: &Connection, query: SearchQuery<'_>) -> Result<Vec<SearchResult>> {
    let limit = query.limit.clamp(1, 100) as i64;
    let mut stmt = conn.prepare(
        r#"SELECT d.id,d.path,c.id,c.chunk_no,bm25(chunks_fts),
      snippet(chunks_fts,0,'<mark>','</mark>',' … ',24)
      FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
      JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ?1
      ORDER BY bm25(chunks_fts) LIMIT ?2"#,
    )?;
    let rows = stmt.query_map(params![query.text, limit], |r| {
        Ok(SearchResult {
            document_id: r.get(0)?,
            path: r.get(1)?,
            chunk_id: r.get(2)?,
            chunk_no: r.get(3)?,
            score: r.get(4)?,
            snippet: r.get(5)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}
