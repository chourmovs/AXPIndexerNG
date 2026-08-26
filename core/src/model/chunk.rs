use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Chunk {
    pub id: i64,
    pub document_id: i64,
    pub chunk_no: i64,
    pub text: String,
}
