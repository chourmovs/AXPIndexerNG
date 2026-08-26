use serde::Serialize;

#[derive(Debug, Clone)]
pub struct NewDocument {
    pub source_root: String,
    pub path: String,
    pub path_key: String,
    pub extension: Option<String>,
    pub size_bytes: u64,
    pub modified_unix_ms: Option<i64>,
    pub sha256: String,
    pub indexed_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Document {
    pub id: i64,
    pub source_root: String,
    pub path: String,
    pub path_key: String,
    pub extension: Option<String>,
    pub size_bytes: i64,
    pub modified_unix_ms: Option<i64>,
    pub sha256: String,
    pub indexed_unix_ms: i64,
}
