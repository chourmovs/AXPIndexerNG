use axpindexer_core::{Database, SearchQuery};
use serde::Serialize;
#[derive(Serialize)]
struct Detail<T, U> {
    document: T,
    chunks: U,
}
pub fn route(db: &Database, url: &str) -> (u16, &'static str, String) {
    if url.len() > 8192 {
        return (414, "text/plain", "URL too long".into());
    }
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    match path {
        "/" => (
            200,
            "text/html; charset=utf-8",
            include_str!("../web/index.html").into(),
        ),
        "/app.js" => (
            200,
            "text/javascript; charset=utf-8",
            include_str!("../web/app.js").into(),
        ),
        "/style.css" => (
            200,
            "text/css; charset=utf-8",
            include_str!("../web/style.css").into(),
        ),
        "/health" => match db.health() {
            Ok(h) => (200, "application/json", serde_json::to_string(&h).unwrap()),
            Err(e) => (503, "text/plain", e.to_string()),
        },
        "/api/search" => {
            let p: std::collections::HashMap<_, _> =
                query.split('&').filter_map(|x| x.split_once('=')).collect();
            let q = decode(p.get("q").copied().unwrap_or(""));
            let limit = p
                .get("limit")
                .and_then(|x| x.parse().ok())
                .unwrap_or(20usize)
                .clamp(1, 100);
            match db.search(SearchQuery { text: &q, limit }) {
                Ok(v) => (200, "application/json", serde_json::to_string(&v).unwrap()),
                Err(e) => (400, "text/plain", e.to_string()),
            }
        }
        _ if path.starts_with("/api/document/") => {
            let id = path.trim_start_matches("/api/document/").parse::<i64>();
            match id.ok().and_then(|id| db.document(id).ok().flatten()) {
                Some((document, chunks)) => (
                    200,
                    "application/json",
                    serde_json::to_string(&Detail { document, chunks }).unwrap(),
                ),
                None => (404, "text/plain", "document not found".into()),
            }
        }
        _ => (404, "text/plain", "not found".into()),
    }
}
fn decode(s: &str) -> String {
    let mut out = String::new();
    let mut b = s.bytes();
    while let Some(c) = b.next() {
        match c {
            b'+' => out.push(' '),
            b'%' => {
                if let (Some(a), Some(z)) = (b.next(), b.next()) {
                    if let Ok(v) = u8::from_str_radix(&format!("{}{}", a as char, z as char), 16) {
                        out.push(v as char)
                    }
                }
            }
            _ => out.push(c as char),
        }
    }
    out
}
