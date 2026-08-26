use crate::extractors;
use anyhow::{Context, Result};
use axpindexer_core::{Database, model::NewDocument, paths::path_key};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tracing::warn;
use walkdir::WalkDir;

#[derive(Debug, Default)]
pub struct ScanSummary {
    pub supported: u64,
    pub new: u64,
    pub updated: u64,
    pub unchanged: u64,
    pub deleted: u64,
    pub failed: u64,
    pub duration_ms: u128,
}
pub fn scan(root: &Path, db_path: &Path) -> Result<ScanSummary> {
    let started = Instant::now();
    let canonical = root
        .canonicalize()
        .with_context(|| format!("resolve root {}", root.display()))?;
    let root_s = canonical.to_string_lossy().into_owned();
    let mut db = Database::open(db_path)?;
    let mut seen = HashSet::new();
    let mut out = ScanSummary::default();
    for entry in WalkDir::new(&canonical).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(error=%e,"walk error");
                out.failed += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() || !extractors::supports(entry.path()) {
            continue;
        }
        out.supported += 1;
        let path = entry.path();
        let key = path_key(path);
        seen.insert(key.clone());
        let result = (|| -> Result<()> {
            let meta = fs::metadata(path)?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64);
            let old = db.find_by_key(&key)?;
            if old
                .as_ref()
                .is_some_and(|d| d.size_bytes == meta.len() as i64 && d.modified_unix_ms == mtime)
            {
                out.unchanged += 1;
                return Ok(());
            }
            let bytes = fs::read(path)?;
            let hash = format!("{:x}", Sha256::digest(&bytes));
            let changed = old.as_ref().is_none_or(|d| d.sha256 != hash);
            let chunks = if changed {
                Some(vec![extractors::extract(path)?])
            } else {
                None
            };
            let d = NewDocument {
                source_root: root_s.clone(),
                path: path.to_string_lossy().into_owned(),
                path_key: key,
                extension: path
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(str::to_ascii_lowercase),
                size_bytes: meta.len(),
                modified_unix_ms: mtime,
                sha256: hash,
                indexed_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64,
            };
            db.upsert(&d, chunks.as_deref())?;
            if old.is_some() {
                out.updated += 1
            } else {
                out.new += 1
            }
            Ok(())
        })();
        if let Err(e) = result {
            warn!(path=%path.display(),error=%e,"file skipped");
            out.failed += 1
        }
    }
    for d in db.documents_for_root(&root_s)? {
        if !seen.contains(&d.path_key) {
            db.delete(d.id)?;
            out.deleted += 1
        }
    }
    out.duration_ms = started.elapsed().as_millis();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    #[test]
    fn changes_and_root_safety() {
        let t = tempdir().unwrap();
        let a = t.path().join("a");
        let b = t.path().join("b");
        fs::create_dir_all(a.join("sub")).unwrap();
        fs::create_dir(&b).unwrap();
        fs::write(a.join("x.txt"), "one").unwrap();
        fs::write(a.join("sub/y.md"), "two").unwrap();
        fs::write(b.join("b.txt"), "three").unwrap();
        let db = t.path().join("db");
        assert_eq!(scan(&a, &db).unwrap().new, 2);
        assert_eq!(scan(&a, &db).unwrap().unchanged, 2);
        scan(&b, &db).unwrap();
        fs::remove_file(a.join("sub/y.md")).unwrap();
        assert_eq!(scan(&a, &db).unwrap().deleted, 1);
        assert_eq!(Database::open(db).unwrap().count().unwrap(), 2)
    }
}
