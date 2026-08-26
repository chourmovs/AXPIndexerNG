use anyhow::{Context, Result};
use std::{fs, path::Path};
pub fn extract(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("read UTF-8 text {}", path.display()))
}
