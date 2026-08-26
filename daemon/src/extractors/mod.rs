mod text;
use anyhow::Result;
use std::path::Path;

pub fn supports(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|x| x.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("txt" | "md" | "markdown")
    )
}
pub fn extract(path: &Path) -> Result<String> {
    text::extract(path)
}
