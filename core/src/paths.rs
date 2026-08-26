use std::path::Path;

/// Stable lookup form. Windows paths are case-insensitive; UNC prefixes are retained.
pub fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().replace('/', "\\");
    if cfg!(windows) {
        value.to_lowercase()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separators_are_stable() {
        assert_eq!(path_key(Path::new("a/b")), "a\\b");
    }
}
