# Direct dependencies

| Crate | Used by | Purpose and architectural impact |
|---|---|---|
| `anyhow` | all | Small application/library error context; no runtime service. |
| `rusqlite` (`bundled`) | core | Typed, parameterized SQLite API and statically bundled SQLite/FTS5; the principal native dependency. |
| `serde` | core, client | Derives JSON response serialization. |
| `serde_json` | client | Emits the small HTTP API payloads. |
| `clap` (`derive`) | daemon, client | Validated native command-line parsing. |
| `sha2` | daemon | Streaming-compatible SHA-256 change identity. |
| `walkdir` | daemon | Robust recursive traversal without a platform runtime. |
| `tracing`, `tracing-subscriber` | daemon/client | Contextual operational logging. |
| `tiny_http` | client | Small synchronous HTTP server; avoids an async runtime. |
| `tempfile` | test only | Isolated filesystem/database tests. |

`cargo tree` and `cargo tree -d` are reviewed in PR checks. There is no async runtime, ORM, frontend toolchain, or dynamic SQLite dependency.
