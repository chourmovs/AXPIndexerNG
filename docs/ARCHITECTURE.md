# Architecture

`AXPIndexerDaemon.exe` recursively scans supported files and is the sole owner of index-content writes. `AXPIndexerClient.exe` independently opens the same database for short searches and database-backed document views; it never crawls or starts the daemon.

`core` defines models, schema/repository operations, path identity, and the shared parameterized FTS search contract. Bundled SQLite uses WAL, foreign keys, a five-second busy timeout, and short write transactions, allowing readers and the writer to coexist. FTS5 external-content triggers synchronize chunk insert, update, and delete (including document cascade deletes). Deletion queries are scoped to the canonical source root.

The client defaults to `127.0.0.1`, refuses non-loopback binds, bounds queries to 100 results, limits URL length, enables no CORS, and exposes no filesystem endpoint.

## Future vector path (not implemented)

`chunks → FastEmbed-rs → embedding → sqlite-vec → hybrid FTS + vector ranking`

Release signing is also not implemented. The intended pipeline is build → tests → smoke → **SIGN** → Authenticode verification → ZIP → hash → release.
