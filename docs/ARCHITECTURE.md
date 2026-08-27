# Architecture

Files flow through small format-specific text extractors, deterministic word-window chunking (400 words with 50-word overlap), SQLite FTS5, and FastEmbed into sqlite-vec. Chunk offsets and PDF pages/PPTX slides preserve provenance. SQLite foreign keys and triggers cascade deletions through chunks, FTS, and vector rows.

The daemon is the sole writer. It skips hashing when size and mtime match, hashes metadata changes, skips extraction when content hashes match, and scopes deletion reconciliation to the canonical source root. The client is read-oriented and resolves viewer requests only by database document ID.

Hybrid search applies Reciprocal Rank Fusion (`1/(60 + rank)`) to lexical BM25 and vector KNN result lists. Both source ranks remain visible. Schema and embedding identity/dimension metadata reject incompatible reuse.
