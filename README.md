# AXPIndexer NG

AXPIndexer NG is a clean-sheet native Rust document index. It deliberately separates two processes: **AXPIndexerDaemon** owns filesystem scanning and index writes; **AXPIndexerClient** reads the shared SQLite database directly and provides CLI search, a localhost HTTP API, and an offline indexed-text viewer. SQLite WAL is their concurrency contract—there is no daemon search RPC.

PR1 supports recursive UTF-8 `.txt`, `.md`, and `.markdown` indexing and FTS5 lexical search. SQLite is bundled; distributed binaries need no Python, Node, Java, or SQLite installation.

```console
AXPIndexerDaemon health --db axpindex.db
AXPIndexerDaemon scan --root C:\Docs --db axpindex.db
AXPIndexerDaemon status --db axpindex.db
AXPIndexerClient search --db axpindex.db --query "reactor pressure" --limit 20
AXPIndexerClient serve --db axpindex.db --host 127.0.0.1 --port 8765
```

The viewer shows only database-backed paths, metadata, and extracted chunks; it cannot read arbitrary files. DOCX, PPTX, PDF, OCR, embeddings, sqlite-vec, hybrid ranking, tray/service support, installers, remote HTTP, and signing are intentionally deferred. PR1 binaries are unsigned.

See [architecture](docs/ARCHITECTURE.md) and [dependency rationale](docs/DEPENDENCIES.md).
