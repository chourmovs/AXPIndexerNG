# Architecture

AXPIndexerNG keeps daemon, client and tray as separate applications sharing an explicit SQLite path.

```text
Catalog
  +-- Source A: D:\Process
  +-- Source B: D:\Quality
  +-- Source C: \\SERVER\Documentation
  |
  v
axpindex.db
  +-- metadata and index signature
  +-- sources
  +-- documents -> source_id
  +-- chunks
  +-- FTS5
  +-- sqlite-vec
```

A source is an administrative unit of indexing. A catalog/database is a unit of search. A source is **not** a
database. All enabled sources are scanned independently into one catalog and global search reads that catalog.
Database paths remain explicit in public APIs so future multiple catalogs are possible without implementing
federation now.

## Desktop process topology

```text
Tray
  +-- manages sources through shared Python APIs
  +-- writes atomic local control requests
  +-- opens/starts the separate client
  |
  v
Persistent daemon
  +-- scans enabled sources sequentially
  +-- extracts and chunks supported documents
  +-- embeds bounded cross-document batches
  +-- commits bounded document batches to SQLite

Client
  +-- opens the same catalog read-only
  +-- serves localhost search UI
```

The portable bootstrap explicitly adds `shared`, `daemon`, `client` and `tray` to `PYTHONPATH`. Background
processes use bundled `pythonw.exe`, carry explicit environment variables and retain the daemon/client boundary.

## Schema and migration

Schema **3**, chunker **2** and embedding-input **2** form the current contract. Opening a schema-2 alpha3
catalog performs one forward migration: distinct `documents.source_root` values become `sources` rows and
`documents.source_id` is backfilled. Documents, chunks, contentless FTS rows, vectors, metadata and index
signature are preserved; embedding/chunking semantics do not change. Unknown schema versions still fail.

Deleting a `sources` row cascades through only its documents, chunks and vector/FTS trigger cleanup. Disabling a
source retains searchable content. Parent/child overlap checks use normalized path components, so `D:\Process`
covers `D:\Process\Batch` but not `D:\ProcessArchive`.

## Source states and deletion safety

| State | Meaning |
|---|---|
| `idle` | Last complete scan succeeded or source awaits work. |
| `scanning` | The daemon is enumerating/indexing this source. |
| `paused` | Work is paused at a safe file/batch boundary; heartbeat continues. |
| `offline` | Root cannot be reached; indexed content is preserved. |
| `error` | Traversal/indexing was incomplete; indexed content is preserved. |
| `disabled` | Future scans are skipped; existing content remains searchable. |

Reconciliation of missing files occurs only when root enumeration completes. Root failure, a protected subtree,
interruption or partial traversal suppresses all destructive missing-file reconciliation for that source. One bad
file and one bad source do not abort other sources.

## Runtime control and resilience

`data/runtime/daemon_state.json` is atomically replaced about every two seconds with PID, explicit state, current
source/file, counters, totals, error and heartbeat timestamp. `control.json` carries atomic local commands; no
network control API or messaging dependency is used. OS-backed locks enforce one tray per data directory and one
daemon per catalog/data directory even when stale lock files survive crashes.

The tray treats heartbeats older than 90 seconds as stale, deliberately tolerating thread-scheduling delays on
slow or battery-throttled systems while the normal publication interval remains about two seconds. A stale
heartbeat is not proof that the daemon is dead: the catalog-specific OS instance lock is authoritative. The tray
probes that lock before every candidate restart and suppresses duplicate launches while it remains owned.
Auto-restart is enabled by default, limited to once per 60 seconds and gated by persisted desired state, so an
intentional stop is not immediately undone. Manual restart also waits for the old lock owner to exit rather than
blindly spawning a replacement. Exiting the tray exits only the UI and preserves daemon operation.

## Retrieval Engine V2

Accent-insensitive FTS5 and exact sqlite-vec cosine KNN retrieve broad candidate sets. Reciprocal rank fusion
combines ranks, exact identifier/filename/quoted phrase evidence supplies a deterministic safeguard, and optional
multilingual ColBERT reranks a bounded head. Results are diversified by document. Source ID, label and path are
payload metadata only and do not affect relevance.
