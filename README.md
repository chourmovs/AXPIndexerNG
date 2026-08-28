# AXPIndexerNG

AXPIndexerNG is a lightweight Python 3.11 desktop document indexer. It uses one portable SQLite catalog with
FTS5, sqlite-vec and FastEmbed; it does not use LanceDB, a custom executable or a system Python installation.
Supported inputs are TXT, Markdown, PDF, DOCX and PPTX.

## Desktop usage

1. Extract the portable ZIP to a normal writable directory.
2. Double-click `AXPIndexerTray.pyw` (or `AXPIndexerTray.vbs` where Windows Script Host is available).
3. On first launch, the tray-started daemon validates `model-cache` and downloads the selected embedding model
   when it is missing or incomplete. The heartbeat remains alive and reports `downloading_model` or
   `waiting_for_model`; transient network failures are retried every 60 seconds instead of terminating the daemon.
4. Right-click the tray icon, choose **Sources...**, then **Explorer...**.
5. Select a folder, complete drive, or add a UNC path such as `\\SERVER\Documentation`.
6. Indexing starts in the background. The tray exposes progress, pause/resume, scan and restart controls.
7. Choose **Search...** to start/reuse the separate localhost client and open the browser UI.

One database contains multiple indexed locations. A source is an administrative scan root, **not a database**;
search remains global across all indexed sources. Disabling a source stops future scans but deliberately leaves
its existing content searchable. Removing it deletes only that source's indexed documents/chunks/vectors and
never touches files on disk.

An unavailable USB disk, mapped drive, VPN location or UNC share is marked `offline`. Existing results are kept:
**offline does not mean deleted**. Deletion reconciliation runs only after complete successful enumeration.
Temporary Office lock files such as `~$document.docx` are ignored. Extraction, embedding and database failures
are isolated per document; an embedding batch is recursively split so one malformed document cannot stop the
remaining source scan.

Normal desktop use writes the database, settings, heartbeat and rotating UTF-8 logs under `data/`. Set
`AXPINDEXER_DATA_DIR` to override that directory. **Start with Windows** registers only the current user under
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`; no elevation is required.

## CLI and diagnostics

Portable BAT launchers configure package paths automatically. From a development checkout, set
`PYTHONPATH=shared;daemon;client;tray` on Windows (`:` on Unix).

```text
python -m axp_daemon health --db data\axpindex.db
python -m axp_daemon source add --db data\axpindex.db --path D:\Process
python -m axp_daemon source list --db data\axpindex.db
python -m axp_daemon scan-source --db data\axpindex.db --id 1
python -m axp_daemon run --db data\axpindex.db --model-cache model-cache --allow-download --scan-interval 300
python -m axp_daemon status --db data\axpindex.db
python -m axp_client search --db data\axpindex.db --query "reactor pressure"
python -m axp_tray self-test --db data\axpindex.db
```

The daemon also supports `source enable`, `source disable`, `source remove` and one-shot `scan`/`reindex`.
The default persistent scan interval is 300 seconds, measured from the end of the previous cycle. `Scan now`
wakes the existing daemon; it never starts a concurrent scan.

## Models and retrieval

The runtime ZIP still excludes the model cache. Desktop settings enable explicit first-launch provisioning by
default (`download_missing_models: true`); set it to `false` for a strictly offline installation and provision
`model-cache` manually. `FASTEMBED_CACHE_PATH`, `AXPINDEXER_DATA_DIR` and `PYTHONPATH` are propagated to every
spawned process. Balanced indexing uses multilingual MiniLM (384 dimensions); quality indexing uses
multilingual E5-large (1024 dimensions). The database index signature locks the exact model, dimension and
cosine metric.

The client `fast`, `hybrid` and `quality` profiles are retrieval strategies independent of the daemon's dense
embedding profile. Retrieval Engine V2 preserves FTS5, cosine KNN, hybrid RRF, exact identifier/filename
safeguards, document diversification, explain mode and optional ColBERT reranking. Before results reach the
client, weak candidates are removed unless they meet at least one absolute relevance condition: exact match,
cosine similarity of 0.35, or 50% lexical coverage of the meaningful query terms. Explain mode reports the
thresholds and filtered candidate count.

Install runtime dependencies with `pip install -r requirements-runtime.txt`; development checks additionally
use `requirements-dev.txt`.
