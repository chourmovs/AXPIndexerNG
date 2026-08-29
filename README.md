# AXPIndexerNG

AXPIndexerNG is a lightweight Python 3.11 desktop document indexer. It uses one portable SQLite catalog with
FTS5, sqlite-vec and FastEmbed; it does not use LanceDB, a custom executable or a system Python installation.
## Document ingestion and coverage

**Full-content indexing:** TXT, Markdown, PDF, DOCX, PPTX, XLSX and CSV. XLSX worksheets and CSV rows retain
their tabular structure in searchable text.

**Metadata-only indexing:** every other useful regular file format is represented by one compact searchable
chunk containing its filename, extension, folder/path and basic filesystem metadata. The binary contents are not
read or parsed. Office/LibreOffice lock files and conservative transient suffixes (`.tmp`, `.part`,
`.crdownload`) are ignored.

Coverage uses mutually exclusive outcomes for every file seen: **content** (contents extracted), **metadata only**
(searchable file metadata), **ignored** (explicit temporary policy), or **failed** (eligible but could not be
indexed). **Absorption** is `(content + metadata) / seen`; **content coverage** is `content / seen`. Thus absorption
describes files represented in the index, while content coverage describes files whose actual contents were parsed.

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

### Live indexing dashboard

The Sources window separates **exact completed-scan coverage** from **estimated live progress**. A value such as
`~68%` is an estimate using the last successfully completed source size (or an existing-document baseline for a
migrated source). A first-ever scan without a meaningful baseline stays indeterminate: AXPIndexerNG intentionally
does not traverse a filesystem twice merely to manufacture a percentage. Completed coverage remains authoritative
and is retained when a later scan is interrupted, offline, or fails.

The dashboard reports the current source, file and processing stage; completed/outcome counters; rolling file and
chunk rates; elapsed time and a conservative ETA. Resource fields are sampled by the tray at most once per second
and show daemon CPU, RSS and I/O rates, system CPU/RAM, battery or AC state, and filesystem-stat sizes for the SQLite
database and WAL. Resource-monitor failures are displayed as unavailable and never affect indexing.
