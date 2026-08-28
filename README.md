# AXPIndexerNG

AXPIndexerNG is a clean-sheet Python 3.11 document indexer. It **no longer uses Rust** and **no longer uses LanceDB**. Its single process boundary contract is SQLite + FTS5 + sqlite-vec; embeddings use FastEmbed and ONNX Runtime CPU.

## Design

Two independent applications share `axpindex.db`: the daemon scans, extracts, chunks, embeds, and reconciles files; the client reads the database and provides CLI/localhost search and a viewer. Supported inputs are TXT, Markdown, PDF, DOCX, and PPTX.

Set `PYTHONPATH=shared;daemon;client` on Windows (use `:` on Unix). The model is deliberately outside the runtime. Set `FASTEMBED_CACHE_PATH` to a provisioned model cache; runtime commands never download silently. CI may pass `--allow-download`.

There are two independent uses of the word **quality**. The daemon's `--embedding-profile quality` selects
the 1024-dimensional `intfloat/multilingual-e5-large` dense indexing model instead of the balanced
384-dimensional model. The client's `--profile quality` is a retrieval strategy: hybrid FTS+dense retrieval
followed by ColBERT reranking. It does not select the dense model. The client always reads the database index
signature and loads the matching dense query model and prefixes; it fails clearly when that model is absent or
incompatible rather than silently falling back to balanced.

On Windows, Hugging Face may report `WinError 1314` when the account cannot create cache symlinks. FastEmbed's
copy-based fallback still downloads and uses the model, so this warning is non-fatal and neither Administrator
access nor Developer Mode is required. `HF_HUB_DISABLE_SYMLINKS_WARNING=1` may be set to hide only this warning.

```text
python -m axp_daemon health --db axpindex.db
python -m axp_daemon scan --root documents --db axpindex.db
python -m axp_daemon status --db axpindex.db
python -m axp_client health --db axpindex.db
python -m axp_client search --db axpindex.db --query "reactor pressure"
python -m axp_client serve --db axpindex.db --host 127.0.0.1 --port 8765
```

Install Python 3.11 runtime roots with `pip install -r requirements-runtime.txt`; development checks additionally use `requirements-dev.txt`.
