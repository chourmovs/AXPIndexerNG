# Dependencies and model packs

Retrieval Engine V2 adds no heavyweight Python dependency. SQLite FTS5 and `sqlite-vec` remain the indexes;
FastEmbed supplies both dense `TextEmbedding` and optional `LateInteractionTextEmbedding`. No Torch,
sentence-transformers, FAISS, PyArrow, Pandas, or service is used.

Models are independent caches and are never placed in the runtime ZIP:

| Pack | Model ID | Purpose | Dimension | License / verification |
|---|---|---|---:|---|
| `dense-balanced/` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | default multilingual dense retrieval | 384 | Apache-2.0; verify files against the provider cache metadata |
| `dense-quality/` | `intfloat/multilingual-e5-large` | optional higher-quality prefixed dense retrieval | 1024 | MIT; verify provider cache metadata |
| `reranker-quality/` | `answerdotai/answerai-colbert-small-v1` | multilingual late-interaction reranking | token matrix width is model-defined | Apache-2.0; verify provider cache metadata |

Provision packs explicitly with the existing model-cache process. Runtime search uses local-only mode and
will not download missing artifacts. The committed unit suite uses deterministic stubs; the manual workflow
provisions real models.

The dense embedding profile (`balanced` or `quality`) is index configuration and is recorded in the database
signature. It is distinct from the search retrieval profile (`fast`, `hybrid`, or `quality`); retrieval quality
means hybrid retrieval plus the ColBERT reranker. Clients derive the model ID, vector dimension, and E5 query
prefix from that signature and never substitute a differently sized model.

Windows users without symlink privilege may see Hugging Face's non-fatal `WinError 1314` cache warning. Its
supported copy fallback remains usable; do not run as Administrator or enable Developer Mode solely for this.
Optionally set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to suppress the warning without changing cache behavior.
