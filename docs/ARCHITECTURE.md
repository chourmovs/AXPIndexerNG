# Architecture

AXPIndexerNG Retrieval Engine V2 preserves the daemon/client boundary and the single SQLite database.

## Index contract

Schema **2**, chunker **2**, and embedding-input **2** form a deterministic signature with model ID,
dimension, and the `cosine` distance metric. A mismatch is never migrated implicitly: the operator must run
`AXPIndexerDaemon reindex --root ROOT --db DB`. Balanced indexes use the 384-dimensional multilingual
MiniLM model; optional quality dense indexes use multilingual E5-large (1024 dimensions and its
`query:`/`passage:` prefixes).

## Retrieval stages

1. **Broad first stage:** accent-insensitive Unicode FTS5 retrieves 100 weighted lexical candidates, while
   FastEmbed and exact sqlite-vec cosine KNN retrieve 100 dense candidates. Identifier, title, heading,
   filename, and body fields are kept distinct. Queries are converted to safe quoted atoms.
2. **RRF:** reciprocal rank fusion (`k=60`) joins ranks without pretending BM25 and cosine distance have a
   shared numeric scale. Exact identifier, filename-stem, and quoted-phrase evidence supplies only a small,
   deterministic priority safeguard.
3. **Optional multilingual ColBERT:** quality search batches at most 30 candidate texts through
   `answerdotai/answerai-colbert-small-v1`; query-token MaxSim determines the final reranking and RRF breaks
   ties. Token matrices are held only in a bounded client-process LRU (128 entries), never SQLite.
4. **Diversification:** results are round-robin by document, then second and third chunks are admitted. The
   default final cap is three chunks per document and 20 total results.

Chunker v2 operates separately on every extracted PDF page/PPTX slide, respects headings, paragraphs and
sentences, targets 350 words, caps at 500, and overlaps complete sentences up to roughly 60 words. Displayed
text remains original; filename/title/heading context is used only for dense embedding input.

`fast`, `hybrid`, and `quality` search profiles progressively enable dense retrieval, lexical+RRF, and
ColBERT. Quality mode fails rather than downloading or silently downgrading when its model is absent.
Exact sqlite-vec remains the scale policy; diagnostics expose vector counts and latency so deployments can
warn at 100,000 vectors and gather evidence before considering ANN.
