# Grounded local RAG

AXP Answer is a backend-only, local document-question-answering pipeline:

`question -> existing hybrid retrieval -> deterministic answerability gate -> full-chunk context -> local LLM -> citation validation -> answer/refusal`

Search result relevance is not answerability. Search finds potentially useful material; the stricter, independently
configured gate determines whether content evidence is sufficient. Metadata-only documents can be returned as related
files but cannot support an answer. A rejected question never invokes the model.

Accepted seed chunks are expanded with their immediate neighbors from SQLite, deduplicated, merged when adjacent,
diversified across documents, and constrained to a 24,000-character budget. Sources have server-issued `S#` identities.
The server accepts an answer only when it contains one or more supplied citations and no unknown citation.

## Refusal layers

1. The deterministic retrieval gate rejects absent or weak content evidence.
2. The model is instructed to output `INSUFFICIENT_EVIDENCE` when the precise answer is absent or ambiguous.
3. Server-side validation rejects uncited answers and invented citation IDs.

The local LLM is never treated as a source of truth. Indexed evidence is the source of truth. Decision reasons remain
machine-readable for later calibration and outcome classification.

## Local model and security

Set `chat_backend` to `llama_cpp` and provision a GGUF file at `chat_model_path` (by default
`model-cache/chat/model.gguf`, resolved relative to the installation root). AXP does not download this file. The model
is loaded lazily on the first answerable Ask and cached; health checks do not load it. Generation is CPU-only by default
and limited to one request at a time.

RAG makes no cloud or outbound request and has no cloud fallback. `POST /api/ask` is loopback-only even if the search
server binds to all interfaces. Questions, prompts, evidence, and answers are not written to normal logs. Retrieved
document instructions are delimited, treated as untrusted data, and never followed.

`llama-cpp-python` remains an optional provisioning dependency: this change does not add it to the portable runtime
because a reliable pinned prebuilt Windows CPU wheel has not yet been validated. Consequently the normal portable ZIP
dependency delta is **0 bytes**, and dynamic workstation compilation is not attempted. The separately provisioned
GGUF is never included in the release ZIP. Once a wheel is validated, the portable acceptance check must include
`from llama_cpp import Llama` and record exact before/after ZIP sizes and the pinned version.

## Interfaces

* `GET /api/ask/health` reports configuration, availability, and lazy load state without exposing the model path.
* `POST /api/ask` accepts `{"question": "...", "debug": false}` (64 KiB body and 4,000-character question limits).
* `python -m axp_client ask --db data/axpindex.db --question "..."` runs the same service without the web server.

There is no database schema change, reindex, retrieval-ranking change, embedding change, or chunker change.
