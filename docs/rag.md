# Local grounded answers

AXP's Ask runtime retrieves from the existing index, applies a deterministic answerability gate, and **does not call
the model when evidence is weak**. Accepted evidence is placed in a bounded prompt, generated locally, and checked for
known `[S#]` citations. Metadata-only documents cannot be factual evidence. Search ranking and defaults are unchanged.

## Validated baseline model

The reference is [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), file
`Qwen3-4B-Q4_K_M.gguf`: Qwen3, approximately 4B parameters, Q4_K_M, approximately 2.5 GB, multilingual (including
French and English), under Apache-2.0. It is a validated baseline rather than the only potentially compatible GGUF.
The model and its license are **not repackaged in the AXP ZIP**. Administrators must confirm that model use complies
with organizational policy and the upstream license.

Provision it explicitly and offline:

```console
python -m axp_client chat-model import --file "D:\Downloads\Qwen3-4B-Q4_K_M.gguf"
python -m axp_client chat-model status
python -m axp_client chat-model remove --yes
```

Import validates the regular `.gguf` file and header, atomically copies it to the configured `chat_model_path`, and
records size, SHA-256, original filename, and import time. It never uses HTTP. A missing model does not affect indexing
or Search; Ask health reports `model_missing` and Ask returns a controlled unavailable response. No download occurs at
startup or request time.

## CPU runtime and supply chain

The portable runtime pins `llama-cpp-python==0.3.23` and installs a prebuilt basic CPU wheel from the project's official
wheel index (`https://abetlen.github.io/llama-cpp-python/whl/cpu`). Release installation sets `PIP_ONLY_BINARY` for this
package and fails rather than compiling. The bundled Python is WinPython CPython 3.11.8, Windows x86-64. CUDA, GPU
drivers, compilers, CMake, pip, and system Python are not end-user prerequisites. The release acceptance step imports
`Llama` using the bundled interpreter. Network restrictions prevented recording the exact wheel SHA-256 during this
implementation; release artifact provenance should be captured by CI before publication.

CPU operation uses `n_gpu_layers=0`, an 8192-token context, a 512-token answer reserve, and a safety reserve. Actual
token counts come from the loaded model tokenizer only after the gate accepts. Qwen thinking is disabled through
`chat_template_kwargs`, and any unexpected thinking wrapper is removed before citation validation. Native verbosity is
disabled. Only one lazily loaded model and one generation at a time are allowed.

No Windows resource measurement was available for this change, so peak RSS, loaded idle RSS, utilization, and tokens/s
are intentionally not estimated. Operators should measure them on representative corporate hardware.

## Confidence and calibration

Search relevance is a ranking signal, **not a probability that an answer exists**. The gate keeps vector similarity,
lexical coverage, content identifier/phrase matches, supporting chunks, and document counts separate. A filename match
alone is not factual support, and high lexical coverage with weak semantic support is refused.

Private corpora can be evaluated without committing or copying their cases:

```console
python -m axp_client rag-eval --db data/axpindex.db --cases my-rag-cases.json
python -m axp_client rag-eval --db data/axpindex.db --cases my-rag-cases.json --full --output result.json
```

Gate-only is the default and never invokes the LLM. Full mode also measures generation, refusal, and citation outcomes.
The JSON result is written only when requested and never contains evidence text. Case files support `expected_terms`
as a rough hint—not semantic proof—and optional expected document identifiers for manual retrieval review. False
accept and false refusal rates are reported independently. Initial thresholds remain provisional until tested against
a larger representative corpus.

## Security boundary

Inference is local and has no cloud, tools, or web-search path. Evidence is explicitly untrusted and cannot override the
system prompt. Model decline and invalid/unknown/missing citations become refusals. Ask is loopback-only; browser Origin
must also be loopback on the serving port, cross-site Fetch Metadata is rejected, JSON content type and bounded positive
Content-Length are required, and responses use `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.
Debug output includes only counts, IDs, gate signals, and token diagnostics—not prompts or evidence.
