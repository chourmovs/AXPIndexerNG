# Local AI models

Ask AXP supports two release-curated, compact GGUF models: **Qwen3 1.7B
Q4_K_M** (Fast, recommended for standard workstations) and **SmolLM3 3B
Q4_K_M** (Balanced). Existing manually imported GGUF files remain available as
a Custom local model and are never copied merely to migrate settings.

Downloads happen only after a user selects **Download** or **Download &
activate**. A model ID is resolved through the version-controlled catalog; the
browser cannot provide a URL. Catalog sources use immutable revisions, HTTPS
with normal certificate validation, a restricted Hugging Face redirect list,
exact byte counts and pinned SHA-256 identities. A temporary `.part` is hashed
as it arrives and is atomically published only after its size, hash, and GGUF
header pass verification. No proxy credentials, cookies, or model repository
tokens are requested or stored. When company network policy blocks Hugging
Face, users can import an approved local GGUF instead; AXP never weakens TLS or
changes proxy, VPN, firewall, or certificate settings.

Models live beneath `model-cache/chat/models`, outside the release archive.
Activation updates the selected ID and compatible path together, unloads the
old local backend explicitly, and does not require an AXP restart. Active
memory-mapped models cannot be removed.

## Privacy and devices

CPU inference uses the portable, pinned `llama-cpp-python` runtime. The device
preference supports Auto, CPU, and Intel GPU. Intel display adapter discovery
does not itself claim compatibility: an optional, cryptographically pinned
portable llama.cpp SYCL pack must successfully list an Intel device. The pack
is user-local; AXP installs no driver and changes neither system PATH nor the
registry. Its owned llama server binds only `127.0.0.1` on a random port, and
localhost IPC explicitly bypasses HTTP proxies. Indexed content and prompts
never leave the PC.

**Benchmark this PC** is designed to use a deterministic synthetic prompt, not
indexed documents. Results are hardware/model/runtime-specific and must be
remeasured after any of those identities change.
