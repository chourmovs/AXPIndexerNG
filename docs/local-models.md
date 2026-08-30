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

Release maintainers can run `python scripts/verify_chat_model_catalog.py` to
check each immutable revision's remote size and LFS SHA-256 metadata without
downloading either multi-gigabyte model. This is an explicit release check and
is not part of normal startup or CI network activity.

Models live beneath `model-cache/chat/models`, outside the release archive.
Activation updates the selected ID and compatible path together, unloads the
old local backend explicitly, and does not require an AXP restart. Active
memory-mapped models cannot be removed.

## Privacy and devices

PR22 inference uses the portable, pinned CPU `llama-cpp-python` runtime. Auto
therefore resolves deterministically to CPU. Windows Intel display-adapter
detection is exposed separately from accelerator availability: detection never
claims that GPU inference is working. The Intel GPU option is disabled because
PR22 does not ship a qualified SYCL backend or accelerator pack. Selecting a
forced Intel device through the API is rejected as `intel_gpu_unavailable`, and
health reports CPU as the effective device with the machine-readable reason.

Intel GPU inference is intentionally deferred until a later release can ship
and test a pinned portable runtime, loopback-only process lifecycle, and
proxy-bypassed local IPC. AXP installs no drivers, changes neither PATH nor the
registry, and requires no administrator access. Indexed content and prompts
remain on the PC. PR22 does not advertise a benchmark control because no
production benchmark endpoint is included.
