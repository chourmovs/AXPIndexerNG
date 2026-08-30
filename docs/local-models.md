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

Release maintainers can install `requirements-catalog-verifier.txt` and run
`python scripts/verify_chat_model_catalog.py` to check each immutable
revision's remote size and canonical LFS/content SHA-256 through the Hugging
Face repository metadata API without downloading either multi-gigabyte model.
Xet hashes, when reported, are diagnostic identities and are never treated as
the downloaded file SHA-256. This is an explicit release check and is not part
of normal startup or CI network activity.

Models live beneath `model-cache/chat/models`, outside the release archive.
Activation updates the selected ID and compatible path together, unloads the
old local backend explicitly, and does not require an AXP restart. Selection is
lazy: **Installed** means the verified file exists, **Selected** means AXP will
use it, **Loading** covers native initialization, **Ready** is shown only after
a successful load, and **Load failed** records a failed attempt. Thus selected
does not mean ready; the first answerable question may trigger loading. Active
memory-mapped models cannot be removed. Retry is offered only for failures that
may be transient, never for a deterministic CPU incompatibility.

## Privacy and devices

The Windows release builds pinned `llama-cpp-python==0.3.23` from source rather
than consuming the opaque upstream CPU wheel. Its policy is x86-64 with AVX:
`GGML_NATIVE=OFF`, `GGML_AVX=ON`, AVX2/BMI2/AVX512 off, and CUDA, Vulkan and
SYCL off. This avoids build-runner-native tuning. FMA, F16C, BMI2 and AVX2 are
reported for diagnostics but are not required by this baseline. AXP uses
`py-cpuinfo` CPUID/OSXSAVE-aware flags plus Windows' OS processor-feature API
to preflight AVX before entering llama.cpp. Unsupported systems report
`backend_cpu_incompatible`, are non-retryable, and do not repeatedly attempt a
native model load. A defensive mapping also recognizes Windows status
`0xc000001d` (illegal instruction). The full traceback remains in `client.log`.

Auto resolves deterministically to CPU. Windows Intel display-adapter
detection is exposed separately from accelerator availability: detection never
claims that GPU inference is working. The Intel GPU option is disabled because
PR23 does not ship a qualified SYCL backend or accelerator pack. Selecting a
forced Intel device through the API is rejected as `intel_gpu_unavailable`, and
health reports CPU as the effective device with the machine-readable reason.

Intel GPU inference is intentionally deferred until a later release can ship
and test a pinned portable runtime, loopback-only process lifecycle, and
proxy-bypassed local IPC. AXP installs no drivers, changes neither PATH nor the
registry, and requires no administrator access. Indexed content and prompts
remain on the PC. PR23 does not advertise a benchmark control because no
production benchmark endpoint is included.

The model cache is outside release archives and survives application upgrades.
The release catalog verifier gates tagged packaging. Release smoke records the
llama.cpp system/build report and CPU capabilities, but no tiny redistributable
real GGUF is currently included; CI therefore cannot reproduce every older
workstation CPU or claim full model-load qualification.
