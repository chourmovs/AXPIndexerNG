"""Release-owned allow-list of local chat models.

URLs are deliberately derived on the server from immutable repository revisions;
the browser can never nominate a download location.
"""
from dataclasses import asdict, dataclass, field

CATALOG_VERSION = 1


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    profile: str
    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    display_size: str
    license: str = "Apache-2.0"
    context_size: int = 6144
    max_answer_tokens: int = 384
    temperature: float = 0.2
    chat_template_kwargs: dict = field(default_factory=dict)

    def public(self):
        value = asdict(self)
        for private in ("revision", "filename", "sha256"):
            value.pop(private)
        value["quantization"] = "Q4_K_M"
        return value

    @property
    def url(self):
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{self.filename}"


MODELS = (
    ModelProfile("qwen3-1.7b-q4km", "Qwen3 1.7B", "fast", "ggml-org/Qwen3-1.7B-GGUF",
                 "daeb8e2d528a760970442092f6bf1e55c3b659eb", "Qwen3-1.7B-Q4_K_M.gguf",
                 "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5",
                 1_280_000_000, "1.28 GB", chat_template_kwargs={"enable_thinking": False}),
    ModelProfile("smollm3-3b-q4km", "SmolLM3 3B", "balanced", "ggml-org/SmolLM3-3B-GGUF",
                 "8a3f806f5f5d76b4a4385b736f4f8ce3f5eecb91", "SmolLM3-Q4_K_M.gguf",
                 "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
                 1_915_305_312, "1.92 GB", chat_template_kwargs={"enable_thinking": False}),
)


def catalog_model(model_id):
    return next((model for model in MODELS if model.id == model_id), None)
