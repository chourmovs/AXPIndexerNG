"""Release-owned catalog for optional native inference accelerators."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AcceleratorRelease:
    id: str
    upstream_repository: str
    tag: str
    commit: str
    asset: str
    exact_size: int
    sha256: str

    @property
    def url(self):
        return f"https://github.com/{self.upstream_repository}/releases/download/{self.tag}/{self.asset}"

    def public(self):
        value = asdict(self)
        value["display_size"] = "approximately 120 MB"
        return value


INTEL_SYCL = AcceleratorRelease(
    id="intel-sycl-b10516",
    upstream_repository="ggml-org/llama.cpp",
    tag="b10516",
    commit="b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9",
    asset="llama-b10516-bin-win-sycl-x64.zip",
    exact_size=119_741_566,
    sha256="b9a5a42ddc4033f05003b127d3fd18583565b33971eb01723c6711e95ece42b4",
)

