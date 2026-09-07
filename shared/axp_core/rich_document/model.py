from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

RICH_DOCUMENT_SCHEMA_VERSION = 1
DOCX_RICH_EXTRACTOR_VERSION = 1


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RichAsset:
    id: str
    media_type: str
    byte_size: int
    sha256: str
    relationship_id: str | None = None
    alt_text: str | None = None
    title: str | None = None
    width_emu: int | None = None
    height_emu: int | None = None
    browser_renderable: bool = False
    filename: str = ""

    def to_dict(self):
        return {"id": self.id, "media_type": self.media_type, "byte_size": self.byte_size,
                "sha256": self.sha256, "relationship_id": self.relationship_id,
                "alt_text": self.alt_text, "title": self.title, "width_emu": self.width_emu,
                "height_emu": self.height_emu, "browser_renderable": self.browser_renderable,
                "filename": self.filename}


@dataclass(frozen=True)
class RichDocument:
    schema_version: int
    document_id: int
    filename: str
    sha256: str
    blocks: tuple[Mapping[str, Any], ...]
    assets: tuple[RichAsset, ...]
    fidelity: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "blocks", tuple(_freeze(dict(block)) for block in self.blocks))
        object.__setattr__(self, "diagnostics", _freeze(dict(self.diagnostics)))

    def to_manifest(self):
        return {"schema_version": self.schema_version,
                "extractor_version": DOCX_RICH_EXTRACTOR_VERSION,
                "document": {"id": self.document_id, "filename": self.filename,
                             "sha256": self.sha256, "format": "docx", "fidelity": self.fidelity},
                "blocks": _thaw(self.blocks), "assets": [asset.to_dict() for asset in self.assets],
                "diagnostics": _thaw(self.diagnostics)}

    @classmethod
    def from_manifest(cls, value):
        document = value["document"]
        return cls(value["schema_version"], document["id"], document["filename"], document["sha256"],
                   tuple(value["blocks"]), tuple(RichAsset(**asset) for asset in value["assets"]),
                   document["fidelity"], value["diagnostics"])
