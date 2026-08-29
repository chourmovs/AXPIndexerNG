from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceBlock:
    id: str
    document_id: int
    title: str
    filename: str
    path: str
    page_no: int | None
    section_heading: str
    chunk_ids: list[int]
    chunk_nos: list[int]
    relevance_score: float
    text: str

    def source(self):
        value = asdict(self)
        value.pop("text")
        value.pop("chunk_nos")
        return value


@dataclass(frozen=True)
class ContextResult:
    prompt_text: str
    blocks: list[EvidenceBlock] = field(default_factory=list)


SearchHit = dict[str, Any]
