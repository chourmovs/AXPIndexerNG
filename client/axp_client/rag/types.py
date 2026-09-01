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
    relevance_signals: dict = field(default_factory=dict)
    seed_chunk_id: int | None = None
    seed_chunk_no: int | None = None
    document_rank: int | None = None
    passage_rank: int | None = None

    def source(self):
        value = asdict(self)
        value.pop("text")
        value.pop("chunk_nos")
        return value


@dataclass(frozen=True)
class ContextResult:
    prompt_text: str
    blocks: list[EvidenceBlock] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


SearchHit = dict[str, Any]
