from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    page_no: int | None
    char_start: int
    char_end: int


def chunk_text(text, page_no=None, target_words=400, overlap_words=50):
    spans = []
    start = None
    import re

    for m in re.finditer(r"\S+", text):
        if start is None:
            start = m.start()
        spans.append((m.start(), m.end()))
    out = []
    step = max(1, target_words - overlap_words)
    for i in range(0, len(spans), step):
        group = spans[i : i + target_words]
        if not group:
            break
        a, b = group[0][0], group[-1][1]
        out.append(Chunk(text[a:b], page_no, a, b))
        if i + target_words >= len(spans):
            break
    return out
