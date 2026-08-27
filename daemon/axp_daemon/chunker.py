import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    page_no: int | None
    char_start: int
    char_end: int
    section_heading: str = ""


WORD_RE = re.compile(r"\S+")
SENTENCE_RE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)


def _is_heading(value):
    text = value.strip()
    words = text.split()
    return bool(text) and len(words) <= 12 and (text.endswith(":") or text.startswith("#") or text.isupper())


def _sentences(text, offset):
    result = []
    for match in SENTENCE_RE.finditer(text):
        value = match.group(0).strip()
        if value:
            left = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            result.append((value, offset + left, offset + left + len(value)))
    return result


def chunk_text(text, page_no=None, target_words=350, overlap_words=60, max_words=500):
    """Deterministic paragraph/sentence chunker; callers invoke it per page/slide."""
    units = []
    heading = ""
    for paragraph in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        raw = paragraph.group(0).strip()
        if not raw:
            continue
        start = paragraph.start() + len(paragraph.group(0)) - len(paragraph.group(0).lstrip())
        if _is_heading(raw):
            heading = raw.lstrip("# ").rstrip(":").strip()
            units.append((raw, start, start + len(raw), heading, True))
            continue
        for sentence, a, b in _sentences(raw, start):
            words = len(WORD_RE.findall(sentence))
            if words <= max_words:
                units.append((sentence, a, b, heading, False))
            else:  # Last resort only: split a pathological sentence at word boundaries.
                spans = list(WORD_RE.finditer(sentence))
                for i in range(0, len(spans), max_words):
                    group = spans[i : i + max_words]
                    x, y = group[0].start(), group[-1].end()
                    units.append((sentence[x:y], a + x, a + y, heading, False))
    chunks = []
    i = 0
    while i < len(units):
        selected = []
        count = 0
        active_heading = units[i][3]
        j = i
        while j < len(units):
            unit = units[j]
            size = len(WORD_RE.findall(unit[0]))
            # A heading starts a new structural chunk once useful content exists.
            if selected and unit[4] and count >= 40:
                break
            if selected and count + size > max_words:
                break
            selected.append(unit)
            count += size
            if not active_heading:
                active_heading = unit[3]
            j += 1
            if count >= target_words:
                break
        if not selected:
            break
        a, b = selected[0][1], selected[-1][2]
        chunks.append(Chunk(text[a:b].strip(), page_no, a, b, active_heading))
        if j >= len(units):
            break
        # Preserve complete sentence overlap, bounded to the requested size.
        overlap = 0
        next_i = j
        for k in range(j - 1, i, -1):
            size = len(WORD_RE.findall(units[k][0]))
            if overlap and overlap + size > overlap_words:
                break
            overlap += size
            next_i = k
            if overlap >= overlap_words:
                break
        i = max(i + 1, next_i)
    return chunks
