import re

from .legacy_office import extract_markdown

_SLIDE_HEADING = re.compile(r"(?im)(?=^#{1,6}\s+slide\s+(\d+)\b)")


def extract(path):
    """Return presentation Markdown in slide order with slide numbers when exposed."""
    text = extract_markdown(path)
    parts = [part.strip() for part in _SLIDE_HEADING.split(text) if part.strip()]
    sections = []
    page = None
    for part in parts:
        if part.isdigit():
            page = int(part)
        else:
            sections.append((part, page))
    return sections or [(text, None)]
