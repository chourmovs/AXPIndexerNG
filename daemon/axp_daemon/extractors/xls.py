import re

from .legacy_office import extract_markdown

_SHEET_HEADING = re.compile(r"(?im)(?=^#{1,6}\s+(?:sheet\s*:\s*)?[^\n]+\s*$)")


def extract(path):
    """Return workbook Markdown, retaining office-oxide's sheet boundaries and order."""
    text = extract_markdown(path)
    sections = [part.strip() for part in _SHEET_HEADING.split(text) if part.strip()]
    return [(part, None) for part in sections] or [(text, None)]
