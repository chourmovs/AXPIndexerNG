import re

# Requires both a letter and a digit, permits common equipment separators, and
# deliberately rejects ordinary words/numbers to keep identifier boosts precise.
IDENTIFIER_RE = re.compile(r"(?<![\w])(?=[A-Za-z0-9][A-Za-z0-9._/-]{1,31}(?![\w]))(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)*(?![\w])")


def normalize_identifier(value):
    return re.sub(r"[-._/\s]", "", value).upper()


def extract_identifiers(*values):
    found = {}
    for value in values:
        for match in IDENTIFIER_RE.finditer(value or ""):
            original = match.group(0)
            normalized = normalize_identifier(original)
            if len(normalized) >= 2:
                found.setdefault(normalized, original)
    return tuple(found.items())
