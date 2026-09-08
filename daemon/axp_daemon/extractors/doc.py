from .legacy_office import extract_markdown


def extract(path):
    """Extract binary Word content in source order; pagination is not inferred."""
    return [(extract_markdown(path), None)]
