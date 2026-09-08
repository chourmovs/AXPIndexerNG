"""Safe, data-only access to legacy Office files through office-oxide."""


class OfficeExtractionError(RuntimeError):
    """An extraction failure with a stable, diagnostic-safe reason code."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def extract_markdown(path):
    """Parse without invoking Office, macros, links, or embedded actions."""
    try:
        from office_oxide import to_markdown

        text = to_markdown(str(path))
    except Exception as exc:  # parser exposes one public exception class
        message = str(exc).casefold()
        if any(word in message for word in ("password", "encrypted", "encryption")):
            code = "office_encrypted"
        elif any(word in message for word in ("unsupported", "not supported", "variant")):
            code = "office_unsupported_variant"
        elif any(word in message for word in ("invalid", "corrupt", "format", "ole", "cfb")):
            code = "office_invalid"
        else:
            code = "office_extraction_failed"
        raise OfficeExtractionError(code, type(exc).__name__) from exc
    text = str(text or "").strip()
    if not text:
        raise OfficeExtractionError("office_content_empty")
    return text
