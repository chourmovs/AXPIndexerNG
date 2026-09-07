from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

MAX_DOCX_BYTES = 100 * 1024 * 1024
MAX_ZIP_ENTRIES = 10_000
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_ASSET_BYTES = 50 * 1024 * 1024
MAX_ASSETS = 512
MAX_BLOCKS = 100_000
MAX_TABLE_CELLS = 250_000
MAX_TEXT_CHARACTERS = 20_000_000
MAX_NESTING_DEPTH = 16


class RichDocumentError(Exception):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code)


def validate_docx(path: Path):
    try:
        if path.stat().st_size > MAX_DOCX_BYTES:
            raise RichDocumentError("rich_document_too_complex")
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_ENTRIES or sum(item.file_size for item in entries) > MAX_UNCOMPRESSED_BYTES:
                raise RichDocumentError("rich_document_too_complex")
            media = [item for item in entries if item.filename.lower().startswith("word/media/")]
            if len(media) > MAX_ASSETS or any(item.file_size > MAX_ASSET_BYTES for item in media):
                raise RichDocumentError("rich_document_too_complex")
            if "word/document.xml" not in archive.namelist():
                raise RichDocumentError("rich_document_invalid")
    except (BadZipFile, OSError) as exc:
        raise RichDocumentError("rich_document_invalid") from exc
