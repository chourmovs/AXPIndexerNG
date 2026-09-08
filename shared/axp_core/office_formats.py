"""Authoritative Office ingestion registry shared by discovery and dispatch."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficeFormat:
    family: str
    extractor: str
    legacy: bool = False


OFFICE_FORMATS = {
    ".doc": OfficeFormat("word", "doc", legacy=True),
    ".docx": OfficeFormat("word", "docx"),
    ".xls": OfficeFormat("excel", "xls", legacy=True),
    ".xlsx": OfficeFormat("excel", "xlsx"),
    ".ppt": OfficeFormat("powerpoint", "ppt", legacy=True),
    ".pptx": OfficeFormat("powerpoint", "pptx"),
}

OFFICE_EXTENSIONS = frozenset(OFFICE_FORMATS)
