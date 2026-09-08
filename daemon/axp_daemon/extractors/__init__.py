from axp_core.office_formats import OFFICE_FORMATS

from . import csv, doc, docx, pdf, ppt, pptx, text, xls, xlsx

EXTRACTORS = {
    ".txt": text.extract,
    ".md": text.extract,
    ".markdown": text.extract,
    ".pdf": pdf.extract,
    ".docx": docx.extract,
    ".pptx": pptx.extract,
    ".xlsx": xlsx.extract,
    ".csv": csv.extract,
}

_OFFICE_EXTRACTORS = {
    "doc": doc.extract, "docx": docx.extract,
    "xls": xls.extract, "xlsx": xlsx.extract,
    "ppt": ppt.extract, "pptx": pptx.extract,
}
EXTRACTORS.update({extension: _OFFICE_EXTRACTORS[spec.extractor]
                   for extension, spec in OFFICE_FORMATS.items()})


def extract(path):
    return EXTRACTORS[path.suffix.lower()](path)
