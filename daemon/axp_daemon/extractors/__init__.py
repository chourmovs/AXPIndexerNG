from . import csv, docx, pdf, pptx, text, xlsx

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


def extract(path):
    return EXTRACTORS[path.suffix.lower()](path)
