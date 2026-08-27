def extract(path):
    from docx import Document

    return [("\n".join(p.text for p in Document(path).paragraphs), None)]
