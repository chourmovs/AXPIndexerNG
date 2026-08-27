def extract(path):
    import fitz

    with fitz.open(path) as doc:
        return [(p.get_text(), i + 1) for i, p in enumerate(doc)]
