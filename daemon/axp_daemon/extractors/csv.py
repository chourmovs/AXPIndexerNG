import csv


def _encoding(path):
    with path.open("rb") as stream:
        sample = stream.read(64 * 1024)
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("csv", sample, 0, min(1, len(sample)), "unsupported text encoding")


def extract(path):
    encoding = _encoding(path)
    with path.open("r", encoding=encoding, newline="") as stream:
        sample = stream.read(64 * 1024)
        stream.seek(0)
        first_physical_line = next((line for line in sample.splitlines() if line.strip()), "")
        delimiter = max((",", ";", "\t"), key=first_physical_line.count)
        rows = csv.reader(stream, delimiter=delimiter)
        first = next(rows, None)
        while first is not None and not any(value.strip() for value in first):
            first = next(rows, None)
        if first is None:
            return
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False
        headers = [value.strip() or f"Column {index}" for index, value in enumerate(first, 1)] if has_header else None
        pending = []

        def render(number, row):
            values = [(headers[index] if headers and index < len(headers) else f"Column {index + 1}", value)
                      for index, value in enumerate(row) if value.strip()]
            return f"Row {number} | " + " | ".join(f"{key}={value}" for key, value in values)

        if not has_header:
            pending.append(render(1, first))
        for number, row in enumerate(rows, 2):
            if not any(value.strip() for value in row):
                continue
            pending.append(render(number, row))
            if len(pending) >= 100:
                yield "\n".join(pending), None
                pending.clear()
        if pending:
            yield "\n".join(pending), None
