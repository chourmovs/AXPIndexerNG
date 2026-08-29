from dataclasses import dataclass

from .types import ContextResult, EvidenceBlock


@dataclass(frozen=True)
class ContextConfig:
    max_documents: int = 6
    max_seeds_per_document: int = 3
    neighbor_radius: int = 1
    max_blocks: int = 12
    character_budget: int = 24_000


def _format(block):
    lines = [
        f"[{block.id}]", f"Document: {block.filename or block.title}", f"Document ID: {block.document_id}",
        f"Chunks: {min(block.chunk_nos)}-{max(block.chunk_nos)}", f"Path: {block.path}",
    ]
    if block.page_no is not None:
        lines.append(f"Page: {block.page_no}")
    if block.section_heading:
        lines.append(f"Section: {block.section_heading}")
    return "\n".join(lines) + "\n\n" + block.text


def build_context(con, hits, config=ContextConfig()):
    selected, counts, documents = [], {}, []
    for hit in hits:
        doc = int(hit["document_id"])
        if doc not in counts:
            if len(documents) >= config.max_documents:
                continue
            documents.append(doc)
            counts[doc] = 0
        if counts[doc] < config.max_seeds_per_document:
            selected.append(hit)
            counts[doc] += 1

    by_doc = {}
    relevance = {}
    for hit in selected:
        doc, number = int(hit["document_id"]), int(hit["chunk_no"])
        by_doc.setdefault(doc, set()).update(range(max(0, number - config.neighbor_radius), number + config.neighbor_radius + 1))
        relevance[doc] = max(relevance.get(doc, 0), float(hit.get("relevance_score") or 0))
    blocks = []
    for doc in documents:
        numbers = sorted(by_doc.get(doc, ()))
        if not numbers:
            continue
        placeholders = ",".join("?" for _ in numbers)
        rows = con.execute(
            f"""SELECT c.id,c.chunk_no,c.text,c.page_no,c.section_heading,d.title,d.filename,d.path
                FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=? AND c.chunk_no IN ({placeholders}) ORDER BY c.chunk_no""",
            [doc, *numbers],
        ).fetchall()
        groups = []
        for row in rows:
            if not groups or row["chunk_no"] != groups[-1][-1]["chunk_no"] + 1:
                groups.append([])
            groups[-1].append(row)
        for group in groups:
            row = group[0]
            blocks.append(EvidenceBlock(
                id="", document_id=doc, title=row["title"], filename=row["filename"], path=row["path"],
                page_no=next((x["page_no"] for x in group if x["page_no"] is not None), None),
                section_heading=next((x["section_heading"] for x in group if x["section_heading"]), ""),
                chunk_ids=[x["id"] for x in group], chunk_nos=[x["chunk_no"] for x in group],
                relevance_score=relevance[doc], text="\n\n".join(x["text"] for x in group),
            ))
    accepted, rendered, used = [], [], 0
    for raw in blocks[: config.max_blocks]:
        block = EvidenceBlock(**{**raw.__dict__, "id": f"S{len(accepted) + 1}"})
        value = _format(block)
        separator = 2 if rendered else 0
        remaining = config.character_budget - used - separator
        if remaining <= 0:
            break
        if len(value) > remaining:
            # Preserve Unicode code points; retain metadata and as much full text as fits.
            header = value[: value.find("\n\n") + 2]
            if len(header) >= remaining:
                break
            value = header + block.text[: remaining - len(header)]
            block = EvidenceBlock(**{**block.__dict__, "text": value[len(header):]})
        accepted.append(block)
        rendered.append(value)
        used += len(value) + separator
    return ContextResult("\n\n".join(rendered), accepted)
