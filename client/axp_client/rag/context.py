from dataclasses import dataclass

from .types import ContextResult, EvidenceBlock


@dataclass(frozen=True)
class ContextConfig:
    max_documents: int = 6
    max_seeds_per_document: int = 3
    neighbor_radius: int = 1
    max_blocks: int = 12
    character_budget: int | None = None  # legacy/testing fallback
    answer_reserve_tokens: int = 512
    safety_reserve_tokens: int = 512
    max_evidence_tokens: int | None = None


def select_distinct_seeds(hits, maximum):
    """Choose dense, spatially distinct passages; neighbors arrive through expansion."""
    selected = []
    for hit in sorted(hits, key=lambda row: (-float(row.get("evidence_score", row.get("relevance_score")) or 0),
                                              int(row.get("chunk_id") or 0))):
        number = int(hit["chunk_no"])
        if any(abs(number - int(existing["chunk_no"])) <= 1 for existing in selected):
            continue
        selected.append(hit)
        if len(selected) == maximum:
            break
    return selected


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


def build_context(con, hits, config=None, *, token_counter=None, context_window=None, fixed_prompt_tokens=0):
    config = config or ContextConfig()
    selected, documents = [], []
    for hit in hits:
        doc = int(hit["document_id"])
        if doc not in documents:
            if len(documents) >= config.max_documents:
                continue
            documents.append(doc)
    for doc in documents:
        selected.extend(select_distinct_seeds([hit for hit in hits if int(hit["document_id"]) == doc],
                                               config.max_seeds_per_document))

    by_doc = {}
    relevance = {}
    relevance_signals = {}
    for hit in selected:
        doc, number = int(hit["document_id"]), int(hit["chunk_no"])
        by_doc.setdefault(doc, set()).update(range(max(0, number - config.neighbor_radius), number + config.neighbor_radius + 1))
        relevance[doc] = max(relevance.get(doc, 0), float(hit.get("relevance_score") or 0))
        current = relevance_signals.setdefault(doc, {"vector_similarity": 0.0, "lexical_coverage": 0.0,
                                                      "exact_identifier_match": False, "exact_phrase_match": False,
                                                      "exact_filename_match": False})
        current["vector_similarity"] = max(current["vector_similarity"], float(
            hit.get("vector_similarity", hit.get("relevance_score")) or 0))
        current["lexical_coverage"] = max(current["lexical_coverage"], float(hit.get("lexical_coverage") or 0))
        for key in ("exact_identifier_match", "exact_phrase_match", "exact_filename_match"):
            current[key] = current[key] or bool(hit.get(key))
    blocks_by_doc = {}
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
            blocks_by_doc.setdefault(doc, []).append(EvidenceBlock(
                id="", document_id=doc, title=row["title"], filename=row["filename"], path=row["path"],
                page_no=next((x["page_no"] for x in group if x["page_no"] is not None), None),
                section_heading=next((x["section_heading"] for x in group if x["section_heading"]), ""),
                chunk_ids=[x["id"] for x in group], chunk_nos=[x["chunk_no"] for x in group],
                relevance_score=relevance[doc], text="\n\n".join(x["text"] for x in group),
                relevance_signals=relevance_signals[doc],
            ))
    # Admit each document's primary block before any document's secondary block.
    blocks = []
    for index in range(max((len(value) for value in blocks_by_doc.values()), default=0)):
        for doc in documents:
            if index < len(blocks_by_doc.get(doc, ())):
                blocks.append(blocks_by_doc[doc][index])
    accepted, rendered, used = [], [], 0
    token_budget = None
    physical_budget = None
    if token_counter is not None and context_window is not None:
        physical_budget = max(0, context_window - config.answer_reserve_tokens - config.safety_reserve_tokens
                              - fixed_prompt_tokens)
        token_budget = (physical_budget if config.max_evidence_tokens is None else
                        min(physical_budget, config.max_evidence_tokens))
    for raw in blocks[: config.max_blocks]:
        block = EvidenceBlock(**{**raw.__dict__, "id": f"S{len(accepted) + 1}"})
        value = _format(block)
        separator = 2 if rendered else 0
        remaining = (config.character_budget - used - separator) if config.character_budget is not None else None
        if token_budget is not None and token_counter("\n\n".join([*rendered, value])) > token_budget:
            # Prefer complete evidence blocks. Only truncate if this is the first/only block.
            if rendered:
                break
            header = value[: value.find("\n\n") + 2]
            # A merged neighbor block retains chunk order: try complete chunks before text truncation.
            chunk_texts = block.text.split("\n\n")
            whole = None
            for count in range(1, len(chunk_texts) + 1):
                candidate_block = EvidenceBlock(**{**block.__dict__, "chunk_ids": block.chunk_ids[:count],
                    "chunk_nos": block.chunk_nos[:count], "text": "\n\n".join(chunk_texts[:count])})
                candidate = _format(candidate_block)
                if token_counter(candidate) <= token_budget:
                    whole = (candidate_block, candidate)
                else:
                    break
            if whole is not None:
                block, value = whole
                accepted.append(block)
                rendered.append(value)
                used += len(value) + separator
                continue
            words = block.text.split()
            low, high, best = 0, len(words), ""
            while low <= high:
                middle = (low + high) // 2
                candidate = header + " ".join(words[:middle])
                if token_counter(candidate) <= token_budget:
                    best, low = candidate, middle + 1
                else:
                    high = middle - 1
            if not best or best == header:
                break
            value = best
            block = EvidenceBlock(**{**block.__dict__, "text": value[len(header):]})
        if remaining is None:
            remaining = len(value)
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
    prompt = "\n\n".join(rendered)
    evidence_tokens = token_counter(prompt) if token_counter else None
    return ContextResult(prompt, accepted, {"evidence_tokens": evidence_tokens, "evidence_budget_tokens": token_budget,
        "physical_context_budget_tokens": physical_budget, "max_evidence_tokens": config.max_evidence_tokens,
        "selected_documents": len({block.document_id for block in accepted}), "selected_blocks": len(accepted),
        "selected_seed_chunks": [int(hit["chunk_no"]) for hit in selected],
        "selected_chunk_ranges": [[min(block.chunk_nos), max(block.chunk_nos)] for block in accepted]})
