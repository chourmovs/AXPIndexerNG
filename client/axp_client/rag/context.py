import logging
from dataclasses import dataclass

from .types import ContextResult, EvidenceBlock

LOGGER = logging.getLogger("axp_client")


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
    for hit in sorted(hits, key=lambda row: (int(row.get("passage_rank") or 10**9),
                                              -float(row.get("passage_score", row.get("evidence_score", row.get("relevance_score"))) or 0),
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
    selected = []
    first_seen = {}
    for index, hit in enumerate(hits):
        first_seen.setdefault(int(hit["document_id"]), index)
    documents = sorted(first_seen, key=lambda doc: (
        min((int(hit.get("document_rank") or 10**9) for hit in hits if int(hit["document_id"]) == doc),
            default=10**9), first_seen[doc]))[:config.max_documents]
    for doc in documents:
        selected.extend(select_distinct_seeds([hit for hit in hits if int(hit["document_id"]) == doc],
                                               config.max_seeds_per_document))
    blocks_by_doc = {}
    for hit in selected:
        doc, seed_no = int(hit["document_id"]), int(hit["chunk_no"])
        numbers = list(range(max(0, seed_no - config.neighbor_radius), seed_no + config.neighbor_radius + 1))
        placeholders = ",".join("?" for _ in numbers)
        rows = con.execute(
            f"""SELECT c.id,c.chunk_no,c.text,c.page_no,c.section_heading,d.title,d.filename,d.path
                FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=? AND c.chunk_no IN ({placeholders}) ORDER BY c.chunk_no""",
            [doc, *numbers],
        ).fetchall()
        if not rows:
            continue
        seed = next((row for row in rows if int(row["chunk_no"]) == seed_no), None)
        if seed is None:
            continue
        signals = {"vector_similarity": float(hit.get("vector_similarity", hit.get("relevance_score")) or 0),
                   "lexical_coverage": float(hit.get("lexical_coverage") or 0),
                   "scoped_lexical_rank": hit.get("scoped_lexical_rank"),
                   **{key: bool(hit.get(key)) for key in
                      ("exact_identifier_match", "exact_phrase_match", "exact_filename_match")}}
        blocks_by_doc.setdefault(doc, []).append(EvidenceBlock(
            id="", document_id=doc, title=seed["title"], filename=seed["filename"], path=seed["path"],
            page_no=seed["page_no"], section_heading=seed["section_heading"] or "",
            chunk_ids=[row["id"] for row in rows], chunk_nos=[row["chunk_no"] for row in rows],
            relevance_score=float(hit.get("relevance_score") or 0),
            text="\n\n".join(row["text"] for row in rows), relevance_signals=signals,
            seed_chunk_id=int(seed["id"]), seed_chunk_no=seed_no,
            document_rank=documents.index(doc) + 1,
            passage_rank=int(hit.get("passage_rank") or len(blocks_by_doc.get(doc, ())) + 1)))
    # Admit each document's primary block before any document's secondary block.
    blocks = []
    for index in range(max((len(value) for value in blocks_by_doc.values()), default=0)):
        for doc in documents:
            if index < len(blocks_by_doc.get(doc, ())):
                blocks.append(blocks_by_doc[doc][index])
    accepted, rendered = [], []
    token_budget = None
    physical_budget = None
    if token_counter is not None and context_window is not None:
        physical_budget = max(0, context_window - config.answer_reserve_tokens - config.safety_reserve_tokens
                              - fixed_prompt_tokens)
        token_budget = (physical_budget if config.max_evidence_tokens is None else
                        min(physical_budget, config.max_evidence_tokens))
    token_cache = {}
    def count(value):
        if value not in token_cache:
            token_cache[value] = token_counter(value)
        return token_cache[value]

    active_token_budget = token_budget
    active_character_budget = config.character_budget

    def fits(value):
        prompt = "\n\n".join([*rendered, value])
        return ((active_token_budget is None or count(prompt) <= active_token_budget) and
                (active_character_budget is None or len(prompt) <= active_character_budget))

    for block_index, raw in enumerate(blocks[:config.max_blocks]):
        # During the primary round, reserve a proportional share for every selected document.
        primary_number = block_index + 1
        if block_index < len(documents):
            active_token_budget = (None if token_budget is None else
                                   max(1, token_budget * primary_number // len(documents)))
            active_character_budget = (None if config.character_budget is None else
                                       max(1, config.character_budget * primary_number // len(documents)))
        else:
            active_token_budget = token_budget
            active_character_budget = config.character_budget
        raw = EvidenceBlock(**{**raw.__dict__, "id": f"S{len(accepted) + 1}"})
        chunk_rows = list(zip(raw.chunk_ids, raw.chunk_nos, raw.text.split("\n\n")))
        seed_index = next(index for index, row in enumerate(chunk_rows) if row[1] == raw.seed_chunk_no)
        variants = [chunk_rows]
        neighbors = [row for index, row in enumerate(chunk_rows) if index != seed_index]
        if neighbors:
            nearest = min(neighbors, key=lambda row: abs(row[1] - raw.seed_chunk_no))
            variants.append(sorted([chunk_rows[seed_index], nearest], key=lambda row: row[1]))
        variants.append([chunk_rows[seed_index]])
        chosen = None
        for rows in variants:
            candidate = EvidenceBlock(**{**raw.__dict__, "chunk_ids": [row[0] for row in rows],
                "chunk_nos": [row[1] for row in rows], "text": "\n\n".join(row[2] for row in rows)})
            value = _format(candidate)
            if fits(value):
                chosen = candidate, value
                break
        if chosen is None:
            seed = variants[-1][0]
            candidate = EvidenceBlock(**{**raw.__dict__, "chunk_ids": [seed[0]], "chunk_nos": [seed[1]]})
            words, low, high = seed[2].split(), 1, len(seed[2].split())
            while low <= high:
                middle = (low + high) // 2
                trial = EvidenceBlock(**{**candidate.__dict__, "text": " ".join(words[:middle])})
                value = _format(trial)
                if fits(value):
                    chosen, low = (trial, value), middle + 1
                else:
                    high = middle - 1
        if chosen is None:
            continue
        block, value = chosen
        accepted.append(block)
        rendered.append(value)
    prompt = "\n\n".join(rendered)
    evidence_tokens = count(prompt) if token_counter else None
    admitted_documents = len({block.document_id for block in accepted})
    if len(documents) >= 2 and admitted_documents == 1 and (token_budget or 0) >= 512:
        LOGGER.error("RAG context fairness violation selected_documents=%s admitted_documents=%s "
                     "budget_tokens=%s", len(documents), admitted_documents, token_budget)
    return ContextResult(prompt, accepted, {"evidence_tokens": evidence_tokens, "evidence_budget_tokens": token_budget,
        "physical_context_budget_tokens": physical_budget, "max_evidence_tokens": config.max_evidence_tokens,
        "selected_documents": admitted_documents, "selected_blocks": len(accepted),
        "requested_documents": len(documents),
        "admitted_seeds": [{"document_rank": block.document_rank, "document_id": block.document_id,
            "filename": block.filename, "passage_rank": block.passage_rank,
            "seed_chunk_no": block.seed_chunk_no, "page_no": block.page_no,
            "passage_score": block.relevance_score,
            "scoped_lexical_rank": block.relevance_signals.get("scoped_lexical_rank")} for block in accepted],
        "selected_seed_chunks": [int(hit["chunk_no"]) for hit in selected],
        "selected_chunk_ranges": [[min(block.chunk_nos), max(block.chunk_nos)] for block in accepted]})
