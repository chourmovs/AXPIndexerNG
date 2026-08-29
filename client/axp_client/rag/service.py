import logging
import threading
import time
import uuid

from .answerability import decide_answerability
from .citations import validate_citations
from .context import build_context
from .prompts import SYSTEM_PROMPT, user_prompt

LOGGER = logging.getLogger("axp_client")
RETRIEVAL_LIMIT = 24


class ChatBusyError(Exception):
    pass


class ChatUnavailableError(Exception):
    pass


class GenerationFailedError(Exception):
    pass


class RagService:
    def __init__(self, *, backend, search_fn, connect_fn, db, embedder):
        self.backend, self.search_fn, self.connect_fn = backend, search_fn, connect_fn
        self.db, self.embedder = db, embedder
        self._generation_lock = threading.Lock()

    def health(self):
        return self.backend.health()

    def ask(self, question, *, debug=False):
        request_id, started = uuid.uuid4().hex[:12], time.perf_counter()
        if not self.health().get("available"):
            raise ChatUnavailableError
        retrieval_start = time.perf_counter()
        with self.connect_fn(self.db, readonly=True) as con:
            result = self.search_fn(con, self.embedder, question, limit=RETRIEVAL_LIMIT, profile="hybrid", explain=True)
            candidates = result.get("results", result)
            document_ids = sorted({int(row["document_id"]) for row in candidates})
            modes = {}
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                modes = {row["id"]: dict(row) for row in con.execute(
                    f"SELECT id,ingestion_mode,filename FROM documents WHERE id IN ({placeholders})", document_ids
                )}
            content = [row for row in candidates if modes.get(int(row["document_id"]), {}).get("ingestion_mode", "content") == "content"]
            metadata = [row for row in candidates if modes.get(int(row["document_id"]), {}).get("ingestion_mode") == "metadata"]
            related = list({int(row["document_id"]): {"document_id": int(row["document_id"]),
                            "filename": modes[int(row["document_id"])]["filename"]} for row in metadata}.values())
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            decision = decide_answerability(content)
            base = {"status": "insufficient_evidence", "answerable": False, "answer": None, "sources": [],
                    "related_documents": related, "decision": decision.public()}
            if not decision.answerable:
                base["timings"] = {"retrieval_ms": retrieval_ms, "total_ms": (time.perf_counter() - started) * 1000}
                if debug:
                    base["debug"] = self._debug(candidates, content, decision, None)
                LOGGER.info("RAG request id=%s decision=%s results=%s documents=%s", request_id, decision.reason.value,
                            len(candidates), decision.content_documents)
                return base
            context_start = time.perf_counter()
            context = build_context(con, content)
        context_ms = (time.perf_counter() - context_start) * 1000
        if not context.blocks:
            base["decision"] = {"reason": "no_context_evidence", "best_relevance": decision.best_relevance}
            return base
        if not self._generation_lock.acquire(blocking=False):
            raise ChatBusyError
        generation_start = time.perf_counter()
        try:
            try:
                answer = self.backend.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt(question, context.prompt_text))
            except Exception as exc:
                LOGGER.error("Local generation failed request id=%s type=%s", request_id, type(exc).__name__)
                raise GenerationFailedError from exc
        finally:
            self._generation_lock.release()
        generation_ms = (time.perf_counter() - generation_start) * 1000
        if (answer or "").strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
            base["decision"] = {"reason": "model_declined", "best_relevance": decision.best_relevance}
        else:
            valid, cited = validate_citations(answer, [block.id for block in context.blocks])
            if valid:
                base.update(status="answered", answerable=True, answer=answer,
                            sources=[block.source() for block in context.blocks if block.id in cited])
            else:
                base["status"] = "ungrounded_generation"
                base["decision"] = {"reason": "invalid_citations", "best_relevance": decision.best_relevance}
        base["timings"] = {"retrieval_ms": retrieval_ms, "context_ms": context_ms, "generation_ms": generation_ms,
                           "total_ms": (time.perf_counter() - started) * 1000}
        if debug:
            base["debug"] = self._debug(candidates, content, decision, context)
        LOGGER.info("RAG request id=%s decision=%s results=%s documents=%s generation_ms=%.1f", request_id,
                    base["decision"]["reason"], len(candidates), decision.content_documents, generation_ms)
        return base

    @staticmethod
    def _debug(candidates, content, decision, context):
        return {"retrieval_candidates": len(candidates), "content_candidates": len(content),
                "gate": {"supporting_chunks": decision.supporting_chunks, "content_documents": decision.content_documents},
                "selected_evidence_ids": [] if context is None else [x.id for x in context.blocks],
                "selected_chunk_ids": [] if context is None else [x.chunk_ids for x in context.blocks],
                "context_characters": 0 if context is None else len(context.prompt_text)}
