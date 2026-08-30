import threading
import sqlite3

import pytest

from axp_client.rag.context import ContextConfig, build_context
from axp_client.rag.llama_cpp_backend import GenerationCancelled, LlamaCppBackend
from axp_client.rag.model_catalog import catalog_model


class StreamModel:
    n_threads = 4
    n_threads_batch = 8
    n_batch = 512

    def __init__(self, chunks):
        self.chunks = chunks
        self.invocation = None

    def create_chat_completion(self, *, stream=False, **invocation):
        self.invocation = {**invocation, "stream": stream}
        return iter(self.chunks)

    @staticmethod
    def tokenize(text, **_):
        return text.split()


def test_stream_reconstruction_progress_and_telemetry(tmp_path):
    model = StreamModel([
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "<think>hidden"}}]},
        {"choices": [{"delta": {"content": "</think>Answer [S1]"}, "finish_reason": "stop"}]},
    ])
    backend = LlamaCppBackend(tmp_path / "unused")
    backend._model = model
    assert backend.generate(system_prompt="system", user_prompt="user") == "Answer [S1]"
    assert model.invocation["stream"] is True
    progress = backend.generation_progress()
    assert progress["sequence"] == progress["generated_fragments"] == 2
    assert progress["generated_characters"] == len("<think>hidden</think>Answer [S1]")
    assert progress["time_to_first_token_ms"] is not None
    assert backend.last_telemetry["completion_tokens"] == 2
    assert backend.last_telemetry["finish_reason"] == "stop"


def test_cooperative_cancel_closes_stream(tmp_path):
    backend = LlamaCppBackend(tmp_path / "unused")
    closed = threading.Event()

    class CancelStream:
        def __iter__(self):
            return self

        def __next__(self):
            backend.request_cancel()
            return {"choices": [{"delta": {"content": "not exposed"}}]}

        def close(self):
            closed.set()

    class Model(StreamModel):
        def create_chat_completion(self, *, stream=False, **invocation):
            assert stream
            return CancelStream()

    backend._model = Model([])
    with pytest.raises(GenerationCancelled):
        backend.generate(system_prompt="system", user_prompt="user")
    assert closed.is_set()
    assert backend.loaded and backend.health()["model_state"] != "failed"
    assert backend.generation_progress()["phase"] == "cancelled"


def test_curated_latency_policies_are_exact():
    qwen, smol = catalog_model("qwen3-1.7b-q4km"), catalog_model("smollm3-3b-q4km")
    assert (qwen.max_answer_tokens, qwen.max_evidence_tokens, qwen.max_context_documents,
            qwen.max_context_blocks, qwen.max_seeds_per_document) == (192, 2048, 3, 6, 2)
    assert (smol.max_answer_tokens, smol.max_evidence_tokens, smol.max_context_documents,
            smol.max_context_blocks, smol.max_seeds_per_document) == (256, 3072, 4, 8, 2)


def test_evidence_cap_and_physical_budget(tmp_path):
    db = tmp_path / "context.db"
    with sqlite3.connect(db) as con:
        con.executescript("""CREATE TABLE documents(id INTEGER PRIMARY KEY,title TEXT,filename TEXT,path TEXT);
            CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,text TEXT,
            page_no INTEGER,section_heading TEXT);
            INSERT INTO documents VALUES(1,'Doc','doc.txt','/doc.txt');
            INSERT INTO chunks VALUES(1,1,0,'one two three four five six',NULL,'');""")
        con.row_factory = sqlite3.Row
        hit = {"document_id": 1, "chunk_no": 0, "relevance_score": .9}
        capped = build_context(con, [hit], ContextConfig(neighbor_radius=0, max_evidence_tokens=10),
                               token_counter=lambda value: len(value.split()), context_window=6144)
        physical = build_context(con, [hit], ContextConfig(neighbor_radius=0, max_evidence_tokens=2048,
                                 answer_reserve_tokens=192, safety_reserve_tokens=512),
                                 token_counter=lambda value: len(value.split()), context_window=6144,
                                 fixed_prompt_tokens=6000)
    assert capped.diagnostics["evidence_budget_tokens"] == 10
    assert physical.diagnostics["physical_context_budget_tokens"] == 0
    assert physical.diagnostics["evidence_budget_tokens"] == 0
