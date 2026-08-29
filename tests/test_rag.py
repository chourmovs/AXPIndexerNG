import json
import sqlite3
import threading

import pytest
from axp_client.rag.answerability import DecisionReason, decide_answerability
from axp_client.rag.backend import FakeChatBackend
from axp_client.rag.citations import validate_citations
from axp_client.rag.context import ContextConfig, build_context
from axp_client.rag.service import ChatBusyError, GenerationFailedError, RagService


def hit(score, document_id=1, chunk_no=0, **extra):
    return {"document_id": document_id, "chunk_no": chunk_no, "relevance_score": score,
            "lexical_coverage": score, **extra}


def database(tmp_path, documents=((1, "content"),)):
    path = tmp_path / "rag.db"
    with sqlite3.connect(path) as con:
        con.executescript("""CREATE TABLE documents(id INTEGER PRIMARY KEY,title TEXT,filename TEXT,path TEXT,
                          ingestion_mode TEXT); CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,
                          chunk_no INTEGER,text TEXT,page_no INTEGER,section_heading TEXT);""")
        for document_id, mode in documents:
            con.execute("INSERT INTO documents VALUES(?,?,?,?,?)",
                        (document_id, f"Document {document_id}", f"doc{document_id}.txt", f"/docs/{document_id}", mode))
            if mode == "content":
                for number in range(3):
                    con.execute("INSERT INTO chunks(document_id,chunk_no,text,page_no,section_heading) VALUES(?,?,?,?,?)",
                                (document_id, number, f"evidence {document_id}/{number}", number + 1, "Section"))
    return path


def connect(path, readonly=False):
    del readonly
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def service(tmp_path, rows, response="Grounded answer. [S1]", documents=((1, "content"),), backend=None):
    db = database(tmp_path, documents)
    fake = backend or FakeChatBackend(response)
    def search(*args, **kwargs):
        return {"results": rows, "candidate_counts": {}}

    return RagService(backend=fake, search_fn=search, connect_fn=connect, db=db, embedder=None), fake


def test_answerability_rules():
    assert decide_answerability([]).reason == DecisionReason.NO_CONTENT_EVIDENCE
    assert decide_answerability([hit(0.39)]).reason == DecisionReason.WEAK_RETRIEVAL
    assert decide_answerability([hit(0.78)]).reason == DecisionReason.STRONG_EVIDENCE
    assert decide_answerability([hit(0.56), hit(0.51, chunk_no=1), hit(0.47, chunk_no=2)]).reason == DecisionReason.MULTIPLE_SUPPORT
    exact = hit(0.46, exact_content_identifier_match=True, lexical_coverage=0.4)
    assert decide_answerability([exact]).reason == DecisionReason.EXACT_SUPPORTED


@pytest.mark.parametrize("rows,reason", [([], "no_content_evidence"), ([hit(0.39)], "weak_retrieval")])
def test_gate_refuses_without_generation(tmp_path, rows, reason):
    rag, backend = service(tmp_path, rows)
    response = rag.ask("secret question")
    assert response["answerable"] is False
    assert response["decision"]["reason"] == reason
    assert backend.calls == []


def test_metadata_is_related_not_evidence(tmp_path):
    rag, backend = service(tmp_path, [hit(0.99, document_id=2)], documents=((2, "metadata"),))
    response = rag.ask("question")
    assert response["sources"] == []
    assert response["related_documents"] == [{"document_id": 2, "filename": "doc2.txt"}]
    assert backend.calls == []


def test_grounded_generation_and_prompt_boundary(tmp_path):
    rag, backend = service(tmp_path, [hit(0.78, chunk_no=1)])
    response = rag.ask("why?")
    assert response["status"] == "answered" and response["sources"][0]["id"] == "S1"
    assert backend.calls and "--- BEGIN EVIDENCE ---" in backend.calls[0]["user_prompt"]
    assert "evidence 1/1" not in backend.calls[0]["system_prompt"]


def test_progress_reports_real_pipeline_order_and_only_final_contains_answer(tmp_path):
    rag, _ = service(tmp_path, [hit(0.78)], "Unsupported answer [S99]")
    events = []
    response = rag.ask("why?", progress=events.append)
    assert [event["event"] for event in events] == ["retrieval_started", "retrieval_complete", "gate_complete",
                                                       "context_ready", "generation_started", "validation_started"]
    assert all("Unsupported answer" not in json.dumps(event) for event in events)
    assert response["status"] == "ungrounded_generation" and response["answer"] is None


def test_rejected_progress_stops_before_generation(tmp_path):
    rag, _ = service(tmp_path, [])
    events = []
    rag.ask("absent?", progress=events.append)
    assert [event["event"] for event in events] == ["retrieval_started", "retrieval_complete", "gate_complete"]


@pytest.mark.parametrize("answer,status,reason", [
    ("Answer without source", "ungrounded_generation", "invalid_citations"),
    ("Invented [S99]", "ungrounded_generation", "invalid_citations"),
    ("INSUFFICIENT_EVIDENCE", "insufficient_evidence", "model_declined"),
])
def test_generation_refusal_layers(tmp_path, answer, status, reason):
    rag, _ = service(tmp_path, [hit(0.78)], answer)
    response = rag.ask("question")
    assert response["answerable"] is False and response["status"] == status
    assert response["decision"]["reason"] == reason


def test_neighbors_merge_and_deduplicate(tmp_path):
    db = database(tmp_path)
    with connect(db) as con:
        context = build_context(con, [hit(0.8, chunk_no=1), hit(0.7, chunk_no=2)])
    assert len(context.blocks) == 1
    assert context.blocks[0].chunk_nos == [0, 1, 2]
    assert context.prompt_text.count("evidence 1/1") == 1


def test_context_diversification_and_budget(tmp_path):
    db = database(tmp_path, ((1, "content"), (2, "content"), (3, "content")))
    rows = [hit(0.9 - number / 100, document_id=1, chunk_no=number % 3) for number in range(8)]
    rows += [hit(0.7, document_id=2), hit(0.65, document_id=3)]
    with connect(db) as con:
        context = build_context(con, rows, ContextConfig(max_documents=2, max_seeds_per_document=1,
                                neighbor_radius=0, character_budget=240))
    assert len({block.document_id for block in context.blocks}) == 2
    assert len(context.prompt_text) <= 240


def test_citation_validation():
    assert validate_citations("Fact [S1]", ["S1"])[0]
    assert not validate_citations("Fact", ["S1"])[0]
    assert not validate_citations("Fact [S2]", ["S1"])[0]


def test_generation_failure_is_controlled(tmp_path):
    class Broken(FakeChatBackend):
        def generate(self, **kwargs):
            raise RuntimeError("secret evidence must not leak")

    rag, _ = service(tmp_path, [hit(0.9)], backend=Broken())
    with pytest.raises(GenerationFailedError):
        rag.ask("question")


def test_generation_lock_rejects_second_request(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Holding(FakeChatBackend):
        def generate(self, **kwargs):
            entered.set()
            release.wait(2)
            return "Answer [S1]"

    rag, _ = service(tmp_path, [hit(0.9)], backend=Holding())
    thread = threading.Thread(target=rag.ask, args=("first",))
    thread.start()
    assert entered.wait(1)
    with pytest.raises(ChatBusyError):
        rag.ask("second")
    release.set()
    thread.join()


def test_mini_evaluation_manifest_has_required_categories():
    cases = json.loads((__import__("pathlib").Path(__file__).parent / "rag_cases" / "cases.json").read_text())
    counts = {kind: sum(case["kind"] == kind for case in cases) for kind in {case["kind"] for case in cases}}
    assert all(counts.get(kind, 0) >= minimum for kind, minimum in
               {"answerable": 5, "unanswerable": 5, "metadata_only": 2, "ambiguous": 2}.items())
