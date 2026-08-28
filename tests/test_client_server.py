import http.client
import sqlite3
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest
from axp_client import server


def sqlite_connect(path, readonly=False):
    del readonly
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@pytest.fixture(autouse=True)
def direct_sqlite(monkeypatch):
    monkeypatch.setattr(server, "connect", sqlite_connect)


def add_document(db, path, document_id=42):
    with sqlite_connect(db) as con:
        con.executescript(
            """CREATE TABLE sources(id INTEGER PRIMARY KEY,path TEXT,path_key TEXT,label TEXT,kind TEXT,
                                      created_ms INTEGER,updated_ms INTEGER);
               CREATE TABLE documents(id INTEGER PRIMARY KEY,source_id INTEGER,path TEXT,path_key TEXT,
                                      extension TEXT,size_bytes INTEGER,modified_unix_ms INTEGER,sha256 TEXT,
                                      indexed_unix_ms INTEGER,title TEXT,filename TEXT);"""
        )
        source_id = con.execute(
            """INSERT INTO sources(path,path_key,label,kind,created_ms,updated_ms)
               VALUES('K:\\','k:','Test source','drive',0,0)"""
        ).lastrowid
        con.execute(
            """INSERT INTO documents(
               id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename)
               VALUES(?,?,?,?,'.txt',0,0,'hash',0,'Manual','manual.txt')""",
            (document_id, source_id, str(path), str(path).casefold()),
        )


@contextmanager
def running_server(db, opener):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(db, None, opener))
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def post(httpd, target):
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=2)
    connection.request("POST", target)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body


def test_open_document_uses_database_path(tmp_path):
    db = tmp_path / "index.db"
    document = tmp_path / "manual.txt"
    document.touch()
    add_document(db, document)
    opened = []
    with running_server(db, opened.append) as httpd:
        status, _ = post(httpd, "/api/document/42/open?path=C%3A%5Cmalicious.exe")
    assert status == 200
    assert opened == [document]


def test_open_document_unknown_and_missing(tmp_path):
    db = tmp_path / "index.db"
    missing = tmp_path / "missing.txt"
    add_document(db, missing)
    opened = []
    with running_server(db, opened.append) as httpd:
        unknown_status, _ = post(httpd, "/api/document/999/open")
        missing_status, _ = post(httpd, "/api/document/42/open")
    assert unknown_status == 404
    assert missing_status == 410
    assert not opened


def test_shutdown_address_security():
    assert server._is_loopback("127.0.0.1")
    assert server._is_loopback("::1")
    assert not server._is_loopback("192.0.2.10")
    handler_type = server.make_handler("unused.db", None)
    handler = handler_type.__new__(handler_type)
    handler.path = "/api/shutdown"
    handler.client_address = ("192.0.2.10", 1234)
    response = {}
    handler.send_json = lambda value, status=200: response.update(value=value, status=status)
    handler.do_POST()
    assert response == {"value": {"error": "shutdown is only available locally"}, "status": 403}


def test_shutdown_endpoint_accepts_loopback(tmp_path):
    db = tmp_path / "unused.db"
    with running_server(db, lambda _: None) as httpd:
        status, _ = post(httpd, "/api/shutdown")
    assert status == 200
