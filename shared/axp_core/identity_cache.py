"""Small single-flight identity result cache, never a stale document catalog."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from contextlib import contextmanager

from .tracing import current_trace, span

_condition = threading.Condition()
_results = OrderedDict()
_building = set()
_latest = OrderedDict()
MAX_ENTRIES = 128
MAX_KEY_CHARS = 8192


def ensure_identity_generation(con):
    """Writer-only additive migration, transactional with catalog changes."""
    con.execute("""CREATE TABLE IF NOT EXISTS identity_generation(
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), token TEXT NOT NULL, revision INTEGER NOT NULL)""")
    con.execute("INSERT OR IGNORE INTO identity_generation VALUES(1,lower(hex(randomblob(16))),0)")
    for table, fields in (("documents", ("filename", "title", "path", "path_key", "source_id",
                                        "extension", "modified_unix_ms")),
                          ("sources", ("label", "path"))):
        for action in ("INSERT", "DELETE", "UPDATE"):
            predicate = ""
            event = action
            if action == "UPDATE":
                event += " OF " + ",".join(fields)
                predicate = " WHEN " + " OR ".join(f"old.{f} IS NOT new.{f}" for f in fields)
            con.execute(f"""CREATE TRIGGER IF NOT EXISTS identity_generation_{table}_{action.lower()}
                AFTER {event} ON {table}{predicate} BEGIN
                UPDATE identity_generation SET revision=revision+1 WHERE singleton=1; END""")


@contextmanager
def identity_resolution(con, question, scope, limit):
    """Yield (cached_ids, publish) inside one SQLite read snapshot.

    Legacy catalogs and uncommitted writers bypass caching. Keys include all
    ranking/scope inputs and an in-database incarnation plus relevant revision.
    Completed entries contain <=48 integer IDs; no passage or catalog is cached.
    """
    with span("Cache identité : vérification / attente", "identity_cache.identity_resolution") as measured:
        if con.in_transaction:
            measured.detail(cache="bypass", invalidation_reason="caller_transaction")
            yield None, lambda ids: None
            return
        con.execute("SAVEPOINT identity_resolution")
        claimed = False
        key = None
        try:
            present = con.execute("SELECT 1 FROM sqlite_master WHERE name='identity_generation'").fetchone()
            if not present:
                measured.detail(cache="bypass", invalidation_reason="legacy_catalog_no_generation")
                yield None, lambda ids: None
                return
            generation = tuple(con.execute("SELECT token,revision FROM identity_generation WHERE singleton=1").fetchone())
            db = next((row[2] for row in con.execute("PRAGMA database_list") if row[1] == "main"), "")
            scope_key = json.dumps(vars(scope), sort_keys=True)
            key = (db, generation, question, scope_key, limit)
            if not db or len(db) + len(question) + len(scope_key) > MAX_KEY_CHARS:
                measured.detail(cache="bypass", invalidation_reason="memory_bound_or_memory_database")
                yield None, lambda ids: None
                return
            with _condition:
                while key in _building:
                    _condition.wait(.05)
                    if current_trace():
                        current_trace().check()
                cached = _results.get(key)
                reason = "cold" if db not in _latest else (
                    "identity_metadata_changed" if _latest[db] != generation else "query_not_cached")
                _latest[db] = generation
                _latest.move_to_end(db)
                while len(_latest) > MAX_ENTRIES:
                    _latest.popitem(last=False)
                if cached is not None:
                    _results.move_to_end(key)
                elif len(_building) < 16:
                    _building.add(key)
                    claimed = True
            measured.detail(cache="hit" if cached is not None else "miss", generation=f"{generation[0]}:{generation[1]}",
                            invalidation_reason=None if cached is not None else reason)
            measured.count(documents=len(cached) if cached is not None else 0)
            def publish(ids):
                if claimed:
                    with _condition:
                        _results[key] = tuple(ids)
                        _results.move_to_end(key)
                        while len(_results) > MAX_ENTRIES:
                            _results.popitem(last=False)
            yield list(cached) if cached is not None else None, publish
        finally:
            if claimed:
                with _condition:
                    _building.discard(key)
                    _condition.notify_all()
            con.execute("RELEASE identity_resolution")
