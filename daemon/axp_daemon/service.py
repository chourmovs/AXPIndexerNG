from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from axp_core.database import connect
from axp_core.locking import AlreadyLocked, FileLock, daemon_lock_path
from axp_core.metadata import ensure_index_signature
from axp_core.runtime import atomic_write_json, configure_logging, read_json, runtime_paths
from axp_core.sources import get_source, list_sources, remove_source

from .embeddings import Embedder
from .indexer import scan_source
from .scanner import discover

LOGGER = configure_logging("axp_daemon", "daemon.log")


def now_ms():
    return int(time.time() * 1000)


def send_control(command, **values):
    payload = {"command": command, "requested_ms": now_ms(), "request_id": f"{os.getpid()}-{time.time_ns()}", **values}
    atomic_write_json(runtime_paths()["control"], payload)
    if command == "stop":
        atomic_write_json(runtime_paths()["desired"], {"state": "stopped", "updated_ms": now_ms()})
    return payload


class StatePublisher:
    def __init__(self, interval_s=2, warning_interval_s=60, launch_mode="interactive"):
        self.path = runtime_paths()["state"]
        self.started_ms = now_ms()
        self.value = {
            "pid": os.getpid(), "state": "starting", "started_ms": self.started_ms, "launch_mode": launch_mode,
            "heartbeat_ms": now_ms(), "current_source_id": None, "current_source": None, "current_file": None,
            "current_stage": "idle", "source_scan_started_ms": None, "scan_cycle_started_ms": None,
            "files_discovered": 0, "files_processed": 0, "files_new": 0, "files_modified": 0,
            "files_seen": 0, "files_content": 0, "files_metadata": 0, "files_ignored": 0,
            "files_failed": 0, "files_completed": 0, "files_unchanged": 0,
            "chunks_generated": 0, "chunks_embedded": 0,
            "current_batch_documents": 0, "current_batch_chunks": 0,
            "progress_baseline": 0, "progress_baseline_kind": "none",
            "documents_total": 0, "chunks_total": 0, "model_download_attempts": 0, "last_error": None,
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.interval_s = interval_s
        self.warning_interval_s = warning_interval_s
        self.write_failures = 0
        self.last_failure_warning = None
        self.thread = threading.Thread(target=self._run, name="axp-heartbeat", daemon=True)

    def start(self):
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.value.update(values)

    def _write(self):
        with self.lock:
            value = {**self.value, "heartbeat_ms": now_ms(),
                     "heartbeat_write_failures": self.write_failures}
        atomic_write_json(self.path, value)

    def _publish_safely(self):
        try:
            self._write()
        except (OSError, RuntimeError) as exc:
            self.write_failures += 1
            now = time.monotonic()
            if self.last_failure_warning is None or now - self.last_failure_warning >= self.warning_interval_s:
                LOGGER.warning("Heartbeat publication failed attempt=%s: %r", self.write_failures, exc)
                self.last_failure_warning = now
            return False
        if self.write_failures:
            LOGGER.info("Heartbeat publication recovered after %s failed writes", self.write_failures)
            self.write_failures = 0
            self.last_failure_warning = None
        return True

    def _run(self):
        while not self.stop_event.wait(self.interval_s):
            self._publish_safely()
        self._publish_safely()

    def close(self, state="stopped"):
        self.update(state=state, current_file=None, current_stage="idle")
        self.stop_event.set()
        self.thread.join(timeout=3)


class DaemonControl:
    def __init__(self, publisher):
        self.publisher = publisher
        self.stop = False
        self.paused = False
        self.scan_requested = False
        self.reindex_source_id = None
        self.remove_source_id = None
        self.last_request_id = None

    def poll(self):
        request = read_json(runtime_paths()["control"], {}) or {}
        request_id = request.get("request_id")
        if not request_id or request_id == self.last_request_id:
            return
        self.last_request_id = request_id
        command = request.get("command")
        LOGGER.info("Control request: %s", command)
        if command == "stop":
            self.stop = True
            self.publisher.update(state="stopping")
        elif command == "pause":
            self.paused = True
            self.publisher.update(state="paused", current_stage="paused")
        elif command == "resume":
            self.paused = False
            self.scan_requested = True
            self.publisher.update(state="idle")
        elif command == "scan":
            self.scan_requested = True
        elif command == "reindex":
            self.reindex_source_id = int(request["source_id"])
            self.scan_requested = True
        elif command == "remove":
            self.remove_source_id = int(request["source_id"])
            self.scan_requested = True

    def should_stop(self):
        self.poll()
        return self.stop

    def wait_if_paused(self):
        self.poll()
        while self.paused and not self.stop:
            self.publisher.update(state="paused")
            time.sleep(0.25)
            self.poll()
        if not self.stop:
            self.publisher.update(state="scanning")

    def source_started(self, source, started_ms):
        source_values = dict(source)
        baseline = source_values.get("last_seen_count", 0)
        counters = {
            key: 0 for key in (
                "files_seen", "files_processed", "files_completed", "files_content", "files_metadata",
                "files_ignored", "files_failed", "files_new", "files_modified", "files_unchanged",
                "chunks_generated", "chunks_embedded",
            )
        }
        self.publisher.update(state="scanning", current_source_id=source["id"], current_source=source["path"],
                              current_file=None, current_stage="discovering", source_scan_started_ms=started_ms,
                              progress_baseline=baseline or 0,
                              progress_baseline_kind="previous_complete_scan" if baseline else "none",
                              current_batch_documents=0, current_batch_chunks=0,
                              last_error=None, **counters)

    def stage(self, stage):
        self.publisher.update(current_stage=stage)

    def batch(self, stage, documents=0, chunks=0):
        self.publisher.update(current_stage=stage, current_file=None,
                              current_batch_documents=documents, current_batch_chunks=chunks)

    def current_file(self, source, path, result):
        self.publisher.update(
            state="scanning", current_source_id=source["id"], current_source=source["path"], current_file=str(path),
            files_discovered=result["files_discovered"], files_processed=result["files_scanned"],
            files_new=result["files_new"], files_modified=result["files_modified"], files_failed=result["files_failed"],
            files_completed=result["files_completed"], files_unchanged=result["files_unchanged"],
            files_seen=result["files_seen"], files_content=result["files_content"],
            files_metadata=result["files_metadata"], files_ignored=result["files_ignored"],
            chunks_generated=result["chunks_generated"], chunks_embedded=result["chunks_embedded"],
        )

    def progress(self, result):
        self.publisher.update(files_new=result["files_new"], files_modified=result["files_modified"],
                              files_completed=result["files_completed"], files_unchanged=result["files_unchanged"],
                              files_seen=result["files_seen"], files_content=result["files_content"],
                              files_metadata=result["files_metadata"], files_ignored=result["files_ignored"],
                              files_failed=result["files_failed"], chunks_generated=result["chunks_generated"],
                              chunks_embedded=result["chunks_embedded"])

    def batch_committed(self, con, result):
        self.progress(result)
        self.publisher.update(current_batch_documents=0, current_batch_chunks=0, **_catalog_totals(con))

    def file_error(self, path, exc):
        LOGGER.warning("Indexing failed for %s: %s", path, exc)
        self.publisher.update(last_error=f"{path}: {exc}")


def _catalog_totals(con):
    return {
        "documents_total": con.execute("SELECT count(*) FROM documents").fetchone()[0],
        "chunks_total": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
    }


def _safe_reindex(con, source_id):
    source = get_source(con, source_id)
    root = Path(source["path"])
    if not root.exists() or not root.is_dir():
        raise RuntimeError("Source is offline; old indexed content was preserved")
    preflight = discover(root, recursive=bool(source["recursive"]))
    try:
        for _ in preflight:
            pass
    except OSError as exc:
        raise RuntimeError("Source is offline; old indexed content was preserved") from exc
    if not preflight.complete:
        raise RuntimeError(f"Source enumeration is incomplete; old indexed content was preserved: {preflight.errors[-1]}")
    con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.source_id=?)", (source_id,))
    con.execute("DELETE FROM documents WHERE source_id=?", (source_id,))
    con.commit()


def _wait_for_model_retry(control, seconds):
    deadline = time.monotonic() + max(1, seconds)
    while time.monotonic() < deadline:
        control.poll()
        if control.stop:
            return False
        time.sleep(0.25)
    return True


def _provision_embedder(profile, model_cache, allow_download, retry_s, publisher, control):
    attempts = 0
    while not control.stop:
        try:
            embedder = Embedder(profile, cache_dir=model_cache, local_only=True)
            publisher.update(state="starting", last_error=None, model_download_attempts=attempts)
            return embedder
        except Exception as local_error:
            if not allow_download:
                raise RuntimeError(
                    f"Required embedding model {profile!r} is missing or incomplete in {model_cache!s}. "
                    "Start the desktop tray with model downloads enabled or provision this cache manually."
                ) from local_error
        attempts += 1
        publisher.update(state="downloading_model", model_download_attempts=attempts, last_error=None)
        LOGGER.warning("Embedding cache missing or incomplete; provisioning profile %s (attempt %s)", profile, attempts)
        try:
            embedder = Embedder(profile, cache_dir=model_cache, local_only=False)
            LOGGER.info("Embedding model %s is ready in %s", embedder.model_id, model_cache)
            publisher.update(state="starting", last_error=None, model_download_attempts=attempts)
            return embedder
        except Exception as download_error:
            message = (
                f"Could not provision embedding model {profile!r} in {model_cache!s}: {download_error}. "
                f"Retrying in {max(1, retry_s)} seconds."
            )
            LOGGER.exception(message)
            publisher.update(state="waiting_for_model", last_error=message, model_download_attempts=attempts)
            if not _wait_for_model_retry(control, retry_s):
                return None
    return None


def run_daemon(db, model_cache, embedding_profile="balanced", scan_interval=300, embedding_batch_size=64,
               allow_model_download=False, model_download_retry_s=60, launch_mode="interactive"):
    paths = runtime_paths()
    try:
        lock = FileLock(daemon_lock_path(db, paths["runtime"])).acquire()
    except AlreadyLocked:
        return {"status": "already_running"}
    atomic_write_json(paths["desired"], {"state": "running", "updated_ms": now_ms()})
    publisher = StatePublisher(launch_mode=launch_mode)
    control = DaemonControl(publisher)
    publisher.start()
    try:
        embedder = _provision_embedder(embedding_profile, model_cache, allow_model_download,
                                      model_download_retry_s, publisher, control)
        if embedder is None:
            return {"status": "stopped"}
        con = connect(db, dimension=embedder.dimension)
        ensure_index_signature(con, embedder.model_id, embedder.dimension, embedder.distance_metric)
        publisher.update(state="idle", **_catalog_totals(con))
        first_cycle = True
        while not control.stop:
            control.poll()
            if control.paused:
                control.wait_if_paused()
                continue
            if first_cycle or control.scan_requested:
                first_cycle = False
                control.scan_requested = False
                publisher.update(scan_cycle_started_ms=now_ms())
                if control.remove_source_id is not None:
                    remove_id = control.remove_source_id
                    control.remove_source_id = None
                    try:
                        remove_source(con, remove_id)
                        LOGGER.info("Removed indexed source %s at a safe scan boundary", remove_id)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error("Could not remove source %s: %s", remove_id, exc)
                        publisher.update(state="error", last_error=str(exc))
                reindex_id = control.reindex_source_id
                control.reindex_source_id = None
                cycle_sources = list_sources(con, enabled_only=True)
                if reindex_id is not None:
                    try:
                        _safe_reindex(con, reindex_id)
                        cycle_sources = [get_source(con, reindex_id)]
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error("Reindex refused for source %s: %s", reindex_id, exc)
                        publisher.update(state="error", last_error=str(exc))
                        cycle_sources = []
                for source in cycle_sources:
                    if control.should_stop():
                        break
                    try:
                        result = scan_source(con, source["id"], embedder,
                                             embedding_batch_size=embedding_batch_size, control=control)
                        LOGGER.info(
                            "Source scan completed: %s status=%s seen=%s content=%s metadata=%s ignored=%s failed=%s chunks=%s",
                            source["path"], result["status"], result["files_seen"], result["files_content"],
                            result["files_metadata"], result["files_ignored"], result["files_failed"],
                            result["chunks_embedded"],
                        )
                        metadata_formats = sorted(
                            ((ext, values["metadata"]) for ext, values in result["extension_breakdown"].items()
                             if values["metadata"]), key=lambda item: (-item[1], item[0]))[:5]
                        if metadata_formats:
                            LOGGER.info("Metadata-only formats: %s", " ".join(f"{ext}={count}" for ext, count in metadata_formats))
                    except Exception as exc:
                        LOGGER.exception("Source scan failed: %s", source["path"])
                        publisher.update(state="error", last_error=str(exc))
                if control.remove_source_id is not None:
                    remove_id = control.remove_source_id
                    control.remove_source_id = None
                    try:
                        remove_source(con, remove_id)
                        LOGGER.info("Removed indexed source %s at a safe scan boundary", remove_id)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error("Could not remove source %s: %s", remove_id, exc)
                        publisher.update(state="error", last_error=str(exc))
                publisher.update(state="idle" if not control.paused else "paused", current_source_id=None,
                                 current_source=None, current_file=None, current_stage="idle",
                                 current_batch_documents=0, current_batch_chunks=0,
                                 source_scan_started_ms=None, **_catalog_totals(con))
                cycle_end = time.monotonic()
            else:
                cycle_end = time.monotonic()
            while not control.stop and not control.scan_requested and time.monotonic() - cycle_end < scan_interval:
                control.poll()
                time.sleep(0.25)
            if not control.stop:
                control.scan_requested = True
        publisher.update(state="stopping")
        con.close()
        return {"status": "stopped"}
    except Exception as exc:
        LOGGER.exception("Persistent daemon failed")
        publisher.update(state="error", last_error=str(exc))
        return {"status": "error", "error": str(exc)}
    finally:
        publisher.close("stopped" if control.stop else publisher.value.get("state", "error"))
        lock.release()
