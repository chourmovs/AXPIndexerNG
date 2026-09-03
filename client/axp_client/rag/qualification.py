"""Deterministic, local-only qualification of every catalog chat model.

The runner deliberately knows nothing about downloads or model selection.  Callers
provide isolated backend factories and a restoration callback, which makes the
same coordinator suitable for the HTTP UI, CLI, and unit tests.
"""
from __future__ import annotations

import platform
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from axp_core.runtime import atomic_write_json

from .benchmark import BenchmarkRunner
from .model_catalog import CATALOG_VERSION, MODELS

SCHEMA_VERSION = 1
TERMINAL_STATES = {"complete", "complete_with_errors", "failed", "cancelled"}
PROFILE_CONFIG = {
    "standard": {"stability_generations": 3, "request_timeout_s": 90, "load_timeout_s": 90,
                 "cpu_timeout_s": 90, "rag_words": 1536},
    "stress": {"stability_generations": 10, "request_timeout_s": 90, "load_timeout_s": 90,
               "cpu_timeout_s": 90, "rag_words": 1900},
}


def _citations(answer):
    return re.findall(r"\[(S\d+)\]", answer, flags=re.I)


def _valid_citations(answer):
    found = _citations(answer)
    return bool(found) and all(value.upper() == "S1" for value in found)


def validate_scalar(answer):
    unit = re.search(r"(?:g\s*/\s*cm(?:³|\^?3)|g\s*cm[-−]?3)", answer, re.I)
    ok = "0.74" in answer and bool(unit) and _valid_citations(answer) and len(answer.split()) <= 80
    return ok, [] if ok else ["expected_value_unit_citation_or_concise_answer_missing"]


def validate_grounding(answer):
    lower = answer.lower()
    refusal = "insufficient_evidence" in lower or ("liquid density" in lower and
        any(word in lower for word in ("not supplied", "not provided", "does not provide", "cannot determine")))
    ok = refusal and "3.4" not in answer
    return ok, [] if ok else ["unsupported_liquid_density"]


def validate_packaging(answer):
    lower = answer.lower()
    refusal = any(value in lower for value in ("not specify", "not provided", "no packaging", "cannot determine"))
    invented = bool(re.search(r"type\s*(?:iii|3)\s*(?:container|packag)", lower))
    ok = refusal and not invented
    return ok, [] if ok else ["invented_packaging"]


def validate_citation(answer):
    ok = _valid_citations(answer)
    return ok, [] if ok else ["missing_or_unknown_citation"]


def validate_summary(answer):
    lower = answer.lower().strip()
    required = ("0.72", "98", "-4", "negligible")
    forbidden = ("viscosity", "odor", "vapour pressure", "vapor pressure", "melting point")
    complete = bool(re.search(r"(?:[.!?]|\[S1\])$", lower)) and not lower.endswith(("and", ",", ":", "-"))
    ok = all(value in lower for value in required) and _valid_citations(answer) and complete and not any(
        value in lower for value in forbidden)
    return ok, [] if ok else ["incomplete_incorrect_or_invented_summary"]


def validate_materials(answer):
    lower = answer.lower()
    required = ("stainless steel", "ptfe", "glass")
    extras = ("aluminum", "aluminium", "copper", "polyethylene", "pvc")
    ok = all(value in lower for value in required) and not any(value in lower for value in extras) and _valid_citations(answer)
    return ok, [] if ok else ["storage_list_incorrect"]


SYNTHETIC_TESTS = (
    ("scalar_lookup", "[S1]\nMaterial: TEST-MTBE\nDensity at 20 °C: 0.74 g/cm³\nBoiling point: 55 °C",
     "What is the density of TEST-MTBE?", validate_scalar),
    ("unsupported_liquid_density", "[S1]\nMaterial: TEST-HEPTANE\nRelative vapor density (air = 1): 3.4\nBoiling point: 98 °C",
     "What is the liquid density of TEST-HEPTANE?", validate_grounding),
    ("packaging_trap", "[S1]\nMaterial: TEST-AMMONIA\nTransport classification:\nADR packing group III.\nNo packaging type or container specification is provided.",
     "What packaging is possible for TEST-AMMONIA?", validate_packaging),
    ("closed_citation", "[S1]\nMaterial TEST-CITATION has code ALPHA-7.",
     "What code is listed for TEST-CITATION? Cite the source.", validate_citation),
    ("summary", "[S1]\nMaterial TEST-SOLVENT\nAppearance: colorless liquid\nDensity: 0.72 g/cm³\nBoiling point: 98 °C\nFlash point: -4 °C\nWater solubility: negligible",
     "Summarize the main physical properties of TEST-SOLVENT.", validate_summary),
    ("direct_lookup", "[S1]\nMaterial: TEST-MATERIAL\nApproved storage materials:\n- stainless steel\n- PTFE\n- glass",
     "Which storage materials are listed for TEST-MATERIAL?", validate_materials),
)


@dataclass
class QualificationJob:
    state: str = "idle"
    profile: str = "standard"
    current_model: str | None = None
    model_index: int = 0
    model_total: int = 0
    phase: str | None = None
    backend: str | None = None
    completed_tests: int = 0
    total_tests: int = 0
    started_at: float | None = None
    updated_at: float | None = None
    elapsed_seconds: float = 0
    cancel_requested: bool = False
    report: dict | None = None
    error: str | None = None

    def public(self):
        value = asdict(self)
        if self.started_at: value["elapsed_seconds"] = round(time.time() - self.started_at, 1)
        return value


class ModelQualificationRunner:
    """Sequential all-catalog qualification coordinator."""
    def __init__(self, backend_factory, installed, report_root, *, restore=None, hardware=None,
                 models=None, axp_version="unknown", runtime_version="unknown"):
        self.backend_factory, self.installed = backend_factory, installed
        self.report_root = Path(report_root); self.restore = restore or (lambda: None)
        self.hardware = hardware or {}; self.models = tuple(models if models is not None else MODELS)
        self.axp_version, self.runtime_version = axp_version, runtime_version
        self.job = QualificationJob(); self._cancel = threading.Event(); self._backend = None
        self._lock = threading.Lock()

    def start(self, profile="standard", model_id=None):
        if profile not in PROFILE_CONFIG: raise ValueError("invalid_qualification_profile")
        with self._lock:
            if self.job.state not in ({"idle"} | TERMINAL_STATES): raise RuntimeError("qualification_busy")
            selected = [m for m in self.models if model_id is None or m.id == model_id]
            if model_id and not selected: raise ValueError("model_not_found")
            now = time.time(); total = len(selected) * (len(SYNTHETIC_TESTS) + 6)
            self.job = QualificationJob("preparing", profile, model_total=len(selected), total_tests=total,
                                        started_at=now, updated_at=now)
            self._selected = selected; self._cancel.clear()
        threading.Thread(target=self._run, daemon=True, name="axp-model-qualification").start()
        return self.job.public()

    def run(self, profile="standard", model_id=None):
        self.start(profile, model_id)
        while self.job.state not in TERMINAL_STATES: time.sleep(.01)
        return self.job.report

    def cancel(self):
        if self.job.state in ({"idle"} | TERMINAL_STATES): raise RuntimeError("qualification_not_active")
        self.job.cancel_requested = True; self._cancel.set()
        backend = self._backend
        if backend:
            if not getattr(backend, "request_load_cancel", lambda: False)():
                getattr(backend, "request_cancel", lambda: False)()
        return self.job.public()

    def _progress(self, phase, backend="Intel GPU", increment=0):
        self.job.phase, self.job.backend = phase, backend
        self.job.completed_tests += increment; self.job.updated_at = time.time()
        if self._cancel.is_set(): raise InterruptedError

    @staticmethod
    def _answer(backend, evidence, question):
        answer = backend.generate(system_prompt="Answer only from the evidence. Cite [S1]. Never invent facts.",
                                  user_prompt=f"Evidence:\n{evidence}\n\nQuestion: {question}")
        return answer if isinstance(answer, str) else getattr(answer, "text", str(answer or ""))

    def _model(self, model):
        result = {"model_id": model.id, "name": model.name, "quantization": model.quantization,
                  "size_bytes": model.size_bytes, "profile": model.public(), "status": "RUNTIME_FAILED"}
        backend = self.backend_factory(model, "intel_gpu", 64); self._backend = backend
        try:
            self._progress("Intel load/offload qualification")
            started = time.perf_counter(); backend.ensure_loaded(); load_ms = (time.perf_counter() - started) * 1000
            health = backend.health()
            result["runtime"] = {"model_load_ms": load_ms, **{k: health.get(k) for k in (
                "sycl_device_name", "gpu_offload_confirmed", "offloaded_layers", "total_layers", "gpu_buffer_bytes",
                "context_memory_bytes", "kv_memory_bytes", "failure_type")}}
            if not health.get("gpu_offload_confirmed"):
                result["runtime"]["failure_type"] = "intel_gpu_offload_not_confirmed"; return result
            bench = BenchmarkRunner(lambda _: backend, lambda _: backend, model.id, self.hardware)
            bench.job.profile = "quick"; bench.job.started_at = time.time()
            quick_cold = bench._measure(backend, "intel_cold"); self._progress("Quick cold", increment=1)
            quick_warm = bench._measure(backend, "intel_warm"); self._progress("Quick warm", increment=1)
            old = bench.job.profile; bench.job.profile = "rag"
            rag = bench._measure(backend, "intel_rag"); bench.job.profile = old
            self._progress("RAG-size performance", increment=1)
            result["performance"] = {"quick_cold": quick_cold, "quick_warm": quick_warm, "rag": rag}
            protocol = []
            for name, evidence, question, validator in SYNTHETIC_TESTS:
                self._progress(name.replace("_", " ").title())
                answer = self._answer(backend, evidence, question); passed, reasons = validator(answer)
                protocol.append({"id": name, "passed": passed, "reasons": reasons,
                                 "citations": _citations(answer), "synthetic_answer": answer})
                self._progress(name, increment=1)
            result["protocol"] = protocol
            stability = {"requested_generations": PROFILE_CONFIG[self.job.profile]["stability_generations"],
                         "successful_generations": 0, "failed_generations": 0, "device_lost_count": 0,
                         "recovery_attempts": 0, "recovery_successes": 0, "other_backend_errors": []}
            for _ in range(stability["requested_generations"]):
                self._progress("Short stability sequence")
                try:
                    self._answer(backend, "[S1]\nThe deterministic code is READY.", "What is the code?")
                    stability["successful_generations"] += 1
                except Exception as exc:
                    stability["failed_generations"] += 1
                    if "DEVICE_LOST" in str(exc).upper(): stability["device_lost_count"] += 1
                    else: stability["other_backend_errors"].append(type(exc).__name__)
                self._progress("Short stability sequence", increment=1)
            result["stability"] = stability
            passed = sum(item["passed"] for item in protocol)
            result["dimensions"] = {"runtime": "PASS", "protocol": f"{passed}/{len(protocol)}",
                "grounding": "PASS" if all(protocol[i]["passed"] for i in (1, 2)) else "FAIL",
                "citations": "PASS" if protocol[3]["passed"] else "FAIL",
                "stability": f'{stability["successful_generations"]}/{stability["requested_generations"]}'}
            if passed != len(protocol): result["status"] = "PROTOCOL_FAILED"
            elif stability["failed_generations"]: result["status"] = "UNSTABLE"
            elif stability["device_lost_count"]: result["status"] = "QUALIFIED_WITH_WARNINGS"
            else: result["status"] = "QUALIFIED"
            result["cpu_baseline"] = {"status": "SKIPPED_OPTIONAL"}
            result["field_observation"] = {"status": "SKIPPED_CORPUS_UNAVAILABLE"}
            return result
        except InterruptedError: raise
        except Exception as exc:
            result["runtime"] = {**result.get("runtime", {}), "failure_type": type(exc).__name__, "error": str(exc)[:500]}
            return result
        finally:
            try: backend.close()
            finally: self._backend = None

    def _recommend(self, results):
        passed = [r for r in results if r["status"] in ("QUALIFIED", "QUALIFIED_WITH_WARNINGS")]
        for result in results:
            result["recommendations"] = (["UNSTABLE ON INTEL"] if result["status"] == "UNSTABLE" else
                ["QUALITY FAILED"] if result["status"] == "PROTOCOL_FAILED" else [])
        if not passed: return
        def metric(r, name): return r.get("performance", {}).get(name, {}).get("generation_ms") or float("inf")
        min(passed, key=lambda r: metric(r, "quick_warm"))["recommendations"].append("FAST LOOKUP CANDIDATE")
        min(passed, key=lambda r: metric(r, "rag"))["recommendations"].append("BALANCED RAG CANDIDATE")

    def _persist(self, report):
        self.report_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        atomic_write_json(self.report_root / f"qualification-{stamp}.json", report)
        atomic_write_json(self.report_root / "last.json", report)

    def _run(self):
        results = []
        try:
            for index, model in enumerate(self._selected, 1):
                self.job.current_model, self.job.model_index = model.name, index
                if not self.installed(model):
                    results.append({"model_id": model.id, "name": model.name, "status": "NOT_INSTALLED",
                                    "recommendations": []}); continue
                if self._cancel.is_set(): raise InterruptedError
                results.append(self._model(model))
            self._recommend(results); state = "complete_with_errors" if any(
                r["status"] in ("RUNTIME_FAILED", "PROTOCOL_FAILED", "UNSTABLE") for r in results) else "complete"
        except InterruptedError: state = "cancelled"
        except Exception as exc: self.job.error, state = str(exc)[:500], "failed"
        finally:
            try: self.restore()
            except Exception as exc:
                self.job.error = f"runtime_restore_failed: {exc}"[:500]; state = "failed"
            report = {"qualification_schema_version": SCHEMA_VERSION, "axp_version": self.axp_version,
                "catalog_version": CATALOG_VERSION, "llama_cpp_runtime_version": self.runtime_version,
                "timestamp": datetime.now(timezone.utc).isoformat(), "profile": self.job.profile,
                "profile_config": PROFILE_CONFIG[self.job.profile], "hardware": {"cpu": platform.processor(), **self.hardware},
                "state": state.upper(), "models": results}
            self.job.report = report; self._persist(report); self.job.state = state; self.job.updated_at = time.time()
