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
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from axp_core.runtime import atomic_write_json
from axp_core.build_info import build_info

from .benchmark import BenchmarkRunner
from .citations import classify_citations
from .final_protocol import generate_final_answer
from .model_catalog import CATALOG_VERSION, MODELS

SCHEMA_VERSION = 2
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
    return classify_citations(answer, ["S1"])[0] == "valid"


def normalize_validation_text(answer):
    """Normalize harmless typography for validation without changing the answer."""
    value = unicodedata.normalize("NFKC", answer or "").replace("−", "-").replace("\xa0", " ")
    value = re.sub(r"(?<=\d),(?=\d)", ".", value.casefold())
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("g/cm^3", "g/cm3").replace("g·cm-3", "g/cm3").replace("g cm-3", "g/cm3")


def validate_scalar(answer):
    value = normalize_validation_text(answer)
    unit = re.search(r"g\s*/\s*cm3", value)
    ok = bool(re.search(r"(?<!\d)0\.74(?!\d)", value)) and bool(unit) and _valid_citations(answer) and len(value.split()) <= 80
    return ok, [] if ok else ["expected_value_unit_citation_or_concise_answer_missing"]


def validate_grounding(answer):
    lower = normalize_validation_text(answer)
    property_term = r"(?:liquid\s+)?density"
    missing_term = (r"(?:not\s+(?:explicitly\s+)?(?:provided|specified|stated|given|available|defined)"
                    r"|unavailable|cannot\s+(?:be\s+)?(?:determined|inferred))")
    refusal = lower == "insufficient_evidence" or bool(re.search(
        rf"(?:{property_term}.{{0,45}}{missing_term}|"
        rf"(?:evidence|information|details|data).{{0,45}}(?:does|do)\s+not\s+"
        rf"(?:provide|specify|state|give|contain).{{0,45}}{property_term}|"
        rf"no\s+(?:specific\s+)?{property_term}\s+(?:is\s+)?(?:provided|specified|stated|given|available)|"
        rf"only\s+(?:the\s+)?relative\s+vapou?r\s+density\s+(?:is\s+)?(?:provided|given)|"
        rf"(?:available\s+)?(?:data|details|information).{{0,45}}vapou?r\s+density.{{0,30}}not\s+(?:the\s+)?liquid(?:'s)?\s+density|"
        r"insufficient\s+evidence)", lower))

    liquid_value = bool(re.search(
        r"(?:liquid\s+density(?:\s+of\s+test-heptane)?|density\s+of\s+test-heptane)"
        r"\s*(?:is|=|:|would\s+be|is\s+approximately|≈|~)\s*(?:approximately\s+)?(?:3\.4|4\.19)\b",
        lower))
    calculation = bool(re.search(
        r"(?:3\.4\s*(?:\*|×|x)\s*1\.225|(?:infer|calculat|would\s+be|gives?).{0,55}"
        r"(?:density\s+of\s+test-heptane|4\.19\s*kg\s*/?\s*m3)|"
        r"density\s+of\s+test-heptane.{0,35}4\.19\s*kg\s*/?\s*m3)", lower))
    citation_valid = _valid_citations(answer) or lower == "insufficient_evidence"
    reasons = []
    if liquid_value or calculation:
        # Keep the PR56 umbrella reason for report/API compatibility; the next
        # item identifies which deterministic relation check fired.
        reasons.append("unsupported_liquid_density")
    if liquid_value:
        reasons.append("unsupported_liquid_density_asserted")
    if calculation:
        reasons.append("unsupported_density_calculation")
    if not refusal:
        reasons.append("missing_grounding_refusal")
    if not citation_valid:
        reasons.append("invalid_citation")
    return not reasons, reasons


def _has_unnegated_concept(text, concept):
    """Return whether a concept is asserted rather than mentioned as unavailable."""
    for match in re.finditer(concept, text):
        prefix = text[max(0, match.start() - 55):match.start()]
        if re.search(r"(?:\bno\b|\bnot\b|\bwithout\b|\bneither\b).{0,35}$", prefix):
            continue
        suffix = text[match.end():match.end() + 45]
        if re.match(r".{0,25}\b(?:not|unavailable|unspecified)\b", suffix):
            continue
        return True
    return False


def validate_packaging(answer):
    lower = normalize_validation_text(answer)
    refusal = lower == "insufficient_evidence" or bool(re.search(
        r"(?:no\s+(?:(?:specific|ibc|type\s*(?:iii|3))\s+)?(?:packaging(?:\s+type)?|container(?:\s+specification)?)|"
        r"(?:packaging(?:\s+type)?|container\s+specification).{0,35}"
        r"(?:not\s+(?:explicitly\s+)?(?:provided|specified|stated|given|defined)|cannot\s+be\s+determined)|"
        r"(?:no|there\s+are\s+no)\s+(?:specific\s+)?details.{0,20}(?:actual\s+)?packaging|"
        r"permitted\s+packaging.{0,20}cannot\s+be\s+determined|"
        r"only\s+(?:the\s+)?adr\s+packing\s+group\s+(?:is\s+)?(?:provided|given|stated)|"
        r"does\s+not\s+specify.{0,25}(?:packaging|container)|insufficient_evidence)", lower))
    concept = r"\b(?:type\s*(?:iii|3)\s*(?:container|packag\w*)|ibc|drums?|barrels?|bottles?)\b"
    invented = _has_unnegated_concept(lower, concept)
    citation_valid = _valid_citations(answer) or lower == "insufficient_evidence"
    reasons = []
    if invented:
        reasons.append("invented_packaging")
    if not refusal:
        reasons.append("missing_packaging_refusal")
    if not citation_valid:
        reasons.append("invalid_citation")
    return not reasons, reasons


def validate_citation(answer):
    ok = "alpha-7" in normalize_validation_text(answer) and _valid_citations(answer)
    return ok, [] if ok else ["missing_or_unknown_citation"]


def validate_summary(answer):
    lower = normalize_validation_text(answer)
    required = ("colorless liquid", "0.72", "98", "-4", "negligible")
    forbidden = ("viscosity", "odor", "vapour pressure", "vapor pressure", "melting point")
    complete = not bool(re.search(r"(?:\b(?:and|or|with|including|such\s+as)|[,;:\-])$", lower))
    invented = any(_has_unnegated_concept(lower, rf"\b{re.escape(value)}\b") for value in forbidden)
    reasons = []
    if not all(value in lower for value in required):
        reasons.append("missing_required_property")
    if invented:
        reasons.append("invented_property")
    if not _valid_citations(answer):
        reasons.append("invalid_citation")
    if not complete:
        reasons.append("truncated_answer")
    return not reasons, reasons


def validate_materials(answer):
    lower = normalize_validation_text(answer)
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
        return generate_final_answer(backend=backend, question=question, evidence=evidence,
                                     allowed_citation_ids=["S1"])

    def _model(self, model):
        result = {"model_id": model.id, "name": model.name, "quantization": model.quantization,
                  "size_bytes": model.size_bytes, "profile": model.public(), "status": "RUNTIME_FAILED"}
        # The catalog object is passed through untouched. Benchmark limits are
        # per generate() call and must never become a temporary model profile.
        backend = self.backend_factory(model, "intel_gpu", None); self._backend = backend
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
            quick_cold = bench._measure(backend, "intel_cold", 32); self._progress("Quick cold", increment=1)
            quick_warm = bench._measure(backend, "intel_warm", 32); self._progress("Quick warm", increment=1)
            old = bench.job.profile; bench.job.profile = "rag"
            rag = bench._measure(backend, "intel_rag", 64); bench.job.profile = old
            self._progress("RAG-size performance", increment=1)
            result["performance"] = {"quick_cold": quick_cold, "quick_warm": quick_warm, "rag": rag}
            protocol = []
            for name, evidence, question, validator in SYNTHETIC_TESTS:
                self._progress(name.replace("_", " ").title())
                final = self._answer(backend, evidence, question)
                passed, reasons = validator(final.answer)
                telemetry = final.generation_telemetry
                protocol.append({"id": name, "passed": passed, "reasons": reasons,
                    "expected_rule": {
                        "scalar_lookup": "The density is 0.74 g/cm³ with an allowed citation.",
                        "unsupported_liquid_density": "Do not equate relative vapor density with liquid density.",
                        "packaging_trap": "The evidence does not specify permitted packaging.",
                        "closed_citation": "Return ALPHA-7 and cite only S1.",
                        "summary": "Report every supplied property without inventing properties.",
                        "direct_lookup": "List all three approved materials and no unsupported material.",
                    }[name],
                    "answer": final.answer, "citations": final.citation_validation["citations"],
                    "citation_validation": final.citation_validation["status"],
                    "query_intent": final.query_intent, "response_mode": final.response_mode,
                    "target_words": final.target_words,
                    "requested_answer_tokens": final.requested_answer_tokens,
                    "effective_answer_tokens": final.effective_answer_tokens,
                    "reasoning_budget_tokens": telemetry.get("reasoning_budget_tokens"),
                    "finish_reason": telemetry.get("finish_reason"),
                    "canonicalized": final.canonicalized,
                    "truncated_tail_cleaned": final.truncated_tail_cleaned})
                self._progress(name, increment=1)
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
            final_health = backend.health()
            for target, candidates in {
                "device_lost_count": ("device_loss_count", "device_lost_count", "intel_device_lost_count"),
                "recovery_attempts": ("recovery_attempts", "intel_recovery_attempts"),
                "recovery_successes": ("recovery_successes", "intel_recovery_successes"),
            }.items():
                stability[target] = max(stability[target], *(int(final_health.get(key) or 0) for key in candidates))
            stability["recovery_attempts"] = max(
                stability["recovery_attempts"], int(bool(final_health.get("device_recovery_attempted"))))
            stability["recovery_successes"] = max(
                stability["recovery_successes"], int(bool(final_health.get("device_recovery_succeeded"))))
            result["stability"] = stability
            passed = sum(item["passed"] for item in protocol)
            result["axp_protocol"] = {"protocol_type": "axp_final_answer_protocol", "passed": passed,
                                      "total": len(protocol), "tests": protocol}
            result["profile_diagnostic"] = {key: getattr(model, key, None) for key in (
                "max_answer_tokens", "reasoning_enabled", "reasoning_budget_tokens", "min_visible_answer_tokens",
                "temperature", "top_p", "top_k", "repeat_penalty", "context_size")}
            result["dimensions"] = {"runtime": "PASS", "axp_protocol": f"{passed}/{len(protocol)}",
                "grounding": "PASS" if all(protocol[i]["passed"] for i in (1, 2)) else "FAIL",
                "citations": "PASS" if protocol[3]["passed"] else "FAIL",
                "stability": f'{stability["successful_generations"]}/{stability["requested_generations"]}'}
            critical_ok = all(protocol[i]["passed"] for i in (1, 2, 3))
            if stability["failed_generations"]: result["status"] = "UNSTABLE"
            elif passed == 6 and not critical_ok: result["status"] = "QUALIFIED_WITH_WARNINGS"
            elif passed == 6 and stability["device_lost_count"]: result["status"] = "QUALIFIED_WITH_WARNINGS"
            elif passed == 6: result["status"] = "QUALIFIED"
            elif passed == 5 and critical_ok: result["status"] = "QUALIFIED_WITH_WARNINGS"
            elif passed >= 3: result["status"] = "PROTOCOL_PARTIAL"
            else: result["status"] = "PROTOCOL_FAILED"
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
        for result in results:
            result["recommendations"] = (["UNSTABLE ON INTEL"] if result["status"] == "UNSTABLE" else
                ["QUALITY FAILED"] if result["status"] == "PROTOCOL_FAILED" else [])
        stable = [r for r in results if r.get("dimensions", {}).get("runtime") == "PASS" and
                  not r.get("stability", {}).get("failed_generations")]
        def tests(result):
            return {item["id"]: item["passed"] for item in result.get("axp_protocol", {}).get("tests", [])}
        fast = [r for r in stable if all(tests(r).get(name) for name in
                ("scalar_lookup", "unsupported_liquid_density", "packaging_trap", "closed_citation"))]
        balanced = [r for r in stable if all(tests(r).get(name) for name in
                    ("unsupported_liquid_density", "packaging_trap", "closed_citation", "summary", "direct_lookup"))]
        def metric(r, name): return r.get("performance", {}).get(name, {}).get("generation_ms") or float("inf")
        if fast: min(fast, key=lambda r: metric(r, "rag"))["recommendations"].append("FAST LOOKUP CANDIDATE")
        if balanced: min(balanced, key=lambda r: metric(r, "rag"))["recommendations"].append("BALANCED RAG CANDIDATE")

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
            report = {"qualification_schema_version": SCHEMA_VERSION, "build": build_info(),
                "axp_version": self.axp_version,
                "catalog_version": CATALOG_VERSION, "llama_cpp_runtime_version": self.runtime_version,
                "timestamp": datetime.now(timezone.utc).isoformat(), "profile": self.job.profile,
                "profile_config": PROFILE_CONFIG[self.job.profile], "hardware": {"cpu": platform.processor(), **self.hardware},
                "state": state.upper(), "models": results}
            self.job.report = report; self._persist(report); self.job.state = state; self.job.updated_at = time.time()
