"""Compile Skills into PR59 retrieval plans and governed response recipes."""
from __future__ import annotations

from axp_client.rag.spiral import (HOT_DOCUMENTS, HOT_YEARS, WARM_DOCUMENTS, WARM_YEARS,
                                   RetrievalScope, SpiralPlan, SpiralStage,
                                   _years_ago_ms, default_spiral_plan)
from axp_core.path_keys import canonical_path_key, sql_path_prefix


class SkillScopeUnavailableError(ValueError):
    code = "skill_scope_unavailable"


class ProjectSkillScopeUnavailableError(SkillScopeUnavailableError):
    code = "project_skill_scope_unavailable"


def compile_response_instruction(skill):
    if not skill.answer_guidance and not skill.answer_sections:
        return None
    lines = ["SKILL RESPONSE RECIPE"]
    if skill.answer_guidance:
        lines += ["", "Focus:", skill.answer_guidance]
    if skill.answer_sections:
        lines += ["", "Preferred structure:"]
        for index, section in enumerate(skill.answer_sections, 1):
            requirement = "required" if section.required else "optional"
            lines += ["", f"{index}. {section.title} ({requirement})", f"   {section.guidance}"]
    lines += ["", "Do not invent content merely to fill a section.",
              "For a required section unsupported by evidence, state briefly that indexed evidence does not establish it.",
              "Omit optional sections when evidence does not support them."]
    return "\n".join(lines)


def _resolved_paths(con, paths):
    resolved, unresolved = [], []
    for path in paths:
        canonical = canonical_path_key(path).rstrip("\\/")
        patterns = (sql_path_prefix(canonical),)
        # Read-only compatibility for early development databases which stored
        # Windows separators as '/'. Production scanner keys remain '\\'.
        if "\\" in canonical:
            patterns += (sql_path_prefix(canonical).replace("\\\\", "/"),)
        row = con.execute("SELECT 1 FROM documents WHERE " + " OR ".join(
            "lower(path_key) LIKE ? ESCAPE '\\'" for _ in patterns) + " LIMIT 1", patterns).fetchone()
        (resolved if row else unresolved).append(canonical)
    return tuple(resolved), tuple(unresolved)


def compile_retrieval_plan(skill, con, *, search_depth=0, now_ms=None, project_context=None):
    retrieval = skill.retrieval
    project_root = None
    relative_diagnostics = {}
    requested_path_count = len(retrieval.relative_paths if retrieval.scope_kind == "project_relative"
                               else retrieval.path_prefixes)
    if retrieval.scope_kind == "project_relative":
        if project_context is None:
            raise ProjectSkillScopeUnavailableError("project context is required")
        project_root = project_context.root_path
        requested = tuple(str(__import__("pathlib").PureWindowsPath(project_root, relative))
                          for relative in retrieval.relative_paths)
        resolved, missing = _resolved_paths(con, requested)
        relative_diagnostics = {"resolved_relative_paths": tuple(
            relative for relative, absolute in zip(retrieval.relative_paths, requested) if absolute in resolved),
            "missing_relative_paths": tuple(
                relative for relative, absolute in zip(retrieval.relative_paths, requested) if absolute in missing),
            "relative_scope_count": len(resolved)}
        if not resolved and retrieval.mode == "strict":
            raise ProjectSkillScopeUnavailableError("project Skill scope is unavailable in indexed content")
        paths = resolved
    else:
        paths = tuple(canonical_path_key(path).rstrip("\\/") for path in retrieval.path_prefixes)
    has_territory = bool(paths or retrieval.extensions)
    if not has_territory:
        return default_spiral_plan(search_depth=search_depth, now_ms=now_ms), {"resolved": 0, "unresolved": 0}
    resolved, unresolved = _resolved_paths(con, paths)
    # Extension-only territory is resolvable without pretending it is a filesystem location.
    available = bool(resolved or (not retrieval.path_prefixes and retrieval.extensions))
    diagnostics = {"resolved": len(resolved), "unresolved": len(unresolved), **relative_diagnostics,
                   "scope_kind": retrieval.scope_kind, "scope_mode": retrieval.mode,
                   "requested_path_count": requested_path_count, "resolved_path_count": len(resolved),
                   "scope_available": available, "skill_scope_unresolved": not available}
    if not available:
        if retrieval.mode == "strict":
            raise SkillScopeUnavailableError("configured strict Skill scope is unavailable")
        return default_spiral_plan(search_depth=search_depth, now_ms=now_ms), diagnostics
    scope = RetrievalScope(path_prefixes=resolved, extensions=retrieval.extensions)
    if search_depth:
        stages = [SpiralStage("skill_scope_expanded", "scoped_lexical", scope,
                              retrieval.max_documents, allow_early_stop=True)]
    else:
        stages = [SpiralStage("skill_identity", "metadata_routed", scope,
                              min(retrieval.max_documents, 20))]
        if retrieval.temporal_policy == "recent_first":
            stages.extend((
                SpiralStage("skill_hot", "scoped_lexical", RetrievalScope(
                    path_prefixes=resolved, extensions=retrieval.extensions,
                    modified_after_ms=_years_ago_ms(HOT_YEARS, now_ms)), min(retrieval.max_documents, HOT_DOCUMENTS)),
                SpiralStage("skill_warm", "scoped_lexical", RetrievalScope(
                    path_prefixes=resolved, extensions=retrieval.extensions,
                    modified_after_ms=_years_ago_ms(WARM_YEARS, now_ms)), min(retrieval.max_documents, WARM_DOCUMENTS))))
        stages.append(SpiralStage("skill_scope", "scoped_lexical", scope, retrieval.max_documents,
                                  allow_early_stop=True))
    if retrieval.mode == "prefer":
        if project_root:
            stages.append(SpiralStage("project_root", "scoped_lexical",
                                      RetrievalScope(path_prefixes=(project_root,), extensions=retrieval.extensions),
                                      retrieval.max_documents, allow_early_stop=True))
        stages.append(SpiralStage("global", "global_hybrid", max_documents=retrieval.max_documents,
                                  allow_early_stop=False))
    return SpiralPlan(tuple(stages), allow_global_fallback=retrieval.mode == "prefer"), diagnostics
