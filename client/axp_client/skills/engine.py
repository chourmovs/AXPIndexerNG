"""Persistent Skill resolution facade owned by the client runtime."""
from __future__ import annotations

from dataclasses import dataclass, field

from .compiler import compile_response_instruction, compile_retrieval_plan
from .matching import match_skill
from .store import SkillStore


class SkillSelectionError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class SkillExecution:
    skill: object | None
    selection: str
    match_reason: str | None
    retrieval_plan: object | None = None
    business_context: str | None = None
    response_instruction: str | None = None
    diagnostics: dict = field(default_factory=dict)


class SkillEngine:
    def __init__(self, store):
        self.store = store if isinstance(store, SkillStore) else SkillStore(store)

    def list_skills(self):
        return self.store.list_skills()

    def get(self, skill_id):
        return self.store.get(skill_id)

    @property
    def invalid(self):
        return self.store.invalid

    def resolve(self, question, skill_id="auto"):
        if skill_id == "none":
            return SkillExecution(None, "none", None, diagnostics={"match": "disabled_by_request"})
        if skill_id != "auto":
            skill = self.store.get(skill_id)
            if skill is None:
                raise SkillSelectionError("skill_not_found")
            if not skill.enabled:
                raise SkillSelectionError("skill_disabled")
            return self._execution(skill, "manual", "manual")
        skill, reason, diagnostics = match_skill(question, self.store.list_skills())
        return self._execution(skill, "auto", reason, diagnostics) if skill else SkillExecution(
            None, "auto", None, diagnostics=diagnostics)

    @staticmethod
    def _execution(skill, selection, reason, diagnostics=None):
        return SkillExecution(skill, selection, reason, business_context=skill.business_context or None,
                              response_instruction=compile_response_instruction(skill),
                              diagnostics=diagnostics or {"match": "matched"})

    def compile(self, execution, con, *, search_depth=0):
        if execution.skill is None:
            return execution
        plan, diagnostics = compile_retrieval_plan(execution.skill, con, search_depth=search_depth)
        return SkillExecution(execution.skill, execution.selection, execution.match_reason, plan,
                              execution.business_context, execution.response_instruction,
                              {**execution.diagnostics, **diagnostics})
