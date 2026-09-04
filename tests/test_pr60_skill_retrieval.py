import sqlite3

import pytest

from axp_client.skills import SkillScopeUnavailableError, compile_retrieval_plan, parse_skill
from test_pr60_skill_schema import skill_value


def _connection(path="x:/example/project-x/file.pdf"):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE documents(path_key TEXT)")
    con.execute("INSERT INTO documents VALUES (?)", (path,))
    return con


def test_prefer_and_strict_plans_reuse_spiral_stages():
    value = skill_value()
    value["retrieval"]["path_prefixes"] = ["X:\\Example\\Project-X"]
    skill = parse_skill(value)
    plan, diagnostics = compile_retrieval_plan(skill, _connection(), now_ms=1_700_000_000_000)
    assert [stage.name for stage in plan.stages] == [
        "skill_identity", "skill_hot", "skill_warm", "skill_scope", "global"]
    assert plan.allow_global_fallback is True
    assert diagnostics["resolved"] == 1
    value["retrieval"]["mode"] = "strict"
    plan, _ = compile_retrieval_plan(parse_skill(value), _connection())
    assert plan.allow_global_fallback is False
    assert plan.stages[-1].name == "skill_scope"


def test_unresolved_prefer_falls_back_and_strict_fails():
    value = skill_value()
    value["retrieval"]["path_prefixes"] = ["X:\\Missing"]
    plan, diagnostics = compile_retrieval_plan(parse_skill(value), _connection())
    assert plan.stages[0].name == "identity"
    assert diagnostics["skill_scope_unresolved"] is True
    value["retrieval"]["mode"] = "strict"
    with pytest.raises(SkillScopeUnavailableError):
        compile_retrieval_plan(parse_skill(value), _connection())
