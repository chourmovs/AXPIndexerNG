import sqlite3

from axp_client.skills import compile_retrieval_plan, parse_skill
from axp_core.path_keys import canonical_path_key, sql_path_prefix
from axp_daemon.scanner import path_key
from test_pr60_skill_schema import skill_value


def test_windows_and_unc_path_contract_preserves_scanner_keys():
    variants = (r"K:\R&D\R&D\SEQUENCES", "K:\\R&D\\R&D\\SEQUENCES\\",
                r"k:\r&d\r&d\sequences")
    assert {canonical_path_key(value).rstrip("\\/") for value in variants} == {
        r"k:\r&d\r&d\sequences"}
    assert all(path_key(value) == canonical_path_key(value) for value in variants)
    assert canonical_path_key(r"\\server\share\SEQUENCES") == r"\\server\share\sequences"
    assert sql_path_prefix("K:\\100%_done\\") == r"k:\\100\%\_done%"


def test_absolute_strict_scope_resolves_without_global_fallback():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE documents(path_key TEXT)")
    con.execute("INSERT INTO documents VALUES (?)", (r"k:\r&d\r&d\sequences\domino.docx",))
    value = skill_value()
    value["schema_version"] = 2
    value["retrieval"].pop("path_prefixes")
    value["retrieval"].update(mode="strict", scope={"kind": "absolute",
        "path_prefixes": [r"K:\R&D\R&D\SEQUENCES"], "relative_paths": []})
    plan, diagnostics = compile_retrieval_plan(parse_skill(value), con)
    assert diagnostics["scope_available"] is True
    assert diagnostics["requested_path_count"] == diagnostics["resolved_path_count"] == 1
    assert plan.allow_global_fallback is False
    assert all(stage.strategy != "global_hybrid" for stage in plan.stages)
