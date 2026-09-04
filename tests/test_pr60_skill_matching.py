from axp_client.skills import SkillEngine, SkillStore

from test_pr60_skill_schema import skill_value


def _engine(tmp_path, values):
    import json
    tmp_path.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values):
        (tmp_path / f"{index}.skill.json").write_text(json.dumps(value))
    return SkillEngine(SkillStore(tmp_path))


def test_identifier_auto_match_and_manual_none(tmp_path):
    engine = _engine(tmp_path, [skill_value("project")])
    execution = engine.resolve("Summarize PX-001 development")
    assert execution.skill.id == "project"
    assert execution.match_reason == "identifier"
    assert engine.resolve("Summarize PX-001 development", "none").skill is None


def test_weak_keyword_does_not_hijack_and_equal_matches_are_ambiguous(tmp_path):
    first, second = skill_value("first"), skill_value("second")
    first["match"] = second["match"] = {"identifiers": [], "phrases": ["project x"], "keywords": []}
    engine = _engine(tmp_path, [first, second])
    ambiguous = engine.resolve("Tell me about Project X")
    assert ambiguous.skill is None
    assert ambiguous.diagnostics["match"] == "ambiguous_skill_match"
    weak = skill_value("weak")
    weak["match"] = {"identifiers": [], "phrases": [], "keywords": ["process"]}
    assert _engine(tmp_path / "weak", [weak]).resolve("What is the process?").skill is None
