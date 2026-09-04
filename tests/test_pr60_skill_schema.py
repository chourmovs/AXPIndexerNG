import json

import pytest

from axp_client.skills import SkillStore, SkillValidationError, parse_skill


def skill_value(skill_id="example"):
    return {"schema_version": 1, "id": skill_id, "name": "Example", "description": "",
            "enabled": True, "priority": 50,
            "match": {"identifiers": ["PX-001"], "phrases": [], "keywords": []},
            "retrieval": {"mode": "prefer", "path_prefixes": [], "extensions": [],
                          "temporal_policy": "recent_first", "max_documents": 32},
            "business_context": "", "answer": {"guidance": "", "sections": []},
            "evidence_policy": "strict"}


def test_schema_is_strict_and_rejects_future_versions():
    value = skill_value()
    value["typo"] = True
    with pytest.raises(SkillValidationError, match="unknown field"):
        parse_skill(value)
    value = skill_value()
    value["schema_version"] = 2
    with pytest.raises(SkillValidationError) as caught:
        parse_skill(value)
    assert caught.value.code == "skill_schema_unsupported"


def test_store_isolates_malformed_files_and_hot_reloads(tmp_path):
    (tmp_path / "good.skill.json").write_text(json.dumps(skill_value()))
    (tmp_path / "broken.skill.json").write_text("{")
    store = SkillStore(tmp_path)
    assert [skill.id for skill in store.list_skills()] == ["example"]
    assert store.invalid[0]["file"] == "broken.skill.json"
    value = skill_value()
    value["name"] = "Updated"
    (tmp_path / "good.skill.json").write_text(json.dumps(value))
    assert store.get("example").name == "Updated"
