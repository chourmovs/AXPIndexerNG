"""Validated, immutable Skill schema v1 runtime objects."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SKILL_SCHEMA_VERSION = 1
SKILL_FILE_MAX_BYTES = 64 * 1024
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_EXTENSION = re.compile(r"^\.[a-z0-9][a-z0-9._+-]{0,15}$")


class SkillValidationError(ValueError):
    def __init__(self, detail: str, code: str = "skill_invalid"):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SkillMatchSpec:
    identifiers: tuple[str, ...]
    phrases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class SkillRetrievalSpec:
    mode: str
    path_prefixes: tuple[str, ...]
    extensions: tuple[str, ...]
    temporal_policy: str
    max_documents: int


@dataclass(frozen=True)
class SkillSection:
    title: str
    guidance: str
    required: bool


@dataclass(frozen=True)
class SkillSpec:
    schema_version: int
    id: str
    name: str
    description: str
    enabled: bool
    priority: int
    match: SkillMatchSpec
    retrieval: SkillRetrievalSpec
    business_context: str
    answer_guidance: str
    answer_sections: tuple[SkillSection, ...]
    evidence_policy: str

    def to_dict(self):
        value = asdict(self)
        value["answer"] = {"guidance": value.pop("answer_guidance"),
                           "sections": value.pop("answer_sections")}
        return value


def _object(value, field, keys):
    if not isinstance(value, dict):
        raise SkillValidationError(f"{field} must be an object")
    unknown = sorted(set(value) - set(keys))
    missing = sorted(set(keys) - set(value))
    if unknown:
        raise SkillValidationError(f"{field} contains unknown field '{unknown[0]}'")
    if missing:
        raise SkillValidationError(f"{field}.{missing[0]} is required")


def _string(value, field, maximum, *, empty=True):
    if not isinstance(value, str):
        raise SkillValidationError(f"{field} must be a string")
    if not empty and not value.strip():
        raise SkillValidationError(f"{field} must not be empty")
    if len(value) > maximum:
        raise SkillValidationError(f"{field} must be at most {maximum} characters")
    return value.strip()


def _strings(value, field, maximum_count, maximum_length=256):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SkillValidationError(f"{field} must be an array of strings")
    if len(value) > maximum_count:
        raise SkillValidationError(f"{field} must contain at most {maximum_count} values")
    result = []
    for item in value:
        cleaned = _string(item, field, maximum_length, empty=False)
        if cleaned in result:
            raise SkillValidationError(f"{field} contains a duplicate value")
        result.append(cleaned)
    return tuple(result)


def parse_skill(value) -> SkillSpec:
    top = ("schema_version", "id", "name", "description", "enabled", "priority", "match",
           "retrieval", "business_context", "answer", "evidence_policy")
    _object(value, "skill", top)
    version = value["schema_version"]
    if type(version) is not int or version != SKILL_SCHEMA_VERSION:
        raise SkillValidationError("unsupported skill schema_version", "skill_schema_unsupported")
    skill_id = _string(value["id"], "id", 64, empty=False)
    if not _ID.fullmatch(skill_id):
        raise SkillValidationError("id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    if type(value["enabled"]) is not bool:
        raise SkillValidationError("enabled must be a boolean")
    if type(value["priority"]) is not int or not -1000 <= value["priority"] <= 1000:
        raise SkillValidationError("priority must be an integer from -1000 to 1000")

    match = value["match"]
    _object(match, "match", ("identifiers", "phrases", "keywords"))
    match_spec = SkillMatchSpec(*(_strings(match[key], f"match.{key}", 64) for key in
                                  ("identifiers", "phrases", "keywords")))

    retrieval = value["retrieval"]
    _object(retrieval, "retrieval", ("mode", "path_prefixes", "extensions", "temporal_policy",
                                      "max_documents"))
    if retrieval["mode"] not in ("prefer", "strict"):
        raise SkillValidationError("retrieval.mode must be 'prefer' or 'strict'")
    if retrieval["temporal_policy"] not in ("recent_first", "all_history"):
        raise SkillValidationError("retrieval.temporal_policy must be 'recent_first' or 'all_history'")
    maximum = retrieval["max_documents"]
    if type(maximum) is not int or not 16 <= maximum <= 48:
        raise SkillValidationError("retrieval.max_documents must be an integer from 16 to 48")
    paths = _strings(retrieval["path_prefixes"], "retrieval.path_prefixes", 32, 1000)
    extensions = tuple(item.casefold() for item in
                       _strings(retrieval["extensions"], "retrieval.extensions", 32, 16))
    if any(not _EXTENSION.fullmatch(item) for item in extensions):
        raise SkillValidationError("retrieval.extensions values must be lowercase extensions beginning with '.'")
    retrieval_spec = SkillRetrievalSpec(retrieval["mode"], paths, extensions,
                                        retrieval["temporal_policy"], maximum)

    answer = value["answer"]
    _object(answer, "answer", ("guidance", "sections"))
    sections = answer["sections"]
    if not isinstance(sections, list) or len(sections) > 8:
        raise SkillValidationError("answer.sections must be an array with at most 8 sections")
    parsed_sections = []
    for index, section in enumerate(sections):
        field = f"answer.sections[{index}]"
        _object(section, field, ("title", "guidance", "required"))
        if type(section["required"]) is not bool:
            raise SkillValidationError(f"{field}.required must be a boolean")
        parsed_sections.append(SkillSection(_string(section["title"], f"{field}.title", 120, empty=False),
                                            _string(section["guidance"], f"{field}.guidance", 500),
                                            section["required"]))
    if value["evidence_policy"] != "strict":
        raise SkillValidationError("evidence_policy must be 'strict'")
    return SkillSpec(version, skill_id, _string(value["name"], "name", 120, empty=False),
                     _string(value["description"], "description", 1000), value["enabled"], value["priority"],
                     match_spec, retrieval_spec, _string(value["business_context"], "business_context", 2000),
                     _string(answer["guidance"], "answer.guidance", 1500), tuple(parsed_sections), "strict")
