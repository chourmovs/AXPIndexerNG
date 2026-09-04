"""Filesystem-backed Skill registry with polling-based hot reload."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import SKILL_FILE_MAX_BYTES, SkillSpec, SkillValidationError, parse_skill


class SkillStore:
    def __init__(self, path):
        self.path = Path(path)
        self._fingerprint = None
        self._skills: dict[str, SkillSpec] = {}
        self._invalid: list[dict] = []

    def _current_fingerprint(self):
        try:
            return tuple((item.name, item.stat().st_mtime_ns, item.stat().st_size)
                         for item in sorted(self.path.glob("*.skill.json")))
        except OSError:
            return ()

    def reload_if_changed(self):
        fingerprint = self._current_fingerprint()
        if fingerprint == self._fingerprint:
            return False
        skills, invalid, by_id = {}, [], {}
        for filename, _mtime, size in fingerprint:
            path = self.path / filename
            try:
                if size > SKILL_FILE_MAX_BYTES:
                    raise SkillValidationError("Skill JSON file must be at most 64 KiB")
                value = json.loads(path.read_text(encoding="utf-8"))
                skill = parse_skill(value)
                by_id.setdefault(skill.id, []).append((filename, skill))
            except SkillValidationError as exc:
                invalid.append({"file": filename, "error": exc.code, "detail": exc.detail})
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                invalid.append({"file": filename, "error": "skill_invalid", "detail": str(exc)})
        for skill_id, entries in sorted(by_id.items()):
            if len(entries) > 1:
                for filename, _skill in entries:
                    invalid.append({"file": filename, "error": "skill_invalid",
                                    "detail": f"duplicate skill id '{skill_id}'"})
            else:
                filename, skill = entries[0]
                skills[skill_id] = skill
        self._skills, self._invalid, self._fingerprint = skills, sorted(invalid, key=lambda x: x["file"]), fingerprint
        return True

    def list_skills(self):
        self.reload_if_changed()
        return tuple(self._skills[key] for key in sorted(self._skills))

    def get(self, skill_id):
        self.reload_if_changed()
        return self._skills.get(skill_id)

    @property
    def invalid(self):
        self.reload_if_changed()
        return tuple(dict(item) for item in self._invalid)
