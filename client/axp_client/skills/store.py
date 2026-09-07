"""Thread-safe filesystem-backed Skill registry with observable reload state."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import SKILL_FILE_MAX_BYTES, SkillSpec, SkillValidationError, parse_skill

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillStoreStatus:
    path: str
    directory_exists: bool
    readable: bool
    valid_count: int
    invalid_count: int
    last_reload_ms: int | None
    reload_generation: int
    error: str | None

    def to_dict(self):
        return asdict(self)


class SkillStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._fingerprint = None
        self._skills: dict[str, SkillSpec] = {}
        self._invalid: tuple[dict, ...] = ()
        self._generation = 0
        self._last_reload_ms = None
        self._error = None
        try:
            self.path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._error = str(exc)
        LOGGER.info("Skill store initialized path=%s", self.path)
        self.force_reload()

    def _current_fingerprint(self):
        if not self.path.is_dir():
            raise OSError(f"Skill directory is unavailable: {self.path}")
        entries = []
        for item in sorted(self.path.glob("*.skill.json")):
            stat = item.stat()
            entries.append((item.name, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
                            getattr(stat, "st_ino", 0)))
        return tuple(entries)

    def _reload(self, forced=False):
        with self._lock:
            try:
                fingerprint = self._current_fingerprint()
            except OSError as exc:
                self._error = str(exc)
                self._generation += bool(forced)
                self._last_reload_ms = int(time.time() * 1000)
                return False
            if not forced and fingerprint == self._fingerprint:
                return False
            skills, invalid, by_id = {}, [], {}
            for entry in fingerprint:
                filename, _mtime, _ctime, size, _inode = entry
                path = self.path / filename
                try:
                    if size > SKILL_FILE_MAX_BYTES:
                        raise SkillValidationError("Skill JSON file must be at most 64 KiB")
                    skill = parse_skill(json.loads(path.read_text(encoding="utf-8")))
                    by_id.setdefault(skill.id, []).append((filename, skill))
                except SkillValidationError as exc:
                    invalid.append({"file": filename, "error": exc.code, "detail": exc.detail})
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    invalid.append({"file": filename, "error": "skill_invalid", "detail": str(exc)})
            for skill_id, entries in sorted(by_id.items()):
                if len(entries) > 1:
                    invalid.extend({"file": filename, "error": "skill_invalid",
                                    "detail": f"duplicate skill id '{skill_id}'"}
                                   for filename, _skill in entries)
                else:
                    skills[skill_id] = entries[0][1]
            self._skills = skills
            self._invalid = tuple(sorted(invalid, key=lambda item: item["file"]))
            self._fingerprint = fingerprint
            self._error = None
            self._generation += 1
            self._last_reload_ms = int(time.time() * 1000)
            LOGGER.info("Skill store loaded valid=%s invalid=%s", len(skills), len(invalid))
            for item in self._invalid:
                LOGGER.warning("Invalid Skill file=%s code=%s detail=%s", item["file"],
                               item["error"], item["detail"])
            return True

    def reload_if_changed(self):
        return self._reload()

    def force_reload(self):
        with self._lock:
            self._fingerprint = None
        self._reload(forced=True)
        return self.snapshot()

    def list_skills(self):
        self.reload_if_changed()
        with self._lock:
            return tuple(self._skills[key] for key in sorted(self._skills))

    def get(self, skill_id):
        self.reload_if_changed()
        with self._lock:
            return self._skills.get(skill_id)

    @property
    def invalid(self):
        self.reload_if_changed()
        with self._lock:
            return tuple(dict(item) for item in self._invalid)

    def status(self):
        self.reload_if_changed()
        with self._lock:
            exists = self.path.is_dir()
            readable = exists and os.access(self.path, os.R_OK)
            return SkillStoreStatus(str(self.path), exists, readable, len(self._skills),
                                    len(self._invalid), self._last_reload_ms,
                                    self._generation, self._error)

    def snapshot(self):
        skills = self.list_skills()
        return {"status": self.status().to_dict(), "skills": skills, "invalid": self.invalid}
