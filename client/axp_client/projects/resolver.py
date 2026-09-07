"""Conservative metadata-only project resolution from indexed paths."""
from __future__ import annotations

import re
from pathlib import PureWindowsPath

from .context import ProjectContext, ProjectResolution

_WORDS = re.compile(r"[\w-]+", re.UNICODE)
_STOP = {"montre", "moi", "montre-moi", "la", "le", "les", "du", "de", "des", "un", "une", "dans",
         "résume", "resume", "projet", "project", "show", "me", "the", "for", "summarize"}


def _tokens(value):
    return tuple(word.casefold() for word in _WORDS.findall(value))


class ProjectResolver:
    def resolve(self, con, question, *, relative_paths, skill_match=None):
        anchors = [tuple(part.casefold() for part in path.replace("/", "\\").split("\\") if part)
                   for path in relative_paths]
        candidates = {}
        for row in con.execute("SELECT path, path_key FROM documents").fetchall():
            path = row["path"] if hasattr(row, "keys") else row[0]
            parts = PureWindowsPath(path).parts
            folded = tuple(part.casefold() for part in parts)
            for anchor in anchors:
                for index in range(len(folded) - len(anchor)):
                    if folded[index:index + len(anchor)] != anchor or index == 0:
                        continue
                    root_parts = parts[:index]
                    root = str(PureWindowsPath(*root_parts))
                    key = root.casefold().rstrip("\\")
                    candidates[key] = ProjectContext(key, root_parts[-1], root, key,
                                                     "indexed_path", None)
        if not candidates:
            return ProjectResolution(None, "project_not_found")
        consumed = set(_STOP)
        if skill_match:
            for values in (skill_match.identifiers, skill_match.phrases, skill_match.keywords):
                for value in values:
                    consumed.update(_tokens(value))
        identity = tuple(token for token in _tokens(question) if token not in consumed)
        values = tuple(candidates.values())
        if not identity:
            if len(values) == 1:
                return ProjectResolution(values[0])
            return ProjectResolution(None, "project_required",
                                     tuple({"name": item.name} for item in values))
        ranked = []
        for item in values:
            segments = _tokens(item.root_path)
            name_tokens = _tokens(item.name)
            coverage = sum(token in segments for token in identity)
            rank = (int(any(token in segments for token in identity)),
                    int(tuple(identity) == name_tokens), int(coverage == len(identity)), coverage, 1)
            ranked.append((rank, item))
        best = max(rank for rank, _item in ranked)
        winners = [item for rank, item in ranked if rank == best]
        if best[0] == 0:
            return ProjectResolution(None, "project_not_found")
        if len(winners) != 1:
            return ProjectResolution(None, "project_ambiguous",
                                     tuple({"name": item.name} for item in winners))
        winner = winners[0]
        matched = " ".join(token for token in identity if token in _tokens(winner.root_path))
        return ProjectResolution(ProjectContext(winner.key, winner.name, winner.root_path,
                                                winner.root_path_key, winner.resolution, matched))
