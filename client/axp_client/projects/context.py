from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectContext:
    key: str
    name: str
    root_path: str
    root_path_key: str
    resolution: str
    matched_identity: str | None


@dataclass(frozen=True)
class ProjectResolution:
    context: ProjectContext | None
    error: str | None = None
    candidates: tuple[dict, ...] = ()

