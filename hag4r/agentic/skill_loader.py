from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from hag4r.agentic.state import is_subpath


_SKILL_SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
RUNTIME_SKILL_SUITE_NAME = "run-diaggen-pipeline"
RUNTIME_SKILL_ROOT_REL = Path(".agents") / "skills" / RUNTIME_SKILL_SUITE_NAME
RUNTIME_SKILL_VIRTUAL_ROOT = f"/.agents/skills/{RUNTIME_SKILL_SUITE_NAME}/"
RUNTIME_SKILL_READ_GLOB = f"{RUNTIME_SKILL_VIRTUAL_ROOT}**"


@dataclass(frozen=True)
class RuntimeSkill:
    name: str
    path: Path
    skill_dir: Path
    content: str
    sha256: str
    title: str
    virtual_path: str
    virtual_skill_dir: str
    skill_suite: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "skill_dir": str(self.skill_dir),
            "virtual_path": self.virtual_path,
            "virtual_skill_dir": self.virtual_skill_dir,
            "sha256": self.sha256,
            "title": self.title,
            "skill_suite": self.skill_suite,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_source_title(content: str, fallback: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.startswith("name:"):
                title = stripped.split(":", 1)[1].strip().strip("'\"")
                if title:
                    return title
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def _validate_skill_source_name(source_kind: str, source_name: str) -> None:
    if not _SKILL_SOURCE_NAME_RE.fullmatch(source_name):
        raise ValueError(f"Invalid HAG4R {source_kind} name: {source_name!r}")


def runtime_skill_root(repo_root: Path | None = None) -> Path:
    root = (repo_root or _repo_root()).resolve()
    return (root / RUNTIME_SKILL_ROOT_REL).resolve()


def runtime_skill_virtual_path(path: Path, *, repo_root: Path | None = None) -> str:
    root = (repo_root or _repo_root()).resolve()
    resolved_path = path.resolve()
    relative = resolved_path.relative_to(root)
    return "/" + relative.as_posix()


def _runtime_skill_dir_name(skill_name: str) -> str:
    return skill_name.replace("_", "-")


def _build_runtime_skill(
    *,
    name: str,
    skill_dir: Path,
    skill_path: Path,
    root: Path,
    skills_root: Path,
) -> RuntimeSkill:
    if not is_subpath(skill_dir, skills_root):
        raise ValueError(f"Refusing to load skill directory outside canonical HAG4R runtime skill suite root: {skill_dir}")
    if not is_subpath(skill_dir, root):
        raise ValueError(f"Refusing to load skill directory outside HAG4R repo root: {skill_dir}")
    content = _read_skill_source(skill_path, root=root, source_root=skills_root, source_kind="skill")
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return RuntimeSkill(
        name=name,
        path=skill_path,
        skill_dir=skill_dir,
        content=content,
        sha256=sha256,
        title=_extract_source_title(content, fallback=name),
        virtual_path=runtime_skill_virtual_path(skill_path, repo_root=root),
        virtual_skill_dir=runtime_skill_virtual_path(skill_dir, repo_root=root),
        skill_suite=RUNTIME_SKILL_SUITE_NAME,
    )


def _read_skill_source(path: Path, *, root: Path, source_root: Path, source_kind: str) -> str:
    if not is_subpath(path, source_root):
        raise ValueError(f"Refusing to load {source_kind} outside canonical HAG4R runtime {source_kind} root: {path}")
    if not is_subpath(path, root):
        raise ValueError(f"Refusing to load {source_kind} outside HAG4R repo root: {path}")
    if not path.exists():
        raise FileNotFoundError(f"HAG4R {source_kind} is missing under canonical root {source_root}: {path}")
    if not path.is_file():
        raise ValueError(f"HAG4R {source_kind} path is not a file: {path}")

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"HAG4R {source_kind} is empty: {path}")
    return content


def load_stage_skill(skill_name: str, *, repo_root: Path | None = None) -> RuntimeSkill:
    _validate_skill_source_name("skill", skill_name)

    root = (repo_root or _repo_root()).resolve()
    skills_root = runtime_skill_root(root)
    skill_dir = (skills_root / _runtime_skill_dir_name(skill_name)).resolve()
    skill_path = (skill_dir / "SKILL.md").resolve()
    return _build_runtime_skill(
        name=skill_name,
        skill_dir=skill_dir,
        skill_path=skill_path,
        root=root,
        skills_root=skills_root,
    )


def load_pipeline_skill(*, repo_root: Path | None = None) -> RuntimeSkill:
    root = (repo_root or _repo_root()).resolve()
    skills_root = runtime_skill_root(root)
    skill_path = (skills_root / "SKILL.md").resolve()
    return _build_runtime_skill(
        name=RUNTIME_SKILL_SUITE_NAME,
        skill_dir=skills_root,
        skill_path=skill_path,
        root=root,
        skills_root=skills_root,
    )


def load_stage_mini_skill(
    parent_skill_name: str,
    mini_skill_name: str,
    *,
    repo_root: Path | None = None,
) -> RuntimeSkill:
    _validate_skill_source_name("skill", parent_skill_name)
    _validate_skill_source_name("skill", mini_skill_name)

    root = (repo_root or _repo_root()).resolve()
    skills_root = runtime_skill_root(root)
    parent_dir = (skills_root / _runtime_skill_dir_name(parent_skill_name)).resolve()
    skill_dir = (parent_dir / _runtime_skill_dir_name(mini_skill_name)).resolve()
    skill_path = (skill_dir / "SKILL.md").resolve()
    if not is_subpath(skill_dir, parent_dir):
        raise ValueError(f"Refusing to load mini skill outside parent skill directory: {skill_dir}")
    if not is_subpath(skill_dir, skills_root):
        raise ValueError(f"Refusing to load mini skill outside canonical HAG4R runtime skill suite root: {skill_dir}")
    if not is_subpath(skill_dir, root):
        raise ValueError(f"Refusing to load mini skill outside HAG4R repo root: {skill_dir}")
    return _build_runtime_skill(
        name=f"{parent_skill_name}/{mini_skill_name}",
        skill_dir=skill_dir,
        skill_path=skill_path,
        root=root,
        skills_root=skills_root,
    )


__all__ = [
    "RuntimeSkill",
    "RUNTIME_SKILL_READ_GLOB",
    "RUNTIME_SKILL_ROOT_REL",
    "RUNTIME_SKILL_SUITE_NAME",
    "RUNTIME_SKILL_VIRTUAL_ROOT",
    "load_pipeline_skill",
    "load_stage_mini_skill",
    "load_stage_skill",
    "runtime_skill_root",
    "runtime_skill_virtual_path",
]
