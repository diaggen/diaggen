from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from hag4r.agentic.state import (
    ArtifactRef,
    ArtifactRole,
    PreflightReport,
    Stage,
    StageRunResult,
    preflight_paths,
)


VIEW_NAMES = ("top", "bottom", "front", "back", "left", "right")


class EnvName:
    ORCHESTRATOR = "hag4r"
    HAG4R_TOOLS = "hag4r_mesh"
    OMNIPART = "omnipart"
    SEGMENTATION = "hag4r_segmentation"
    GENESIS = "genesis_world-main"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path, repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root()
    return path if path.is_absolute() else root / path


def _path_for_cwd(path: Path, *, cwd: Path, repo_root: Path) -> Path:
    resolved = _path_for_repo(path, repo_root)
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return resolved
    return Path(os.path.relpath(resolved, cwd))


def _conda_env_args(env: str, repo_root: Path | None = None) -> tuple[str, str]:
    root = repo_root or _repo_root()
    explicit_prefix = Path(env)
    if explicit_prefix.is_absolute() or len(explicit_prefix.parts) > 1:
        return ("-p", str(explicit_prefix))

    local_prefix = root / ".conda" / env
    if local_prefix.exists():
        return ("-p", str(local_prefix))
    return ("-n", env)


def resolve_conda_env(env: str, repo_root: Path | None = None) -> str:
    return _conda_env_args(env, repo_root=repo_root)[1]


def _conda_run_argv(env: str, command: Iterable[object], repo_root: Path | None = None) -> tuple[str, ...]:
    return ("conda", "run", *_conda_env_args(env, repo_root), *(str(arg) for arg in command))


def _conda_module_argv(
    env: str,
    module: str,
    args: Iterable[object],
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    return _conda_run_argv(env, ("python", "-m", module, *args), repo_root=repo_root)


def _artifact(
    role: ArtifactRole,
    path: Path,
    stage: Stage,
    description: str = "",
    required: bool = True,
) -> ArtifactRef:
    return ArtifactRef(role=role, path=path, stage=stage, description=description, required=required)


def _verify_artifacts(artifacts: tuple[ArtifactRef, ...]) -> str:
    missing = [str(artifact.path) for artifact in artifacts if artifact.required and not artifact.path.exists()]
    return ", ".join(missing)


def preflight_command(
    *,
    name: str,
    stage: Stage,
    cwd: Path,
    read_paths: tuple[Path, ...] = (),
    write_paths: tuple[Path, ...] = (),
    conda_env: str | None = None,
    repo_root: Path | None = None,
    allow_external_writes: bool = False,
) -> PreflightReport:
    root = repo_root or _repo_root()
    return preflight_paths(
        name=name,
        stage=stage,
        cwd=cwd,
        repo_root=root,
        read_paths=read_paths,
        write_paths=write_paths,
        conda_env=conda_env,
        allow_external_writes=allow_external_writes,
    )


def _run_subprocess_stage(
    *,
    name: str,
    stage: Stage,
    argv: tuple[str, ...],
    cwd: Path,
    expected_artifacts: tuple[ArtifactRef, ...],
    read_paths: tuple[Path, ...] = (),
    write_paths: tuple[Path, ...] = (),
    conda_env: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int = 3600,
    log_dir: Path | None = None,
    repo_root: Path | None = None,
    allow_external_writes: bool = False,
) -> StageRunResult:
    root = repo_root or _repo_root()
    report = preflight_command(
        name=name,
        stage=stage,
        cwd=cwd,
        read_paths=read_paths,
        write_paths=write_paths,
        conda_env=conda_env,
        repo_root=root,
        allow_external_writes=allow_external_writes,
    )
    if not report.ok:
        message = "; ".join(issue.message for issue in report.issues)
        return StageRunResult(name=name, stage=stage, success=False, artifacts=expected_artifacts, error=message)

    resolved_log_dir = log_dir or root / "outputs/agentic_asset_refinement/logs"
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = resolved_log_dir / f"{name}.stdout.log"
    stderr_path = resolved_log_dir / f"{name}.stderr.log"
    cmd_path = resolved_log_dir / f"{name}.cmd.txt"
    cmd_path.write_text(" ".join(argv) + "\n", encoding="utf-8")

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    missing = _verify_artifacts(expected_artifacts) if completed.returncode == 0 else ""
    success = completed.returncode == 0 and not missing
    error = ""
    if completed.returncode != 0:
        error = f"{name} exited with return code {completed.returncode}"
    elif missing:
        error = f"{name} did not produce required artifact(s): {missing}"
    return StageRunResult(
        name=name,
        stage=stage,
        success=success,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        artifacts=expected_artifacts,
        error=error,
    )


def _dict_stage_result(
    *,
    name: str,
    stage: Stage,
    payload: dict,
    artifacts: tuple[ArtifactRef, ...],
) -> StageRunResult:
    missing = _verify_artifacts(artifacts)
    success = not missing
    return StageRunResult(
        name=name,
        stage=stage,
        success=success,
        artifacts=artifacts,
        metrics={"payload": payload},
        error=f"missing required artifact(s): {missing}" if missing else "",
    )


__all__ = [
    "EnvName",
    "VIEW_NAMES",
    "_artifact",
    "_conda_env_args",
    "_conda_module_argv",
    "_conda_run_argv",
    "_dict_stage_result",
    "_path_for_cwd",
    "_path_for_repo",
    "_repo_root",
    "_run_subprocess_stage",
    "_verify_artifacts",
    "preflight_command",
    "resolve_conda_env",
]
