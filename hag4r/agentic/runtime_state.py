from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import stat
import threading
from functools import wraps
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hag4r.agentic.diagnostic_cues import (
    DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION,
    build_diagnostic_repair_brief,
)
from hag4r.agentic.genesis_vlm_schemas import normalize_persisted_vlm_final_recommendation
from hag4r.agentic.run_config import (
    RunConfig,
    run_config_audit_state,
)
from hag4r.agentic.skill_loader import load_pipeline_skill, runtime_skill_root
from hag4r.tools.genesis.config import DEFAULT_GENESIS_ROOT
from hag4r.tools.genesis.diagnostic_timing import DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
from hag4r.tools.genesis.live_protocol import DEFAULT_READY_TIMEOUT_S
from hag4r.tools.common import VIEW_NAMES


RuntimeState = dict[str, Any]

RUNTIME_STATE_SCHEMA_VERSION = "2026-07-skill-suite-runtime-state-v2"
WORKER_ISOLATION_SCHEMA_VERSION = "hag4r-worker-isolation-v1"
DEFAULT_RUNS_ROOT = Path("outputs/agentic_asset_refinement")
DIAGNOSTICS_SNAPSHOT_CANONICAL_INPUT_PATH_KEYS: tuple[str, ...] = (
    "inferred_params_path",
    "monolithic_mesh_path",
    "mesh_processing_request_path",
)
SLURM_GPU_BINDING_ENV_KEYS: tuple[str, ...] = (
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_LOCALID",
    "SLURM_PROCID",
    "SLURM_STEP_ID",
    "SLURM_JOB_ID",
)
SLURM_GPU_BINDING_VALUE_KEYS: tuple[str, ...] = (
    "SLURM_STEP_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_LOCALID",
)

FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE: tuple[str, ...] = (
    "image_cleanup",
    "sam3_omnipart_2d_segmentation",
    "omnipart_generate_parts",
    "hag4r_gpt_staged_material_inference",
    "hag4r_assign_params_to_prims",
    "hag4r_combined_to_monolithic",
)
DIAGNOSTIC_STAGE_SEQUENCE: tuple[str, ...] = ("genesis_live_diagnostic_loop",)
POST_MESH_TEXTURE_STAGE_SEQUENCE: tuple[str, ...] = ("hag4r_post_mesh_texture",)
FINAL_EXPORT_STAGE_SEQUENCE: tuple[str, ...] = ("hag4r_final_export_bundle",)
FULL_IMAGE_STAGE_SEQUENCE: tuple[str, ...] = (
    *FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE,
    *POST_MESH_TEXTURE_STAGE_SEQUENCE,
    *FINAL_EXPORT_STAGE_SEQUENCE,
)
FULL_IMAGE_THROUGH_IMAGE_CLEANUP_STAGE_SEQUENCE: tuple[str, ...] = (
    FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE[:1]
)
FULL_IMAGE_THROUGH_OMNIPART_STAGE_SEQUENCE: tuple[str, ...] = (
    FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE[:3]
)
FULL_IMAGE_THROUGH_MATERIAL_STAGE_SEQUENCE: tuple[str, ...] = (
    FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE[:4]
)
DIAGNOSTICS_SNAPSHOT_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    # Import-only schema migration; this is not an active runtime stage name.
    "image_cleanup": ("hag4r_" + "vlm_image_cleanup",),
}
POST_MESH_PROCESSING_DIAGNOSTICS_STAGE_SEQUENCE: tuple[str, ...] = (
    *DIAGNOSTIC_STAGE_SEQUENCE,
    *POST_MESH_TEXTURE_STAGE_SEQUENCE,
    *FINAL_EXPORT_STAGE_SEQUENCE,
)
POST_MESH_PROCESSING_DIAGNOSTICS_REROUTE_AND_EXPORT_STAGE_SEQUENCE: tuple[str, ...] = (
    *FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE,
    *DIAGNOSTIC_STAGE_SEQUENCE,
    *POST_MESH_TEXTURE_STAGE_SEQUENCE,
    *FINAL_EXPORT_STAGE_SEQUENCE,
)
BYPASS_DIAGNOSTICS_ROUTE = "bypass_diagnostics"
PIPELINE_SKILL_ROOT = ".agents/skills/run-diaggen-pipeline"
STAGE_SKILL_PATHS: dict[str, str] = {
    "image_cleanup": f"{PIPELINE_SKILL_ROOT}/image-cleanup/SKILL.md",
    "segmentation": f"{PIPELINE_SKILL_ROOT}/segmentation/SKILL.md",
    "omnipart": f"{PIPELINE_SKILL_ROOT}/omnipart/SKILL.md",
    "material_inference": f"{PIPELINE_SKILL_ROOT}/material-inference/SKILL.md",
    "mesh_processing": f"{PIPELINE_SKILL_ROOT}/mesh-processing/SKILL.md",
    "genesis_diagnostics": f"{PIPELINE_SKILL_ROOT}/genesis-diagnostics/SKILL.md",
    "post_mesh_texture": f"{PIPELINE_SKILL_ROOT}/post-mesh-texture/SKILL.md",
    "final_export": f"{PIPELINE_SKILL_ROOT}/final-export/SKILL.md",
}

RUN_MODES = frozenset({"full_image", "post_mesh_processing_diagnostics"})
TERMINATE_AFTER_STAGE_OPTIONS = frozenset({"image_cleanup", "omnipart", "material_inference"})
REVISION_AWARE_RUN_MODES = frozenset({"full_image", "post_mesh_processing_diagnostics"})
RUN_STATUSES = frozenset({"seeded", "running", "success", "failed"})
STATE_STATUSES = RUN_STATUSES
DIAGNOSTIC_STAGE_STATUSES = frozenset(
    {"pending", "pending_retry", "retryable", "running", "success", "failed", "halted"}
)
DIAGNOSTIC_VERDICTS = frozenset({"accept", "revise"})
TRANSITION_ACTIONS = frozenset(
    {
        "post_mesh_texture",
        "final_export",
        "reroute",
        "report_only",
        "retry_diagnostics",
        "retry_diagnostics_cleanup",
        "halt",
    }
)
REVISION_STATUSES = frozenset(
    {
        "active",
        "diagnostics_accepted",
        "ready_for_export",
        "ready_for_export_unaccepted",
        "accepted",
        "exported_unaccepted",
        "superseded",
    }
)
DIAGNOSTIC_EXPORT_POLICY = "export_after_revision_budget_revise"
LEGACY_RUNTIME_CONTRACT_KEYS = frozenset(
    {
        "representation",
        "expected_" + "representation",
        "representation_" + "decision",
        "representation_" + "status",
        "representation_" + "audit",
        "shell_" + "thickness_m",
        "wall_" + "thickness_m",
        "surface_processing_audit_path",
        "final_homogeneous_params_path",
        "uniform_params_path",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_profile_event_fail_open(
    run_root: str | Path,
    event: str,
    *,
    stage: str = "",
    operation: str = "",
    status: str = "",
) -> None:
    try:
        from hag4r.agentic.runtime_profiler import record_event, record_tool_status

        if event == "tool_status":
            record_tool_status(
                run_root,
                stage=stage,
                tool_name=operation,
                status=status,
            )
        else:
            record_event(
                run_root,
                event,
                stage=stage,
                operation=operation,
                status=status,
            )
    except Exception:
        # Profiling is observability-only and must never affect pipeline state.
        return


def _to_json_friendly(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_json_friendly(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_to_json_friendly(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_friendly(item) for key, item in value.items()}
    return str(value)


def _resolve_read_path(path: str | Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _resolve_write_path(path: str | Path, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_write_path(path: Path, *, repo_root: Path, run_root: Path | None = None) -> None:
    outputs_root = repo_root / "outputs"
    if _is_subpath(path, outputs_root):
        return
    if run_root is not None and _is_subpath(path, run_root):
        return
    raise ValueError(f"Refusing runtime-state write path outside HAG4R outputs or run root: {path}")


def _nonempty_env_value(environ: Mapping[str, str], key: str) -> str | None:
    value = environ.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_worker_gpu_binding(environ: Mapping[str, str]) -> dict[str, Any]:
    cuda_visible_devices = _nonempty_env_value(environ, "CUDA_VISIBLE_DEVICES")
    slurm = {
        key: value
        for key in SLURM_GPU_BINDING_ENV_KEYS
        if (value := _nonempty_env_value(environ, key)) is not None
    }
    if cuda_visible_devices is not None:
        source = "env"
        value = cuda_visible_devices
        effective_cuda_visible_devices = cuda_visible_devices
    elif slurm:
        source = "slurm"
        value = next((slurm[key] for key in SLURM_GPU_BINDING_VALUE_KEYS if key in slurm), None)
        effective_cuda_visible_devices = None
    else:
        source = "unbound"
        value = None
        effective_cuda_visible_devices = None
    gpu_binding: dict[str, Any] = {
        "source": source,
        "value": value,
        "effective_cuda_visible_devices": effective_cuda_visible_devices,
        "conflict_checked": True,
    }
    if slurm:
        gpu_binding["slurm"] = slurm
    return gpu_binding


def validate_single_cuda_visible_devices(value: str, *, source: str) -> str:
    """Return one canonical CUDA visibility token for diagnostics."""
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string containing exactly one GPU token")
    token = value.strip()
    if not token:
        raise ValueError(f"{source} must contain exactly one non-empty GPU token")
    if "," in token or any(character.isspace() for character in token):
        raise ValueError(
            f"{source} must contain exactly one GPU token without commas or whitespace; "
            "pass one explicit token to the affected GPU-bearing child process"
        )
    return token


def worker_id_from_environment(environ: Mapping[str, str], *, run_id: str) -> str:
    return _nonempty_env_value(environ, "HAG4R_WORKER_ID") or run_id


def _build_worker_isolation(
    *,
    state: RuntimeState,
    worker_id: str,
    artifact_root: str | Path,
    gpu_binding: dict[str, Any],
) -> dict[str, Any]:
    paths = state.get("paths", {})
    diagnostics = state.get("diagnostics", {})
    final_export = state.get("final_export", {})
    final_export_dir = (
        final_export.get("dir")
        if isinstance(final_export, dict) and final_export.get("dir")
        else paths.get("final_export_dir")
        if isinstance(paths, dict)
        else None
    )
    worker_isolation = {
        "schema_version": WORKER_ISOLATION_SCHEMA_VERSION,
        "worker_id": worker_id,
        "run_id": state["run_id"],
        "run_root": str(Path(str(state["run_root"])).expanduser().resolve()),
        "state_path": str(state_path(state["run_root"])),
        "artifact_root": str(Path(artifact_root).expanduser().resolve()),
        "output_tag": state["output_tag"],
        "final_export_dir": str(Path(str(final_export_dir)).expanduser().resolve()) if final_export_dir else None,
        "diagnostic_workspace_dir": str(Path(str(paths["diagnostic_workspace_dir"])).expanduser().resolve()),
        "diagnostic_generated_episodes_dir": str(
            Path(str(paths["diagnostic_generated_episodes_dir"])).expanduser().resolve()
        ),
        "gpu_binding": dict(gpu_binding),
        "genesis_root": str(Path(str(diagnostics["genesis_root"])).expanduser().resolve()),
        "genesis_env_path": diagnostics.get("genesis_env_path"),
        "genesis_live_command": str(diagnostics["genesis_live_command"]),
        "live_launch_evidence_paths": [],
    }
    if worker_isolation["genesis_env_path"] is not None:
        worker_isolation["genesis_env_path"] = str(
            Path(str(worker_isolation["genesis_env_path"])).expanduser().resolve()
        )
    return _to_json_friendly(worker_isolation)


def _worker_path(worker_isolation: dict[str, Any], key: str) -> Path:
    value = worker_isolation.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"worker_isolation.{key} is required")
    return Path(str(value)).expanduser().resolve()


def validate_worker_isolation_paths(state: RuntimeState) -> None:
    worker_isolation = state.get("worker_isolation")
    if not isinstance(worker_isolation, dict):
        raise ValueError("runtime state is missing worker_isolation")
    required_fields = (
        "schema_version",
        "worker_id",
        "run_id",
        "run_root",
        "state_path",
        "artifact_root",
        "output_tag",
        "final_export_dir",
        "diagnostic_workspace_dir",
        "diagnostic_generated_episodes_dir",
        "gpu_binding",
        "genesis_root",
        "genesis_env_path",
        "genesis_live_command",
    )
    missing = [
        field
        for field in required_fields
        if field not in worker_isolation
        or (field not in {"genesis_env_path"} and worker_isolation.get(field) in (None, ""))
    ]
    if missing:
        raise ValueError("worker_isolation missing required field(s): " + ", ".join(missing))
    if worker_isolation["schema_version"] != WORKER_ISOLATION_SCHEMA_VERSION:
        raise ValueError(f"unknown worker_isolation schema_version: {worker_isolation['schema_version']}")
    if str(worker_isolation["run_id"]) != str(state["run_id"]):
        raise ValueError("worker_isolation.run_id must match state.run_id")
    if str(worker_isolation["output_tag"]) != str(state["output_tag"]):
        raise ValueError("worker_isolation.output_tag must match state.output_tag")

    gpu_binding = worker_isolation.get("gpu_binding")
    if not isinstance(gpu_binding, dict):
        raise ValueError("worker_isolation.gpu_binding must be an object")
    if gpu_binding.get("conflict_checked") is not True:
        raise ValueError("worker_isolation.gpu_binding.conflict_checked must be true")

    run_root = Path(str(state["run_root"])).expanduser().resolve()
    if _worker_path(worker_isolation, "run_root") != run_root:
        raise ValueError("worker_isolation.run_root must match state.run_root")
    if _worker_path(worker_isolation, "state_path") != state_path(run_root):
        raise ValueError("worker_isolation.state_path must equal runtime_state.state_path(run_root)")

    paths = state.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("runtime state paths must be an object")
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ValueError("runtime state diagnostics must be an object")
    if _worker_path(worker_isolation, "diagnostic_workspace_dir") != Path(
        str(paths["diagnostic_workspace_dir"])
    ).expanduser().resolve():
        raise ValueError("worker_isolation.diagnostic_workspace_dir must match state.paths")
    if _worker_path(worker_isolation, "diagnostic_generated_episodes_dir") != Path(
        str(paths["diagnostic_generated_episodes_dir"])
    ).expanduser().resolve():
        raise ValueError("worker_isolation.diagnostic_generated_episodes_dir must match state.paths")
    if _worker_path(worker_isolation, "genesis_root") != Path(str(diagnostics["genesis_root"])).expanduser().resolve():
        raise ValueError("worker_isolation.genesis_root must match state.diagnostics.genesis_root")

    worker_genesis_env = worker_isolation.get("genesis_env_path")
    diagnostics_genesis_env = diagnostics.get("genesis_env_path")
    if worker_genesis_env != diagnostics_genesis_env:
        raise ValueError("worker_isolation.genesis_env_path must match state.diagnostics.genesis_env_path")
    if str(worker_isolation["genesis_live_command"]) != str(diagnostics["genesis_live_command"]):
        raise ValueError("worker_isolation.genesis_live_command must match state.diagnostics.genesis_live_command")

    final_export = state.get("final_export", {})
    if isinstance(final_export, dict) and final_export.get("dir"):
        if _worker_path(worker_isolation, "final_export_dir") != Path(str(final_export["dir"])).expanduser().resolve():
            raise ValueError("worker_isolation.final_export_dir must match state.final_export.dir")

    artifact_root = _worker_path(worker_isolation, "artifact_root")
    diagnostic_workspace_dir = _worker_path(worker_isolation, "diagnostic_workspace_dir")
    diagnostic_generated_episodes_dir = _worker_path(worker_isolation, "diagnostic_generated_episodes_dir")
    run_mode = str(state.get("run_mode", ""))
    if run_mode in {"full_image", "post_mesh_processing_diagnostics"}:
        if not _is_subpath(artifact_root, run_root):
            raise ValueError(f"worker_isolation.artifact_root must be under run_root: {artifact_root}")
        for key, path in (
            ("diagnostic_workspace_dir", diagnostic_workspace_dir),
            ("diagnostic_generated_episodes_dir", diagnostic_generated_episodes_dir),
        ):
            if not _is_subpath(path, artifact_root):
                raise ValueError(f"worker_isolation.{key} must be under artifact_root: {path}")


def worker_cuda_visible_devices(state: RuntimeState) -> str | None:
    worker_isolation = state.get("worker_isolation")
    if not isinstance(worker_isolation, dict):
        return None
    gpu_binding = worker_isolation.get("gpu_binding")
    if not isinstance(gpu_binding, dict):
        return None
    value = gpu_binding.get("effective_cuda_visible_devices")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _install_worker_isolation(
    state: RuntimeState,
    *,
    worker_id: str | None,
    artifact_root: str | Path,
    gpu_binding: dict[str, Any] | None,
) -> None:
    resolved_worker_id = worker_id.strip() if worker_id is not None else str(state["run_id"])
    if not resolved_worker_id:
        resolved_worker_id = str(state["run_id"])
    resolved_gpu_binding = dict(gpu_binding) if gpu_binding is not None else build_worker_gpu_binding(os.environ)
    state["worker_isolation"] = _build_worker_isolation(
        state=state,
        worker_id=resolved_worker_id,
        artifact_root=artifact_root,
        gpu_binding=resolved_gpu_binding,
    )
    validate_worker_isolation_paths(state)
    validate_existing_worker_isolation_collision(state)


def _refresh_worker_isolation_for_active_paths(state: RuntimeState) -> None:
    worker_isolation = state.get("worker_isolation")
    if not isinstance(worker_isolation, dict):
        return
    artifact_root = (
        active_revision_root(state)
        if _supports_diagnostic_repair_revisions(state) and state.get("active_revision")
        else Path(str(worker_isolation["artifact_root"])).expanduser().resolve()
    )
    refreshed = _build_worker_isolation(
        state=state,
        worker_id=str(worker_isolation["worker_id"]),
        artifact_root=artifact_root,
        gpu_binding=dict(worker_isolation["gpu_binding"]),
    )
    refreshed["live_launch_evidence_paths"] = list(worker_isolation.get("live_launch_evidence_paths", []))
    state["worker_isolation"] = _to_json_friendly(refreshed)
    validate_worker_isolation_paths(state)


def _worker_collision_field_value(worker_isolation: dict[str, Any], key: str) -> str:
    if key not in worker_isolation or worker_isolation[key] in (None, ""):
        raise ValueError(f"existing worker_isolation is missing collision field: {key}")
    value = worker_isolation[key]
    if key in {"artifact_root", "final_export_dir"}:
        return str(Path(str(value)).expanduser().resolve())
    return str(value)


def validate_existing_worker_isolation_collision(state: RuntimeState) -> None:
    worker_isolation = state.get("worker_isolation")
    if not isinstance(worker_isolation, dict):
        raise ValueError("runtime state is missing worker_isolation")
    path = state_path(state["run_root"])
    if not path.exists():
        return
    existing_state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing_state, dict):
        raise ValueError(f"existing runtime state must be a JSON object: {path}")
    existing_worker_isolation = existing_state.get("worker_isolation")
    if not isinstance(existing_worker_isolation, dict):
        return

    collision_fields = ("worker_id", "run_id", "artifact_root", "output_tag", "final_export_dir")
    differing_fields = [
        field
        for field in collision_fields
        if _worker_collision_field_value(existing_worker_isolation, field)
        != _worker_collision_field_value(worker_isolation, field)
    ]
    if differing_fields:
        raise ValueError(
            "existing runtime state worker_isolation collides with this worker contract: "
            f"state_path={path}; differing field(s): {', '.join(differing_fields)}"
        )


def _resolve_runs_root(runs_root: str | Path, repo_root: Path) -> Path:
    resolved = _resolve_write_path(runs_root, repo_root)
    if not _is_subpath(resolved, repo_root) and not _is_subpath(resolved, repo_root / "outputs"):
        raise ValueError(f"runs_root must be inside the HAG4R repository: {resolved}")
    return resolved


def _resolve_runtime_root_path(run_root: str | Path) -> Path:
    path = Path(run_root).expanduser()
    if path.is_absolute():
        try:
            return (_repo_root() / "outputs" / path.relative_to("/outputs")).resolve()
        except ValueError:
            pass
    return path.resolve()


def state_path(run_root: str | Path) -> Path:
    return _resolve_runtime_root_path(run_root) / "state.json"


def _validate_runtime_status_domains(state: RuntimeState) -> None:
    if "status" in state and state["status"] not in RUN_STATUSES:
        raise ValueError(f"state.status must be one of {sorted(RUN_STATUSES)}")
    stages = state.get("stages", {})
    diagnostic_stage = stages.get(DIAGNOSTIC_STAGE_SEQUENCE[0]) if isinstance(stages, dict) else None
    if isinstance(diagnostic_stage, dict) and "status" in diagnostic_stage:
        diagnostic_status = diagnostic_stage["status"]
        if diagnostic_status not in DIAGNOSTIC_STAGE_STATUSES:
            raise ValueError(
                "diagnostic stage status must be one of "
                f"{sorted(DIAGNOSTIC_STAGE_STATUSES)}"
            )
    revisions = state.get("revisions", {})
    if revisions is not None and not isinstance(revisions, dict):
        raise ValueError("runtime state revisions must be an object")
    if isinstance(revisions, dict):
        for revision_id, revision in revisions.items():
            if not isinstance(revision, dict):
                raise ValueError(f"runtime revision must be an object: {revision_id}")
            status = revision.get("status")
            if status not in REVISION_STATUSES:
                raise ValueError(
                    f"revision status must be one of {sorted(REVISION_STATUSES)}: "
                    f"{revision_id}={status!r}"
                )


def load_state(run_root: str | Path) -> RuntimeState:
    path = state_path(run_root)
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"runtime state must contain a JSON object: {path}")
    if state.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
        raise ValueError(
            "runtime state schema mismatch: "
            f"expected={RUNTIME_STATE_SCHEMA_VERSION!r}, "
            f"actual={state.get('schema_version')!r}"
        )
    _validate_runtime_status_domains(state)
    return state


def _write_state_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(_to_json_friendly(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def save_state(state: RuntimeState, run_root: str | Path | None = None) -> Path:
    resolved_run_root = (
        _resolve_runtime_root_path(run_root)
        if run_root is not None
        else _resolve_runtime_root_path(state["run_root"])
    )
    root = Path(str(state.get("repo_root", _repo_root()))).expanduser().resolve()
    if not _is_subpath(resolved_run_root, root) and not _is_subpath(resolved_run_root, root / "outputs"):
        raise ValueError(f"Refusing to save runtime state outside the HAG4R repository: {resolved_run_root}")
    path = state_path(resolved_run_root)
    payload = _to_json_friendly(state)
    if payload.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"runtime state schema_version must be {RUNTIME_STATE_SCHEMA_VERSION!r}"
        )
    _validate_runtime_status_domains(payload)
    _write_state_json(path, payload)
    if payload.get("run_mode") in REVISION_AWARE_RUN_MODES and payload.get("active_revision"):
        refresh_active_revision_snapshot(payload, resolved_run_root, payload=payload)
    return path


def update_state(run_root: str | Path, **patch: Any) -> RuntimeState:
    state = load_state(run_root)
    state.update(_to_json_friendly(patch))
    state["updated_at"] = _utc_now()
    save_state(state, run_root)
    return state


def append_history(
    run_root: str | Path,
    agent: str,
    note: str = "",
    *,
    event: str = "note",
    detail: dict[str, Any] | None = None,
) -> RuntimeState:
    state = load_state(run_root)
    state.setdefault("history", []).append(
        {
            "timestamp": _utc_now(),
            "agent": agent,
            "event": event,
            "note": note,
            "detail": _to_json_friendly(detail or {}),
        }
    )
    state["updated_at"] = _utc_now()
    save_state(state, run_root)
    return state


def stage_done(run_root: str | Path, stage_name: str) -> bool:
    stage = load_state(run_root).get("stages", {}).get(stage_name)
    return bool(isinstance(stage, dict) and stage.get("ok") is True)


def _pending_stage() -> dict[str, Any]:
    return {
        "ok": False,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "outputs": {},
        "stats": {},
        "artifacts": {},
        "tool_calls": [],
        "error": "",
    }


def record_stage(
    run_root: str | Path,
    stage_name: str,
    ok: bool,
    *,
    outputs: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
    status: str | None = None,
    allow_new: bool = False,
) -> RuntimeState:
    state = load_state(run_root)
    stages = state.setdefault("stages", {})
    if stage_name not in stages and not allow_new:
        raise KeyError(f"stage is not in runtime state: {stage_name}")
    existing = stages.get(stage_name, _pending_stage())
    now = _utc_now()
    resolved_status = status or ("success" if ok else "failed")
    if stage_name == DIAGNOSTIC_STAGE_SEQUENCE[0] and resolved_status not in DIAGNOSTIC_STAGE_STATUSES:
        raise ValueError(
            "diagnostic stage status must be one of "
            f"{sorted(DIAGNOSTIC_STAGE_STATUSES)}"
        )
    stages[stage_name] = {
        **existing,
        "ok": bool(ok),
        "status": resolved_status,
        "finished_at": now,
        "outputs": _to_json_friendly(outputs or {}),
        "stats": _to_json_friendly(stats or {}),
        "artifacts": _to_json_friendly(artifacts or {}),
        "error": error or "",
    }
    if existing.get("started_at") is None:
        stages[stage_name]["started_at"] = now

    completed = state.setdefault("completed_global_stage_sequence", [])
    if ok and stage_name not in completed:
        completed.append(stage_name)

    state.setdefault("runtime_events", []).append(
        {
            "timestamp": now,
            "event": "stage_recorded",
            "stage_name": stage_name,
            "ok": bool(ok),
            "status": resolved_status,
        }
    )
    state["updated_at"] = now
    save_state(state, run_root)
    _record_profile_event_fail_open(
        run_root,
        "deterministic_stage_recorded",
        stage=stage_name,
        operation="record_stage",
        status=resolved_status,
    )
    return state


def record_tool_call(
    run_root: str | Path,
    stage_name: str,
    tool_name: str,
    *,
    status: str,
    inputs_summary: dict[str, Any] | None = None,
    outputs_summary: dict[str, Any] | None = None,
    error: str = "",
    allow_new_stage: bool = False,
) -> RuntimeState:
    state = load_state(run_root)
    stages = state.setdefault("stages", {})
    if stage_name not in stages and not allow_new_stage:
        raise KeyError(f"stage is not in runtime state: {stage_name}")
    stage = stages.setdefault(stage_name, _pending_stage())
    now = _utc_now()
    if status == "started" and stage.get("started_at") is None:
        stage["started_at"] = now
    stage.setdefault("tool_calls", []).append(
        {
            "timestamp": now,
            "tool_name": tool_name,
            "status": status,
            "inputs_summary": _to_json_friendly(inputs_summary or {}),
            "outputs_summary": _to_json_friendly(outputs_summary or {}),
            "error": error,
        }
    )
    state["updated_at"] = now
    save_state(state, run_root)
    _record_profile_event_fail_open(
        run_root,
        "tool_status",
        stage=stage_name,
        operation=tool_name,
        status=status,
    )
    return state


def _validate_common(
    *,
    run_id: str,
    diagnostics: dict[str, Any],
) -> None:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if diagnostics["max_runs"] < 1:
        raise ValueError("diagnostic max_runs must be >= 1")
    if diagnostics["max_episodes"] < 1:
        raise ValueError("diagnostic max_episodes must be >= 1")
    if diagnostics["max_actions_per_episode"] < 1:
        raise ValueError("diagnostic max_actions_per_episode must be >= 1")
    if diagnostics["episode_timeout_s"] < 1:
        raise ValueError("diagnostic episode_timeout_s must be >= 1")
    if diagnostics["live_port"] < 0 or diagnostics["live_port"] > 65535:
        raise ValueError("diagnostic live_port must be between 0 and 65535")
    if diagnostics["live_ready_timeout_s"] < 1:
        raise ValueError("diagnostic live_ready_timeout_s must be >= 1")
    if diagnostics["live_heartbeat_ms"] < 1:
        raise ValueError("diagnostic live_heartbeat_ms must be >= 1")
    if diagnostics["live_client_lease_timeout_ms"] < 1:
        raise ValueError("diagnostic live_client_lease_timeout_ms must be >= 1")
    if diagnostics["probe_max_vertices"] < 1:
        raise ValueError("diagnostic probe_max_vertices must be >= 1")
    if not math.isfinite(float(diagnostics["probe_max_speed_m_s"])) or diagnostics["probe_max_speed_m_s"] <= 0:
        raise ValueError("diagnostic probe_max_speed_m_s must be finite and > 0")
    if diagnostics["probe_max_duration_steps"] < DEFAULT_DIAGNOSTIC_SIMULATE_STEPS:
        raise ValueError(
            f"diagnostic probe_max_duration_steps must be >= {DEFAULT_DIAGNOSTIC_SIMULATE_STEPS}"
        )


def _object_name_from_source(path: Path) -> str:
    return path.stem


def _output_tag(run_id: str, output_tag: str | None) -> str:
    if output_tag:
        return output_tag
    return f"{run_id}_agentic"


def _build_diagnostics_config(
    *,
    enabled: bool,
    max_runs: int,
    max_episodes: int,
    max_actions_per_episode: int,
    episode_timeout_s: int,
    live_host: str,
    live_port: int,
    live_ready_timeout_s: int,
    live_heartbeat_ms: int,
    live_client_lease_timeout_ms: int,
    probe_max_vertices: int,
    probe_max_distance_m: float,
    probe_max_speed_m_s: float,
    probe_max_duration_steps: int,
    genesis_root: str | Path,
    genesis_env_path: str | Path | None = None,
    genesis_live_command: str = "python -m genesis.live.server",
) -> dict[str, Any]:
    diagnostics = {
        "enabled": bool(enabled),
        "max_runs": int(max_runs),
        "max_episodes": int(max_episodes),
        "max_actions_per_episode": int(max_actions_per_episode),
        "episode_timeout_s": int(episode_timeout_s),
        "live_host": live_host,
        "live_port": int(live_port),
        "live_ready_timeout_s": int(live_ready_timeout_s),
        "live_heartbeat_ms": int(live_heartbeat_ms),
        "live_client_lease_timeout_ms": int(live_client_lease_timeout_ms),
        "probe_max_vertices": int(probe_max_vertices),
        "probe_max_distance_m": float(probe_max_distance_m),
        "probe_max_speed_m_s": float(probe_max_speed_m_s),
        "probe_max_duration_steps": int(probe_max_duration_steps),
        "genesis_root": str(Path(genesis_root).expanduser().resolve()),
        "genesis_env_path": (
            str(Path(genesis_env_path).expanduser().resolve())
            if genesis_env_path is not None
            else None
        ),
        "genesis_live_command": str(genesis_live_command).strip(),
        "episodes": [],
        "active_episode_index": None,
    }
    if not diagnostics["genesis_live_command"]:
        raise ValueError("genesis_live_command must be non-empty")
    return diagnostics


def _common_paths(
    *,
    repo_root: Path,
    run_root: Path,
    output_tag: str,
    omnipart_output_dir: Path,
    material_params_source_path: Path | None,
    sim_diagnostics_report_path: Path | None,
    artifact_root: Path | None = None,
) -> dict[str, str | None]:
    final_export_dir = repo_root / "outputs/run_pipeline" / output_tag
    final_export_sim_diagnostics_dir = final_export_dir / "asset_refinement" / "sim_diagnostics"
    sim_diagnostics_dir = (artifact_root / "sim_diagnostics") if artifact_root is not None else run_root / "sim_diagnostics"
    diagnostics_report = sim_diagnostics_report_path or sim_diagnostics_dir / "diagnostic_summary.json"
    inferred_params_path = (
        artifact_root / "material_inference" / f"{output_tag}.json"
        if artifact_root is not None
        else repo_root / "outputs/infer_params" / output_tag / f"{output_tag}.json"
    )
    mesh_processing_dir = (
        artifact_root / "mesh_processing"
        if artifact_root is not None
        else repo_root / "outputs/mesh_processing" / output_tag
    )
    source_combined_mesh_path = omnipart_output_dir / "mesh_combined.mesh"
    partwise_params_path = mesh_processing_dir / "assign_params_to_prims" / "volumetric_params.npz"
    monolithic_mesh_path = mesh_processing_dir / "combined_to_monolithic" / f"{output_tag}.mesh"
    metric_mesh_scaling_path = mesh_processing_dir / "combined_to_monolithic" / "metric_mesh_scaling.json"
    volume_topology_path = mesh_processing_dir / "combined_to_monolithic" / "volume_topology.json"
    mesh_request_path = mesh_processing_dir / "combined_to_monolithic" / f"{output_tag}.per_part_volume_meshing_request.json"
    post_mesh_texture_dir = mesh_processing_dir / "post_mesh_texture"
    monolithic_params_dir = (
        mesh_processing_dir / "assign_monolithic_mesh_params"
        if artifact_root is not None
        else repo_root / "outputs/assign_monolithic_mesh_params" / output_tag
    )
    paths: dict[str, str | None] = {
        "logs_dir": str((artifact_root / "logs") if artifact_root is not None else run_root / "logs"),
        "final_export_dir": str(final_export_dir),
        "final_asset_refinement_dir": str(final_export_dir / "asset_refinement"),
        "final_export_sim_diagnostics_dir": str(final_export_sim_diagnostics_dir),
        "omnipart_output_dir": str(omnipart_output_dir),
        "omnipart_appearance_manifest_path": str(
            omnipart_output_dir / "appearance" / "appearance_manifest.json"
        ),
        "material_params_source_path": str(material_params_source_path) if material_params_source_path else None,
        "inferred_params_path": str(inferred_params_path),
        "part_labels_path": str(omnipart_output_dir / "part_labels.npz"),
        "source_combined_mesh_path": str(source_combined_mesh_path),
        "partwise_params_path": str(partwise_params_path),
        "metric_mesh_scaling_path": str(metric_mesh_scaling_path),
        "volume_topology_path": str(volume_topology_path),
        "mesh_processing_request_path": str(mesh_request_path),
        "monolithic_mesh_path": str(monolithic_mesh_path),
        "monolithic_params_dir": str(monolithic_params_dir),
        "monolithic_params_path": str(monolithic_params_dir / "heterogeneous_params.npz"),
        "post_mesh_texture_dir": str(post_mesh_texture_dir),
        "post_mesh_texture_request_path": str(
            post_mesh_texture_dir / "post_mesh_texture_request.json"
        ),
        "textured_visual_mesh_path": str(post_mesh_texture_dir / "visual_mesh.glb"),
        "albedo_path": str(post_mesh_texture_dir / "albedo.png"),
        "visual_to_physics_binding_path": str(
            post_mesh_texture_dir / "visual_to_physics.npz"
        ),
        "visual_manifest_path": str(post_mesh_texture_dir / "visual_manifest.json"),
        "texture_qa_report_path": str(post_mesh_texture_dir / "qa" / "report.json"),
        "diagnostic_workspace_dir": str(sim_diagnostics_dir),
        "diagnostic_workspace_index_path": str(sim_diagnostics_dir / "index.json"),
        "diagnostic_workspace_markdown_index_path": str(sim_diagnostics_dir / "index.md"),
        "diagnostic_log_path": str(sim_diagnostics_dir / "diagnostic_log.md"),
        "diagnostic_observations_digest_path": str(sim_diagnostics_dir / "observations_digest.json"),
        "diagnostic_generated_episodes_dir": str((sim_diagnostics_dir / "episodes") if artifact_root is not None else final_export_sim_diagnostics_dir / "episodes"),
        "sim_diagnostics_dir": str(sim_diagnostics_dir),
        "sim_diagnostics_report_path": str(diagnostics_report),
        "sim_diagnostics_markdown_report_path": str(sim_diagnostics_dir / "diagnostic_report.md"),
        "sim_diagnostics_cues_path": str(sim_diagnostics_dir / "diagnostic_cues.json"),
        "final_export_manifest_path": str(final_export_dir / "final_export_manifest.json"),
        "final_mesh_path": str(final_export_dir / "final_mesh.mesh"),
        "final_heterogeneous_params_path": str(final_export_dir / "heterogeneous_params.npz"),
        "final_inferred_params_path": str(final_export_dir / "inferred_params.json"),
        "final_volume_topology_path": str(final_export_dir / "volume_topology.json"),
        "final_metric_mesh_scaling_path": str(final_export_dir / "metric_mesh_scaling.json"),
        "final_appearance_dir": str(final_export_dir / "appearance"),
    }
    read_only_keys = {"material_params_source_path"}
    for key, value in paths.items():
        if value is None:
            continue
        value_path = Path(value)
        if key in read_only_keys:
            continue
        if value_path == omnipart_output_dir or _is_subpath(value_path, omnipart_output_dir):
            continue
        _assert_write_path(value_path, repo_root=repo_root, run_root=run_root)
    return paths


def build_full_image_revision_paths(
    *,
    repo_root: Path,
    run_root: Path,
    revision_root_path: Path,
    output_tag: str,
    object_name: str,
    source_image: Path,
    material_params_source_path: Path | None,
    sim_diagnostics_report_path: Path | None,
) -> dict[str, str | None]:
    omnipart_root = revision_root_path / "omnipart"
    omnipart_output_dir = omnipart_root / object_name
    paths = _common_paths(
        repo_root=repo_root,
        run_root=run_root,
        output_tag=output_tag,
        omnipart_output_dir=omnipart_output_dir,
        material_params_source_path=material_params_source_path,
        sim_diagnostics_report_path=sim_diagnostics_report_path,
        artifact_root=revision_root_path,
    )
    paths.update(
        {
            "source_image": str(source_image),
            "image_cleanup_dir": str(revision_root_path / "image_cleanup"),
            "image_cleanup_attempt_1_path": str(revision_root_path / "image_cleanup" / "attempt_01.png"),
            "image_cleanup_attempt_2_path": str(revision_root_path / "image_cleanup" / "attempt_02.png"),
            "cleaned_image_path": str(revision_root_path / "image_cleanup" / f"{object_name}.png"),
            "cleanup_report_path": str(revision_root_path / "image_cleanup" / f"{object_name}_image_cleanup.json"),
            "object_description_path": str(revision_root_path / "object_description" / f"{object_name}.json"),
            "segmentation_dir": str(revision_root_path / "segmentation" / object_name),
            "sam3_prompt_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_sam3_prompt.json"),
            "sam3_raw_npz_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_sam3_raw.npz"),
            "sam3_raw_metadata_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_sam3_raw.json"),
            "processed_rgba_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_processed.png"),
            "processed_white_bg_png_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_white_bg.png"),
            "processed_black_bg_png_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_black_bg.png"),
            "image_white_bg_tensor_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_image_white_bg.npy"),
            "image_black_bg_tensor_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_image_black_bg.npy"),
            "group_ids_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_group_ids.npy"),
            "mask_exr_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_mask.exr"),
            "ordered_mask_input_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_ordered_mask_input.npy"),
            "ordered_mask_vis_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_ordered_mask_vis.png"),
            "segmentation_overlay_path": str(revision_root_path / "segmentation" / object_name / f"{object_name}_segmentation_overlay.png"),
            "segmentation_manifest_path": str(revision_root_path / "segmentation" / object_name / "segmentation_manifest.json"),
            "omnipart_root": str(omnipart_root),
        }
    )
    for key, value in paths.items():
        if value is None or key in {"source_image", "material_params_source_path"}:
            continue
        _assert_write_path(Path(str(value)), repo_root=repo_root, run_root=run_root)
    return paths


def _final_export_payload(paths: dict[str, str | None], *, include_cleanup: bool, include_diagnostics: bool) -> dict[str, Any]:
    artifact_paths = {
        "mesh": paths["final_mesh_path"],
        "heterogeneous_params": paths["final_heterogeneous_params_path"],
        "inferred_params": paths["final_inferred_params_path"],
        "volume_topology": paths["final_volume_topology_path"],
        "metric_mesh_scaling": paths["final_metric_mesh_scaling_path"],
        "appearance": paths["final_appearance_dir"],
    }
    source_directories: dict[str, str | None] = {}
    if include_cleanup:
        source_directories["image_cleanup"] = str(Path(str(paths["cleaned_image_path"])).parent)
    if include_diagnostics:
        source_directories["sim_diagnostics"] = paths["sim_diagnostics_dir"]
    return {
        "dir": paths["final_export_dir"],
        "manifest_path": paths["final_export_manifest_path"],
        "artifact_paths": artifact_paths,
        "source_directories": source_directories,
    }


def _initial_stages(
    stage_sequence: tuple[str, ...],
    planned_diagnostic_stage_sequence: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    stages = {stage_name: _pending_stage() for stage_name in stage_sequence}
    for stage_name in planned_diagnostic_stage_sequence:
        stages.setdefault(stage_name, _pending_stage())
    return stages


def _runtime_skill_suite_state(repo_root: Path) -> dict[str, str]:
    source_root = repo_root
    if not (runtime_skill_root(source_root) / "SKILL.md").is_file():
        source_root = _repo_root()
    skill = load_pipeline_skill(repo_root=source_root)
    return {
        "name": skill.name,
        "skill_suite": skill.skill_suite,
        "title": skill.title,
        "path": str(skill.path),
        "virtual_path": skill.virtual_path,
        "sha256": skill.sha256,
        "source_repo_root": str(source_root),
    }


def _base_state(
    *,
    repo_root: Path,
    runs_root: Path,
    run_root: Path,
    run_id: str,
    run_mode: str,
    fidelity: str,
    object_name: str,
    output_tag: str,
    inputs: dict[str, Any],
    run_config: RunConfig,
    diagnostics: dict[str, Any],
    mesh_extra_args: tuple[str, ...],
    paths: dict[str, Any],
    planned_global_stage_sequence: tuple[str, ...],
    planned_diagnostic_stage_sequence: tuple[str, ...],
    runtime_kind: str = "skill_suite",
    runtime_entrypoint: str = "run-diaggen-pipeline",
    terminate_after_stage: str | None = None,
) -> RuntimeState:
    now = _utc_now()
    return {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "runtime_kind": runtime_kind,
        "runtime_entrypoint": runtime_entrypoint,
        "repo_root": str(repo_root),
        "runs_root": str(runs_root),
        "run_root": str(run_root),
        "state_path": str(state_path(run_root)),
        "run_id": run_id,
        "run_mode": run_mode,
        "status": "seeded",
        "created_at": now,
        "updated_at": now,
        "fidelity": fidelity,
        "object_name": object_name,
        "output_tag": output_tag,
        "inputs": _to_json_friendly(inputs),
        "run_config": _to_json_friendly(run_config_audit_state(run_config)),
        "runtime_skill_suite": _to_json_friendly(_runtime_skill_suite_state(repo_root)),
        "diagnostics": _to_json_friendly(diagnostics),
        "mesh_config": {"mesh_extra_args": list(mesh_extra_args)},
        "paths": _to_json_friendly(paths),
        "terminate_after_stage": terminate_after_stage,
        "planned_global_stage_sequence": list(planned_global_stage_sequence),
        "planned_diagnostic_stage_sequence": list(planned_diagnostic_stage_sequence),
        "completed_global_stage_sequence": [],
        "stages": _initial_stages(planned_global_stage_sequence, planned_diagnostic_stage_sequence),
        "history": [],
        "runtime_events": [],
        "route_state": {},
        "diagnostic_repair_brief": {},
        "final_export": _final_export_payload(
            paths,
            include_cleanup=run_mode == "full_image",
            include_diagnostics=bool(diagnostics["enabled"]),
        ),
    }


def _install_bypass_diagnostics_route(state: RuntimeState) -> None:
    state.setdefault("route_state", {})["orchestrator_action_route"] = {
        "route": BYPASS_DIAGNOSTICS_ROUTE,
        "status": "selected",
        "reason": "genesis_diagnostics.enable=false",
        "diagnostics_enabled": False,
        "skipped_stage": DIAGNOSTIC_STAGE_SEQUENCE[0],
        "next_stage_skill_path": STAGE_SKILL_PATHS["post_mesh_texture"],
    }


FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE: dict[str, tuple[str, ...]] = {
    "segmentation": (
        STAGE_SKILL_PATHS["segmentation"],
        STAGE_SKILL_PATHS["omnipart"],
        STAGE_SKILL_PATHS["material_inference"],
        STAGE_SKILL_PATHS["mesh_processing"],
    ),
    "material_inference": (
        STAGE_SKILL_PATHS["material_inference"],
        STAGE_SKILL_PATHS["mesh_processing"],
    ),
    "mesh_processing": (STAGE_SKILL_PATHS["mesh_processing"],),
}

FULL_IMAGE_COPY_PREFIX_STAGES_BY_ROUTE: dict[str, tuple[str, ...]] = {
    "segmentation": ("image_cleanup",),
    "material_inference": (
        "image_cleanup",
        "sam3_omnipart_2d_segmentation",
        "omnipart_generate_parts",
    ),
    "mesh_processing": (
        "image_cleanup",
        "sam3_omnipart_2d_segmentation",
        "omnipart_generate_parts",
        "hag4r_gpt_staged_material_inference",
    ),
}

_DIAGNOSTIC_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "max_runs",
        "max_episodes",
        "max_actions_per_episode",
        "episode_timeout_s",
        "live_host",
        "live_port",
        "live_ready_timeout_s",
        "live_heartbeat_ms",
        "live_client_lease_timeout_ms",
        "probe_max_vertices",
        "probe_max_distance_m",
        "probe_max_speed_m_s",
        "probe_max_duration_steps",
        "genesis_root",
        "genesis_env_path",
        "genesis_live_command",
    }
)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(_to_json_friendly(value)))


def _find_legacy_runtime_contract_fields(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, dict):
        hits = [
            f"{prefix}.{key}"
            for key in value
            if key in LEGACY_RUNTIME_CONTRACT_KEYS or str(key).startswith("tri" + "_")
        ]
        for key, item in value.items():
            hits.extend(_find_legacy_runtime_contract_fields(item, f"{prefix}.{key}"))
        return hits
    if isinstance(value, list):
        hits: list[str] = []
        for index, item in enumerate(value):
            hits.extend(_find_legacy_runtime_contract_fields(item, f"{prefix}[{index}]"))
        return hits
    return []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_lineage_record(path: Path, *, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "kind": "file",
        "role": role,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _copied_file_lineage_record(
    source: Path,
    destination: Path,
    *,
    role: str,
) -> dict[str, Any]:
    record = _file_lineage_record(source, role=role)
    destination_stat = destination.stat()
    destination_sha256 = _sha256_file(destination)
    record.update(
        {
            "destination_path": str(destination),
            "destination_size_bytes": destination_stat.st_size,
            "destination_sha256": destination_sha256,
            "byte_identical": (
                record["size_bytes"] == destination_stat.st_size
                and record["sha256"] == destination_sha256
            ),
        }
    )
    if record["byte_identical"] is not True:
        raise RuntimeError(
            "copied diagnostics snapshot input is not byte-identical: "
            f"role={role}, source={source}, destination={destination}"
        )
    return record


def _directory_lineage_record(path: Path, *, role: str) -> dict[str, Any]:
    state_json = path / "state.json"
    if not state_json.is_file():
        raise FileNotFoundError(f"diagnostics revision snapshot is missing state.json: {state_json}")
    state_stat = state_json.stat()
    return {
        "kind": "directory",
        "role": role,
        "path": str(path),
        "state_json_path": str(state_json),
        "state_json_size_bytes": state_stat.st_size,
        "state_json_mtime_ns": state_stat.st_mtime_ns,
        "state_json_sha256": _sha256_file(state_json),
    }


def validate_external_lineage(state: RuntimeState) -> None:
    lineage = state.get("external_lineage", {})
    if not isinstance(lineage, dict):
        return
    for key, record in lineage.items():
        if not isinstance(record, dict):
            raise ValueError(f"external lineage record must be an object: {key}")
        kind = record.get("kind")
        if kind is None:
            # Import preflight metadata is retained beside lineage records for
            # compatibility, but it does not identify an external source.
            continue
        if kind == "directory":
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"external lineage directory is missing: {key}={path}")
            state_json = Path(str(record.get("state_json_path", path / "state.json"))).expanduser().resolve()
            if not state_json.is_file():
                raise FileNotFoundError(f"external lineage directory state is missing: {key}={state_json}")
            stat = state_json.stat()
            if int(record.get("state_json_size_bytes", -1)) != stat.st_size:
                raise ValueError(f"external lineage directory state size changed: {key}={state_json}")
            if int(record.get("state_json_mtime_ns", -1)) != stat.st_mtime_ns:
                raise ValueError(f"external lineage directory state mtime changed: {key}={state_json}")
            if str(record.get("state_json_sha256", "")) != _sha256_file(state_json):
                raise ValueError(f"external lineage directory state sha256 changed: {key}={state_json}")
            continue
        if kind != "file":
            raise ValueError(f"external lineage record has unsupported kind: {key}={kind}")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"external lineage source is missing: {key}={path}")
        stat = path.stat()
        if int(record.get("size_bytes", -1)) != stat.st_size:
            raise ValueError(f"external lineage source size changed: {key}={path}")
        if int(record.get("mtime_ns", -1)) != stat.st_mtime_ns:
            raise ValueError(f"external lineage source mtime changed: {key}={path}")
        if str(record.get("sha256", "")) != _sha256_file(path):
            raise ValueError(f"external lineage source sha256 changed: {key}={path}")


def revision_root(state: RuntimeState, revision_id: str) -> Path:
    revisions = state.get("revisions", {})
    if not isinstance(revisions, dict) or revision_id not in revisions:
        raise KeyError(f"unknown runtime revision: {revision_id}")
    revision = revisions[revision_id]
    if not isinstance(revision, dict):
        raise ValueError(f"revision metadata must be an object: {revision_id}")
    root = revision.get("revision_root") or Path(str(state["run_root"])) / "revisions" / revision_id
    return Path(str(root)).expanduser().resolve()


def active_revision_root(state: RuntimeState) -> Path:
    active_revision = str(state.get("active_revision", "") or "")
    if not active_revision:
        raise RuntimeError("runtime state has no active revision")
    return revision_root(state, active_revision)


def accepted_revision_root(state: RuntimeState) -> Path:
    accepted_revision = str(state.get("accepted_revision", "") or "")
    if not accepted_revision:
        raise RuntimeError("runtime state has no accepted revision")
    return revision_root(state, accepted_revision)


def next_revision_id(state: RuntimeState) -> str:
    indices: list[int] = []
    revisions = state.get("revisions", {})
    if isinstance(revisions, dict):
        for revision_id in revisions:
            text = str(revision_id)
            if text.startswith("revision_"):
                try:
                    indices.append(int(text.removeprefix("revision_")))
                except ValueError:
                    continue
    return f"revision_{(max(indices) + 1) if indices else 0:04d}"


def _revision_metadata(
    *,
    revision_id: str,
    revision_root_path: Path,
    status: str,
    base_revision: str | None = None,
    rerun_route: str | None = None,
    copied_stage_prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "revision_id": revision_id,
        "revision_root": str(revision_root_path),
        "status": status,
        "base_revision": base_revision,
        "rerun_route": rerun_route,
        "copied_stage_prefix": list(copied_stage_prefix),
        "created_at": now,
        "updated_at": now,
    }


def regenerate_active_revision_paths(state: RuntimeState) -> RuntimeState:
    if not _supports_diagnostic_repair_revisions(state) or not state.get("active_revision"):
        return state
    repo_root = Path(str(state["repo_root"])).expanduser().resolve()
    run_root = Path(str(state["run_root"])).expanduser().resolve()
    revision_root_path = active_revision_root(state)
    source_image = Path(str(state.get("inputs", {}).get("source_image") or state.get("paths", {}).get("source_image"))).expanduser().resolve()
    material_source_raw = state.get("inputs", {}).get("material_params_json") or state.get("paths", {}).get("material_params_source_path")
    material_source = Path(str(material_source_raw)).expanduser().resolve() if material_source_raw else None
    paths = build_full_image_revision_paths(
        repo_root=repo_root,
        run_root=run_root,
        revision_root_path=revision_root_path,
        output_tag=str(state["output_tag"]),
        object_name=str(state["object_name"]),
        source_image=source_image,
        material_params_source_path=material_source,
        sim_diagnostics_report_path=None,
    )
    state["paths"] = _to_json_friendly(paths)
    state["final_export"] = _final_export_payload(
        paths,
        include_cleanup=True,
        include_diagnostics=bool(state.get("diagnostics", {}).get("enabled")),
    )
    _refresh_worker_isolation_for_active_paths(state)
    return state


def initialize_full_image_revision(state: RuntimeState) -> RuntimeState:
    if state.get("run_mode") != "full_image":
        return state
    revision_id = str(state.get("active_revision") or "revision_0000")
    run_root = Path(str(state["run_root"])).expanduser().resolve()
    state["active_revision"] = revision_id
    state.setdefault("accepted_revision", None)
    state.setdefault("exported_revision", None)
    revisions = state.setdefault("revisions", {})
    if not isinstance(revisions, dict):
        raise ValueError("runtime state revisions must be an object")
    revisions.setdefault(
        revision_id,
        _revision_metadata(
            revision_id=revision_id,
            revision_root_path=run_root / "revisions" / revision_id,
            status="active",
        ),
    )
    return regenerate_active_revision_paths(state)


def _supports_diagnostic_repair_revisions(state: Mapping[str, Any]) -> bool:
    """Whether this lifecycle owns full-image revision paths and repair lineage."""
    return state.get("run_mode") == "full_image" or (
        state.get("run_mode") == "post_mesh_processing_diagnostics"
        and state.get("diagnostics_reroute_and_export") is True
    )


def _is_final_revise_export_policy(
    state: Mapping[str, Any],
    decision: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether ``state`` satisfies the complete scoped final-revise contract.

    The policy is derived from the mode flags, durable diagnostic terminal, parent
    decision, and sequential revision budget.  A caller may pass a candidate
    decision while applying the transition; otherwise the persisted decision is
    used.  Returning ``False`` for an incomplete contract keeps ordinary and
    historical paths on their existing transitions, while every positive match
    is fully validated before a texture or export write is permitted.
    """
    if (
        state.get("run_mode") != "post_mesh_processing_diagnostics"
        or state.get("diagnostics_reroute_and_export") is not True
        or state.get("diagnostics_stage_only") is not False
    ):
        return False

    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    try:
        max_runs = int(diagnostics.get("max_runs", 0) or 0)
    except (TypeError, ValueError):
        return False
    if max_runs < 2:
        return False

    stages = state.get("stages")
    diagnostic_stage = stages.get(DIAGNOSTIC_STAGE_SEQUENCE[0]) if isinstance(stages, Mapping) else None
    if (
        not isinstance(diagnostic_stage, Mapping)
        or diagnostic_stage.get("ok") is not True
        or diagnostic_stage.get("status") != "success"
    ):
        return False
    terminal = diagnostics.get("terminal")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("status") != "success"
        or terminal.get("validated") is not True
        or terminal.get("tool_name") != "submit_diagnostic_recommendation"
    ):
        return False
    recommendation = terminal.get("recommendation")
    if not isinstance(recommendation, Mapping):
        return False
    try:
        normalized = normalize_persisted_vlm_final_recommendation(dict(recommendation))
    except ValueError:
        return False
    route = str(normalized.get("route", "") or "")
    cues = normalized.get("diagnostic_cues", [])
    if (
        normalized.get("recommendation") != "revise"
        or route not in FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE
        or not isinstance(cues, list)
        or not any(isinstance(cue, str) and cue.strip() for cue in cues)
        or str(terminal.get("route", "") or "") != route
        or terminal.get("ready") is not False
    ):
        return False

    route_state = state.get("route_state")
    persisted_decision = (
        route_state.get("orchestrator_diagnostic_decision")
        if isinstance(route_state, Mapping)
        else None
    )
    active_decision: Mapping[str, Any] | None = decision if decision is not None else persisted_decision
    if not isinstance(active_decision, Mapping):
        return False
    if (
        active_decision.get("diagnostic_verdict") != "revise"
        or active_decision.get("transition_action") != "post_mesh_texture"
        or active_decision.get("diagnostic_export_policy") != DIAGNOSTIC_EXPORT_POLICY
        or active_decision.get("child_recommendation") != "revise"
        or active_decision.get("recommendation_route") != route
        or active_decision.get("resulting_run_status") != "running"
        or active_decision.get("failure_kind", "")
        or active_decision.get("halt_reason", "")
    ):
        return False
    if "diagnostic_cues" not in active_decision:
        return False
    decision_cues = active_decision.get("diagnostic_cues")
    if decision_cues != list(cues):
        return False
    if active_decision.get("diagnostic_repair_brief", {}) not in ({}, None):
        return False
    if str(active_decision.get("diagnostic_verdict_reason", "") or "").strip():
        return False
    if (
        state.get("diagnostic_repair_brief", {}) not in ({}, None)
        or not isinstance(route_state, Mapping)
        or route_state.get("diagnostic_repair_brief", {}) not in ({}, None)
        or route_state.get("force_rerun_stage_skill_paths", []) != []
        or route_state.get("first_rerouted_stage_skill_path", "") != ""
        or active_decision.get("selected_reentry_stage_skill_paths")
        != [
            STAGE_SKILL_PATHS["post_mesh_texture"],
            STAGE_SKILL_PATHS["final_export"],
        ]
    ):
        return False

    active_revision = str(state.get("active_revision", "") or "")
    revisions = state.get("revisions")
    if not isinstance(revisions, Mapping) or not active_revision:
        return False
    try:
        revision_ids = sorted(
            (str(revision_id) for revision_id in revisions),
            key=lambda value: int(value.removeprefix("revision_")),
        )
    except (TypeError, ValueError):
        return False
    expected_revision_ids = [f"revision_{index:04d}" for index in range(max_runs)]
    if revision_ids != expected_revision_ids or active_revision != expected_revision_ids[-1]:
        return False
    try:
        revision_count = int(
            (route_state or {}).get("diagnostic_revision_count", 0)
            if isinstance(route_state, Mapping)
            else 0
        )
    except (TypeError, ValueError):
        return False
    if revision_count != max_runs - 1:
        return False
    decision_count = active_decision.get("diagnostic_revision_count")
    try:
        if int(decision_count) != max_runs - 1:
            return False
    except (TypeError, ValueError):
        return False
    if "diagnostic_revision_history" in state:
        history = state.get("diagnostic_revision_history")
        if not isinstance(history, list) or len(history) != max_runs:
            return False
    if state.get("accepted_revision") not in (None, ""):
        return False
    if state.get("exported_revision") not in (None, "", active_revision):
        return False
    return True


def refresh_active_revision_snapshot(
    state: RuntimeState,
    run_root: str | Path,
    *,
    payload: RuntimeState | None = None,
) -> Path:
    del run_root
    if state.get("run_mode") not in REVISION_AWARE_RUN_MODES or not state.get("active_revision"):
        raise RuntimeError("active revision snapshots are only available for revision-aware runs")
    snapshot_path = active_revision_root(state) / "state.json"
    return _write_state_json(snapshot_path, payload or state)


def _reset_revision_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    reset = {key: _json_clone(value) for key, value in diagnostics.items() if key in _DIAGNOSTIC_CONFIG_KEYS}
    reset.setdefault("enabled", bool(diagnostics.get("enabled")))
    reset["episodes"] = []
    reset["active_episode_index"] = None
    return reset


def _sanitized_repair_cue(cue: dict[str, Any], *, base_revision: str) -> dict[str, Any]:
    del base_revision
    route = str(cue.get("route", "") or "")
    diagnostic_cues = cue.get("diagnostic_cues")
    if not isinstance(diagnostic_cues, list):
        legacy_field = {
            "segmentation": "segmentation_hints",
            "material_inference": "material_inference_hints",
            "mesh_processing": "mesh_processing_hints",
        }.get(route, "")
        diagnostic_cues = cue.get(legacy_field, []) if legacy_field else []
    normalized_cues = [str(item).strip() for item in diagnostic_cues if isinstance(item, str) and item.strip()]
    if route not in FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE or not normalized_cues:
        raise ValueError("persisted diagnostic repair cue requires a legal route and non-empty diagnostic_cues")
    return {"route": route, "diagnostic_cues": normalized_cues}


def _sanitized_repair_brief(brief: dict[str, Any], *, base_revision: str) -> dict[str, Any]:
    if not isinstance(brief, dict) or not brief:
        return {}
    schema_version = brief.get("schema_version")
    if schema_version not in {DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION, "hag4r-diagnostic-repair-brief-v2"}:
        raise ValueError(
            "unsupported diagnostic repair brief schema_version: "
            f"{brief.get('schema_version')!r}"
        )
    diagnostic_cues = brief.get("diagnostic_cues")
    if schema_version == "hag4r-diagnostic-repair-brief-v2":
        diagnostic_cues = brief.get("selected_stage_hints", [])
    if not isinstance(diagnostic_cues, list) or not any(
        isinstance(item, str) and item.strip() for item in diagnostic_cues
    ):
        raise ValueError("diagnostic repair brief requires non-empty diagnostic_cues")
    allowed_keys = (
        "route", "designated_destination_stage", "probe_report_summary",
        "diagnostic_summary_path", "first_rerouted_stage_skill_path",
    )
    sanitized = {key: _json_clone(brief.get(key)) for key in allowed_keys if key in brief}
    sanitized["schema_version"] = DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION
    sanitized["diagnostic_cues"] = [
        item.strip() for item in diagnostic_cues if isinstance(item, str) and item.strip()
    ]
    sanitized["source_revision"] = base_revision
    return sanitized


def _rewrite_path_strings(value: Any, *, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, str):
        old = old_root.as_posix()
        if value == old:
            return new_root.as_posix()
        if value.startswith(old + "/"):
            return new_root.as_posix() + value[len(old) :]
        return value
    if isinstance(value, list):
        return [_rewrite_path_strings(item, old_root=old_root, new_root=new_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_path_strings(item, old_root=old_root, new_root=new_root) for key, item in value.items()}
    return value


def _snapshot_absolute_path(
    snapshot_state: dict[str, Any],
    key: str,
    *,
    required: bool,
) -> Path | None:
    if key not in snapshot_state or snapshot_state[key] is None:
        if required:
            raise ValueError(f"diagnostics revision snapshot {key} must be an absolute path")
        return None
    value = str(snapshot_state[key]).strip()
    if not value:
        raise ValueError(f"diagnostics revision snapshot {key} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"diagnostics revision snapshot {key} must be an absolute path: {value}")
    return path


def _preflight_diagnostics_revision_snapshot_paths(
    *,
    snapshot_state: RuntimeState,
    snapshot_root: Path,
    repo_root: Path,
) -> tuple[RuntimeState, dict[str, Any]]:
    state = _json_clone(snapshot_state)
    snapshot_root = snapshot_root.expanduser().resolve()
    local_run_root = snapshot_root.parent.parent
    source_repo_root = _snapshot_absolute_path(snapshot_state, "repo_root", required=True)
    assert source_repo_root is not None
    source_run_root = _snapshot_absolute_path(snapshot_state, "run_root", required=False)
    active_revision = str(snapshot_state.get("active_revision", "") or "")
    source_revision_root: Path | None = None
    revisions = snapshot_state.get("revisions", {})
    if active_revision and isinstance(revisions, dict):
        revision = revisions.get(active_revision)
        if isinstance(revision, dict):
            source_revision_root = _snapshot_absolute_path(
                revision,
                "revision_root",
                required=False,
            )
    if source_revision_root is None and source_run_root is not None and active_revision:
        source_revision_root = source_run_root / "revisions" / active_revision

    report: dict[str, Any] = {
        "schema_version": "hag4r-diagnostics-snapshot-preflight-v1",
        "source": "snapshot_state_absolute_path_rebase",
        "snapshot_root": snapshot_root.as_posix(),
        "source_repo_root": source_repo_root.as_posix(),
        "source_run_root": source_run_root.as_posix() if source_run_root is not None else "",
        "source_revision_root": source_revision_root.as_posix() if source_revision_root is not None else "",
        "snapshot_state_path": (snapshot_root / "state.json").as_posix(),
        "snapshot_state_sha256": _sha256_file(snapshot_root / "state.json"),
        "stage_alias_normalizations": [],
        "rebased_paths": [],
        "unresolved_paths": [],
    }
    if not active_revision or snapshot_root.name != active_revision or snapshot_root.parent.name != "revisions":
        report["skipped"] = True
        report["skip_reason"] = "provided_path_is_not_active_revision_directory"
        state["diagnostics_snapshot_preflight"] = report
        return state, report

    source_roots = (
        (source_revision_root, snapshot_root, "snapshot_revision_root"),
        (source_run_root, local_run_root, "snapshot_run_root"),
        (source_repo_root, repo_root, "repo_root"),
    )

    def rebase_path(path: Path) -> tuple[str, str] | None:
        for source_root, local_root, rule in source_roots:
            if source_root is None:
                continue
            try:
                relative = path.relative_to(source_root)
            except ValueError:
                continue
            return (local_root / relative).as_posix(), rule
        return None

    def rewrite(value: Any, json_path: str = "$") -> Any:
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                return value
            rebased = rebase_path(path)
            if rebased is None:
                return value
            new_value, rule = rebased
            report["rebased_paths"].append(
                {"json_path": json_path, "from": value, "to": new_value, "rule": rule}
            )
            return new_value
        if isinstance(value, list):
            return [rewrite(item, f"{json_path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: rewrite(item, f"{json_path}.{key}") for key, item in value.items()}
        return value

    state = rewrite(state)
    state["repo_root"] = repo_root.as_posix()
    state["run_root"] = local_run_root.as_posix()
    state["runs_root"] = local_run_root.parent.as_posix()
    if active_revision:
        state["active_revision"] = active_revision
        revisions = state.setdefault("revisions", {})
        if not isinstance(revisions, dict):
            raise ValueError("diagnostics revision snapshot revisions must be an object")
        revision = revisions.setdefault(active_revision, {})
        if not isinstance(revision, dict):
            raise ValueError(f"diagnostics revision metadata must be an object: {active_revision}")
        revision["revision_root"] = snapshot_root.as_posix()
    state["diagnostics_snapshot_preflight"] = report
    return state, report


def _normalize_diagnostics_snapshot_stage_aliases(
    state: RuntimeState,
    report: dict[str, Any],
) -> RuntimeState:
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("diagnostics revision snapshot stages must be an object")
    completed = state.get("completed_global_stage_sequence", [])
    if not isinstance(completed, list):
        raise ValueError("diagnostics revision snapshot completed_global_stage_sequence must be a list")

    for canonical, aliases in DIAGNOSTICS_SNAPSHOT_STAGE_ALIASES.items():
        canonical_payload = stages.get(canonical)
        present_aliases = [alias for alias in aliases if alias in stages]
        if canonical_payload is not None:
            if not isinstance(canonical_payload, dict):
                raise ValueError(f"diagnostics revision snapshot stage must be an object: {canonical}")
            for alias in present_aliases:
                alias_payload = stages[alias]
                if not isinstance(alias_payload, dict):
                    raise ValueError(f"diagnostics revision snapshot stage must be an object: {alias}")
                if (
                    bool(alias_payload.get("ok")) != bool(canonical_payload.get("ok"))
                    or str(alias_payload.get("status", "")) != str(canonical_payload.get("status", ""))
                ):
                    raise ValueError(
                        "diagnostics revision snapshot has conflicting canonical and legacy stage records: "
                        f"{canonical}, {alias}"
                    )
            continue
        if not present_aliases:
            continue
        if len(present_aliases) != 1:
            raise ValueError(
                f"diagnostics revision snapshot has ambiguous legacy aliases for {canonical}: "
                + ", ".join(present_aliases)
            )
        alias = present_aliases[0]
        alias_payload = stages[alias]
        if not isinstance(alias_payload, dict):
            raise ValueError(f"diagnostics revision snapshot stage must be an object: {alias}")
        stages[canonical] = _json_clone(alias_payload)
        normalized_completed: list[Any] = []
        for stage_name in completed:
            normalized = canonical if stage_name == alias else stage_name
            if normalized not in normalized_completed:
                normalized_completed.append(normalized)
        state["completed_global_stage_sequence"] = normalized_completed
        report["stage_alias_normalizations"].append(
            {
                "schema_version": "hag4r-diagnostics-snapshot-stage-alias-v1",
                "source_stage": alias,
                "canonical_stage": canonical,
                "source_ok": bool(alias_payload.get("ok")),
                "source_status": str(alias_payload.get("status", "")),
            }
        )
    state["diagnostics_snapshot_preflight"] = report
    return state


def _rewrite_json_files_under(root: Path, *, old_root: Path, new_root: Path) -> None:
    if not root.exists():
        return
    paths = (root.rglob("*.json") if root.is_dir() else (root,))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = _rewrite_path_strings(payload, old_root=old_root, new_root=new_root)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_path_replace(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"cannot copy missing revision artifact: {source}")
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _copy_file_atomic(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"cannot copy missing revision artifact: {source}")
    if source == target:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_prefix_artifacts(
    *,
    base_state: RuntimeState,
    new_state: RuntimeState,
    prefix_stages: tuple[str, ...],
) -> None:
    stage_path_groups: dict[str, tuple[str, ...]] = {
        "image_cleanup": (
            "image_cleanup_dir",
            "object_description_path",
        ),
        "sam3_omnipart_2d_segmentation": ("segmentation_dir",),
        "omnipart_generate_parts": ("omnipart_root",),
        "hag4r_gpt_staged_material_inference": ("inferred_params_path",),
    }
    base_paths = base_state.get("paths", {})
    new_paths = new_state.get("paths", {})
    copied_targets: list[Path] = []
    for stage_name in prefix_stages:
        stage = base_state.get("stages", {}).get(stage_name, {})
        if not isinstance(stage, dict) or stage.get("ok") is not True:
            raise RuntimeError(f"cannot copy incomplete base revision stage: {stage_name}")
        for key in stage_path_groups.get(stage_name, ()):
            source = Path(str(base_paths[key])).expanduser().resolve()
            target = Path(str(new_paths[key])).expanduser().resolve()
            if key.endswith("_path") and source.is_file():
                _copy_path_replace(source, target)
                copied_targets.append(target)
            else:
                _copy_path_replace(source, target)
                copied_targets.append(target)
    old_root = active_revision_root(base_state)
    new_root = active_revision_root(new_state)
    for target in copied_targets:
        _rewrite_json_files_under(target, old_root=old_root, new_root=new_root)


def validate_copied_prefix_metadata(
    state: RuntimeState,
    *,
    base_revision: str,
    new_revision: str,
) -> None:
    base_root = revision_root(state, base_revision).as_posix()
    new_root = revision_root(state, new_revision)
    if not new_root.exists():
        return
    offenders = [
        path
        for path in new_root.rglob("*.json")
        if base_root in path.read_text(encoding="utf-8")
    ]
    copied_prefix = state.get("revisions", {}).get(new_revision, {}).get("copied_stage_prefix", [])
    if isinstance(copied_prefix, list):
        for stage_name in copied_prefix:
            stage_payload = state.get("stages", {}).get(str(stage_name), {})
            if base_root in json.dumps(_to_json_friendly(stage_payload), sort_keys=True):
                offenders.append(new_root / f"<state.stages.{stage_name}>")
    if offenders:
        raise ValueError(
            "copied revision metadata still references the base revision: "
            + ", ".join(str(path) for path in offenders[:5])
        )


def validate_revision_path_ownership(state: RuntimeState) -> None:
    if state.get("run_mode") not in REVISION_AWARE_RUN_MODES or not state.get("active_revision"):
        return
    repo_root = Path(str(state["repo_root"])).expanduser().resolve()
    run_root = Path(str(state["run_root"])).expanduser().resolve()
    active_root = active_revision_root(state)
    owned_exceptions = {
        "source_image",
        "material_params_source_path",
        "final_export_dir",
        "final_asset_refinement_dir",
        "final_export_sim_diagnostics_dir",
        "final_export_manifest_path",
        "final_mesh_path",
        "final_heterogeneous_params_path",
        "final_inferred_params_path",
        "final_volume_topology_path",
        "final_metric_mesh_scaling_path",
        "final_appearance_dir",
    }
    for key, value in state.get("paths", {}).items():
        if value is None or key in owned_exceptions:
            continue
        path = Path(str(value)).expanduser().resolve()
        if _is_subpath(path, active_root):
            continue
        if key.startswith("final_") and _is_subpath(path, repo_root / "outputs" / "run_pipeline"):
            continue
        if _is_subpath(path, run_root) and key == "state_path":
            continue
        raise ValueError(f"full-image revision path is not active-revision scoped: paths.{key}={path}")


def bind_pending_diagnostics_worker_gpu_from_environment(
    run_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeState:
    """Bind one quiescent unbound diagnostics run to one explicit GPU.

    Eligible runs are either a full-image run with the complete successful
    pre-diagnostic prefix, or a non-stage-only post-mesh snapshot run before
    any diagnostic, texture, or export work has started.
    """
    resolved_run_root = _resolve_runtime_root_path(run_root)
    top_level_state_path = state_path(resolved_run_root)
    if not top_level_state_path.is_file():
        raise FileNotFoundError(f"runtime state is missing: {top_level_state_path}")

    state = load_state(resolved_run_root)
    if state.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
        raise ValueError(
            "pending diagnostics GPU binding requires the current runtime-state schema"
        )
    recorded_run_root = state.get("run_root")
    if not isinstance(recorded_run_root, str) or not recorded_run_root:
        raise ValueError("runtime state run_root must be a non-empty canonical path")
    if Path(recorded_run_root).expanduser().resolve() != resolved_run_root:
        raise ValueError("requested run_root must match state.run_root")
    recorded_state_path = state.get("state_path")
    if not isinstance(recorded_state_path, str) or not recorded_state_path:
        raise ValueError("runtime state state_path must be a non-empty canonical path")
    if Path(recorded_state_path).expanduser().resolve() != top_level_state_path:
        raise ValueError("requested run_root must match state.state_path")

    _validate_runtime_status_domains(state)
    validate_revision_path_ownership(state)
    validate_worker_isolation_paths(state)
    validate_existing_worker_isolation_collision(state)

    active_state_path = active_revision_root(state) / "state.json"
    if not active_state_path.is_file():
        raise FileNotFoundError(
            f"active-revision runtime state is missing: {active_state_path}"
        )
    active_state = json.loads(active_state_path.read_text(encoding="utf-8"))
    if not isinstance(active_state, dict):
        raise ValueError(
            f"active-revision runtime state must contain a JSON object: {active_state_path}"
        )
    _validate_runtime_status_domains(active_state)
    if active_state != state:
        raise ValueError(
            "top-level and active-revision runtime state ledgers must match before GPU binding"
        )

    run_mode = state.get("run_mode")
    if run_mode not in {"full_image", "post_mesh_processing_diagnostics"}:
        raise ValueError(
            "pending diagnostics GPU binding requires run_mode=full_image "
            "or non-stage-only post_mesh_processing_diagnostics"
        )
    if (
        run_mode == "post_mesh_processing_diagnostics"
        and state.get("diagnostics_stage_only") is not False
    ):
        raise ValueError(
            "pending diagnostics GPU binding rejects stage-only "
            "post_mesh_processing_diagnostics runs"
        )
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("enabled") is not True:
        raise ValueError("pending diagnostics GPU binding requires diagnostics.enabled=true")
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("runtime state stages must be an object")
    for stage_name in (
        *DIAGNOSTIC_STAGE_SEQUENCE,
        *POST_MESH_TEXTURE_STAGE_SEQUENCE,
        *FINAL_EXPORT_STAGE_SEQUENCE,
    ):
        stage = stages.get(stage_name)
        if (
            not isinstance(stage, dict)
            or stage.get("ok") is not False
            or stage.get("status") != "pending"
        ):
            raise ValueError(f"stage must be pending before diagnostics GPU binding: {stage_name}")
    completed = state.get("completed_global_stage_sequence")
    if _supports_diagnostic_repair_revisions(state):
        for stage_name in FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE:
            stage = stages.get(stage_name)
            if (
                not isinstance(stage, dict)
                or stage.get("ok") is not True
                or stage.get("status") != "success"
            ):
                raise ValueError(
                    "pre-diagnostic stage must be successful before diagnostics "
                    f"GPU binding: {stage_name}"
                )
        if completed != list(FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE):
            raise ValueError(
                "completed_global_stage_sequence must exactly match the successful "
                "pre-diagnostic stages"
            )
    elif completed != []:
        raise ValueError(
            "non-stage-only post-mesh diagnostics GPU binding requires an empty "
            "completed_global_stage_sequence"
        )

    for key in ("diagnostic_runtime_attachments", "diagnostic_runtime_sessions"):
        value = state.get(key, [])
        if not isinstance(value, list) or value:
            raise ValueError(f"{key} must be absent or an empty list before GPU binding")
    worker_isolation = state["worker_isolation"]
    launch_evidence = worker_isolation.get("live_launch_evidence_paths", [])
    if not isinstance(launch_evidence, list) or launch_evidence:
        raise ValueError(
            "worker_isolation.live_launch_evidence_paths must be absent or empty before GPU binding"
        )
    expected_unbound_binding = {
        "source": "unbound",
        "value": None,
        "effective_cuda_visible_devices": None,
        "conflict_checked": True,
    }
    before_gpu_binding = worker_isolation.get("gpu_binding")
    if before_gpu_binding != expected_unbound_binding:
        raise ValueError(
            "worker_isolation.gpu_binding must equal the exact unbound binding before recovery"
        )

    source_environ = os.environ if environ is None else environ
    for key in SLURM_GPU_BINDING_ENV_KEYS:
        if key not in source_environ:
            continue
        raw_slurm_value = source_environ[key]
        if not isinstance(raw_slurm_value, str) or raw_slurm_value != "":
            raise ValueError(
                f"pending diagnostics GPU binding rejects nonempty Slurm environment key: {key}"
            )
    raw_cuda_visible_devices = source_environ.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(raw_cuda_visible_devices, str):
        raise ValueError("CUDA_VISIBLE_DEVICES must be a string containing exactly one GPU token")
    if (
        not raw_cuda_visible_devices
        or "," in raw_cuda_visible_devices
        or any(character.isspace() for character in raw_cuda_visible_devices)
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one non-empty GPU token without commas or whitespace"
        )
    after_gpu_binding = {
        "source": "env",
        "value": raw_cuda_visible_devices,
        "effective_cuda_visible_devices": raw_cuda_visible_devices,
        "conflict_checked": True,
    }

    candidate = _json_clone(state)
    candidate["worker_isolation"]["gpu_binding"] = _json_clone(after_gpu_binding)
    timestamp = _utc_now()
    candidate.setdefault("history", []).append(
        {
            "timestamp": timestamp,
            "agent": "hag4r.agentic.runtime_state",
            "event": "pending_diagnostics_worker_gpu_bound_from_environment",
            "note": "Bound the pending diagnostics worker to one explicit environment GPU token.",
            "detail": {
                "before_gpu_binding": _json_clone(before_gpu_binding),
                "after_gpu_binding": _json_clone(after_gpu_binding),
            },
        }
    )
    candidate["updated_at"] = timestamp

    _validate_runtime_status_domains(candidate)
    validate_revision_path_ownership(candidate)
    validate_worker_isolation_paths(candidate)
    validate_existing_worker_isolation_collision(candidate)
    save_state(candidate, resolved_run_root)

    persisted = load_state(resolved_run_root)
    persisted_active = json.loads(active_state_path.read_text(encoding="utf-8"))
    if persisted_active != persisted:
        raise RuntimeError(
            "pending diagnostics GPU binding did not synchronize both runtime-state ledgers"
        )
    return persisted


def _validate_diagnostics_revision_snapshot(snapshot_state: RuntimeState, snapshot_root: Path) -> None:
    active_revision = str(snapshot_state.get("active_revision", "") or "")
    if not active_revision:
        raise ValueError("diagnostics revision snapshot state must record an active_revision")
    expected_snapshot_root = active_revision_root(snapshot_state)
    if snapshot_root.resolve() != expected_snapshot_root:
        raise ValueError(
            "diagnostics_revision_snapshot_dir must point to the active revision directory, not the top-level run root: "
            f"provided={snapshot_root.resolve()}, expected={expected_snapshot_root}"
        )
    missing_stages = [
        stage_name
        for stage_name in FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE
        if snapshot_state.get("stages", {}).get(stage_name, {}).get("ok") is not True
    ]
    if missing_stages:
        raise RuntimeError(
            "diagnostics revision snapshot has not completed full pre-export mesh processing: "
            + ", ".join(missing_stages)
        )
    paths = snapshot_state.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("diagnostics revision snapshot state paths must be an object")
    required_path_keys = [
        "object_description_path",
        "cleaned_image_path",
        "part_labels_path",
        "inferred_params_path",
        "mesh_processing_request_path",
        "monolithic_mesh_path",
        "monolithic_params_path",
        "volume_topology_path",
        "metric_mesh_scaling_path",
    ]
    missing_paths: list[str] = []
    for key in required_path_keys:
        value = paths.get(key)
        if not value:
            missing_paths.append(f"paths.{key}=<missing>")
            continue
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            missing_paths.append(f"paths.{key}={path}")
    omnipart_output_dir = Path(str(paths.get("omnipart_output_dir") or "")).expanduser().resolve()
    for view_name in VIEW_NAMES:
        path = omnipart_output_dir / f"mesh_combined_part_labels_{view_name}.png"
        if not path.is_file():
            missing_paths.append(str(path))
    if missing_paths:
        raise FileNotFoundError(
            "diagnostics revision snapshot is missing required pre-export artifact(s): "
            + ", ".join(missing_paths)
        )
    mesh_path = Path(str(paths["monolithic_mesh_path"])).expanduser().resolve()
    if mesh_path.suffix.lower() != ".mesh":
        raise ValueError(f"diagnostics revision snapshot monolithic_mesh_path must be .mesh: {mesh_path}")
    with np.load(Path(str(paths["monolithic_params_path"])).expanduser().resolve(), allow_pickle=False) as data:
        keys = set(data.files)
        forbidden = sorted(key for key in keys if key.startswith("tri" + "_") or "thickness" in key)
        if forbidden:
            raise ValueError("diagnostics revision snapshot params contain legacy key(s): " + ", ".join(forbidden))
        missing = sorted(key for key in ("tet_E_nu", "tet_density", "tet_part_labels") if key not in keys)
        if missing:
            raise ValueError("diagnostics revision snapshot params missing required key(s): " + ", ".join(missing))
    inferred_payload = json.loads(Path(str(paths["inferred_params_path"])).read_text(encoding="utf-8"))
    if not isinstance(inferred_payload, dict):
        raise ValueError("diagnostics revision snapshot inferred_params.json must be an object")
    legacy_hits = _find_legacy_runtime_contract_fields(inferred_payload)
    if legacy_hits:
        raise ValueError("diagnostics revision snapshot inferred params contain legacy field(s): " + ", ".join(legacy_hits))


def mark_active_revision_diagnostics_accepted(
    run_root: str | Path,
    *,
    reason: str,
) -> RuntimeState:
    state = load_state(run_root)
    validate_external_lineage(state)
    if state.get("run_mode") not in REVISION_AWARE_RUN_MODES:
        raise RuntimeError("diagnostics acceptance requires a revision-aware run")
    active_revision = str(state.get("active_revision", "") or "")
    if not active_revision:
        raise RuntimeError("cannot accept diagnostics without an active revision")
    reason = reason.strip()
    if not reason:
        raise ValueError("diagnostics acceptance reason must be non-empty")
    revision = state["revisions"][active_revision]
    if revision.get("status") != "active":
        raise RuntimeError(
            "diagnostics acceptance requires active revision status=active: "
            f"{active_revision} status={revision.get('status')}"
        )
    diagnostic_stage = state.get("stages", {}).get(DIAGNOSTIC_STAGE_SEQUENCE[0], {})
    if (
        not isinstance(diagnostic_stage, dict)
        or diagnostic_stage.get("ok") is not True
        or diagnostic_stage.get("status") != "success"
    ):
        raise RuntimeError("diagnostics acceptance requires a successful diagnostic stage")
    decision = state.get("route_state", {}).get(
        "orchestrator_diagnostic_decision", {}
    )
    if (
        not isinstance(decision, dict)
        or decision.get("diagnostic_verdict") != "accept"
        or decision.get("transition_action") != "post_mesh_texture"
    ):
        raise RuntimeError(
            "diagnostics acceptance requires a durable parent accept decision"
        )
    timestamp = _utc_now()
    revision["status"] = "diagnostics_accepted"
    revision["diagnostics_accepted_at"] = timestamp
    revision["diagnostics_accepted_reason"] = reason
    revision["updated_at"] = timestamp
    state["updated_at"] = timestamp
    validate_revision_path_ownership(state)
    save_state(state, run_root)
    return load_state(run_root)


def mark_active_revision_ready_for_export(run_root: str | Path, *, reason: str) -> RuntimeState:
    from hag4r.tools.post_mesh_texture import validate_post_mesh_texture_bundle

    state = load_state(run_root)
    validate_external_lineage(state)
    if state.get("run_mode") not in REVISION_AWARE_RUN_MODES:
        raise RuntimeError("export readiness requires a revision-aware run")
    active_revision = str(state.get("active_revision", "") or "")
    if not active_revision:
        raise RuntimeError("cannot mark export-ready without an active revision")
    reason = reason.strip()
    if not reason:
        raise ValueError("export-readiness reason must be non-empty")
    revision = state["revisions"][active_revision]
    diagnostics_enabled = bool(state.get("diagnostics", {}).get("enabled"))
    decision = state.get("route_state", {}).get("orchestrator_diagnostic_decision", {})
    persisted_policy = decision.get("diagnostic_export_policy") if isinstance(decision, dict) else None
    if persisted_policy not in (None, "", DIAGNOSTIC_EXPORT_POLICY):
        raise RuntimeError("export readiness contains an unknown diagnostic export policy")
    final_revise_policy = _is_final_revise_export_policy(state)
    if persisted_policy == DIAGNOSTIC_EXPORT_POLICY and not final_revise_policy:
        raise RuntimeError("export readiness contains an invalid final-revise export policy")
    if final_revise_policy:
        expected_source_status = "active"
        target_status = "ready_for_export_unaccepted"
    else:
        expected_source_status = "diagnostics_accepted" if diagnostics_enabled else "active"
        target_status = "ready_for_export"
    if revision.get("status") != expected_source_status:
        raise RuntimeError(
            "texture completion cannot promote this revision: "
            f"{active_revision} status={revision.get('status')}, "
            f"expected={expected_source_status}"
        )
    texture_stage_name = POST_MESH_TEXTURE_STAGE_SEQUENCE[0]
    texture_stage = state.get("stages", {}).get(texture_stage_name, {})
    if (
        not isinstance(texture_stage, dict)
        or texture_stage.get("ok") is not True
        or texture_stage.get("status") != "success"
    ):
        raise RuntimeError("export readiness requires a successful post-mesh texture stage")
    required_paths = (
        "textured_visual_mesh_path",
        "visual_to_physics_binding_path",
        "visual_manifest_path",
        "texture_qa_report_path",
    )
    missing = [
        key
        for key in required_paths
        if not Path(str(state.get("paths", {}).get(key, ""))).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "post-mesh texture required artifacts are missing: " + ", ".join(missing)
        )
    validate_post_mesh_texture_bundle(
        Path(str(state["paths"]["post_mesh_texture_dir"])),
        validate_glb=True,
    )
    timestamp = _utc_now()
    revision["status"] = target_status
    revision["ready_for_export_at"] = timestamp
    revision["ready_for_export_reason"] = reason
    revision["updated_at"] = timestamp
    route_state = state.setdefault("route_state", {})
    if isinstance(route_state, dict):
        route_state["force_rerun_stage_skill_paths"] = []
    state["updated_at"] = timestamp
    validate_revision_path_ownership(state)
    save_state(state, run_root)
    return load_state(run_root)


def resolve_export_revision(state: RuntimeState, *, explicit_reexport: bool = False) -> tuple[str, Path]:
    if (
        state.get("run_mode") not in REVISION_AWARE_RUN_MODES
        or not state.get("active_revision")
    ):
        return "", Path(str(state["run_root"])).expanduser().resolve()
    accepted_revision = state.get("accepted_revision")
    active_revision = str(state["active_revision"])
    if explicit_reexport and accepted_revision:
        revision_id = str(accepted_revision)
    else:
        revision_id = active_revision
    status = str(state.get("revisions", {}).get(revision_id, {}).get("status", ""))
    allowed_statuses = {"ready_for_export", "ready_for_export_unaccepted", "accepted"}
    if status not in allowed_statuses:
        raise RuntimeError(f"revision is not exportable: {revision_id} status={status}")
    if status == "ready_for_export_unaccepted" and not _is_final_revise_export_policy(state):
        raise RuntimeError(
            "unaccepted export requires the validated final-revise export policy"
        )
    return revision_id, revision_root(state, revision_id)


def promote_exported_revision(
    run_root: str | Path,
    revision_id: str,
    *,
    manifest_payload: dict[str, Any] | None = None,
) -> RuntimeState:
    state = load_state(run_root)
    validate_external_lineage(state)
    revisions = state.get("revisions", {})
    if not isinstance(revisions, dict) or revision_id not in revisions:
        raise KeyError(f"unknown exported revision: {revision_id}")
    previous_accepted = state.get("accepted_revision")
    source_status = revisions[revision_id].get("status")
    if source_status == "ready_for_export_unaccepted":
        if revision_id != str(state.get("active_revision", "") or ""):
            raise RuntimeError(
                "unaccepted export promotion requires the active revision: "
                f"{revision_id} active={state.get('active_revision')}"
            )
        if state.get("accepted_revision") not in (None, ""):
            raise RuntimeError(
                "unaccepted export promotion requires accepted_revision to remain empty"
            )
        if not _is_final_revise_export_policy(state):
            raise RuntimeError(
                "unaccepted export promotion requires the validated final-revise policy"
            )
        revisions[revision_id]["status"] = "exported_unaccepted"
        revisions[revision_id]["exported_at"] = _utc_now()
        revisions[revision_id]["manifest_payload"] = _to_json_friendly(manifest_payload or {})
        revisions[revision_id]["updated_at"] = _utc_now()
        state["accepted_revision"] = None
        state["exported_revision"] = revision_id
        state["updated_at"] = _utc_now()
        validate_revision_path_ownership(state)
        save_state(state, run_root)
        return state
    if source_status not in {"ready_for_export", "accepted"}:
        raise RuntimeError(
            "export promotion requires revision status ready_for_export or accepted: "
            f"{revision_id} status={source_status}"
        )
    if previous_accepted and previous_accepted != revision_id and previous_accepted in revisions:
        revisions[previous_accepted]["status"] = "superseded"
        revisions[previous_accepted]["updated_at"] = _utc_now()
    revisions[revision_id]["status"] = "accepted"
    revisions[revision_id]["accepted_at"] = _utc_now()
    revisions[revision_id]["manifest_payload"] = _to_json_friendly(manifest_payload or {})
    revisions[revision_id]["updated_at"] = _utc_now()
    state["accepted_revision"] = revision_id
    state["exported_revision"] = revision_id
    state["updated_at"] = _utc_now()
    validate_revision_path_ownership(state)
    save_state(state, run_root)
    return state


def create_repair_revision_from_diagnostic_decision(run_root: str | Path) -> RuntimeState:
    state = load_state(run_root)
    validate_external_lineage(state)
    if not _supports_diagnostic_repair_revisions(state):
        raise RuntimeError(
            "diagnostic repair revisions require a full-image run or an opt-in reroute snapshot"
        )
    route_state = state.get("route_state", {})
    decision = route_state.get("orchestrator_diagnostic_decision", {}) if isinstance(route_state, dict) else {}
    route = str(decision.get("recommendation_route") or decision.get("route") or "")
    if route not in FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE:
        raise ValueError(f"diagnostic route cannot create a repair revision: {route}")
    base_revision = str(state.get("active_revision", "") or "")
    if not base_revision:
        raise RuntimeError("cannot create repair revision without an active revision")
    new_revision = next_revision_id(state)
    expected_base_revision = f"revision_{int(new_revision.removeprefix('revision_')) - 1:04d}"
    if base_revision != expected_base_revision:
        raise RuntimeError(
            "diagnostic repair revisions must be based on the immediate previous active revision: "
            f"active={base_revision}, expected={expected_base_revision}"
        )
    base_revision_status = str(state.get("revisions", {}).get(base_revision, {}).get("status", ""))
    if base_revision_status != "active":
        raise RuntimeError(f"base revision is not eligible for diagnostic repair: {base_revision} status={base_revision_status}")
    missing_base_stages = [
        stage_name
        for stage_name in (
            *FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE,
            *DIAGNOSTIC_STAGE_SEQUENCE,
        )
        if state.get("stages", {}).get(stage_name, {}).get("ok") is not True
    ]
    if missing_base_stages:
        raise RuntimeError(
            "base revision is incomplete and cannot seed a diagnostic repair revision: "
            + ", ".join(missing_base_stages)
        )
    terminal = state.get("diagnostics", {}).get("terminal", {})
    if not isinstance(terminal, dict) or terminal.get("status") != "success":
        raise RuntimeError("base revision diagnostics have no successful terminal recommendation")
    run_root_path = Path(str(state["run_root"])).expanduser().resolve()
    prefix_stages = FULL_IMAGE_COPY_PREFIX_STAGES_BY_ROUTE[route]
    new_state = _json_clone(state)
    # Diagnostic ownership ledgers belong to the active revision's runtime
    # attempt, not to the immutable revision history.  A repair revision must
    # therefore start with empty ledgers even when the superseded revision has
    # closed attachment/session records.  The superseded revision snapshot is
    # left untouched so its historical runtime evidence remains auditable.
    new_state["diagnostic_runtime_attachments"] = []
    new_state["diagnostic_runtime_sessions"] = []
    new_state["active_revision"] = new_revision
    revisions = new_state.setdefault("revisions", {})
    revisions[base_revision]["status"] = "superseded"
    revisions[base_revision]["superseded_by"] = new_revision
    revisions[base_revision]["updated_at"] = _utc_now()
    revisions[new_revision] = _revision_metadata(
        revision_id=new_revision,
        revision_root_path=run_root_path / "revisions" / new_revision,
        status="active",
        base_revision=base_revision,
        rerun_route=route,
        copied_stage_prefix=prefix_stages,
    )
    new_state["accepted_revision"] = None
    new_state["exported_revision"] = None
    new_state["diagnostics"] = _reset_revision_diagnostics(new_state.get("diagnostics", {}))
    new_state["completed_global_stage_sequence"] = list(prefix_stages)
    new_state["stages"] = _initial_stages(
        tuple(new_state.get("planned_global_stage_sequence", ())),
        tuple(new_state.get("planned_diagnostic_stage_sequence", ())),
    )
    base_revision_root_path = active_revision_root(state)
    for stage_name in prefix_stages:
        copied_stage = _rewrite_path_strings(
            _json_clone(state["stages"][stage_name]),
            old_root=base_revision_root_path,
            new_root=run_root_path / "revisions" / new_revision,
        )
        copied_stage["status"] = "reused"
        copied_stage.setdefault("stats", {})["reused_from_revision"] = base_revision
        copied_stage.setdefault("outputs", {})["reused_from_revision"] = base_revision
        new_state["stages"][stage_name] = copied_stage
    new_state = regenerate_active_revision_paths(new_state)
    force_stage_skill_paths = FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE[route]
    old_cues = state.get("sim_diagnostic_cues") or state.get("route_state", {}).get("sim_diagnostic_cues", [])
    repair_cues = [
        _sanitized_repair_cue(cue, base_revision=base_revision)
        for cue in old_cues
        if isinstance(cue, dict)
    ]
    prior_route_state = state.get("route_state", {}) if isinstance(state.get("route_state", {}), dict) else {}
    old_brief = (
        prior_route_state.get("diagnostic_repair_brief")
        or state.get("diagnostic_repair_brief")
        or decision.get("diagnostic_repair_brief")
        or {}
    )
    repair_brief = _sanitized_repair_brief(old_brief, base_revision=base_revision)
    prior_revision_count = int(state.get("route_state", {}).get("diagnostic_revision_count", 0) or 0)
    expected_revision_count = prior_revision_count + 1
    requested_revision_count = int(decision.get("diagnostic_revision_count", 0) or 0)
    if requested_revision_count != expected_revision_count:
        raise RuntimeError(
            "diagnostic repair revision count must advance exactly once after a successful repair: "
            f"requested={requested_revision_count}, expected={expected_revision_count}"
        )
    new_state["route_state"] = {
        "diagnostic_revision_count": requested_revision_count,
        "force_rerun_stage_skill_paths": list(force_stage_skill_paths),
        "first_rerouted_stage_skill_path": force_stage_skill_paths[0] if force_stage_skill_paths else "",
        "sim_diagnostic_cues": repair_cues,
        "diagnostic_repair_brief": repair_brief,
        "active_repair_revision": {
            "base_revision": base_revision,
            "new_revision": new_revision,
            "route": route,
            "copied_stage_prefix": list(prefix_stages),
        },
    }
    new_state["sim_diagnostic_cues"] = repair_cues
    new_state["diagnostic_repair_brief"] = repair_brief
    new_state["updated_at"] = _utc_now()
    _copy_prefix_artifacts(base_state=state, new_state=new_state, prefix_stages=prefix_stages)
    validate_copied_prefix_metadata(new_state, base_revision=base_revision, new_revision=new_revision)
    validate_revision_path_ownership(new_state)
    save_state(new_state, run_root)
    return new_state


def reset_active_revision_stage_runtime_terminals(run_root: str | Path) -> RuntimeState:
    """Clear inherited terminal submissions for stages rerun by an active repair.

    ``stage_runtime`` is a run-level execution ledger.  Repair revisions clone
    the source state, so terminal submissions from the bypass/full source must
    not block the replacement stage or its downstream texture/export stages.
    Prefix stages copied as immutable provenance retain their terminal records;
    only stages outside that prefix are reopened.  The active repair metadata
    is required so this cannot be used to rewrite an ordinary completed run.
    """
    state = load_state(run_root)
    route_state = state.get("route_state")
    active_repair = route_state.get("active_repair_revision") if isinstance(route_state, dict) else None
    active_revision = str(state.get("active_revision") or "")
    if not active_revision or not isinstance(active_repair, dict):
        raise RuntimeError("active state is not a diagnostic repair revision")
    if str(active_repair.get("new_revision") or "") != active_revision:
        raise RuntimeError("active repair revision metadata does not match active_revision")
    route = str(active_repair.get("route") or "")
    if route not in FULL_IMAGE_COPY_PREFIX_STAGES_BY_ROUTE:
        raise RuntimeError(f"active repair revision has unsupported route: {route}")
    stage_runtime = state.setdefault("stage_runtime", {})
    rerun_stage_runtime_keys: dict[str, tuple[str, ...]] = {
        "segmentation": (
            "segmentation",
            "omnipart",
            "material_inference",
            "mesh_processing",
            "post_mesh_texture",
            "final_export",
        ),
        "material_inference": (
            "material_inference",
            "mesh_processing",
            "post_mesh_texture",
            "final_export",
        ),
        "mesh_processing": (
            "mesh_processing",
            "post_mesh_texture",
            "final_export",
        ),
    }
    reopened_stages: list[str] = []
    for stage_name in rerun_stage_runtime_keys[route]:
        record = stage_runtime.get(stage_name)
        if not isinstance(record, dict) or "terminal" not in record:
            continue
        record.pop("terminal", None)
        record.pop("diagnostic_response_summary", None)
        reopened_stages.append(stage_name)
    state["updated_at"] = _utc_now()
    save_state(state, run_root)
    append_history(
        run_root,
        "reset_active_revision_stage_runtime_terminals",
        "reopened inherited terminal ledgers for repair stages"
        + (": " + ", ".join(reopened_stages) if reopened_stages else ""),
        event="repair_stage_runtime_terminals_reset",
        detail={"route": route, "reopened_stages": reopened_stages},
    )
    return state


def reset_active_revision_diagnostic_runtime_ledgers(run_root: str | Path) -> RuntimeState:
    """Recover a repair revision whose inherited diagnostic ledgers are closed.

    Repair revisions own a fresh diagnostic execution namespace.  This
    migration is intentionally narrow: it only accepts an active repair
    revision with no active diagnostic payload and terminalized prior runtime
    ownership records, then persists empty active ledgers.  Historical records
    remain in the superseded revision snapshot.
    """
    state = load_state(run_root)
    active_revision = str(state.get("active_revision") or "")
    route_state = state.get("route_state")
    active_repair = route_state.get("active_repair_revision") if isinstance(route_state, dict) else None
    if not active_revision or not isinstance(active_repair, dict):
        raise RuntimeError("active state is not a diagnostic repair revision")
    if str(active_repair.get("new_revision") or "") != active_revision:
        raise RuntimeError("active repair revision metadata does not match active_revision")
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("state.diagnostics must be an object")
    if diagnostics.get("terminal") is not None or diagnostics.get("episodes"):
        raise RuntimeError("cannot reset diagnostic ledgers after active revision diagnostic execution")
    sessions = state.get("diagnostic_runtime_sessions", [])
    attachments = state.get("diagnostic_runtime_attachments", [])
    if not isinstance(sessions, list) or not isinstance(attachments, list):
        raise ValueError("diagnostic runtime ledgers must be lists")
    if any(
        not isinstance(session, Mapping)
        or session.get("lifecycle_state") not in {"closed", "closed_failed"}
        for session in sessions
    ):
        raise RuntimeError("cannot reset diagnostic ledgers while a runtime session is active")
    if any(
        not isinstance(attachment, Mapping)
        or not attachment.get("completed_at")
        for attachment in attachments
    ):
        raise RuntimeError("cannot reset diagnostic ledgers while an attachment is active")
    state["diagnostic_runtime_sessions"] = []
    state["diagnostic_runtime_attachments"] = []
    state["updated_at"] = _utc_now()
    save_state(state, run_root)
    return state


def _diagnostic_terminal(state: RuntimeState) -> dict[str, Any]:
    diagnostics = state.get("diagnostics", {})
    terminal = diagnostics.get("terminal") if isinstance(diagnostics, dict) else None
    if not isinstance(terminal, dict):
        raise RuntimeError("diagnostics completed without terminal state")
    forbidden = {"diagnostic_verdict", "diagnostic_verdict_reason", "transition_action", "final_verdict"}
    claimed = sorted(forbidden.intersection(terminal))
    if claimed:
        raise ValueError("diagnostic child terminal claims parent-owned field(s): " + ", ".join(claimed))
    return terminal


def _validated_terminal_diagnostic_recommendation(state: RuntimeState) -> dict[str, Any]:
    terminal = _diagnostic_terminal(state)
    if terminal.get("tool_name") != "submit_diagnostic_recommendation":
        raise RuntimeError("diagnostics completed without a submitted recommendation")
    if terminal.get("status") != "success" or terminal.get("validated") is not True:
        raise RuntimeError(f"diagnostic recommendation is not validated: {terminal.get('status', 'missing')}")
    recommendation = terminal.get("recommendation")
    if not isinstance(recommendation, dict):
        raise RuntimeError("validated diagnostic recommendation is missing from terminal state")
    recommendation = normalize_persisted_vlm_final_recommendation(recommendation)
    forbidden = {"diagnostic_verdict", "diagnostic_verdict_reason", "transition_action", "final_verdict", "status"}
    claimed = sorted(forbidden.intersection(recommendation))
    if claimed:
        raise ValueError("diagnostic recommendation claims parent-owned field(s): " + ", ".join(claimed))
    route = str(recommendation.get("route", ""))
    if str(terminal.get("route", "")) != route:
        raise ValueError("diagnostic terminal route does not match its recommendation")
    recommendation_ready = recommendation["recommendation"] == "accept"
    if terminal.get("ready") is not recommendation_ready:
        raise ValueError("diagnostic terminal ready flag does not match its recommendation")
    allowed_routes = {"accept", *FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE}
    if route not in allowed_routes:
        raise ValueError(f"unsupported diagnostic recommendation route: {route}")
    if route == "accept":
        if recommendation.get("recommendation") != "accept" or recommendation.get("ready") is not True:
            raise ValueError("accept diagnostic route requires recommendation=accept and ready=true")
    else:
        if recommendation.get("recommendation") != "revise":
            raise ValueError("repair diagnostic route requires recommendation=revise and ready=false")
        cues = recommendation.get("diagnostic_cues", [])
        if not isinstance(cues, list) or not any(str(cue).strip() for cue in cues):
            raise ValueError("diagnostic recommendation has no concrete diagnostic_cues")
    return _json_clone(recommendation)


def _validated_parent_diagnostic_decision(
    recommendation: dict[str, Any] | None,
    diagnostic_verdict: str | None,
    diagnostic_verdict_reason: str | None,
) -> tuple[str | None, str]:
    if recommendation is None:
        if diagnostic_verdict is not None:
            raise ValueError("terminal diagnostic failure requires diagnostic_verdict=None")
        if not isinstance(diagnostic_verdict_reason, str) or not diagnostic_verdict_reason.strip():
            raise ValueError("terminal diagnostic failure requires a nonempty parent-authored reason")
        reason = diagnostic_verdict_reason.strip()
        return None, reason
    child_recommendation = recommendation.get("recommendation")
    if child_recommendation == "revise":
        if diagnostic_verdict not in {None, "revise"}:
            raise ValueError("a validated child revision cannot be overridden to accept")
        route = str(recommendation.get("route", ""))
        cues = recommendation.get("diagnostic_cues", [])
        if route not in FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE or not any(
            isinstance(cue, str) and cue.strip() for cue in cues
        ):
            raise ValueError(
                "validated child revision requires an actionable non-accept route and diagnostic cues"
            )
        return "revise", ""
    if child_recommendation != "accept":
        raise ValueError(f"unsupported child diagnostic recommendation: {child_recommendation}")
    if diagnostic_verdict != "accept":
        raise ValueError("a validated child accept requires an explicit parent accept verdict")
    if not isinstance(diagnostic_verdict_reason, str) or not diagnostic_verdict_reason.strip():
        raise ValueError("accept requires a nonempty parent-authored diagnostic verdict reason")
    return "accept", diagnostic_verdict_reason.strip()


def _build_diagnostic_summary_fallback(
    state: RuntimeState,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    recommendation = terminal.get("recommendation")
    if not isinstance(recommendation, dict):
        recommendation = {}
    operational = terminal.get("operational_outcome")
    artifact_warning = dict(operational) if isinstance(operational, Mapping) else {}
    return {
        "summary_source": "terminal_fallback",
        "diagnostic_recommendation": _json_clone(recommendation),
        "recommended_route": recommendation.get("route"),
        "reason": recommendation.get("reason", ""),
        "diagnostic_cues": list(recommendation.get("diagnostic_cues", [])),
        "artifact_warning": artifact_warning,
        "probe_report": {
            "episode_count": len(
                [item for item in diagnostics.get("episodes", []) if isinstance(item, dict)]
            ),
            "route_adjudication": _json_clone(diagnostics.get("route_adjudication", {})),
            "probes": _json_clone(diagnostics.get("tool_results", [])),
            "coverage_summary": _json_clone(terminal.get("coverage_summary", {})),
        },
    }


def _diagnostic_summary_for_handoff(state: RuntimeState) -> tuple[dict[str, Any], str]:
    diagnostics = state.get("diagnostics", {})
    artifact_paths = diagnostics.get("artifact_paths", {}) if isinstance(diagnostics, dict) else {}
    summary_path = str(artifact_paths.get("summary", "") or "") if isinstance(artifact_paths, dict) else ""
    if not summary_path:
        summary_path = str(state.get("paths", {}).get("sim_diagnostics_report_path", "") or "")
    terminal = _diagnostic_terminal(state)
    if summary_path:
        path = Path(summary_path).expanduser().resolve()
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = None
        if isinstance(summary, dict):
            return summary, str(path)
    return _build_diagnostic_summary_fallback(state, terminal), ""


def _classify_diagnostic_terminal_outcome(
    terminal: Mapping[str, Any],
) -> tuple[str, str, bool, str]:
    business = terminal.get("business_outcome")
    operational = terminal.get("operational_outcome")
    if isinstance(business, Mapping):
        business_status = str(business.get("status", ""))
        if business_status not in {"not_authored", "recommendation_valid", "business_halt"}:
            raise ValueError(f"unknown diagnostic business outcome: {business_status}")
        operational_status = (
            str(operational.get("status", "pending"))
            if isinstance(operational, Mapping)
            else "pending"
        )
        retryable = bool(operational.get("retryable", False)) if isinstance(operational, Mapping) else False
        failure_kind = str(operational.get("failure_kind", "")) if isinstance(operational, Mapping) else ""
        return business_status, operational_status, retryable, failure_kind
    legacy_status = str(terminal.get("status", ""))
    if legacy_status == "success" and terminal.get("validated") is True:
        return "recommendation_valid", "complete", False, ""
    if legacy_status == "halted":
        return "business_halt", "complete", False, "diagnostic_terminal_halted"
    if legacy_status in {"failed", "rejected"}:
        return "legacy_failure", "complete", False, f"diagnostic_terminal_{legacy_status}"
    return "not_authored", "pending", False, ""


def _diagnostic_cue_from_recommendation(
    recommendation: dict[str, Any],
    *,
    probe_report: dict[str, Any],
    summary_path: str,
    first_rerouted_stage_skill_path: str,
) -> dict[str, Any]:
    del probe_report, summary_path, first_rerouted_stage_skill_path
    return {
        "route": str(recommendation["route"]),
        "diagnostic_cues": list(recommendation["diagnostic_cues"]),
    }


def apply_diagnostic_recommendation(
    run_root: str | Path,
    *,
    diagnostic_verdict: str | None = None,
    diagnostic_verdict_reason: str | None = None,
) -> dict[str, Any]:
    """Apply a strict accept decision or deterministically consume a child revision."""
    state = load_state(run_root)
    if not state:
        raise FileNotFoundError(f"runtime state does not exist: {state_path(run_root)}")
    validate_external_lineage(state)
    run_mode = str(state.get("run_mode", ""))
    if run_mode not in RUN_MODES:
        raise RuntimeError(f"diagnostic recommendation cannot be applied for run_mode={run_mode}")
    terminal = _diagnostic_terminal(state)
    terminal_status = str(terminal.get("status", ""))
    business_status, operational_status, operational_retryable, operational_failure_kind = (
        _classify_diagnostic_terminal_outcome(terminal)
    )
    stage = state.get("stages", {}).get(DIAGNOSTIC_STAGE_SEQUENCE[0], {})
    stage_status = str(stage.get("status", "")) if isinstance(stage, dict) else ""
    terminal_failed = business_status in {"business_halt", "legacy_failure"}
    if terminal_failed:
        recommendation = None
    elif business_status == "recommendation_valid":
        recommendation = _validated_terminal_diagnostic_recommendation(state)
        if stage_status != "success":
            raise RuntimeError("validated diagnostic recommendation requires diagnostic stage status=success")
    else:
        recommendation = None
    if business_status == "not_authored" and operational_retryable:
        parent_verdict, parent_reason = None, ""
    elif business_status == "recommendation_valid" and operational_status in {
        "cleanup_retryable",
        "lease_expired_retryable",
    }:
        parent_verdict, parent_reason = None, ""
    else:
        parent_verdict, parent_reason = _validated_parent_diagnostic_decision(
            recommendation,
            diagnostic_verdict,
            diagnostic_verdict_reason,
        )

    route_state = state.setdefault("route_state", {})
    prior_revision_count = int(route_state.get("diagnostic_revision_count", 0) or 0)
    route = str(recommendation.get("route", "")) if recommendation is not None else ""
    ready = recommendation.get("recommendation") == "accept" if recommendation is not None else False
    selected_stage_skill_paths: tuple[str, ...] = ()
    repair_brief: dict[str, Any] = {}
    failure_kind = ""
    halt_reason = ""
    diagnostic_export_policy = ""
    diagnostic_cues: list[str] = (
        list(recommendation.get("diagnostic_cues", []))
        if recommendation is not None and recommendation.get("recommendation") == "revise"
        else []
    )
    next_revision_count = prior_revision_count

    if business_status == "business_halt" or business_status == "legacy_failure":
        transition_action = "halt"
        resulting_run_status = "failed"
        failure_kind = operational_failure_kind or {
            "halted": "diagnostic_terminal_halted",
            "rejected": "diagnostic_terminal_rejected",
        }.get(terminal_status, "diagnostic_terminal_failed")
        halt_reason = str(terminal.get("error") or parent_reason)
    elif business_status == "not_authored" and operational_retryable:
        transition_action = "retry_diagnostics"
        resulting_run_status = "running"
        failure_kind = operational_failure_kind or "diagnostic_operational_retry"
    elif business_status == "recommendation_valid" and operational_status in {
        "cleanup_retryable",
        "lease_expired_retryable",
    }:
        transition_action = "retry_diagnostics_cleanup"
        resulting_run_status = "running"
        failure_kind = operational_failure_kind or "diagnostic_cleanup_retry"
    elif parent_verdict == "accept":
        if run_mode == "post_mesh_processing_diagnostics" and state.get("diagnostics_stage_only") is True:
            transition_action = "report_only"
            resulting_run_status = "success"
        else:
            transition_action = "post_mesh_texture"
            resulting_run_status = "running"
            selected_stage_skill_paths = (
                STAGE_SKILL_PATHS["post_mesh_texture"],
                STAGE_SKILL_PATHS["final_export"],
            )
    elif _supports_diagnostic_repair_revisions(state):
        max_runs = int(state.get("diagnostics", {}).get("max_runs", 1) or 1)
        if prior_revision_count >= max_runs - 1:
            scoped_mode = (
                run_mode == "post_mesh_processing_diagnostics"
                and state.get("diagnostics_reroute_and_export") is True
                and state.get("diagnostics_stage_only") is False
            )
            revisions = state.get("revisions")
            final_revision_boundary = (
                isinstance(revisions, dict)
                and max_runs >= 2
                and len(revisions) == max_runs
                and str(state.get("active_revision", "") or "")
                == f"revision_{max_runs - 1:04d}"
            )
            if (
                scoped_mode
                and final_revision_boundary
                and recommendation is not None
                and recommendation.get("recommendation") == "revise"
                and route in FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE
                and diagnostic_cues
            ):
                transition_action = "post_mesh_texture"
                resulting_run_status = "running"
                selected_stage_skill_paths = (
                    STAGE_SKILL_PATHS["post_mesh_texture"],
                    STAGE_SKILL_PATHS["final_export"],
                )
                diagnostic_export_policy = DIAGNOSTIC_EXPORT_POLICY
                cue = _diagnostic_cue_from_recommendation(
                    recommendation,
                    probe_report={},
                    summary_path="",
                    first_rerouted_stage_skill_path="",
                )
                route_state["sim_diagnostic_cues"] = [cue]
                state["sim_diagnostic_cues"] = [cue]
            else:
                transition_action = "halt"
                resulting_run_status = "failed"
                failure_kind = "revision_budget_exhausted"
                halt_reason = (
                    "diagnostic revision budget exhausted: "
                    f"prior_revision_count={prior_revision_count}, max_runs={max_runs}"
                )
        else:
            transition_action = "reroute"
            resulting_run_status = "running"
            selected_stage_skill_paths = FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE[route]
            first_rerouted_stage_skill_path = selected_stage_skill_paths[0]
            summary, summary_path = _diagnostic_summary_for_handoff(state)
            probe_report = summary.get("probe_report", {})
            if not isinstance(probe_report, dict):
                probe_report = {}
            repair_brief = build_diagnostic_repair_brief(
                recommendation=recommendation,
                probe_report=probe_report,
                summary_path=summary_path,
                first_rerouted_stage_skill_path=first_rerouted_stage_skill_path,
                source_revision=str(state.get("active_revision", "") or ""),
            )
            if not repair_brief or not repair_brief.get("diagnostic_cues"):
                raise RuntimeError("validated diagnostic recommendation did not produce a concrete repair brief")
            cue = _diagnostic_cue_from_recommendation(
                recommendation,
                probe_report=probe_report,
                summary_path=summary_path,
                first_rerouted_stage_skill_path=first_rerouted_stage_skill_path,
            )
            next_revision_count = prior_revision_count + 1
            route_state["sim_diagnostic_cues"] = [cue]
            route_state["diagnostic_repair_brief"] = repair_brief
            state["sim_diagnostic_cues"] = [cue]
            state["diagnostic_repair_brief"] = repair_brief
    else:
        transition_action = "report_only"
        resulting_run_status = "success"

    if transition_action not in TRANSITION_ACTIONS:
        raise AssertionError(f"unknown diagnostic transition action: {transition_action}")
    first_stage_skill_path = selected_stage_skill_paths[0] if selected_stage_skill_paths else ""
    decision = {
        "tool_name": "apply_diagnostic_recommendation",
        "status": "success",
        "child_recommendation": recommendation.get("recommendation") if recommendation is not None else None,
        "recommendation_route": route or None,
        "recommended_ready": ready,
        "diagnostic_verdict": parent_verdict,
        "transition_action": transition_action,
        "resulting_run_status": resulting_run_status,
        "source_revision": str(state.get("active_revision", "") or ""),
        "selected_reentry_stage_skill_paths": list(selected_stage_skill_paths),
        "first_rerouted_stage_skill_path": first_stage_skill_path if transition_action == "reroute" else "",
        "diagnostic_revision_count": next_revision_count,
        "diagnostic_repair_brief": repair_brief,
        "diagnostic_cues": diagnostic_cues,
        "failure_kind": failure_kind,
        "halt_reason": halt_reason,
    }
    if diagnostic_export_policy:
        decision["diagnostic_export_policy"] = diagnostic_export_policy
    if parent_reason:
        decision["diagnostic_verdict_reason"] = parent_reason
    route_state["orchestrator_diagnostic_decision"] = decision
    # A reroute only consumes its revision budget after the replacement revision
    # is fully created. Keep the parent ledger at the old count until then.
    route_state["diagnostic_revision_count"] = (
        prior_revision_count if transition_action == "reroute" else next_revision_count
    )
    route_state["force_rerun_stage_skill_paths"] = (
        list(selected_stage_skill_paths) if transition_action == "reroute" else []
    )
    route_state["first_rerouted_stage_skill_path"] = (
        first_stage_skill_path if transition_action == "reroute" else ""
    )
    if transition_action != "reroute":
        route_state["diagnostic_repair_brief"] = {}
        state["diagnostic_repair_brief"] = {}
    if diagnostic_export_policy and not _is_final_revise_export_policy(state, decision):
        raise RuntimeError(
            "final diagnostic revise export policy failed its durable state validation"
        )
    state["status"] = resulting_run_status
    save_state(state, run_root)

    if transition_action == "post_mesh_texture" and not diagnostic_export_policy:
        state = mark_active_revision_diagnostics_accepted(
            run_root,
            reason=parent_reason,
        )
    elif transition_action == "reroute":
        state = create_repair_revision_from_diagnostic_decision(run_root)

    state = append_history(
        run_root,
        "apply_diagnostic_recommendation",
        parent_reason or "Applied validated child revision route and diagnostic cues.",
        event="diagnostic_recommendation_applied",
        detail={
            "diagnostic_recommendation": _json_clone(recommendation) if recommendation is not None else None,
            "diagnostic_terminal": _json_clone(terminal),
            "parent_diagnostic_decision": decision,
        },
    )
    result = {
        "status": "success",
        "diagnostic_recommendation": recommendation.get("recommendation") if recommendation is not None else None,
        "recommendation_route": route or None,
        "recommended_ready": ready,
        "diagnostic_verdict": parent_verdict,
        "transition_action": transition_action,
        "run_status": resulting_run_status,
        "selected_stage_skill_paths": list(selected_stage_skill_paths),
        "first_stage_skill_path": first_stage_skill_path,
        "diagnostic_repair_brief": repair_brief,
        "diagnostic_export_policy": diagnostic_export_policy,
        "diagnostic_cues": diagnostic_cues,
        "failure_kind": failure_kind,
        "halt_reason": halt_reason,
        "active_revision": str(state.get("active_revision", "") or ""),
        "state_path": str(state_path(run_root)),
    }
    if parent_reason:
        result["diagnostic_verdict_reason"] = parent_reason
    return result


def seed_full_image_state(
    *,
    source_image: str | Path,
    run_id: str,
    run_config: RunConfig,
    repo_root: str | Path | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    fidelity: str = "medium",
    output_tag: str | None = None,
    object_description: str = "",
    object_description_path: str | Path | None = None,
    material_params_json: str | Path | None = None,
    user_hints: str = "",
    mesh_extra_args: tuple[str, ...] = (),
    sim_diagnostics_max_runs: int = 2,
    sim_diagnostics_max_episodes: int = 4,
    sim_diagnostics_max_actions_per_episode: int = 5,
    sim_diagnostics_report_path: str | Path | None = None,
    sim_diagnostics_episode_timeout_s: int = 3600,
    sim_diagnostics_live_host: str = "127.0.0.1",
    sim_diagnostics_live_port: int = 0,
    sim_diagnostics_live_ready_timeout_s: int = int(DEFAULT_READY_TIMEOUT_S),
    sim_diagnostics_live_heartbeat_ms: int = 1000,
    sim_diagnostics_live_client_lease_timeout_ms: int = 30000,
    sim_diagnostics_probe_max_vertices: int = 8,
    sim_diagnostics_probe_max_distance_m: float = 0.25,
    sim_diagnostics_probe_max_speed_m_s: float = 1.0,
    sim_diagnostics_probe_max_duration_steps: int = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
    genesis_root: str | Path | None = None,
    genesis_env_path: str | Path | None = None,
    genesis_live_command: str | None = None,
    worker_id: str | None = None,
    gpu_binding: dict[str, Any] | None = None,
    terminate_after_stage: str | None = None,
    runtime_kind: str = "skill_suite",
    runtime_entrypoint: str = "run-diaggen-pipeline",
) -> RuntimeState:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    resolved_runs_root = _resolve_runs_root(runs_root, root)
    resolved_run_root = resolved_runs_root / run_id
    diagnostics = _build_diagnostics_config(
        enabled=run_config.genesis_diagnostics.enable,
        max_runs=sim_diagnostics_max_runs,
        max_episodes=sim_diagnostics_max_episodes,
        max_actions_per_episode=sim_diagnostics_max_actions_per_episode,
        episode_timeout_s=sim_diagnostics_episode_timeout_s,
        live_host=sim_diagnostics_live_host,
        live_port=sim_diagnostics_live_port,
        live_ready_timeout_s=sim_diagnostics_live_ready_timeout_s,
        live_heartbeat_ms=sim_diagnostics_live_heartbeat_ms,
        live_client_lease_timeout_ms=sim_diagnostics_live_client_lease_timeout_ms,
        probe_max_vertices=sim_diagnostics_probe_max_vertices,
        probe_max_distance_m=sim_diagnostics_probe_max_distance_m,
        probe_max_speed_m_s=sim_diagnostics_probe_max_speed_m_s,
        probe_max_duration_steps=sim_diagnostics_probe_max_duration_steps,
        genesis_root=genesis_root or run_config.genesis_diagnostics.genesis_root or DEFAULT_GENESIS_ROOT,
        genesis_env_path=genesis_env_path or run_config.genesis_diagnostics.genesis_env_path,
        genesis_live_command=genesis_live_command or run_config.genesis_diagnostics.genesis_live_command,
    )
    if terminate_after_stage is not None and terminate_after_stage not in TERMINATE_AFTER_STAGE_OPTIONS:
        allowed = ", ".join(sorted(TERMINATE_AFTER_STAGE_OPTIONS))
        raise ValueError(f"terminate_after_stage must be one of: {allowed}")
    _validate_common(run_id=run_id, diagnostics=diagnostics)
    _assert_write_path(resolved_run_root, repo_root=root, run_root=resolved_run_root)

    resolved_source_image = _resolve_read_path(source_image, root)
    assert resolved_source_image is not None
    if not resolved_source_image.is_file():
        raise FileNotFoundError(f"source_image is missing or is not a file: {resolved_source_image}")
    object_name = _object_name_from_source(resolved_source_image)
    resolved_output_tag = _output_tag(run_id, output_tag)
    revision_id = "revision_0000"
    revision_root_path = resolved_run_root / "revisions" / revision_id
    object_description_source = _resolve_read_path(object_description_path, root) if object_description_path is not None else None
    if object_description_source is not None and not object_description_source.is_file():
        raise FileNotFoundError(f"object_description_path is missing or is not a file: {object_description_source}")
    material_params_source = _resolve_read_path(material_params_json, root)
    diagnostics_report = (
        _resolve_write_path(sim_diagnostics_report_path, root)
        if sim_diagnostics_report_path is not None
        else None
    )
    if diagnostics_report is not None and not _is_subpath(diagnostics_report, revision_root_path / "sim_diagnostics"):
        raise ValueError(
            "full-image sim_diagnostics_report_path must be inside the active revision diagnostics workspace: "
            f"{diagnostics_report}"
        )
    paths = build_full_image_revision_paths(
        repo_root=root,
        run_root=resolved_run_root,
        revision_root_path=revision_root_path,
        output_tag=resolved_output_tag,
        object_name=object_name,
        source_image=resolved_source_image,
        material_params_source_path=material_params_source,
        sim_diagnostics_report_path=diagnostics_report,
    )
    inputs = {
        "source_image": str(resolved_source_image),
        "omnipart_output_dir": None,
        "object_description": object_description,
        "object_description_path": str(object_description_source) if object_description_source is not None else None,
        "object_description_source_path": str(object_description_source) if object_description_source is not None else None,
        "material_params_json": str(material_params_source) if material_params_source else None,
        "user_hints": user_hints,
    }
    if terminate_after_stage == "image_cleanup":
        planned_global_stage_sequence = FULL_IMAGE_THROUGH_IMAGE_CLEANUP_STAGE_SEQUENCE
        planned_diagnostic_stage_sequence = ()
    elif terminate_after_stage == "omnipart":
        planned_global_stage_sequence = FULL_IMAGE_THROUGH_OMNIPART_STAGE_SEQUENCE
        planned_diagnostic_stage_sequence = ()
    elif terminate_after_stage == "material_inference":
        planned_global_stage_sequence = FULL_IMAGE_THROUGH_MATERIAL_STAGE_SEQUENCE
        planned_diagnostic_stage_sequence = ()
    else:
        planned_global_stage_sequence = (
            (
                *FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE,
                *DIAGNOSTIC_STAGE_SEQUENCE,
                *POST_MESH_TEXTURE_STAGE_SEQUENCE,
                *FINAL_EXPORT_STAGE_SEQUENCE,
            )
            if diagnostics["enabled"]
            else FULL_IMAGE_STAGE_SEQUENCE
        )
        planned_diagnostic_stage_sequence = DIAGNOSTIC_STAGE_SEQUENCE if diagnostics["enabled"] else ()
    state = _base_state(
        repo_root=root,
        runs_root=resolved_runs_root,
        run_root=resolved_run_root,
        run_id=run_id,
        run_mode="full_image",
        fidelity=fidelity,
        object_name=object_name,
        output_tag=resolved_output_tag,
        inputs=inputs,
        run_config=run_config,
        diagnostics=diagnostics,
        mesh_extra_args=mesh_extra_args,
        paths=paths,
        planned_global_stage_sequence=planned_global_stage_sequence,
        planned_diagnostic_stage_sequence=planned_diagnostic_stage_sequence,
        runtime_kind=runtime_kind,
        runtime_entrypoint=runtime_entrypoint,
        terminate_after_stage=terminate_after_stage,
    )
    state["active_revision"] = revision_id
    state["output_tag_status"] = "explicit" if output_tag else "default"
    state["accepted_revision"] = None
    state["exported_revision"] = None
    state["revisions"] = {
        revision_id: _revision_metadata(
            revision_id=revision_id,
            revision_root_path=revision_root_path,
            status="active",
        )
    }
    if terminate_after_stage is None and not diagnostics["enabled"]:
        _install_bypass_diagnostics_route(state)
    external_lineage = {
        "source_image": _file_lineage_record(resolved_source_image, role="source_image"),
    }
    if object_description_source is not None:
        external_lineage["object_description_source_path"] = _file_lineage_record(
            object_description_source,
            role="object_description_source_path",
        )
    if material_params_source is not None:
        external_lineage["material_params_json"] = _file_lineage_record(
            material_params_source,
            role="material_params_json",
        )
    state["external_lineage"] = external_lineage
    validate_revision_path_ownership(state)
    _install_worker_isolation(
        state,
        worker_id=worker_id,
        artifact_root=revision_root_path,
        gpu_binding=gpu_binding,
    )
    save_state(state, resolved_run_root)
    return state


def _cleanup_owned_diagnostics_run_root(
    run_root: Path,
    *,
    reservation_identity: tuple[int, int],
) -> None:
    try:
        current = run_root.lstat()
    except FileNotFoundError:
        return
    current_identity = (current.st_dev, current.st_ino)
    if current_identity != reservation_identity or not stat.S_ISDIR(current.st_mode):
        raise RuntimeError(
            "refusing to clean diagnostics run-root reservation because the path is no "
            f"longer the directory created by this initializer: {run_root}"
        )
    shutil.rmtree(run_root)


def _load_validated_diagnostics_revision_snapshot(
    *,
    diagnostics_revision_snapshot_dir: str | Path,
    repo_root: Path,
    allow_post_mesh_repair_snapshot: bool = False,
) -> tuple[Path, RuntimeState, dict[str, Any]]:
    snapshot_root = _resolve_read_path(
        diagnostics_revision_snapshot_dir,
        repo_root,
    )
    assert snapshot_root is not None
    if not snapshot_root.is_dir():
        raise FileNotFoundError(
            "diagnostics_revision_snapshot_dir is missing or is not a directory: "
            f"{snapshot_root}"
        )
    snapshot_state_path = snapshot_root / "state.json"
    if not snapshot_state_path.is_file():
        raise FileNotFoundError(
            f"diagnostics revision snapshot is missing state.json: {snapshot_state_path}"
        )
    snapshot_state = json.loads(snapshot_state_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot_state, dict):
        raise ValueError(
            "diagnostics revision snapshot state must be a JSON object: "
            f"{snapshot_state_path}"
        )
    source_run_mode = snapshot_state.get("run_mode")
    valid_source_modes = {"full_image"}
    if allow_post_mesh_repair_snapshot:
        valid_source_modes.add("post_mesh_processing_diagnostics")
    if source_run_mode not in valid_source_modes:
        raise ValueError(
            "diagnostics_revision_snapshot_dir must point to a full-image revision snapshot"
            " or, for diagnostics-stage-only runs, a canonical post-mesh repair revision"
        )
    snapshot_state, snapshot_preflight = (
        _preflight_diagnostics_revision_snapshot_paths(
            snapshot_state=snapshot_state,
            snapshot_root=snapshot_root,
            repo_root=repo_root,
        )
    )
    snapshot_state = _normalize_diagnostics_snapshot_stage_aliases(
        snapshot_state,
        snapshot_preflight,
    )
    _validate_diagnostics_revision_snapshot(snapshot_state, snapshot_root)
    return snapshot_root, snapshot_state, snapshot_preflight


def _preflight_diagnostics_snapshot_before_run_root_reservation(
    *,
    diagnostics_revision_snapshot_dir: str | Path,
    repo_root: str | Path | None = None,
    diagnostics_stage_only: bool = False,
    **_: Any,
) -> None:
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else _repo_root()
    )
    _, snapshot_state, _ = _load_validated_diagnostics_revision_snapshot(
        diagnostics_revision_snapshot_dir=diagnostics_revision_snapshot_dir,
        repo_root=root,
        allow_post_mesh_repair_snapshot=diagnostics_stage_only,
    )
    if not diagnostics_stage_only:
        from hag4r.tools.post_mesh_texture import load_appearance_rebake_evidence

        appearance = snapshot_state.get("paths", {}).get("omnipart_appearance_manifest_path")
        if not appearance:
            raise FileNotFoundError(
                "non-stage-only diagnostics snapshot is missing "
                "paths.omnipart_appearance_manifest_path"
            )
        load_appearance_rebake_evidence(Path(str(appearance)))


def _with_new_diagnostics_run_root_reservation(
    *,
    pre_reservation: Any | None = None,
) -> Any:
    def decorate(function: Any) -> Any:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> RuntimeState:
            if not kwargs.get("require_new_run_root", False):
                return function(*args, **kwargs)
            if pre_reservation is not None:
                pre_reservation(*args, **kwargs)
            run_id = str(kwargs["run_id"])
            if not run_id.strip():
                raise ValueError("run_id must be non-empty")
            repo_value = kwargs.get("repo_root")
            root = (
                Path(repo_value).expanduser().resolve()
                if repo_value is not None
                else _repo_root()
            )
            resolved_runs_root = _resolve_runs_root(
                kwargs.get("runs_root", DEFAULT_RUNS_ROOT),
                root,
            )
            run_root = (resolved_runs_root / run_id).resolve()
            _assert_write_path(run_root, repo_root=root)
            run_root.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_root.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    "diagnostics-only initializer requires a new run root; "
                    f"refusing to reuse existing path: {run_root}"
                ) from exc
            reserved = run_root.lstat()
            reservation_identity = (reserved.st_dev, reserved.st_ino)
            try:
                return function(*args, **kwargs)
            except BaseException as error:
                try:
                    _cleanup_owned_diagnostics_run_root(
                        run_root,
                        reservation_identity=reservation_identity,
                    )
                except Exception as cleanup_error:
                    error.add_note(
                        f"owned run-root reservation cleanup failed: {cleanup_error}"
                    )
                raise

        return wrapped

    return decorate


@_with_new_diagnostics_run_root_reservation(
    pre_reservation=_preflight_diagnostics_snapshot_before_run_root_reservation
)
def seed_post_mesh_processing_diagnostics_state(
    *,
    diagnostics_revision_snapshot_dir: str | Path,
    diagnostics_stage_only: bool = False,
    diagnostics_reroute_and_export: bool = False,
    require_new_run_root: bool = False,
    run_id: str,
    run_config: RunConfig,
    repo_root: str | Path | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    output_tag: str | None = None,
    sim_diagnostics_max_runs: int = 2,
    sim_diagnostics_max_episodes: int = 4,
    sim_diagnostics_max_actions_per_episode: int = 5,
    sim_diagnostics_report_path: str | Path | None = None,
    sim_diagnostics_episode_timeout_s: int = 3600,
    sim_diagnostics_live_host: str = "127.0.0.1",
    sim_diagnostics_live_port: int = 0,
    sim_diagnostics_live_ready_timeout_s: int = int(DEFAULT_READY_TIMEOUT_S),
    sim_diagnostics_live_heartbeat_ms: int = 1000,
    sim_diagnostics_live_client_lease_timeout_ms: int = 30000,
    sim_diagnostics_probe_max_vertices: int = 8,
    sim_diagnostics_probe_max_distance_m: float = 0.25,
    sim_diagnostics_probe_max_speed_m_s: float = 1.0,
    sim_diagnostics_probe_max_duration_steps: int = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
    genesis_root: str | Path | None = None,
    genesis_env_path: str | Path | None = None,
    genesis_live_command: str | None = None,
    worker_id: str | None = None,
    gpu_binding: dict[str, Any] | None = None,
    runtime_kind: str = "skill_suite",
    runtime_entrypoint: str = "run-diaggen-pipeline",
) -> RuntimeState:
    if diagnostics_stage_only and diagnostics_reroute_and_export:
        raise ValueError(
            "diagnostics_stage_only and diagnostics_reroute_and_export are mutually exclusive"
        )
    if not run_config.genesis_diagnostics.enable:
        raise ValueError("post_mesh_processing_diagnostics requires Genesis diagnostics to be enabled in the run config")
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    resolved_runs_root = _resolve_runs_root(runs_root, root)
    resolved_run_root = resolved_runs_root / run_id

    snapshot_root, snapshot_state, snapshot_preflight = (
        _load_validated_diagnostics_revision_snapshot(
            diagnostics_revision_snapshot_dir=diagnostics_revision_snapshot_dir,
            repo_root=root,
            allow_post_mesh_repair_snapshot=diagnostics_stage_only,
        )
    )
    if not diagnostics_stage_only:
        from hag4r.tools.post_mesh_texture import load_appearance_rebake_evidence

        snapshot_appearance = snapshot_state.get("paths", {}).get(
            "omnipart_appearance_manifest_path"
        )
        if not snapshot_appearance:
            raise FileNotFoundError(
                "non-stage-only diagnostics snapshot is missing "
                "paths.omnipart_appearance_manifest_path"
            )
        load_appearance_rebake_evidence(Path(str(snapshot_appearance)))

    diagnostics = _build_diagnostics_config(
        enabled=run_config.genesis_diagnostics.enable,
        max_runs=sim_diagnostics_max_runs,
        max_episodes=sim_diagnostics_max_episodes,
        max_actions_per_episode=sim_diagnostics_max_actions_per_episode,
        episode_timeout_s=sim_diagnostics_episode_timeout_s,
        live_host=sim_diagnostics_live_host,
        live_port=sim_diagnostics_live_port,
        live_ready_timeout_s=sim_diagnostics_live_ready_timeout_s,
        live_heartbeat_ms=sim_diagnostics_live_heartbeat_ms,
        live_client_lease_timeout_ms=sim_diagnostics_live_client_lease_timeout_ms,
        probe_max_vertices=sim_diagnostics_probe_max_vertices,
        probe_max_distance_m=sim_diagnostics_probe_max_distance_m,
        probe_max_speed_m_s=sim_diagnostics_probe_max_speed_m_s,
        probe_max_duration_steps=sim_diagnostics_probe_max_duration_steps,
        genesis_root=genesis_root or run_config.genesis_diagnostics.genesis_root or DEFAULT_GENESIS_ROOT,
        genesis_env_path=genesis_env_path or run_config.genesis_diagnostics.genesis_env_path,
        genesis_live_command=genesis_live_command or run_config.genesis_diagnostics.genesis_live_command,
    )
    fidelity = str(snapshot_state.get("fidelity", "medium"))
    _validate_common(run_id=run_id, diagnostics=diagnostics)
    _assert_write_path(resolved_run_root, repo_root=root, run_root=resolved_run_root)
    snapshot_input_sources = {
        key: Path(str(snapshot_state["paths"][key])).expanduser().resolve()
        for key in DIAGNOSTICS_SNAPSHOT_CANONICAL_INPUT_PATH_KEYS
    }
    revision_id = "revision_0000"
    revision_root_path = resolved_run_root / "revisions" / revision_id
    if not require_new_run_root and revision_root_path.exists():
        shutil.rmtree(revision_root_path)
    shutil.copytree(snapshot_root, revision_root_path)
    copied_state_path = revision_root_path / "state.json"
    copied_state_path.unlink(missing_ok=True)

    state = _rewrite_path_strings(_json_clone(snapshot_state), old_root=snapshot_root, new_root=revision_root_path)
    object_name = str(state.get("object_name", snapshot_root.name))
    resolved_output_tag = (
        output_tag
        or (run_id if diagnostics_reroute_and_export else str(state.get("output_tag") or _output_tag(run_id, None)))
    )
    sim_diagnostics_dir = revision_root_path / "sim_diagnostics"
    shutil.rmtree(sim_diagnostics_dir, ignore_errors=True)
    diagnostics_report = (
        _resolve_write_path(sim_diagnostics_report_path, root)
        if sim_diagnostics_report_path is not None
        else sim_diagnostics_dir / "diagnostic_summary.json"
    )
    if not _is_subpath(diagnostics_report, sim_diagnostics_dir):
        raise ValueError(
            "post_mesh_processing_diagnostics sim_diagnostics_report_path must be inside the diagnostics revision workspace: "
            f"{diagnostics_report}"
        )
    snapshot_paths = state.get("paths", {})
    if not isinstance(snapshot_paths, dict):
        raise ValueError("diagnostics revision snapshot state paths must be an object")
    source_image_raw = snapshot_paths.get("source_image")
    if not source_image_raw:
        raise ValueError(
            "diagnostics revision snapshot state is missing paths.source_image"
        )
    material_source_raw = (
        state.get("inputs", {}).get("material_params_json")
        or snapshot_paths.get("material_params_source_path")
    )
    paths = build_full_image_revision_paths(
        repo_root=root,
        run_root=resolved_run_root,
        revision_root_path=revision_root_path,
        output_tag=resolved_output_tag,
        object_name=object_name,
        source_image=Path(str(source_image_raw)).expanduser().resolve(),
        material_params_source_path=(
            Path(str(material_source_raw)).expanduser().resolve()
            if material_source_raw
            else None
        ),
        sim_diagnostics_report_path=diagnostics_report,
    )
    snapshot_input_lineage: dict[str, dict[str, Any]] = {}
    for key, source in snapshot_input_sources.items():
        destination = Path(str(paths[key])).expanduser().resolve()
        _copy_file_atomic(source, destination)
        snapshot_input_lineage[f"diagnostics_snapshot_{key}"] = (
            _copied_file_lineage_record(
                source,
                destination,
                role=f"diagnostics_snapshot_{key}",
            )
        )
    post_mesh_texture_dir = Path(str(paths["post_mesh_texture_dir"]))
    shutil.rmtree(post_mesh_texture_dir, ignore_errors=True)
    inputs = {
        "source_image": None,
        "diagnostics_revision_snapshot_dir": str(snapshot_root),
    }
    planned_global_stage_sequence = (
        DIAGNOSTIC_STAGE_SEQUENCE
        if diagnostics_stage_only
        else (
            POST_MESH_PROCESSING_DIAGNOSTICS_REROUTE_AND_EXPORT_STAGE_SEQUENCE
            if diagnostics_reroute_and_export
            else POST_MESH_PROCESSING_DIAGNOSTICS_STAGE_SEQUENCE
        )
    )
    now = _utc_now()
    state.update(
        {
            "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
            "runtime_kind": runtime_kind,
            "runtime_entrypoint": runtime_entrypoint,
            "repo_root": str(root),
            "runs_root": str(resolved_runs_root),
            "run_root": str(resolved_run_root),
            "state_path": str(state_path(resolved_run_root)),
            "run_id": run_id,
            "run_mode": "post_mesh_processing_diagnostics",
            "diagnostics_stage_only": diagnostics_stage_only,
            "diagnostics_reroute_and_export": diagnostics_reroute_and_export,
            "status": "seeded",
            "created_at": now,
            "updated_at": now,
            "fidelity": fidelity,
            "object_name": object_name,
            "output_tag": resolved_output_tag,
            "inputs": _to_json_friendly(inputs),
            "run_config": _to_json_friendly(run_config_audit_state(run_config)),
            "runtime_skill_suite": _to_json_friendly(_runtime_skill_suite_state(root)),
            "diagnostics": _to_json_friendly(diagnostics),
            "diagnostic_runtime_attachments": [],
            "diagnostic_runtime_sessions": [],
            "paths": _to_json_friendly(paths),
            "planned_global_stage_sequence": list(planned_global_stage_sequence),
            "planned_diagnostic_stage_sequence": list(DIAGNOSTIC_STAGE_SEQUENCE),
            "completed_global_stage_sequence": (
                list(FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE)
                if diagnostics_reroute_and_export
                else []
            ),
            "stages": _initial_stages(planned_global_stage_sequence, DIAGNOSTIC_STAGE_SEQUENCE),
            "history": [],
            "runtime_events": [],
            "route_state": {},
            "sim_diagnostic_cues": [],
            "diagnostic_repair_brief": {},
            "final_export": {},
            "active_revision": revision_id,
            "accepted_revision": None,
            "exported_revision": None,
            "revisions": {
                revision_id: _revision_metadata(
                    revision_id=revision_id,
                    revision_root_path=revision_root_path,
                    status="active",
                    base_revision=str(snapshot_state.get("active_revision") or snapshot_root.name),
                )
            },
            "external_lineage": {
                "diagnostics_revision_snapshot_dir": _directory_lineage_record(
                    snapshot_root,
                    role="diagnostics_revision_snapshot_dir",
                ),
                "diagnostics_snapshot_preflight": _to_json_friendly(snapshot_preflight),
                **snapshot_input_lineage,
            },
        }
    )
    if diagnostics_reroute_and_export:
        source_stages = snapshot_state.get("stages", {})
        if not isinstance(source_stages, dict):
            raise ValueError("diagnostics revision snapshot state stages must be an object")
        for stage_name in FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE:
            source_stage = source_stages.get(stage_name)
            if not isinstance(source_stage, dict):
                raise ValueError(f"diagnostics revision snapshot is missing stage payload: {stage_name}")
            imported_stage = _rewrite_path_strings(
                _json_clone(source_stage), old_root=snapshot_root, new_root=revision_root_path
            )
            imported_stage["ok"] = True
            imported_stage["status"] = "success"
            state["stages"][stage_name] = imported_stage
        state["revisions"][revision_id]["base_revision"] = str(
            snapshot_state.get("active_revision") or snapshot_root.name
        )
        state["final_export"] = _final_export_payload(
            paths, include_cleanup=True, include_diagnostics=True
        )
    validate_revision_path_ownership(state)
    _install_worker_isolation(
        state,
        worker_id=worker_id,
        artifact_root=revision_root_path,
        gpu_binding=gpu_binding,
    )
    save_state(state, resolved_run_root)
    return state


__all__ = [
    "BYPASS_DIAGNOSTICS_ROUTE",
    "DIAGNOSTIC_STAGE_STATUSES",
    "DIAGNOSTIC_STAGE_SEQUENCE",
    "DIAGNOSTIC_VERDICTS",
    "DIAGNOSTIC_EXPORT_POLICY",
    "FINAL_EXPORT_STAGE_SEQUENCE",
    "FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE",
    "FULL_IMAGE_REPAIR_STAGE_SKILLS_BY_ROUTE",
    "FULL_IMAGE_STAGE_SEQUENCE",
    "FULL_IMAGE_THROUGH_IMAGE_CLEANUP_STAGE_SEQUENCE",
    "FULL_IMAGE_THROUGH_MATERIAL_STAGE_SEQUENCE",
    "POST_MESH_PROCESSING_DIAGNOSTICS_STAGE_SEQUENCE",
    "POST_MESH_PROCESSING_DIAGNOSTICS_REROUTE_AND_EXPORT_STAGE_SEQUENCE",
    "POST_MESH_TEXTURE_STAGE_SEQUENCE",
    "RUNTIME_STATE_SCHEMA_VERSION",
    "REVISION_STATUSES",
    "REVISION_AWARE_RUN_MODES",
    "RUN_STATUSES",
    "RuntimeState",
    "TRANSITION_ACTIONS",
    "WORKER_ISOLATION_SCHEMA_VERSION",
    "accepted_revision_root",
    "active_revision_root",
    "append_history",
    "apply_diagnostic_recommendation",
    "bind_pending_diagnostics_worker_gpu_from_environment",
    "build_full_image_revision_paths",
    "build_worker_gpu_binding",
    "create_repair_revision_from_diagnostic_decision",
    "initialize_full_image_revision",
    "_is_final_revise_export_policy",
    "load_state",
    "mark_active_revision_ready_for_export",
    "mark_active_revision_diagnostics_accepted",
    "next_revision_id",
    "record_stage",
    "record_tool_call",
    "regenerate_active_revision_paths",
    "reset_active_revision_diagnostic_runtime_ledgers",
    "reset_active_revision_stage_runtime_terminals",
    "refresh_active_revision_snapshot",
    "resolve_export_revision",
    "save_state",
    "seed_full_image_state",
    "seed_post_mesh_processing_diagnostics_state",
    "stage_done",
    "state_path",
    "update_state",
    "validate_copied_prefix_metadata",
    "validate_external_lineage",
    "validate_existing_worker_isolation_collision",
    "validate_worker_isolation_paths",
    "validate_revision_path_ownership",
    "validate_single_cuda_visible_devices",
    "worker_cuda_visible_devices",
    "worker_id_from_environment",
    "revision_root",
]
