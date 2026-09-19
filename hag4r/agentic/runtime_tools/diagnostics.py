from __future__ import annotations
import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from PIL import Image, ImageDraw
from hag4r.agentic.genesis_vlm_schemas import (
    GenesisVlmActionLimits,
    GenesisVlmSchemaError,
    VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION,
    VLM_ACTION_SCHEMA_VERSION,
    VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION,
    VLM_PRE_EPISODE_SCHEMA_VERSION,
    VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION,
    validate_diagnostic_anchor_target_edit,
    validate_diagnostic_anchor_target_intent,
    validate_diagnostic_anchor_region,
    validate_diagnostic_probe_target_edit,
    validate_diagnostic_probe_target_intent,
    validate_diagnostic_session_plan,
    normalize_persisted_vlm_final_recommendation,
    validate_vlm_action_decision,
    validate_vlm_final_recommendation,
    validate_vlm_pre_episode_plan,
    validate_vlm_reflection_evidence,
    validate_source_semantic_material_audit,
)
from hag4r.agentic.diagnostic_force_limited_controller import (
    DIAGNOSTIC_FORCE_LIMITED_POLICY_ID,
    calibration_provenance as diagnostic_force_limited_calibration_provenance,
    completed_controller_telemetry,
    controller_state_from_action,
    force_limited_probe_schedule,
    policy_hash as diagnostic_force_limited_policy_hash,
    policy_payload as diagnostic_force_limited_policy_payload,
    validate_controller_policy,
)
from hag4r.tools.genesis.diagnostic_timing import (
    DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
    DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
    DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
    DIAGNOSTIC_RENDER_FPS,
    DIAGNOSTIC_SCENE_TIMESTEP_S,
    min_diagnostic_captured_frames,
)
from hag4r.agentic.runtime_state import (
    STAGE_SKILL_PATHS,
    append_history,
    load_state,
    record_stage,
    save_state,
    state_path,
)
from hag4r.agentic.state import ObservationSignal, SimDiagnosticRoute
from hag4r.tools.genesis.config import (
    build_agentic_genesis_config_bundle,
    diagnostic_asset_geometry_payload,
    read_mesh_vertex_geometry,
)
from hag4r.tools.genesis.live_protocol import STATUS_UNSUPPORTED
from hag4r.tools.genesis.live_client import (
    ADAPTIVE_COMPILED_PROBE_RESUME_SCHEMA_VERSION,
    CompletedProbeMeasurement,
    GENESIS_RUNTIME_LOG_MAX_LINE_CHARS,
    GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS,
    GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS,
    GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM,
    GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM,
    GENESIS_RUNTIME_LOG_SCHEMA_VERSION,
)
from hag4r.tools.part_grounding import (
    build_part_grounding_context,
    compact_part_grounding_table,
    render_part_grounding_markdown_table,
)
from hag4r.tools.probe_targeting import (
    MIN_GRABBED_SELECTED_PART_FRACTION,
    PROBE_TARGET_COMPILE_SCHEMA_VERSION,
    PROBE_TARGET_EDIT_SCHEMA_VERSION,
    compile_anchor_target_from_context,
    compile_probe_target_from_context,
    mesh_box_to_env_aabb,
    render_anchor_target_preview,
    render_probe_target_preview,
    validate_grabbed_vertices_against_part,
)
from hag4r.tools.triple_view_evidence import (
    TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION,
    TRIPLE_VIEW_PANEL_ORDER,
    sha256_file,
    stitch_triptych,
    triple_view_cameras,
    write_triple_view_manifest,
)
DIAGNOSTIC_STAGE_NAME = "genesis_live_diagnostic_loop"
DIAGNOSTIC_TERMINAL_TOOL_NAMES = frozenset({"submit_diagnostic_recommendation", "halt_diagnostics"})
DIAGNOSTIC_SETUP_SCHEMA_VERSION = "hag4r-diagnostic-setup-trial-v1"
DIAGNOSTIC_ANCHOR_SETUP_VALIDATION_SCHEMA_VERSION = "hag4r-diagnostic-anchor-setup-validation-v1"
DIAGNOSTIC_SETUP_VERDICTS = frozenset({"accept_setup", "revise_setup", "proceed_with_concerns"})
DIAGNOSTIC_SETUP_UNCERTAINTY = frozenset({"low", "medium", "high"})
MAX_DIAGNOSTIC_SETUP_TRIALS = 3
MAX_DIAGNOSTIC_COMPILED_PROBES_PER_EPISODE = 3
LiveToolHandler = Callable[..., dict[str, Any]]


def _completed_default_probe_window(
    *,
    requested_steps: Any,
    completed_steps: Any,
    runtime_step_adaptation: Any,
) -> bool:
    """Return whether runtime completed one default compiled-probe window.

    The public/default request remains exactly 100 steps.  A runtime-owned
    motion adaptation may dispatch a longer window, but only its exact,
    schema-tagged effective duration can satisfy the completion gate.
    """
    if (
        isinstance(requested_steps, bool)
        or not isinstance(requested_steps, int)
        or requested_steps != DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
        or isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
    ):
        return False
    if runtime_step_adaptation is None:
        return completed_steps == DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
    if not isinstance(runtime_step_adaptation, Mapping):
        return False
    applied = runtime_step_adaptation.get("applied")
    effective_steps = runtime_step_adaptation.get("effective_steps")
    return bool(
        runtime_step_adaptation.get("schema_version")
        == ADAPTIVE_COMPILED_PROBE_RESUME_SCHEMA_VERSION
        and runtime_step_adaptation.get("action_type") == "compiled_probe"
        and runtime_step_adaptation.get("requested_steps")
        == DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
        and isinstance(applied, bool)
        and not isinstance(effective_steps, bool)
        and isinstance(effective_steps, int)
        and effective_steps >= DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
        and completed_steps == effective_steps
        and (applied or effective_steps == DEFAULT_DIAGNOSTIC_SIMULATE_STEPS)
    )


def _completed_force_limited_probe_window(
    *,
    application: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool:
    """Check a diagnostics-owned fixed command window, never visual response."""

    schedule = application.get("force_limited_schedule")
    if not isinstance(schedule, Mapping):
        return False
    load_steps = schedule.get("load_steps")
    requested_steps = application.get("requested_duration_steps")
    completed_steps = payload.get("steps_completed")
    return bool(
        isinstance(load_steps, int)
        and not isinstance(load_steps, bool)
        and load_steps > 0
        and requested_steps == load_steps
        and completed_steps == load_steps
        and payload.get("runtime_step_adaptation") is None
    )


def _episode_force_limited_policy(episode: Mapping[str, Any]) -> bool:
    """Return true only for episode suites created by the calibrated v1 path."""

    return (
        episode.get("force_limited_controller_policy") == diagnostic_force_limited_policy_payload()
        and episode.get("force_limited_calibration_provenance")
        == diagnostic_force_limited_calibration_provenance()
    )
LIVE_TOOL_HANDLERS: dict[str, LiveToolHandler] = {}
SEGMENTATION_REENTRY_STAGE_SKILL_PATHS = (
    STAGE_SKILL_PATHS["segmentation"],
    STAGE_SKILL_PATHS["omnipart"],
    STAGE_SKILL_PATHS["material_inference"],
    STAGE_SKILL_PATHS["mesh_processing"],
)
MAX_INLINE_LIVE_RESULT_CHARS = 4096
LIVE_VISUAL_TOOL_NAMES = frozenset({"simulation_reset", "simulate"})
LIVE_TOOL_SUCCESS_STATUSES = frozenset({"ok", "applied"})
LIVE_VISUAL_SOURCE_KIND_BY_TOOL = {
    "simulation_reset": "live_reset_part_segmentation_observation",
    "simulate": "live_simulate_part_segmentation_observation",
}
_ACTIVE_DIAGNOSTIC_RUN_ROOT: ContextVar[str | None] = ContextVar("active_diagnostic_run_root", default=None)
_ACTIVE_LIVE_TOOL_HANDLERS: ContextVar[Mapping[str, LiveToolHandler] | None] = ContextVar(
    "active_live_tool_handlers",
    default=None,
)
_TERMINAL_LOCKS: dict[str, threading.Lock] = {}
_TERMINAL_LOCKS_GUARD = threading.Lock()
def _normalize_run_root(run_root: str | Path) -> str:
    return str(Path(run_root).expanduser().resolve())
def active_diagnostic_run_root() -> str | None:
    return _ACTIVE_DIAGNOSTIC_RUN_ROOT.get()
@contextmanager
def bind_active_diagnostic_run(run_root: str | Path) -> Iterator[str]:
    normalized = _normalize_run_root(run_root)
    token = _ACTIVE_DIAGNOSTIC_RUN_ROOT.set(normalized)
    try:
        yield normalized
    finally:
        _ACTIVE_DIAGNOSTIC_RUN_ROOT.reset(token)
def active_live_tool_handlers() -> Mapping[str, LiveToolHandler]:
    handlers = _ACTIVE_LIVE_TOOL_HANDLERS.get()
    if handlers is not None:
        return handlers
    return LIVE_TOOL_HANDLERS
@contextmanager
def bind_live_tool_handlers(handlers: Mapping[str, LiveToolHandler]) -> Iterator[Mapping[str, LiveToolHandler]]:
    bound_handlers = dict(handlers)
    token = _ACTIVE_LIVE_TOOL_HANDLERS.set(bound_handlers)
    try:
        yield bound_handlers
    finally:
        _ACTIVE_LIVE_TOOL_HANDLERS.reset(token)
@contextmanager
def _terminal_decision_lock(run_root: str | Path) -> Iterator[None]:
    normalized = _normalize_run_root(run_root)
    with _TERMINAL_LOCKS_GUARD:
        lock = _TERMINAL_LOCKS.setdefault(normalized, threading.Lock())
    with lock:
        yield
def _require_bound_diagnostic_run_root(run_root: str | Path) -> str:
    active = active_diagnostic_run_root()
    if active is None:
        raise RuntimeError("diagnostic tool called without active diagnostic run binding")
    requested = _normalize_run_root(run_root)
    if requested != active:
        # The active run_root binding is the source of truth; the run_root the
        # LLM echoes into its tool args is just a label. Weaker open models
        # (observed with Kimi-K2.6) occasionally introduce typos like
        # ".../class_174_pototato_instance_..." vs ".../class_174_potato_..."
        # which would otherwise tank the whole diagnostic loop. We log the
        # mismatch and use the bound run_root, which is what the worker
        # actually launched with.
        import logging
        logging.getLogger(__name__).warning(
            "diagnostic tool run_root mismatch (using active binding): "
            "active=%s requested=%s",
            active,
            requested,
        )
    return active
def _require_state(run_root: str | Path) -> dict[str, Any]:
    state = load_state(run_root)
    if not state:
        raise FileNotFoundError(f"runtime state does not exist: {state_path(run_root)}")
    if not state.get("diagnostics", {}).get("enabled"):
        raise ValueError("Genesis diagnostics are disabled for this runtime state.")
    return state
def _diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    diagnostics = state.setdefault("diagnostics", {})
    diagnostics.setdefault("diagnostic_session_plans", [])
    diagnostics.setdefault("pre_episode_plans", [])
    diagnostics.setdefault("episodes", [])
    diagnostics.setdefault("active_episode_index", None)
    diagnostics.setdefault("tool_results", [])
    diagnostics.setdefault("expected_observations", [])
    diagnostics.setdefault("actual_observations", [])
    diagnostics.setdefault("visual_evidence", [])
    diagnostics.setdefault("triple_view_evidence", [])
    diagnostics.setdefault("turn_based_violations", [])
    diagnostics.setdefault("agent_attempts", [])
    diagnostics.setdefault("setup_trials", [])
    diagnostics.setdefault("active_setup_trial_by_anchor", {})
    diagnostics.setdefault("final_setup_trial_by_anchor", {})
    diagnostics.setdefault("geometry_bounds_measurements", [])
    diagnostics.setdefault("geometry_context_measurements", [])
    diagnostics.setdefault("live_geometry_context_measurements", [])
    diagnostics.setdefault("part_grounding_context", None)
    diagnostics.setdefault("part_grounding_context_measurements", [])
    diagnostics.setdefault("part_grounding_artifacts", [])
    diagnostics.setdefault("anchor_target_intents", [])
    diagnostics.setdefault("active_anchor_target_intent_by_anchor", {})
    diagnostics.setdefault("compiled_anchor_targets", [])
    diagnostics.setdefault("active_compiled_anchor_target_by_anchor", {})
    diagnostics.setdefault("anchor_target_edit_trials", [])
    diagnostics.setdefault("anchor_target_preview_artifacts", [])
    diagnostics.setdefault("probe_target_intents", [])
    diagnostics.setdefault("active_probe_target_intent_id", None)
    diagnostics.setdefault("compiled_probe_targets", [])
    diagnostics.setdefault("active_compiled_probe_target_id", None)
    diagnostics.setdefault("probe_target_edit_trials", [])
    diagnostics.setdefault("probe_target_preview_artifacts", [])
    diagnostics.setdefault("probe_target_validations", [])
    diagnostics.setdefault("probe_target_media_validations", [])
    diagnostics.setdefault("probe_target_applications", [])
    diagnostics.setdefault("source_semantic_material_audits", [])
    diagnostics.setdefault("active_source_semantic_material_audit_id", None)
    diagnostics.setdefault("source_semantic_material_artifact_registration", None)
    diagnostics.setdefault("source_semantic_material_artifact_registrations", [])
    diagnostics.setdefault("active_source_semantic_material_artifact_registration_id", None)
    # v2 owns every executable identity separately from the semantic plan.
    diagnostics.setdefault("diagnostic_region_ledger", [])
    diagnostics.setdefault("active_region_id", None)
    diagnostics.setdefault("region_episode_ids", {})
    diagnostics.setdefault("region_probe_intent_ids", {})
    diagnostics.setdefault("region_current_probe_compile_ids", {})
    diagnostics.setdefault("anchor_probe_separation_results", [])
    diagnostics.setdefault("diagnostic_probe_attempts", [])
    diagnostics.setdefault("probe_target_reflections", [])
    diagnostics.setdefault("episode_concerns", [])
    diagnostics.setdefault("region_settlements", [])
    diagnostics.setdefault("paired_comparison_policies", [])
    diagnostics.setdefault("pair_direction_locks", [])
    diagnostics.setdefault("probe_measurements", [])
    diagnostics.setdefault("paired_comparison_records", [])
    diagnostics.setdefault("coverage_summary", None)
    diagnostics.setdefault("diagnostic_attempt_archives", [])
    diagnostics.setdefault("active_diagnostic_attempt_id", None)
    active_audit_id = diagnostics.get("active_source_semantic_material_audit_id")
    if active_audit_id is not None:
        active_revision_id = str(state.get("active_revision") or "")
        if not any(
            isinstance(audit, Mapping)
            and audit.get("audit_id") == active_audit_id
            and audit.get("active_revision_id") == active_revision_id
            for audit in diagnostics["source_semantic_material_audits"]
        ):
            diagnostics["active_source_semantic_material_audit_id"] = None
    return diagnostics


_FRESH_UNPLANNED_EMPTY_LIST_KEYS = frozenset({
    "diagnostic_session_plans", "pre_episode_plans", "episodes", "tool_results", "expected_observations",
    "actual_observations", "visual_evidence", "triple_view_evidence", "turn_based_violations", "agent_attempts",
    "setup_trials", "geometry_bounds_measurements", "live_geometry_context_measurements", "part_grounding_artifacts",
    "anchor_target_intents", "compiled_anchor_targets",
    "anchor_target_edit_trials", "anchor_target_preview_artifacts", "probe_target_intents", "compiled_probe_targets",
    "probe_target_edit_trials", "probe_target_preview_artifacts", "probe_target_validations", "probe_target_media_validations",
    "probe_target_applications", "source_semantic_material_audits", "source_semantic_material_artifact_registrations",
    "diagnostic_region_ledger", "anchor_probe_separation_results", "diagnostic_probe_attempts", "probe_target_reflections",
    "episode_concerns", "region_settlements", "model_reflections", "terminal_attempts",
    "paired_comparison_policies", "pair_direction_locks", "probe_measurements", "paired_comparison_records",
    "geometry_bounds_measurements", "videos", "video_paths", "triptych_videos", "triptych_video_paths",
    "triptych_sequences",
})
_FRESH_UNPLANNED_READONLY_MEASUREMENT_KEYS = frozenset({
    "geometry_context_measurements", "part_grounding_context_measurements",
})
_FRESH_UNPLANNED_READONLY_EVENT_KEYS = frozenset({"runtime_owned_events"})
_FRESH_UNPLANNED_EMPTY_MAP_KEYS = frozenset({
    "active_setup_trial_by_anchor", "final_setup_trial_by_anchor", "active_anchor_target_intent_by_anchor",
    "active_compiled_anchor_target_by_anchor", "region_episode_ids", "region_probe_intent_ids",
    "region_current_probe_compile_ids",
})
_FRESH_UNPLANNED_NONE_KEYS = frozenset({
    "active_episode_index", "active_probe_target_intent_id", "active_compiled_probe_target_id",
    "active_source_semantic_material_audit_id", "source_semantic_material_artifact_registration",
    "active_source_semantic_material_artifact_registration_id", "active_region_id", "coverage_summary", "terminal",
    "artifact_paths", "active_diagnostic_session_plan", "session_plan_version",
    "route_adjudication", "active_agent_invocation_id", "status",
})
_FRESH_UNPLANNED_READONLY_INPUT_KEYS = frozenset({"part_grounding_context"})
_FRESH_UNPLANNED_CONFIG_KEYS = frozenset({
    "enabled", "max_runs", "max_episodes", "max_actions_per_episode", "episode_timeout_s", "live_host", "live_port",
    "live_ready_timeout_s", "live_heartbeat_ms", "live_client_lease_timeout_ms", "probe_max_vertices",
    "probe_max_distance_m", "probe_max_speed_m_s", "probe_max_duration_steps", "genesis_root", "genesis_env_path",
    "genesis_live_command",
})
_FRESH_UNPLANNED_ALLOWED_KEYS = _FRESH_UNPLANNED_EMPTY_LIST_KEYS | _FRESH_UNPLANNED_EMPTY_MAP_KEYS | _FRESH_UNPLANNED_NONE_KEYS | _FRESH_UNPLANNED_CONFIG_KEYS | _FRESH_UNPLANNED_READONLY_INPUT_KEYS | _FRESH_UNPLANNED_READONLY_MEASUREMENT_KEYS | _FRESH_UNPLANNED_READONLY_EVENT_KEYS

_DIAGNOSTIC_ATTEMPT_ARCHIVE_SCHEMA_VERSION = "hag4r-diagnostic-attempt-archive-v1"
_DIAGNOSTIC_ATTEMPT_ARCHIVE_KEYS = frozenset(
    {
        "schema_version",
        "archive_id",
        "archived_at",
        "recovery_cause",
        "recovery_event_id",
        "source_attachment_id",
        "source_agent_invocation_id",
        "source_session_handles",
        "source_attempt_id",
        "execution_snapshot",
        "removed_route_projection",
    }
)
_DIAGNOSTIC_ATTEMPT_ARCHIVED_LIST_KEYS = _FRESH_UNPLANNED_EMPTY_LIST_KEYS | frozenset({"runtime_owned_events"})
_DIAGNOSTIC_ATTEMPT_ARCHIVED_MAP_KEYS = _FRESH_UNPLANNED_EMPTY_MAP_KEYS
_DIAGNOSTIC_ATTEMPT_ARCHIVED_NONE_KEYS = _FRESH_UNPLANNED_NONE_KEYS
_DIAGNOSTIC_ATTEMPT_PRESERVED_KEYS = (
    _FRESH_UNPLANNED_CONFIG_KEYS
    | _FRESH_UNPLANNED_READONLY_INPUT_KEYS
    | _FRESH_UNPLANNED_READONLY_MEASUREMENT_KEYS
)
_DIAGNOSTIC_ATTEMPT_METADATA_KEYS = frozenset(
    {"diagnostic_attempt_archives", "active_diagnostic_attempt_id"}
)
_DIAGNOSTIC_ATTEMPT_CLASSIFIED_KEYS = (
    _DIAGNOSTIC_ATTEMPT_ARCHIVED_LIST_KEYS
    | _DIAGNOSTIC_ATTEMPT_ARCHIVED_MAP_KEYS
    | _DIAGNOSTIC_ATTEMPT_ARCHIVED_NONE_KEYS
    | _DIAGNOSTIC_ATTEMPT_PRESERVED_KEYS
    | _DIAGNOSTIC_ATTEMPT_METADATA_KEYS
)
_DIAGNOSTIC_ROUTE_PROJECTION_KEYS = frozenset(
    {
        "diagnostic_recommendation",
        "diagnostic_route",
        "orchestrator_diagnostic_decision",
        "sim_diagnostic_cues",
        "diagnostic_repair_brief",
        "force_rerun_stage_skill_paths",
        "first_rerouted_stage_skill_path",
    }
)


def _validate_diagnostic_attempt_archives(value: Any, active_attempt_id: Any) -> bool:
    if not isinstance(value, list):
        return False
    # First-attempt state has neither archive nor namespace.  A nonempty
    # archive is inseparable from a successor namespace, otherwise an old
    # attempt could silently reuse first-attempt artifact paths.
    if not value:
        return active_attempt_id is None
    if (
        not isinstance(active_attempt_id, str)
        or not re.fullmatch(r"diagnostic_attempt_\d{4}", active_attempt_id)
        or active_attempt_id == "diagnostic_attempt_0001"
    ):
        return False
    seen: set[tuple[str, str]] = set()
    for index, archive in enumerate(value, start=1):
        if not isinstance(archive, Mapping) or set(archive) != _DIAGNOSTIC_ATTEMPT_ARCHIVE_KEYS:
            return False
        if archive.get("schema_version") != _DIAGNOSTIC_ATTEMPT_ARCHIVE_SCHEMA_VERSION:
            return False
        if archive.get("archive_id") != f"diagnostic_attempt_archive_{index:04d}":
            return False
        if not all(isinstance(archive.get(key), str) and archive[key] for key in (
            "archived_at", "recovery_cause", "recovery_event_id", "source_attachment_id", "source_agent_invocation_id",
            "source_attempt_id",
        )):
            return False
        if not isinstance(archive.get("source_session_handles"), list) or not all(
            isinstance(handle, str) and handle for handle in archive["source_session_handles"]
        ):
            return False
        identity = (archive["source_attachment_id"], archive["recovery_event_id"])
        if identity in seen or not isinstance(archive.get("execution_snapshot"), Mapping):
            return False
        if not isinstance(archive.get("removed_route_projection"), Mapping):
            return False
        seen.add(identity)
    return True


def _next_diagnostic_attempt_id(archives: list[dict[str, Any]]) -> str:
    highest = 1
    for archive in archives:
        source_attempt_id = archive.get("source_attempt_id")
        if isinstance(source_attempt_id, str):
            match = re.fullmatch(r"diagnostic_attempt_(\d{4})", source_attempt_id)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"diagnostic_attempt_{highest + 1:04d}"


def archive_and_reset_diagnostic_execution_attempt(
    state: dict[str, Any],
    *,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a terminal transport-loss attempt and prepare a successor.

    The MCP runtime invokes this while it owns the run lock, then persists the
    whole successor transaction.  It deliberately does not touch ownership
    ledgers or live handles: those must already be terminalized by the server.
    """
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("state.diagnostics must be an object")
    required_recovery = {
        "cause", "event_id", "source_attachment_id", "source_agent_invocation_id",
    }
    if set(recovery) != required_recovery or not all(
        isinstance(recovery[key], str) and recovery[key]
        for key in required_recovery
    ):
        raise ValueError("diagnostic attempt recovery metadata is malformed")
    unknown = set(diagnostics) - _DIAGNOSTIC_ATTEMPT_CLASSIFIED_KEYS
    if unknown:
        raise ValueError(
            "diagnostic attempt reset encountered unclassified active state: "
            + ", ".join(sorted(unknown))
        )
    archives = diagnostics.get("diagnostic_attempt_archives", [])
    active_attempt_id = diagnostics.get("active_diagnostic_attempt_id")
    if not _validate_diagnostic_attempt_archives(archives, active_attempt_id):
        raise ValueError("diagnostic attempt archive metadata is malformed")
    archive_key = (recovery["source_attachment_id"], recovery["event_id"])
    for archive in archives:
        if (archive["source_attachment_id"], archive["recovery_event_id"]) == archive_key:
            if is_fresh_unplanned_diagnostics_state(state):
                return dict(archive)
            raise ValueError(
                "diagnostic attempt archive already exists but active successor state is not fresh"
            )
    source_attempt_id = active_attempt_id or "diagnostic_attempt_0001"
    execution_snapshot = {
        key: copy.deepcopy(diagnostics.get(key))
        for key in (
            _DIAGNOSTIC_ATTEMPT_ARCHIVED_LIST_KEYS
            | _DIAGNOSTIC_ATTEMPT_ARCHIVED_MAP_KEYS
            | _DIAGNOSTIC_ATTEMPT_ARCHIVED_NONE_KEYS
        )
        if key in diagnostics
    }
    route_state = state.get("route_state")
    if route_state is not None and not isinstance(route_state, dict):
        raise ValueError("state.route_state must be an object when present")
    removed_route_projection = {
        "route_state": {
            key: copy.deepcopy(route_state[key])
            for key in _DIAGNOSTIC_ROUTE_PROJECTION_KEYS
            if isinstance(route_state, dict) and key in route_state
        },
        "root": {
            key: copy.deepcopy(state[key])
            for key in ("sim_diagnostic_cues", "diagnostic_repair_brief")
            if key in state
        },
    }
    archive = {
        "schema_version": _DIAGNOSTIC_ATTEMPT_ARCHIVE_SCHEMA_VERSION,
        "archive_id": f"diagnostic_attempt_archive_{len(archives) + 1:04d}",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "recovery_cause": recovery["cause"],
        "recovery_event_id": recovery["event_id"],
        "source_attachment_id": recovery["source_attachment_id"],
        "source_agent_invocation_id": recovery["source_agent_invocation_id"],
        "source_session_handles": [
            str(session["live_session_handle"])
            for session in state.get("diagnostic_runtime_sessions", [])
            if isinstance(session, Mapping)
            and session.get("attachment_id") == recovery["source_attachment_id"]
            and isinstance(session.get("live_session_handle"), str)
            and session["live_session_handle"]
        ],
        "source_attempt_id": source_attempt_id,
        "execution_snapshot": execution_snapshot,
        "removed_route_projection": removed_route_projection,
    }
    archives.append(archive)
    for key in _DIAGNOSTIC_ATTEMPT_ARCHIVED_LIST_KEYS:
        diagnostics[key] = []
    for key in _DIAGNOSTIC_ATTEMPT_ARCHIVED_MAP_KEYS:
        diagnostics[key] = {}
    for key in _DIAGNOSTIC_ATTEMPT_ARCHIVED_NONE_KEYS:
        diagnostics[key] = None
    diagnostics["diagnostic_attempt_archives"] = archives
    diagnostics["active_diagnostic_attempt_id"] = _next_diagnostic_attempt_id(archives)
    if isinstance(route_state, dict):
        for key in _DIAGNOSTIC_ROUTE_PROJECTION_KEYS:
            route_state.pop(key, None)
    state.pop("sim_diagnostic_cues", None)
    state.pop("diagnostic_repair_brief", None)
    return dict(archive)


def _fresh_preplan_readonly_measurements_are_valid(
    diagnostics: Mapping[str, Any],
) -> bool:
    geometry_measurements = diagnostics.get("geometry_context_measurements", [])
    if (
        not isinstance(geometry_measurements, list)
        or len(geometry_measurements) > 1
        or any(
            not isinstance(measurement, Mapping)
            or set(measurement) != {"measurement_index", "geometry_context"}
            or isinstance(measurement["measurement_index"], bool)
            or not isinstance(measurement["measurement_index"], int)
            or measurement["measurement_index"] != index
            or not isinstance(measurement["geometry_context"], Mapping)
            for index, measurement in enumerate(geometry_measurements)
        )
    ):
        return False
    grounding_measurements = diagnostics.get("part_grounding_context_measurements", [])
    if (
        not isinstance(grounding_measurements, list)
        or len(grounding_measurements) > 1
        or any(
            not isinstance(measurement, Mapping)
            or set(measurement) != {"measurement_index", "source", "part_grounding_context"}
            or isinstance(measurement["measurement_index"], bool)
            or not isinstance(measurement["measurement_index"], int)
            or measurement["measurement_index"] != index
            or measurement["source"] != "compute_part_grounding_context"
            or not isinstance(measurement["part_grounding_context"], Mapping)
            for index, measurement in enumerate(grounding_measurements)
        )
    ):
        return False
    readonly_events = diagnostics.get("runtime_owned_events", [])
    expected_events: dict[str, tuple[str, str, dict[str, Any]]] = {}
    if geometry_measurements:
        expected_events["diagnostic_geometry_context_computed"] = (
            "computed minimal diagnostic geometry context",
            "runtime_event",
            _runtime_event_detail_preview(
                {"geometry_context": dict(geometry_measurements[0])}
            ),
        )
    if grounding_measurements:
        expected_events["diagnostic_part_grounding_context_computed"] = (
            "computed runtime-owned diagnostic part grounding context",
            "runtime_event",
            _runtime_event_detail_preview(
                {
                    "part_grounding_context": _part_grounding_measurement_preview(
                        dict(grounding_measurements[0])
                    )
                }
            ),
        )
    recovered_attempt = diagnostics.get("active_diagnostic_attempt_id") is not None
    if recovered_attempt and readonly_events == []:
        # A successor preserves the immutable source measurements but archives
        # the prior attempt's runtime-event evidence.  Its own event stream
        # starts empty until it performs fresh work.
        return True
    if not isinstance(readonly_events, list) or len(readonly_events) != len(expected_events):
        return False
    seen_events: set[str] = set()
    for index, event in enumerate(readonly_events):
        if (
            not isinstance(event, Mapping)
            or set(event) != {"event_id", "event", "note", "detail", "provenance"}
            or isinstance(event["event_id"], bool)
            or not isinstance(event["event_id"], int)
            or event["event_id"] != index + 1
            or event["event"] not in expected_events
            or event["event"] in seen_events
        ):
            return False
        note, provenance, detail = expected_events[event["event"]]
        if event["note"] != note or event["provenance"] != provenance or event["detail"] != detail:
            return False
        seen_events.add(event["event"])
    return seen_events == set(expected_events)


def is_fresh_unplanned_diagnostics_state(state: Mapping[str, Any]) -> bool:
    """Only attachment and deterministic pre-plan read-only state may precede v2."""
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    sessions = state.get("diagnostic_runtime_sessions", [])
    archives = diagnostics.get("diagnostic_attempt_archives", [])
    active_attempt_id = diagnostics.get("active_diagnostic_attempt_id")
    if not _validate_diagnostic_attempt_archives(archives, active_attempt_id):
        return False
    if not isinstance(sessions, list):
        return False
    if not archives and sessions:
        return False
    if any(
        not isinstance(session, Mapping)
        or session.get("lifecycle_state") not in {"closed", "closed_failed"}
        for session in sessions
    ):
        return False
    # This is intentionally a literal allowlist: an unfamiliar persisted field
    # is evidence-bearing until a v2 migration explicitly classifies it.
    terminal = diagnostics.get("terminal")
    retry_terminal = (
        isinstance(terminal, Mapping)
        and terminal.get("business_outcome") == {"status": "not_authored", "recommendation": None}
        and isinstance(terminal.get("operational_outcome"), Mapping)
        and terminal["operational_outcome"].get("status") in {
            "lease_expired_retryable",
            "cleanup_retryable",
        }
        and terminal["operational_outcome"].get("retryable") is True
        and terminal.get("validated") is False
    )
    if any(key not in _FRESH_UNPLANNED_ALLOWED_KEYS | _DIAGNOSTIC_ATTEMPT_METADATA_KEYS for key in diagnostics):
        return False
    for key in _FRESH_UNPLANNED_EMPTY_LIST_KEYS:
        if key in diagnostics and diagnostics[key] != []:
            return False
    for key in _FRESH_UNPLANNED_EMPTY_MAP_KEYS:
        if key in diagnostics and diagnostics[key] != {}:
            return False
    for key in _FRESH_UNPLANNED_NONE_KEYS:
        if key == "terminal" and retry_terminal:
            continue
        if key in diagnostics and diagnostics[key] is not None:
            return False
    if not _fresh_preplan_readonly_measurements_are_valid(diagnostics):
        return False
    if "part_grounding_context" in diagnostics and diagnostics["part_grounding_context"] is not None and not isinstance(diagnostics["part_grounding_context"], Mapping):
        return False
    # Config is read-only at this boundary; require the ordinary config shapes,
    # rather than accepting a marker object or evidence payload under a config key.
    if diagnostics.get("enabled") is not True:
        return False
    for key in _FRESH_UNPLANNED_CONFIG_KEYS - {"enabled", "genesis_env_path"}:
        if key in diagnostics and isinstance(diagnostics[key], (dict, list, tuple, set)):
            return False
    if "genesis_env_path" in diagnostics and diagnostics["genesis_env_path"] is not None and not isinstance(diagnostics["genesis_env_path"], str):
        return False
    return True


def require_v2_session_state(state: Mapping[str, Any], *, allow_fresh_unplanned: bool = False) -> Mapping[str, Any] | None:
    diagnostics = state.get("diagnostics")
    plan = diagnostics.get("active_diagnostic_session_plan") if isinstance(diagnostics, Mapping) else None
    if isinstance(plan, Mapping):
        try:
            normalized = validate_diagnostic_session_plan(plan)
        except GenesisVlmSchemaError as exc:
            raise GenesisVlmSchemaError(f"diagnostics active v2 plan is malformed or unsupported: {exc}") from exc
        plans = diagnostics.get("diagnostic_session_plans") if isinstance(diagnostics, Mapping) else None
        if (
            normalized != plan
            or diagnostics.get("session_plan_version") != VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION
            or not isinstance(plans, list)
            or len(plans) != 1
            or plans[0] != plan
        ):
            raise GenesisVlmSchemaError(
                "diagnostics v2 plan lineage is unsupported, corrupt, missing, duplicated, "
                "mismatched, or tampered"
            )
        return plan
    if allow_fresh_unplanned and is_fresh_unplanned_diagnostics_state(state):
        return None
    found = plan.get("schema_version") if isinstance(plan, Mapping) else diagnostics.get("session_plan_version") if isinstance(diagnostics, Mapping) else None
    raise GenesisVlmSchemaError(
        f"diagnostics requires {VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION}; found {found or 'missing'} in a non-fresh state"
    )
def _diagnostic_workspace_episodes_root(state: dict[str, Any]) -> Path:
    root = Path(str(state["paths"]["diagnostic_workspace_dir"])).expanduser().resolve()
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not _validate_diagnostic_attempt_archives(
        diagnostics.get("diagnostic_attempt_archives", []),
        diagnostics.get("active_diagnostic_attempt_id"),
    ):
        raise ValueError("diagnostic attempt metadata is malformed")
    attempt_id = diagnostics.get("active_diagnostic_attempt_id")
    if attempt_id is None:
        return root / "episodes"
    if not isinstance(attempt_id, str) or not re.fullmatch(r"diagnostic_attempt_\d{4}", attempt_id):
        raise ValueError("active diagnostic attempt namespace is malformed")
    return root / "attempts" / attempt_id / "episodes"
def _diagnostic_generated_episodes_root(state: dict[str, Any]) -> Path:
    root = Path(str(state["paths"]["diagnostic_generated_episodes_dir"])).expanduser().resolve()
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not _validate_diagnostic_attempt_archives(
        diagnostics.get("diagnostic_attempt_archives", []),
        diagnostics.get("active_diagnostic_attempt_id"),
    ):
        raise ValueError("diagnostic attempt metadata is malformed")
    attempt_id = diagnostics.get("active_diagnostic_attempt_id")
    if attempt_id is None:
        return root
    if not isinstance(attempt_id, str) or not re.fullmatch(r"diagnostic_attempt_\d{4}", attempt_id):
        raise ValueError("active diagnostic attempt namespace is malformed")
    return root / attempt_id
def _active_diagnostic_mesh_path(state: dict[str, Any]) -> Path:
    return Path(str(state["paths"]["monolithic_mesh_path"])).expanduser().resolve()
def _optional_path(paths: dict[str, Any], key: str) -> Path | None:
    value = paths.get(key)
    return Path(str(value)) if value else None
def _assert_workspace_episode_write_path(path: Path, workspace_episodes_root: Path) -> None:
    resolved = path.expanduser().resolve()
    root = workspace_episodes_root.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing diagnostic workspace episode write outside episodes root: {resolved}") from exc
def _assert_generated_episode_write_path(path: Path, generated_episodes_root: Path) -> None:
    resolved = path.expanduser().resolve()
    root = generated_episodes_root.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing diagnostic generated episode write outside episodes root: {resolved}") from exc
def _episode_record_by_index(diagnostics: dict[str, Any], episode_index: int) -> dict[str, Any]:
    for episode in diagnostics.get("episodes", []):
        if isinstance(episode, dict) and int(episode.get("episode_index", 0)) == int(episode_index):
            return episode
    raise ValueError(f"diagnostic episode is not recorded: {episode_index}")
def _active_episode_record(state: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    active_episode_index = diagnostics.get("active_episode_index")
    if active_episode_index is None:
        raise ValueError("A diagnostic episode suite must be created before live diagnostic tools are used.")
    return _episode_record_by_index(diagnostics, int(active_episode_index))
def _episode_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "episode_index": int(episode["episode_index"]),
        "episode_id": str(episode["episode_id"]),
        "episode_dir": str(episode["episode_dir"]),
        "scene_config_path": str(episode["scene_config_path"]),
        "observations_path": str(episode["observations_path"]),
    }
    anchor = _anchor_region_preview(episode.get("anchor_region"))
    if anchor:
        metadata["anchor_region"] = anchor
    if episode.get("setup_trial_id"):
        metadata["setup_trial_id"] = str(episode["setup_trial_id"])
    if episode.get("anchor_compile_id"):
        metadata["anchor_compile_id"] = str(episode["anchor_compile_id"])
    if episode.get("anchor_preview_evidence_id"):
        metadata["anchor_preview_evidence_id"] = str(episode["anchor_preview_evidence_id"])
    setup_validation = episode.get("anchor_setup_validation")
    if isinstance(setup_validation, dict):
        metadata["anchor_setup_validation"] = {
            key: setup_validation[key]
            for key in (
                "schema_version",
                "status",
                "anchor_id",
                "compile_id",
                "selected_part_id",
                "region_hint",
                "local_anchor_box",
                "executable_pin_box",
                "executable_frame",
                "warning_codes",
                "pinned_vertex_distribution",
            )
            if key in setup_validation
        }
    setup_quality = _setup_quality_preview(episode.get("setup_quality"))
    if setup_quality:
        metadata["setup_quality"] = setup_quality
    return metadata
def _live_tool_succeeded(result: Mapping[str, Any]) -> bool:
    return str(result.get("status", "")) in LIVE_TOOL_SUCCESS_STATUSES
def _successful_compiled_probe_count(diagnostics: Mapping[str, Any], episode_id: str) -> int:
    tool_results = diagnostics.get("tool_results", [])
    count = 0
    for application in diagnostics.get("probe_target_applications", []):
        if not isinstance(application, Mapping):
            continue
        if str(application.get("episode_id", "")) != episode_id:
            continue
        if str(application.get("internal_action", "")) != "box_ee_grasp_and_move":
            continue
        result_status = str(application.get("result_status", ""))
        if result_status in LIVE_TOOL_SUCCESS_STATUSES:
            count += 1
            continue
        tool_result_index = application.get("tool_result_index")
        if (
            isinstance(tool_result_index, bool)
            or not isinstance(tool_result_index, int)
            or not isinstance(tool_results, list)
        ):
            continue
        if tool_result_index < 0 or tool_result_index >= len(tool_results):
            continue
        record = tool_results[tool_result_index]
        result = record.get("result") if isinstance(record, Mapping) else None
        if isinstance(result, Mapping) and _live_tool_succeeded(result):
            count += 1
    return count
def _completed_simulated_step_counts_for_episode(diagnostics: Mapping[str, Any], episode: Mapping[str, Any]) -> list[int]:
    tool_results = diagnostics.get("tool_results", [])
    if not isinstance(tool_results, list):
        return []
    counts: list[int] = []
    for raw_index in episode.get("tool_result_indices", []):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index >= len(tool_results)
        ):
            continue
        record = tool_results[raw_index]
        if not isinstance(record, Mapping) or str(record.get("tool", "")) != "simulate":
            continue
        result = record.get("result")
        if not isinstance(result, Mapping) or not _live_tool_succeeded(result):
            continue
        payload = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        steps_completed = payload.get("steps_completed") if isinstance(payload, Mapping) else None
        if isinstance(steps_completed, bool) or not isinstance(steps_completed, int):
            decision = record.get("decision")
            arguments = decision.get("arguments") if isinstance(decision, Mapping) else {}
            steps_completed = arguments.get("steps") if isinstance(arguments, Mapping) else 0
        if isinstance(steps_completed, bool) or not isinstance(steps_completed, int):
            continue
        counts.append(max(0, steps_completed))
    return counts
def _completed_simulated_steps_for_episode(diagnostics: Mapping[str, Any], episode: Mapping[str, Any]) -> int:
    return sum(_completed_simulated_step_counts_for_episode(diagnostics, episode))
def _max_completed_simulate_steps_for_episode(diagnostics: Mapping[str, Any], episode: Mapping[str, Any]) -> int:
    return max(_completed_simulated_step_counts_for_episode(diagnostics, episode), default=0)
_POST_RUNTIME_FAILURE_DENIED_TOOLS = frozenset(
    {
        "inspect_genesis_runtime_logs",
        "simulation_reset",
        "simulate",
        "query_live_geometry_context",
        "submit_diagnostic_probe_target_intent",
        "compile_diagnostic_probe_target",
        "preview_diagnostic_probe_target",
        "revise_diagnostic_probe_target",
    }
)


def _runtime_log_record_has_lines(record: Mapping[str, Any]) -> bool:
    wrapper = record.get("result")
    if not isinstance(wrapper, Mapping) or wrapper.get("status") != "ok":
        return False
    envelope = wrapper.get("result") if isinstance(wrapper, Mapping) else None
    streams = envelope.get("streams") if isinstance(envelope, Mapping) else None
    return isinstance(streams, Mapping) and any(
        isinstance(stream, Mapping) and bool(stream.get("lines"))
        for stream in streams.values()
    )


def _attachment_owned_events(state: Mapping[str, Any], attachment_id: str) -> list[dict[str, Any]]:
    attachments = state.get("diagnostic_runtime_attachments", [])
    attachment = next(
        (
            item
            for item in attachments
            if isinstance(item, dict) and item.get("attachment_id") == attachment_id
        ),
        None,
    )
    if not isinstance(attachment, dict):
        raise GenesisVlmSchemaError("runtime-log evidence attachment ledger is missing")
    events = [
        event
        for key in ("pre_session_tool_events", "terminal_tool_events")
        for event in attachment.get(key, [])
        if isinstance(event, dict)
    ]
    events.extend(
        {
            **event,
            "attachment_id": event.get("attachment_id", session.get("attachment_id")),
            "episode_id": event.get("episode_id", session.get("episode_id")),
            "live_session_handle": event.get(
                "live_session_handle", session.get("live_session_handle")
            ),
        }
        for session in state.get("diagnostic_runtime_sessions", [])
        if isinstance(session, dict) and session.get("attachment_id") == attachment_id
        for event in session.get("tool_events", [])
        if isinstance(event, dict)
    )
    return sorted(events, key=lambda event: int(event.get("sequence_index", -1)))


def _current_runtime_attachment(
    state: Mapping[str, Any], diagnostics: Mapping[str, Any], *, allow_completed: bool = False
) -> dict[str, Any]:
    attachments = [
        item
        for item in state.get("diagnostic_runtime_attachments", [])
        if isinstance(item, dict) and item.get("status") in ({"attached", "completed"} if allow_completed else {"attached"})
    ]
    invocation_id = diagnostics.get("active_agent_invocation_id")
    if invocation_id:
        attachments = [
            item for item in attachments if item.get("agent_invocation_id") == invocation_id
        ]
    if len(attachments) != 1:
        raise GenesisVlmSchemaError(
            "runtime-log evidence requires exactly one current diagnostic attachment"
        )
    return attachments[0]


def _current_attachment_has_runtime_failure(
    state: Mapping[str, Any], diagnostics: Mapping[str, Any], *, allow_completed: bool = False
) -> bool:
    current_reflections = [
        reflection
        for reflection in diagnostics.get("model_reflections", [])
        if isinstance(reflection, Mapping) and reflection.get("phase") == "runtime_failure"
    ]
    try:
        attachment = _current_runtime_attachment(state, diagnostics, allow_completed=allow_completed)
    except GenesisVlmSchemaError:
        return bool(current_reflections)
    attachment_id = attachment.get("attachment_id")
    invocation_id = attachment.get("agent_invocation_id")
    if isinstance(attachment.get("runtime_failure"), Mapping):
        return True
    if any(
        isinstance(session, Mapping)
        and session.get("attachment_id") == attachment_id
        and isinstance(session.get("runtime_failure"), Mapping)
        for session in state.get("diagnostic_runtime_sessions", [])
    ):
        return True
    return any(
        isinstance(reflection, Mapping)
        and reflection.get("phase") == "runtime_failure"
        and (
            reflection.get("attachment_id") == attachment_id
            or reflection.get("agent_invocation_id") == invocation_id
        )
        for reflection in current_reflections
    )


def _require_correlated_runtime_log_failure_evidence(
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    require_close_event: bool,
    allow_completed_attachment: bool = False,
) -> dict[str, Any]:
    attachment = _current_runtime_attachment(state, diagnostics, allow_completed=allow_completed_attachment)
    invocation_id = str(attachment.get("agent_invocation_id", ""))
    attachment_id = str(attachment.get("attachment_id", ""))
    sessions = [
        item
        for item in state.get("diagnostic_runtime_sessions", [])
        if isinstance(item, dict)
        and item.get("attachment_id") == attachment_id
        and isinstance(item.get("runtime_failure"), dict)
    ]
    if len(sessions) != 1:
        raise GenesisVlmSchemaError("runtime-log evidence requires exactly one correlated failed session")
    session = sessions[0]
    failure = dict(session["runtime_failure"])
    if attachment.get("runtime_failure") != failure:
        raise GenesisVlmSchemaError(
            "attachment and episode runtime-log failure markers disagree"
        )
    episode_id = str(failure.get("episode_id", ""))
    handle = str(failure.get("live_session_handle", ""))
    if (
        failure.get("attachment_id") != attachment_id
        or session.get("agent_invocation_id") != invocation_id
        or session.get("episode_id") != episode_id
        or session.get("live_session_handle") != handle
    ):
        raise GenesisVlmSchemaError("runtime-log failure marker crosses ownership identity")
    cited_index = failure.get("cited_tool_result_index")
    if isinstance(cited_index, bool) or not isinstance(cited_index, int):
        raise GenesisVlmSchemaError("runtime-log evidence stores an invalid cited result index")
    tool_results = diagnostics.get("tool_results", [])
    if cited_index < 0 or cited_index >= len(tool_results):
        raise GenesisVlmSchemaError("runtime-log evidence cites a missing result")
    if failure.get("cited_tool_result_ref") != f"tool_result:{cited_index + 1}":
        raise GenesisVlmSchemaError("runtime-log evidence has a 1-based/0-based result-index mismatch")
    record = tool_results[cited_index]
    if (
        not isinstance(record, Mapping)
        or record.get("tool") != "inspect_genesis_runtime_logs"
        or record.get("episode_id") != episode_id
        or not _runtime_log_record_has_lines(record)
    ):
        raise GenesisVlmSchemaError("runtime-log evidence must cite a nonempty successful owned inspection")
    events = _attachment_owned_events(state, attachment_id)
    inspection_sequence = failure.get("inspection_event_sequence_index")
    fault_sequence = failure.get("event_sequence_index")
    inspection_event = next(
        (
            event
            for event in events
            if event.get("sequence_index") == inspection_sequence
            and event.get("tool") == "inspect_genesis_runtime_logs"
            and event.get("status") == "ok"
            and event.get("tool_result_index") == cited_index
        ),
        None,
    )
    fault_event = next(
        (
            event
            for event in events
            if event.get("sequence_index") == fault_sequence
            and event.get("tool") == "record_diagnostic_evidence"
            and event.get("status") == "ok"
            and event.get("reflection_index") == failure.get("reflection_index")
        ),
        None,
    )
    if not inspection_event or not fault_event or not int(inspection_sequence) < int(fault_sequence):
        raise GenesisVlmSchemaError("runtime-log inspection must precede its runtime_failure reflection")
    for event in (inspection_event, fault_event):
        if (
            event.get("attachment_id") != attachment_id
            or event.get("episode_id") != episode_id
            or event.get("live_session_handle") != handle
        ):
            raise GenesisVlmSchemaError("runtime-log evidence crosses attachment, episode, or handle ownership")
    reflection_index = failure.get("reflection_index")
    reflections = diagnostics.get("model_reflections", [])
    if (
        isinstance(reflection_index, bool)
        or not isinstance(reflection_index, int)
        or reflection_index < 0
        or reflection_index >= len(reflections)
    ):
        raise GenesisVlmSchemaError("runtime-log failure marker cites an invalid reflection")
    fault_reflection = reflections[reflection_index]
    fault_refs = (
        fault_reflection.get("artifact_refs")
        if isinstance(fault_reflection, Mapping)
        else None
    )
    cited_ref_matches = (
        isinstance(fault_refs, list)
        and len(fault_refs) == 1
        and isinstance(fault_refs[0], Mapping)
        and fault_refs[0].get("kind") == "tool_result"
        and fault_refs[0].get("ref") == f"tool_result:{cited_index + 1}"
    )
    if (
        not isinstance(fault_reflection, Mapping)
        or fault_reflection.get("phase") != "runtime_failure"
        or fault_reflection.get("reflection_index") != reflection_index
        or fault_reflection.get("attachment_id") != attachment_id
        or fault_reflection.get("agent_invocation_id") != invocation_id
        or fault_reflection.get("episode_id") != episode_id
        or fault_reflection.get("live_session_handle") != handle
        or fault_reflection.get("ownership_event_sequence_index") != fault_sequence
        or not cited_ref_matches
    ):
        raise GenesisVlmSchemaError("runtime-log failure reflection correlation is malformed")
    close_event = next(
        (
            event
            for event in events
            if int(event.get("sequence_index", -1)) > int(fault_sequence)
            and event.get("tool") == "close_genesis_live_session"
            and event.get("status") == "ok"
            and event.get("attachment_id") == attachment_id
            and event.get("episode_id") == episode_id
            and event.get("live_session_handle") == handle
        ),
        None,
    )
    if require_close_event and (not close_event or session.get("lifecycle_state") != "closed"):
        raise GenesisVlmSchemaError("runtime_failure must be followed by a clean close of the same session")
    close_sequence = int(close_event["sequence_index"]) if close_event else None
    denied = [
        event.get("tool")
        for event in events
        if int(event.get("sequence_index", -1)) > int(fault_sequence)
        and event.get("tool") in _POST_RUNTIME_FAILURE_DENIED_TOOLS
    ]
    if denied:
        raise GenesisVlmSchemaError(f"runtime_failure was followed by forbidden live work: {denied}")
    return {
        "attachment_id": attachment_id,
        "agent_invocation_id": invocation_id,
        "episode_id": episode_id,
        "live_session_handle": handle,
        "tool_result_index": cited_index,
        "inspection_event_sequence_index": int(inspection_sequence),
        "runtime_failure_event_sequence_index": int(fault_sequence),
        "close_event_sequence_index": close_sequence,
    }


def _anchor_region_preview(anchor: Any) -> dict[str, Any]:
    if not isinstance(anchor, dict):
        return {}
    preview = {
        key: anchor[key]
        for key in (
            "anchor_id",
            "name",
            "anchor_type",
            "semantic_region",
            "uncertainty",
            "concerns",
        )
        if key in anchor
    }
    if "concerns" in preview and isinstance(preview["concerns"], list):
        preview["concerns"] = [str(item) for item in preview["concerns"][:8]]
    return preview
def _setup_trial_preview(trial: Any) -> dict[str, Any]:
    if not isinstance(trial, dict):
        return {}
    reflection = trial.get("reflection") if isinstance(trial.get("reflection"), dict) else {}
    artifacts = trial.get("preview_artifacts") if isinstance(trial.get("preview_artifacts"), dict) else {}
    preview = {
        key: trial[key]
        for key in (
            "schema_version",
            "trial_id",
            "trial_index",
            "anchor_id",
            "anchor_compile_id",
            "anchor_preview_evidence_id",
            "revision_request",
            "final_status",
            "created_at_event_id",
        )
        if key in trial
    }
    anchor = _anchor_region_preview(trial.get("anchor_region"))
    if anchor:
        preview["anchor_region"] = anchor
    if artifacts:
        preview["preview_artifacts"] = {
            key: artifacts[key]
            for key in ("png_path", "renderer", "overlay_frame")
            if key in artifacts
        }
    validation = trial.get("anchor_setup_validation") if isinstance(trial.get("anchor_setup_validation"), dict) else {}
    if validation:
        preview["anchor_setup_validation"] = {
            key: validation[key]
            for key in (
                "schema_version",
                "status",
                "anchor_id",
                "compile_id",
                "selected_part_id",
                "region_hint",
                "local_anchor_box",
                "executable_pin_box",
                "executable_frame",
                "warning_codes",
                "pinned_vertex_distribution",
            )
            if key in validation
        }
    if reflection:
        preview["reflection"] = {
            "verdict": reflection.get("verdict"),
            "concerns": _string_list(reflection.get("concerns"))[:8],
            "uncertainty": reflection.get("uncertainty"),
            "revision_request": reflection.get("revision_request"),
        }
    return preview
def _setup_quality_preview(setup_quality: Any) -> dict[str, Any]:
    if not isinstance(setup_quality, dict):
        return {}
    preview = {
        key: setup_quality[key]
        for key in (
            "final_status",
            "verdict",
            "uncertainty",
            "revision_request",
            "preview_png_path",
        )
        if key in setup_quality
    }
    if "concerns" in setup_quality:
        preview["concerns"] = _string_list(setup_quality.get("concerns"))[:8]
    return preview
def _session_plan_preview(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    return {
        "schema_version": str(plan.get("schema_version", "")),
        "session_intent": str(plan.get("session_intent", "")),
        "candidate_region_count": plan.get("candidate_region_count"),
        "regions": [{key: region.get(key) for key in ("region_id", "name", "semantic_region", "risk_rank", "setup_anchor")}
                    for region in plan.get("regions", []) if isinstance(region, Mapping)],
        "omitted_region_summary": list(plan.get("omitted_region_summary", [])),
    }
def _geometry_context_preview(measurement: Any) -> dict[str, Any]:
    if not isinstance(measurement, dict):
        return {}
    context = measurement.get("geometry_context")
    if not isinstance(context, dict):
        return {}
    return {
        "measurement_index": int(measurement.get("measurement_index", 0)),
        "geometry_context": context,
    }
def _part_grounding_context_preview(context: Any) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    return compact_part_grounding_table(context)
def _part_grounding_measurement_preview(measurement: Any) -> dict[str, Any]:
    if not isinstance(measurement, dict):
        return {}
    context = measurement.get("part_grounding_context")
    if not isinstance(context, Mapping):
        return {}
    return {
        "measurement_index": int(measurement.get("measurement_index", 0)),
        "source": str(measurement.get("source", "")),
        "part_grounding_context": _part_grounding_context_preview(context),
    }
def _live_geometry_context_preview(measurement: Any) -> dict[str, Any]:
    if not isinstance(measurement, dict):
        return {}
    context = measurement.get("geometry_context")
    if not isinstance(context, dict):
        return {}
    preview = {
        "measurement_index": int(measurement.get("measurement_index", 0)),
        "tool_result_index": int(measurement.get("tool_result_index", 0)),
        "env_id": int(measurement.get("env_id", 0)),
        "obj_id": int(measurement.get("obj_id", 0)),
        "geometry_context": context,
    }
    return preview
def _write_episode_observations(episode: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    observations_path = Path(str(episode["observations_path"])).expanduser().resolve()
    episode_dir = Path(str(episode["episode_dir"])).expanduser().resolve()
    _assert_workspace_episode_write_path(observations_path, episode_dir.parent)
    tool_results = list(diagnostics.get("tool_results", []))
    visual_evidence = list(diagnostics.get("visual_evidence", []))
    triple_view_evidence = list(diagnostics.get("triple_view_evidence", []))
    live_geometry_context_measurements = list(diagnostics.get("live_geometry_context_measurements", []))
    payload = {
        "schema_version": "hag4r-diagnostic-episode-observations-v1",
        "episode_index": int(episode["episode_index"]),
        "episode_id": str(episode["episode_id"]),
        "intent": str(episode.get("intent", "")),
        "status": str(episode.get("status", "")),
        "tool_result_indices": list(episode.get("tool_result_indices", [])),
        "visual_evidence_indices": list(episode.get("visual_evidence_indices", [])),
        "triple_view_evidence": triple_view_evidence,
        "live_geometry_context_measurement_indices": list(episode.get("live_geometry_context_measurement_indices", [])),
        "expected_observations": list(episode.get("expected_observations", [])),
        "actual_observations": list(episode.get("actual_observations", [])),
        "video_path": str(episode.get("video_path", "")),
        "video": episode.get("video", {}),
        "triptych_png_sequence": episode.get("triptych_png_sequence", {}),
        "triptych_video_path": str(episode.get("triptych_video_path", "")),
        "triptych_video": episode.get("triptych_video", {}),
        "media_status": str(episode.get("media_status", "")),
        "tool_results": [
            tool_results[index]
            for index in episode.get("tool_result_indices", [])
            if isinstance(index, int) and 0 <= index < len(tool_results)
        ],
        "visual_evidence": [
            visual_evidence[index]
            for index in episode.get("visual_evidence_indices", [])
            if isinstance(index, int) and 0 <= index < len(visual_evidence)
        ],
        "live_geometry_context_measurements": [
            live_geometry_context_measurements[index]
            for index in episode.get("live_geometry_context_measurement_indices", [])
            if isinstance(index, int) and 0 <= index < len(live_geometry_context_measurements)
        ],
    }
    anchor = _anchor_region_preview(episode.get("anchor_region"))
    if anchor:
        payload["anchor_region"] = anchor
    if episode.get("setup_trial_id"):
        payload["setup_trial_id"] = str(episode["setup_trial_id"])
    if episode.get("anchor_compile_id"):
        payload["anchor_compile_id"] = str(episode["anchor_compile_id"])
    if episode.get("anchor_preview_evidence_id"):
        payload["anchor_preview_evidence_id"] = str(episode["anchor_preview_evidence_id"])
    setup_validation = episode.get("anchor_setup_validation")
    if isinstance(setup_validation, dict):
        payload["anchor_setup_validation"] = {
            key: setup_validation[key]
            for key in (
                "schema_version",
                "status",
                "anchor_id",
                "compile_id",
                "selected_part_id",
                "region_hint",
                "local_anchor_box",
                "executable_pin_box",
                "executable_frame",
                "warning_codes",
                "pinned_vertex_distribution",
            )
            if key in setup_validation
        }
    setup_quality = _setup_quality_preview(episode.get("setup_quality"))
    if setup_quality:
        payload["setup_quality"] = setup_quality
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    observations_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
def _runtime_owned_workspace_paths(state: dict[str, Any]) -> dict[str, Path]:
    paths = state["paths"]
    workspace_dir = Path(str(paths["diagnostic_workspace_dir"])).expanduser().resolve()
    owned_paths = {
        "index_json": Path(str(paths["diagnostic_workspace_index_path"])).expanduser().resolve(),
        "index_md": Path(str(paths["diagnostic_workspace_markdown_index_path"])).expanduser().resolve(),
        "diagnostic_log": Path(str(paths["diagnostic_log_path"])).expanduser().resolve(),
        "observations_digest": Path(str(paths["diagnostic_observations_digest_path"])).expanduser().resolve(),
        "part_grounding_context": workspace_dir / "part_grounding_context.json",
        "part_grounding_table": workspace_dir / "part_grounding_table.md",
    }
    for name, path in owned_paths.items():
        try:
            path.relative_to(workspace_dir)
        except ValueError as exc:
            raise ValueError(f"runtime-owned diagnostic workspace file is outside workspace: {name}={path}") from exc
    return {"workspace_dir": workspace_dir, **owned_paths}
def _runtime_event_detail_preview(detail: Any) -> Any:
    if not isinstance(detail, dict):
        return detail
    preview: dict[str, Any] = {}
    for key in (
        "tool_name",
        "tool",
        "status",
        "episode_id",
        "episode_index",
        "phase",
        "route",
        "route_relevance",
        "knowledge_base_entry_ids",
        "evidence_mode",
        "ready",
        "next_action",
        "error",
        "observation",
        "reflection",
        "route_adjudication",
        "path",
        "artifact_paths",
    ):
        if key in detail:
            preview[key] = detail[key]
    if "decision" in detail and isinstance(detail["decision"], dict):
        decision = detail["decision"]
        preview["decision"] = {
            key: decision[key]
            for key in ("decision_id", "tool", "expected_observation", "rationale")
            if key in decision
        }
    if "anchors" in detail:
        preview["diagnostic_session_plan"] = _session_plan_preview(detail)
    if "session_plan" in detail:
        session_plan = _session_plan_preview(detail["session_plan"])
        if session_plan:
            preview["diagnostic_session_plan"] = session_plan
    if "setup_trial" in detail:
        setup_trial = _setup_trial_preview(detail["setup_trial"])
        if setup_trial:
            preview["setup_trial"] = setup_trial
    if "geometry_context" in detail:
        geometry_context = _geometry_context_preview(detail["geometry_context"])
        if geometry_context:
            preview["geometry_context"] = geometry_context
    if "part_grounding_context" in detail:
        value = detail["part_grounding_context"]
        if isinstance(value, dict) and "part_grounding_context" in value:
            part_grounding_context = _part_grounding_measurement_preview(value)
        else:
            part_grounding_context = _part_grounding_context_preview(value)
        if part_grounding_context:
            preview["part_grounding_context"] = part_grounding_context
    if "live_geometry_context" in detail:
        live_geometry_context = _live_geometry_context_preview(detail["live_geometry_context"])
        if live_geometry_context:
            preview["live_geometry_context"] = live_geometry_context
    if "setup_quality" in detail:
        setup_quality = _setup_quality_preview(detail["setup_quality"])
        if setup_quality:
            preview["setup_quality"] = setup_quality
    if "anchor_region" in detail:
        anchor = _anchor_region_preview(detail["anchor_region"])
        if anchor:
            preview["anchor_region"] = anchor
    if "episode" in detail and isinstance(detail["episode"], dict):
        episode = dict(detail["episode"])
        anchor = _anchor_region_preview(episode.get("anchor_region"))
        if anchor:
            episode["anchor_region"] = anchor
        preview["episode"] = {
            key: episode[key]
            for key in (
                "episode_index",
                "episode_id",
                "episode_dir",
                "scene_config_path",
                "observations_path",
                "anchor_region",
                "setup_trial_id",
                "setup_quality",
                "anchor_compile_id",
                "anchor_preview_evidence_id",
                "anchor_setup_validation",
            )
            if key in episode
        }
    if "pinning" in detail and isinstance(detail["pinning"], dict):
        pinning = detail["pinning"]
        boxes = pinning.get("boxes", [])
        preview["pinning"] = {
            "enabled": bool(pinning.get("enabled", False)),
            "boxes": [
                {
                    key: box[key]
                    for key in (
                        "name",
                        "source",
                        "anchor_id",
                        "anchor_type",
                        "semantic_region",
                        "uncertainty",
                        "concerns",
                    )
                    if isinstance(box, dict) and key in box
                }
                for box in boxes[:4]
            ]
            if isinstance(boxes, list)
            else [],
        }
    if "result" in detail and isinstance(detail["result"], dict):
        result = detail["result"]
        preview["result"] = {
            key: result[key]
            for key in ("tool", "status", "error", "payload_artifacts")
            if key in result
        }
    return preview or {"keys": sorted(str(key) for key in detail.keys())}
def _append_diagnostic_runtime_event(
    state: dict[str, Any],
    *,
    provenance: str,
    event: str,
    note: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    allowed = {
        "runtime_event",
        "live_tool_result",
        "typed_model_reflection",
        "typed_model_anchor_target",
        "typed_model_anchor_target_edit",
        "typed_model_probe_target",
        "typed_model_probe_target_edit",
        "runtime_anchor_target_compiler",
        "runtime_anchor_target_preview",
        "runtime_probe_target_compiler",
        "runtime_probe_target_preview",
        "terminal_recommendation",
        "video_generation",
        "artifact_generation",
        "visual_sequence_materialization",
        "summary_report",
        "material_audit",
    }
    if provenance not in allowed:
        raise ValueError(f"unknown diagnostic runtime event provenance: {provenance}")
    diagnostics = _diagnostics(state)
    events = diagnostics.setdefault("runtime_owned_events", [])
    if not isinstance(events, list):
        raise ValueError("diagnostics.runtime_owned_events must be a list")
    previous_ids = [int(item.get("event_id", 0)) for item in events if isinstance(item, dict)]
    record = {
        "event_id": (max(previous_ids) if previous_ids else 0) + 1,
        "provenance": provenance,
        "event": str(event),
        "note": str(note),
        "detail": _runtime_event_detail_preview(detail),
    }
    for key in ("episode_id", "tool_name", "tool"):
        value = detail.get(key)
        if value:
            record["tool_name" if key == "tool" else key] = str(value)
    if "decision" in detail and isinstance(detail["decision"], dict) and detail["decision"].get("tool"):
        record["tool_name"] = str(detail["decision"]["tool"])
    events.append(record)
    return record
def _runtime_file_refs(state: dict[str, Any]) -> dict[str, str]:
    paths = state.get("paths", {}) if isinstance(state.get("paths"), dict) else {}
    return {
        "workspace_index_json": str(paths.get("diagnostic_workspace_index_path", "")),
        "workspace_index_md": str(paths.get("diagnostic_workspace_markdown_index_path", "")),
        "diagnostic_log": str(paths.get("diagnostic_log_path", "")),
        "observations_digest": str(paths.get("diagnostic_observations_digest_path", "")),
    }


def _coverage_summary_for_runtime_projection(state: dict[str, Any]) -> Any:
    """Project coverage without making it a revision persistence dependency."""
    diagnostics = _diagnostics(state)
    terminal = diagnostics.get("terminal")
    recommendation = terminal.get("recommendation") if isinstance(terminal, Mapping) else None
    if isinstance(recommendation, Mapping) and recommendation.get("recommendation") == "revise":
        return None
    return (
        build_diagnostic_coverage_summary(state)
        if diagnostics.get("active_diagnostic_session_plan") is not None
        else diagnostics.get("coverage_summary")
    )


def _build_observations_digest(state: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    coverage_summary = _coverage_summary_for_runtime_projection(state)
    runtime_events = [event for event in diagnostics.get("runtime_owned_events", []) if isinstance(event, dict)]
    provenance_categories = sorted({str(event.get("provenance", "")) for event in runtime_events if event.get("provenance")})
    terminal = diagnostics.get("terminal") if isinstance(diagnostics.get("terminal"), dict) else {}
    route_state = state.get("route_state", {}) if isinstance(state.get("route_state"), dict) else {}
    diagnostic_route = route_state.get("diagnostic_route", {}) if isinstance(route_state.get("diagnostic_route"), dict) else {}
    report_refs = {
        "diagnostic_summary": str(state.get("paths", {}).get("sim_diagnostics_report_path", "")),
        "diagnostic_report": str(state.get("paths", {}).get("sim_diagnostics_markdown_report_path", "")),
        "diagnostic_cues": str(state.get("paths", {}).get("sim_diagnostics_cues_path", "")),
        **_runtime_file_refs(state),
    }
    return {
        "schema_version": "hag4r-diagnostic-observations-digest-v1",
        "run_id": state.get("run_id", ""),
        "status": diagnostics.get("status", ""),
        "provenance_categories": provenance_categories,
        "runtime_owned_event_ids": [event.get("event_id") for event in runtime_events],
        "runtime_owned_events": [
            {
                "event_id": event.get("event_id"),
                "provenance": event.get("provenance", ""),
                "event": event.get("event", ""),
                "note": event.get("note", ""),
                "episode_id": event.get("episode_id", ""),
                "tool_name": event.get("tool_name", ""),
                "detail": event.get("detail", {}),
            }
            for event in runtime_events
        ],
        "diagnostic_session_plan": _session_plan_preview(diagnostics.get("active_diagnostic_session_plan")),
        "setup_trials": [
            preview
            for trial in diagnostics.get("setup_trials", [])
            if (preview := _setup_trial_preview(trial))
        ],
        "geometry_context_measurements": [
            preview
            for measurement in diagnostics.get("geometry_context_measurements", [])
            if (preview := _geometry_context_preview(measurement))
        ],
        "part_grounding_context": _part_grounding_context_preview(diagnostics.get("part_grounding_context")),
        "part_grounding_context_measurements": [
            preview
            for measurement in diagnostics.get("part_grounding_context_measurements", [])
            if (preview := _part_grounding_measurement_preview(measurement))
        ],
        "part_grounding_artifacts": list(diagnostics.get("part_grounding_artifacts", [])),
        "source_semantic_material_audits": [
            {
                "audit_id": audit.get("audit_id", ""),
                "active_revision_id": audit.get("active_revision_id", ""),
                "status_counts": audit.get("status_counts", {}),
                "canonical_material_hint_count": len(audit.get("canonical_material_hints", [])),
                "scope": "constitutive material-plan evidence only; not realized structural compliance",
            }
            for audit in diagnostics.get("source_semantic_material_audits", [])
            if isinstance(audit, Mapping)
        ],
        "active_source_semantic_material_audit_id": diagnostics.get("active_source_semantic_material_audit_id"),
        "coverage_summary": coverage_summary,
        "anchor_target_intents": [
            _anchor_target_intent_summary(intent)
            for intent in diagnostics.get("anchor_target_intents", [])
            if isinstance(intent, Mapping)
        ],
        "active_anchor_target_intent_by_anchor": dict(diagnostics.get("active_anchor_target_intent_by_anchor", {}))
        if isinstance(diagnostics.get("active_anchor_target_intent_by_anchor"), Mapping)
        else {},
        "compiled_anchor_targets": [
            _compiled_anchor_target_summary(compiled)
            for compiled in diagnostics.get("compiled_anchor_targets", [])
            if isinstance(compiled, Mapping)
        ],
        "active_compiled_anchor_target_by_anchor": dict(diagnostics.get("active_compiled_anchor_target_by_anchor", {}))
        if isinstance(diagnostics.get("active_compiled_anchor_target_by_anchor"), Mapping)
        else {},
        "anchor_target_edit_trials": _redact_internal_anchor_target(list(diagnostics.get("anchor_target_edit_trials", []))),
        "anchor_target_preview_artifacts": [
            _anchor_target_preview_summary(preview)
            for preview in diagnostics.get("anchor_target_preview_artifacts", [])
            if isinstance(preview, Mapping)
        ],
        "probe_target_intents": [
            _probe_target_intent_summary(intent)
            for intent in diagnostics.get("probe_target_intents", [])
            if isinstance(intent, Mapping)
        ],
        "compiled_probe_targets": [
            _compiled_probe_target_summary(compiled)
            for compiled in diagnostics.get("compiled_probe_targets", [])
            if isinstance(compiled, Mapping)
        ],
        "probe_target_edit_trials": _redact_internal_probe_target(list(diagnostics.get("probe_target_edit_trials", []))),
        "probe_target_preview_artifacts": [
            _probe_target_preview_summary(preview)
            for preview in diagnostics.get("probe_target_preview_artifacts", [])
            if isinstance(preview, Mapping)
        ],
        "probe_target_validations": [
            _probe_target_validation_summary(validation)
            for validation in diagnostics.get("probe_target_validations", [])
            if isinstance(validation, Mapping)
        ],
        "probe_target_applications": [
            {
                "application_index": application.get("application_index"),
                "tool": application.get("tool", ""),
                "target_id": application.get("target_id", ""),
                "compile_id": application.get("compile_id", ""),
                "decision_id": application.get("decision_id"),
                "tool_result_index": application.get("tool_result_index"),
                "internal_action": application.get("internal_action", ""),
                "requested_duration_steps": application.get("requested_duration_steps"),
                "effective_resume_steps": application.get("effective_resume_steps"),
                "adaptive_resume_applied": application.get("adaptive_resume_applied"),
                "estimated_motion_steps": application.get("estimated_motion_steps"),
                "adaptive_resume_source": application.get("adaptive_resume_source", ""),
                "probe_target_validation": application.get("probe_target_validation", {}),
                "episode_id": application.get("episode_id", ""),
            }
            for application in diagnostics.get("probe_target_applications", [])
            if isinstance(application, Mapping)
        ],
        "live_geometry_context_measurements": [
            preview
            for measurement in diagnostics.get("live_geometry_context_measurements", [])
            if (preview := _live_geometry_context_preview(measurement))
        ],
        "episodes": [
            {
                "episode_index": episode.get("episode_index"),
                "episode_id": episode.get("episode_id", ""),
                "status": episode.get("status", ""),
                "intent": episode.get("intent", ""),
                "observations_path": episode.get("observations_path", ""),
                "tool_result_indices": list(episode.get("tool_result_indices", [])),
                "visual_evidence_indices": list(episode.get("visual_evidence_indices", [])),
                "video_path": episode.get("video_path", ""),
                "video": episode.get("video", {}),
                "triptych_png_sequence": episode.get("triptych_png_sequence", {}),
                "triptych_video_path": episode.get("triptych_video_path", ""),
                "triptych_video": episode.get("triptych_video", {}),
                "media_status": episode.get("media_status", ""),
                "anchor_region": _anchor_region_preview(episode.get("anchor_region")),
                "setup_trial_id": episode.get("setup_trial_id", ""),
                "setup_quality": _setup_quality_preview(episode.get("setup_quality")),
                "anchor_compile_id": episode.get("anchor_compile_id", ""),
                "anchor_preview_evidence_id": episode.get("anchor_preview_evidence_id", ""),
                "anchor_setup_validation": episode.get("anchor_setup_validation", {}),
            }
            for episode in diagnostics.get("episodes", [])
            if isinstance(episode, dict)
        ],
        "live_tool_results": [
            {
                "index": index,
                "tool": result.get("tool", ""),
                "episode_id": result.get("episode_id", ""),
                "expected_observation": result.get("expected_observation", ""),
                "actual_observation": result.get("actual_observation", ""),
                "status": result.get("result", {}).get("status", "") if isinstance(result.get("result"), dict) else "",
                "visual_evidence": bool(result.get("visual_evidence")),
            }
            for index, result in enumerate(diagnostics.get("tool_results", []))
            if isinstance(result, dict)
        ],
        "expected_observations": list(diagnostics.get("expected_observations", [])),
        "actual_observations": list(diagnostics.get("actual_observations", [])),
        "visual_evidence_refs": [
            {
                "index": index,
                "episode_id": evidence.get("episode_id", ""),
                "available": evidence.get("available", False),
                "count": evidence.get("count", 0),
                "triple_view_evidence_id": evidence.get("triple_view_evidence_id", ""),
                "triptych_png_paths": list(evidence.get("triptych_png_paths", []))
                if isinstance(evidence.get("triptych_png_paths"), list)
                else [],
                "model_visible_as": evidence.get("model_visible_as", ""),
            }
            for index, evidence in enumerate(diagnostics.get("visual_evidence", []))
            if isinstance(evidence, dict)
        ],
        "triple_view_evidence_refs": [
            {
                "index": index,
                "evidence_id": evidence.get("evidence_id", ""),
                "source_kind": evidence.get("source_kind", ""),
                "episode_id": evidence.get("episode_id", ""),
                "target_id": evidence.get("target_id", ""),
                "compile_id": evidence.get("compile_id", ""),
                "triptych_png_path": evidence.get("triptych_png_path", ""),
                "triple_view_manifest_path": evidence.get("triple_view_manifest_path", ""),
                "panel_order": list(evidence.get("panel_order", [])) if isinstance(evidence.get("panel_order"), list) else [],
            }
            for index, evidence in enumerate(diagnostics.get("triple_view_evidence", []))
            if isinstance(evidence, dict)
        ],
        "model_reflections": list(diagnostics.get("model_reflections", [])),
        "source_semantic_material_audits": list(diagnostics.get("source_semantic_material_audits", [])),
        "source_semantic_material_audit_scope": "constitutive material-plan evidence only; not realized structural compliance",
        "route_adjudication": diagnostics.get("route_adjudication", {}),
        "terminal": {
            "status": terminal.get("status", ""),
            "tool_name": terminal.get("tool_name", ""),
            "route": terminal.get("route", ""),
            "ready": bool(terminal.get("ready", False)),
            "validated": bool(terminal.get("validated", False)),
            "error": terminal.get("error", ""),
            "evidence_mode": terminal.get("evidence_mode", ""),
            "live_probe_evidence": terminal.get("live_probe_evidence", {}),
            "runtime_failure_evidence": terminal.get("runtime_failure_evidence", {}),
        },
        "route": diagnostic_route.get("route", terminal.get("route", "")),
        "ready": bool(diagnostic_route.get("ready", terminal.get("ready", False))),
        "evidence_mode": diagnostic_route.get("evidence_mode", terminal.get("evidence_mode", "")),
        "videos": list(diagnostics.get("videos", [])),
        "video_paths": list(diagnostics.get("video_paths", [])),
        "triptych_videos": list(diagnostics.get("triptych_videos", [])),
        "triptych_video_paths": list(diagnostics.get("triptych_video_paths", [])),
        "report_refs": report_refs,
        "runtime_owned_files": _runtime_file_refs(state),
    }
def _write_runtime_diagnostic_log(path: Path, state: dict[str, Any]) -> None:
    diagnostics = _diagnostics(state)
    events = [event for event in diagnostics.get("runtime_owned_events", []) if isinstance(event, dict)]
    lines = ["# HAG4R Diagnostic Log", ""]
    for event in events:
        lines.extend(
            [
                f"## {int(event.get('event_id', 0)):04d} {event.get('event', '')}",
                "",
                f"- provenance: {event.get('provenance', '')}",
                f"- note: {event.get('note', '')}",
            ]
        )
        if event.get("episode_id"):
            lines.append(f"- episode_id: {event['episode_id']}")
        if event.get("tool_name"):
            lines.append(f"- tool_name: {event['tool_name']}")
        detail = event.get("detail")
        if detail:
            detail_text = json.dumps(detail, indent=2, sort_keys=True, default=str)
            if len(detail_text) > 4000:
                detail_text = detail_text[:4000].rstrip() + "\n..."
            lines.extend(["", "```json", detail_text, "```"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _diagnostic_workspace_virtual_prefix(state: dict[str, Any]) -> str:
    run_root = Path(str(state["run_root"])).expanduser().resolve()
    run_tag = run_root.name
    revision_id = str(state.get("active_revision") or "revision_0000")
    return f"/outputs/agentic_asset_refinement/{run_tag}/revisions/{revision_id}/sim_diagnostics/"


def _workspace_relative_artifact_entry(
    *,
    artifact_id: str,
    artifact_path: Path,
    workspace_dir: Path,
    virtual_prefix: str,
    source_path: str = "",
    source_virtual_path: str = "",
) -> dict[str, Any]:
    relative_path = artifact_path.relative_to(workspace_dir).as_posix()
    return {
        "artifact_id": artifact_id,
        "path": relative_path,
        "virtual_path": virtual_prefix + relative_path,
        "lineage": {
            "source_path": source_path,
            "source_virtual_path": source_virtual_path,
        },
    }


def _write_part_grounding_artifacts(paths: dict[str, Path], state: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = _diagnostics(state)
    context = diagnostics.get("part_grounding_context")
    if not isinstance(context, Mapping):
        return []
    context_path = paths["part_grounding_context"]
    table_path = paths["part_grounding_table"]
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    table_path.write_text(render_part_grounding_markdown_table(context), encoding="utf-8")
    workspace_dir = Path(str(state["paths"]["diagnostic_workspace_dir"])).expanduser().resolve()
    virtual_prefix = _diagnostic_workspace_virtual_prefix(state)
    return [
        _workspace_relative_artifact_entry(
            artifact_id="part_grounding_context",
            artifact_path=context_path,
            workspace_dir=workspace_dir,
            virtual_prefix=virtual_prefix,
        ),
        _workspace_relative_artifact_entry(
            artifact_id="part_grounding_table",
            artifact_path=table_path,
            workspace_dir=workspace_dir,
            virtual_prefix=virtual_prefix,
        ),
    ]


_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_PATH_KEYS = {
    "raw_source_image": "source_image",
    "cleaned_source_image": "cleaned_image_path",
    "object_description": "object_description_path",
    "material_inference_payload": "inferred_params_path",
}
_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS = frozenset(
    {*_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_PATH_KEYS, "part_grounding_context"}
)


def _material_audit_registration_records(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise GenesisVlmSchemaError("material audit diagnostics state is malformed")
    records = diagnostics.get("source_semantic_material_artifact_registrations", [])
    if not isinstance(records, list) or not all(isinstance(item, Mapping) for item in records):
        raise GenesisVlmSchemaError("material audit artifact registration lineage is malformed")
    return [dict(item) for item in records]


def _active_material_audit_registration(state: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = state.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise GenesisVlmSchemaError("material audit diagnostics state is malformed")
    revision_id = str(state.get("active_revision") or "")
    registration_id = diagnostics.get("active_source_semantic_material_artifact_registration_id")
    if not isinstance(registration_id, str) or not registration_id:
        raise GenesisVlmSchemaError("material audit requires an immutable active-revision artifact registration")
    records = _material_audit_registration_records(state)
    matching = [item for item in records if item.get("registration_id") == registration_id]
    if len(matching) != 1:
        raise GenesisVlmSchemaError("material audit active artifact registration lineage is ambiguous or missing")
    registration = matching[0]
    if registration.get("active_revision_id") != revision_id:
        raise GenesisVlmSchemaError("material audit artifact registration has stale revision")
    same_revision = [item for item in records if item.get("active_revision_id") == revision_id]
    if len(same_revision) != 1:
        raise GenesisVlmSchemaError("material audit artifact registration lineage has duplicate active-revision records")
    legacy = diagnostics.get("source_semantic_material_artifact_registration")
    if not isinstance(legacy, Mapping) or dict(legacy) != registration:
        raise GenesisVlmSchemaError("material audit active artifact registration pointer is inconsistent")
    return registration


def initialize_source_semantic_material_artifact_registration(
    state: dict[str, Any],
    *,
    runtime_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Freeze the five audit inputs at a normal active-revision session boundary."""
    diagnostics = _diagnostics(state)
    paths = state.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("state.paths must be an object")
    revision_id = str(state.get("active_revision") or "revision_0000")
    existing = [
        item for item in _material_audit_registration_records(state)
        if item.get("active_revision_id") == revision_id
    ]
    if existing:
        if len(existing) != 1:
            raise GenesisVlmSchemaError("material audit artifact registration lineage has duplicate active-revision records")
        registration = existing[0]
        if diagnostics.get("active_source_semantic_material_artifact_registration_id") != registration.get("registration_id"):
            raise GenesisVlmSchemaError("material audit active artifact registration pointer is inconsistent")
        if diagnostics.get("source_semantic_material_artifact_registration") != registration:
            raise GenesisVlmSchemaError("material audit active artifact registration pointer is inconsistent")
        return registration
    candidates: dict[str, Path | None] = {
        artifact_id: _optional_path(dict(paths), path_key)
        for artifact_id, path_key in _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_PATH_KEYS.items()
    }
    candidates["part_grounding_context"] = runtime_paths["part_grounding_context"]
    artifacts: list[dict[str, Any]] = []
    for artifact_id in sorted(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS):
        candidate = candidates[artifact_id]
        if candidate is None or not candidate.is_file():
            source_path = str(candidate.expanduser().resolve()) if candidate is not None else ""
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "path": "",
                    "source_path": source_path,
                    "source_repository_relative_path": "",
                    "source_virtual_path": "",
                    "workspace_path": "",
                    "workspace_virtual_path": "",
                    "active_revision_id": revision_id,
                    "sha256": "",
                    "source_sha256": "",
                    "workspace_sha256": "",
                    "available": False,
                    "unavailable_reason": "missing authoritative artifact",
                }
            )
            continue
        path = candidate.expanduser().resolve()
        if artifact_id == "part_grounding_context":
            # This runtime-owned artifact already occupies the canonical
            # workspace slot.  Freeze its hash there rather than shadowing the
            # workspace index's established part_grounding_context entry.
            workspace_copy = path
        else:
            workspace_artifact_dir = runtime_paths["workspace_dir"] / "material_audit_authoritative" / revision_id
            workspace_artifact_dir.mkdir(parents=True, exist_ok=True)
            suffix = path.suffix or ".bin"
            workspace_copy = workspace_artifact_dir / f"{artifact_id}{suffix}"
            if workspace_copy.exists():
                raise GenesisVlmSchemaError(
                    f"material audit registration snapshot path already exists: {artifact_id}"
                )
            shutil.copy2(path, workspace_copy)
            workspace_copy = workspace_copy.expanduser().resolve()
        source_digest = sha256_file(path)
        workspace_digest = sha256_file(workspace_copy)
        if source_digest != workspace_digest:
            raise ValueError(f"material audit authoritative artifact copy hash mismatch: {artifact_id}")
        repo_root = Path(str(state.get("repo_root", ""))).expanduser().resolve()
        try:
            repository_relative_path = path.relative_to(repo_root).as_posix()
        except ValueError:
            repository_relative_path = ""
        workspace_path = workspace_copy.relative_to(runtime_paths["workspace_dir"]).as_posix()
        entry: dict[str, Any] = {
            "artifact_id": artifact_id,
            "path": workspace_path,
            "source_path": str(path),
            "source_repository_relative_path": repository_relative_path,
            "source_virtual_path": f"/{repository_relative_path}" if repository_relative_path else "",
            "workspace_path": workspace_path,
            "workspace_virtual_path": (
                _diagnostic_workspace_virtual_prefix(state) + workspace_path if workspace_path else ""
            ),
            "virtual_path": _diagnostic_workspace_virtual_prefix(state) + workspace_path,
            "lineage": {
                "source_path": str(path),
                "source_virtual_path": f"/{repository_relative_path}" if repository_relative_path else "",
            },
            "active_revision_id": revision_id,
            "sha256": source_digest,
            "source_sha256": source_digest,
            "workspace_sha256": workspace_digest,
            "available": True,
        }
        artifacts.append(entry)
    if {item["artifact_id"] for item in artifacts} != _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS:
        raise GenesisVlmSchemaError("material audit registration must contain exactly five authoritative artifacts")
    registration = {
        "schema_version": "hag4r-source-semantic-material-artifact-registration-v1",
        "registration_id": f"source_semantic_material_registration_{len(diagnostics['source_semantic_material_artifact_registrations']) + 1:04d}",
        "active_revision_id": revision_id,
        "complete": all(item.get("available") is True for item in artifacts),
        "artifacts": artifacts,
    }
    diagnostics["source_semantic_material_artifact_registrations"].append(registration)
    diagnostics["active_source_semantic_material_artifact_registration_id"] = registration["registration_id"]
    # Retain this current pointer for report compatibility; lineage is append-only above.
    diagnostics["source_semantic_material_artifact_registration"] = registration
    return registration


def _write_runtime_workspace_json_index(
    path: Path,
    state: dict[str, Any],
    *,
    part_grounding_artifacts: list[dict[str, Any]],
) -> None:
    diagnostics = _diagnostics(state)
    paths = state["paths"]
    workspace_dir = Path(str(paths["diagnostic_workspace_dir"])).expanduser().resolve()
    generated_episodes = Path(str(paths["diagnostic_generated_episodes_dir"])).expanduser().resolve()
    virtual_prefix = _diagnostic_workspace_virtual_prefix(state)
    generated_files = []
    if generated_episodes.is_dir():
        generated_files = [
            item.relative_to(workspace_dir).as_posix()
            for item in sorted(generated_episodes.rglob("*"))
            if item.is_file() and item.suffix.lower() in {".png", ".mp4"}
        ]
    part_grounding_context = diagnostics.get("part_grounding_context")
    if part_grounding_context is not None and not isinstance(part_grounding_context, Mapping):
        raise ValueError("diagnostics.part_grounding_context must be an object or null")
    registration = diagnostics.get("source_semantic_material_artifact_registration")
    registered_artifacts = (
        [dict(item) for item in registration.get("artifacts", []) if isinstance(item, Mapping)]
        if isinstance(registration, Mapping)
        else []
    )
    authoritative_artifacts = [item for item in registered_artifacts if item.get("available") is True]
    unavailable_authoritative_artifacts = [item for item in registered_artifacts if item.get("available") is not True]
    coverage_summary = _coverage_summary_for_runtime_projection(state)
    index = {
        "schema_version": "hag4r-diagnostic-workspace-index-v1",
        "run_id": state.get("run_id", ""),
        "revision_id": str(state.get("active_revision") or "revision_0000"),
        "workspace": {
            "path": str(workspace_dir),
            "virtual_path": virtual_prefix,
        },
        "generated_media": {
            "episodes_dir": str(generated_episodes),
            "files": generated_files,
        },
        "part_grounding": (
            compact_part_grounding_table(part_grounding_context)
            if part_grounding_context is not None
            else {}
        ),
        "artifacts": [
            *[item for item in part_grounding_artifacts if item.get("artifact_id") not in _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS],
            *authoritative_artifacts,
        ],
        "unavailable_artifacts": unavailable_authoritative_artifacts,
        "source_semantic_material_artifact_registration": (
            dict(registration) if isinstance(registration, Mapping) else None
        ),
        "active_source_semantic_material_artifact_registration_id": diagnostics.get(
            "active_source_semantic_material_artifact_registration_id"
        ),
        "source_semantic_material_audits": [
            {
                "audit_id": audit.get("audit_id", ""),
                "active_revision_id": audit.get("active_revision_id", ""),
                "status_counts": audit.get("status_counts", {}),
            }
            for audit in diagnostics.get("source_semantic_material_audits", [])
            if isinstance(audit, Mapping)
        ],
        "active_source_semantic_material_audit_id": diagnostics.get("active_source_semantic_material_audit_id"),
        "coverage_summary": coverage_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
def _write_runtime_workspace_markdown_index(path: Path, state: dict[str, Any]) -> None:
    diagnostics = _diagnostics(state)
    paths = state["paths"]
    index_json_path = Path(str(paths["diagnostic_workspace_index_path"])).expanduser().resolve()
    artifact_count = 0
    unavailable_count = 0
    workspace_virtual_path = ""
    generated_media = str(paths.get("diagnostic_generated_episodes_dir", ""))
    if index_json_path.is_file():
        index = json.loads(index_json_path.read_text(encoding="utf-8"))
        artifact_count = len(index.get("artifacts", []))
        unavailable_count = len(index.get("unavailable_artifacts", []))
        workspace_virtual_path = str(index.get("workspace", {}).get("virtual_path", ""))
        generated_media = str(index.get("generated_media", {}).get("episodes_dir", generated_media))
    terminal = diagnostics.get("terminal") if isinstance(diagnostics.get("terminal"), dict) else {}
    coverage_summary = index.get("coverage_summary") if index_json_path.is_file() else diagnostics.get("coverage_summary")
    lines = [
        "# HAG4R Diagnostic Workspace",
        "",
        f"- run_id: {state.get('run_id', '')}",
        f"- workspace: {workspace_virtual_path or paths.get('diagnostic_workspace_dir', '')}",
        f"- copied_artifacts: {artifact_count}",
        f"- unavailable_artifacts: {unavailable_count}",
        f"- episode_suites: {len(diagnostics.get('episodes', []))}",
        f"- generated_media_location: {generated_media}",
        f"- terminal_status: {terminal.get('status', '')}",
        f"- report_status: {diagnostics.get('status', '')}",
        f"- coverage_status: {coverage_summary.get('regions', [{}])[0].get('status', '') if isinstance(coverage_summary, Mapping) and coverage_summary.get('regions') else ''}",
        "",
        "## Runtime-Owned Files",
        "",
        f"- index_json: {paths.get('diagnostic_workspace_index_path', '')}",
        f"- index_md: {paths.get('diagnostic_workspace_markdown_index_path', '')}",
        f"- diagnostic_log: {paths.get('diagnostic_log_path', '')}",
        f"- observations_digest: {paths.get('diagnostic_observations_digest_path', '')}",
        "",
        "## Workspace Contents",
        "",
        "Read `index.json` first for copied artifact paths, unavailable artifacts, generated-media policy, and runtime facts.",
    ]
    if isinstance(coverage_summary, Mapping):
        lines.extend(["", "## Coverage Summary", "", f"- schema_version: {coverage_summary.get('schema_version', '')}", f"- selected_region_count: {coverage_summary.get('selected_region_count', '')}", "", "| region_id | risk_rank | status | episode_id | settlement_id |", "|---|---:|---|---|---|"])
        for region in coverage_summary.get("regions", []):
            if isinstance(region, Mapping):
                lines.append(f"| {region.get('region_id', '')} | {region.get('risk_rank', '')} | {region.get('status', '')} | {region.get('episode_id', '') or ''} | {region.get('settlement_id', '') or ''} |")
    if diagnostics.get("episodes"):
        lines.extend(["", "## Episode Suites", ""])
        for episode in diagnostics.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            lines.extend(
                [
                    f"- {episode.get('episode_id', '')}: {episode.get('status', '')}",
                    f"  - observations: {episode.get('observations_path', '')}",
                    f"  - generated_episode_dir: {episode.get('generated_episode_dir', '')}",
                    f"  - video: {episode.get('video_path', '')}",
                ]
            )
    part_grounding_context = diagnostics.get("part_grounding_context")
    if isinstance(part_grounding_context, Mapping):
        part_grounding = compact_part_grounding_table(part_grounding_context)
        lines.extend(["", "## Part Grounding", ""])
        labels = part_grounding.get("labels", {}) if isinstance(part_grounding.get("labels"), dict) else {}
        mesh = part_grounding.get("mesh", {}) if isinstance(part_grounding.get("mesh"), dict) else {}
        lines.extend(
            [
                f"- schema_version: {part_grounding.get('schema_version', '')}",
                f"- label_array_key: {labels.get('array_key', '')}",
                f"- primitive_kind: {mesh.get('primitive_kind', '')}",
                "",
                "| part_id | name | material | primitives | vertices | bbox_m |",
                "|---:|---|---|---:|---:|---|",
            ]
        )
        for part in part_grounding.get("parts", []):
            if not isinstance(part, dict):
                continue
            bbox = part.get("bbox_m", [])
            bbox_text = json.dumps(bbox, separators=(",", ":")) if bbox else ""
            lines.append(
                "| "
                f"{part.get('part_id', '')} | "
                f"{part.get('part_name', '')} | "
                f"{part.get('major_material_name', '')} | "
                f"{part.get('primitive_count', 0)} | "
                f"{part.get('vertex_count', 0)} | "
                f"{bbox_text} |"
            )
    if diagnostics.get("anchor_target_intents"):
        lines.extend(
            [
                "",
                "## Anchor Target Intents",
                "",
                "| intent_id | anchor_id | part_id | region_hint | uncertainty | anchor_intent |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for intent in diagnostics.get("anchor_target_intents", []):
            if not isinstance(intent, Mapping):
                continue
            summary = _anchor_target_intent_summary(intent)
            lines.append(
                "| "
                f"{summary.get('intent_id', '')} | "
                f"{summary.get('anchor_id', '')} | "
                f"{summary.get('selected_part_id', '')} | "
                f"{summary.get('region_hint', '')} | "
                f"{summary.get('uncertainty', '')} | "
                f"{summary.get('anchor_intent', '')} |"
            )
    if diagnostics.get("compiled_anchor_targets") or diagnostics.get("anchor_target_preview_artifacts"):
        lines.extend(["", "## Anchor Target Compiles", ""])
        compiled_anchors = [
            _compiled_anchor_target_summary(compiled)
            for compiled in diagnostics.get("compiled_anchor_targets", [])
            if isinstance(compiled, Mapping)
        ]
        if compiled_anchors:
            lines.extend(
                [
                    "| compile_id | anchor_id | part_id | region_hint | status | warnings | primitive_purity |",
                    "|---|---|---:|---|---|---|---:|",
                ]
            )
            for compiled in compiled_anchors:
                lines.append(
                    "| "
                    f"{compiled.get('compile_id', '')} | "
                    f"{compiled.get('anchor_id', '')} | "
                    f"{compiled.get('selected_part_id', '')} | "
                    f"{compiled.get('region_hint', '')} | "
                    f"{compiled.get('validation_status', '')} | "
                    f"{','.join(compiled.get('warning_codes', []))} | "
                    f"{compiled.get('primitive_purity', '')} |"
                )
        previews = [
            _anchor_target_preview_summary(preview)
            for preview in diagnostics.get("anchor_target_preview_artifacts", [])
            if isinstance(preview, Mapping)
        ]
        if previews:
            lines.extend(["", "## Anchor Target Previews", ""])
            for preview in previews:
                lines.append(
                    f"- {preview.get('trial_id', '')}: triptych={preview.get('triptych_png_path', '')} "
                    f"manifest={preview.get('triple_view_manifest_path', '')}"
                )
    if diagnostics.get("probe_target_intents") or diagnostics.get("compiled_probe_targets"):
        lines.extend(["", "## Probe Targets", ""])
        intents = [
            _probe_target_intent_summary(intent)
            for intent in diagnostics.get("probe_target_intents", [])
            if isinstance(intent, Mapping)
        ]
        if intents:
            lines.extend(["### Intents", "", "| target_id | part_id | region_hint | target_intent |", "|---|---:|---|---|"])
            for intent in intents:
                lines.append(
                    "| "
                    f"{intent.get('target_id', '')} | "
                    f"{intent.get('selected_part_id', '')} | "
                    f"{intent.get('region_hint', '')} | "
                    f"{intent.get('target_intent', '')} |"
                )
        compiled_targets = [
            _compiled_probe_target_summary(compiled)
            for compiled in diagnostics.get("compiled_probe_targets", [])
            if isinstance(compiled, Mapping)
        ]
        if compiled_targets:
            lines.extend(
                [
                    "",
                    "### Compiled Targets",
                    "",
                    "| compile_id | target_id | part_id | region_hint | status | warnings | pin_overlap |",
                    "|---|---|---:|---|---|---|---:|",
                ]
            )
            for compiled in compiled_targets:
                metrics = compiled.get("metrics", {}) if isinstance(compiled.get("metrics"), dict) else {}
                lines.append(
                    "| "
                    f"{compiled.get('compile_id', '')} | "
                    f"{compiled.get('target_id', '')} | "
                    f"{compiled.get('selected_part_id', '')} | "
                    f"{compiled.get('region_hint', '')} | "
                    f"{compiled.get('validation_status', '')} | "
                    f"{','.join(compiled.get('warning_codes', []))} | "
                    f"{metrics.get('pin_target_overlap_ratio', '')} |"
                )
        previews = [
            _probe_target_preview_summary(preview)
            for preview in diagnostics.get("probe_target_preview_artifacts", [])
            if isinstance(preview, Mapping)
        ]
        if previews:
            lines.extend(["", "### Previews", ""])
            for preview in previews:
                lines.append(
                    f"- {preview.get('trial_id', '')}: triptych={preview.get('triptych_png_path', '')} "
                    f"manifest={preview.get('triple_view_manifest_path', '')}"
                )
    triple_view_evidence = [
        evidence
        for evidence in diagnostics.get("triple_view_evidence", [])
        if isinstance(evidence, Mapping)
    ]
    if triple_view_evidence:
        lines.extend(["", "## Triple-View Evidence", ""])
        for evidence in triple_view_evidence:
            lines.append(
                f"- {evidence.get('evidence_id', '')}: {evidence.get('source_kind', '')} "
                f"triptych={evidence.get('triptych_png_path', '')} "
                f"manifest={evidence.get('triple_view_manifest_path', '')}"
            )
    validations = [
        _probe_target_validation_summary(validation)
        for validation in diagnostics.get("probe_target_validations", [])
        if isinstance(validation, Mapping)
    ]
    if validations:
        lines.extend(
            [
                "",
                "### Grabbed Vertex Validations",
                "",
                "| compile_id | status | grabbed_selected_part_fraction | warnings | hard_errors |",
                "|---|---|---:|---|---|",
            ]
        )
        for validation in validations:
            lines.append(
                "| "
                f"{validation.get('compile_id', '')} | "
                f"{validation.get('status', '')} | "
                f"{validation.get('grabbed_selected_part_fraction', '')} | "
                f"{','.join(validation.get('warning_codes', []))} | "
                f"{','.join(validation.get('hard_errors', []))} |"
            )
    artifact_paths = diagnostics.get("artifact_paths") if isinstance(diagnostics.get("artifact_paths"), dict) else {}
    if artifact_paths:
        lines.extend(["", "## Reports", ""])
        for key, value in sorted(artifact_paths.items()):
            lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
def refresh_runtime_owned_diagnostic_files(
    run_root: str | Path,
    state: dict[str, Any],
    *,
    provenance: str,
    event: str | None,
    note: str,
    detail: dict[str, Any],
) -> None:
    del run_root
    if event:
        _append_diagnostic_runtime_event(
            state,
            provenance=provenance,
            event=event,
            note=note,
            detail=detail,
        )
    runtime_paths = _runtime_owned_workspace_paths(state)
    part_grounding_artifacts = _write_part_grounding_artifacts(runtime_paths, state)
    _write_runtime_workspace_json_index(
        runtime_paths["index_json"],
        state,
        part_grounding_artifacts=part_grounding_artifacts,
    )
    _write_runtime_workspace_markdown_index(runtime_paths["index_md"], state)
    _write_runtime_diagnostic_log(runtime_paths["diagnostic_log"], state)
    digest = _build_observations_digest(state)
    runtime_paths["observations_digest"].parent.mkdir(parents=True, exist_ok=True)
    runtime_paths["observations_digest"].write_text(
        json.dumps(digest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
def _save_diagnostics(
    run_root: str | Path,
    state: dict[str, Any],
    *,
    event: str,
    note: str,
    detail: dict[str, Any],
    provenance: str = "runtime_event",
) -> None:
    refresh_runtime_owned_diagnostic_files(
        run_root,
        state,
        provenance=provenance,
        event=event,
        note=note,
        detail=detail,
    )
    save_state(state, run_root)
    append_history(run_root, "genesis_diagnostics", note, event=event, detail=detail)
def _compact_actual_observation(result: dict[str, Any]) -> str:
    status = str(result.get("status", ""))
    error = result.get("error")
    if isinstance(error, dict) and error.get("message"):
        return f"{status}: {error['message']}"
    if result.get("result"):
        return f"{status}: Genesis live tool returned structured telemetry."
    return status or "Genesis live tool returned no status."
def _visual_evidence_from_live_result(
    *,
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    if tool_name != "pause_and_observe" and tool_name not in LIVE_VISUAL_TOOL_NAMES:
        return None
    live_result = result.get("result")
    if not isinstance(live_result, dict):
        return None
    renderer = live_result.get("renderer") if isinstance(live_result.get("renderer"), dict) else {}
    frame_ids = [int(frame_id) for frame_id in live_result.get("frame_ids", []) if isinstance(frame_id, int)]
    part_segmentation_png_paths = _string_list(live_result.get("part_segmentation_png_paths"))
    rgb_png_paths = _string_list(live_result.get("rgb_png_paths"))
    triptych_png_paths = _string_list(live_result.get("triptych_png_paths"))
    depth_png_paths = _string_list(live_result.get("depth_png_paths"))
    von_mises_png_paths = _string_list(live_result.get("von_mises_png_paths"))
    triple_view_manifest_paths = _string_list(live_result.get("triple_view_manifest_paths"))
    triple_view_evidence_ids = _string_list(live_result.get("triple_view_evidence_ids"))
    count = int(live_result.get("count", len(frame_ids)) or len(frame_ids))
    sequence_error = live_result.get("sequence_error")
    if not (part_segmentation_png_paths or rgb_png_paths or triptych_png_paths or depth_png_paths or von_mises_png_paths):
        reason = "frame sequence unavailable"
        if isinstance(sequence_error, dict) and sequence_error.get("message"):
            reason = str(sequence_error["message"])
        return {
            "tool": tool_name,
            "available": False,
            "reason": str(renderer.get("reason") or reason or result.get("status")),
            "frame_ids": frame_ids,
            "count": count,
            "renderer": renderer,
            "sequence_error": sequence_error,
        }
    is_historical_rgb = (
        str(renderer.get("mode", "")) == "rgb_triptych"
        or (bool(rgb_png_paths) and not part_segmentation_png_paths)
    )
    evidence = {
        "tool": tool_name,
        "available": True,
        "frame_ids": frame_ids,
        "count": count,
        "triptych_png_paths": triptych_png_paths,
        "triple_view_manifest_path": str(live_result.get("triple_view_manifest_path", "")),
        "triple_view_evidence_id": str(live_result.get("triple_view_evidence_id", "")),
        "triple_view_manifest_paths": triple_view_manifest_paths,
        "triple_view_evidence_ids": triple_view_evidence_ids,
        "depth_png_paths": depth_png_paths,
        "von_mises_png_paths": von_mises_png_paths,
        "renderer": renderer,
        "sequence_error": sequence_error,
    }
    if is_historical_rgb:
        evidence["rgb_png_paths"] = rgb_png_paths or triptych_png_paths
        evidence["model_visible_as"] = (
            "diagnostic_tool_result_triple_view_rgb_evidence"
            if triptych_png_paths
            else "diagnostic_tool_result_rgb_frame_sequence_evidence"
        )
    else:
        evidence["part_segmentation_png_paths"] = part_segmentation_png_paths or triptych_png_paths
        evidence["model_visible_as"] = (
            "diagnostic_tool_result_triple_view_part_segmentation_evidence"
            if triptych_png_paths
            else "diagnostic_tool_result_part_segmentation_frame_sequence_evidence"
        )
    return evidence
def _write_rgba_frame_png(field: dict[str, Any], path: Path) -> dict[str, Any]:
    shape = field.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or not all(isinstance(item, int) for item in shape)
        or shape[2] != 4
    ):
        raise GenesisVlmSchemaError("pause_and_observe RGB frame shape must be [height, width, 4]")
    height, width, _channels = shape
    encoded = field.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        raise GenesisVlmSchemaError("pause_and_observe RGB frame is missing data_base64")
    rgba = base64.b64decode(encoded)
    expected_bytes = int(height) * int(width) * 4
    if len(rgba) != expected_bytes:
        raise GenesisVlmSchemaError(
            f"pause_and_observe RGB frame byte count mismatch: expected {expected_bytes}, got {len(rgba)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGBA", (int(width), int(height)), rgba).save(path)
    return {
        key: value
        for key, value in field.items()
        if key != "data_base64"
    }
def _next_triple_view_evidence_id(state: dict[str, Any]) -> str:
    diagnostics = _diagnostics(state)
    return f"triple_view_{len(diagnostics['triple_view_evidence']) + 1:06d}"
def _compact_triple_view_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": str(manifest.get("evidence_id", "")),
        "source_kind": str(manifest.get("source_kind", "")),
        "target_id": str(manifest.get("target_id", "")),
        "compile_id": str(manifest.get("compile_id", "")),
        "trial_id": str(manifest.get("trial_id", "")),
        "episode_id": str(manifest.get("episode_id", "")),
        "tool_result_index": manifest.get("tool_result_index"),
        "panel_order": list(manifest.get("panel_order", [])) if isinstance(manifest.get("panel_order"), list) else [],
        "triptych_png_path": str(manifest.get("triptych_png_path", "")),
        "triple_view_manifest_path": str(manifest.get("manifest_path", "")),
        "triptych_sha256": str(manifest.get("triptych_sha256", "")),
        "triptych_dimensions": dict(manifest.get("triptych_dimensions", {}))
        if isinstance(manifest.get("triptych_dimensions"), Mapping)
        else {},
        "validation": dict(manifest.get("validation", {})) if isinstance(manifest.get("validation"), Mapping) else {},
    }
def _append_triple_view_evidence_once(state: dict[str, Any], record: Mapping[str, Any]) -> None:
    diagnostics = _diagnostics(state)
    evidence_id = str(record.get("evidence_id", ""))
    if not evidence_id:
        raise ValueError("triple-view evidence record is missing evidence_id")
    if any(isinstance(item, Mapping) and str(item.get("evidence_id")) == evidence_id for item in diagnostics["triple_view_evidence"]):
        return
    diagnostics["triple_view_evidence"].append(dict(record))
def _materialize_pause_and_observe_visual_frame(
    state: dict[str, Any],
    *,
    decision_id: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    if result.get("tool") != "pause_and_observe":
        return result
    live_result = result.get("result")
    if not isinstance(live_result, dict):
        return result
    triple_view = live_result.get("triple_view") if isinstance(live_result.get("triple_view"), dict) else None
    if triple_view is not None:
        captures = triple_view.get("captures") if isinstance(triple_view.get("captures"), dict) else {}
        missing = [view_name for view_name in TRIPLE_VIEW_PANEL_ORDER if view_name not in captures]
        extra = sorted(set(captures) - set(TRIPLE_VIEW_PANEL_ORDER))
        if missing or extra:
            raise GenesisVlmSchemaError(
                f"pause_and_observe triple_view captures must be exactly {list(TRIPLE_VIEW_PANEL_ORDER)}; "
                f"missing={missing}, extra={extra}"
            )
        compacted = dict(result)
        compact_live_result = dict(live_result)
        payload_dir = _diagnostic_payload_dir(state) / "visual_evidence"
        source_dir = payload_dir / f"decision_{decision_id:04d}_triple_view"
        evidence_id = _next_triple_view_evidence_id(state)
        source_png_paths_by_view: dict[str, Path] = {}
        cameras_by_view: dict[str, dict[str, Any]] = {}
        views_extra_by_view: dict[str, dict[str, Any]] = {}
        compact_captures: dict[str, dict[str, Any]] = {}
        for view_name in TRIPLE_VIEW_PANEL_ORDER:
            capture = captures[view_name]
            if not isinstance(capture, dict):
                raise GenesisVlmSchemaError(f"pause_and_observe triple_view capture is malformed for {view_name}")
            frame = capture.get("frame") if isinstance(capture.get("frame"), dict) else {}
            rgb = frame.get("rgb") if isinstance(frame.get("rgb"), dict) else {}
            png_path = source_dir / f"{view_name}.png"
            compact_rgb = _write_rgba_frame_png(rgb, png_path)
            compact_frame = dict(frame)
            compact_frame["rgb"] = {**compact_rgb, "png_path": str(png_path)}
            for component in ("depth", "von_mises"):
                value = compact_frame.get(component)
                if isinstance(value, dict) and "data_base64" in value:
                    compact_frame[component] = {key: item for key, item in value.items() if key != "data_base64"}
            camera = capture.get("camera") if isinstance(capture.get("camera"), dict) else {}
            frame_id = int(capture.get("frame_id"))
            sim_time_s = float(capture.get("sim_time_s"))
            source_png_paths_by_view[view_name] = png_path
            cameras_by_view[view_name] = dict(camera)
            views_extra_by_view[view_name] = {"frame_id": frame_id, "sim_time_s": sim_time_s}
            compact_captures[view_name] = {
                "frame_id": frame_id,
                "sim_time_s": sim_time_s,
                "camera": dict(camera),
                "frame": compact_frame,
            }
        triptych_path = payload_dir / f"decision_{decision_id:04d}_pause_and_observe_triptych.png"
        triptych_metadata = stitch_triptych(source_png_paths_by_view, triptych_path)
        manifest_path = payload_dir / f"decision_{decision_id:04d}_pause_and_observe_triple_view_manifest.json"
        episode = _active_episode_record(state)
        manifest = write_triple_view_manifest(
            manifest_path,
            evidence_id=evidence_id,
            source_kind="live_pause_observation",
            episode_id=str(episode["episode_id"]),
            tool_result_index=decision_id - 1,
            source_png_paths_by_view=source_png_paths_by_view,
            triptych_png_path=triptych_path,
            cameras_by_view=cameras_by_view,
            views_extra_by_view=views_extra_by_view,
            extra={"triptych": triptych_metadata},
        )
        manifest["manifest_path"] = str(manifest_path)
        compact_live_result["triple_view"] = {
            **triple_view,
            "captures": compact_captures,
            "triptych_png_path": str(triptych_path),
            "triple_view_manifest_path": str(manifest_path),
            "triple_view_evidence_id": evidence_id,
        }
        compact_live_result["frame"] = compact_captures["top"]["frame"]
        compact_live_result["renderer"] = compact_captures["top"]["frame"].get("renderer", {})
        compact_live_result["frame_ids"] = [int(capture["frame_id"]) for capture in compact_captures.values()]
        compact_live_result["rgb_png_paths"] = [str(triptych_path)]
        compact_live_result["triptych_png_paths"] = [str(triptych_path)]
        compact_live_result["depth_png_paths"] = []
        compact_live_result["von_mises_png_paths"] = []
        compact_live_result["triple_view_manifest_path"] = str(manifest_path)
        compact_live_result["triple_view_evidence_id"] = evidence_id
        compact_live_result["count"] = 1
        compact_live_result["sequence_error"] = None
        compacted["result"] = compact_live_result
        _append_triple_view_evidence_once(state, _compact_triple_view_record(manifest))
        return compacted
    frame = live_result.get("frame")
    if not isinstance(frame, dict):
        return result
    compacted = dict(result)
    compact_live_result = dict(live_result)
    compact_frame = dict(frame)
    renderer = frame.get("renderer") if isinstance(frame.get("renderer"), dict) else {}
    frame_id = int(result.get("frame_id") or 0)
    compact_live_result["renderer"] = renderer
    compact_live_result["frame_ids"] = [frame_id] if frame_id else []
    compact_live_result["rgb_png_paths"] = []
    compact_live_result["depth_png_paths"] = []
    compact_live_result["von_mises_png_paths"] = []
    compact_live_result["count"] = 0
    compact_live_result["sequence_error"] = None
    rgb = frame.get("rgb") if isinstance(frame.get("rgb"), dict) else {}
    if frame.get("supported") and rgb.get("status") == "ok" and isinstance(rgb.get("data_base64"), str):
        payload_dir = _diagnostic_payload_dir(state) / "visual_evidence"
        rgb_path = payload_dir / f"decision_{decision_id:04d}_pause_and_observe_frame_{frame_id:06d}_rgb.png"
        compact_frame["rgb"] = {
            **_write_rgba_frame_png(rgb, rgb_path),
            "png_path": str(rgb_path),
        }
        compact_live_result["rgb_png_paths"] = [str(rgb_path)]
        compact_live_result["count"] = 1
    elif isinstance(rgb, dict) and "data_base64" in rgb:
        compact_frame["rgb"] = {
            key: value
            for key, value in rgb.items()
            if key != "data_base64"
        }
    for component in ("depth", "von_mises"):
        value = compact_frame.get(component)
        if isinstance(value, dict) and "data_base64" in value:
            compact_frame[component] = {
                key: item
                for key, item in value.items()
                if key != "data_base64"
            }
    compact_live_result["frame"] = compact_frame
    compacted["result"] = compact_live_result
    return compacted
def _sequence_manifest_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]
    return [0, frame_count - 1]
def _materialize_live_visual_sequence(
    state: dict[str, Any],
    *,
    decision_id: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    tool_name = str(result.get("tool", ""))
    if tool_name not in LIVE_VISUAL_TOOL_NAMES:
        return result
    live_result = result.get("result")
    if not isinstance(live_result, dict):
        return result
    triple_view_sequence = live_result.get("triple_view_sequence")
    if not isinstance(triple_view_sequence, Mapping):
        return result
    frames = triple_view_sequence.get("frames")
    if not isinstance(frames, list):
        return result
    compacted = dict(result)
    compact_live_result = dict(live_result)
    payload_dir = _diagnostic_payload_dir(state) / "visual_evidence"
    episode = _active_episode_record(state)
    renderer = live_result.get("renderer") if isinstance(live_result.get("renderer"), Mapping) else {}
    if str(renderer.get("mode", "")) == "rgb_triptych" or (
        live_result.get("rgb_png_paths") and not live_result.get("part_segmentation_png_paths")
    ):
        source_kind = {
            "simulation_reset": "live_reset_observation",
            "simulate": "live_simulate_observation",
        }[tool_name]
    else:
        source_kind = LIVE_VISUAL_SOURCE_KIND_BY_TOOL[tool_name]
    manifest_paths: list[str] = []
    evidence_ids: list[str] = []
    for frame_list_index in _sequence_manifest_frame_indices(len(frames)):
        frame = frames[frame_list_index]
        if not isinstance(frame, Mapping):
            continue
        views = frame.get("views")
        if not isinstance(views, Mapping):
            continue
        source_png_paths_by_view: dict[str, Path] = {}
        for view_name in TRIPLE_VIEW_PANEL_ORDER:
            view = views.get(view_name)
            if not isinstance(view, Mapping):
                continue
            source_path = str(view.get("source_png_path", "")).strip()
            if source_path:
                source_png_paths_by_view[view_name] = Path(source_path).expanduser()
        if set(source_png_paths_by_view) != set(TRIPLE_VIEW_PANEL_ORDER):
            continue
        sequence_index = int(frame.get("sequence_index", frame_list_index) or frame_list_index)
        evidence_id = _next_triple_view_evidence_id(state)
        manifest_path = (
            payload_dir
            / f"decision_{decision_id:04d}_{tool_name}_frame_{sequence_index:06d}_triple_view_manifest.json"
        )
        views_extra = {}
        for view_name in TRIPLE_VIEW_PANEL_ORDER:
            view = views.get(view_name) if isinstance(views.get(view_name), Mapping) else {}
            extra_view = {
                "sequence_index": sequence_index,
                "frame_id": frame.get("frame_id"),
                "server_view_label": str(
                    view.get("server_view_label", "")
                ),
            }
            camera = view.get("camera")
            if isinstance(camera, Mapping):
                extra_view["camera"] = dict(camera)
            views_extra[view_name] = extra_view
        manifest = write_triple_view_manifest(
            manifest_path,
            evidence_id=evidence_id,
            source_kind=source_kind,
            episode_id=str(episode["episode_id"]),
            tool_result_index=decision_id - 1,
            source_png_paths_by_view=source_png_paths_by_view,
            triptych_png_path=str(frame.get("triptych_png_path", "")),
            views_extra_by_view=views_extra,
            extra={
                "source_tool": tool_name,
                "sequence_index": sequence_index,
                "frame_list_index": frame_list_index,
            },
        )
        manifest["manifest_path"] = str(manifest_path)
        _append_triple_view_evidence_once(state, _compact_triple_view_record(manifest))
        manifest_paths.append(str(manifest_path))
        evidence_ids.append(evidence_id)
    compact_live_result["triple_view_manifest_paths"] = manifest_paths
    compact_live_result["triple_view_evidence_ids"] = evidence_ids
    if manifest_paths and not compact_live_result.get("triple_view_manifest_path"):
        compact_live_result["triple_view_manifest_path"] = manifest_paths[-1]
    if evidence_ids and not compact_live_result.get("triple_view_evidence_id"):
        compact_live_result["triple_view_evidence_id"] = evidence_ids[-1]
    compact_live_result.setdefault("depth_png_paths", [])
    compact_live_result.setdefault("von_mises_png_paths", [])
    compacted["result"] = compact_live_result
    return compacted
def _repo_visible_path(state: dict[str, Any], path: Path) -> Path:
    repo_root = Path(str(state["repo_root"])).expanduser().resolve()
    resolved = path.expanduser().resolve()
    outputs_root = repo_root / "outputs"
    try:
        return outputs_root / resolved.relative_to(outputs_root.resolve())
    except ValueError:
        return path
def _diagnostic_payload_dir(state: dict[str, Any]) -> Path:
    diagnostics_dir = _repo_visible_path(state, Path(str(state["paths"]["diagnostic_workspace_dir"])).expanduser())
    return diagnostics_dir / "live_payloads"
def _json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, default=str))
def _artifact_ref(*, path: Path, value: Any) -> dict[str, Any]:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return {
        "artifact_path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
def _externalize_large_live_result(
    state: dict[str, Any],
    *,
    decision_id: int,
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "inspect_genesis_runtime_logs":
        return _validate_inline_runtime_log_result(result)
    payload_dir = _diagnostic_payload_dir(state)
    compacted = dict(result)
    live_result = compacted.get("result")
    if not isinstance(live_result, dict):
        return compacted
    compact_live_result: dict[str, Any] = {}
    artifact_paths: dict[str, dict[str, Any]] = {}
    always_inline_keys = {
        "last_resume_available",
        "resume_sequence_id",
        "frame_ids",
        "renderer",
        "part_segmentation_png_paths",
        "rgb_png_paths",
        "triptych_png_paths",
        "triple_view_manifest_path",
        "triple_view_evidence_id",
        "triple_view_manifest_paths",
        "triple_view_evidence_ids",
        "depth_png_paths",
        "von_mises_png_paths",
        "count",
        "sequence_error",
    }
    for key, value in live_result.items():
        if key in always_inline_keys or _json_size(value) <= MAX_INLINE_LIVE_RESULT_CHARS:
            compact_live_result[key] = value
            continue
        payload_dir.mkdir(parents=True, exist_ok=True)
        path = payload_dir / f"decision_{decision_id:04d}_{tool_name}_{key}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        reference = _artifact_ref(path=path, value=value)
        artifact_paths[key] = reference
        compact_live_result[key] = {
            "externalized": True,
            **reference,
        }
    compacted["result"] = compact_live_result
    if artifact_paths:
        compacted["payload_artifacts"] = artifact_paths
    return compacted


def _validate_inline_runtime_log_result(result: dict[str, Any]) -> dict[str, Any]:
    if set(result) != {"tool", "status", "result"}:
        raise ValueError("Genesis runtime log result has unexpected wrapper fields")
    if result.get("tool") != "inspect_genesis_runtime_logs" or result.get("status") != "ok":
        raise ValueError("Genesis runtime log result has an invalid tool or status")
    envelope = result.get("result")
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "stream",
        "contains",
        "context_lines",
        "limits",
        "output_truncated",
        "streams",
    }:
        raise ValueError("Genesis runtime log result has an invalid envelope")
    if envelope.get("schema_version") != GENESIS_RUNTIME_LOG_SCHEMA_VERSION:
        raise ValueError("Genesis runtime log result has an unsupported schema version")
    stream = envelope.get("stream")
    if stream not in {"stdout", "stderr", "both"}:
        raise ValueError("Genesis runtime log result has an invalid selected stream")
    contains = envelope.get("contains")
    if not isinstance(contains, str) or len(contains) > GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS:
        raise ValueError("Genesis runtime log result has an invalid literal query")
    context_lines = envelope.get("context_lines")
    if isinstance(context_lines, bool) or not isinstance(context_lines, int) or not 0 <= context_lines <= 5:
        raise ValueError("Genesis runtime log result has invalid context_lines")
    if not isinstance(envelope.get("output_truncated"), bool):
        raise ValueError("Genesis runtime log result has an invalid output_truncated flag")
    limits = envelope.get("limits")
    expected_limits = {
        "max_scan_lines_per_stream": GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM,
        "max_returned_lines_per_stream": GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM,
        "max_line_chars": GENESIS_RUNTIME_LOG_MAX_LINE_CHARS,
        "max_model_visible_chars": GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS,
    }
    if limits != expected_limits:
        raise ValueError("Genesis runtime log result has invalid limits")
    streams = envelope.get("streams")
    if not isinstance(streams, dict) or not streams or set(streams) - {"stdout", "stderr"}:
        raise ValueError("Genesis runtime log result has invalid streams")
    selected_streams = {"stdout", "stderr"} if stream == "both" else {stream}
    if set(streams) != selected_streams:
        raise ValueError("Genesis runtime log result streams do not match the selected stream")
    stream_order = [name for name in ("stdout", "stderr") if name in streams]
    if list(streams) != stream_order:
        raise ValueError("Genesis runtime log streams are not deterministically ordered")
    expected_stream_keys = {
        "requested_cursor",
        "actual_cursor",
        "scan_end",
        "next_cursor",
        "eof",
        "scan_truncated",
        "output_truncated",
        "returned_line_count",
        "lines",
    }
    stream_output_truncated = False
    for stream_name, stream_result in streams.items():
        if not isinstance(stream_result, dict) or set(stream_result) != expected_stream_keys:
            raise ValueError(f"Genesis runtime log {stream_name} result has invalid fields")
        lines = stream_result.get("lines")
        returned_line_count = stream_result.get("returned_line_count")
        if (
            isinstance(returned_line_count, bool)
            or not isinstance(returned_line_count, int)
            or returned_line_count < 0
            or not isinstance(lines, list)
            or len(lines) != returned_line_count
        ):
            raise ValueError(f"Genesis runtime log {stream_name} line count is invalid")
        if len(lines) > GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM:
            raise ValueError(f"Genesis runtime log {stream_name} exceeds the line limit")
        requested_cursor = stream_result.get("requested_cursor")
        actual_cursor = stream_result.get("actual_cursor")
        scan_end = stream_result.get("scan_end")
        next_cursor = stream_result.get("next_cursor")
        for field_name, value, minimum in (
            ("requested_cursor", requested_cursor, 1),
            ("actual_cursor", actual_cursor, 1),
            ("scan_end", scan_end, 0),
            ("next_cursor", next_cursor, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Genesis runtime log {stream_name} has invalid {field_name}")
        if actual_cursor > requested_cursor or scan_end < actual_cursor - 1 or next_cursor != scan_end + 1:
            raise ValueError(f"Genesis runtime log {stream_name} has inconsistent cursor relations")
        examined_count = max(0, scan_end - actual_cursor + 1)
        if examined_count > GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM:
            raise ValueError(f"Genesis runtime log {stream_name} exceeds the scan limit")
        for flag_name in ("eof", "scan_truncated", "output_truncated"):
            if not isinstance(stream_result.get(flag_name), bool):
                raise ValueError(f"Genesis runtime log {stream_name} has invalid {flag_name} flag")
        if stream_result["scan_truncated"] and (
            stream_result["eof"] or examined_count != GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM
        ):
            raise ValueError(f"Genesis runtime log {stream_name} has inconsistent scan truncation")
        previous_line_number = 0
        for line in lines:
            if not isinstance(line, dict) or set(line) != {"line_number", "text", "truncated"}:
                raise ValueError(f"Genesis runtime log {stream_name} contains an invalid line")
            line_number = line["line_number"]
            if (
                isinstance(line_number, bool)
                or not isinstance(line_number, int)
                or line_number <= previous_line_number
                or line_number < actual_cursor
                or line_number > scan_end
            ):
                raise ValueError(f"Genesis runtime log {stream_name} contains an invalid line number")
            if not isinstance(line["text"], str) or len(line["text"]) > GENESIS_RUNTIME_LOG_MAX_LINE_CHARS:
                raise ValueError(f"Genesis runtime log {stream_name} contains an overlong line")
            if not isinstance(line["truncated"], bool):
                raise ValueError(f"Genesis runtime log {stream_name} contains an invalid line truncation flag")
            previous_line_number = line_number
        stream_output_truncated = stream_output_truncated or stream_result["output_truncated"]
    if envelope["output_truncated"] != stream_output_truncated:
        raise ValueError("Genesis runtime log envelope truncation flag does not match its streams")
    compact = {
        "tool": "inspect_genesis_runtime_logs",
        "status": "ok",
        "tool_result_index": 9_999_999_999_999_999_999,
        "result": envelope,
    }
    if len(json.dumps(compact, sort_keys=True, separators=(",", ":"))) > GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS:
        raise ValueError("Genesis runtime log result exceeds the model-visible character limit")
    return result
def _diagnostic_route_payload(
    state: dict[str, Any],
    route: str,
    ready: bool,
    *,
    evidence_mode: str | None = None,
) -> dict[str, Any]:
    run_mode = state.get("run_mode")
    original_route = route
    effective_route = route
    non_actionable_reason = ""
    diagnostics_only = (
        run_mode == "post_mesh_processing_diagnostics"
        and state.get("diagnostics_reroute_and_export") is not True
    )
    supports_repair_reentry = (
        run_mode == "full_image"
        or state.get("diagnostics_reroute_and_export") is True
    )
    recommended_stage_skill_paths: tuple[str, ...]
    if ready or effective_route == SimDiagnosticRoute.ACCEPT.value or non_actionable_reason or diagnostics_only:
        recommended_stage_skill_paths = ()
    elif effective_route == SimDiagnosticRoute.SEGMENTATION.value:
        recommended_stage_skill_paths = (
            SEGMENTATION_REENTRY_STAGE_SKILL_PATHS if supports_repair_reentry else ()
        )
    elif effective_route == SimDiagnosticRoute.MATERIAL_INFERENCE.value:
        recommended_stage_skill_paths = (
            (STAGE_SKILL_PATHS["material_inference"], STAGE_SKILL_PATHS["mesh_processing"])
            if supports_repair_reentry
            else ()
        )
    elif effective_route == SimDiagnosticRoute.MESH_PROCESSING.value:
        recommended_stage_skill_paths = (
            (STAGE_SKILL_PATHS["mesh_processing"],)
            if supports_repair_reentry
            else ()
        )
    else:
        recommended_stage_skill_paths = ()
    payload = {
        "route": original_route,
        "original_route": original_route,
        "effective_route": effective_route,
        "ready": bool(ready),
        "degraded": False,
        "degradation_reason": "",
        "non_actionable_reason": non_actionable_reason,
        "recommended_stage_skill_paths": list(recommended_stage_skill_paths),
        "reentry_stage_skill_paths": list(recommended_stage_skill_paths),
        "force_rerun_stage_skill_paths": [],
    }
    if evidence_mode is not None:
        payload["evidence_mode"] = evidence_mode
    return payload


def _simulate_probe_validation_result(result: Mapping[str, Any]) -> dict[str, Any]:
    public_result = dict(result)
    live_result = result.get("result")
    if not isinstance(live_result, Mapping):
        return public_result
    action = live_result.get("action")
    if not isinstance(action, Mapping):
        return public_result
    action_payload = _load_externalized_live_payload(action)
    if not action_payload:
        return public_result
    details = action.get("details")
    if not isinstance(details, Mapping):
        details = action_payload.get("details")
    if not isinstance(details, Mapping):
        details = action_payload
    public_result["result"] = {**dict(live_result), **dict(details)}
    return public_result


def _require_executable_probe_aabb_box(compiled: Mapping[str, Any]) -> dict[str, Any]:
    payload = compiled.get("executable_probe_aabb_box")
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError("compiled probe target is missing executable_probe_aabb_box; recompile it")
    if payload.get("frame") != "env_local":
        raise GenesisVlmSchemaError("compiled probe target executable_probe_aabb_box.frame must be env_local")
    executable_box = _require_finite_vector_values(
        payload.get("box"),
        context="compiled_probe_target.executable_probe_aabb_box.box",
        length=6,
    )
    if any(executable_box[axis] >= executable_box[axis + 3] for axis in range(3)):
        raise GenesisVlmSchemaError(
            "compiled_probe_target.executable_probe_aabb_box.box min values must be strictly less than max values"
        )
    if payload.get("source_frame") != "local_mesh":
        raise GenesisVlmSchemaError("compiled probe target executable_probe_aabb_box.source_frame must be local_mesh")
    if payload.get("scene_encoded_mesh_transform") is not False:
        raise GenesisVlmSchemaError(
            "compiled probe target executable_probe_aabb_box.scene_encoded_mesh_transform must be false"
        )
    return {"frame": "env_local", "box": executable_box}
def _prepare_simulate_live_arguments(
    run_root: str,
    *,
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    episode: dict[str, Any],
    decision: dict[str, Any],
    decision_id: int,
    episode_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    arguments = dict(decision["arguments"])
    action = arguments.get("action")
    if not isinstance(action, Mapping):
        arguments["action"] = None
        return arguments, None, None
    compile_id = str(action["compiled_probe_target_id"])
    compiled = _compiled_probe_target_by_id(diagnostics, compile_id)
    target_id = str(compiled["target_id"])
    action_type = str(action["type"])
    controller_id = f"{compile_id}_box"
    force_schedule: Mapping[str, Any] | None = None
    if action_type == "compiled_probe":
        owner_episode_id = str(compiled.get("simulate_episode_id", "")).strip()
        current_episode_id = str(episode["episode_id"])
        if owner_episode_id != current_episode_id:
            raise GenesisVlmSchemaError(
                "compiled probe target belongs to another diagnostic episode; "
                f"requested={compile_id}, owner={owner_episode_id or 'none'}, current={current_episode_id}"
            )
        region_id = _region_id_for_episode(state, current_episode_id)
        if str(compiled.get("region_id", "")) != region_id:
            raise GenesisVlmSchemaError("compiled probe target crosses v2 region ownership")
        region = _planned_region(state, region_id)
        pair_policy = _pair_policy_for_region(diagnostics, region_id)
        current_locality_required = _mechanics_locality_required(
            pair_policy, str(compiled.get("mechanics_probe_mode", "none"))
        )
        if current_locality_required != bool(compiled.get("mechanics_locality_required", False)):
            raise GenesisVlmSchemaError("compiled mechanics locality no longer matches the immutable current pair policy")
        if current_locality_required and str(compiled.get("region_hint", "")) not in {"tip", "edge_band"}:
            raise GenesisVlmSchemaError("compiled local mechanics probe must retain tip or edge_band")
        if str(compiled.get("motion_axis", "+Y")) not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise GenesisVlmSchemaError("compiled probe has invalid motion_axis")
        current_count = sum(1 for item in diagnostics["diagnostic_probe_attempts"] if isinstance(item, Mapping) and item.get("region_id") == region_id)
        if current_count >= MAX_DIAGNOSTIC_COMPILED_PROBES_PER_EPISODE:
            arguments["action"] = None
            return arguments, {
                "tool": "simulate",
                "status": "rejected",
                "result": {
                    "mode": "bounded",
                    "steps_requested": int(arguments["steps"]),
                    "steps_completed": 0,
                    "paused_guaranteed": True,
                    "action": {
                        "requested": True,
                        "type": "compiled_probe",
                        "compiled_probe_target_id": compile_id,
                        "status": "rejected",
                    },
                },
                "error": {
                    "code": "compiled_probe_limit_exceeded",
                    "message": (
                        "active diagnostic region already has the maximum number of dispatched compiled_probe "
                        f"executions ({MAX_DIAGNOSTIC_COMPILED_PROBES_PER_EPISODE}); use plain simulate for "
                        "continued observation or submit a terminal recommendation"
                    ),
                    "details": {
                        "episode_id": current_episode_id,
                        "limit": MAX_DIAGNOSTIC_COMPILED_PROBES_PER_EPISODE,
                        "dispatched_attempt_count": current_count,
                        "requested_compile_id": compile_id,
                    },
                },
            }, None
        if region.get("qualifying_attempt_id"):
            raise GenesisVlmSchemaError("a qualifying probe candidate is already frozen for this region")
        reflections = [item for item in diagnostics["probe_target_reflections"] if isinstance(item, Mapping) and item.get("region_id") == region_id and item.get("compile_id") == compile_id]
        if len(reflections) != 1 or reflections[0].get("semantic_match") != "matched":
            raise GenesisVlmSchemaError("compiled probe dispatch requires exactly one current matched probe reflection")
        if len(compiled.get("semantic_group_part_ids", [compiled.get("selected_part_id")])) > 1:
            _verify_probe_semantic_group_lineage(_ensure_part_grounding_context(state), compiled)
        reflection = reflections[0]
        attachment = _current_runtime_attachment(state, diagnostics)
        owner_snapshot = reflection.get("owner_snapshot")
        expected_owner = {
            "attachment_id": attachment.get("attachment_id"),
            "agent_invocation_id": attachment.get("agent_invocation_id"),
            "active_revision_id": str(state.get("active_revision") or ""),
            "region_id": region_id,
            "episode_id": current_episode_id,
            "compile_id": compile_id,
        }
        if (
            reflection.get("active_revision_id") != expected_owner["active_revision_id"]
            or not isinstance(owner_snapshot, Mapping)
            or any(owner_snapshot.get(key) != value for key, value in expected_owner.items())
        ):
            raise GenesisVlmSchemaError(
                "compiled probe dispatch requires a current same-owner probe reflection"
            )
        separation = _record_anchor_probe_separation(state, region_id=region_id, probe_compile=compiled)
        if separation["validation_outcome"] == "rejected_revision_required":
            _save_diagnostics(
                run_root, state, event="diagnostic_anchor_probe_overlap_rejected",
                note="declared distinct pair has positive same-part overlap",
                detail={"anchor_probe_separation": separation}, provenance="runtime_probe_target_compiler",
            )
            raise GenesisVlmSchemaError("positive overlap is rejected before dispatch and consumes no attempt")
        if separation["relationship_to_probe"] == "overlap_exception":
            concern_id = _overlap_concern_id(region_id)
            concerns = [
                item for item in diagnostics["episode_concerns"]
                if isinstance(item, dict) and item.get("concern_id") == concern_id
            ]
            if (
                len(concerns) != 1
                or concerns[0].get("episode_id") != current_episode_id
                or concerns[0].get("state") not in {"episode_bound_pending_measurement", "pending_probe_evidence"}
            ):
                raise GenesisVlmSchemaError("overlap exception separation lacks its bound pending runtime concern")
            concerns[0].update({"state": "pending_probe_evidence", "separation_id": separation["separation_id"]})
        probe_box = separation["probe_aabb"]
        for complete in diagnostics["diagnostic_region_ledger"]:
            if not isinstance(complete, Mapping) or complete.get("status") != "coverage_complete" or complete.get("region_id") == region_id:
                continue
            prior = next((item for item in diagnostics["diagnostic_probe_attempts"] if isinstance(item, Mapping) and item.get("attempt_id") == complete.get("qualifying_attempt_id")), None)
            if prior and prior.get("selected_part_id") == compiled.get("selected_part_id") and isinstance(prior.get("probe_aabb"), list) and _aabb_iou(probe_box, prior["probe_aabb"]) >= 0.5:
                raise GenesisVlmSchemaError("same-part candidate duplicates completed coverage (IoU >= 0.5)")
        active_compile_id = str(episode.get("active_simulate_probe_compile_id", "")).strip()
        if active_compile_id:
            raise GenesisVlmSchemaError(
                "simulate compiled_probe requires no currently active probe; "
                f"requested={compile_id}, active={active_compile_id}"
            )
        internal_action = "box_ee_grasp_and_move"
        runtime_action_type = "compiled_probe"
        executable_aabb_box = _require_executable_probe_aabb_box(compiled)
        policy = pair_policy or {
            "pair_id": None, "policy_fingerprint": "ordinary-v1-default",
            "normalized_distance_scale": 0.5, "speed_m_s": 0.6,
        }
        motion_axis, effective_pair_execution_fingerprint, pending_direction_lock = _effective_pair_direction_policy(
            diagnostics, policy=pair_policy, compiled=compiled, region_id=region_id, compile_id=compile_id,
        )
        force_limited = _episode_force_limited_policy(episode)
        force_schedule = (
            force_limited_probe_schedule(
                aabb_box=executable_aabb_box["box"],
                distance_scale=float(policy["normalized_distance_scale"]),
                speed_m_s=float(policy["speed_m_s"]),
                timestep_s=DIAGNOSTIC_SCENE_TIMESTEP_S,
            )
            if force_limited
            else None
        )
        dispatch_token = uuid.uuid4().hex
        register_arguments: dict[str, Any] = {
            "action_id": f"{compile_id}_compiled_probe_{decision_id:04d}",
            "action_type": runtime_action_type,
            "probe_apply": {
                "probe_id": compile_id,
                "action": internal_action,
                "env_id": int(action.get("env_id", 0)),
                "duration_steps": int(force_schedule["load_steps"]) if force_schedule else int(arguments["steps"]),
                "measurement": {
                    "dispatch_token": dispatch_token,
                    "anchor_id": str(separation["region_id"]),
                    **(
                        {
                            "schedule": {
                                "load_steps": int(force_schedule["load_steps"]),
                                "recovery_steps": int(force_schedule["recovery_steps"]),
                            }
                        }
                        if force_schedule
                        else {}
                    ),
                },
                "diagnostics_only": True,
                "controllers": [
                    {
                        "controller_id": controller_id,
                        "aabb_box": executable_aabb_box,
                        "distance_scale": float(policy["normalized_distance_scale"]),
                        "speed": float(policy["speed_m_s"]),
                        "motion_axis": motion_axis,
                    }
                ],
            },
            "metadata": {
                "decision_id": decision_id,
                "target_id": target_id,
                "compile_id": compile_id,
                "selected_part_id": compiled.get("selected_part_id"),
                "semantic_group_part_ids": list(compiled.get("semantic_group_part_ids", [compiled.get("selected_part_id")])),
                "semantic_group_part_grounding": list(compiled.get("semantic_group_part_grounding", [])),
                **episode_metadata,
            },
        }
    elif action_type == "release_probe":
        active_compile_id = str(episode.get("active_simulate_probe_compile_id", "")).strip()
        if active_compile_id != compile_id:
            raise GenesisVlmSchemaError(
                "simulate release_probe requires an active compiled probe target from a prior successful "
                f"simulate compiled_probe call; requested={compile_id}, active={active_compile_id or 'none'}"
            )
        internal_action = "probe_release"
        runtime_action_type = "release_probe"
        active_dispatch_token = str(episode.get("active_simulate_probe_dispatch_token", "")).strip()
        attempts = [
            item for item in diagnostics["diagnostic_probe_attempts"]
            if isinstance(item, Mapping)
            and item.get("compile_id") == compile_id
            and item.get("episode_id") == episode["episode_id"]
            and item.get("dispatch_token") == active_dispatch_token
        ]
        if not active_dispatch_token or len(attempts) != 1:
            raise GenesisVlmSchemaError("release_probe requires the current compiled-probe dispatch token")
        attempt = attempts[0]
        register_arguments = {
            "action_id": f"{compile_id}_release_probe_{decision_id:04d}",
            "action_type": runtime_action_type,
            "probe_release": {
                "probe_id": compile_id,
                "action": internal_action,
                "duration_steps": int(arguments["steps"]),
                "diagnostics_only": True,
                "controllers": [{"controller_id": controller_id}],
            },
            "metadata": {
                "decision_id": decision_id,
                "target_id": target_id,
                "compile_id": compile_id,
                **episode_metadata,
            },
        }
    else:
        raise GenesisVlmSchemaError(f"unsupported simulate action type: {action_type}")
    register_handler = active_live_tool_handlers().get("register_probe_action")
    if register_handler is None:
        return arguments, {
            "tool": "simulate",
            "status": STATUS_UNSUPPORTED,
            "result": {},
            "error": {
                "code": "live_handler_not_configured",
                "message": "No Genesis registered-action handler is attached to this Codex diagnostic stage run.",
            },
        }, None
    registration_result = register_handler(
        run_root=run_root,
        arguments=register_arguments,
        decision=decision,
    )
    if str(registration_result.get("status", "")) not in {"ok", "applied"}:
        return arguments, {
            "tool": "simulate",
            "status": str(registration_result.get("status", "error")) or "error",
            "result": {"registration_result": _redact_internal_probe_target(registration_result)},
            "error": registration_result.get("error"),
        }, None
    if action_type == "compiled_probe":
        # The registered action is the durable dispatch boundary.  Nothing before
        # this point has reached the simulator, so a rejected registration must
        # not consume an attempt slot.  Conversely, write this record before the
        # simulate handler is called: a process crash or handler exception still
        # leaves an owned, auditable attempt behind.
        attachment = _current_runtime_attachment(state, diagnostics)
        if pending_direction_lock is not None:
            diagnostics["pair_direction_locks"].append(pending_direction_lock)
            _save_diagnostics(
                run_root, state, event="diagnostic_pair_direction_locked",
                note=f"locked {pending_direction_lock['pair_id']} to {pending_direction_lock['axis']}",
                detail={"pair_direction_lock": pending_direction_lock}, provenance="runtime_event",
            )
        attempt = {
            "attempt_id": f"attempt_{len(diagnostics['diagnostic_probe_attempts']) + 1:04d}",
            "attempt_index": len(diagnostics["diagnostic_probe_attempts"]),
            "region_id": region_id,
            "episode_id": current_episode_id,
            "compile_id": compile_id,
            "selected_part_id": compiled.get("selected_part_id"),
            "semantic_group_part_ids": list(compiled.get("semantic_group_part_ids", [compiled.get("selected_part_id")])),
            "semantic_group_part_grounding": list(compiled.get("semantic_group_part_grounding", [])),
            "probe_aabb": probe_box,
            "decision_id": decision_id,
            "registered_action_id": str(register_arguments["action_id"]),
            "attachment_id": str(attachment["attachment_id"]),
            "agent_invocation_id": str(attachment["agent_invocation_id"]),
            "active_revision_id": str(state.get("active_revision") or ""),
            "registration_result": _redact_internal_probe_target(registration_result),
            "dispatch_status": "registered_dispatched",
            "dispatch_token": dispatch_token,
            "pair_id": policy["pair_id"],
            "pair_policy_fingerprint": policy["policy_fingerprint"],
            "motion_axis": motion_axis,
            "effective_pair_execution_fingerprint": effective_pair_execution_fingerprint,
            **(
                {
                    "force_limited_schedule": force_schedule,
                    "force_limited_policy_id": DIAGNOSTIC_FORCE_LIMITED_POLICY_ID,
                    "force_limited_policy_hash": diagnostic_force_limited_policy_hash(),
                    "force_limited_calibration_provenance": diagnostic_force_limited_calibration_provenance(),
                }
                if force_schedule
                else {}
            ),
        }
        diagnostics["diagnostic_probe_attempts"].append(attempt)
        _save_diagnostics(
            run_root,
            state,
            event="diagnostic_probe_attempt_dispatched",
            note=f"registered action {attempt['registered_action_id']} before simulate handler",
            detail={"attempt": attempt},
            provenance="runtime_event",
        )
    if action_type == "compiled_probe" and force_schedule:
        arguments["steps"] = int(force_schedule["load_steps"])
    elif action_type == "release_probe" and _episode_force_limited_policy(episode):
        force_schedule = attempt.get("force_limited_schedule")
        if not isinstance(force_schedule, Mapping):
            raise GenesisVlmSchemaError("release_probe lacks its frozen force-limited schedule")
        arguments["steps"] = int(force_schedule["recovery_steps"])
    arguments["action"] = {
        "type": runtime_action_type,
        "action_id": str(register_arguments["action_id"]),
        **({"force_limited_schedule": dict(force_schedule)} if action_type == "compiled_probe" and force_schedule else {}),
    }
    application = {
        "application_index": len(diagnostics["probe_target_applications"]),
        "tool": "simulate",
        "target_id": target_id,
        "compile_id": compile_id,
        "decision_id": decision_id,
        "internal_tool": "sim.resume",
        "internal_action": internal_action,
        "registered_action_id": str(register_arguments["action_id"]),
        "controller_id": controller_id,
        "duration_steps": int(arguments["steps"]),
        "requested_duration_steps": int(arguments["steps"]),
        "registration_result": _redact_internal_probe_target(registration_result),
        **episode_metadata,
    }
    if action_type == "compiled_probe":
        application["attempt_id"] = attempt["attempt_id"]
        application["region_id"] = region_id
        application["selected_part_id"] = compiled.get("selected_part_id")
        application["semantic_group_part_ids"] = list(compiled.get("semantic_group_part_ids", [compiled.get("selected_part_id")]))
        application["semantic_group_part_grounding"] = list(compiled.get("semantic_group_part_grounding", []))
        application["distance_scale"] = float(policy["normalized_distance_scale"])
        application["dispatch_token"] = dispatch_token
        application["pair_id"] = policy["pair_id"]
        application["pair_policy_fingerprint"] = policy["policy_fingerprint"]
        application["motion_axis"] = motion_axis
        application["effective_pair_execution_fingerprint"] = effective_pair_execution_fingerprint
        application["env_id"] = int(action.get("env_id", 0))
        application["executable_probe_aabb_box"] = executable_aabb_box
        application["transformed_scene_fit_probe_box"] = compiled["genesis_aabb_box"]
        if force_schedule:
            application["force_limited_schedule"] = force_schedule
            application["force_limited_policy_id"] = DIAGNOSTIC_FORCE_LIMITED_POLICY_ID
            application["force_limited_policy_hash"] = diagnostic_force_limited_policy_hash()
            application["force_limited_calibration_provenance"] = diagnostic_force_limited_calibration_provenance()
    else:
        application["dispatch_token"] = str(attempt["dispatch_token"])
        application["region_id"] = attempt.get("region_id")
        application["pair_id"] = attempt.get("pair_id")
        application["pair_policy_fingerprint"] = attempt.get("pair_policy_fingerprint")
        if force_schedule:
            application["force_limited_schedule"] = dict(force_schedule)
            application["force_limited_policy_id"] = DIAGNOSTIC_FORCE_LIMITED_POLICY_ID
            application["force_limited_policy_hash"] = diagnostic_force_limited_policy_hash()
            application["force_limited_calibration_provenance"] = diagnostic_force_limited_calibration_provenance()
    return arguments, None, {
        "compiled": compiled,
        "target_id": target_id,
        "compile_id": compile_id,
        "action_type": action_type,
        "application": application,
        "force_limited": bool(force_schedule),
        "validate_grabbed_vertices": action_type == "compiled_probe",
    }


def _persist_failed_runtime_log_inspection(
    run_root: str,
    *,
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    episode: dict[str, Any],
    decision: dict[str, Any],
    expected_observation: str,
    error: BaseException,
) -> None:
    tool_result_index = len(diagnostics["tool_results"])
    episode_metadata = _episode_metadata(episode)
    actual_observation = f"{type(error).__name__}: {error}"
    record = {
        "tool": "inspect_genesis_runtime_logs",
        "tool_result_index": tool_result_index,
        "decision": decision,
        "result": {
            "tool": "inspect_genesis_runtime_logs",
            "status": "error",
            "error": {
                "code": "genesis_runtime_log_inspection_failed",
                "message": actual_observation,
            },
        },
        "expected_observation": expected_observation,
        "actual_observation": actual_observation,
        **episode_metadata,
    }
    diagnostics["tool_results"].append(record)
    diagnostics["expected_observations"].append(expected_observation)
    diagnostics["actual_observations"].append(actual_observation)
    episode.setdefault("tool_result_indices", []).append(tool_result_index)
    episode.setdefault("expected_observations", []).append(expected_observation)
    episode.setdefault("actual_observations", []).append(actual_observation)
    _write_episode_observations(episode, diagnostics)
    _save_diagnostics(
        run_root,
        state,
        event="diagnostic_live_tool_failed",
        note=actual_observation,
        detail=record,
        provenance="live_tool_result",
    )
    setattr(error, "diagnostic_tool_result_index", tool_result_index)


def _persist_compiled_probe_dispatch_failure(
    run_root: str,
    *,
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    episode: dict[str, Any],
    decision: dict[str, Any],
    expected_observation: str,
    episode_metadata: dict[str, Any],
    simulate_probe: Mapping[str, Any],
    error: BaseException,
    stage: str,
) -> None:
    """Close the durable attempt linkage before re-raising a dispatch failure."""
    application_seed = simulate_probe.get("application")
    attempt_id = str(application_seed.get("attempt_id", "")) if isinstance(application_seed, Mapping) else ""
    attempt = next(
        (
            item
            for item in diagnostics["diagnostic_probe_attempts"]
            if isinstance(item, dict) and item.get("attempt_id") == attempt_id
        ),
        None,
    )
    if attempt is None:
        raise GenesisVlmSchemaError(
            "compiled probe dispatch failed without its durable registered-dispatch attempt"
        ) from error
    tool_result_index = len(diagnostics["tool_results"])
    application_index = len(diagnostics["probe_target_applications"])
    failure = {
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
        "registered_action_id": attempt["registered_action_id"],
        "compile_id": attempt["compile_id"],
        "decision_id": attempt["decision_id"],
        "tool_result_index": tool_result_index,
        "application_index": application_index,
    }
    application = {
        **dict(application_seed or {}),
        "application_index": application_index,
        "tool_result_index": tool_result_index,
        "result_status": "error",
        "active_revision_id": str(state.get("active_revision") or ""),
        "failure": failure,
    }
    result = {
        "tool": "simulate",
        "status": "error",
        "result": {},
        "error": {
            "code": "compiled_probe_dispatch_exception",
            "message": f"{type(error).__name__}: {error}",
            "stage": stage,
        },
    }
    record = {
        "tool": "simulate",
        "tool_result_index": tool_result_index,
        "decision": decision,
        "result": result,
        "expected_observation": expected_observation,
        "actual_observation": f"{type(error).__name__}: {error}",
        "compiled_probe_target": _compiled_probe_target_summary(simulate_probe["compiled"]),
        "dispatch_failure": failure,
        **episode_metadata,
    }
    diagnostics["probe_target_applications"].append(application)
    diagnostics["tool_results"].append(record)
    diagnostics["expected_observations"].append(expected_observation)
    diagnostics["actual_observations"].append(record["actual_observation"])
    episode.setdefault("tool_result_indices", []).append(tool_result_index)
    episode.setdefault("expected_observations", []).append(expected_observation)
    episode.setdefault("actual_observations", []).append(record["actual_observation"])
    attempt.update(
        {
            "dispatch_status": "handler_raised" if stage == "simulate_handler" else "post_handler_raised",
            "tool_result_index": tool_result_index,
            "application_index": application_index,
            "result_status": "error",
            "qualifying": False,
            "failure": failure,
        }
    )
    _write_episode_observations(episode, diagnostics)
    _save_diagnostics(
        run_root,
        state,
        event="diagnostic_probe_attempt_dispatch_failed",
        note=f"compiled probe dispatch raised {type(error).__name__} during {stage}",
        detail={"attempt": attempt, "application": application, "tool_result": record},
        provenance="live_tool_result",
    )


def _persist_completed_probe_measurement(
    diagnostics: dict[str, Any], *, result: dict[str, Any], simulate_probe: Mapping[str, Any], tool_result_index: int
) -> None:
    payload = result.get("result")
    if not isinstance(payload, dict):
        return
    completed = payload.pop("_completed_probe_measurement", None)
    if simulate_probe.get("action_type") != "release_probe":
        if completed is not None:
            raise GenesisVlmSchemaError("completed probe measurement is valid only for release_probe")
        return
    if completed is None:
        return
    if not isinstance(completed, CompletedProbeMeasurement):
        raise GenesisVlmSchemaError("Genesis completed probe measurement bypassed typed evidence admission")
    application = simulate_probe.get("application")
    if not isinstance(application, Mapping):
        raise GenesisVlmSchemaError("completed probe measurement lacks dispatched application")
    expected_token = str(application.get("dispatch_token", "")).strip()
    if completed.dispatch_token != expected_token:
        raise GenesisVlmSchemaError("Genesis completed probe dispatch token does not match its application")
    controller_effectiveness = simulate_probe.get("completed_controller_effectiveness")
    if simulate_probe.get("force_limited"):
        if not isinstance(controller_effectiveness, Mapping):
            raise GenesisVlmSchemaError("completed force-limited probe has no validated controller telemetry")
        schedule = controller_effectiveness.get("schedule")
        if not isinstance(schedule, Mapping):
            raise GenesisVlmSchemaError("completed force-limited probe has an invalid frozen recovery schedule")
        post_release_steps = int(schedule.get("recovery_steps", 0))
        if post_release_steps <= 0:
            raise GenesisVlmSchemaError("completed force-limited probe has an invalid frozen recovery schedule")
        if completed.post_release.simulation_step - completed.under_load.simulation_step != post_release_steps:
            raise GenesisVlmSchemaError("completed force-limited probe endpoints do not match the frozen recovery schedule")
        if not all(
            isinstance(value, Mapping)
            for value in (
                completed.schedule,
                completed.controller_policy,
                completed.controller_telemetry,
                completed.force_summary,
            )
        ):
            raise GenesisVlmSchemaError("completed force-limited probe omitted typed server controller evidence")
    record = {
        "dispatch_token": completed.dispatch_token,
        "identity": dict(completed.identity),
        "under_load": completed.under_load.as_dict(),
        "post_release": completed.post_release.as_dict(),
        "completion_tool_result_index": tool_result_index,
        "compile_id": application.get("compile_id"), "episode_id": application.get("episode_id"),
        "region_id": application.get("region_id"), "pair_id": application.get("pair_id"),
        "pair_policy_fingerprint": application.get("pair_policy_fingerprint"),
        **({"controller_effectiveness": dict(controller_effectiveness)} if isinstance(controller_effectiveness, Mapping) else {}),
        **(
            {
                "server_completed_controller_evidence": {
                    "schedule": dict(completed.schedule),
                    "controller_policy": dict(completed.controller_policy),
                    "controller_telemetry": dict(completed.controller_telemetry),
                    "force_summary": dict(completed.force_summary),
                },
                "force_limited_calibration_provenance": dict(application["force_limited_calibration_provenance"]),
            }
            if simulate_probe.get("force_limited")
            else {}
        ),
    }
    diagnostics["probe_measurements"].append(record)


def _record_live_tool(
    run_root: str,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    expected_observation: str,
    rationale: str,
    observe_after: dict[str, bool] | None = None,
) -> dict[str, Any]:
    state = _require_state(run_root)
    require_v2_session_state(state)
    diagnostics = _diagnostics(state)
    probe_max_duration_steps = int(diagnostics["probe_max_duration_steps"])
    if probe_max_duration_steps < DEFAULT_DIAGNOSTIC_SIMULATE_STEPS:
        raise GenesisVlmSchemaError(
            f"diagnostic probe_max_duration_steps must be >= {DEFAULT_DIAGNOSTIC_SIMULATE_STEPS}"
        )
    action_limits = GenesisVlmActionLimits(
        max_resume_steps=probe_max_duration_steps,
        max_probe_vertices=int(diagnostics["probe_max_vertices"]),
        max_probe_distance_m=float(diagnostics["probe_max_distance_m"]),
        max_probe_speed_m_s=float(diagnostics["probe_max_speed_m_s"]),
        max_probe_duration_steps=probe_max_duration_steps,
    )
    decision_id = len(diagnostics["tool_results"]) + 1
    decision = validate_vlm_action_decision(
        {
            "schema_version": VLM_ACTION_SCHEMA_VERSION,
            "decision_id": decision_id,
            "tool": tool_name,
            "arguments": arguments,
            "observe_after": observe_after
            or {
                "contact": False,
                "deformation": False,
                "material": False,
                "frame": tool_name == "pause_and_observe" or tool_name in LIVE_VISUAL_TOOL_NAMES,
            },
            "expected_observation": expected_observation,
            "rationale": rationale,
        },
        limits=action_limits,
    )
    episode = _active_episode_record(state)
    if tool_name == "simulate" and not bool(episode.get("simulation_reset_completed", False)):
        raise GenesisVlmSchemaError("simulate requires a prior successful simulation_reset in the active episode")
    last_tool_name = _latest_episode_tool_name(diagnostics, episode)
    episode_metadata = _episode_metadata(episode)
    turn_based_violation = _turn_based_live_tool_order_violation(
        tool_name=tool_name,
        previous_tool_name=last_tool_name,
        decision_id=decision_id,
        episode_metadata=episode_metadata,
    )
    if turn_based_violation is not None:
        violation_index = len(diagnostics["turn_based_violations"])
        diagnostics["turn_based_violations"].append(turn_based_violation)
        episode.setdefault("turn_based_violation_indices", []).append(violation_index)
        _save_diagnostics(
            run_root,
            state,
            event="diagnostic_turn_based_violation",
            note=turn_based_violation["message"],
            detail=turn_based_violation,
            provenance="runtime_event",
        )
    handler_arguments = dict(decision["arguments"])
    prepared_result: dict[str, Any] | None = None
    simulate_probe: dict[str, Any] | None = None
    if tool_name == "simulate":
        handler_arguments, prepared_result, simulate_probe = _prepare_simulate_live_arguments(
            run_root,
            state=state,
            diagnostics=diagnostics,
            episode=episode,
            decision=decision,
            decision_id=decision_id,
            episode_metadata=episode_metadata,
        )
    try:
        handler = active_live_tool_handlers().get(tool_name)
        if prepared_result is not None:
            result = prepared_result
        elif handler is None:
            result = {
                "tool": tool_name,
                "status": STATUS_UNSUPPORTED,
                "result": {},
                "error": {
                    "code": "live_handler_not_configured",
                    "message": "No Genesis live handler is attached to this Codex diagnostic stage run.",
                },
            }
        else:
            result = handler(run_root=run_root, arguments=handler_arguments, decision=decision)
        if tool_name == "simulate" and isinstance(simulate_probe, dict) and simulate_probe.get("force_limited"):
            live_payload = result.get("result") if isinstance(result.get("result"), Mapping) else {}
            action_payload = live_payload.get("action") if isinstance(live_payload, Mapping) else None
            if simulate_probe.get("action_type") == "compiled_probe":
                controller_state = controller_state_from_action(action_payload)
                simulate_probe["dispatch_controller_policy"] = validate_controller_policy(controller_state)
            elif simulate_probe.get("action_type") == "release_probe":
                controller_state = controller_state_from_action(action_payload)
                schedule = simulate_probe["application"].get("force_limited_schedule")
                if not isinstance(schedule, Mapping):
                    raise GenesisVlmSchemaError("release_probe lacks frozen force-limited schedule")
                completed = live_payload.get("_completed_probe_measurement")
                if not isinstance(completed, CompletedProbeMeasurement):
                    raise GenesisVlmSchemaError("release_probe omitted the typed completed force-limited measurement")
                simulate_probe["completed_controller_effectiveness"] = completed_controller_telemetry(
                    controller_state=controller_state,
                    schedule=schedule,
                    completed_measurement=completed,
                )
        if tool_name == "simulate" and isinstance(simulate_probe, Mapping):
            _persist_completed_probe_measurement(
                diagnostics,
                result=result,
                simulate_probe=simulate_probe,
                tool_result_index=len(diagnostics["tool_results"]),
            )
        result = _materialize_pause_and_observe_visual_frame(state, decision_id=decision_id, result=result)
        result = _materialize_live_visual_sequence(state, decision_id=decision_id, result=result)
        result = _externalize_large_live_result(state, decision_id=decision_id, tool_name=tool_name, result=result)
    except BaseException as error:
        if tool_name == "inspect_genesis_runtime_logs":
            _persist_failed_runtime_log_inspection(
                run_root,
                state=state,
                diagnostics=diagnostics,
                episode=episode,
                decision=decision,
                expected_observation=expected_observation,
                error=error,
            )
        if tool_name == "simulate" and isinstance(simulate_probe, Mapping) and simulate_probe.get("action_type") == "compiled_probe":
            _persist_compiled_probe_dispatch_failure(
                run_root,
                state=state,
                diagnostics=diagnostics,
                episode=episode,
                decision=decision,
                expected_observation=expected_observation,
                episode_metadata=episode_metadata,
                simulate_probe=simulate_probe,
                error=error,
                stage="simulate_handler",
            )
        raise
    actual_observation = _compact_actual_observation(result)
    tool_result_index = len(diagnostics["tool_results"])
    probe_validation: dict[str, Any] | None = None
    if simulate_probe is not None and bool(simulate_probe.get("validate_grabbed_vertices")):
        try:
            context = _ensure_part_grounding_context(state)
            probe_validation = validate_grabbed_vertices_against_part(
                context,
                simulate_probe["compiled"],
                _simulate_probe_validation_result(result),
            )
            probe_validation = {
                **probe_validation,
                "target_id": simulate_probe["target_id"],
                "compile_id": simulate_probe["compile_id"],
                "semantic_group_part_grounding": list(
                    simulate_probe["compiled"].get("semantic_group_part_grounding", [])
                ),
                "decision_id": decision_id,
                "validation_index": len(diagnostics["probe_target_validations"]),
                "tool_result_index": tool_result_index,
                **episode_metadata,
            }
            actual_observation = (
                f"{actual_observation} Probe target validation status={probe_validation['status']}; "
                f"grabbed_selected_part_fraction={probe_validation.get('grabbed_selected_part_fraction', 0.0):.3f}."
            )
        except BaseException as error:
            if simulate_probe.get("action_type") == "compiled_probe":
                _persist_compiled_probe_dispatch_failure(
                    run_root,
                    state=state,
                    diagnostics=diagnostics,
                    episode=episode,
                    decision=decision,
                    expected_observation=expected_observation,
                    episode_metadata=episode_metadata,
                    simulate_probe=simulate_probe,
                    error=error,
                    stage="probe_validation",
                )
            raise
    record = {
        "tool": tool_name,
        "tool_result_index": tool_result_index,
        "decision": decision,
        "result": result,
        "expected_observation": expected_observation,
        "actual_observation": actual_observation,
        **episode_metadata,
    }
    if turn_based_violation is not None:
        record["turn_based_violation"] = turn_based_violation
    if simulate_probe is not None:
        record["compiled_probe_target"] = _compiled_probe_target_summary(simulate_probe["compiled"])
    if probe_validation is not None:
        record["probe_target_validation"] = _probe_target_validation_summary(probe_validation)
        diagnostics["probe_target_validations"].append(probe_validation)
    visual_evidence = _visual_evidence_from_live_result(
        tool_name=tool_name,
        result=result,
    )
    if visual_evidence is not None:
        visual_evidence = {**visual_evidence, **episode_metadata}
        record["visual_evidence"] = visual_evidence
        visual_index = len(diagnostics["visual_evidence"])
        diagnostics["visual_evidence"].append(visual_evidence)
        episode.setdefault("visual_evidence_indices", []).append(visual_index)
    if simulate_probe is not None:
        live_result = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        runtime_step_adaptation = (
            live_result.get("runtime_step_adaptation") if isinstance(live_result, Mapping) else None
        )
        application = {
            **simulate_probe["application"],
            "tool_result_index": tool_result_index,
            "result_status": str(result.get("status", "")),
            "active_revision_id": str(state.get("active_revision") or ""),
        }
        if simulate_probe.get("action_type") == "compiled_probe" and isinstance(runtime_step_adaptation, Mapping):
            application["effective_resume_steps"] = int(runtime_step_adaptation["effective_steps"])
            application["adaptive_resume_applied"] = bool(runtime_step_adaptation.get("applied", False))
            application["estimated_motion_steps"] = runtime_step_adaptation.get("estimated_motion_steps")
            application["adaptive_resume_source"] = str(runtime_step_adaptation.get("source", ""))
        if simulate_probe.get("action_type") == "compiled_probe" and simulate_probe.get("force_limited"):
            application["dispatch_controller_policy"] = dict(simulate_probe["dispatch_controller_policy"])
        elif simulate_probe.get("action_type") == "release_probe" and simulate_probe.get("force_limited"):
            application["completed_controller_effectiveness"] = dict(simulate_probe["completed_controller_effectiveness"])
        if probe_validation is not None:
            application["probe_target_validation"] = _probe_target_validation_summary(probe_validation)
        diagnostics["probe_target_applications"].append(application)
        if simulate_probe.get("action_type") == "compiled_probe":
            attempt_id = str(application.get("attempt_id", ""))
            attempt = next((item for item in diagnostics["diagnostic_probe_attempts"] if isinstance(item, dict) and item.get("attempt_id") == attempt_id), None)
            if attempt is None:
                raise GenesisVlmSchemaError("compiled probe application lacks its pre-dispatch attempt")
            attempt.update({"dispatch_status": "completed", "tool_result_index": tool_result_index,
                            "result_status": str(result.get("status", "")), "application_index": application["application_index"]})
            steps = result.get("result", {}).get("steps_completed") if isinstance(result.get("result"), Mapping) else None
            hard_error = isinstance(result.get("error"), Mapping) and bool(result["error"])
            fraction = float(probe_validation.get("grabbed_selected_part_fraction", 0.0)) if probe_validation else 0.0
            payload = result.get("result", {}) if isinstance(result.get("result"), Mapping) else {}
            full_probe_window = _completed_force_limited_probe_window(
                application=application,
                payload=payload,
            ) if simulate_probe.get("force_limited") else _completed_default_probe_window(
                requested_steps=application.get("requested_duration_steps"),
                completed_steps=steps,
                runtime_step_adaptation=runtime_step_adaptation,
            )
            qualifies = bool(_live_tool_succeeded(result) and not hard_error and full_probe_window and fraction >= 0.50)
            attempt["qualifying"] = qualifies
            attempt["grabbed_selected_part_fraction"] = fraction
            failure_reasons: list[str] = []
            if not _live_tool_succeeded(result):
                failure_reasons.append("simulate_handler_returned_non_success")
            if hard_error:
                failure_reasons.append("simulate_result_contains_error")
            if not full_probe_window:
                failure_reasons.append("default_probe_window_not_completed")
            if probe_validation is None or str(probe_validation.get("status", "")) != "ok":
                failure_reasons.append("probe_target_purity_not_ok")
            if failure_reasons:
                attempt["failure"] = {
                    "stage": "simulate_result",
                    "reasons": failure_reasons,
                    "registered_action_id": attempt["registered_action_id"],
                    "tool_result_index": tool_result_index,
                    "application_index": application["application_index"],
                }
            if qualifies:
                region = _planned_region(state, str(attempt["region_id"]))
                if region.get("qualifying_attempt_id"):
                    raise GenesisVlmSchemaError("a second qualifying candidate is forbidden for the region")
                region["qualifying_attempt_id"] = attempt_id
                region["qualifying_compile_id"] = attempt["compile_id"]
        if _live_tool_succeeded(result):
            if simulate_probe.get("action_type") == "compiled_probe":
                episode["active_simulate_probe_compile_id"] = str(simulate_probe["compile_id"])
                episode["active_simulate_probe_controller_id"] = str(application["controller_id"])
                episode["active_simulate_probe_dispatch_token"] = str(application["dispatch_token"])
            elif simulate_probe.get("action_type") == "release_probe":
                episode.pop("active_simulate_probe_compile_id", None)
                episode.pop("active_simulate_probe_controller_id", None)
                episode.pop("active_simulate_probe_dispatch_token", None)
    if tool_name == "simulation_reset" and _live_tool_succeeded(result):
        episode["simulation_reset_completed"] = True
        episode["simulation_reset_decision_id"] = decision_id
    diagnostics["tool_results"].append(record)
    if tool_name == "query_live_geometry_context" and isinstance(result, dict):
        live_geometry_context = result.get("result")
        if isinstance(live_geometry_context, dict):
            measurement = {
                "measurement_index": len(diagnostics["live_geometry_context_measurements"]),
                "tool_result_index": tool_result_index,
                "env_id": int(arguments.get("env_id", 0)),
                "obj_id": int(arguments.get("obj_id", 0)),
                "geometry_context": live_geometry_context,
            }
            diagnostics["live_geometry_context_measurements"].append(measurement)
            episode["live_geometry_context_cache"] = measurement
            episode.setdefault("live_geometry_context_measurement_indices", []).append(
                len(diagnostics["live_geometry_context_measurements"]) - 1
            )
            record["live_geometry_context"] = measurement
    diagnostics["expected_observations"].append(expected_observation)
    diagnostics["actual_observations"].append(actual_observation)
    episode.setdefault("tool_result_indices", []).append(tool_result_index)
    episode.setdefault("expected_observations", []).append(expected_observation)
    episode.setdefault("actual_observations", []).append(actual_observation)
    _write_episode_observations(episode, diagnostics)
    _save_diagnostics(
        run_root,
        state,
        event="diagnostic_live_tool_recorded",
        note=f"{tool_name} returned {result.get('status', 'unknown')}",
        detail=record,
        provenance="live_tool_result",
    )
    return record
def _diagnostic_action_limits(diagnostics: dict[str, Any]) -> GenesisVlmActionLimits:
    probe_max_duration_steps = int(diagnostics["probe_max_duration_steps"])
    if probe_max_duration_steps < DEFAULT_DIAGNOSTIC_SIMULATE_STEPS:
        raise GenesisVlmSchemaError(
            f"diagnostic probe_max_duration_steps must be >= {DEFAULT_DIAGNOSTIC_SIMULATE_STEPS}"
        )
    return GenesisVlmActionLimits(
        max_resume_steps=probe_max_duration_steps,
        max_probe_vertices=int(diagnostics["probe_max_vertices"]),
        max_probe_distance_m=float(diagnostics["probe_max_distance_m"]),
        max_probe_speed_m_s=float(diagnostics["probe_max_speed_m_s"]),
        max_probe_duration_steps=probe_max_duration_steps,
    )
def _latest_episode_tool_name(diagnostics: dict[str, Any], episode: dict[str, Any]) -> str | None:
    tool_results = diagnostics.get("tool_results", [])
    if not isinstance(tool_results, list):
        return None
    for raw_index in reversed(episode.get("tool_result_indices", [])):
        if not isinstance(raw_index, int) or raw_index < 0 or raw_index >= len(tool_results):
            continue
        record = tool_results[raw_index]
        if isinstance(record, dict):
            tool_name = str(record.get("tool", "")).strip()
            if tool_name and tool_name != "query_live_geometry_context":
                return tool_name
    return None
def _turn_based_live_tool_order_violation(
    *,
    tool_name: str,
    previous_tool_name: str | None,
    decision_id: int,
    episode_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    del tool_name, previous_tool_name, decision_id, episode_metadata
    return None
def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple):
        return [text for item in value if (text := str(item).strip())]
    text = str(value).strip()
    return [text] if text else []
def _normalize_terminal_recommendation_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GenesisVlmSchemaError(f"recommendation must be a JSON object, got {type(payload).__name__}")
    return dict(payload)
def _require_nonempty_runtime_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenesisVlmSchemaError(f"{context} must be a non-empty string")
    return value.strip()
def record_diagnostic_session_plan(
    run_root: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Record the immutable v2 plan. Submit one complete exact-shape payload, not progressive schema discovery: top level is exactly schema_version, session_intent, candidate_region_count, regions, omitted_region_summary, paired_comparisons; when candidate_region_count <= 4, omitted_region_summary is exactly []. Each region is exactly region_id, name, semantic_region, physical_hypothesis, desired_interaction, episode_intent, termination_condition, risk_rank, selection_rationale, setup_anchor. setup_anchor is exactly anchor_type, anchor_region, uncertainty (low|medium|high), relationship_to_probe (distinct|overlap_exception), relationship_rationale, concerns; anchor_type is support_contact|grip_root|joint_hinge and each globally unique concern has exactly concern_id, concern_type, summary. `paired_comparisons` is required: use [] only when no pair of selected regions needs structural/material-response comparison; each item is exactly {pair_id, region_ids:[two region IDs]}. Before submit, inspect every unordered pair A,B: if A.semantic_region and B.setup_anchor.anchor_region are one semantic part/material group, and B.semantic_region and A.setup_anchor.anchor_region are another, A,B MUST be one pair. When source evidence and current canonical semantics identify two selected groups with a relative structural/material-response contrast, reserve those reciprocal candidates as selected slots before ranking or filling independent risks, then construct and declare their pair; do not use unrelated third anchors or independent episodes to avoid the comparison. Extra unrelated regions and independent hypotheses do not exempt; a missing reciprocal pair must not be submitted. Each semantic comparison pair's two `physical_hypothesis` strings must state the ordered greater/lesser expected response, source-semantic strength, and runtime-owned common policy. For a paired ordering in those strings, immutable source object-class semantics plus visible functional construction/role are authoritative; current MaterialInference prose/numerics are diagnostic evidence, not ordering authority. A cotton-covered/thin/curved adjective cannot invert an established shape-holding, reinforced, or structural-support role into the more-compliant side; if source role is not established, use weak/uncertain strength rather than inventing an opposite ordering from that adjective. Membership carries no expected winner, knob, threshold, or telemetry request."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    require_v2_session_state(state, allow_fresh_unplanned=True)
    diagnostics = _diagnostics(state)
    if diagnostics.get("active_diagnostic_session_plan") or diagnostics.get("diagnostic_session_plans"):
        raise GenesisVlmSchemaError("diagnostic session plan is immutable and may be submitted only once")
    if any(isinstance(session, Mapping) and session.get("lifecycle_state") not in {"closed", "closed_failed"}
           for session in state.get("diagnostic_runtime_sessions", [])):
        raise GenesisVlmSchemaError("cannot submit diagnostic session plan while a live session is open")
    validated_plan = validate_diagnostic_session_plan(plan)
    diagnostics["diagnostic_session_plans"].append(validated_plan)
    diagnostics["active_diagnostic_session_plan"] = validated_plan
    diagnostics["session_plan_version"] = VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION
    diagnostics["diagnostic_region_ledger"] = [
        {
            "region_id": region["region_id"], "risk_rank": region["risk_rank"],
            "region": region, "status": "active", "episode_id": None,
            "probe_intent_id": None, "qualifying_attempt_id": None,
        }
        for region in validated_plan["regions"]
    ]
    # Policy is exclusively backend-owned.  The fingerprint is durable so a
    # peer cannot be silently retuned after its counterpart has run.
    policies: list[dict[str, Any]] = []
    for pair in validated_plan["paired_comparisons"]:
        policy = {
            "pair_id": pair["pair_id"], "region_ids": list(pair["region_ids"]),
            "normalized_distance_scale": 0.5, "speed_m_s": 0.6,
            "probe_window_steps": DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
            "release_window_steps": DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
        }
        fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        policy["policy_fingerprint"] = fingerprint
        policies.append(policy)
    diagnostics["paired_comparison_policies"] = policies
    pair_by_region = {
        region_id: policy for policy in policies for region_id in policy["region_ids"]
    }
    for region in diagnostics["diagnostic_region_ledger"]:
        policy = pair_by_region.get(region["region_id"])
        if policy is not None:
            region["pair_id"] = policy["pair_id"]
            region["pair_policy_fingerprint"] = policy["policy_fingerprint"]
    # An overlap exception is semantic plan input, but the concern that makes
    # it non-ignorable is runtime-owned.  Persist it with the immutable plan,
    # before any setup reflection or episode allocation can hide it.
    revision = str(state.get("active_revision") or "")
    diagnostics["episode_concerns"].extend(
        {
            "concern_id": _overlap_concern_id(str(region["region_id"])),
            "origin": "runtime",
            "concern_type": "anchor_probe_overlap",
            "region_id": str(region["region_id"]),
            "episode_id": None,
            "state": "pending_setup_acknowledgement",
            "declaration": "overlap_exception",
            "plan_declaration_ref": {
                "region_id": str(region["region_id"]),
                "relationship_to_probe": "overlap_exception",
            },
            "active_revision_id": revision,
        }
        for region in validated_plan["regions"]
        if region["setup_anchor"]["relationship_to_probe"] == "overlap_exception"
    )
    diagnostics["active_region_id"] = validated_plan["regions"][0]["region_id"]
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_session_plan_recorded",
        note=validated_plan["session_intent"],
        detail=validated_plan,
    )
    return {"status": "success", "session_plan": validated_plan, "state_path": str(state_path(bound_run_root))}


def _pair_policy_for_region(diagnostics: Mapping[str, Any], region_id: str) -> dict[str, Any] | None:
    policies = diagnostics.get("paired_comparison_policies", [])
    matches = [item for item in policies if isinstance(item, Mapping) and region_id in item.get("region_ids", [])]
    if not matches:
        return None
    if len(matches) != 1:
        raise GenesisVlmSchemaError("region has ambiguous paired-comparison policy")
    return dict(matches[0])


def _effective_pair_direction_policy(
    diagnostics: Mapping[str, Any], *, policy: Mapping[str, Any] | None,
    compiled: Mapping[str, Any], region_id: str, compile_id: str,
) -> tuple[str, str, dict[str, Any] | None]:
    axis = str(compiled.get("motion_axis", "+Y"))
    if axis not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise GenesisVlmSchemaError("compiled probe has invalid motion_axis")
    if policy is None or not bool(compiled.get("mechanics_locality_required", False)):
        return axis, str(policy["policy_fingerprint"]) if policy else "ordinary-v1-default", None
    pair_id = str(policy["pair_id"])
    matches = [
        item for item in diagnostics.get("pair_direction_locks", [])
        if isinstance(item, Mapping) and str(item.get("pair_id")) == pair_id
    ]
    if len(matches) > 1:
        raise GenesisVlmSchemaError("pair has ambiguous durable direction locks")
    lock = dict(matches[0]) if matches else None
    if lock is not None and lock.get("axis") != axis:
        raise GenesisVlmSchemaError(
            f"paired mechanics probe must use locked signed axis {lock.get('axis')}, not {axis}"
        )
    effective_axis = str(lock["axis"]) if lock else axis
    fingerprint = hashlib.sha256(
        json.dumps(
            {"policy_fingerprint": str(policy["policy_fingerprint"]), "motion_axis": effective_axis},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if lock is not None and lock.get("effective_pair_execution_fingerprint") != fingerprint:
        raise GenesisVlmSchemaError("pair direction lock fingerprint is inconsistent with immutable controller policy")
    pending = None if lock is not None else {
        "pair_id": pair_id,
        "axis": effective_axis,
        "first_region_id": region_id,
        "first_compile_id": compile_id,
        "active_revision_id": str(compiled.get("active_revision_id", "")),
        "controller_policy_fingerprint": str(policy["policy_fingerprint"]),
        "effective_pair_execution_fingerprint": fingerprint,
    }
    return effective_axis, fingerprint, pending


def _overlap_concern_id(region_id: str) -> str:
    return f"runtime:region:{region_id}:anchor_probe_overlap"
def _require_finite_vector_values(value: Any, *, context: str, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise GenesisVlmSchemaError(f"{context} must be a finite {length}-vector")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise GenesisVlmSchemaError(f"{context} must contain only finite numbers")
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise GenesisVlmSchemaError(f"{context} must contain only finite numbers")
    return vector
def _geometry_context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    geometry = diagnostic_asset_geometry_payload(_active_diagnostic_mesh_path(state))
    raw_bounds = geometry.get("raw_bounds") if isinstance(geometry, dict) else {}
    if not isinstance(raw_bounds, dict) or not raw_bounds.get("available"):
        raise GenesisVlmSchemaError("cannot compute geometry context because active mesh bounds are unavailable")
    mins = _require_finite_vector_values(raw_bounds.get("min"), context="geometry_context.raw_bounds.min", length=3)
    maxs = _require_finite_vector_values(raw_bounds.get("max"), context="geometry_context.raw_bounds.max", length=3)
    if any(left >= right for left, right in zip(mins, maxs)):
        raise GenesisVlmSchemaError("cannot compute geometry context because active mesh bbox is degenerate")
    scene_fit = geometry.get("scene_fit") if isinstance(geometry, dict) else {}
    if not isinstance(scene_fit, dict):
        raise GenesisVlmSchemaError("cannot compute geometry context because active mesh scene fit is unavailable")
    scale_value = scene_fit.get("scale")
    if isinstance(scale_value, bool) or not isinstance(scale_value, int | float):
        raise GenesisVlmSchemaError("geometry_context.scene_fit.scale must be finite")
    scale = float(scale_value)
    if not math.isfinite(scale):
        raise GenesisVlmSchemaError("geometry_context.scene_fit.scale must be finite")
    translation = _require_finite_vector_values(
        scene_fit.get("translation"),
        context="geometry_context.scene_fit.translation",
        length=3,
    )
    return {
        "local_mesh_bbox": [*mins, *maxs],
        "local_to_world_transform": [
            [scale, 0.0, 0.0, translation[0]],
            [0.0, scale, 0.0, translation[1]],
            [0.0, 0.0, scale, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
def _required_state_path(state: dict[str, Any], key: str) -> Path:
    paths = state.get("paths", {})
    if not isinstance(paths, dict) or not paths.get(key):
        raise ValueError(f"cannot build part grounding context because paths.{key} is missing")
    path = Path(str(paths[key])).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cannot build part grounding context because paths.{key} does not exist: {path}")
    return path
def build_part_grounding_context_from_state(
    state: dict[str, Any],
    *,
    label_view_artifact_ids: tuple[str, ...] = (),
    generated_asset_json_path: Path | None = None,
) -> dict[str, Any]:
    """Build runtime-owned final-mesh per-part grounding context from state paths."""
    return build_part_grounding_context(
        run_id=str(state.get("run_id", "")),
        object_name=str(state.get("object_name", "")),
        inferred_params_path=_required_state_path(state, "inferred_params_path"),
        part_labels_path=_required_state_path(state, "part_labels_path"),
        monolithic_mesh_path=_required_state_path(state, "monolithic_mesh_path"),
        monolithic_params_path=_required_state_path(state, "monolithic_params_path"),
        label_view_artifact_ids=label_view_artifact_ids,
        generated_asset_json_path=generated_asset_json_path,
    )
def _part_grounding_context_from_state(state: dict[str, Any]) -> dict[str, Any]:
    label_view_artifact_ids = tuple(f"omnipart_view_{view_name}" for view_name in ("top", "bottom", "front", "back", "left", "right"))
    return build_part_grounding_context_from_state(state, label_view_artifact_ids=label_view_artifact_ids)
def _record_part_grounding_context(
    state: dict[str, Any],
    context: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    diagnostics["part_grounding_context"] = context
    measurement = {
        "measurement_index": len(diagnostics["part_grounding_context_measurements"]),
        "source": str(source),
        "part_grounding_context": context,
    }
    diagnostics["part_grounding_context_measurements"].append(measurement)
    return measurement
def _ensure_part_grounding_context(state: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    context = diagnostics.get("part_grounding_context")
    if isinstance(context, dict) and context.get("schema_version"):
        return context
    context = _part_grounding_context_from_state(state)
    _record_part_grounding_context(state, context, source="probe_targeting")
    return context
def diagnostic_triple_view_cameras_from_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context = _ensure_part_grounding_context(state)
    mesh = context.get("mesh") if isinstance(context.get("mesh"), Mapping) else {}
    mesh_box = mesh.get("bbox_m") if isinstance(mesh, Mapping) else None
    transform = context.get("transform") if isinstance(context.get("transform"), Mapping) else None
    if mesh_box is None or transform is None:
        raise GenesisVlmSchemaError("part grounding context is missing mesh.bbox_m or transform for triple-view cameras")
    try:
        env_box = mesh_box_to_env_aabb(mesh_box, transform)["box"]
    except Exception as exc:
        raise GenesisVlmSchemaError(f"failed to compute env-local triple-view camera bbox: {exc}") from exc
    return triple_view_cameras(env_box, fly_to=False)
def _probe_target_intent_by_id(diagnostics: dict[str, Any], target_id: str) -> dict[str, Any]:
    for intent in diagnostics.get("probe_target_intents", []):
        if isinstance(intent, dict) and str(intent.get("target_id")) == str(target_id):
            return intent
    raise GenesisVlmSchemaError(f"diagnostic probe target intent is not recorded: {target_id}")
def _session_anchor_ids(state: dict[str, Any]) -> set[str]:
    plan = require_v2_session_state(state)
    assert isinstance(plan, Mapping)
    # Target compiler still calls this a set of anchor ids; v2 identities are region ids.
    return {str(region["region_id"]) for region in plan.get("regions", []) if isinstance(region, Mapping)}


def _planned_region(state: dict[str, Any], region_id: str) -> dict[str, Any]:
    require_v2_session_state(state)
    diagnostics = _diagnostics(state)
    matches = [item for item in diagnostics["diagnostic_region_ledger"] if isinstance(item, dict) and item.get("region_id") == region_id]
    if len(matches) != 1:
        raise GenesisVlmSchemaError(f"region_id is not uniquely owned by the active v2 plan: {region_id}")
    record = matches[0]
    if record.get("status") not in {"active", "episode_defined"}:
        raise GenesisVlmSchemaError(f"region {region_id} is settled or unavailable")
    region = record.get("region")
    if not isinstance(region, dict):
        raise GenesisVlmSchemaError(f"region {region_id} has malformed immutable plan declaration")
    return record


def _part_grounding_part_by_id(context: Mapping[str, Any], part_id: int) -> Mapping[str, Any]:
    parts = context.get("parts")
    if not isinstance(parts, list):
        raise GenesisVlmSchemaError("part grounding context is missing canonical parts for semantic group validation")
    matches = [part for part in parts if isinstance(part, Mapping) and part.get("part_id") == part_id]
    if len(matches) != 1:
        raise GenesisVlmSchemaError(f"semantic group part {part_id} is absent from canonical source semantics")
    return matches[0]


def _verify_probe_semantic_group_lineage(
    context: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> None:
    """Verify immutable agent-authored member snapshots against fresh canonical grounding."""
    selected_part_id = intent.get("selected_part_id")
    group = intent.get("semantic_group_part_ids", [selected_part_id])
    grounding = intent.get("semantic_group_part_grounding", [])
    if not isinstance(group, list) or not group:
        raise GenesisVlmSchemaError("probe semantic group lineage is missing")
    if len(group) == 1:
        if grounding != []:
            raise GenesisVlmSchemaError("legacy singleton probe target must not carry semantic group member grounding")
        return
    if not isinstance(grounding, list) or len(grounding) != len(group):
        raise GenesisVlmSchemaError("non-singleton semantic group requires complete member grounding lineage")
    for index, part_id in enumerate(group):
        entry = grounding[index]
        if not isinstance(entry, Mapping) or entry.get("part_id") != part_id:
            raise GenesisVlmSchemaError("semantic group member grounding order must match semantic_group_part_ids")
        part = _part_grounding_part_by_id(context, int(part_id))
        if entry.get("part_name") != part.get("part_name") or entry.get("part_semantics") != part.get("part_semantics"):
            raise GenesisVlmSchemaError(
                "semantic group member grounding must exactly match the current canonical part_name and part_semantics"
            )


def _region_id_for_episode(state: dict[str, Any], episode_id: str) -> str:
    diagnostics = _diagnostics(state)
    matches = [entry for entry in diagnostics["diagnostic_region_ledger"] if isinstance(entry, Mapping) and entry.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise GenesisVlmSchemaError(f"episode {episode_id} lacks unique v2 region ownership")
    return str(matches[0]["region_id"])
def _anchor_target_intent_by_anchor(diagnostics: dict[str, Any], anchor_id: str) -> dict[str, Any]:
    active_by_anchor = diagnostics.get("active_anchor_target_intent_by_anchor")
    active_intent_id = ""
    if isinstance(active_by_anchor, Mapping):
        active_intent_id = str(active_by_anchor.get(str(anchor_id), ""))
    for intent in diagnostics.get("anchor_target_intents", []):
        if not isinstance(intent, dict):
            continue
        if active_intent_id and str(intent.get("intent_id")) == active_intent_id:
            return intent
    for intent in reversed(diagnostics.get("anchor_target_intents", [])):
        if isinstance(intent, dict) and str(intent.get("anchor_id")) == str(anchor_id):
            return intent
    raise GenesisVlmSchemaError(f"diagnostic anchor target intent is not recorded: {anchor_id}")
def _compiled_probe_target_by_id(diagnostics: dict[str, Any], compile_id: str) -> dict[str, Any]:
    for compiled in diagnostics.get("compiled_probe_targets", []):
        if isinstance(compiled, dict) and str(compiled.get("compile_id")) == str(compile_id):
            return compiled
    raise GenesisVlmSchemaError(f"compiled diagnostic probe target is not recorded: {compile_id}")
def _compiled_anchor_target_by_id(diagnostics: dict[str, Any], compile_id: str) -> dict[str, Any]:
    for compiled in diagnostics.get("compiled_anchor_targets", []):
        if isinstance(compiled, dict) and str(compiled.get("compile_id")) == str(compile_id):
            return compiled
    raise GenesisVlmSchemaError(f"compiled diagnostic anchor target is not recorded: {compile_id}")
def _active_compiled_anchor_target_for_setup(
    diagnostics: dict[str, Any],
    anchor_id: str,
) -> dict[str, Any]:
    active_by_anchor = diagnostics.get("active_compiled_anchor_target_by_anchor")
    compile_id = str(active_by_anchor.get(str(anchor_id), "")) if isinstance(active_by_anchor, Mapping) else ""
    if not compile_id:
        raise GenesisVlmSchemaError(
            f"preview_diagnostic_episode_setup requires an active compiled anchor target for {anchor_id}; "
            "call compile_diagnostic_anchor_target and preview_diagnostic_anchor_target first"
        )
    compiled = _compiled_anchor_target_by_id(diagnostics, compile_id)
    if str(compiled.get("anchor_id", "")) != str(anchor_id):
        raise GenesisVlmSchemaError(
            f"active compiled anchor target {compile_id} belongs to {compiled.get('anchor_id', '')}, not {anchor_id}"
        )
    validation = compiled.get("validation") if isinstance(compiled.get("validation"), Mapping) else {}
    hard_errors = validation.get("hard_errors") if isinstance(validation.get("hard_errors"), list) else []
    if str(validation.get("status", "")) == "error" or hard_errors:
        raise GenesisVlmSchemaError(f"active compiled anchor target {compile_id} is not valid for setup preview")
    return compiled
def _required_anchor_preview_evidence_for_setup(
    diagnostics: dict[str, Any],
    *,
    anchor_id: str,
    compile_id: str,
) -> dict[str, Any]:
    for evidence in reversed(diagnostics.get("triple_view_evidence", [])):
        if not isinstance(evidence, Mapping):
            continue
        if (
            str(evidence.get("source_kind", "")) == "static_anchor_preview"
            and str(evidence.get("target_id", "")) == str(anchor_id)
            and str(evidence.get("compile_id", "")) == str(compile_id)
            and list(evidence.get("panel_order", [])) == list(TRIPLE_VIEW_PANEL_ORDER)
        ):
            return {
                "evidence_id": str(evidence.get("evidence_id", "")),
                "source_kind": "static_anchor_preview",
                "target_id": str(evidence.get("target_id", "")),
                "compile_id": str(evidence.get("compile_id", "")),
                "panel_order": list(evidence.get("panel_order", [])),
                "triple_view_manifest_path": str(evidence.get("triple_view_manifest_path", "")),
                "triptych_png_path": str(evidence.get("triptych_png_path", "")),
            }
    raise GenesisVlmSchemaError(
        "preview_diagnostic_episode_setup requires current same-anchor, same-compile static_anchor_preview evidence"
    )
def _latest_anchor_pin_box(state: dict[str, Any]) -> list[float] | None:
    diagnostics = _diagnostics(state)
    for trial in reversed(diagnostics.get("setup_trials", [])):
        if not isinstance(trial, dict):
            continue
        pinning = trial.get("derived_pinning")
        boxes = pinning.get("boxes") if isinstance(pinning, dict) else None
        if isinstance(boxes, list) and boxes and isinstance(boxes[0], dict) and isinstance(boxes[0].get("box"), list):
            return [float(value) for value in boxes[0]["box"]]
    return None


def _owning_anchor_snapshot(state: dict[str, Any], *, region_id: str) -> dict[str, Any]:
    """Resolve this region's accepted static-anchor trial, never global history."""
    diagnostics = _diagnostics(state)
    active = diagnostics.get("active_setup_trial_by_anchor")
    trial_id = str(active.get(region_id, "")) if isinstance(active, Mapping) else ""
    if not trial_id:
        raise GenesisVlmSchemaError(f"mechanics probe requires an active owning setup trial for region {region_id}")
    trial = _setup_trial_by_id(diagnostics, trial_id)
    if str(trial.get("anchor_id", "")) != region_id:
        raise GenesisVlmSchemaError("owning static anchor trial does not match probe region")
    pinning = trial.get("derived_pinning")
    boxes = pinning.get("boxes") if isinstance(pinning, Mapping) else None
    if not isinstance(boxes, list) or len(boxes) != 1 or not isinstance(boxes[0], Mapping):
        raise GenesisVlmSchemaError("owning static anchor trial must contain exactly one pin box")
    box = _require_finite_vector_values(boxes[0].get("box"), context="owning static anchor box", length=6)
    if any(box[axis] >= box[axis + 3] for axis in range(3)):
        raise GenesisVlmSchemaError("owning static anchor box must have strict min < max")
    return {
        "anchor_id": region_id,
        "trial_id": str(trial["trial_id"]),
        "anchor_compile_id": str(trial.get("anchor_compile_id", "")),
        "active_revision_id": str(state.get("active_revision") or ""),
        "box": box,
    }


def _mechanics_locality_required(pair_policy: Mapping[str, Any] | None, mechanics_probe_mode: str) -> bool:
    return bool(pair_policy) and mechanics_probe_mode in {
        "bending", "compliance", "relative_structural_response",
    }
def _compile_warning_codes(compiled: Mapping[str, Any]) -> list[str]:
    validation = compiled.get("validation") if isinstance(compiled.get("validation"), Mapping) else {}
    warnings = validation.get("warning_codes", []) if isinstance(validation.get("warning_codes"), list) else []
    return [str(item) for item in warnings]
def _compiled_probe_target_summary(compiled: Mapping[str, Any]) -> dict[str, Any]:
    metrics = compiled.get("metrics") if isinstance(compiled.get("metrics"), Mapping) else {}
    primitive_purity = metrics.get("primitive_purity") if isinstance(metrics.get("primitive_purity"), Mapping) else {}
    pin_overlap = metrics.get("pin_overlap") if isinstance(metrics.get("pin_overlap"), Mapping) else {}
    validation = compiled.get("validation") if isinstance(compiled.get("validation"), Mapping) else {}
    return {
        "target_id": str(compiled.get("target_id", "")),
        "compile_id": str(compiled.get("compile_id", "")),
        "selected_part_id": compiled.get("selected_part_id"),
        "semantic_group_part_ids": list(compiled.get("semantic_group_part_ids", [compiled.get("selected_part_id")])),
        "semantic_group_part_grounding": list(compiled.get("semantic_group_part_grounding", [])),
        "region_hint": str(compiled.get("region_hint", "")),
        "part_name": str(compiled.get("part_name", "")),
        "part_semantics": str(compiled.get("part_semantics", "")),
        "mechanics_probe_mode": str(compiled.get("mechanics_probe_mode", "none")),
        "mechanics_locality_required": bool(compiled.get("mechanics_locality_required", False)),
        "locality_strategy": str(compiled.get("locality_strategy", "ordinary_semantic_hint")),
        "motion_axis": str(compiled.get("motion_axis", "+Y")),
        "validation_status": str(validation.get("status", "")),
        "warning_codes": _compile_warning_codes(compiled),
        "target_patch_qualified": bool(primitive_purity.get("selected_part_fraction", 0.0) >= MIN_GRABBED_SELECTED_PART_FRACTION),
    }
def _compiled_anchor_target_summary(compiled: Mapping[str, Any]) -> dict[str, Any]:
    metrics = compiled.get("metrics") if isinstance(compiled.get("metrics"), Mapping) else {}
    primitive_purity = metrics.get("primitive_purity") if isinstance(metrics.get("primitive_purity"), Mapping) else {}
    validation = compiled.get("validation") if isinstance(compiled.get("validation"), Mapping) else {}
    return {
        "anchor_id": str(compiled.get("anchor_id", "")),
        "intent_id": str(compiled.get("intent_id", "")),
        "compile_id": str(compiled.get("compile_id", "")),
        "selected_part_id": compiled.get("selected_part_id"),
        "region_hint": str(compiled.get("region_hint", "")),
        "region_provenance": str(compiled.get("region_provenance", "")),
        "part_name": str(compiled.get("part_name", "")),
        "part_semantics": str(compiled.get("part_semantics", "")),
        "validation_status": str(validation.get("status", "")),
        "warning_codes": _compile_warning_codes(compiled),
        "hard_errors": list(validation.get("hard_errors", [])) if isinstance(validation.get("hard_errors"), list) else [],
        "primitive_purity": primitive_purity.get("selected_part_fraction"),
        "target_volume_m3": metrics.get("target_volume_m3"),
        "selected_part_bbox_volume_m3": metrics.get("selected_part_bbox_volume_m3"),
        "target_to_part_volume_ratio": metrics.get("target_to_part_volume_ratio"),
        "triple_view_evidence_id": str(compiled.get("triple_view_evidence_id", "")),
    }
def _probe_target_intent_summary(intent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_id": str(intent.get("target_id", "")),
        "selected_part_id": intent.get("selected_part_id"),
        "semantic_group_part_ids": list(intent.get("semantic_group_part_ids", [intent.get("selected_part_id")])),
        "semantic_group_part_grounding": list(intent.get("semantic_group_part_grounding", [])),
        "region_hint": str(intent.get("region_hint", "")),
        "target_intent": str(intent.get("target_intent", "")),
        "desired_interaction": str(intent.get("desired_interaction", "")),
        "uncertainty": str(intent.get("uncertainty", "")),
        "mechanics_probe_mode": str(intent.get("mechanics_probe_mode", "none")),
        "mechanics_locality_required": bool(intent.get("mechanics_locality_required", False)),
        "motion_axis": str(intent.get("motion_axis", "+Y")),
    }
def _anchor_target_intent_summary(intent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": str(intent.get("intent_id", "")),
        "intent_index": intent.get("intent_index"),
        "anchor_id": str(intent.get("anchor_id", "")),
        "selected_part_id": intent.get("selected_part_id"),
        "region_hint": str(intent.get("region_hint", "")),
        "anchor_intent": str(intent.get("anchor_intent", "")),
        "physical_boundary_condition": str(intent.get("physical_boundary_condition", "")),
        "uncertainty": str(intent.get("uncertainty", "")),
    }
def _probe_target_preview_summary(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_id": str(preview.get("target_id", "")),
        "compile_id": str(preview.get("compile_id", "")),
        "trial_id": str(preview.get("trial_id", "")),
        "preview_manifest_path": str(preview.get("preview_manifest_path", "")),
        "triptych_png_path": str(preview.get("triptych_png_path", "")),
        "triple_view_manifest_path": str(preview.get("triple_view_manifest_path", "")),
        "triple_view_evidence_id": str(preview.get("triple_view_evidence_id", "")),
        "panel_order": list(preview.get("panel_order", [])) if isinstance(preview.get("panel_order"), list) else [],
        "view_png_paths": dict(preview.get("view_png_paths", {})) if isinstance(preview.get("view_png_paths"), dict) else {},
        "motion_axis": str(preview.get("motion_axis", "+Y")),
        "motion_axis_arrow": bool(preview.get("motion_axis_arrow", False)),
    }
def _anchor_target_preview_summary(preview: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_id": str(preview.get("anchor_id", "")),
        "compile_id": str(preview.get("compile_id", "")),
        "trial_id": str(preview.get("trial_id", "")),
        "preview_manifest_path": str(preview.get("preview_manifest_path", "")),
        "triptych_png_path": str(preview.get("triptych_png_path", "")),
        "triple_view_manifest_path": str(preview.get("triple_view_manifest_path", "")),
        "triple_view_evidence_id": str(preview.get("triple_view_evidence_id", "")),
        "panel_order": list(preview.get("panel_order", [])) if isinstance(preview.get("panel_order"), list) else [],
        "view_png_paths": dict(preview.get("view_png_paths", {})) if isinstance(preview.get("view_png_paths"), dict) else {},
    }
def _probe_target_validation_summary(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_id": str(validation.get("target_id", "")),
        "compile_id": str(validation.get("compile_id", "")),
        "semantic_group_part_ids": list(validation.get("semantic_group_part_ids", [validation.get("selected_part_id")])),
        "semantic_group_part_grounding": list(validation.get("semantic_group_part_grounding", [])),
        "tool_result_index": validation.get("tool_result_index"),
        "status": str(validation.get("status", "")),
        "grabbed_selected_part_fraction": validation.get("grabbed_selected_part_fraction"),
        "warning_codes": list(validation.get("warning_codes", [])) if isinstance(validation.get("warning_codes"), list) else [],
        "hard_errors": list(validation.get("hard_errors", [])) if isinstance(validation.get("hard_errors"), list) else [],
    }
def _normalize_triple_view_evidence_ref(ref: str) -> str:
    text = str(ref).strip()
    if text.startswith("triple_view:"):
        text = text.split(":", 1)[1].strip()
    if not text:
        raise GenesisVlmSchemaError("probe target edit evidence_refs entries must be non-empty")
    return text
def _triple_view_evidence_by_id(diagnostics: Mapping[str, Any], evidence_id: str) -> Mapping[str, Any]:
    for evidence in diagnostics.get("triple_view_evidence", []):
        if isinstance(evidence, Mapping) and str(evidence.get("evidence_id", "")) == str(evidence_id):
            return evidence
    raise GenesisVlmSchemaError(f"probe target edit references unknown triple-view evidence: {evidence_id}")
def _validate_probe_target_edit_evidence_refs(
    diagnostics: Mapping[str, Any],
    *,
    target_id: str,
    compile_id: str,
    edit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence_refs = edit.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise GenesisVlmSchemaError("probe target edit requires at least one triple-view evidence ref")
    resolved: list[dict[str, Any]] = []
    for ref in evidence_refs:
        evidence_id = _normalize_triple_view_evidence_ref(str(ref))
        evidence = _triple_view_evidence_by_id(diagnostics, evidence_id)
        source_kind = str(evidence.get("source_kind", ""))
        evidence_target_id = str(evidence.get("target_id", ""))
        evidence_compile_id = str(evidence.get("compile_id", ""))
        if source_kind != "static_probe_preview":
            raise GenesisVlmSchemaError(
                f"probe target edit evidence {evidence_id} must be a static_probe_preview, got {source_kind}"
            )
        if evidence_target_id != str(target_id):
            raise GenesisVlmSchemaError(
                f"probe target edit evidence {evidence_id} belongs to target {evidence_target_id}, "
                f"not {target_id}"
            )
        if evidence_compile_id != str(compile_id):
            raise GenesisVlmSchemaError(
                f"probe target edit evidence {evidence_id} belongs to compile {evidence_compile_id}, "
                f"not {compile_id}"
            )
        panel_order = evidence.get("panel_order")
        if list(panel_order) != list(TRIPLE_VIEW_PANEL_ORDER):
            raise GenesisVlmSchemaError(f"probe target edit evidence {evidence_id} has invalid panel_order")
        resolved.append(
            {
                "evidence_id": evidence_id,
                "source_kind": source_kind,
                "target_id": evidence_target_id,
                "compile_id": evidence_compile_id,
                "triple_view_manifest_path": str(evidence.get("triple_view_manifest_path", "")),
                "triptych_png_path": str(evidence.get("triptych_png_path", "")),
            }
        )
    return resolved
def _validate_anchor_target_edit_evidence_refs(
    diagnostics: Mapping[str, Any],
    *,
    anchor_id: str,
    compile_id: str,
    edit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence_refs = edit.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise GenesisVlmSchemaError("anchor target edit requires at least one triple-view evidence ref")
    active_by_anchor = diagnostics.get("active_compiled_anchor_target_by_anchor")
    active_compile_id = str(active_by_anchor.get(str(anchor_id), "")) if isinstance(active_by_anchor, Mapping) else ""
    if active_compile_id != str(compile_id):
        raise GenesisVlmSchemaError(
            f"anchor target edit source compile {compile_id} is stale; active compile is {active_compile_id}"
        )
    resolved: list[dict[str, Any]] = []
    for ref in evidence_refs:
        evidence_id = _normalize_triple_view_evidence_ref(str(ref))
        evidence = _triple_view_evidence_by_id(diagnostics, evidence_id)
        source_kind = str(evidence.get("source_kind", ""))
        evidence_target_id = str(evidence.get("target_id", ""))
        evidence_compile_id = str(evidence.get("compile_id", ""))
        if source_kind != "static_anchor_preview":
            raise GenesisVlmSchemaError(
                f"anchor target edit evidence {evidence_id} must be a static_anchor_preview, got {source_kind}"
            )
        if evidence_target_id != str(anchor_id):
            raise GenesisVlmSchemaError(
                f"anchor target edit evidence {evidence_id} belongs to anchor {evidence_target_id}, not {anchor_id}"
            )
        if evidence_compile_id != str(compile_id):
            raise GenesisVlmSchemaError(
                f"anchor target edit evidence {evidence_id} belongs to compile {evidence_compile_id}, not {compile_id}"
            )
        panel_order = evidence.get("panel_order")
        if list(panel_order) != list(TRIPLE_VIEW_PANEL_ORDER):
            raise GenesisVlmSchemaError(f"anchor target edit evidence {evidence_id} has invalid panel_order")
        resolved.append(
            {
                "evidence_id": evidence_id,
                "source_kind": source_kind,
                "anchor_id": evidence_target_id,
                "compile_id": evidence_compile_id,
                "triple_view_manifest_path": str(evidence.get("triple_view_manifest_path", "")),
                "triptych_png_path": str(evidence.get("triptych_png_path", "")),
            }
        )
    return resolved
def _probe_targets_dir(state: dict[str, Any], target_id: str) -> Path:
    sim_diagnostics_dir = _repo_visible_path(state, Path(str(state["paths"]["sim_diagnostics_dir"])).expanduser())
    return sim_diagnostics_dir / "probe_targets" / target_id
def _anchor_targets_dir(state: dict[str, Any], anchor_id: str) -> Path:
    sim_diagnostics_dir = _repo_visible_path(state, Path(str(state["paths"]["sim_diagnostics_dir"])).expanduser())
    return sim_diagnostics_dir / "anchor_targets" / anchor_id
def _render_and_record_probe_target_preview(
    *,
    run_root: str,
    state: dict[str, Any],
    compile_id: str,
) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    context = _ensure_part_grounding_context(state)
    compiled = _compiled_probe_target_by_id(diagnostics, compile_id)
    target_id = str(compiled["target_id"])
    trial_id = f"{compile_id}_preview_{len(diagnostics['probe_target_preview_artifacts']) + 1:02d}"
    preview = render_probe_target_preview(
        context,
        compiled,
        output_dir=_probe_targets_dir(state, target_id),
        target_id=target_id,
        trial_id=trial_id,
        anchor_pin_box=compiled.get("anchor_pin_box") if isinstance(compiled.get("anchor_pin_box"), list) else None,
    )
    record = {
        "target_id": target_id,
        "compile_id": compile_id,
        "trial_id": trial_id,
        "preview_manifest_path": preview["manifest_path"],
        "view_png_paths": {name: data["path"] for name, data in preview["views"].items()},
        "triptych_png_path": preview["triptych_png_path"],
        "triple_view_manifest_path": preview["triple_view_manifest_path"],
        "triple_view_evidence_id": preview["triple_view_evidence_id"],
        "panel_order": list(preview["panel_order"]),
        "motion_axis": compiled.get("motion_axis", "+Y"),
        "motion_axis_arrow": bool(preview["manifest"].get("motion_axis_arrow", False)),
        "manifest": preview["manifest"],
        "triple_view_manifest": preview["triple_view_manifest"],
    }
    diagnostics["probe_target_preview_artifacts"].append(record)
    triple_record = _compact_triple_view_record(
        {
            **preview["triple_view_manifest"],
            "manifest_path": preview["triple_view_manifest_path"],
        }
    )
    _append_triple_view_evidence_once(state, triple_record)
    _save_diagnostics(
        run_root,
        state,
        event="diagnostic_probe_target_preview_rendered",
        note=f"rendered probe target preview {trial_id}",
        detail={"probe_target_preview": _probe_target_preview_summary(record)},
        provenance="runtime_probe_target_preview",
    )
    return {
        "status": "success",
        **_probe_target_preview_summary(record),
        "visibility_status": "preview_rendered",
        "next_action": "simulate_or_revise_diagnostic_probe_target",
    }
def _render_and_record_anchor_target_preview(
    *,
    run_root: str,
    state: dict[str, Any],
    compile_id: str,
) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    context = _ensure_part_grounding_context(state)
    compiled = _compiled_anchor_target_by_id(diagnostics, compile_id)
    anchor_id = str(compiled["anchor_id"])
    trial_id = f"{compile_id}_preview_{len(diagnostics['anchor_target_preview_artifacts']) + 1:02d}"
    preview = render_anchor_target_preview(
        context,
        compiled,
        output_dir=_anchor_targets_dir(state, anchor_id),
        anchor_id=anchor_id,
        trial_id=trial_id,
    )
    record = {
        "anchor_id": anchor_id,
        "compile_id": compile_id,
        "trial_id": trial_id,
        "preview_manifest_path": preview["manifest_path"],
        "view_png_paths": {name: data["path"] for name, data in preview["views"].items()},
        "triptych_png_path": preview["triptych_png_path"],
        "triple_view_manifest_path": preview["triple_view_manifest_path"],
        "triple_view_evidence_id": preview["triple_view_evidence_id"],
        "panel_order": list(preview["panel_order"]),
        "manifest": preview["manifest"],
        "triple_view_manifest": preview["triple_view_manifest"],
    }
    diagnostics["anchor_target_preview_artifacts"].append(record)
    compiled["triple_view_evidence_id"] = preview["triple_view_evidence_id"]
    triple_record = _compact_triple_view_record(
        {
            **preview["triple_view_manifest"],
            "manifest_path": preview["triple_view_manifest_path"],
        }
    )
    _append_triple_view_evidence_once(state, triple_record)
    _save_diagnostics(
        run_root,
        state,
        event="diagnostic_anchor_target_preview_rendered",
        note=f"rendered anchor target preview {trial_id}",
        detail={"anchor_target_preview": _anchor_target_preview_summary(record)},
        provenance="runtime_anchor_target_preview",
    )
    return {
        "status": "success",
        **_anchor_target_preview_summary(record),
        "visibility_status": "preview_rendered",
        "next_action": "revise_diagnostic_anchor_target_or_preview_diagnostic_episode_setup",
    }
def _redact_internal_probe_target(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_internal_probe_target(item)
            for key, item in value.items()
            if key
            not in {
                "aabb_box",
                "genesis_aabb_box",
                "selected_vertex_indices",
                "resolved_vertices",
                "affected_vertices",
                "resolved_object_local_vertices",
                "affected_object_local_vertices",
            }
        }
    if isinstance(value, list):
        return [_redact_internal_probe_target(item) for item in value]
    return value
def _redact_internal_anchor_target(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_internal_anchor_target(item)
            for key, item in value.items()
            if key
            not in {
                "genesis_aabb_box",
                "selected_vertex_indices",
            }
        }
    if isinstance(value, list):
        return [_redact_internal_anchor_target(item) for item in value]
    return value
def compute_geometry_context() -> dict[str, Any]:
    """Return legacy minimal active mesh geometry context for audits; not an anchor authoring surface."""
    bound_run_root = active_diagnostic_run_root()
    if bound_run_root is None:
        raise RuntimeError("diagnostic tool called without active diagnostic run binding")
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    context = _geometry_context_from_state(state)
    measurement = {
        "measurement_index": len(diagnostics["geometry_context_measurements"]),
        "geometry_context": context,
    }
    diagnostics["geometry_context_measurements"].append(measurement)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_geometry_context_computed",
        note="computed minimal diagnostic geometry context",
        detail={"geometry_context": measurement},
    )
    return {"geometry_context": context}
def compute_part_grounding_context() -> dict[str, Any]:
    """Return active final-mesh per-part grounding context for diagnostic targeting."""
    bound_run_root = active_diagnostic_run_root()
    if bound_run_root is None:
        raise RuntimeError("diagnostic tool called without active diagnostic run binding")
    state = _require_state(bound_run_root)
    context = _part_grounding_context_from_state(state)
    measurement = _record_part_grounding_context(state, context, source="compute_part_grounding_context")
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_part_grounding_context_computed",
        note="computed runtime-owned diagnostic part grounding context",
        detail={"part_grounding_context": _part_grounding_measurement_preview(measurement)},
    )
    return {"part_grounding_context": compact_part_grounding_table(context)}
def submit_diagnostic_anchor_target_intent(
    run_root: str,
    region_id: str,
    anchor_intent: dict[str, Any],
) -> dict[str, Any]:
    """Record semantic anchor target intent with selected final-mesh part_id and region_hint."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    region_record = _planned_region(state, str(region_id))
    context = _ensure_part_grounding_context(state)
    anchor_id_text = str(region_id).strip()
    if not anchor_id_text:
        raise GenesisVlmSchemaError("submit_diagnostic_anchor_target_intent.anchor_id must be non-empty")
    if not isinstance(anchor_intent, Mapping):
        raise GenesisVlmSchemaError("submit_diagnostic_anchor_target_intent.anchor_intent must be an object")
    payload_anchor_id = str(anchor_intent.get("anchor_id", "")).strip()
    if payload_anchor_id != anchor_id_text:
        raise GenesisVlmSchemaError("submit_diagnostic_anchor_target_intent anchor_id does not match payload anchor_id")
    session_anchor_ids = _session_anchor_ids(state)
    validated = validate_diagnostic_anchor_target_intent(
        anchor_intent,
        part_grounding_context=context,
        session_anchor_ids=session_anchor_ids,
    )
    safe_anchor_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", anchor_id_text).strip("_.-") or "anchor"
    intent_index = len(diagnostics["anchor_target_intents"])
    intent_id = f"{safe_anchor_id}_anchor_target_intent_{intent_index + 1:02d}"
    record = {
        "intent_id": intent_id,
        "intent_index": intent_index,
        "region_id": anchor_id_text,
        **validated,
    }
    diagnostics["anchor_target_intents"].append(record)
    diagnostics["active_anchor_target_intent_by_anchor"][anchor_id_text] = intent_id
    summary = _anchor_target_intent_summary(record)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_anchor_target_intent_recorded",
        note=f"recorded anchor target intent {intent_id}",
        detail={"anchor_target_intent": summary},
        provenance="typed_model_anchor_target",
    )
    return {
        "status": "success",
        "intent_id": intent_id,
        "region_id": anchor_id_text,
        "selected_part_id": record["selected_part_id"],
        "region_hint": record["region_hint"],
        "next_action": "compile_diagnostic_anchor_target",
        "setup_note": "compile and preview the anchor target before setup preview derives runtime pinning",
    }
def compile_diagnostic_anchor_target(run_root: str, anchor_id: str) -> dict[str, Any]:
    """Compile a semantic anchor target intent into a runtime-owned part-grounded box."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    _planned_region(state, str(anchor_id))
    context = _ensure_part_grounding_context(state)
    anchor_id_text = str(anchor_id).strip()
    if not anchor_id_text:
        raise GenesisVlmSchemaError("compile_diagnostic_anchor_target.anchor_id must be non-empty")
    intent = _anchor_target_intent_by_anchor(diagnostics, anchor_id_text)
    try:
        compiled = compile_anchor_target_from_context(
            context,
            selected_part_id=int(intent["selected_part_id"]),
            region_hint=str(intent["region_hint"]),
        )
    except Exception as exc:
        raise GenesisVlmSchemaError(str(exc)) from exc
    safe_anchor_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", anchor_id_text).strip("_.-") or "anchor"
    compile_count = sum(
        1
        for item in diagnostics["compiled_anchor_targets"]
        if isinstance(item, dict) and str(item.get("anchor_id")) == anchor_id_text
    )
    compile_id = f"{safe_anchor_id}_anchor_compile_{compile_count + 1:02d}"
    compiled = {
        **compiled,
        "compile_id": compile_id,
        "anchor_id": anchor_id_text,
        "intent_id": str(intent["intent_id"]),
        "anchor_intent": str(intent["anchor_intent"]),
        "physical_boundary_condition": str(intent["physical_boundary_condition"]),
        "compile_index": len(diagnostics["compiled_anchor_targets"]),
        "region_id": str(anchor_id),
        "active_revision_id": str(state.get("active_revision") or ""),
    }
    diagnostics["compiled_anchor_targets"].append(compiled)
    diagnostics["active_compiled_anchor_target_by_anchor"][anchor_id_text] = compile_id
    summary = _compiled_anchor_target_summary(compiled)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_anchor_target_compiled",
        note=f"compiled anchor target {compile_id}",
        detail={"compiled_anchor_target": summary},
        provenance="runtime_anchor_target_compiler",
    )
    return {
        "status": "success",
        **summary,
        "next_action": "preview_diagnostic_anchor_target",
    }
def preview_diagnostic_anchor_target(run_root: str, compile_id: str) -> dict[str, Any]:
    """Render top/ne_3q/sw_3q previews for a compiled runtime-owned anchor target."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    return _render_and_record_anchor_target_preview(run_root=bound_run_root, state=state, compile_id=compile_id)
def revise_diagnostic_anchor_target(run_root: str, compile_id: str, edit: dict[str, Any]) -> dict[str, Any]:
    """Apply a bounded relative edit to a compiled anchor target and render a new preview."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    context = _ensure_part_grounding_context(state)
    previous = _compiled_anchor_target_by_id(diagnostics, compile_id)
    anchor_id = str(previous["anchor_id"])
    intent = _anchor_target_intent_by_anchor(diagnostics, anchor_id)
    edit_index = len(diagnostics["anchor_target_edit_trials"])
    try:
        validated_edit = validate_diagnostic_anchor_target_edit(edit)
        edit_evidence_refs = _validate_anchor_target_edit_evidence_refs(
            diagnostics,
            anchor_id=anchor_id,
            compile_id=compile_id,
            edit=validated_edit,
        )
        compiled = compile_anchor_target_from_context(
            context,
            selected_part_id=int(intent["selected_part_id"]),
            region_hint=str(intent["region_hint"]),
            edit=validated_edit,
        )
    except Exception as exc:
        rejected = {
            "edit_index": edit_index,
            "status": "rejected",
            "source_compile_id": compile_id,
            "anchor_id": anchor_id,
            "error": str(exc),
        }
        diagnostics["anchor_target_edit_trials"].append(rejected)
        _save_diagnostics(
            bound_run_root,
            state,
            event="diagnostic_anchor_target_edit_rejected",
            note=str(exc),
            detail={"anchor_target_edit": rejected},
            provenance="typed_model_anchor_target_edit",
        )
        if isinstance(exc, GenesisVlmSchemaError):
            raise
        raise GenesisVlmSchemaError(str(exc)) from exc
    safe_anchor_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", anchor_id).strip("_.-") or "anchor"
    compile_count = sum(
        1
        for item in diagnostics["compiled_anchor_targets"]
        if isinstance(item, dict) and str(item.get("anchor_id")) == anchor_id
    )
    new_compile_id = f"{safe_anchor_id}_anchor_compile_{compile_count + 1:02d}"
    compiled = {
        **compiled,
        "compile_id": new_compile_id,
        "anchor_id": anchor_id,
        "intent_id": str(intent["intent_id"]),
        "anchor_intent": str(intent["anchor_intent"]),
        "physical_boundary_condition": str(intent["physical_boundary_condition"]),
        "source_compile_id": compile_id,
        "compile_index": len(diagnostics["compiled_anchor_targets"]),
        "edit_evidence_refs": edit_evidence_refs,
        "region_id": anchor_id,
        "active_revision_id": str(state.get("active_revision") or ""),
    }
    edit_record = {
        "edit_index": edit_index,
        "status": "accepted",
        "source_compile_id": compile_id,
        "compile_id": new_compile_id,
        "anchor_id": anchor_id,
        "edit": validated_edit,
        "resolved_evidence_refs": edit_evidence_refs,
    }
    diagnostics["anchor_target_edit_trials"].append(edit_record)
    diagnostics["compiled_anchor_targets"].append(compiled)
    diagnostics["active_compiled_anchor_target_by_anchor"][anchor_id] = new_compile_id
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_anchor_target_revised",
        note=f"revised anchor target {new_compile_id}",
        detail={"compiled_anchor_target": _compiled_anchor_target_summary(compiled), "anchor_target_edit": edit_record},
        provenance="typed_model_anchor_target_edit",
    )
    return _render_and_record_anchor_target_preview(run_root=bound_run_root, state=state, compile_id=new_compile_id)
def submit_diagnostic_probe_target_intent(run_root: str, region_id: str, target_intent: dict[str, Any]) -> dict[str, Any]:
    """Record a semantic diagnostic probe target intent with selected part_id and region_hint."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    region_record = _planned_region(state, str(region_id))
    region_id = str(region_id)
    episode_id = str(region_record.get("episode_id") or "")
    if not episode_id:
        raise GenesisVlmSchemaError("probe target intent requires the region's defined episode")
    context = _ensure_part_grounding_context(state)
    try:
        validated = validate_diagnostic_probe_target_intent(target_intent, part_grounding_context=context)
    except GenesisVlmSchemaError:
        raise
    except Exception as exc:
        raise GenesisVlmSchemaError(str(exc)) from exc
    _verify_probe_semantic_group_lineage(context, validated)
    pair_policy = _pair_policy_for_region(diagnostics, region_id)
    locality_required = _mechanics_locality_required(pair_policy, str(validated["mechanics_probe_mode"]))
    if locality_required and validated["region_hint"] not in {"tip", "edge_band"}:
        raise GenesisVlmSchemaError(
            "paired bending/compliance/relative_structural_response probe intent requires region_hint tip or edge_band"
        )
    target_id = f"probe_target_{len(diagnostics['probe_target_intents']) + 1:04d}"
    record = {
        "target_id": target_id,
        "intent_index": len(diagnostics["probe_target_intents"]),
        "region_id": region_id,
        "simulate_episode_id": episode_id,
        "pair_id": pair_policy["pair_id"] if pair_policy else None,
        "mechanics_locality_required": locality_required,
        **validated,
    }
    diagnostics["probe_target_intents"].append(record)
    diagnostics["active_probe_target_intent_id"] = target_id
    diagnostics["region_probe_intent_ids"][region_id] = target_id
    region_record["probe_intent_id"] = target_id
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_probe_target_intent_recorded",
        note=f"recorded probe target intent {target_id}",
        detail={"probe_target_intent": _probe_target_intent_summary(record)},
        provenance="typed_model_probe_target",
    )
    return {
        "status": "success",
        "target_id": target_id,
        "region_id": region_id,
        "selected_part_id": record["selected_part_id"],
        "semantic_group_part_ids": list(record["semantic_group_part_ids"]),
        "semantic_group_part_grounding": list(record["semantic_group_part_grounding"]),
        "region_hint": record["region_hint"],
        "mechanics_probe_mode": record["mechanics_probe_mode"],
        "motion_axis": record["motion_axis"],
        "mechanics_locality_required": locality_required,
        "next_action": "compile_diagnostic_probe_target",
    }
def compile_diagnostic_probe_target(run_root: str, target_id: str) -> dict[str, Any]:
    """Compile a semantic probe target intent into a runtime-owned Genesis BoxEE target."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    context = _ensure_part_grounding_context(state)
    intent = _probe_target_intent_by_id(diagnostics, target_id)
    region_id = str(intent.get("region_id", ""))
    region_record = _planned_region(state, region_id)
    active_episode = _active_episode_record(state) if diagnostics.get("active_episode_index") is not None else None
    pair_policy = _pair_policy_for_region(diagnostics, region_id)
    locality_required = _mechanics_locality_required(pair_policy, str(intent.get("mechanics_probe_mode", "none")))
    if locality_required != bool(intent.get("mechanics_locality_required")):
        raise GenesisVlmSchemaError("probe intent mechanics locality no longer matches immutable pair policy")
    if locality_required and str(intent.get("region_hint")) not in {"tip", "edge_band"}:
        raise GenesisVlmSchemaError("mechanics-conditioned paired probe target requires tip or edge_band")
    anchor_snapshot = _owning_anchor_snapshot(state, region_id=region_id) if locality_required else None
    anchor_pin_box = list(anchor_snapshot["box"]) if anchor_snapshot else _latest_anchor_pin_box(state)
    try:
        _verify_probe_semantic_group_lineage(context, intent)
        compiled = compile_probe_target_from_context(
            context,
            selected_part_id=int(intent["selected_part_id"]),
            region_hint=str(intent["region_hint"]),
            anchor_pin_box=anchor_pin_box,
            mechanics_locality_required=locality_required,
            mechanics_probe_mode=str(intent.get("mechanics_probe_mode", "none")),
            motion_axis=str(intent.get("motion_axis", "+Y")),
        )
    except Exception as exc:
        raise GenesisVlmSchemaError(str(exc)) from exc
    compile_count = sum(
        1
        for item in diagnostics["compiled_probe_targets"]
        if isinstance(item, dict) and str(item.get("target_id")) == str(target_id)
    )
    compile_id = f"{target_id}_compile_{compile_count + 1:02d}"
    compiled = {
        **compiled,
        "compile_id": compile_id,
        "target_id": target_id,
        "intent_id": target_id,
        "anchor_pin_box": anchor_pin_box,
        "owning_anchor_snapshot": anchor_snapshot,
        "mechanics_probe_mode": str(intent.get("mechanics_probe_mode", "none")),
        "mechanics_locality_required": locality_required,
        "motion_axis": str(intent.get("motion_axis", "+Y")),
        "motion_axis_rationale": str(intent.get("motion_axis_rationale", "")),
        "compile_index": len(diagnostics["compiled_probe_targets"]),
        "active_revision_id": str(state.get("active_revision") or ""),
        "semantic_group_part_ids": list(intent["semantic_group_part_ids"]),
        "semantic_group_part_grounding": list(intent["semantic_group_part_grounding"]),
    }
    if active_episode is None or str(active_episode.get("episode_id")) != str(region_record.get("episode_id")):
        raise GenesisVlmSchemaError("probe compilation requires the owning region episode to be active")
    compiled["simulate_episode_id"] = str(active_episode["episode_id"])
    compiled["region_id"] = region_id
    diagnostics["compiled_probe_targets"].append(compiled)
    diagnostics["active_compiled_probe_target_id"] = compile_id
    diagnostics["region_current_probe_compile_ids"][region_id] = compile_id
    summary = _compiled_probe_target_summary(compiled)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_probe_target_compiled",
        note=f"compiled probe target {compile_id}",
        detail={"compiled_probe_target": summary},
        provenance="runtime_probe_target_compiler",
    )
    return {
        "status": "success",
        **summary,
        "next_action": "preview_diagnostic_probe_target",
    }
def preview_diagnostic_probe_target(run_root: str, compile_id: str) -> dict[str, Any]:
    """Render top/ne_3q/sw_3q previews for a compiled runtime-owned probe target."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    return _render_and_record_probe_target_preview(run_root=bound_run_root, state=state, compile_id=compile_id)


def _compiled_box(compiled: Mapping[str, Any], *, label: str) -> list[float]:
    payload = compiled.get("executable_probe_aabb_box") or compiled.get("executable_anchor_aabb_box") or compiled.get("genesis_aabb_box")
    if isinstance(payload, Mapping):
        payload = payload.get("box")
    return _require_finite_vector_values(payload, context=label, length=6)


def _positive_intersection_volume(left: list[float], right: list[float]) -> float:
    widths = [max(0.0, min(left[i + 3], right[i + 3]) - max(left[i], right[i])) for i in range(3)]
    return widths[0] * widths[1] * widths[2]


def _aabb_iou(left: list[float], right: list[float]) -> float:
    intersection = _positive_intersection_volume(left, right)
    left_volume = max(0.0, (left[3] - left[0]) * (left[4] - left[1]) * (left[5] - left[2]))
    right_volume = max(0.0, (right[3] - right[0]) * (right[4] - right[1]) * (right[5] - right[2]))
    union = left_volume + right_volume - intersection
    return intersection / union if union > 0 else 0.0


def _v2_owner_snapshot(state: dict[str, Any], *, region_id: str, episode_id: str | None, current_id: str, current_kind: str) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    attachment = _current_runtime_attachment(state, diagnostics)
    return {
        "attachment_id": str(attachment["attachment_id"]),
        "agent_invocation_id": str(attachment["agent_invocation_id"]),
        "active_revision_id": str(state.get("active_revision") or ""),
        "region_id": region_id, "episode_id": episode_id,
        current_kind: current_id,
    }


def _resolve_v2_preview_refs(
    state: dict[str, Any], *, refs: Any, region_id: str, episode_id: str | None,
    compile_id: str, preview_id: str, source_kind: str, target_key: str, target_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve typed v2 preview claims through existing triple-view evidence."""
    if not isinstance(refs, list) or not refs or not all(isinstance(item, Mapping) for item in refs):
        raise GenesisVlmSchemaError("typed reflection requires nonempty preview evidence refs")
    diagnostics = _diagnostics(state)
    evidence = _triple_view_evidence_by_id(diagnostics, preview_id)
    if (evidence.get("source_kind") != source_kind or str(evidence.get("compile_id", "")) != compile_id
            or str(evidence.get("target_id", "")) != target_id):
        raise GenesisVlmSchemaError("typed reflection preview is stale or does not belong to the current compile")
    normalized: list[dict[str, Any]] = []
    for raw in refs:
        ref = str(raw.get("ref", ""))
        if raw.get("kind") != "visual_evidence" or ref not in {preview_id, f"triple_view:{preview_id}"}:
            raise GenesisVlmSchemaError("typed reflection must cite exactly the current runtime-owned preview")
        normalized.append({"kind": "visual_evidence", "ref": preview_id, "source_kind": source_kind,
                           target_key: target_id, "compile_id": compile_id})
    snapshot = _v2_owner_snapshot(state, region_id=region_id, episode_id=episode_id, current_id=preview_id, current_kind="preview_evidence_id")
    snapshot.update({"compile_id": compile_id, target_key: target_id})
    return normalized, snapshot


def record_diagnostic_probe_target_reflection(
    run_root: str, region_id: str, compile_id: str, semantic_match: str,
    evidence_refs: list[dict[str, Any]], explanation: str,
) -> dict[str, Any]:
    bound = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound)
    diagnostics = _diagnostics(state)
    record = _planned_region(state, str(region_id))
    if semantic_match not in {"matched", "mismatched", "unresolved"}:
        raise GenesisVlmSchemaError("probe semantic_match must be matched, mismatched, or unresolved")
    compiled = _compiled_probe_target_by_id(diagnostics, compile_id)
    if compiled.get("region_id") != str(region_id) or diagnostics["region_current_probe_compile_ids"].get(str(region_id)) != compile_id:
        raise GenesisVlmSchemaError("probe reflection must bind the current region-owned compile")
    previews = [item for item in diagnostics["probe_target_preview_artifacts"] if isinstance(item, Mapping) and item.get("compile_id") == compile_id]
    if len(previews) != 1:
        raise GenesisVlmSchemaError("probe reflection requires exactly one current probe preview")
    preview_id = str(previews[0].get("triple_view_evidence_id", ""))
    resolved_refs, owner_snapshot = _resolve_v2_preview_refs(
        state, refs=evidence_refs, region_id=str(region_id), episode_id=str(record.get("episode_id") or "") or None,
        compile_id=compile_id, preview_id=preview_id, source_kind="static_probe_preview",
        target_key="target_id", target_id=str(compiled.get("target_id", "")),
    )
    reflection = {"reflection_id": f"probe_reflection_{len(diagnostics['probe_target_reflections']) + 1:04d}",
                  "region_id": str(region_id), "episode_id": record.get("episode_id"), "compile_id": compile_id,
                  "semantic_match": semantic_match, "evidence_refs": resolved_refs, "owner_snapshot": owner_snapshot,
                  "explanation": _require_nonempty_runtime_text(explanation, "probe reflection explanation"),
                  "active_revision_id": str(state.get("active_revision") or "")}
    diagnostics["probe_target_reflections"].append(reflection)
    _save_diagnostics(bound, state, event="diagnostic_probe_target_reflection_recorded", note=semantic_match, detail=reflection)
    return {"status": "success", "reflection": reflection, "next_action": "simulate_or_revise_diagnostic_probe_target"}


def _record_anchor_probe_separation(state: dict[str, Any], *, region_id: str, probe_compile: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    record = _planned_region(state, region_id)
    episode_id = str(record.get("episode_id") or "")
    episode = _episode_record_by_index(diagnostics, next(int(item["episode_index"]) for item in diagnostics["episodes"] if isinstance(item, Mapping) and item.get("episode_id") == episode_id))
    anchor_compile = _compiled_anchor_target_by_id(diagnostics, str(episode.get("anchor_compile_id") or ""))
    anchor_part = anchor_compile.get("selected_part_id")
    probe_part = probe_compile.get("selected_part_id")
    anchor_box = _compiled_box(anchor_compile, label="compiled anchor AABB")
    probe_box = _compiled_box(probe_compile, label="compiled probe AABB")
    volume = _positive_intersection_volume(anchor_box, probe_box)
    declaration = str(record["region"]["setup_anchor"]["relationship_to_probe"])
    same_part = anchor_part == probe_part
    derived_basis_status = (
        "different_grounded_parts" if not same_part
        else "same_part_zero_positive_intersection" if volume == 0
        else "plan_authored_overlap_exception" if declaration == "overlap_exception"
        else "rejected_positive_overlap"
    )
    validation_outcome = (
        "validated_distinct"
        if derived_basis_status in {"different_grounded_parts", "same_part_zero_positive_intersection"}
        else "exception_pending_concern_discharge"
        if derived_basis_status == "plan_authored_overlap_exception"
        else "rejected_revision_required"
    )
    owner_snapshot = _v2_owner_snapshot(
        state, region_id=region_id, episode_id=episode_id,
        current_id=str(probe_compile["compile_id"]), current_kind="probe_compile_id",
    )
    owner_snapshot.update({"anchor_compile_id": anchor_compile["compile_id"], "probe_compile_id": probe_compile["compile_id"]})
    existing = [
        item for item in diagnostics["anchor_probe_separation_results"]
        if isinstance(item, Mapping)
        and item.get("region_id") == region_id
        and item.get("episode_id") == episode_id
        and item.get("anchor_compile_id") == anchor_compile["compile_id"]
        and item.get("probe_compile_id") == probe_compile["compile_id"]
        and item.get("active_revision_id") == str(state.get("active_revision") or "")
    ]
    if existing:
        if len(existing) != 1:
            raise GenesisVlmSchemaError("v2 separation pair has duplicate immutable records")
        _require_canonical_anchor_probe_separation(existing[0])
        return dict(existing[0])
    result = {"separation_id": f"separation_{len(diagnostics['anchor_probe_separation_results']) + 1:04d}", "region_id": region_id,
              "episode_id": episode_id, "anchor_compile_id": anchor_compile["compile_id"], "probe_compile_id": probe_compile["compile_id"],
              "anchor_part_id": anchor_part, "probe_part_id": probe_part, "anchor_aabb": anchor_box, "probe_aabb": probe_box,
              "positive_intersection_volume": volume, "relationship_to_probe": declaration,
              "derived_basis_status": derived_basis_status,
              "validation_outcome": validation_outcome,
              # Internal projection only.  v2 readers never accept it as a
              # fallback or source of authority.
              "outcome": validation_outcome,
              "active_revision_id": str(state.get("active_revision") or ""), "owner_snapshot": owner_snapshot}
    diagnostics["anchor_probe_separation_results"].append(result)
    return result


def _require_canonical_anchor_probe_separation(separation: Mapping[str, Any]) -> None:
    """Reject a v2 separation record unless its server-derived facts agree."""
    required = (
        "region_id", "episode_id", "anchor_compile_id", "probe_compile_id",
        "anchor_part_id", "probe_part_id", "anchor_aabb", "probe_aabb",
        "positive_intersection_volume", "relationship_to_probe",
        "derived_basis_status", "validation_outcome", "owner_snapshot",
    )
    if any(key not in separation for key in required):
        raise GenesisVlmSchemaError("v2 separation is missing canonical basis/outcome lineage")
    anchor_box = _require_finite_vector_values(separation["anchor_aabb"], context="separation.anchor_aabb", length=6)
    probe_box = _require_finite_vector_values(separation["probe_aabb"], context="separation.probe_aabb", length=6)
    if any(anchor_box[axis] >= anchor_box[axis + 3] or probe_box[axis] >= probe_box[axis + 3] for axis in range(3)):
        raise GenesisVlmSchemaError("v2 separation has a degenerate AABB")
    expected_volume = _positive_intersection_volume(anchor_box, probe_box)
    observed_volume = separation["positive_intersection_volume"]
    if (
        isinstance(observed_volume, bool)
        or not isinstance(observed_volume, int | float)
        or not math.isclose(float(observed_volume), expected_volume, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise GenesisVlmSchemaError("v2 separation intersection volume disagrees with its AABBs")
    declaration = separation["relationship_to_probe"]
    if declaration not in {"distinct", "overlap_exception"}:
        raise GenesisVlmSchemaError("v2 separation has an unsupported plan declaration")
    same_part = separation["anchor_part_id"] == separation["probe_part_id"]
    expected_basis = (
        "different_grounded_parts" if not same_part
        else "same_part_zero_positive_intersection" if expected_volume == 0.0
        else "plan_authored_overlap_exception" if declaration == "overlap_exception"
        else "rejected_positive_overlap"
    )
    expected_validation = (
        "validated_distinct"
        if expected_basis in {"different_grounded_parts", "same_part_zero_positive_intersection"}
        else "exception_pending_concern_discharge"
        if expected_basis == "plan_authored_overlap_exception"
        else "rejected_revision_required"
    )
    if separation["derived_basis_status"] != expected_basis or separation["validation_outcome"] != expected_validation:
        raise GenesisVlmSchemaError("v2 separation basis/outcome disagrees with compiled geometry and declaration")


def verify_v2_anchor_probe_separation_records(state: Mapping[str, Any]) -> None:
    """Independently verify compact persisted v2 separation lineage.

    This is deliberately read-only so workspace and trace consumers can
    reject corruption without invoking settlement code or synthesizing state.
    """
    diagnostics = state.get("diagnostics") if isinstance(state.get("diagnostics"), Mapping) else {}
    separations = diagnostics.get("anchor_probe_separation_results", [])
    if not isinstance(separations, list):
        raise GenesisVlmSchemaError("v2 separation ledger must be a list")
    ledger = diagnostics.get("diagnostic_region_ledger", [])
    if not isinstance(ledger, list):
        raise GenesisVlmSchemaError("v2 region ledger must be a list")
    revision = str(state.get("active_revision") or "")
    seen: set[str] = set()
    rejected_compile_ids: set[str] = set()
    for separation in separations:
        if not isinstance(separation, Mapping):
            raise GenesisVlmSchemaError("v2 separation ledger has a malformed record")
        separation_id = str(separation.get("separation_id") or "")
        if not separation_id or separation_id in seen:
            raise GenesisVlmSchemaError("v2 separation ledger has duplicate or missing separation_id")
        seen.add(separation_id)
        _require_canonical_anchor_probe_separation(separation)
        anchor_matches = [
            item for item in diagnostics.get("compiled_anchor_targets", [])
            if isinstance(item, Mapping) and item.get("compile_id") == separation.get("anchor_compile_id")
        ]
        probe_matches = [
            item for item in diagnostics.get("compiled_probe_targets", [])
            if isinstance(item, Mapping) and item.get("compile_id") == separation.get("probe_compile_id")
        ]
        if len(anchor_matches) != 1 or len(probe_matches) != 1:
            raise GenesisVlmSchemaError("v2 separation requires both persisted compiled anchor and probe targets")
        anchor, probe = anchor_matches[0], probe_matches[0]
        region_rows = [
            row for row in ledger
            if isinstance(row, Mapping) and row.get("region_id") == separation.get("region_id")
        ]
        if len(region_rows) != 1 or region_rows[0].get("episode_id") != separation.get("episode_id"):
            raise GenesisVlmSchemaError("v2 separation disagrees with region/episode ownership")
        region = region_rows[0].get("region")
        if (
            not isinstance(region, Mapping)
            or not isinstance(region.get("setup_anchor"), Mapping)
            or region["setup_anchor"].get("relationship_to_probe") != separation.get("relationship_to_probe")
        ):
            raise GenesisVlmSchemaError("v2 separation declaration disagrees with immutable region plan")
        for compiled, prefix in ((anchor, "anchor"), (probe, "probe")):
            compiled_box = _compiled_box(compiled, label=f"persisted {prefix} compiled AABB")
            if (
                compiled.get("selected_part_id") != separation.get(f"{prefix}_part_id")
                or compiled_box != separation.get(f"{prefix}_aabb")
            ):
                raise GenesisVlmSchemaError("v2 separation disagrees with its persisted compiled target")
        owner = separation.get("owner_snapshot")
        if (
            separation.get("active_revision_id") != revision
            or anchor.get("active_revision_id") != revision
            or probe.get("active_revision_id") != revision
            or anchor.get("anchor_id") != separation.get("region_id")
            or probe.get("region_id") != separation.get("region_id")
            or probe.get("simulate_episode_id") != separation.get("episode_id")
            or not isinstance(owner, Mapping)
            or owner.get("active_revision_id") != revision
            or any(
            owner.get(key) != separation.get(key)
            for key in ("region_id", "episode_id", "anchor_compile_id", "probe_compile_id")
            )
        ):
            raise GenesisVlmSchemaError("v2 separation owner snapshot disagrees with identity lineage")
        attachments = [
            item for item in state.get("diagnostic_runtime_attachments", [])
            if isinstance(item, Mapping) and item.get("attachment_id") == owner.get("attachment_id")
        ]
        sessions = [
            item for item in state.get("diagnostic_runtime_sessions", [])
            if isinstance(item, Mapping)
            and item.get("episode_id") == separation.get("episode_id")
            and item.get("attachment_id") == owner.get("attachment_id")
            and item.get("agent_invocation_id") == owner.get("agent_invocation_id")
            and item.get("active_revision_id") == revision
        ]
        if (
            len(attachments) != 1
            or attachments[0].get("agent_invocation_id") != owner.get("agent_invocation_id")
            or len(sessions) != 1
        ):
            raise GenesisVlmSchemaError("v2 separation owner is stale or does not own the live episode")
        if separation.get("validation_outcome") == "rejected_revision_required":
            rejected_compile_ids.add(str(separation["probe_compile_id"]))
    if rejected_compile_ids:
        for attempt in diagnostics.get("diagnostic_probe_attempts", []):
            if isinstance(attempt, Mapping) and str(attempt.get("compile_id")) in rejected_compile_ids:
                raise GenesisVlmSchemaError("rejected v2 separation may not be referenced by a probe attempt")
        for settlement in diagnostics.get("region_settlements", []):
            if not isinstance(settlement, Mapping):
                continue
            lineage = settlement.get("evidence_lineage")
            separation = lineage.get("separation") if isinstance(lineage, Mapping) else None
            if isinstance(separation, Mapping) and str(separation.get("probe_compile_id")) in rejected_compile_ids:
                raise GenesisVlmSchemaError("rejected v2 separation may not be referenced by settlement coverage")
        for row in ledger:
            if isinstance(row, Mapping) and row.get("qualifying_attempt_id"):
                qualifying = [
                    item for item in diagnostics.get("diagnostic_probe_attempts", [])
                    if isinstance(item, Mapping) and item.get("attempt_id") == row.get("qualifying_attempt_id")
                ]
                if len(qualifying) == 1 and str(qualifying[0].get("compile_id")) in rejected_compile_ids:
                    raise GenesisVlmSchemaError("rejected v2 separation may not own a qualifying region pointer")
        for projection in (diagnostics.get("coverage_summary"), diagnostics.get("terminal")):
            if not isinstance(projection, Mapping):
                continue
            serialized = json.dumps(projection, sort_keys=True)
            if any(compile_id in serialized for compile_id in rejected_compile_ids):
                raise GenesisVlmSchemaError("rejected v2 separation may not be referenced by successful coverage or terminal")


def record_diagnostic_episode_concern_disposition(
    run_root: str, region_id: str, episode_id: str, concern_id: str, disposition: str,
    evidence_refs: list[dict[str, Any]], explanation: str,
) -> dict[str, Any]:
    bound = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound)
    diagnostics = _diagnostics(state)
    record = _planned_region(state, region_id)
    if record.get("episode_id") != episode_id or concern_id != _overlap_concern_id(region_id):
        raise GenesisVlmSchemaError("only the exact runtime-owned overlap concern may be disposed")
    concerns = [item for item in diagnostics["episode_concerns"] if isinstance(item, dict) and item.get("concern_id") == concern_id]
    if (
        len(concerns) != 1
        or concerns[0].get("state") != "pending_probe_evidence"
        or concerns[0].get("episode_id") != episode_id
    ):
        raise GenesisVlmSchemaError("overlap concern must be uniquely pending and may transition only once")
    active_revision_id = str(state.get("active_revision") or "")
    if concerns[0].get("active_revision_id") != active_revision_id:
        raise GenesisVlmSchemaError("overlap concern belongs to a stale diagnostic revision")
    if disposition not in {"resolved", "unresolved"} or not isinstance(evidence_refs, list) or not evidence_refs:
        raise GenesisVlmSchemaError("overlap concern disposition requires resolved/unresolved and evidence")
    if not record.get("qualifying_attempt_id"):
        raise GenesisVlmSchemaError("overlap concern may be disposed only after a qualifying candidate")
    attempt = next((item for item in diagnostics["diagnostic_probe_attempts"] if isinstance(item, Mapping) and item.get("attempt_id") == record["qualifying_attempt_id"]), None)
    if (
        not isinstance(attempt, Mapping)
        or attempt.get("region_id") != region_id
        or attempt.get("episode_id") != episode_id
        or attempt.get("active_revision_id") != active_revision_id
    ):
        raise GenesisVlmSchemaError("overlap concern qualifying attempt is stale or foreign")
    index = attempt.get("tool_result_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(diagnostics["tool_results"]):
        raise GenesisVlmSchemaError("overlap concern requires the qualifying post-probe tool result")
    expected_ref = f"tool_result:{index + 1}"
    if not isinstance(evidence_refs, list) or not evidence_refs or any(not isinstance(item, Mapping) or item.get("kind") != "tool_result" or item.get("ref") != expected_ref for item in evidence_refs):
        raise GenesisVlmSchemaError("overlap concern refs must cite exactly the current qualifying post-probe result")
    reflections = [item for item in diagnostics.get("model_reflections", []) if isinstance(item, Mapping)
                   and item.get("phase") == "post_probe" and item.get("episode_id") == episode_id
                   and item.get("region_id") == region_id and item.get("attachment_id") == _current_runtime_attachment(state, diagnostics).get("attachment_id")
                   and item.get("agent_invocation_id") == _current_runtime_attachment(state, diagnostics).get("agent_invocation_id")
                   and item.get("active_revision_id") == str(state.get("active_revision") or "")
                   and any(isinstance(ref, Mapping) and ref.get("kind") == "tool_result" and ref.get("ref") == expected_ref for ref in item.get("artifact_refs", []))]
    if len(reflections) == 1:
        reflection = reflections[0]
        reflection_sequence = reflection.get("ownership_event_sequence_index")
        if not isinstance(reflection_sequence, int):
            raise GenesisVlmSchemaError("overlap concern post_probe reflection lacks ownership event sequence")
        result_event = _material_audit_ownership_event(state, attachment_id=str(reflection["attachment_id"]), tool_result_index=index, episode_id=episode_id, before_sequence_index=reflection_sequence)
        reflection_index = reflection.get("reflection_index")
        if reflection_sequence <= int(result_event["sequence_index"]):
            raise GenesisVlmSchemaError("overlap concern post_probe reflection must follow the qualifying result")
    elif not reflections:
        # Layer-2 keeps healthy post_probe reflections strictly post-close. An
        # overlap exception must nevertheless be resolved before that close
        # can settle coverage, so accept only this narrow pre-close route from
        # the exact same qualifying compiled-probe result.
        attachment = _current_runtime_attachment(state, diagnostics)
        sessions = [
            item for item in state.get("diagnostic_runtime_sessions", [])
            if isinstance(item, Mapping)
            and item.get("attachment_id") == attachment.get("attachment_id")
            and item.get("agent_invocation_id") == attachment.get("agent_invocation_id")
            and item.get("episode_id") == episode_id
            and item.get("lifecycle_state") not in {"closed", "closed_failed"}
        ]
        if len(sessions) != 1:
            raise GenesisVlmSchemaError("pre-close overlap concern disposition requires one current open owned live session")
        result_events = [
            item for item in sessions[0].get("tool_events", [])
            if isinstance(item, Mapping)
            and item.get("tool_result_index") == index
            and item.get("attachment_id") == attachment.get("attachment_id")
            and item.get("episode_id") == episode_id
        ]
        if len(result_events) != 1:
            raise GenesisVlmSchemaError("pre-close overlap concern disposition lacks the exact owned qualifying result event")
        reflection_index = None
    else:
        raise GenesisVlmSchemaError("overlap concern has ambiguous current same-owner post_probe reflections")
    separation = [item for item in diagnostics["anchor_probe_separation_results"] if isinstance(item, Mapping) and item.get("region_id") == region_id and item.get("episode_id") == episode_id and item.get("probe_compile_id") == attempt.get("compile_id")]
    if len(separation) != 1:
        raise GenesisVlmSchemaError("overlap concern requires exactly one current qualifying compile-pair separation")
    _require_canonical_anchor_probe_separation(separation[0])
    separation_owner = separation[0].get("owner_snapshot")
    if (
        separation[0].get("active_revision_id") != active_revision_id
        or not isinstance(separation_owner, Mapping)
        or separation_owner.get("active_revision_id") != active_revision_id
        or concerns[0].get("separation_id") != separation[0].get("separation_id")
        or separation_owner.get("attachment_id") != attempt.get("attachment_id")
        or separation_owner.get("agent_invocation_id") != attempt.get("agent_invocation_id")
    ):
        raise GenesisVlmSchemaError("overlap concern current separation does not match frozen qualifying attempt ownership")
    snapshot = _v2_owner_snapshot(state, region_id=region_id, episode_id=episode_id, current_id=expected_ref, current_kind="post_probe_result_ref")
    snapshot.update({"concern_id": concern_id, "qualifying_attempt_id": attempt["attempt_id"], "qualifying_compile_id": attempt.get("compile_id"), "separation_id": separation[0].get("separation_id"), "reflection_index": reflection_index, "tool_result_index": index})
    concerns[0].update({"state": disposition, "evidence_refs": [{"kind": "tool_result", "ref": expected_ref}], "owner_snapshot": snapshot,
                        "explanation": _require_nonempty_runtime_text(explanation, "overlap concern explanation")})
    _save_diagnostics(bound, state, event="diagnostic_overlap_concern_disposed", note=disposition, detail=concerns[0])
    return {"status": "success", "concern": dict(concerns[0])}
def revise_diagnostic_probe_target(run_root: str, compile_id: str, edit: dict[str, Any]) -> dict[str, Any]:
    """Apply a bounded relative edit to a compiled target and render a new preview."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    context = _ensure_part_grounding_context(state)
    previous = _compiled_probe_target_by_id(diagnostics, compile_id)
    target_id = str(previous["target_id"])
    intent = _probe_target_intent_by_id(diagnostics, target_id)
    active_episode = _active_episode_record(state) if diagnostics.get("active_episode_index") is not None else None
    active_episode_id = str(active_episode["episode_id"]) if active_episode is not None else ""
    owner_episode_id = str(previous.get("simulate_episode_id", "")).strip()
    if owner_episode_id and active_episode_id and owner_episode_id != active_episode_id:
        raise GenesisVlmSchemaError(
            "compiled probe target belongs to another diagnostic episode; "
            f"requested={compile_id}, owner={owner_episode_id}, current={active_episode_id}"
        )
    edit_index = len(diagnostics["probe_target_edit_trials"])
    try:
        _verify_probe_semantic_group_lineage(context, intent)
        validated_edit = validate_diagnostic_probe_target_edit(edit)
        active_compile_id = str(diagnostics.get("active_compiled_probe_target_id", ""))
        if active_compile_id != str(compile_id):
            raise GenesisVlmSchemaError(
                f"probe target edit source compile {compile_id} is stale; active compile is {active_compile_id}"
            )
        edit_evidence_refs = _validate_probe_target_edit_evidence_refs(
            diagnostics,
            target_id=target_id,
            compile_id=compile_id,
            edit=validated_edit,
        )
        locality_required = bool(previous.get("mechanics_locality_required", False))
        if locality_required and str(intent.get("region_hint")) not in {"tip", "edge_band"}:
            raise GenesisVlmSchemaError("mechanics-conditioned paired probe target requires tip or edge_band")
        anchor_snapshot = previous.get("owning_anchor_snapshot") if isinstance(previous.get("owning_anchor_snapshot"), Mapping) else None
        if locality_required and anchor_snapshot is None:
            raise GenesisVlmSchemaError("mechanics-conditioned revision is missing its frozen owning anchor snapshot")
        compiled = compile_probe_target_from_context(
            context,
            selected_part_id=int(intent["selected_part_id"]),
            region_hint=str(intent["region_hint"]),
            edit=validated_edit,
            anchor_pin_box=previous.get("anchor_pin_box") if isinstance(previous.get("anchor_pin_box"), list) else None,
            mechanics_locality_required=locality_required,
            mechanics_probe_mode=str(previous.get("mechanics_probe_mode", intent.get("mechanics_probe_mode", "none"))),
            motion_axis=str(previous.get("motion_axis", intent.get("motion_axis", "+Y"))),
        )
    except Exception as exc:
        rejected = {
            "edit_index": edit_index,
            "status": "rejected",
            "source_compile_id": compile_id,
            "target_id": target_id,
            "error": str(exc),
        }
        diagnostics["probe_target_edit_trials"].append(rejected)
        _save_diagnostics(
            bound_run_root,
            state,
            event="diagnostic_probe_target_edit_rejected",
            note=str(exc),
            detail={"probe_target_edit": rejected},
            provenance="typed_model_probe_target_edit",
        )
        if isinstance(exc, GenesisVlmSchemaError):
            raise
        raise GenesisVlmSchemaError(str(exc)) from exc
    compile_count = sum(
        1
        for item in diagnostics["compiled_probe_targets"]
        if isinstance(item, dict) and str(item.get("target_id")) == target_id
    )
    new_compile_id = f"{target_id}_compile_{compile_count + 1:02d}"
    compiled = {
        **compiled,
        "compile_id": new_compile_id,
        "target_id": target_id,
        "intent_id": target_id,
        "source_compile_id": compile_id,
        "anchor_pin_box": previous.get("anchor_pin_box"),
        "owning_anchor_snapshot": anchor_snapshot,
        "mechanics_probe_mode": previous.get("mechanics_probe_mode", intent.get("mechanics_probe_mode", "none")),
        "mechanics_locality_required": locality_required,
        "motion_axis": previous.get("motion_axis", intent.get("motion_axis", "+Y")),
        "motion_axis_rationale": previous.get("motion_axis_rationale", intent.get("motion_axis_rationale", "")),
        "compile_index": len(diagnostics["compiled_probe_targets"]),
        "active_revision_id": str(state.get("active_revision") or ""),
        "region_id": previous.get("region_id"),
        "edit_evidence_refs": edit_evidence_refs,
        "semantic_group_part_ids": list(intent["semantic_group_part_ids"]),
        "semantic_group_part_grounding": list(intent["semantic_group_part_grounding"]),
    }
    if active_episode_id or owner_episode_id:
        compiled["simulate_episode_id"] = active_episode_id or owner_episode_id
    edit_record = {
        "edit_index": edit_index,
        "status": "accepted",
        "source_compile_id": compile_id,
        "compile_id": new_compile_id,
        "target_id": target_id,
        "edit": validated_edit,
        "resolved_evidence_refs": edit_evidence_refs,
    }
    diagnostics["probe_target_edit_trials"].append(edit_record)
    diagnostics["compiled_probe_targets"].append(compiled)
    diagnostics["active_compiled_probe_target_id"] = new_compile_id
    if previous.get("region_id"):
        diagnostics["region_current_probe_compile_ids"][str(previous["region_id"])] = new_compile_id
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_probe_target_revised",
        note=f"revised probe target {new_compile_id}",
        detail={"compiled_probe_target": _compiled_probe_target_summary(compiled), "probe_target_edit": edit_record},
        provenance="typed_model_probe_target_edit",
    )
    return _render_and_record_probe_target_preview(run_root=bound_run_root, state=state, compile_id=new_compile_id)
def record_diagnostic_pre_episode_plan(
    run_root: str,
    episode_intent: str,
    pinning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record and validate a Genesis diagnostic pre-episode plan."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    plan = validate_vlm_pre_episode_plan(
        {
            "schema_version": VLM_PRE_EPISODE_SCHEMA_VERSION,
            "episode_intent": episode_intent,
            "pinning": pinning or {"enabled": False, "boxes": []},
        }
    )
    diagnostics["pre_episode_plans"].append(plan)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_pre_episode_plan_recorded",
        note=plan["episode_intent"],
        detail=plan,
    )
    return {"status": "success", "plan": plan, "state_path": str(state_path(bound_run_root))}
def _require_json_object(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenesisVlmSchemaError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value
def _pin_boxes_from_pinning(pinning: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    if not isinstance(pinning, dict) or not pinning.get("enabled"):
        return ()
    boxes = pinning.get("boxes", ())
    if not isinstance(boxes, list | tuple):
        raise GenesisVlmSchemaError("pinning.boxes must be a list when pinning is enabled")
    return tuple(
        {
            key: box[key]
            for key in ("anchor_id", "name", "obj_id", "box", "reason", "source")
            if key in box
        }
        for box in boxes
        if isinstance(box, dict)
    )
def _active_session_anchor(state: dict[str, Any], region_id: str) -> dict[str, Any]:
    """Compatibility-shaped view used by the existing anchor compiler; data stays v2 owned."""
    record = _planned_region(state, region_id)
    region = record["region"]
    setup = region["setup_anchor"]
    return {
        "anchor_id": region["region_id"], "name": region["name"],
        "anchor_type": setup["anchor_type"], "semantic_region": setup["anchor_region"],
        "physical_hypothesis": region["physical_hypothesis"], "episode_intent": region["episode_intent"],
        "termination_condition": region["termination_condition"], "uncertainty": setup["uncertainty"],
        "concerns": [concern["summary"] for concern in setup["concerns"]],
        "region_id": region["region_id"], "setup_anchor": setup,
    }
def _anchor_box_from_semantics(state: dict[str, Any], anchor: dict[str, Any]) -> list[float]:
    geometry = diagnostic_asset_geometry_payload(Path(str(state["paths"]["monolithic_mesh_path"])))
    bounds = geometry.get("raw_bounds") if isinstance(geometry, dict) else {}
    if not isinstance(bounds, dict) or not bounds.get("available"):
        raise GenesisVlmSchemaError("cannot resolve semantic anchor because asset mesh bounds are unavailable")
    mins = [float(value) for value in bounds["min"]]
    maxs = [float(value) for value in bounds["max"]]
    if len(mins) != 3 or len(maxs) != 3 or any(left >= right for left, right in zip(mins, maxs)):
        raise GenesisVlmSchemaError("cannot resolve semantic anchor because asset mesh bounds are degenerate")
    text = " ".join(str(anchor.get(key, "")) for key in ("anchor_id", "name", "semantic_region")).lower()
    extents = [right - left for left, right in zip(mins, maxs)]
    main_axis = max(range(3), key=lambda axis: extents[axis])
    box_min = list(mins)
    box_max = list(maxs)
    slab = 0.25
    def apply_slab(axis: int, side: str) -> None:
        if side == "min":
            box_max[axis] = mins[axis] + extents[axis] * slab
        elif side == "max":
            box_min[axis] = maxs[axis] - extents[axis] * slab
        else:
            raise AssertionError(f"unsupported semantic slab side: {side}")
    if any(token in text for token in ("lower", "below", "base", "bottom", "foot", "support", "stand")):
        box_max[1] = mins[1] + extents[1] * slab
    elif any(token in text for token in ("top", "upper")):
        box_min[1] = maxs[1] - extents[1] * slab
    elif (
        anchor.get("anchor_type") == "joint_hinge"
        or any(token in text for token in ("joint", "hinge", "bend", "u-shaped", "root"))
    ):
        apply_slab(main_axis, "max")
    elif "left" in text:
        box_max[0] = mins[0] + extents[0] * slab
    elif "right" in text:
        box_min[0] = maxs[0] - extents[0] * slab
    elif "front" in text:
        box_max[2] = mins[2] + extents[2] * slab
    elif "back" in text or "rear" in text:
        box_min[2] = maxs[2] - extents[2] * slab
    else:
        for axis in range(3):
            center = (mins[axis] + maxs[axis]) * 0.5
            half = extents[axis] * 0.125
            box_min[axis] = center - half
            box_max[axis] = center + half
    return [*box_min, *box_max]
def _anchor_for_setup_revision(anchor: dict[str, Any], revision_request: str | None) -> dict[str, Any]:
    if not revision_request:
        return anchor
    revised = dict(anchor)
    revised["semantic_region"] = (
        f"{anchor['semantic_region']} Setup revision request for this trial: {revision_request}"
    )
    return revised
def _pinning_from_anchor_region(state: dict[str, Any], anchor: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    box = _anchor_box_from_semantics(state, anchor)
    pin_box = {
        "name": anchor["name"],
        "obj_id": 0,
        "box": box,
        "reason": anchor["physical_hypothesis"],
        "source": f"diagnostic_session_anchor:{anchor['anchor_id']}",
        "anchor_id": anchor["anchor_id"],
        "anchor_type": anchor["anchor_type"],
        "semantic_region": anchor["semantic_region"],
        "uncertainty": anchor["uncertainty"],
        "concerns": list(anchor["concerns"]),
    }
    return anchor, {"enabled": True, "boxes": [pin_box]}
def _next_runtime_event_id(state: dict[str, Any]) -> int:
    diagnostics = _diagnostics(state)
    events = diagnostics.get("runtime_owned_events", [])
    previous_ids = [int(item.get("event_id", 0)) for item in events if isinstance(item, dict)]
    return (max(previous_ids) if previous_ids else 0) + 1
def _setup_trials_for_anchor(diagnostics: dict[str, Any], anchor_id: str) -> list[dict[str, Any]]:
    return [
        trial
        for trial in diagnostics.get("setup_trials", [])
        if isinstance(trial, dict) and str(trial.get("anchor_id")) == anchor_id
    ]
def _setup_trial_by_id(diagnostics: dict[str, Any], trial_id: str) -> dict[str, Any]:
    for trial in diagnostics.get("setup_trials", []):
        if isinstance(trial, dict) and str(trial.get("trial_id")) == trial_id:
            return trial
    raise GenesisVlmSchemaError(f"diagnostic setup trial is not recorded: {trial_id}")
def _validate_semantic_setup_revision_request(revision_request: str | None) -> str | None:
    if revision_request is None:
        return None
    text = str(revision_request).strip()
    if not text:
        return None
    lowered = text.lower()
    forbidden_terms = (
        "pinning",
        "boxes",
        " box",
        "box:",
        "box=",
        "scene_json",
        "model_deformable_json",
        "generated_asset_json",
        "controllers",
        "probe",
        "resume_simulation",
        "pause_and_observe",
    )
    if any(term in lowered for term in forbidden_terms) or re.search(r"\bbox\b", lowered):
        raise GenesisVlmSchemaError(
            "diagnostic setup revision_request must be semantic-only; raw runtime setup, pins, probes, "
            "controllers, and Genesis JSON are runtime-owned"
        )
    if "{" in text or "}" in text:
        raise GenesisVlmSchemaError("diagnostic setup revision_request must not contain JSON object payloads")
    if re.search(r"\[[\s\d.,+\-eE]{12,}\]", text):
        raise GenesisVlmSchemaError("diagnostic setup revision_request must not contain coordinate-list payloads")
    return text
def _validate_setup_reflection(
    *,
    verdict: str,
    concerns: list[str],
    uncertainty: str,
    revision_request: str | None,
    trial_index: int,
) -> dict[str, Any]:
    normalized_verdict = str(verdict)
    if normalized_verdict not in DIAGNOSTIC_SETUP_VERDICTS:
        raise GenesisVlmSchemaError(
            f"diagnostic setup verdict must be one of {sorted(DIAGNOSTIC_SETUP_VERDICTS)}, got {normalized_verdict}"
        )
    normalized_uncertainty = str(uncertainty)
    if normalized_uncertainty not in DIAGNOSTIC_SETUP_UNCERTAINTY:
        raise GenesisVlmSchemaError(
            f"diagnostic setup uncertainty must be one of {sorted(DIAGNOSTIC_SETUP_UNCERTAINTY)}, got {normalized_uncertainty}"
        )
    if not isinstance(concerns, list):
        raise GenesisVlmSchemaError("diagnostic setup concerns must be a list of strings")
    normalized_concerns: list[str] = []
    for index, item in enumerate(concerns):
        if not isinstance(item, str):
            raise GenesisVlmSchemaError(f"diagnostic setup concerns[{index}] must be a string")
        text = item.strip()
        if text:
            normalized_concerns.append(text)
    normalized_revision = _validate_semantic_setup_revision_request(revision_request)
    if normalized_verdict == "revise_setup" and trial_index < MAX_DIAGNOSTIC_SETUP_TRIALS and not normalized_revision:
        raise GenesisVlmSchemaError("revise_setup requires a semantic revision_request before the third setup trial")
    return {
        "verdict": normalized_verdict,
        "concerns": normalized_concerns,
        "uncertainty": normalized_uncertainty,
        "revision_request": normalized_revision,
    }
def _bbox_overlap_volume(left: list[float], right: list[float]) -> tuple[list[float], float]:
    overlap_min = [max(float(left[axis]), float(right[axis])) for axis in range(3)]
    overlap_max = [min(float(left[axis + 3]), float(right[axis + 3])) for axis in range(3)]
    extents = [overlap_max[axis] - overlap_min[axis] for axis in range(3)]
    volume = extents[0] * extents[1] * extents[2] if all(extent > 0.0 for extent in extents) else 0.0
    return [*overlap_min, *overlap_max], volume
def _bbox_volume(box: list[float]) -> float:
    extents = [float(box[axis + 3]) - float(box[axis]) for axis in range(3)]
    return extents[0] * extents[1] * extents[2] if all(extent > 0.0 for extent in extents) else 0.0
def _bbox_span_ratios(box: list[float], outer_box: list[float]) -> list[float]:
    ratios: list[float] = []
    for axis in range(3):
        outer_span = float(outer_box[axis + 3]) - float(outer_box[axis])
        if outer_span <= 0.0:
            raise GenesisVlmSchemaError("cannot compute bbox span ratios against a degenerate outer box")
        ratios.append((float(box[axis + 3]) - float(box[axis])) / outer_span)
    return ratios
def _bbox_corners(box: list[float]) -> list[tuple[float, float, float]]:
    mins = [float(value) for value in box[:3]]
    maxs = [float(value) for value in box[3:]]
    return [
        (x, y, z)
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]
def _validate_local_to_world_transform(transform: Any) -> list[list[float]]:
    if not isinstance(transform, list | tuple) or len(transform) != 4:
        raise GenesisVlmSchemaError("geometry_context.local_to_world_transform must be a finite 4x4 matrix")
    normalized: list[list[float]] = []
    for row_index, row in enumerate(transform):
        if not isinstance(row, list | tuple) or len(row) != 4:
            raise GenesisVlmSchemaError(
                f"geometry_context.local_to_world_transform[{row_index}] must be a finite 4-vector"
            )
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in row):
            raise GenesisVlmSchemaError("geometry_context.local_to_world_transform must contain only finite numbers")
        normalized_row = [float(value) for value in row]
        if any(not math.isfinite(value) for value in normalized_row):
            raise GenesisVlmSchemaError("geometry_context.local_to_world_transform must contain only finite numbers")
        normalized.append(normalized_row)
    return normalized
def _transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    values = [x, y, z, 1.0]
    transformed = [sum(matrix[row][col] * values[col] for col in range(4)) for row in range(4)]
    if any(not math.isfinite(value) for value in transformed):
        raise GenesisVlmSchemaError("local_to_world_transform produced a non-finite point")
    w = transformed[3]
    if w == 0.0:
        raise GenesisVlmSchemaError("local_to_world_transform produced a homogeneous point with w=0")
    return transformed[0] / w, transformed[1] / w, transformed[2] / w
def _transform_local_aabb_to_world_aabb(local_box: list[float], transform: list[list[float]]) -> list[float]:
    transformed = [_transform_point(transform, corner) for corner in _bbox_corners(local_box)]
    mins = [min(point[axis] for point in transformed) for axis in range(3)]
    maxs = [max(point[axis] for point in transformed) for axis in range(3)]
    if any(left >= right for left, right in zip(mins, maxs)):
        raise GenesisVlmSchemaError("local_to_world_transform produced a degenerate executable pin box")
    return [*mins, *maxs]
def _anchor_setup_soft_warnings(
    *,
    anchor: dict[str, Any],
    compiled: Mapping[str, Any],
    local_box: list[float],
    local_mesh_bbox: list[float],
    local_anchor_volume: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    metrics = compiled.get("metrics") if isinstance(compiled.get("metrics"), Mapping) else {}
    primitive_purity = metrics.get("primitive_purity") if isinstance(metrics.get("primitive_purity"), Mapping) else {}
    selected_part_fraction = primitive_purity.get("selected_part_fraction")
    if isinstance(selected_part_fraction, int | float) and float(selected_part_fraction) < 0.5:
        warnings.append(
            {
                "code": "low_primitive_purity",
                "message": "compiled anchor target includes a low fraction of selected-part primitives",
                "primitive_purity": float(selected_part_fraction),
            }
        )
    mesh_volume = _bbox_volume(local_mesh_bbox)
    volume_ratio = local_anchor_volume / mesh_volume if mesh_volume > 0.0 else 0.0
    span_ratios = _bbox_span_ratios(local_box, local_mesh_bbox)
    mesh_height = float(local_mesh_bbox[4]) - float(local_mesh_bbox[1])
    box_height = float(local_box[4]) - float(local_box[1])
    lower_threshold = float(local_mesh_bbox[1]) + 0.25 * mesh_height
    if (
        float(local_box[1]) <= lower_threshold
        and box_height <= 0.35 * mesh_height
        and span_ratios[0] >= 0.85
        and span_ratios[2] >= 0.85
    ):
        warnings.append(
            {
                "code": "broad_lower_slab",
                "message": "compiled anchor target overlaps a broad lower slab; this is audit-only for support setup",
                "span_ratios": span_ratios,
            }
        )
    if volume_ratio >= 0.5:
        warnings.append(
            {
                "code": "high_anchor_volume_coverage",
                "message": "compiled anchor target covers a large fraction of the whole mesh bbox",
                "volume_ratio": volume_ratio,
            }
        )
    if (
        str(anchor.get("anchor_type", "")) in {"grip_root", "joint_hinge"}
        and sum(1 for ratio in span_ratios if ratio >= 0.85) >= 2
    ):
        warnings.append(
            {
                "code": "broad_semantic_local_anchor",
                "message": "local anchor type usually expects a narrower compiled setup target",
                "span_ratios": span_ratios,
            }
        )
    return warnings
def _selected_vertex_count(compiled: Mapping[str, Any]) -> int:
    selected_vertices = compiled.get("selected_vertex_indices")
    if isinstance(selected_vertices, list):
        return len(selected_vertices)
    value = compiled.get("selected_vertex_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0
def _mesh_vertex_indices_inside_aabb(mesh_path: Path, box: list[float]) -> list[int]:
    normalized_box = _require_finite_vector_values(box, context="serialized static anchor env_local box", length=6)
    if any(normalized_box[axis] > normalized_box[axis + 3] for axis in range(3)):
        raise GenesisVlmSchemaError("serialized static anchor env_local box min values must be <= max values")
    geometry = read_mesh_vertex_geometry(mesh_path)
    if geometry is None or not geometry.vertices:
        raise GenesisVlmSchemaError(f"cannot validate serialized static anchor box because mesh vertices are unavailable: {mesh_path}")
    selected: list[int] = []
    for index, vertex in enumerate(geometry.vertices):
        if all(normalized_box[axis] <= float(vertex[axis]) <= normalized_box[axis + 3] for axis in range(3)):
            selected.append(index)
    return selected
def _anchor_pinned_vertex_distribution_summary(
    setup_validation: dict[str, Any],
    live_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if live_result is None:
        selected = setup_validation.get("serialized_static_anchor_vertex_indices")
        if isinstance(selected, list):
            return {
                "available": True,
                "total_pinned_vertices": len(selected),
                "selected_part_pinned_vertices": None,
                "selected_part_fraction": None,
                "data_source_key": "raw_mesh_static_anchor_preflight",
                "frame": "env_local",
                "source_frame": "local_mesh",
            }
        return {"available": False}
    data = live_result.get("result") if isinstance(live_result.get("result"), Mapping) else live_result
    if not isinstance(data, Mapping):
        return {"available": False}
    for key in ("pinned_object_local_vertices", "resolved_object_local_vertices", "affected_object_local_vertices"):
        vertices = data.get(key)
        if isinstance(vertices, list):
            return {
                "available": True,
                "total_pinned_vertices": len(vertices),
                "selected_part_pinned_vertices": None,
                "selected_part_fraction": None,
                "data_source_key": key,
            }
    return {"available": False}
def _validate_compiled_anchor_for_setup(
    state: dict[str, Any],
    *,
    anchor: dict[str, Any],
    compiled: dict[str, Any],
    preview_evidence: dict[str, Any],
) -> dict[str, Any]:
    if "mesh_frame_box_m" not in compiled:
        raise GenesisVlmSchemaError("compiled anchor target is missing mesh_frame_box_m")
    local_anchor_box = _require_finite_vector_values(
        compiled["mesh_frame_box_m"],
        context="compiled_anchor_target.mesh_frame_box_m",
        length=6,
    )
    if any(local_anchor_box[axis] >= local_anchor_box[axis + 3] for axis in range(3)):
        raise GenesisVlmSchemaError("compiled_anchor_target.mesh_frame_box_m min values must be strictly less than max values")
    extents = [local_anchor_box[axis + 3] - local_anchor_box[axis] for axis in range(3)]
    if any(extent <= 1.0e-9 for extent in extents):
        raise GenesisVlmSchemaError("compiled_anchor_target.mesh_frame_box_m must have nonzero thickness on all axes")
    selected_primitive_count = int(compiled.get("selected_primitive_count", 0))
    selected_vertex_count = _selected_vertex_count(compiled)
    if selected_primitive_count <= 0:
        raise GenesisVlmSchemaError("compiled anchor target selected_primitive_count must be positive")
    if selected_vertex_count <= 0:
        raise GenesisVlmSchemaError("compiled anchor target selected_vertex_count must be positive")
    geometry_context = _geometry_context_from_state(state)
    transform = _validate_local_to_world_transform(geometry_context["local_to_world_transform"])
    local_mesh_bbox = [float(value) for value in geometry_context["local_mesh_bbox"]]
    local_mesh_overlap_box, local_mesh_overlap_volume = _bbox_overlap_volume(local_anchor_box, local_mesh_bbox)
    if local_mesh_overlap_volume <= 0.0:
        raise GenesisVlmSchemaError("compiled anchor target mesh_frame_box_m must overlap local_mesh_bbox")
    local_anchor_volume = _bbox_volume(local_anchor_box)
    if local_anchor_volume <= 0.0:
        raise GenesisVlmSchemaError("compiled anchor target mesh_frame_box_m volume must be positive")
    transformed_scene_fit_anchor_box = _transform_local_aabb_to_world_aabb(local_anchor_box, transform)
    world_mesh_bbox = _transform_local_aabb_to_world_aabb(local_mesh_bbox, transform)
    world_mesh_overlap_box, world_mesh_overlap_volume = _bbox_overlap_volume(transformed_scene_fit_anchor_box, world_mesh_bbox)
    if world_mesh_overlap_volume <= 0.0:
        raise GenesisVlmSchemaError("compiled anchor target executable pin box must overlap transformed world mesh bbox")
    executable_pin_box = list(local_anchor_box)
    raw_mesh_vertex_indices = _mesh_vertex_indices_inside_aabb(_active_diagnostic_mesh_path(state), executable_pin_box)
    if not raw_mesh_vertex_indices:
        raise GenesisVlmSchemaError("serialized static anchor env_local box selected no raw mesh vertices")
    metrics = compiled.get("metrics") if isinstance(compiled.get("metrics"), Mapping) else {}
    primitive_purity = metrics.get("primitive_purity") if isinstance(metrics.get("primitive_purity"), Mapping) else {}
    warnings = _anchor_setup_soft_warnings(
        anchor=anchor,
        compiled=compiled,
        local_box=local_anchor_box,
        local_mesh_bbox=local_mesh_bbox,
        local_anchor_volume=local_anchor_volume,
    )
    warning_codes = [str(warning["code"]) for warning in warnings]
    validation = {
        "schema_version": DIAGNOSTIC_ANCHOR_SETUP_VALIDATION_SCHEMA_VERSION,
        "status": "accepted_for_preview",
        "anchor_id": str(compiled.get("anchor_id", "")),
        "compile_id": str(compiled.get("compile_id", "")),
        "selected_part_id": compiled.get("selected_part_id"),
        "region_hint": str(compiled.get("region_hint", "")),
        "local_anchor_box": local_anchor_box,
        "local_mesh_bbox": local_mesh_bbox,
        "local_mesh_overlap_box": local_mesh_overlap_box,
        "local_mesh_overlap_volume": local_mesh_overlap_volume,
        "local_anchor_volume": local_anchor_volume,
        "local_to_world_transform": transform,
        "world_mesh_bbox": world_mesh_bbox,
        "executable_pin_box": executable_pin_box,
        "executable_frame": "genesis_env_local",
        "serialized_static_anchor_frame": "env_local",
        "serialized_static_anchor_box": list(executable_pin_box),
        "serialized_static_anchor_source_frame": "local_mesh",
        "scene_encoded_mesh_transform": False,
        "transformed_scene_fit_anchor_box": transformed_scene_fit_anchor_box,
        "transformed_scene_fit_anchor_usage": "audit_only",
        "serialized_static_anchor_vertex_indices": raw_mesh_vertex_indices,
        "serialized_static_anchor_vertex_count": len(raw_mesh_vertex_indices),
        "serialized_static_anchor_vertex_count_positive": True,
        "world_mesh_overlap_box": world_mesh_overlap_box,
        "world_mesh_overlap_volume": world_mesh_overlap_volume,
        "preview_evidence_id": str(preview_evidence["evidence_id"]),
        "preview_source_kind": str(preview_evidence["source_kind"]),
        "preview_panel_order": list(preview_evidence["panel_order"]),
        "metrics": {
            "primitive_purity": primitive_purity.get("selected_part_fraction"),
            "target_volume_m3": metrics.get("target_volume_m3"),
            "selected_part_bbox_volume_m3": metrics.get("selected_part_bbox_volume_m3"),
            "target_to_part_volume_ratio": metrics.get("target_to_part_volume_ratio"),
            "selected_primitive_count": selected_primitive_count,
            "selected_vertex_count": selected_vertex_count,
        },
        "hard_checks": {
            "finite_local_box_values": True,
            "strict_min_less_than_max": True,
            "positive_local_volume": True,
            "minimum_thickness": True,
            "selected_primitive_count_positive": True,
            "selected_vertex_count_positive": True,
            "local_mesh_overlap_positive": True,
            "world_mesh_overlap_positive": True,
            "valid_transform": True,
            "serialized_static_anchor_vertex_count_positive": True,
            "required_static_anchor_preview_exists": True,
        },
        "warning_codes": warning_codes,
        "warnings": warnings,
    }
    validation["pinned_vertex_distribution"] = _anchor_pinned_vertex_distribution_summary(validation)
    return validation
def _pinning_from_compiled_anchor_setup(
    anchor: dict[str, Any],
    compiled: dict[str, Any],
    setup_validation: dict[str, Any],
) -> dict[str, Any]:
    intent_reason = str(compiled.get("anchor_intent", "")).strip()
    boundary_reason = str(compiled.get("physical_boundary_condition", "")).strip()
    reason = boundary_reason or intent_reason or str(anchor["physical_hypothesis"])
    return {
        "enabled": True,
        "boxes": [
            {
                "name": anchor["name"],
                "obj_id": 0,
                "box": list(setup_validation["executable_pin_box"]),
                "reason": reason,
                "source": f"diagnostic_compiled_anchor_target:{compiled['compile_id']}",
                "anchor_id": anchor["anchor_id"],
                "anchor_type": anchor["anchor_type"],
                "semantic_region": anchor["semantic_region"],
                "uncertainty": anchor["uncertainty"],
                "concerns": list(anchor["concerns"]),
                "selected_part_id": compiled.get("selected_part_id"),
                "region_hint": str(compiled.get("region_hint", "")),
                "compile_id": str(compiled["compile_id"]),
                "executable_frame": "genesis_env_local",
            }
        ],
    }
def _setup_quality_from_trial(trial: dict[str, Any]) -> dict[str, Any]:
    reflection = trial.get("reflection") if isinstance(trial.get("reflection"), dict) else {}
    artifacts = trial.get("preview_artifacts") if isinstance(trial.get("preview_artifacts"), dict) else {}
    return {
        "final_status": str(trial.get("final_status", "")),
        "anchor_semantic_match": reflection.get("anchor_semantic_match"),
        "tested_dof_preservation": reflection.get("tested_dof_preservation"),
        "setup_concern_dispositions": reflection.get("setup_concern_dispositions", []),
        "preview_png_path": str(artifacts.get("png_path", "")),
    }
def _pinning_from_setup_trial(trial: dict[str, Any]) -> dict[str, Any]:
    pinning = trial.get("derived_pinning")
    if not isinstance(pinning, dict) or not pinning.get("enabled"):
        raise GenesisVlmSchemaError(f"diagnostic setup trial has no derived pinning: {trial.get('trial_id', '')}")
    boxes = pinning.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise GenesisVlmSchemaError(f"diagnostic setup trial has no derived pinning boxes: {trial.get('trial_id', '')}")
    return {
        "enabled": True,
        "boxes": [dict(box) for box in boxes if isinstance(box, dict)],
    }
def _select_setup_trial_for_episode(diagnostics: dict[str, Any], anchor_id: str) -> dict[str, Any]:
    trials = _setup_trials_for_anchor(diagnostics, anchor_id)
    if not trials:
        raise GenesisVlmSchemaError(
            "define_episode requires setup preview first: call "
            "preview_diagnostic_episode_setup and record_diagnostic_setup_reflection before live episode creation"
        )
    def usable(trial: dict[str, Any]) -> bool:
        reflection = trial.get("reflection") if isinstance(trial.get("reflection"), dict) else {}
        runtime_overlap_id = _overlap_concern_id(str(trial.get("region_id") or anchor_id))
        exception = any(
            isinstance(item, Mapping)
            and item.get("concern_id") == runtime_overlap_id
            and item.get("state") == "setup_acknowledged"
            and item.get("setup_trial_id") == trial.get("trial_id")
            for item in diagnostics.get("episode_concerns", [])
        )
        setup_dispositions = reflection.get("setup_concern_dispositions", [])
        plan_concerns_resolved = all(
            item.get("disposition") == "resolved"
            for item in setup_dispositions
            if isinstance(item, Mapping) and item.get("concern_id") != runtime_overlap_id
        )
        overlap_ack = any(
            isinstance(item, Mapping)
            and item.get("concern_id") == runtime_overlap_id
            and item.get("disposition") == "acknowledged"
            for item in setup_dispositions
        )
        return (
            bool(trial.get("anchor_compile_id"))
            and isinstance(trial.get("anchor_setup_validation"), dict)
            and isinstance(trial.get("derived_pinning"), dict)
            and reflection.get("anchor_semantic_match") == "matched"
            and reflection.get("tested_dof_preservation") == "preserved"
            and plan_concerns_resolved
            and ((exception and overlap_ack) or (not exception and not overlap_ack))
        )
    accepted = [
        trial
        for trial in trials
        if usable(trial)
        and (
            trial.get("final_status") == "accepted"
        )
    ]
    if accepted:
        return accepted[-1]
    final_by_anchor = diagnostics.setdefault("final_setup_trial_by_anchor", {})
    final_trial_id = final_by_anchor.get(anchor_id) if isinstance(final_by_anchor, dict) else None
    if final_trial_id:
        trial = _setup_trial_by_id(diagnostics, str(final_trial_id))
        if usable(trial):
            return trial
    raise GenesisVlmSchemaError(
        "define_episode requires setup reflection first: call "
        "record_diagnostic_setup_reflection, or preview/revise up to three setup trials before live episode creation"
    )
def _setup_bounds_from_geometry(state: dict[str, Any]) -> tuple[list[float], list[float]]:
    geometry = diagnostic_asset_geometry_payload(Path(str(state["paths"]["monolithic_mesh_path"])))
    bounds = geometry.get("raw_bounds") if isinstance(geometry, dict) else {}
    if not isinstance(bounds, dict) or not bounds.get("available"):
        raise GenesisVlmSchemaError("cannot render diagnostic setup preview because asset mesh bounds are unavailable")
    mins = [float(value) for value in bounds.get("min", [])]
    maxs = [float(value) for value in bounds.get("max", [])]
    if (
        len(mins) != 3
        or len(maxs) != 3
        or any(not math.isfinite(value) for value in (*mins, *maxs))
        or any(left >= right for left, right in zip(mins, maxs))
    ):
        raise GenesisVlmSchemaError("cannot render diagnostic setup preview because asset mesh bounds are degenerate")
    return mins, maxs
def _project_setup_rect(
    *,
    box: list[float],
    axes: tuple[int, int],
    panel: tuple[int, int, int, int],
    mins: list[float],
    maxs: list[float],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = panel
    pad = 18
    plot_left = left + pad
    plot_top = top + pad
    plot_right = right - pad
    plot_bottom = bottom - pad
    axis_a, axis_b = axes
    span_a = maxs[axis_a] - mins[axis_a]
    span_b = maxs[axis_b] - mins[axis_b]
    def px(value: float) -> int:
        return int(round(plot_left + ((value - mins[axis_a]) / span_a) * (plot_right - plot_left)))
    def py(value: float) -> int:
        return int(round(plot_bottom - ((value - mins[axis_b]) / span_b) * (plot_bottom - plot_top)))
    box_mins = box[:3]
    box_maxs = box[3:]
    x0 = px(float(box_mins[axis_a]))
    x1 = px(float(box_maxs[axis_a]))
    y0 = py(float(box_maxs[axis_b]))
    y1 = py(float(box_mins[axis_b]))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
def _write_setup_preview_image(
    state: dict[str, Any],
    *,
    trial: dict[str, Any],
) -> Path:
    mins, maxs = _setup_bounds_from_geometry(state)
    setup_validation = trial.get("anchor_setup_validation")
    if not isinstance(setup_validation, dict):
        raise GenesisVlmSchemaError("cannot render diagnostic setup preview without anchor setup validation")
    pin_box = setup_validation.get("local_anchor_box")
    if (
        not isinstance(pin_box, list)
        or len(pin_box) != 6
        or any(not isinstance(value, int | float) or not math.isfinite(float(value)) for value in pin_box)
    ):
        raise GenesisVlmSchemaError("cannot render diagnostic setup preview because local anchor box is invalid")
    preview_dir = _diagnostic_payload_dir(state) / "setup_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / f"{trial['trial_id']}.png"
    image = Image.new("RGB", (960, 360), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    panels = [
        ("XY/front", (0, 1), (24, 48, 304, 328)),
        ("XZ/top", (0, 2), (340, 48, 620, 328)),
        ("YZ/side", (1, 2), (656, 48, 936, 328)),
    ]
    title = f"{trial['anchor_id']} ({trial['anchor_region']['anchor_type']}) trial {trial['trial_index']}"
    draw.text((24, 18), title, fill=(25, 30, 36))
    for label, axes, panel in panels:
        left, top, right, bottom = panel
        draw.rectangle(panel, fill=(255, 255, 255), outline=(206, 212, 218), width=1)
        object_rect = _project_setup_rect(box=[*mins, *maxs], axes=axes, panel=panel, mins=mins, maxs=maxs)
        setup_rect = _project_setup_rect(box=[float(value) for value in pin_box], axes=axes, panel=panel, mins=mins, maxs=maxs)
        draw.rectangle(object_rect, outline=(90, 98, 108), width=2)
        draw.rectangle(setup_rect, outline=(205, 56, 56), width=3)
        draw.text((left + 10, top + 8), label, fill=(25, 30, 36))
        draw.text((left + 10, bottom - 22), "gray: object  red: setup", fill=(73, 80, 87))
    image.save(path, format="PNG")
    return path
def _create_setup_trial(
    state: dict[str, Any],
    *,
    anchor: dict[str, Any],
    compiled: dict[str, Any],
    preview_evidence: dict[str, Any],
    anchor_setup_validation: dict[str, Any],
    revision_request: str | None,
) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    anchor_id = str(anchor["anchor_id"])
    trial_index = len(_setup_trials_for_anchor(diagnostics, anchor_id)) + 1
    trial_id = f"{re.sub(r'[^A-Za-z0-9_]+', '_', anchor_id).strip('_')}_setup_trial_{trial_index:02d}"
    setup_anchor = _anchor_for_setup_revision(anchor, revision_request)
    pinning = _pinning_from_compiled_anchor_setup(setup_anchor, compiled, anchor_setup_validation)
    geometry_context = {
        "local_mesh_bbox": list(anchor_setup_validation["local_mesh_bbox"]),
        "local_to_world_transform": [list(row) for row in anchor_setup_validation["local_to_world_transform"]],
    }
    trial = {
        "schema_version": DIAGNOSTIC_SETUP_SCHEMA_VERSION,
        "trial_id": trial_id,
        "trial_index": trial_index,
        "anchor_id": anchor_id,
        "anchor_compile_id": str(compiled["compile_id"]),
        "anchor_preview_evidence_id": str(preview_evidence["evidence_id"]),
        "anchor_region": setup_anchor,
        "revision_request": revision_request,
        "compiled_anchor_target": _compiled_anchor_target_summary(compiled),
        "anchor_setup_validation": anchor_setup_validation,
        "geometry_context": geometry_context,
        "proposed_setup": {
            "source": "runtime_compiled_anchor_target",
            "runtime_owner": "hag4r",
            "summary": "runtime-compiled local mesh-frame anchor target serialized as Genesis env_local static anchoring",
            "executable_frame": "genesis_env_local",
        },
        "derived_pinning": {
            "enabled": bool(pinning.get("enabled", False)),
            "boxes": [
                {
                    key: box[key]
                    for key in (
                        "name",
                        "obj_id",
                        "box",
                        "reason",
                        "source",
                        "anchor_id",
                        "anchor_type",
                        "semantic_region",
                        "uncertainty",
                        "concerns",
                        "selected_part_id",
                        "region_hint",
                        "compile_id",
                        "executable_frame",
                    )
                    if isinstance(box, dict) and key in box
                }
                for box in pinning.get("boxes", [])
                if isinstance(box, dict)
            ],
        },
        "preview_artifacts": {},
        "reflection": {
            "verdict": None,
            "concerns": [],
            "uncertainty": None,
            "revision_request": None,
        },
        "final_status": "previewed",
        "created_at_event_id": _next_runtime_event_id(state),
    }
    png_path = _write_setup_preview_image(state, trial=trial)
    trial["preview_artifacts"] = {
        "png_path": str(png_path),
        "renderer": "static_pil_bbox_projection",
        "overlay_frame": "local_mesh",
    }
    diagnostics["setup_trials"].append(trial)
    diagnostics.setdefault("active_setup_trial_by_anchor", {})[anchor_id] = trial_id
    return trial
def preview_diagnostic_episode_setup(
    run_root: str,
    anchor_id: str,
    revision_request: str | None = None,
) -> dict[str, Any]:
    """Preview one pre-simulation setup trial from the active compiled anchor target."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    anchor = _active_session_anchor(state, str(anchor_id))
    trials = _setup_trials_for_anchor(diagnostics, str(anchor_id))
    if len(trials) >= MAX_DIAGNOSTIC_SETUP_TRIALS:
        latest = trials[-1]
        latest["final_status"] = "proceed_with_concerns"
        diagnostics.setdefault("final_setup_trial_by_anchor", {})[str(anchor_id)] = latest["trial_id"]
        _save_diagnostics(
            bound_run_root,
            state,
            event="diagnostic_setup_trial_limit_reached",
            note=f"setup trial limit reached for {anchor_id}",
            detail={"setup_trial": latest},
        )
        return {
            "setup_preview": {
                "trial_id": latest["trial_id"],
                "anchor_id": latest["anchor_id"],
                "anchor_compile_id": str(latest.get("anchor_compile_id", "")),
                "anchor_preview_evidence_id": str(latest.get("anchor_preview_evidence_id", "")),
                "preview_image_path": str(latest.get("preview_artifacts", {}).get("png_path", "")),
                "validation_status": "proceed_with_concerns",
                "warning_codes": ["setup_trial_limit_reached"],
                "next_action": "define_episode",
            },
        }
    normalized_revision = _validate_semantic_setup_revision_request(revision_request)
    compiled = _active_compiled_anchor_target_for_setup(diagnostics, str(anchor_id))
    preview_evidence = _required_anchor_preview_evidence_for_setup(
        diagnostics,
        anchor_id=str(anchor_id),
        compile_id=str(compiled["compile_id"]),
    )
    anchor_setup_validation = _validate_compiled_anchor_for_setup(
        state,
        anchor=anchor,
        compiled=compiled,
        preview_evidence=preview_evidence,
    )
    trial = _create_setup_trial(
        state,
        anchor=anchor,
        compiled=compiled,
        preview_evidence=preview_evidence,
        anchor_setup_validation=anchor_setup_validation,
        revision_request=normalized_revision,
    )
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_setup_trial_previewed",
        note=f"previewed diagnostic setup trial {trial['trial_id']}",
        detail={"setup_trial": trial},
    )
    return {
        "setup_preview": {
            "trial_id": trial["trial_id"],
            "anchor_id": trial["anchor_id"],
            "anchor_compile_id": trial["anchor_compile_id"],
            "anchor_preview_evidence_id": trial["anchor_preview_evidence_id"],
            "preview_image_path": str(trial["preview_artifacts"]["png_path"]),
            "validation_status": str(trial["anchor_setup_validation"]["status"]),
            "warning_codes": list(trial["anchor_setup_validation"]["warning_codes"]),
            "next_action": "record_diagnostic_setup_reflection",
        },
    }
def record_diagnostic_setup_reflection(
    run_root: str,
    trial_id: str,
    anchor_semantic_match: str,
    anchor_semantic_evidence_refs: list[dict[str, Any]],
    anchor_semantic_explanation: str,
    tested_dof_preservation: str,
    tested_dof_evidence_refs: list[dict[str, Any]],
    tested_dof_explanation: str,
    setup_concern_dispositions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist typed setup evidence; negative observations remain durable but cannot define an episode."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    trial = _setup_trial_by_id(diagnostics, str(trial_id))
    anchor = _active_session_anchor(state, str(trial["anchor_id"]))
    allowed_semantic = {"matched", "mismatched", "unresolved"}
    allowed_dof = {"preserved", "not_preserved", "unresolved"}
    if anchor_semantic_match not in allowed_semantic or tested_dof_preservation not in allowed_dof:
        raise GenesisVlmSchemaError("setup semantic and DOF statuses must use the v2 controlled enums")
    region_id = str(anchor["region_id"])
    preview_id = str(trial.get("anchor_preview_evidence_id", ""))
    def refs(value: Any, label: str) -> list[dict[str, Any]]:
        try:
            resolved, _ = _resolve_v2_preview_refs(
                state, refs=value, region_id=region_id, episode_id=None,
                compile_id=str(trial["anchor_compile_id"]), preview_id=preview_id,
                source_kind="static_anchor_preview", target_key="anchor_id", target_id=str(trial["anchor_id"]),
            )
            return resolved
        except GenesisVlmSchemaError as exc:
            raise GenesisVlmSchemaError(f"{label}: {exc}") from exc
    _, owner_snapshot = _resolve_v2_preview_refs(
        state, refs=anchor_semantic_evidence_refs, region_id=region_id, episode_id=None,
        compile_id=str(trial["anchor_compile_id"]), preview_id=preview_id,
        source_kind="static_anchor_preview", target_key="anchor_id", target_id=str(trial["anchor_id"]),
    )
    plan_concerns = anchor["setup_anchor"]["concerns"]
    overlap_exception = anchor["setup_anchor"]["relationship_to_probe"] == "overlap_exception"
    runtime_overlap_id = _overlap_concern_id(region_id)
    runtime_concerns = [
        item for item in diagnostics["episode_concerns"]
        if isinstance(item, Mapping) and item.get("concern_id") == runtime_overlap_id
    ]
    if overlap_exception:
        if (
            len(runtime_concerns) != 1
            or runtime_concerns[0].get("state") != "pending_setup_acknowledgement"
            or runtime_concerns[0].get("region_id") != region_id
            or runtime_concerns[0].get("episode_id") is not None
            or runtime_concerns[0].get("active_revision_id") != str(state.get("active_revision") or "")
        ):
            raise GenesisVlmSchemaError("overlap exception requires one current plan-persisted runtime concern")
    elif runtime_concerns:
        raise GenesisVlmSchemaError("distinct region must not carry a runtime overlap concern")
    expected_ids = {item["concern_id"] for item in plan_concerns}
    if overlap_exception:
        expected_ids.add(runtime_overlap_id)
    if not isinstance(setup_concern_dispositions, list) or len(setup_concern_dispositions) != len(expected_ids):
        raise GenesisVlmSchemaError("setup_concern_dispositions must contain exactly each required setup concern")
    observed_ids: set[str] = set()
    normalized_concerns: list[dict[str, Any]] = []
    for value in setup_concern_dispositions:
        if not isinstance(value, Mapping) or set(value) != {"concern_id", "disposition", "evidence_refs", "explanation"}:
            raise GenesisVlmSchemaError("setup concern disposition has an invalid shape")
        concern_id = str(value.get("concern_id", ""))
        is_runtime_overlap = concern_id == runtime_overlap_id
        allowed_dispositions = {"acknowledged"} if is_runtime_overlap else {"resolved", "unresolved"}
        if concern_id in observed_ids or concern_id not in expected_ids or value.get("disposition") not in allowed_dispositions:
            raise GenesisVlmSchemaError("setup concern disposition is duplicate, foreign, or unsupported")
        observed_ids.add(concern_id)
        normalized_concerns.append({"concern_id": concern_id, "disposition": value["disposition"], "evidence_refs": refs(value.get("evidence_refs"), "setup concern evidence_refs"), "explanation": _require_nonempty_runtime_text(value.get("explanation"), "setup concern explanation")})
    if observed_ids != expected_ids:
        raise GenesisVlmSchemaError("setup_concern_dispositions must cover every required setup concern exactly once")
    reflection = {
        "schema_version": "hag4r-diagnostic-setup-reflection-v2",
        "anchor_semantic_match": anchor_semantic_match,
        "anchor_semantic_evidence_refs": refs(anchor_semantic_evidence_refs, "anchor_semantic_evidence_refs"),
        "anchor_semantic_explanation": _require_nonempty_runtime_text(anchor_semantic_explanation, "anchor_semantic_explanation"),
        "tested_dof_preservation": tested_dof_preservation,
        "tested_dof_evidence_refs": refs(tested_dof_evidence_refs, "tested_dof_evidence_refs"),
        "tested_dof_explanation": _require_nonempty_runtime_text(tested_dof_explanation, "tested_dof_explanation"),
        "setup_concern_dispositions": normalized_concerns,
        "owner_snapshot": {**owner_snapshot, "trial_id": str(trial["trial_id"]), "anchor_compile_id": str(trial["anchor_compile_id"]), "anchor_preview_evidence_id": preview_id},
    }
    trial["reflection"] = reflection
    anchor_id = str(trial["anchor_id"])
    if anchor_semantic_match == "matched" and tested_dof_preservation == "preserved" and all(
        item["disposition"] == ("acknowledged" if item["concern_id"] == runtime_overlap_id else "resolved")
        for item in normalized_concerns
    ):
        trial["final_status"] = "accepted"
        diagnostics.setdefault("final_setup_trial_by_anchor", {})[anchor_id] = trial["trial_id"]
    else:
        trial["final_status"] = "blocked_by_typed_setup_evidence"
    if overlap_exception and trial["final_status"] == "accepted":
        runtime_concerns[0].update({
            "state": "setup_acknowledged",
            "setup_trial_id": str(trial["trial_id"]),
            "setup_owner_snapshot": {**owner_snapshot, "trial_id": str(trial["trial_id"]), "anchor_compile_id": str(trial["anchor_compile_id"]), "anchor_preview_evidence_id": preview_id},
            "setup_evidence_refs": next(
                item["evidence_refs"] for item in normalized_concerns if item["concern_id"] == runtime_overlap_id
            ),
        })
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_setup_reflection_recorded",
        note=f"recorded diagnostic setup reflection for {trial['trial_id']}: {trial['final_status']}",
        detail={"setup_trial": trial},
    )
    return {
        "status": "success",
        "trial": _setup_trial_preview(trial),
        "state_path": str(state_path(bound_run_root)),
    }
def _anchor_episode_markdown(anchor: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Diagnostic episode",
            "",
            f"Anchor: {anchor['name']}",
            f"Semantic region: {anchor['semantic_region']}",
            f"Physical hypothesis: {anchor['physical_hypothesis']}",
            f"Termination condition: {anchor['termination_condition']}",
        )
    )
def _canonical_episode_suite_payloads(
    state: dict[str, Any],
    *,
    episode_index: int,
    scene_config_path: Path,
    live_output_dir: Path,
    pinning: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = state["paths"]
    bundle = build_agentic_genesis_config_bundle(
        run_id=f"{state['run_id']}_diagnostic_{episode_index:04d}",
        asset_mesh_path=Path(str(paths["monolithic_mesh_path"])),
        material_params_path=Path(str(paths["monolithic_params_path"])),
        part_labels_path=Path(str(paths["part_labels_path"])),
        inferred_params_path=Path(str(paths["inferred_params_path"])),
        volume_topology_path=_optional_path(paths, "volume_topology_path"),
        output_dir=live_output_dir,
        target_config_path=scene_config_path,
        diagnostic_cues=tuple(cue for cue in state.get("sim_diagnostic_cues", ()) if isinstance(cue, dict)),
        fixed_pin_boxes=_pin_boxes_from_pinning(pinning),
        box_ee_controller_policy=diagnostic_force_limited_policy_payload(),
    )
    scene_payload = dict(bundle.config_payload)
    return scene_payload, bundle.model_config_payload, bundle.body_config_payload
def create_diagnostic_episode_from_anchor(
    run_root: str,
    region_id: str,
    episode_markdown: str | None = None,
) -> dict[str, Any]:
    """Create one diagnostic episode suite from one semantic anchor region."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    region_record = _planned_region(state, str(region_id))
    anchor = _active_session_anchor(state, str(region_id))
    diagnostics = _diagnostics(state)
    selected_trial = _select_setup_trial_for_episode(diagnostics, str(region_id))
    setup_quality = _setup_quality_from_trial(selected_trial)
    pinning = _pinning_from_setup_trial(selected_trial)
    authored_payload = {
        "source": "diagnostic_anchor_episode_authoring",
        "anchor_region": anchor,
    }
    result = create_diagnostic_episode_suite(
        run_root=bound_run_root,
        episode_intent=anchor["episode_intent"],
        episode_markdown=episode_markdown or _anchor_episode_markdown(anchor),
        scene_json=authored_payload,
        model_deformable_json=authored_payload,
        generated_asset_json=authored_payload,
        pinning=pinning,
    )
    episode = result.get("episode")
    if isinstance(episode, dict):
        if region_record.get("episode_id"):
            raise GenesisVlmSchemaError("a planned region may own exactly one episode")
        episode["anchor_region"] = anchor
        episode["setup_trial_id"] = str(selected_trial["trial_id"])
        episode["setup_quality"] = setup_quality
        episode["anchor_compile_id"] = str(selected_trial["anchor_compile_id"])
        episode["anchor_preview_evidence_id"] = str(selected_trial["anchor_preview_evidence_id"])
        episode["anchor_setup_validation"] = dict(selected_trial["anchor_setup_validation"])
        episode["force_limited_controller_policy"] = diagnostic_force_limited_policy_payload()
        episode["force_limited_calibration_provenance"] = diagnostic_force_limited_calibration_provenance()
        state = _require_state(bound_run_root)
        diagnostics = _diagnostics(state)
        region_record = _planned_region(state, str(region_id))
        recorded = _episode_record_by_index(diagnostics, int(episode["episode_index"]))
        recorded["anchor_region"] = anchor
        recorded["pinning"] = pinning
        recorded["setup_trial_id"] = str(selected_trial["trial_id"])
        recorded["setup_quality"] = setup_quality
        recorded["anchor_compile_id"] = str(selected_trial["anchor_compile_id"])
        recorded["anchor_preview_evidence_id"] = str(selected_trial["anchor_preview_evidence_id"])
        recorded["anchor_setup_validation"] = dict(selected_trial["anchor_setup_validation"])
        recorded["force_limited_controller_policy"] = diagnostic_force_limited_policy_payload()
        recorded["force_limited_calibration_provenance"] = diagnostic_force_limited_calibration_provenance()
        recorded["region_id"] = str(region_id)
        region_record["episode_id"] = str(recorded["episode_id"])
        region_record["status"] = "episode_defined"
        diagnostics["region_episode_ids"][str(region_id)] = str(recorded["episode_id"])
        if anchor["setup_anchor"]["relationship_to_probe"] == "overlap_exception":
            concern_id = _overlap_concern_id(str(region_id))
            concerns = [
                item for item in diagnostics["episode_concerns"]
                if isinstance(item, dict) and item.get("concern_id") == concern_id
            ]
            if (
                len(concerns) != 1
                or concerns[0].get("state") != "setup_acknowledged"
                or concerns[0].get("episode_id") is not None
                or concerns[0].get("setup_trial_id") != str(selected_trial["trial_id"])
            ):
                raise GenesisVlmSchemaError("episode allocation requires one setup-acknowledged plan overlap concern")
            concerns[0].update({
                "episode_id": str(recorded["episode_id"]),
                "state": "episode_bound_pending_measurement",
                "episode_owner_snapshot": {
                    "region_id": str(region_id),
                    "episode_id": str(recorded["episode_id"]),
                    "active_revision_id": str(state.get("active_revision") or ""),
                    "setup_trial_id": str(selected_trial["trial_id"]),
                },
            })
        selected_record = _setup_trial_by_id(diagnostics, str(selected_trial["trial_id"]))
        reflection = selected_record.get("reflection")
        if not isinstance(reflection, dict) or not isinstance(reflection.get("owner_snapshot"), dict):
            raise GenesisVlmSchemaError("selected setup trial lacks its typed owner snapshot")
        setup_owner = reflection["owner_snapshot"]
        if (
            setup_owner.get("region_id") != str(region_id)
            or setup_owner.get("active_revision_id") != str(state.get("active_revision") or "")
        ):
            raise GenesisVlmSchemaError("selected setup trial owner snapshot is stale or foreign")
        # Setup reflection necessarily precedes episode allocation.  Bind that
        # already-verified pre-episode evidence to its one newly-owned episode
        # inside the same allocation mutation; settlement can now verify the
        # complete tuple without inventing any terminal-time linkage.
        setup_owner.update({
            "episode_id": str(recorded["episode_id"]),
            "trial_id": str(selected_record["trial_id"]),
            "region_id": str(region_id),
        })
        selected_record.update({
            "episode_id": str(recorded["episode_id"]),
            "region_id": str(region_id),
            "attachment_id": setup_owner.get("attachment_id"),
            "agent_invocation_id": setup_owner.get("agent_invocation_id"),
            "active_revision_id": setup_owner.get("active_revision_id"),
        })
        selected_record["final_status"] = "used_for_episode"
        _write_episode_observations(recorded, diagnostics)
        _save_diagnostics(
            bound_run_root,
            state,
            event="diagnostic_anchor_episode_created",
            note=anchor["episode_intent"],
            detail={
                "episode": _episode_metadata(recorded),
                "anchor_region": anchor,
                "pinning": pinning,
                "anchor_compile_id": recorded["anchor_compile_id"],
                "anchor_preview_evidence_id": recorded["anchor_preview_evidence_id"],
                "anchor_setup_validation": recorded["anchor_setup_validation"],
                "setup_trial": selected_record,
                "setup_quality": setup_quality,
            },
        )
        returned_episode = dict(episode)
        returned_episode["anchor_region"] = anchor
        returned_episode["pinning"] = pinning
        returned_episode["setup_trial_id"] = str(selected_trial["trial_id"])
        returned_episode["setup_quality"] = setup_quality
        returned_episode["anchor_compile_id"] = str(selected_trial["anchor_compile_id"])
        returned_episode["anchor_preview_evidence_id"] = str(selected_trial["anchor_preview_evidence_id"])
        returned_episode["anchor_setup_validation"] = dict(selected_trial["anchor_setup_validation"])
        result["episode"] = returned_episode
    return result
def define_episode(
    run_root: str,
    region_id: str,
    episode_markdown: str | None = None,
) -> dict[str, Any]:
    """Define the model-facing diagnostic episode from an accepted runtime-owned anchor setup."""
    result = create_diagnostic_episode_from_anchor(
        run_root=run_root,
        region_id=region_id,
        episode_markdown=episode_markdown,
    )
    result["next_action"] = "simulation_reset"
    result["public_tool"] = "define_episode"
    return result
def create_diagnostic_episode_suite(
    run_root: str,
    episode_intent: str,
    episode_markdown: str,
    scene_json: Any,
    model_deformable_json: Any,
    generated_asset_json: Any,
    pinning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a VLM-authored Genesis diagnostic episode suite and mark it active."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    scene_payload = _require_json_object("scene_json", scene_json)
    model_payload = _require_json_object("model_deformable_json", model_deformable_json)
    body_payload = _require_json_object("generated_asset_json", generated_asset_json)
    # This is the normal session-initialization boundary: the active revision's
    # complete, model-visible inputs are frozen before any diagnostic evidence
    # or material audit can be authored.  Ordinary saves only project it.
    _ensure_part_grounding_context(state)
    runtime_paths = _runtime_owned_workspace_paths(state)
    _write_part_grounding_artifacts(runtime_paths, state)
    initialize_source_semantic_material_artifact_registration(state, runtime_paths=runtime_paths)
    workspace_episodes_root = _diagnostic_workspace_episodes_root(state)
    generated_episodes_root = _diagnostic_generated_episodes_root(state)
    episode_index = len(diagnostics["episodes"]) + 1
    episode_id = f"episode_{episode_index:04d}"
    episode_dir = workspace_episodes_root / episode_id
    generated_episode_dir = generated_episodes_root / episode_id
    suite_dir = episode_dir / "suite"
    episode_md_path = episode_dir / "episode.md"
    scene_config_path = suite_dir / "scene.json"
    model_config_path = suite_dir / "model_deformable.json"
    body_config_path = suite_dir / "generated_asset.json"
    agent_scene_config_path = suite_dir / "agent_authored_scene.json"
    agent_model_config_path = suite_dir / "agent_authored_model_deformable.json"
    agent_body_config_path = suite_dir / "agent_authored_generated_asset.json"
    live_output_dir = generated_episode_dir / "live_output"
    log_dir = generated_episode_dir / "logs"
    ready_file_path = generated_episode_dir / "live_ready.json"
    observations_path = episode_dir / "observations.json"
    png_part_segmentation_triptych_dir = live_output_dir / "png_part_segmentation_triptych"
    png_part_segmentation_triptych_view_dir = live_output_dir / "png_part_segmentation_panels"
    png_part_segmentation_sequence_dir = live_output_dir / "png_part_segmentation_triptych_sequence"
    png_part_segmentation_sequence_view_dir = live_output_dir / "png_part_segmentation_sequence_panels"
    png_depth_dir = live_output_dir / "png_depth"
    png_von_mises_dir = live_output_dir / "png_von_mises"
    for path in (
        episode_md_path,
        scene_config_path,
        model_config_path,
        body_config_path,
        agent_scene_config_path,
        agent_model_config_path,
        agent_body_config_path,
        observations_path,
    ):
        _assert_workspace_episode_write_path(path, workspace_episodes_root)
    for path in (
        generated_episode_dir,
        live_output_dir,
        png_part_segmentation_triptych_dir,
        png_part_segmentation_triptych_view_dir,
        png_part_segmentation_sequence_dir,
        png_part_segmentation_sequence_view_dir,
        png_depth_dir,
        png_von_mises_dir,
        log_dir,
        ready_file_path,
    ):
        _assert_generated_episode_write_path(path, generated_episodes_root)
    suite_dir.mkdir(parents=True, exist_ok=True)
    generated_episode_dir.mkdir(parents=True, exist_ok=True)
    live_output_dir.mkdir(parents=True, exist_ok=True)
    png_part_segmentation_triptych_dir.mkdir(parents=True, exist_ok=True)
    for view_name in TRIPLE_VIEW_PANEL_ORDER:
        (png_part_segmentation_triptych_view_dir / view_name).mkdir(parents=True, exist_ok=True)
    png_depth_dir.mkdir(parents=True, exist_ok=True)
    png_von_mises_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_md_path.write_text(str(episode_markdown).rstrip() + "\n", encoding="utf-8")
    for path, payload in (
        (agent_scene_config_path, scene_payload),
        (agent_model_config_path, model_payload),
        (agent_body_config_path, body_payload),
    ):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    scene_payload, model_payload, body_payload = _canonical_episode_suite_payloads(
        state,
        episode_index=episode_index,
        scene_config_path=scene_config_path,
        live_output_dir=live_output_dir,
        pinning=pinning,
    )
    for path, payload in (
        (scene_config_path, scene_payload),
        (model_config_path, model_payload),
        (body_config_path, body_payload),
    ):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    asset_payload = scene_payload["agentic_diagnostics"]["asset"]
    episode = {
        "schema_version": "hag4r-diagnostic-episode-suite-v1",
        "episode_index": episode_index,
        "episode_id": episode_id,
        "intent": str(episode_intent),
        "pinning": pinning or {},
        "status": "active",
        "episode_dir": str(episode_dir),
        "generated_episode_dir": str(generated_episode_dir),
        "episode_md_path": str(episode_md_path),
        "suite_dir": str(suite_dir),
        "scene_config_path": str(scene_config_path),
        "model_config_path": str(model_config_path),
        "body_config_path": str(body_config_path),
        "agent_authored_scene_config_path": str(agent_scene_config_path),
        "agent_authored_model_config_path": str(agent_model_config_path),
        "agent_authored_body_config_path": str(agent_body_config_path),
        "canonicalized_suite": True,
        "live_output_dir": str(live_output_dir),
        "png_part_segmentation_triptych_dir": str(png_part_segmentation_triptych_dir),
        "png_part_segmentation_triptych_view_dir": str(png_part_segmentation_triptych_view_dir),
        "png_part_segmentation_sequence_dir": str(png_part_segmentation_sequence_dir),
        "png_part_segmentation_sequence_view_dir": str(png_part_segmentation_sequence_view_dir),
        "png_depth_dir": str(png_depth_dir),
        "png_von_mises_dir": str(png_von_mises_dir),
        "log_dir": str(log_dir),
        "ready_file_path": str(ready_file_path),
        "observations_path": str(observations_path),
        "tool_result_indices": [],
        "visual_evidence_indices": [],
        "expected_observations": [],
        "actual_observations": [],
        "force_limited_controller_policy": dict(scene_payload["box_ee_controller_policy"]),
        "force_limited_calibration_provenance": diagnostic_force_limited_calibration_provenance(),
    }
    for required_path in (
        episode_md_path,
        scene_config_path,
        model_config_path,
        body_config_path,
        agent_scene_config_path,
        agent_model_config_path,
        agent_body_config_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"diagnostic episode suite file was not written: {required_path}")
    for previous in diagnostics["episodes"]:
        if isinstance(previous, dict) and previous.get("status") == "active":
            previous["status"] = "superseded"
    diagnostics["episodes"].append(episode)
    diagnostics["active_episode_index"] = episode_index
    _write_episode_observations(episode, diagnostics)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_episode_suite_created",
        note=str(episode_intent),
        detail=episode,
    )
    return {"status": "success", "episode": episode, "state_path": str(state_path(bound_run_root))}
def pause_simulation(run_root: str, expected_observation: str = "simulation should pause", rationale: str = "pause before diagnostic observation") -> dict[str, Any]:
    """Pause the Genesis live session for diagnostic observation."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="pause_simulation",
        arguments={},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def resume_simulation(
    run_root: str,
    steps: int = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
    expected_observation: str = "bounded simulation steps should advance physical state and pause afterward",
    rationale: str = "advance bounded simulation before a pause-and-observe turn",
) -> dict[str, Any]:
    """Resume the Genesis live session for a bounded number of steps."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    bounded_steps = int(steps)
    return _record_live_tool(
        bound_run_root,
        tool_name="resume_simulation",
        arguments={"mode": "bounded", "steps": bounded_steps, "pause_after": True},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def simulation_reset(
    run_root: str,
    expected_observation: str = "initial part segmentation triptych should show the reset diagnostic simulation state",
    rationale: str = "reset diagnostic simulation and capture the initial observation",
) -> dict[str, Any]:
    """Reset or initialize the Genesis live diagnostic episode and return part segmentation triptych evidence."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="simulation_reset",
        arguments={},
        expected_observation=expected_observation,
        rationale=rationale,
    )


def inspect_genesis_runtime_logs(
    run_root: str,
    stream: str = "both",
    cursors: dict[str, int] | None = None,
    max_lines: int = 120,
    contains: str = "",
    context_lines: int = 2,
) -> dict[str, Any]:
    """Return bounded raw stdout/stderr from the active owned Genesis live session."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="inspect_genesis_runtime_logs",
        arguments={
            "stream": stream,
            "cursors": cursors,
            "max_lines": max_lines,
            "contains": contains,
            "context_lines": context_lines,
        },
        expected_observation="raw Genesis stdout/stderr should expose only bounded runtime log lines",
        rationale="inspect the owned Genesis process logs without advancing simulation",
    )
def simulate(
    run_root: str,
    steps: int,
    action: dict[str, Any] | None = None,
    expected_observation: str = "part segmentation triptych sequence should show the simulated diagnostic response",
    rationale: str = "advance diagnostic simulation and observe the auto-paused state",
) -> dict[str, Any]:
    """Advance Genesis for bounded steps with optional compiled probe action and return part segmentation evidence."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="simulate",
        arguments={"steps": int(steps), "action": action},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def pause_and_observe(
    run_root: str,
    expected_observation: str = "current paused simulation frame should show the latest physical state",
    rationale: str = "pause and observe the current simulation state without advancing time",
) -> dict[str, Any]:
    """Pause the Genesis live session and observe the current simulation frame."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="pause_and_observe",
        arguments={},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def get_contact_states(run_root: str, expected_observation: str = "contact telemetry sequence should indicate support or penetration", rationale: str = "inspect resume-scoped contact states") -> dict[str, Any]:
    """Read Genesis live contact telemetry states since the latest resume."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="get_contact_states",
        arguments={},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def get_deformation_states(run_root: str, expected_observation: str = "deformation telemetry sequence should show bounded physical response", rationale: str = "inspect resume-scoped deformation states") -> dict[str, Any]:
    """Read Genesis live deformation telemetry states since the latest resume."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="get_deformation_states",
        arguments={},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def query_live_geometry_context(
    run_root: str,
    env_id: int = 0,
    obj_id: int = 0,
    expected_observation: str = "live env_local bbox context should reflect the current deformable mesh positions",
    rationale: str = "inspect live geometry context",
) -> dict[str, Any]:
    """Query the live env_local geometry context for the current deformable mesh positions."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    return _record_live_tool(
        bound_run_root,
        tool_name="query_live_geometry_context",
        arguments={"env_id": int(env_id), "obj_id": int(obj_id)},
        expected_observation=expected_observation,
        rationale=rationale,
    )
def set_material_params(
    run_root: str,
    scope: dict[str, Any] | None = None,
    young: float | None = None,
    poisson: float | None = None,
    friction_mu: float | None = None,
    bending_weight: float | None = None,
    expected_observation: str = "temporary material adjustment should change the observed response",
    rationale: str = "test material sensitivity",
) -> dict[str, Any]:
    """Temporarily set Genesis material parameters for diagnostics."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    arguments: dict[str, Any] = {"scope": scope or {"type": "global"}}
    for key, value in {
        "young": young,
        "poisson": poisson,
        "friction_mu": friction_mu,
        "bending_weight": bending_weight,
    }.items():
        if value is not None:
            arguments[key] = float(value)
    return _record_live_tool(
        bound_run_root,
        tool_name="set_material_params",
        arguments=arguments,
        expected_observation=expected_observation,
        rationale=rationale,
    )
def _workspace_artifact_ids(state: dict[str, Any]) -> set[str]:
    index_path = Path(str(state["paths"]["diagnostic_workspace_index_path"])).expanduser().resolve()
    if not index_path.is_file():
        return set()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_ids: set[str] = set()
    for group_name in ("artifacts", "unavailable_artifacts"):
        for entry in index.get(group_name, []):
            if isinstance(entry, dict) and entry.get("artifact_id"):
                artifact_ids.add(str(entry["artifact_id"]))
    return artifact_ids
def _workspace_artifact_ref_aliases(state: dict[str, Any]) -> dict[str, str]:
    index_path = Path(str(state["paths"]["diagnostic_workspace_index_path"])).expanduser().resolve()
    if not index_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for group_name in ("artifacts", "unavailable_artifacts"):
        for entry in index.get(group_name, []):
            if not isinstance(entry, dict) or not entry.get("artifact_id"):
                continue
            artifact_id = str(entry["artifact_id"])
            aliases[artifact_id] = artifact_id
            for key in ("path", "virtual_path", "source_path"):
                value = entry.get(key)
                if not value:
                    continue
                text = str(value)
                aliases[text] = artifact_id
                aliases[Path(text).name] = artifact_id
    return aliases
def _validate_diagnostic_evidence_refs(state: dict[str, Any], evidence: dict[str, Any]) -> None:
    diagnostics = _diagnostics(state)
    workspace_artifact_ids = _workspace_artifact_ids(state)
    workspace_artifact_ref_aliases = _workspace_artifact_ref_aliases(state)
    tool_results = [item for item in diagnostics.get("tool_results", []) if isinstance(item, dict)]
    visual_evidence = [item for item in diagnostics.get("visual_evidence", []) if isinstance(item, dict)]
    episodes_by_id = {
        str(episode.get("episode_id")): episode
        for episode in diagnostics.get("episodes", [])
        if isinstance(episode, dict) and episode.get("episode_id")
    }
    allowed_reports = {"diagnostic_summary", "diagnostic_report", "diagnostic_cues"}
    runtime_file_aliases = {
        "index_json": "index_json",
        "index.md": "index_md",
        "index_md": "index_md",
        "index.json": "index_json",
        "workspace_index_json": "workspace_index_json",
        "workspace_index_md": "workspace_index_md",
        "workspace_index_virtual_path": "workspace_index_md",
        "diagnostic_log": "diagnostic_log",
        "diagnostic_log.md": "diagnostic_log",
        "observations_digest": "observations_digest",
        "observations_digest.json": "observations_digest",
    }
    for artifact_ref in evidence["artifact_refs"]:
        kind = artifact_ref["kind"]
        ref = artifact_ref["ref"]
        if kind == "workspace_artifact":
            artifact_id = workspace_artifact_ref_aliases.get(ref)
            if artifact_id is None:
                raise GenesisVlmSchemaError(f"unknown workspace_artifact ref: {ref}")
            artifact_ref["ref"] = artifact_id
            continue
        if kind == "tool_result":
            prefix = "tool_result:"
            if not ref.startswith(prefix):
                raise GenesisVlmSchemaError("tool_result ref must be formatted as tool_result:<1-based-index>")
            index_text = ref.removeprefix(prefix)
            if not index_text.isdigit() or int(index_text) < 1 or int(index_text) > len(tool_results):
                raise GenesisVlmSchemaError(f"unknown tool_result ref: {ref}")
            continue
        if kind == "visual_evidence":
            prefix = "visual_evidence:"
            if not ref.startswith(prefix):
                raise GenesisVlmSchemaError("visual_evidence ref must be formatted as visual_evidence:<0-based-index>")
            index_text = ref.removeprefix(prefix)
            if not index_text.isdigit() or int(index_text) < 0 or int(index_text) >= len(visual_evidence):
                raise GenesisVlmSchemaError(f"unknown visual_evidence ref: {ref}")
            continue
        if kind == "video":
            prefix = "video:"
            if not ref.startswith(prefix):
                raise GenesisVlmSchemaError("video ref must be formatted as video:<episode_id>")
            episode_id = ref.removeprefix(prefix)
            episode = episodes_by_id.get(episode_id)
            if not episode or not episode.get("video_path"):
                raise GenesisVlmSchemaError(f"unknown video ref: {ref}")
            continue
        if kind == "report":
            if ref not in allowed_reports:
                raise GenesisVlmSchemaError(f"unknown report ref: {ref}")
            continue
        if kind == "runtime_file":
            runtime_ref = runtime_file_aliases.get(ref)
            if runtime_ref is not None:
                artifact_ref["ref"] = runtime_ref
                continue
            artifact_id = workspace_artifact_ref_aliases.get(ref)
            if artifact_id is not None:
                artifact_ref["kind"] = "workspace_artifact"
                artifact_ref["ref"] = artifact_id
                continue
            if ref not in runtime_file_aliases.values():
                raise GenesisVlmSchemaError(f"unknown runtime_file ref: {ref}")
            continue
        raise GenesisVlmSchemaError(f"unknown artifact ref kind: {kind}")
def record_diagnostic_evidence(run_root: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Record schema-validated diagnostic observation/reflection evidence.
    The evidence object must include exactly these keys: schema_version, phase,
    episode_id, artifact_refs, observation, reflection, uncertainty,
    knowledge_base_entry_ids, route_relevance, next_action. Use schema_version
    "hag4r-genesis-vlm-reflection-v2". artifact_refs is a list of objects such
    as {"kind": "workspace_artifact", "ref": "object_description", "note": "workspace index artifact"}
    or {"kind": "runtime_file", "ref": "observations_digest", "note": "runtime digest"}.
    Use tool_result and visual_evidence refs only after live tools produce them.
    """
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    validated = validate_vlm_reflection_evidence(evidence)
    _validate_diagnostic_evidence_refs(state, validated)
    if validated["phase"] == "post_probe" and validated["episode_id"] != "pre_episode":
        region_id = _region_id_for_episode(state, validated["episode_id"])
        measurements = [
            item for item in diagnostics["probe_measurements"]
            if isinstance(item, Mapping) and item.get("region_id") == region_id
            and item.get("episode_id") == validated["episode_id"]
        ]
        if measurements:
            if len(measurements) != 1:
                raise GenesisVlmSchemaError("post_probe reflection has ambiguous probe measurements")
            sessions = [
                item for item in state.get("diagnostic_runtime_sessions", []) if isinstance(item, Mapping)
                and item.get("episode_id") == validated["episode_id"]
            ]
            if len(sessions) != 1 or sessions[0].get("lifecycle_state") != "closed":
                raise GenesisVlmSchemaError("healthy measured post_probe reflection requires its episode live session to be closed")
            measurement = measurements[0]
            judgment = validated.get("material_response_judgment")
            if not isinstance(judgment, Mapping):
                raise GenesisVlmSchemaError("healthy post_probe reflection requires a binary material_response_judgment")
            derived_route = [str(judgment["suggested_route"])]
            if validated["route_relevance"] != derived_route:
                raise GenesisVlmSchemaError("post_probe route_relevance must equal runtime-derived material_response_judgment route")
            completion_index = measurement["completion_tool_result_index"]
            required_refs = {f"tool_result:{int(completion_index) + 1}"}
            cited_refs = {str(ref["ref"]) for ref in validated["artifact_refs"] if ref["kind"] == "tool_result"}
            if required_refs != cited_refs.intersection(required_refs):
                raise GenesisVlmSchemaError("post_probe reflection must cite its completed measurement tool result")
            if judgment["suggested_route"] == SimDiagnosticRoute.MATERIAL_INFERENCE.value and measurement.get("pair_id") is not None:
                pair_records = [
                    item for item in diagnostics["paired_comparison_records"] if isinstance(item, Mapping)
                    and item.get("pair_id") == measurement.get("pair_id")
                    and item.get("policy_fingerprint") == measurement.get("pair_policy_fingerprint")
                ]
                if len(pair_records) != 1:
                    raise GenesisVlmSchemaError("paired material-contrast judgment requires both members to clean-close first")
                peer_measurements = [
                    item for item in diagnostics["probe_measurements"] if isinstance(item, Mapping)
                    and item.get("dispatch_token") in pair_records[0].get("dispatch_tokens", [])
                ]
                required_pair_refs = {
                    f"tool_result:{int(item['completion_tool_result_index']) + 1}"
                    for item in peer_measurements
                }
                if len(peer_measurements) != 2 or len(required_pair_refs) != 2 or not required_pair_refs.issubset(cited_refs):
                    raise GenesisVlmSchemaError("paired material-contrast judgment must cite both completed measurement tool results")
            validated["route_relevance"] = derived_route
    record = {
        "reflection_index": len(diagnostics.setdefault("model_reflections", [])),
        **validated,
    }
    if diagnostics.get("active_agent_invocation_id"):
        record["agent_invocation_id"] = diagnostics["active_agent_invocation_id"]
    if validated["episode_id"] != "pre_episode":
        try:
            region_id = _region_id_for_episode(state, validated["episode_id"])
            record["region_id"] = region_id
            record["active_revision_id"] = str(state.get("active_revision") or "")
            attachment = _current_runtime_attachment(state, diagnostics)
            record["attachment_id"] = attachment["attachment_id"]
            record["agent_invocation_id"] = attachment["agent_invocation_id"]
        except GenesisVlmSchemaError:
            pass
    diagnostics["model_reflections"].append(record)
    _save_diagnostics(
        bound_run_root,
        state,
        event="diagnostic_evidence_recorded",
        note=validated["observation"],
        detail=record,
        provenance="typed_model_reflection",
    )
    return {"status": "success", "evidence": record, "state_path": str(state_path(bound_run_root))}


def _material_audit_index_entries(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index_path = Path(str(state["paths"]["diagnostic_workspace_index_path"])).expanduser().resolve()
    if not index_path.is_file():
        raise GenesisVlmSchemaError("material audit requires the current diagnostic workspace index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GenesisVlmSchemaError("material audit workspace index is invalid JSON") from exc
    if not isinstance(index, Mapping):
        raise GenesisVlmSchemaError("material audit workspace index is malformed")
    revision_id = str(state.get("active_revision") or "")
    if index.get("revision_id") != revision_id:
        raise GenesisVlmSchemaError("material audit workspace index has stale revision")
    registration = _active_material_audit_registration(state)
    if index.get("active_source_semantic_material_artifact_registration_id") != registration.get("registration_id"):
        raise GenesisVlmSchemaError("material audit workspace index registration pointer is inconsistent")
    if index.get("source_semantic_material_artifact_registration") != registration:
        raise GenesisVlmSchemaError("material audit workspace index registration is inconsistent")
    raw_entries = index.get("artifacts")
    unavailable_entries = index.get("unavailable_artifacts", [])
    if not isinstance(raw_entries, list) or not isinstance(unavailable_entries, list):
        raise GenesisVlmSchemaError("material audit workspace index artifacts are malformed")
    entries: dict[str, dict[str, Any]] = {}
    canonical_entries = [
        entry for entry in [*raw_entries, *unavailable_entries]
        if isinstance(entry, Mapping) and entry.get("artifact_id") in _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS
    ]
    if len(canonical_entries) != len(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS):
        raise GenesisVlmSchemaError("material audit workspace index must contain exactly five authoritative artifacts")
    for entry in canonical_entries:
        if isinstance(entry, dict) and entry.get("artifact_id") in _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS:
            artifact_id = str(entry["artifact_id"])
            if artifact_id in entries:
                raise GenesisVlmSchemaError("material audit workspace index has duplicate authoritative artifact ID")
            entries[artifact_id] = dict(entry)
    if set(entries) != _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS:
        raise GenesisVlmSchemaError("material audit workspace index authoritative artifact IDs are incomplete")
    registration_entries = registration.get("artifacts")
    if not isinstance(registration_entries, list):
        raise GenesisVlmSchemaError("material audit artifact registration artifacts are malformed")
    registered = {str(item.get("artifact_id")): item for item in registration_entries if isinstance(item, Mapping)}
    if (
        len(registration_entries) != len(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS)
        or len(registered) != len(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS)
        or set(registered) != _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS
    ):
        raise GenesisVlmSchemaError("material audit artifact registration must contain exactly five unique artifacts")
    for artifact_id, entry in entries.items():
        if entry != registered[artifact_id]:
            raise GenesisVlmSchemaError(f"material audit workspace index registration entry drift: {artifact_id}")
    if registration.get("complete") is not True or any(entry.get("available") is not True for entry in entries.values()):
        unavailable = sorted(
            artifact_id for artifact_id, entry in entries.items() if entry.get("available") is not True
        )
        raise GenesisVlmSchemaError(
            "material audit requires a complete five-artifact registration; unavailable artifacts: "
            + ", ".join(unavailable)
        )
    return entries


def _resolve_material_audit_authoritative_artifact(
    state: dict[str, Any],
    ref: Mapping[str, Any],
    *,
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    aliases = _workspace_artifact_ref_aliases(state)
    artifact_id = aliases.get(str(ref["ref"]), str(ref["ref"]))
    if artifact_id not in _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS:
        raise GenesisVlmSchemaError(f"material audit source ref is not one of the five authoritative artifacts: {ref['ref']}")
    if artifact_id not in entries:
        raise GenesisVlmSchemaError(f"material audit cited authoritative artifact is unavailable: {artifact_id}")
    entry = entries[artifact_id]
    revision_id = str(state.get("active_revision") or "")
    if not revision_id or entry.get("active_revision_id") != revision_id:
        raise GenesisVlmSchemaError(f"material audit authoritative artifact has stale revision: {artifact_id}")
    source_path = entry.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise GenesisVlmSchemaError(f"material audit authoritative artifact has no path: {artifact_id}")
    path = Path(source_path).expanduser().resolve()
    expected_path = (
        _runtime_owned_workspace_paths(state)["part_grounding_context"]
        if artifact_id == "part_grounding_context"
        else _optional_path(dict(state["paths"]), _MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_PATH_KEYS[artifact_id])
    )
    if expected_path is None or not expected_path.is_file() or path != expected_path.expanduser().resolve():
        raise GenesisVlmSchemaError(f"material audit required authoritative artifact is unavailable or stale: {artifact_id}")
    workspace_path_value = entry.get("path")
    if (
        not isinstance(workspace_path_value, str) or not workspace_path_value
        or not isinstance(entry.get("workspace_path"), str) or not entry["workspace_path"]
        or not isinstance(entry.get("workspace_virtual_path"), str) or not entry["workspace_virtual_path"]
        or not isinstance(entry.get("source_repository_relative_path"), str)
    ):
        raise GenesisVlmSchemaError(f"material audit authoritative artifact has no workspace path: {artifact_id}")
    workspace_path_entry = Path(workspace_path_value)
    workspace_root = _runtime_owned_workspace_paths(state)["workspace_dir"]
    if workspace_path_entry.is_absolute():
        raise GenesisVlmSchemaError("material audit authoritative workspace path must be workspace-relative")
    if workspace_path_value != entry.get("workspace_path"):
        raise GenesisVlmSchemaError("material audit authoritative workspace path is inconsistent")
    workspace_path = (workspace_root / workspace_path_entry).resolve()
    if workspace_root not in workspace_path.parents:
        raise GenesisVlmSchemaError("material audit authoritative workspace path escapes the diagnostic workspace")
    if not path.is_file() or not workspace_path.is_file():
        raise GenesisVlmSchemaError(f"material audit authoritative artifact is unavailable: {artifact_id}")
    if (
        sha256_file(path) != entry.get("source_sha256", entry.get("sha256"))
        or sha256_file(workspace_path) != entry.get("workspace_sha256", entry.get("sha256"))
        or sha256_file(path) != sha256_file(workspace_path)
    ):
        raise GenesisVlmSchemaError(f"material audit authoritative artifact hash drift: {artifact_id}")
    if artifact_id == "part_grounding_context":
        context = _diagnostics(state).get("part_grounding_context")
        if not isinstance(context, Mapping):
            raise GenesisVlmSchemaError("material audit requires active final part grounding")
        serialized = json.dumps(context, indent=2, sort_keys=True, default=str) + "\n"
        if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != entry.get("source_sha256", entry.get("sha256")):
            raise GenesisVlmSchemaError("material audit authoritative artifact hash drift: part_grounding_context")
    return {
        "kind": "workspace_artifact",
        "ref": artifact_id,
        "active_revision_id": revision_id,
        "sha256": str(entry["sha256"]),
    }


def _verify_all_material_audit_authoritative_artifacts(
    state: dict[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    for artifact_id in sorted(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS):
        _resolve_material_audit_authoritative_artifact(
            state,
            {"kind": "workspace_artifact", "ref": artifact_id, "note": "frozen registration"},
            entries=entries,
        )


def _material_audit_ownership_event(
    state: Mapping[str, Any],
    *,
    attachment_id: str,
    tool_result_index: int | None,
    episode_id: str,
    before_sequence_index: int,
) -> dict[str, Any]:
    events: list[Mapping[str, Any]] = []
    for attachment in state.get("diagnostic_runtime_attachments", []):
        if isinstance(attachment, Mapping) and attachment.get("attachment_id") == attachment_id:
            for key in ("pre_session_tool_events", "terminal_tool_events"):
                events.extend(item for item in attachment.get(key, []) if isinstance(item, Mapping))
    for session in state.get("diagnostic_runtime_sessions", []):
        if isinstance(session, Mapping) and session.get("attachment_id") == attachment_id:
            events.extend(item for item in session.get("tool_events", []) if isinstance(item, Mapping))
    matching = [
        event for event in events
        if event.get("episode_id") == episode_id
        and event.get("tool_result_index") == tool_result_index
        and isinstance(event.get("sequence_index"), int)
        and event["sequence_index"] < before_sequence_index
    ]
    if not matching:
        raise GenesisVlmSchemaError("material audit dynamic evidence lacks prior attachment-owned tool event")
    event = dict(max(matching, key=lambda item: int(item["sequence_index"])))
    if event.get("attachment_id") != attachment_id:
        raise GenesisVlmSchemaError("material audit ownership event belongs to another attachment")
    return event


def _require_current_material_audit_attachment_closed(
    state: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    reflection: Mapping[str, Any],
    *,
    revision_id: str,
) -> dict[str, Any]:
    """Bind a material audit to the active owner only after its live work closed."""
    attachment = _current_runtime_attachment(state, diagnostics)
    attachment_id = str(attachment.get("attachment_id") or "")
    invocation_id = str(attachment.get("agent_invocation_id") or "")
    if (
        not attachment_id
        or not invocation_id
        or reflection.get("attachment_id") != attachment_id
        or reflection.get("agent_invocation_id") != invocation_id
        or reflection.get("active_revision_id") != revision_id
    ):
        raise GenesisVlmSchemaError(
            "material audit trigger reflection does not belong to the current runtime attachment/revision"
        )
    current_sessions = [
        session
        for session in state.get("diagnostic_runtime_sessions", [])
        if isinstance(session, Mapping)
        and session.get("attachment_id") == attachment_id
        and session.get("agent_invocation_id") == invocation_id
        and session.get("active_revision_id") == revision_id
    ]
    if any(session.get("lifecycle_state") != "closed" for session in current_sessions):
        raise GenesisVlmSchemaError(
            "material audit requires every current attachment/revision live session to be closed"
        )
    return attachment


def _resolve_material_audit_trigger_refs(
    state: dict[str, Any],
    *,
    request: Mapping[str, Any],
    reflection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reflection_refs = {
        (str(item.get("kind")), str(item.get("ref")), str(item.get("note")))
        for item in reflection.get("artifact_refs", [])
        if isinstance(item, Mapping)
    }
    diagnostics = _diagnostics(state)
    tool_results = diagnostics.get("tool_results", [])
    visuals = diagnostics.get("visual_evidence", [])
    attachment_id = reflection.get("attachment_id")
    invocation_id = reflection.get("agent_invocation_id")
    episode_id = reflection.get("episode_id")
    reflection_sequence = reflection.get("ownership_event_sequence_index")
    revision_id = str(state.get("active_revision") or "")
    if not all(isinstance(value, str) and value for value in (attachment_id, invocation_id, episode_id)):
        raise GenesisVlmSchemaError("material audit trigger reflection lacks authenticated ownership binding")
    if not isinstance(reflection_sequence, int) or reflection_sequence < 0:
        raise GenesisVlmSchemaError("material audit trigger reflection lacks ownership event sequence")
    attachments = [
        item for item in state.get("diagnostic_runtime_attachments", [])
        if isinstance(item, Mapping)
        and item.get("attachment_id") == attachment_id
        and item.get("agent_invocation_id", invocation_id) == invocation_id
    ]
    if len(attachments) != 1:
        raise GenesisVlmSchemaError("material audit trigger reflection lacks matching current attachment ownership")
    resolved: list[dict[str, Any]] = []
    qualifying = False
    for ref in request["trigger_evidence_refs"]:
        canonical = (ref["kind"], ref["ref"], ref["note"])
        if canonical not in reflection_refs:
            raise GenesisVlmSchemaError("material audit trigger ref was not recorded in trigger reflection")
        if ref["kind"] == "workspace_artifact":
            resolved.append({"kind": ref["kind"], "ref": ref["ref"], "note": ref["note"]})
            continue
        prefix = "tool_result:" if ref["kind"] == "tool_result" else "visual_evidence:"
        if not ref["ref"].startswith(prefix) or not ref["ref"].removeprefix(prefix).isdigit():
            raise GenesisVlmSchemaError(f"material audit malformed dynamic ref: {ref['ref']}")
        raw_index = int(ref["ref"].removeprefix(prefix))
        if ref["kind"] == "tool_result":
            index = raw_index - 1
            if index < 0 or index >= len(tool_results) or not isinstance(tool_results[index], Mapping):
                raise GenesisVlmSchemaError(f"material audit unknown tool result: {ref['ref']}")
            tool_result = tool_results[index]
            tool_result_index = index
        else:
            index = raw_index
            if index < 0 or index >= len(visuals) or not isinstance(visuals[index], Mapping):
                raise GenesisVlmSchemaError(f"material audit unknown visual evidence: {ref['ref']}")
            visual = visuals[index]
            if visual.get("episode_id") != episode_id:
                raise GenesisVlmSchemaError("material audit visual evidence belongs to another episode")
            linked_indices = [
                record_index for record_index, record in enumerate(tool_results)
                if isinstance(record, Mapping) and record.get("visual_evidence") == visual
            ]
            if not linked_indices:
                raise GenesisVlmSchemaError("material audit visual evidence has no linked tool result")
            tool_result_index = linked_indices[-1]
            tool_result = tool_results[tool_result_index]
            manifest_path = visual.get("triple_view_manifest_path")
            advertised_hash = visual.get("sha256") or visual.get("manifest_sha256")
            if manifest_path and advertised_hash and sha256_file(Path(str(manifest_path))) != advertised_hash:
                raise GenesisVlmSchemaError("material audit visual evidence existing hash drift")
        if tool_result.get("episode_id") != episode_id:
            raise GenesisVlmSchemaError("material audit dynamic evidence belongs to another episode")
        applications = [
            item for item in diagnostics.get("probe_target_applications", [])
            if isinstance(item, Mapping)
            and item.get("tool_result_index") == tool_result_index
            and item.get("episode_id") == episode_id
            and item.get("compile_id")
            and str(item.get("result_status")) in LIVE_TOOL_SUCCESS_STATUSES
            and item.get("active_revision_id") == revision_id
        ]
        if not applications:
            raise GenesisVlmSchemaError("material audit dynamic evidence is not a successful compiled probe")
        application = applications[-1]
        compiled_candidates = [
            item for item in diagnostics.get("compiled_probe_targets", [])
            if isinstance(item, Mapping)
            and item.get("compile_id") == application.get("compile_id")
            and item.get("target_id") == application.get("target_id")
            and item.get("simulate_episode_id") == episode_id
            and item.get("active_revision_id") == revision_id
        ]
        if len(compiled_candidates) != 1:
            raise GenesisVlmSchemaError("material audit probe application has stale or fabricated compiled-target lineage")
        compiled = compiled_candidates[0]
        result_compiled = tool_result.get("compiled_probe_target")
        if isinstance(result_compiled, Mapping) and result_compiled.get("compile_id") != application["compile_id"]:
            raise GenesisVlmSchemaError("material audit tool result compile lineage disagrees with probe application")
        event = _material_audit_ownership_event(
            state,
            attachment_id=attachment_id,
            tool_result_index=tool_result_index,
            episode_id=episode_id,
            before_sequence_index=reflection_sequence,
        )
        sessions = [
            session for session in state.get("diagnostic_runtime_sessions", [])
            if isinstance(session, Mapping)
            and session.get("attachment_id") == attachment_id
            and session.get("agent_invocation_id") == invocation_id
            and session.get("episode_id") == episode_id
            and session.get("live_session_handle") == event.get("live_session_handle")
        ]
        if len(sessions) != 1:
            raise GenesisVlmSchemaError("material audit dynamic evidence lacks matching attachment-owned live session")
        resolved.append(
            {
                "kind": ref["kind"], "ref": ref["ref"], "note": ref["note"],
                "attachment_id": attachment_id, "agent_invocation_id": invocation_id,
                "episode_id": episode_id, "tool_result_index": tool_result_index,
                "compile_id": str(application["compile_id"]),
                "attempt_id": application.get("application_index"),
                "application_index": application.get("application_index"),
                "target_id": compiled.get("target_id"),
                "active_revision_id": revision_id,
                "ownership_event_sequence_index": event["sequence_index"],
            }
        )
        qualifying = True
    if not qualifying:
        raise GenesisVlmSchemaError("material audit requires dynamic evidence of a successful compiled probe")
    return resolved


def _material_audit_part_table(
    state: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[int, float]:
    workspace_dir = _runtime_owned_workspace_paths(dict(state))["workspace_dir"]

    def snapshot_path(artifact_id: str) -> Path:
        entry = entries[artifact_id]
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
            raise GenesisVlmSchemaError("material audit authoritative workspace path must be workspace-relative")
        if raw_path != entry.get("workspace_path"):
            raise GenesisVlmSchemaError("material audit authoritative workspace path is inconsistent")
        resolved = (workspace_dir / raw_path).resolve()
        if workspace_dir not in resolved.parents:
            raise GenesisVlmSchemaError("material audit authoritative workspace path escapes the diagnostic workspace")
        return resolved

    material_path = snapshot_path("material_inference_payload")
    grounding_path = snapshot_path("part_grounding_context")
    material_payload = json.loads(material_path.read_text(encoding="utf-8"))
    predictions = material_payload.get("predictions") if isinstance(material_payload, Mapping) else None
    grounding_payload = json.loads(grounding_path.read_text(encoding="utf-8"))
    parts = grounding_payload.get("parts") if isinstance(grounding_payload, Mapping) else None
    mesh = grounding_payload.get("mesh") if isinstance(grounding_payload, Mapping) else None
    if not isinstance(predictions, list) or not isinstance(parts, list) or not isinstance(mesh, Mapping):
        raise GenesisVlmSchemaError("material audit requires complete active material and part-grounding payloads")
    mesh_primitive_count = mesh.get("primitive_count")
    if isinstance(mesh_primitive_count, bool) or not isinstance(mesh_primitive_count, int | float) or mesh_primitive_count <= 0:
        raise GenesisVlmSchemaError("material audit requires positive overall final mesh geometry")
    grounded_indices: set[int] = set()
    for part in parts:
        if not isinstance(part, Mapping) or isinstance(part.get("part_index"), bool) or not isinstance(part.get("part_index"), int):
            continue
        final_geometry = part.get("final_geometry")
        primitive_count = final_geometry.get("primitive_count") if isinstance(final_geometry, Mapping) else None
        if isinstance(primitive_count, bool) or not isinstance(primitive_count, int | float) or primitive_count <= 0:
            continue
        grounded_indices.add(int(part["part_index"]))
    table: dict[int, float] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            continue
        index = prediction.get("part_index")
        young = prediction.get("youngs_modulus_pa")
        if isinstance(index, bool) or not isinstance(index, int) or isinstance(young, bool) or not isinstance(young, int | float):
            continue
        modulus = float(young)
        if index not in grounded_indices or not math.isfinite(modulus) or modulus <= 0:
            continue
        table[index] = modulus
    return table


def _canonical_material_repair_hint(invariant: Mapping[str, Any], *, ratio: float, log_ratio: float, stiffer: Mapping[int, float], softer: Mapping[int, float]) -> str:
    return (
        f"Source-semantic material invariant {invariant['invariant_id']} violated: {invariant['semantic_basis']} "
        f"(semantic strength={invariant['semantic_strength']}, calibration confidence={invariant['calibration_confidence']}). {invariant['stiffer_group']['role']} "
        f"parts={invariant['stiffer_group']['part_indices']} E={dict(stiffer)} must be stiffer than "
        f"{invariant['softer_group']['role']} parts={invariant['softer_group']['part_indices']} E={dict(softer)}; "
        f"runtime ratio={ratio:.12g}, log10_ratio={log_ratio:.12g}, hard minimum={invariant['hard_min_ratio']:.12g}, "
        f"target band=[{invariant['target_min_ratio']:.12g}, {invariant['target_max_ratio']:.12g}]. "
        f"Rationale: {invariant['rationale']}. Re-author complete material hypotheses for all groups in this audit scope, "
        "not only the violated E field; an arbitrarily tiny modulus difference does not satisfy this repair. Keep absolute "
        "values, material names, and fill mode source-plausible and Genesis-stable."
    )


def audit_source_semantic_material_invariants(run_root: str, audit: dict[str, Any]) -> dict[str, Any]:
    """Persist one dynamically-triggered, active-revision constitutive material audit."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    state = _require_state(bound_run_root)
    diagnostics = _diagnostics(state)
    request = validate_source_semantic_material_audit(audit)
    revision_id = str(state.get("active_revision") or "")
    if not revision_id:
        raise GenesisVlmSchemaError("material audit requires an active revision")
    terminal = diagnostics.get("terminal")
    if isinstance(terminal, Mapping) and terminal.get("status") not in {None, "", "running"}:
        raise GenesisVlmSchemaError("material audit is forbidden after terminal diagnostics state")
    if any(isinstance(item, Mapping) and item.get("active_revision_id") == revision_id for item in diagnostics["source_semantic_material_audits"]):
        raise GenesisVlmSchemaError("material audit already exists for the active revision")
    reflections = diagnostics.get("model_reflections", [])
    reflection_index = request["trigger_reflection_index"]
    if reflection_index >= len(reflections) or not isinstance(reflections[reflection_index], Mapping):
        raise GenesisVlmSchemaError("material audit trigger_reflection_index is unknown")
    reflection = reflections[reflection_index]
    if reflection.get("reflection_index") != reflection_index:
        raise GenesisVlmSchemaError("material audit trigger reflection index is mismatched")
    if reflection.get("phase") not in {"post_observation", "post_probe"}:
        raise GenesisVlmSchemaError("material audit requires post_observation or post_probe reflection")
    if reflection.get("active_revision_id") != revision_id:
        raise GenesisVlmSchemaError("material audit trigger reflection belongs to a stale revision")
    _require_current_material_audit_attachment_closed(
        state,
        diagnostics,
        reflection,
        revision_id=revision_id,
    )
    if "material_inference" not in reflection.get("route_relevance", []):
        raise GenesisVlmSchemaError("material audit trigger reflection is not material-relevant")
    if not reflection.get("episode_id") or reflection.get("episode_id") == "pre_episode":
        raise GenesisVlmSchemaError("material audit trigger reflection must own a real episode")
    context = diagnostics.get("part_grounding_context")
    if not isinstance(context, Mapping) or not context.get("parts"):
        raise GenesisVlmSchemaError("material audit requires active final part grounding")
    entries = _material_audit_index_entries(state)
    _verify_all_material_audit_authoritative_artifacts(state, entries)
    source_refs = [
        _resolve_material_audit_authoritative_artifact(state, ref, entries=entries)
        for invariant in request["invariants"] for ref in invariant["source_refs"]
    ]
    trigger_refs = _resolve_material_audit_trigger_refs(state, request=request, reflection=reflection)
    table = _material_audit_part_table(state, entries)
    invariant_results: list[dict[str, Any]] = []
    canonical_hints: list[str] = []
    for invariant in request["invariants"]:
        stiffer_indices = invariant["stiffer_group"]["part_indices"]
        softer_indices = invariant["softer_group"]["part_indices"]
        if set(stiffer_indices) & set(softer_indices) or any(index not in table for index in [*stiffer_indices, *softer_indices]):
            raise GenesisVlmSchemaError(
                f"material audit invariant requires positive final mesh geometry for every grouped part: {invariant['invariant_id']}"
            )
        stiffer = {index: table[index] for index in stiffer_indices}
        softer = {index: table[index] for index in softer_indices}
        ratio = min(stiffer.values()) / max(softer.values())
        log_ratio = math.log10(ratio)
        if ratio < invariant["hard_min_ratio"]:
            status = "violated"
        elif ratio < invariant["target_min_ratio"]:
            status = "minimum_only"
        elif ratio <= invariant["target_max_ratio"]:
            status = "within_target"
        else:
            status = "above_target"
        result = {**invariant, "runtime_stiffer_youngs_modulus_pa": stiffer, "runtime_softer_youngs_modulus_pa": softer,
                  "actual_ratio": ratio, "log10_actual_ratio": log_ratio, "status": status}
        if status == "violated":
            hint = _canonical_material_repair_hint(invariant, ratio=ratio, log_ratio=log_ratio, stiffer=stiffer, softer=softer)
            result["required_repair_hint"] = hint
            canonical_hints.append(hint)
        invariant_results.append(result)
    status_counts = {status: sum(item["status"] == status for item in invariant_results) for status in ("violated", "minimum_only", "within_target", "above_target")}
    # V2 only consumes Feature-1's immutable one-audit slot when the session is
    # complete or the computed result is actually violated.  The computation is
    # still returned as an advisory observation during incomplete coverage.
    audit_id = f"source_semantic_material_audit_{len(diagnostics['source_semantic_material_audits']) + 1:04d}"
    registration = _active_material_audit_registration(state)
    resolved_authoritative_artifacts = [
        _resolve_material_audit_authoritative_artifact(
            state,
            {"kind": "workspace_artifact", "ref": artifact_id, "note": "all-five verification"},
            entries=entries,
        )
        for artifact_id in sorted(_MATERIAL_AUDIT_AUTHORITATIVE_ARTIFACT_IDS)
    ]
    record = {
        "schema_version": VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION,
        "audit_id": audit_id, "audit_index": len(diagnostics["source_semantic_material_audits"]),
        "active_revision_id": revision_id, "request": request,
        "trigger_reflection": {key: reflection.get(key) for key in ("reflection_index", "phase", "route_relevance", "episode_id", "attachment_id", "agent_invocation_id", "ownership_event_sequence_index")},
        "request_cited_source_refs": source_refs,
        "authoritative_artifact_registration_id": registration["registration_id"],
        "resolved_authoritative_artifacts": resolved_authoritative_artifacts,
        "resolved_trigger_evidence": trigger_refs,
        "invariants": invariant_results, "canonical_material_hints": canonical_hints, "status_counts": status_counts,
        "attachment_id": reflection["attachment_id"], "agent_invocation_id": reflection["agent_invocation_id"],
    }
    coverage_summary = build_diagnostic_coverage_summary(state)
    coverage_complete = bool(coverage_summary["regions"]) and all(
        item.get("status") == "coverage_complete"
        for item in coverage_summary["regions"]
    )
    if not coverage_complete and not canonical_hints:
        # An incomplete audit with no violated invariant is advisory only.  Do
        # not consume Feature-1's one current-revision audit slot or create an
        # active audit lineage; the caller may continue probing and retry once
        # coverage is complete (or a real violation is observed).
        advisory = {
            **record,
            "persisted": False,
            "coverage_summary": coverage_summary,
            "coverage_timing": "incomplete",
        }
        return {
            "status": "advisory",
            "audit": advisory,
            "state_path": str(state_path(bound_run_root)),
        }
    diagnostics["source_semantic_material_audits"].append(record)
    diagnostics["active_source_semantic_material_audit_id"] = audit_id
    _save_diagnostics(bound_run_root, state, event="source_semantic_material_audit_created", note=request["concern_summary"], detail={"audit_id": audit_id, "active_revision_id": revision_id, "status_counts": status_counts}, provenance="material_audit")
    return {"status": "success", "audit": record, "state_path": str(state_path(bound_run_root))}
def _derive_region_close_settlement(
    state: dict[str, Any], *, attachment_id: str, live_session_handle: str, episode_id: str, close_outcome: str,
) -> dict[str, Any]:
    """Strictly derive close eligibility from owned, current runtime lineage."""
    diagnostics = _diagnostics(state)
    if close_outcome not in {"clean_closed", "runtime_failed"}:
        raise GenesisVlmSchemaError("close settlement outcome must be clean_closed or runtime_failed")
    region_id = _region_id_for_episode(state, episode_id)
    regions = [item for item in diagnostics["diagnostic_region_ledger"] if isinstance(item, dict) and item.get("region_id") == region_id]
    if len(regions) != 1:
        raise GenesisVlmSchemaError("close settlement region ownership is missing or ambiguous")
    region = regions[0]
    revision = str(state.get("active_revision") or "")
    sessions = [item for item in state.get("diagnostic_runtime_sessions", []) if isinstance(item, Mapping) and item.get("attachment_id") == attachment_id and item.get("live_session_handle") == live_session_handle and item.get("episode_id") == episode_id and item.get("active_revision_id") == revision]
    if len(sessions) != 1:
        raise GenesisVlmSchemaError("close settlement requires exactly one owned durable live-session tuple")
    session = sessions[0]
    attachments = [item for item in state.get("diagnostic_runtime_attachments", []) if isinstance(item, Mapping) and item.get("attachment_id") == attachment_id]
    if len(attachments) != 1:
        raise GenesisVlmSchemaError("close settlement requires exactly one durable attachment tuple")
    attachment = attachments[0]
    failure = session.get("runtime_failure")
    # A successful close after a recorded runtime failure remains a runtime
    # failure for terminal adjudication; callers do not get to relabel it as a
    # clean close merely because cleanup completed.
    if close_outcome == "clean_closed" and isinstance(failure, Mapping):
        close_outcome = "runtime_failed"
    expected_lifecycle_states = {"closed_failed", "closed"} if close_outcome == "runtime_failed" else {"closed"}
    if attachment.get("status") not in {"attached", "completed"} or session.get("agent_invocation_id") != attachment.get("agent_invocation_id") or session.get("lifecycle_state") not in expected_lifecycle_states:
        raise GenesisVlmSchemaError("close settlement live session crosses attachment ownership")
    episodes = [item for item in diagnostics["episodes"] if isinstance(item, Mapping) and item.get("episode_id") == episode_id]
    if len(episodes) != 1:
        raise GenesisVlmSchemaError("close settlement requires exactly one owned episode")
    episode = episodes[0]
    attempt_id = region.get("qualifying_attempt_id")
    current_attempts = [
        item for item in diagnostics["diagnostic_probe_attempts"]
        if isinstance(item, Mapping)
        and item.get("region_id") == region_id
        and item.get("episode_id") == episode_id
        and item.get("active_revision_id") == revision
    ]
    # The region pointer and the server-derived qualifying flag are both
    # durable lineage, not agent claims.  A pointer to an attempt whose flag
    # was cleared (or whose record was removed/duplicated) is corruption, not
    # an ordinary incomplete close that may be silently downgraded.
    if attempt_id:
        pointed_attempts = [item for item in current_attempts if item.get("attempt_id") == attempt_id]
        if len(pointed_attempts) != 1 or pointed_attempts[0].get("qualifying") is not True:
            raise GenesisVlmSchemaError("region qualifying attempt pointer or qualifier is stale or corrupt")
    qualifying = [item for item in current_attempts if item.get("qualifying") is True]
    if len(qualifying) > 1:
        raise GenesisVlmSchemaError("region has duplicate qualifying probe attempts")
    attempt = qualifying[0] if qualifying else None
    if attempt is not None and attempt.get("attempt_id") != attempt_id:
        raise GenesisVlmSchemaError("region qualifying attempt pointer is stale or corrupt")
    setup_trials = [
        item for item in diagnostics["setup_trials"]
        if isinstance(item, Mapping)
        and item.get("anchor_id") == region_id
        and item.get("region_id") == region_id
        and item.get("episode_id") == episode_id
        and item.get("attachment_id") == attachment_id
        and item.get("agent_invocation_id") == attachment.get("agent_invocation_id")
        and item.get("active_revision_id") == revision
        and item.get("final_status") == "used_for_episode"
    ]
    if len(setup_trials) != 1:
        raise GenesisVlmSchemaError("close settlement requires exactly one used setup trial")
    setup = setup_trials[0].get("reflection") if len(setup_trials) == 1 and isinstance(setup_trials[0].get("reflection"), Mapping) else {}
    setup_owner = setup.get("owner_snapshot") if isinstance(setup, Mapping) else None
    expected_setup_concerns = {item.get("concern_id") for item in region["region"]["setup_anchor"]["concerns"] if isinstance(item, Mapping)}
    overlap_declared = region["region"]["setup_anchor"].get("relationship_to_probe") == "overlap_exception"
    runtime_overlap_id = _overlap_concern_id(region_id)
    setup_dispositions = setup.get("setup_concern_dispositions", []) if isinstance(setup, Mapping) else []
    setup_concern_ids = [item.get("concern_id") for item in setup_dispositions if isinstance(item, Mapping)] if isinstance(setup_dispositions, list) else []
    plan_setup_dispositions = [
        item for item in setup_dispositions
        if isinstance(item, Mapping) and item.get("concern_id") != runtime_overlap_id
    ]
    runtime_setup_dispositions = [
        item for item in setup_dispositions
        if isinstance(item, Mapping) and item.get("concern_id") == runtime_overlap_id
    ]
    plan_setup_ok = (
        len(setup_concern_ids) == len(set(setup_concern_ids))
        and {item.get("concern_id") for item in plan_setup_dispositions} == expected_setup_concerns
        and all(item.get("disposition") == "resolved" for item in plan_setup_dispositions)
    )
    runtime_setup_ok = (
        len(runtime_setup_dispositions) == 1
        and runtime_setup_dispositions[0].get("disposition") == "acknowledged"
        and bool(runtime_setup_dispositions[0].get("evidence_refs"))
    ) if overlap_declared else not runtime_setup_dispositions
    setup_concerns_ok = plan_setup_ok and runtime_setup_ok
    if isinstance(setup_owner, Mapping) and any(setup_owner.get(key) != value for key, value in {"region_id": region_id, "episode_id": episode_id, "attachment_id": attachment_id, "agent_invocation_id": attachment.get("agent_invocation_id"), "active_revision_id": revision, "trial_id": setup_trials[0].get("trial_id")}.items()):
        raise GenesisVlmSchemaError("setup reflection owner snapshot crosses the close ownership tuple")
    setup_ok = bool(setup.get("anchor_semantic_match") == "matched" and setup.get("tested_dof_preservation") == "preserved" and isinstance(setup_owner, Mapping) and setup_owner.get("region_id") == region_id and setup_owner.get("episode_id") == episode_id and setup_owner.get("attachment_id") == attachment_id and setup_owner.get("agent_invocation_id") == attachment.get("agent_invocation_id") and setup_owner.get("active_revision_id") == revision and setup_owner.get("trial_id") == setup_trials[0].get("trial_id") and setup_concerns_ok)
    probe_reflections = [item for item in diagnostics["probe_target_reflections"] if isinstance(item, Mapping) and item.get("region_id") == region_id and item.get("episode_id") == episode_id and item.get("compile_id") == (attempt or {}).get("compile_id") and item.get("active_revision_id") == revision]
    probe_owner = probe_reflections[0].get("owner_snapshot") if len(probe_reflections) == 1 and isinstance(probe_reflections[0].get("owner_snapshot"), Mapping) else {}
    if probe_reflections and any(probe_owner.get(key) != value for key, value in {"attachment_id": attachment_id, "agent_invocation_id": attachment.get("agent_invocation_id"), "episode_id": episode_id, "region_id": region_id, "active_revision_id": revision}.items()):
        raise GenesisVlmSchemaError("probe reflection owner snapshot crosses the close ownership tuple")
    probe_ok = len(probe_reflections) == 1 and probe_reflections[0].get("semantic_match") == "matched" and probe_owner.get("attachment_id") == attachment_id and probe_owner.get("agent_invocation_id") == attachment.get("agent_invocation_id") and probe_owner.get("episode_id") == episode_id and probe_owner.get("region_id") == region_id and probe_owner.get("active_revision_id") == revision
    separation = [item for item in diagnostics["anchor_probe_separation_results"] if isinstance(item, Mapping) and item.get("region_id") == region_id and item.get("probe_compile_id") == (attempt or {}).get("compile_id")]
    separation_owner = separation[0].get("owner_snapshot") if len(separation) == 1 and isinstance(separation[0].get("owner_snapshot"), Mapping) else {}
    if separation:
        _require_canonical_anchor_probe_separation(separation[0])
    if separation and any(separation_owner.get(key) != value for key, value in {"attachment_id": attachment_id, "agent_invocation_id": attachment.get("agent_invocation_id"), "region_id": region_id, "episode_id": episode_id, "active_revision_id": revision}.items()):
        raise GenesisVlmSchemaError("anchor/probe separation owner snapshot crosses the close ownership tuple")
    separation_ok = len(separation) == 1 and separation[0].get("episode_id") == episode_id and separation[0].get("active_revision_id") == revision and separation_owner.get("attachment_id") == attachment_id and separation_owner.get("agent_invocation_id") == attachment.get("agent_invocation_id") and separation_owner.get("region_id") == region_id and separation_owner.get("episode_id") == episode_id and separation_owner.get("active_revision_id") == revision and separation_owner.get("anchor_compile_id") == separation[0].get("anchor_compile_id") and separation_owner.get("probe_compile_id") == separation[0].get("probe_compile_id") and separation[0].get("validation_outcome") in {"validated_distinct", "exception_pending_concern_discharge"}
    expected_concern_id = runtime_overlap_id
    concerns = [item for item in diagnostics["episode_concerns"] if isinstance(item, Mapping) and item.get("concern_id") == expected_concern_id]
    if overlap_declared and separation and len(concerns) == 1 and concerns[0].get("separation_id") != separation[0].get("separation_id"):
        raise GenesisVlmSchemaError("overlap concern current separation does not match frozen qualifying compile pair")
    concerns_ok = (not overlap_declared) or (len(concerns) == 1 and concerns[0].get("origin") == "runtime" and concerns[0].get("region_id") == region_id and concerns[0].get("episode_id") == episode_id and concerns[0].get("active_revision_id") == revision and concerns[0].get("state") == "resolved")
    if attempt is not None:
        result_index, application_index = attempt.get("tool_result_index"), attempt.get("application_index")
        if isinstance(result_index, bool) or not isinstance(result_index, int) or isinstance(application_index, bool) or not isinstance(application_index, int):
            raise GenesisVlmSchemaError("qualifying attempt lacks result/application lineage")
        results, applications = diagnostics["tool_results"], diagnostics["probe_target_applications"]
        if not (0 <= result_index < len(results) and 0 <= application_index < len(applications)):
            raise GenesisVlmSchemaError("qualifying attempt points outside result/application lineage")
        result, application = results[result_index], applications[application_index]
        matching_applications = [item for item in applications if isinstance(item, Mapping) and item.get("attempt_id") == attempt.get("attempt_id")]
        matching_results = [item for item in results if isinstance(item, Mapping) and item.get("tool_result_index") == result_index]
        payload = result.get("result", {}).get("result", {}) if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping) else {}
        hard_error = isinstance(result.get("result", {}).get("error"), Mapping) and bool(result["result"]["error"]) if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping) else True
        wrapper = result.get("result") if isinstance(result, Mapping) else None
        validation = application.get("probe_target_validation") if isinstance(application, Mapping) else None
        if isinstance(application, Mapping) and isinstance(application.get("force_limited_schedule"), Mapping):
            full_probe_window = _completed_force_limited_probe_window(
                application=application,
                payload=payload,
            )
        else:
            full_probe_window = _completed_default_probe_window(
                requested_steps=application.get("requested_duration_steps") if isinstance(application, Mapping) else None,
                completed_steps=payload.get("steps_completed"),
                runtime_step_adaptation=payload.get("runtime_step_adaptation"),
            )
        attempt_ok = bool(len(matching_applications) == 1 and len(matching_results) == 1 and attempt.get("dispatch_status") == "completed" and isinstance(result, Mapping) and result.get("tool") == "simulate" and result.get("episode_id") == episode_id and isinstance(wrapper, Mapping) and _live_tool_succeeded(wrapper) and attempt.get("result_status") == wrapper.get("status") and isinstance(application, Mapping) and application.get("result_status") == wrapper.get("status") and application.get("region_id") == region_id and application.get("episode_id") == episode_id and application.get("active_revision_id") == revision and application.get("attempt_id") == attempt.get("attempt_id") and application.get("compile_id") == attempt.get("compile_id") and application.get("registered_action_id") == attempt.get("registered_action_id") and attempt.get("attachment_id") == attachment_id and attempt.get("agent_invocation_id") == attachment.get("agent_invocation_id") and full_probe_window and not hard_error and isinstance(validation, Mapping) and validation.get("status") == "ok" and validation.get("compile_id") == attempt.get("compile_id") and validation.get("tool_result_index") == result_index and float(validation.get("grabbed_selected_part_fraction", 0.0)) >= 0.50 and float(attempt.get("grabbed_selected_part_fraction", 0.0)) >= 0.50)
        if not attempt_ok:
            raise GenesisVlmSchemaError("qualifying attempt result, purity, or ownership lineage is corrupt")
    if close_outcome == "clean_closed" and failure is not None:
        raise GenesisVlmSchemaError("clean_closed settlement may not carry a runtime_failure marker")
    if close_outcome == "runtime_failed" and not isinstance(failure, Mapping):
        raise GenesisVlmSchemaError("runtime_failed settlement requires correlated durable runtime_failure evidence")
    if close_outcome == "runtime_failed" and (failure.get("attachment_id") != attachment_id or failure.get("episode_id") != episode_id or failure.get("live_session_handle") != live_session_handle):
        raise GenesisVlmSchemaError("runtime_failed settlement failure marker crosses the owned session tuple")
    failure_evidence: dict[str, Any] | None = None
    if close_outcome == "runtime_failed":
        failure_evidence = _require_correlated_runtime_log_failure_evidence(
            state, diagnostics, require_close_event=False
        )
        if any(failure_evidence.get(key) != failure.get(key) for key in ("attachment_id", "episode_id", "live_session_handle")):
            raise GenesisVlmSchemaError("runtime_failed settlement lacks the correlated runtime-log failure lineage")
    measurements = [
        item for item in diagnostics["probe_measurements"] if isinstance(item, Mapping)
        and item.get("region_id") == region_id and item.get("episode_id") == episode_id
        and item.get("compile_id") == (attempt or {}).get("compile_id")
    ]
    measurement_ok = len(measurements) == 1
    status = "runtime_failed" if close_outcome == "runtime_failed" else "coverage_complete" if attempt and setup_ok and probe_ok and separation_ok and concerns_ok and measurement_ok else "closed_incomplete"
    evidence_lineage = {
        "setup": {
            "trial_id": setup_trials[0].get("trial_id"),
            "reflection_id": setup.get("reflection_id"),
            "anchor_compile_id": setup_trials[0].get("anchor_compile_id"),
            "anchor_preview_evidence_id": setup_trials[0].get("anchor_preview_evidence_id"),
            "anchor_selected_part_id": (
                setup_trials[0].get("anchor_setup_validation", {}).get("selected_part_id")
                if isinstance(setup_trials[0].get("anchor_setup_validation"), Mapping)
                else None
            ),
            "status": setup_trials[0].get("final_status"),
            "anchor_semantic_match": setup.get("anchor_semantic_match"),
            "anchor_semantic_evidence_refs": setup.get("anchor_semantic_evidence_refs", []),
            "tested_dof_preservation": setup.get("tested_dof_preservation"),
            "tested_dof_evidence_refs": setup.get("tested_dof_evidence_refs", []),
            "concern_dispositions": setup.get("setup_concern_dispositions", []),
        },
        "probe": {
            "reflection_id": probe_reflections[0].get("reflection_id") if probe_reflections else None,
            "semantic_match": probe_reflections[0].get("semantic_match") if probe_reflections else None,
            "compile_id": attempt.get("compile_id") if attempt else None,
            "selected_part_id": attempt.get("selected_part_id") if attempt else None,
            "evidence_refs": (
                probe_reflections[0].get("evidence_refs", probe_reflections[0].get("artifact_refs", []))
                if probe_reflections else []
            ),
        },
        "attempt": {
            "attempt_id": attempt.get("attempt_id") if attempt else None,
            "application_index": attempt.get("application_index") if attempt else None,
            "tool_result_index": attempt.get("tool_result_index") if attempt else None,
            "dispatch_status": attempt.get("dispatch_status") if attempt else None,
            "result_status": attempt.get("result_status") if attempt else None,
            "grabbed_selected_part_fraction": attempt.get("grabbed_selected_part_fraction") if attempt else None,
        },
        "measurement": {
            "dispatch_token": measurements[0].get("dispatch_token") if measurements else None,
            "completion_tool_result_index": measurements[0].get("completion_tool_result_index") if measurements else None,
            "valid": measurement_ok,
        },
        "separation": {
            "separation_id": separation[-1].get("separation_id") if separation else None,
            "derived_basis_status": separation[-1].get("derived_basis_status") if separation else None,
            "validation_outcome": separation[-1].get("validation_outcome") if separation else None,
            "anchor_compile_id": separation[-1].get("anchor_compile_id") if separation else None,
            "probe_compile_id": separation[-1].get("probe_compile_id") if separation else None,
            "anchor_aabb": separation[-1].get("anchor_aabb") if separation else None,
            "probe_aabb": separation[-1].get("probe_aabb") if separation else None,
            "anchor_part_id": separation[-1].get("anchor_part_id") if separation else None,
            "probe_part_id": separation[-1].get("probe_part_id") if separation else None,
            "positive_intersection_volume": separation[-1].get("positive_intersection_volume") if separation else None,
            "relationship_to_probe": separation[-1].get("relationship_to_probe") if separation else None,
        },
        "concerns": [
            {
                "concern_id": item.get("concern_id"),
                "state": item.get("state"),
                "evidence_refs": item.get("evidence_refs", []),
                "explanation": item.get("explanation", ""),
            }
            for item in concerns
        ],
        "ownership": {
            "attachment_id": attachment_id,
            "agent_invocation_id": attachment.get("agent_invocation_id"),
            "live_session_handle": live_session_handle,
            "episode_id": episode_id,
            "lifecycle_state": session.get("lifecycle_state"),
            "closed_at": session.get("closed_at"),
            "bound_at": session.get("bound_at"),
        },
        # The close event is appended by the enclosing ownership mutation
        # after settlement.  It is terminal-time evidence, not immutable
        # close-settlement input; retaining it here would create false drift
        # on the first post-close summary recomputation.
        "runtime_failure": (
            {
                key: value
                for key, value in failure_evidence.items()
                if key != "close_event_sequence_index"
            }
            if failure_evidence is not None
            else None
        ),
    }
    return {
        "region_id": region_id,
        "episode_id": episode_id, "attachment_id": attachment_id, "live_session_handle": live_session_handle,
        "active_revision_id": revision, "close_outcome": close_outcome, "status": status,
        "qualifying_attempt_id": attempt_id, "evidence_lineage": evidence_lineage,
    }


def settle_region_on_live_close(
    state: dict[str, Any], *, attachment_id: str, live_session_handle: str, episode_id: str, close_outcome: str,
) -> dict[str, Any]:
    """Persist the one immutable settlement derived by the strict shared verifier."""
    diagnostics = _diagnostics(state)
    derived = _derive_region_close_settlement(state, attachment_id=attachment_id, live_session_handle=live_session_handle, episode_id=episode_id, close_outcome=close_outcome)
    region_id = derived["region_id"]
    existing = [item for item in diagnostics["region_settlements"] if isinstance(item, Mapping) and item.get("region_id") == region_id]
    if existing:
        if len(existing) != 1 or any(existing[0].get(key) != value for key, value in derived.items()):
            raise GenesisVlmSchemaError("region settlement is immutable and conflicts with current derived lineage")
        return dict(existing[0])
    settlement = {"settlement_id": f"settlement_{len(diagnostics['region_settlements']) + 1:04d}", **derived}
    diagnostics["region_settlements"].append(settlement)
    _planned_region(state, region_id)["status"] = settlement["status"]
    policy = _pair_policy_for_region(diagnostics, region_id)
    if policy is not None and settlement["status"] == "coverage_complete":
        members = list(policy["region_ids"])
        peer_settlements = [
            item for item in diagnostics["region_settlements"] if isinstance(item, Mapping)
            and item.get("region_id") in members and item.get("status") == "coverage_complete"
        ]
        if len(peer_settlements) == 2 and {item.get("region_id") for item in peer_settlements} == set(members):
            measurements = [
                item for item in diagnostics["probe_measurements"] if isinstance(item, Mapping)
                and item.get("region_id") in members and item.get("pair_policy_fingerprint") == policy["policy_fingerprint"]
            ]
            if len(measurements) != 2:
                raise GenesisVlmSchemaError("paired close requires two valid measurements under the frozen policy")
            record = {
                "pair_id": policy["pair_id"], "region_ids": members,
                "policy_fingerprint": policy["policy_fingerprint"],
                "settlement_ids": [next(item["settlement_id"] for item in peer_settlements if item["region_id"] == member) for member in members],
                "dispatch_tokens": [next(item["dispatch_token"] for item in measurements if item["region_id"] == member) for member in members],
            }
            existing_pair = [item for item in diagnostics["paired_comparison_records"] if isinstance(item, Mapping) and item.get("pair_id") == policy["pair_id"]]
            if existing_pair:
                if len(existing_pair) != 1 or dict(existing_pair[0]) != record:
                    raise GenesisVlmSchemaError("paired comparison record is immutable and conflicts with close lineage")
            else:
                diagnostics["paired_comparison_records"].append(record)
    return settlement


def build_diagnostic_coverage_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Recompute the compact, auditable v2 coverage projection from durable lineage."""
    plan = require_v2_session_state(state)
    assert isinstance(plan, Mapping)
    diagnostics = _diagnostics(state)
    verify_v2_anchor_probe_separation_records(state)
    ledger = diagnostics["diagnostic_region_ledger"]
    planned_regions = plan["regions"]
    if len(ledger) != len(planned_regions):
        raise GenesisVlmSchemaError("coverage summary requires the region ledger to cover every planned region exactly once")
    for position, (planned, entry) in enumerate(zip(planned_regions, ledger, strict=True)):
        if (
            not isinstance(entry, Mapping)
            or entry.get("region_id") != planned.get("region_id")
            or entry.get("risk_rank") != planned.get("risk_rank")
            or entry.get("region") != planned
        ):
            raise GenesisVlmSchemaError(
                f"coverage summary region ledger entry {position} is not the immutable planned region"
            )
    regions = []
    for entry in ledger:
        if not isinstance(entry, Mapping):
            raise GenesisVlmSchemaError("diagnostic region ledger contains a malformed record")
        matches = [item for item in diagnostics["region_settlements"] if isinstance(item, Mapping) and item.get("region_id") == entry.get("region_id")]
        if len(matches) > 1:
            raise GenesisVlmSchemaError("coverage summary rejects duplicate settlement rows")
        settlement = matches[0] if matches else None
        if settlement is not None:
            if settlement.get("status") == "early_exit":
                region_id = entry.get("region_id")
                has_runtime_session = any(
                    isinstance(session, Mapping)
                    and session.get("episode_id") == entry.get("episode_id")
                    for session in state.get("diagnostic_runtime_sessions", [])
                )
                has_dispatched_attempt = any(
                    isinstance(attempt, Mapping)
                    and attempt.get("region_id") == region_id
                    for attempt in diagnostics["diagnostic_probe_attempts"]
                )
                terminal = diagnostics.get("terminal")
                attempt_index = settlement.get("terminal_attempt_index")
                attempts = diagnostics.get("terminal_attempts", [])
                if (
                    has_runtime_session
                    or has_dispatched_attempt
                    or settlement.get("active_revision_id") != str(state.get("active_revision") or "")
                    or settlement.get("plan_schema_version") != VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION
                    or not settlement.get("terminal_decision_id")
                    or not isinstance(terminal, Mapping)
                    or terminal.get("status") not in {"pending_validation", "running", "success"}
                    or not isinstance(terminal.get("validated"), bool)
                    or (terminal.get("status") == "success" and terminal.get("validated") is not True)
                    or terminal.get("terminal_decision_id") != settlement.get("terminal_decision_id")
                    or not isinstance(attempt_index, int)
                    or isinstance(attempt_index, bool)
                    or not isinstance(attempts, list)
                    or not (0 <= attempt_index < len(attempts))
                    or not isinstance(attempts[attempt_index], Mapping)
                    or attempts[attempt_index].get("status") != "accepted"
                    or attempts[attempt_index].get("normalized_recommendation") != terminal.get("recommendation")
                    or terminal.get("terminal_attempt_index") != attempt_index
                ):
                    raise GenesisVlmSchemaError("coverage summary rejects malformed early-exit settlement")
                regions.append({"region_id": entry.get("region_id"), "risk_rank": entry.get("risk_rank"), "status": "early_exit",
                                "episode_id": None, "qualifying_attempt_id": None,
                                "settlement_id": settlement.get("settlement_id")})
                continue
            required = ("attachment_id", "live_session_handle", "episode_id", "close_outcome")
            if any(not settlement.get(key) for key in required):
                raise GenesisVlmSchemaError("coverage summary settlement lacks a complete ownership tuple")
            derived = _derive_region_close_settlement(
                state,
                attachment_id=str(settlement["attachment_id"]),
                live_session_handle=str(settlement["live_session_handle"]),
                episode_id=str(settlement["episode_id"]),
                close_outcome=str(settlement["close_outcome"]),
            )
            if settlement.get("status") != derived["status"] or settlement.get("evidence_lineage") != derived["evidence_lineage"]:
                raise GenesisVlmSchemaError("coverage summary rejects settlement status or evidence-lineage drift")
        lineage = settlement.get("evidence_lineage") if settlement else None
        compact_evidence = None
        if isinstance(lineage, Mapping):
            settlement_episode_id = settlement.get("episode_id") if settlement else None
            episode = next(
                (item for item in diagnostics.get("episodes", [])
                 if isinstance(item, Mapping) and item.get("episode_id") == settlement_episode_id),
                {},
            )
            compact_evidence = {
                "anchor": lineage.get("setup"),
                "probe": lineage.get("probe"),
                "qualifying_attempt": lineage.get("attempt"),
                "anchor_probe_separation": lineage.get("separation"),
                "concern_states": lineage.get("concerns"),
                "runtime_failure": lineage.get("runtime_failure"),
                "video_path": episode.get("triptych_video_path", "") if isinstance(episode, Mapping) else "",
                "ownership_timing": lineage.get("ownership"),
            }
        regions.append({"region_id": entry.get("region_id"), "risk_rank": entry.get("risk_rank"), "status": settlement.get("status") if settlement else entry.get("status"),
                        "episode_id": settlement.get("episode_id") if settlement else entry.get("episode_id"), "qualifying_attempt_id": settlement.get("qualifying_attempt_id") if settlement else entry.get("qualifying_attempt_id"), "settlement_id": settlement.get("settlement_id") if settlement else None,
                        "coverage_evidence": compact_evidence})
    audits = [item for item in diagnostics["source_semantic_material_audits"] if isinstance(item, Mapping) and item.get("active_revision_id") == str(state.get("active_revision") or "")]
    audit = audits[-1] if audits else None
    summary = {"schema_version": "hag4r-diagnostic-coverage-summary-v2", "plan_schema_version": plan["schema_version"],
               "selected_region_count": len(regions), "regions": regions, "omitted_region_summary": list(plan["omitted_region_summary"]),
               "open_live_sessions": [item.get("live_session_handle") for item in state.get("diagnostic_runtime_sessions", []) if isinstance(item, Mapping) and item.get("lifecycle_state") not in {"closed", "closed_failed"}],
               "material_audit_status": [item.get("status") for item in audit.get("invariants", [])] if audit else []}
    diagnostics["coverage_summary"] = summary
    return summary


def _current_owned_probe_evidence_reflections(
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    route: str | None = None,
    allow_completed_attachment: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return current owned post-probe reflections with proven live evidence.

    The model's route list alone is never a terminal defect.  A current,
    attachment-owned reflection must also cite successful compiled-probe
    evidence through the existing narrow audit-ref resolver.
    """
    attachment = _current_runtime_attachment(state, diagnostics, allow_completed=allow_completed_attachment)
    revision_id = str(state.get("active_revision") or "")
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for reflection in diagnostics.get("model_reflections", []):
        if not isinstance(reflection, Mapping):
            continue
        relevance = reflection.get("route_relevance")
        if (
            reflection.get("phase") not in {"post_observation", "post_probe"}
            or reflection.get("active_revision_id") != revision_id
            or reflection.get("attachment_id") != attachment.get("attachment_id")
            or reflection.get("agent_invocation_id") != attachment.get("agent_invocation_id")
            or not isinstance(relevance, list)
            or (route is not None and route not in relevance)
        ):
            continue
        dynamic_refs = [
            dict(item) for item in reflection.get("artifact_refs", [])
            if isinstance(item, Mapping) and item.get("kind") in {"tool_result", "visual_evidence"}
        ]
        if not dynamic_refs:
            invalid.append(dict(reflection))
            continue
        resolver_refs = dynamic_refs
        if (
            SimDiagnosticRoute.MATERIAL_INFERENCE.value in relevance
            and reflection.get("phase") == "post_probe"
            and reflection.get("episode_id") not in {None, "", "pre_episode"}
        ):
            region_id = str(reflection.get("region_id") or "")
            measurements = [
                item for item in diagnostics["probe_measurements"]
                if isinstance(item, Mapping)
                and item.get("region_id") == region_id
                and item.get("episode_id") == reflection.get("episode_id")
            ]
            if len(measurements) == 1 and measurements[0].get("pair_id") is not None:
                measurement = measurements[0]
                pair_records = [
                    item for item in diagnostics["paired_comparison_records"]
                    if isinstance(item, Mapping)
                    and item.get("pair_id") == measurement.get("pair_id")
                    and item.get("policy_fingerprint") == measurement.get("pair_policy_fingerprint")
                ]
                peer_measurements = [
                    item for item in diagnostics["probe_measurements"]
                    if isinstance(item, Mapping)
                    and len(pair_records) == 1
                    and item.get("dispatch_token") in pair_records[0].get("dispatch_tokens", [])
                ]
                required_pair_refs = {
                    f"tool_result:{int(item['completion_tool_result_index']) + 1}"
                    for item in peer_measurements
                }
                cited_pair_refs = {
                    str(item.get("ref")) for item in dynamic_refs
                    if item.get("kind") == "tool_result"
                }
                if (
                    len(pair_records) != 1
                    or len(peer_measurements) != 2
                    or len(required_pair_refs) != 2
                    or not required_pair_refs.issubset(cited_pair_refs)
                ):
                    invalid.append(dict(reflection))
                    continue
                resolver_refs = [
                    item for item in dynamic_refs
                    if (
                        item.get("kind") == "tool_result"
                        and str(item.get("ref"))
                        == f"tool_result:{int(measurement['completion_tool_result_index']) + 1}"
                    )
                ]
                if len(resolver_refs) != 1:
                    invalid.append(dict(reflection))
                    continue
        try:
            _resolve_material_audit_trigger_refs(
                state,
                request={"trigger_evidence_refs": resolver_refs},
                reflection=reflection,
            )
        except GenesisVlmSchemaError:
            invalid.append(dict(reflection))
            continue
        valid.append(dict(reflection))
    return valid, invalid


def _current_material_concern_reflections(
    state: dict[str, Any], diagnostics: dict[str, Any], *, allow_completed_attachment: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _current_owned_probe_evidence_reflections(
        state, diagnostics, route=SimDiagnosticRoute.MATERIAL_INFERENCE.value,
        allow_completed_attachment=allow_completed_attachment,
    )


def _settled_accept_probe_evidence(
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return strict accept evidence from a verified close settlement."""
    completed = [
        item for item in summary["regions"]
        if isinstance(item, Mapping) and item.get("status") == "coverage_complete"
    ]
    if not completed:
        raise GenesisVlmSchemaError("terminal probe evidence requires a coverage_complete settled region")
    candidates = completed
    if not candidates:
        raise GenesisVlmSchemaError("terminal evidence must be rooted in a settled region with current diagnostic lineage")
    selected = candidates[0]
    lineage = selected.get("coverage_evidence")
    if not isinstance(lineage, Mapping):
        raise GenesisVlmSchemaError("coverage_complete settlement lacks verified terminal evidence lineage")
    attempt = lineage.get("qualifying_attempt")
    probe = lineage.get("probe")
    ownership = lineage.get("ownership_timing")
    if not isinstance(attempt, Mapping) or not isinstance(probe, Mapping) or not isinstance(ownership, Mapping):
        raise GenesisVlmSchemaError("coverage_complete settlement terminal evidence is malformed")
    if not attempt.get("attempt_id") or not isinstance(attempt.get("tool_result_index"), int):
        raise GenesisVlmSchemaError("coverage_complete settlement lacks a qualifying probe result")
    return {
        "region_id": selected["region_id"],
        "settlement_id": selected["settlement_id"],
        "episode_id": selected["episode_id"],
        "qualifying_attempt_id": attempt["attempt_id"],
        "tool_result_index": attempt["tool_result_index"],
        "application_index": attempt.get("application_index"),
        "probe_reflection_id": probe.get("reflection_id"),
        "probe_compile_id": probe.get("compile_id"),
        "attachment_id": ownership.get("attachment_id"),
        "agent_invocation_id": ownership.get("agent_invocation_id"),
    }


def _adjudicate_terminal_recommendation(
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    normalized: Mapping[str, Any],
    *,
    allow_completed_attachment: bool = False,
) -> dict[str, Any]:
    """Keep revision admission route+cues-only and adjudicate accept strictly."""
    if normalized["recommendation"] == "revise":
        return {}
    summary = build_diagnostic_coverage_summary(state)
    if summary["open_live_sessions"]:
        raise GenesisVlmSchemaError("accept recommendation requires every live session closed")
    all_complete = bool(summary["regions"]) and all(
        item.get("status") == "coverage_complete" for item in summary["regions"]
    )
    runtime_failure = _current_attachment_has_runtime_failure(
        state, diagnostics, allow_completed=allow_completed_attachment
    )
    if all_complete and not runtime_failure:
        complete_region_ids = {str(item["region_id"]) for item in summary["regions"]}
        judgments_by_region: dict[str, list[Mapping[str, Any]]] = {}
        for reflection in diagnostics.get("model_reflections", []):
            if not isinstance(reflection, Mapping) or reflection.get("phase") != "post_probe":
                continue
            region_id = str(reflection.get("region_id") or "")
            judgment = reflection.get("material_response_judgment")
            if region_id in complete_region_ids and isinstance(judgment, Mapping):
                judgments_by_region.setdefault(region_id, []).append(judgment)
        if set(judgments_by_region) != complete_region_ids or any(len(items) != 1 for items in judgments_by_region.values()):
            raise GenesisVlmSchemaError("terminal recommendation requires one typed post_probe judgment for every coverage_complete region")
    if runtime_failure:
        raise GenesisVlmSchemaError("accept is forbidden after a recorded blocking runtime_failure")
    audits = [
        item for item in diagnostics.get("source_semantic_material_audits", [])
        if isinstance(item, Mapping) and item.get("active_revision_id") == str(state.get("active_revision") or "")
    ]
    if len(audits) > 1:
        raise GenesisVlmSchemaError("terminal gate requires at most one current material audit")
    audit = audits[0] if audits else None
    material_concerns, invalid_material_concerns = _current_material_concern_reflections(
        state, diagnostics, allow_completed_attachment=allow_completed_attachment
    )
    if invalid_material_concerns:
        raise GenesisVlmSchemaError("current material concern lacks owned successful-probe evidence; halt diagnostics")
    if material_concerns and audit is None:
        raise GenesisVlmSchemaError("current material concern requires a unique current material audit before terminal")
    violated = bool(audit and any(item.get("status") == "violated" for item in audit.get("invariants", [])))
    if violated:
        raise GenesisVlmSchemaError("accept is forbidden while a current material audit is violated")
    if not all_complete:
        raise GenesisVlmSchemaError("accept requires every planned region coverage_complete")
    terminal_evidence = _settled_accept_probe_evidence(state, diagnostics, summary)
    return {
        "summary": summary,
        "coverage_timing": "full",
        "evidence_mode": "probe_window",
        "terminal_evidence": terminal_evidence,
    }


def submit_diagnostic_recommendation(run_root: str, recommendation: dict[str, Any]) -> dict[str, Any]:
    """Submit a schema-validated Genesis diagnostic recommendation."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    with _terminal_decision_lock(bound_run_root):
        state = _require_state(bound_run_root)
        diagnostics = _diagnostics(state)
        existing_terminal = diagnostics.get("terminal")
        if isinstance(existing_terminal, Mapping):
            submitted = validate_vlm_final_recommendation(
                _normalize_terminal_recommendation_payload(recommendation)
            )
            persisted = existing_terminal.get("recommendation")
            if not isinstance(persisted, Mapping):
                raise GenesisVlmSchemaError("existing terminal recommendation is missing or malformed")
            normalized_existing = normalize_persisted_vlm_final_recommendation(persisted)
            if normalized_existing != submitted:
                raise GenesisVlmSchemaError("terminal recommendation already exists and conflicts with this submission")
            attempt_index = existing_terminal.get("terminal_attempt_index")
            attempts = diagnostics.get("terminal_attempts", [])
            if (
                isinstance(attempt_index, bool)
                or not isinstance(attempt_index, int)
                or not isinstance(attempts, list)
                or not (0 <= attempt_index < len(attempts))
                or not isinstance(attempts[attempt_index], Mapping)
                or attempts[attempt_index].get("status") != "accepted"
                or existing_terminal.get("terminal_attempt_index") != attempt_index
                or existing_terminal.get("terminal_decision_id") != f"terminal_{attempt_index + 1:04d}"
            ):
                raise GenesisVlmSchemaError("existing terminal recommendation has corrupt decision-attempt lineage")
            attempt_recommendation = attempts[attempt_index].get("normalized_recommendation")
            if not isinstance(attempt_recommendation, Mapping) or (
                normalize_persisted_vlm_final_recommendation(attempt_recommendation)
                != normalized_existing
            ):
                raise GenesisVlmSchemaError("existing terminal recommendation has corrupt normalized attempt lineage")
            expected_ready = normalized_existing["recommendation"] == "accept"
            if existing_terminal.get("route") != normalized_existing["route"] or existing_terminal.get("ready") is not expected_ready:
                raise GenesisVlmSchemaError("existing terminal recommendation conflicts with its route tuple")
            if normalized_existing["recommendation"] == "accept":
                adjudication = _adjudicate_terminal_recommendation(
                    state, diagnostics, normalized_existing, allow_completed_attachment=True
                )
                if (
                    existing_terminal.get("coverage_summary") != adjudication["summary"]
                    or existing_terminal.get("evidence_mode") != "probe_window"
                    or existing_terminal.get("live_probe_evidence") != adjudication["terminal_evidence"]
                ):
                    raise GenesisVlmSchemaError("existing accept recommendation conflicts with its current evidence lineage")
            response = {
                "status": existing_terminal.get("status"),
                "route": existing_terminal.get("route"),
                "ready": expected_ready,
            }
            if normalized_existing["recommendation"] == "accept":
                response["evidence_mode"] = "probe_window"
            return response
        attempt = {
            "tool_name": "submit_diagnostic_recommendation",
            "raw_recommendation": recommendation,
        }
        diagnostics.setdefault("terminal_attempts", []).append(attempt)
        try:
            normalized = validate_vlm_final_recommendation(_normalize_terminal_recommendation_payload(recommendation))
            attempt["normalized_recommendation"] = normalized
            adjudication = _adjudicate_terminal_recommendation(state, diagnostics, normalized)
            if normalized["recommendation"] == "accept":
                attempt["evidence_mode"] = "probe_window"
                attempt["live_probe_evidence"] = adjudication["terminal_evidence"]
        except GenesisVlmSchemaError as exc:
            attempt["error"] = str(exc)
            attempt["status"] = "rejected"
            _save_diagnostics(
                bound_run_root,
                state,
                event="diagnostic_recommendation_rejected_by_schema",
                note=str(exc),
                detail=attempt,
                provenance="terminal_recommendation",
            )
            raise
        attempt["status"] = "accepted"
        terminal_attempt_index = len(diagnostics["terminal_attempts"]) - 1
        terminal_decision_id = f"terminal_{terminal_attempt_index + 1:04d}"
        terminal = {
            "tool_name": "submit_diagnostic_recommendation",
            "status": "pending_validation",
            "validated": False,
            "recommendation": normalized,
            "route": normalized["route"],
            "ready": normalized["recommendation"] == "accept",
            "error": "",
            "terminal_decision_id": terminal_decision_id,
            "terminal_attempt_index": terminal_attempt_index,
        }
        if normalized["recommendation"] == "accept":
            terminal.update(
                {
                    "evidence_mode": "probe_window",
                    "coverage_timing": "full",
                    "coverage_summary": adjudication["summary"],
                    "live_probe_evidence": adjudication["terminal_evidence"],
                }
            )
        diagnostics.pop("route_adjudication", None)
        diagnostics["terminal"] = terminal
        route_state = state.setdefault("route_state", {})
        recommendation_payload = _diagnostic_route_payload(
            state,
            normalized["route"],
            normalized["recommendation"] == "accept",
            evidence_mode="probe_window" if normalized["recommendation"] == "accept" else None,
        )
        route_state["diagnostic_recommendation"] = dict(normalized)
        route_state["diagnostic_route"] = recommendation_payload
        note = normalized.get("reason") or "; ".join(normalized.get("diagnostic_cues", []))
        _save_diagnostics(
            bound_run_root,
            state,
            event="diagnostic_recommendation_submitted",
            note=note,
            detail=terminal,
            provenance="terminal_recommendation",
        )
        response = {
            "status": "pending_validation",
            "route": normalized["route"],
            "ready": normalized["recommendation"] == "accept",
        }
        if normalized["recommendation"] == "accept":
            response["evidence_mode"] = "probe_window"
        return response
def halt_diagnostics(run_root: str, error: str = "") -> dict[str, Any]:
    """Halt the Genesis diagnostic stage."""
    bound_run_root = _require_bound_diagnostic_run_root(run_root)
    with _terminal_decision_lock(bound_run_root):
        state = _require_state(bound_run_root)
        diagnostics = _diagnostics(state)
        require_v2_session_state(state)
        try:
            episode = _active_episode_record(state)
        except ValueError:
            episode = None
        if episode is not None:
            episode["status"] = "halted"
            _write_episode_observations(episode, diagnostics)
        terminal = {
            "tool_name": "halt_diagnostics",
            "status": "halted",
            "validated": True,
            "business_outcome": build_diagnostic_business_outcome("business_halt"),
            "operational_outcome": build_diagnostic_operational_outcome("complete"),
            "route": None,
            "original_route": None,
            "effective_route": None,
            "ready": False,
            "error": error or "diagnostics halted",
            "recommended_stage_skill_paths": [],
            "reentry_stage_skill_paths": [],
            "force_rerun_stage_skill_paths": [],
        }
        diagnostics["terminal"] = terminal
        halt_route = {
            "route": None,
            "original_route": None,
            "effective_route": None,
            "ready": False,
            "status": "halted",
            "error": terminal["error"],
            "recommended_stage_skill_paths": [],
            "force_rerun_stage_skill_paths": [],
            "reentry_stage_skill_paths": [],
        }
        route_state = state.setdefault("route_state", {})
        route_state["diagnostic_recommendation"] = dict(halt_route)
        route_state["diagnostic_route"] = halt_route
        refresh_runtime_owned_diagnostic_files(
            bound_run_root,
            state,
            provenance="terminal_recommendation",
            event="diagnostic_halted",
            note=terminal["error"],
            detail=terminal,
        )
        save_state(state, bound_run_root)
        record_stage(
            bound_run_root,
            DIAGNOSTIC_STAGE_NAME,
            False,
            status="failed",
            error=terminal["error"],
            allow_new=DIAGNOSTIC_STAGE_NAME not in state.get("stages", {}),
        )
        append_history(
            bound_run_root,
            "halt_diagnostics",
            terminal["error"],
            event="diagnostic_halted",
            detail=terminal,
        )
        return {"status": "halted", "route": terminal["route"], "ready": False, "error": terminal["error"]}
def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    route = summary.get("recommended_route", "")
    ready = summary.get("recommended_ready", False)
    episodes = [episode for episode in summary.get("episodes", []) if isinstance(episode, dict)]
    lines = [
        "# HAG4R Genesis Diagnostic Report",
        "",
        f"- recommended_route: {route}",
        f"- recommended_ready: {ready}",
        f"- evidence_mode: {summary.get('evidence_mode', '')}",
        f"- tool_results: {len(summary.get('tool_results', []))}",
        f"- episode_suites: {len(episodes)}",
        "",
        "## Recommendation",
        "",
        str(
            summary.get("diagnostic_recommendation", {}).get("reason")
            or "\n".join(summary.get("diagnostic_recommendation", {}).get("diagnostic_cues", []))
        ),
    ]
    coverage_summary = summary.get("coverage_summary")
    if isinstance(coverage_summary, Mapping):
        lines.extend(
            [
                "",
                "## Coverage Summary",
                "",
                f"- schema_version: {coverage_summary.get('schema_version', '')}",
                f"- selected_region_count: {coverage_summary.get('selected_region_count', '')}",
                "",
                "| region_id | risk_rank | status | episode_id | settlement_id |",
                "|---|---:|---|---|---|",
            ]
        )
        for region in coverage_summary.get("regions", []):
            if isinstance(region, Mapping):
                lines.append(
                    f"| {region.get('region_id', '')} | {region.get('risk_rank', '')} | "
                    f"{region.get('status', '')} | {region.get('episode_id', '') or ''} | "
                    f"{region.get('settlement_id', '') or ''} |"
                )
    route_adjudication = summary.get("route_adjudication")
    if isinstance(route_adjudication, dict) and route_adjudication:
        lines.extend(
            [
                "",
                "## Route Adjudication",
                "",
                f"- selected_route: {route_adjudication.get('selected_route', '')}",
                f"- reflection_index: {route_adjudication.get('reflection_index', '')}",
                f"- phase: {route_adjudication.get('phase', '')}",
                f"- route_relevance: {', '.join(str(item) for item in route_adjudication.get('route_relevance', []))}",
                f"- observation: {route_adjudication.get('observation', '')}",
                f"- reflection: {route_adjudication.get('reflection', '')}",
                f"- artifact_refs: {json.dumps(route_adjudication.get('artifact_refs', []), sort_keys=True)}",
            ]
        )
    if episodes:
        lines.extend(["", "## Episode Suites", ""])
        for episode in episodes:
            video = episode.get("video") if isinstance(episode.get("video"), dict) else {}
            setup_quality = _setup_quality_preview(episode.get("setup_quality"))
            lines.extend(
                [
                    f"### {episode.get('episode_id', '')}",
                    "",
                    f"- status: {episode.get('status', '')}",
                    f"- intent: {episode.get('intent', '')}",
                    f"- scene: {episode.get('scene_config_path', '')}",
                    f"- observations: {episode.get('observations_path', '')}",
                    f"- live_output: {episode.get('live_output_dir', '')}",
                    f"- logs: {episode.get('log_dir', '')}",
                    f"- video: {episode.get('video_path', '')}",
                    f"- video_status: {video.get('status', '')}",
                    f"- setup_trial_id: {episode.get('setup_trial_id', '')}",
                    f"- setup_final_status: {setup_quality.get('final_status', '')}",
                    f"- setup_uncertainty: {setup_quality.get('uncertainty', '')}",
                    f"- setup_concerns: {json.dumps(setup_quality.get('concerns', []), sort_keys=True)}",
                    f"- video_fps: {video.get('video_fps', video.get('fps', ''))}",
                    f"- simulated_duration_s: {video.get('simulated_duration_s', video.get('duration_s', ''))}",
                    f"- video_playback_duration_s: {video.get('video_playback_duration_s', '')}",
                    f"- png_frames: {video.get('frame_count', '')}",
                    "",
                ]
            )
    runtime_owned_files = summary.get("runtime_owned_files")
    if isinstance(runtime_owned_files, dict):
        lines.extend(["", "## Runtime-Owned Workspace Files", ""])
        for key, value in sorted(runtime_owned_files.items()):
            lines.append(f"- {key}: {value}")
    model_reflections = [item for item in summary.get("model_reflections", []) if isinstance(item, dict)]
    if model_reflections:
        lines.extend(["", "## Model Reflections", ""])
        for reflection in model_reflections:
            lines.extend(
                [
                    f"### reflection_{int(reflection.get('reflection_index', 0)):04d}",
                    "",
                    f"- phase: {reflection.get('phase', '')}",
                    f"- episode_id: {reflection.get('episode_id', '')}",
                    f"- observation: {reflection.get('observation', '')}",
                    f"- reflection: {reflection.get('reflection', '')}",
                    f"- uncertainty: {reflection.get('uncertainty', '')}",
                    "",
                ]
            )
    audits = [item for item in summary.get("source_semantic_material_audits", []) if isinstance(item, dict)]
    if audits:
        lines.extend(["", "## Source-Semantic Material Audits", "", "Constitutive material-plan evidence only; not realized structural compliance.", ""])
        for audit in audits:
            lines.extend(
                [
                    f"- audit_id: {audit.get('audit_id', '')}",
                    f"- active_revision_id: {audit.get('active_revision_id', '')}",
                    f"- status_counts: {json.dumps(audit.get('status_counts', {}), sort_keys=True)}",
                    f"- canonical_material_hint_count: {len(audit.get('canonical_material_hints', []))}",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def _format_video_fps(fps: float) -> str:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"video fps must be positive and finite: {fps}")
    if float(fps).is_integer():
        return f"{int(fps)}fps"
    label = f"{fps:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}fps"
def _resolve_scene_output_png(scene: dict[str, Any], legacy_sim_root: Path) -> Path:
    output_png = scene.get("output_png")
    if not isinstance(output_png, str) or not output_png.strip():
        raise ValueError("diagnostic episode scene must define non-empty output_png for video generation")
    png_dir = Path(output_png).expanduser()
    if not png_dir.is_absolute():
        png_dir = legacy_sim_root / png_dir
    return png_dir.resolve()
def _hag4r_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
def _local_generate_simulation_mp4_script_path() -> Path:
    return _hag4r_repo_root() / "scripts" / "generate_simulation_mp4.sh"
def _expected_generate_simulation_mp4_path(output_root: Path, scene_path: Path) -> Path:
    return (output_root / scene_path.parent.name / f"{scene_path.stem}.mp4").resolve()
def _expected_genesis_script_mp4_path(legacy_sim_root: Path, scene_path: Path) -> Path:
    output_root = (
        legacy_sim_root
        / "simulation"
        / "output_mp4"
        / "generate_simulation_mp4"
    )
    return _expected_generate_simulation_mp4_path(output_root, scene_path)
def _load_externalized_live_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("externalized") is True:
        artifact_path = Path(str(value.get("artifact_path", ""))).expanduser()
        if artifact_path.is_file():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}
def _source_view_path_for_triptych_frame(
    *,
    triptych_path: Path,
    frame: Mapping[str, Any] | None,
    view_name: str,
) -> Path:
    if isinstance(frame, Mapping):
        views = frame.get("views")
        if isinstance(views, Mapping):
            view = views.get(view_name)
            if isinstance(view, Mapping):
                source_path = str(view.get("source_png_path", "")).strip()
                if source_path:
                    return Path(source_path).expanduser()
    part_path = triptych_path.parent.parent / "png_part_segmentation_panels" / view_name / triptych_path.name
    if part_path.is_file():
        return part_path
    return triptych_path.parent.parent / "png_rgb_triptych_views" / view_name / triptych_path.name
def _materialize_episode_triptych_sequence_from_live_tool_results(
    state: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any] | None:
    raw_sequence_dir = Path(
        str(episode.get("png_part_segmentation_triptych_dir") or episode.get("png_rgb_triptych_dir", ""))
    ).expanduser().resolve()
    raw_source_view_root = Path(
        str(episode.get("png_part_segmentation_triptych_view_dir") or episode.get("png_rgb_triptych_view_dir", ""))
    ).expanduser().resolve()
    is_part_segmentation = bool(str(episode.get("png_part_segmentation_triptych_dir", "")).strip())
    if is_part_segmentation:
        sequence_dir = Path(
            str(
                episode.get("png_part_segmentation_sequence_dir")
                or raw_sequence_dir.parent / "png_part_segmentation_triptych_sequence"
            )
        ).expanduser().resolve()
        source_view_root = Path(
            str(
                episode.get("png_part_segmentation_sequence_view_dir")
                or raw_source_view_root.parent / "png_part_segmentation_sequence_panels"
            )
        ).expanduser().resolve()
        episode["png_part_segmentation_sequence_dir"] = str(sequence_dir)
        episode["png_part_segmentation_sequence_view_dir"] = str(source_view_root)
    else:
        sequence_dir = raw_sequence_dir
        source_view_root = raw_source_view_root
    if (sequence_dir / "frame_000000.png").is_file():
        frame_count = len([path for path in sequence_dir.glob("frame_*.png") if path.is_file()])
        return {
            "status": "already_present",
            "sequence_dir": str(sequence_dir),
            "source_view_root": str(source_view_root),
            "raw_sequence_dir": str(raw_sequence_dir),
            "raw_source_view_root": str(raw_source_view_root),
            "frame_count": frame_count,
        }
    diagnostics = _diagnostics(state)
    tool_results = diagnostics.get("tool_results", [])
    if not isinstance(tool_results, list):
        return None
    tool_result_indices = [
        int(index)
        for index in episode.get("tool_result_indices", [])
        if isinstance(index, int) and 0 <= int(index) < len(tool_results)
    ]
    if not tool_result_indices:
        return None
    source_frames: list[tuple[Path, Mapping[str, Any] | None]] = []
    for tool_result_index in tool_result_indices:
        record = tool_results[tool_result_index]
        if not isinstance(record, Mapping):
            continue
        if str(record.get("tool", "")) not in LIVE_VISUAL_TOOL_NAMES:
            continue
        result = record.get("result")
        if not isinstance(result, Mapping):
            continue
        live_result = result.get("result")
        if not isinstance(live_result, Mapping):
            continue
        sequence_payload = _load_externalized_live_payload(live_result.get("triple_view_sequence"))
        payload_frames = sequence_payload.get("frames") if isinstance(sequence_payload, Mapping) else None
        frames_by_path: dict[str, Mapping[str, Any]] = {}
        if isinstance(payload_frames, list):
            for frame in payload_frames:
                if not isinstance(frame, Mapping):
                    continue
                triptych_png_path = str(frame.get("triptych_png_path", "")).strip()
                if triptych_png_path:
                    frames_by_path[str(Path(triptych_png_path).expanduser())] = frame
        for triptych_png_path in _string_list(live_result.get("triptych_png_paths")):
            path = Path(triptych_png_path).expanduser()
            if path.is_file():
                source_frames.append((path, frames_by_path.get(str(path))))
    if not source_frames:
        return None
    source_records: list[tuple[Path, dict[str, Path]]] = []
    for triptych_path, frame in source_frames:
        view_paths: dict[str, Path] = {}
        for view_name in TRIPLE_VIEW_PANEL_ORDER:
            source_view_path = _source_view_path_for_triptych_frame(
                triptych_path=triptych_path,
                frame=frame,
                view_name=view_name,
            )
            if not source_view_path.is_file():
                raise FileNotFoundError(f"triptych source view PNG is missing: {source_view_path}")
            view_paths[view_name] = source_view_path
        source_records.append((triptych_path, view_paths))
    staging_dir = sequence_dir.parent / ".triptych_materialize_staging"
    staged = False
    for triptych_path, view_paths in source_records:
        if triptych_path.resolve().is_relative_to(sequence_dir) or any(
            view_path.resolve().is_relative_to(source_view_root)
            for view_path in view_paths.values()
        ):
            staged = True
            break
    if staged:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        staged_records: list[tuple[Path, dict[str, Path]]] = []
        for sequence_index, (triptych_path, view_paths) in enumerate(source_records):
            staged_triptych_path = staging_dir / f"triptych_{sequence_index:06d}.png"
            shutil.copy2(triptych_path, staged_triptych_path)
            staged_view_paths: dict[str, Path] = {}
            for view_name, view_path in view_paths.items():
                staged_view_path = staging_dir / f"{view_name}_{sequence_index:06d}.png"
                shutil.copy2(view_path, staged_view_path)
                staged_view_paths[view_name] = staged_view_path
            staged_records.append((staged_triptych_path, staged_view_paths))
        source_records = staged_records
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for path in sequence_dir.glob("frame_*.png"):
        path.unlink()
    for view_name in TRIPLE_VIEW_PANEL_ORDER:
        view_dir = source_view_root / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for path in view_dir.glob("frame_*.png"):
            path.unlink()
    copied = 0
    try:
        for sequence_index, (triptych_path, view_paths) in enumerate(source_records):
            destination_name = f"frame_{sequence_index:06d}.png"
            shutil.copy2(triptych_path, sequence_dir / destination_name)
            for view_name in TRIPLE_VIEW_PANEL_ORDER:
                shutil.copy2(view_paths[view_name], source_view_root / view_name / destination_name)
            copied += 1
    finally:
        if staged and staging_dir.exists():
            shutil.rmtree(staging_dir)
    metadata = {
        "status": "materialized",
        "source": "live_tool_result_triptych_sequences",
        "sequence_dir": str(sequence_dir),
        "source_view_root": str(source_view_root),
        "raw_sequence_dir": str(raw_sequence_dir),
        "raw_source_view_root": str(raw_source_view_root),
        "frame_count": copied,
    }
    episode["materialized_triptych_sequence"] = metadata
    return metadata
def _episode_video_settings(episode: dict[str, Any], legacy_sim_root: Path) -> dict[str, Any]:
    del legacy_sim_root
    scene_path = Path(str(episode["scene_config_path"])).expanduser().resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError(f"diagnostic episode scene must be a JSON object: {scene_path}")
    render_png = scene.get("render_png") or {}
    if not isinstance(render_png, dict):
        raise ValueError(f"diagnostic episode render_png must be a JSON object: {scene_path}")
    timestep = float(scene.get("timestep"))
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError(f"diagnostic episode scene timestep must be positive: {scene_path}")
    capture_every_n_steps = render_png.get("capture_every_n_steps", 1)
    if isinstance(capture_every_n_steps, bool) or not isinstance(capture_every_n_steps, int) or capture_every_n_steps <= 0:
        raise ValueError(f"render_png.capture_every_n_steps must be a positive integer: {scene_path}")
    png_dir = Path(str(episode.get("png_part_segmentation_sequence_dir", ""))).expanduser().resolve()
    first_png = png_dir / "frame_000000.png"
    png_source_kind = "part_segmentation_triptych"
    video_scene_path = _triptych_scene_config_path(episode, png_dir) if first_png.is_file() else scene_path
    if not first_png.is_file():
        triptych_png_dir = Path(str(episode.get("png_rgb_triptych_dir", ""))).expanduser().resolve()
        triptych_first_png = triptych_png_dir / "frame_000000.png"
        if not triptych_first_png.is_file():
            legacy_rgb_dir = Path(str(episode.get("png_rgb_dir", ""))).expanduser().resolve()
            legacy_rgb_first = legacy_rgb_dir / "frame_000000.png"
            if legacy_rgb_first.is_file():
                png_dir = legacy_rgb_dir
                first_png = legacy_rgb_first
                png_source_kind = "legacy_rgb"
            else:
                raise FileNotFoundError(
                    "diagnostic episode is missing part segmentation triptych and historical RGB sequences"
                )
        else:
            png_dir = triptych_png_dir
            first_png = triptych_first_png
            png_source_kind = "rgb_triptych"
            video_scene_path = _triptych_scene_config_path(episode, triptych_png_dir)
    frame_paths = sorted(png_dir.glob("frame_*.png"))
    frame_count = len([path for path in frame_paths if path.is_file()])
    min_frame_count = min_diagnostic_captured_frames(
        duration_s=DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
        timestep_s=timestep,
        capture_every_n_steps=capture_every_n_steps,
    )
    if frame_count < min_frame_count:
        raise ValueError(
            f"diagnostic episode is shorter than the default "
            f"{DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S:g}s simulate window: "
            f"frame_count={frame_count}, min_frame_count={min_frame_count}, png_dir={png_dir}"
        )
    simulated_duration_s = frame_count * timestep * capture_every_n_steps
    if simulated_duration_s < DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S:
        raise ValueError(
            f"diagnostic episode simulated duration is below the default "
            f"{DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S:g}s simulate window: "
            f"{simulated_duration_s:.3f}s "
            f"from frame_count={frame_count}, timestep={timestep:g}, "
            f"capture_every_n_steps={capture_every_n_steps}"
        )
    if simulated_duration_s > 60.0:
        raise ValueError(
            f"diagnostic episode simulated duration exceeds 60s: {simulated_duration_s:.3f}s "
            f"from frame_count={frame_count}, timestep={timestep:g}, "
            f"capture_every_n_steps={capture_every_n_steps}"
        )
    time_multiplier = 10.0
    video_fps = 1.0 / (timestep * capture_every_n_steps * time_multiplier)
    video_playback_duration_s = frame_count / video_fps
    return {
        "scene_path": video_scene_path,
        "source_scene_path": scene_path,
        "png_dir": png_dir,
        "png_source_kind": png_source_kind,
        "timestep": timestep,
        "capture_every_n_steps": capture_every_n_steps,
        "fps": video_fps,
        "video_fps": video_fps,
        "fps_label": _format_video_fps(video_fps),
        "frame_count": frame_count,
        "min_frame_count": min_frame_count,
        "min_duration_s": DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
        "default_simulate_duration_s": DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
        "default_simulate_steps": DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
        "duration_s": simulated_duration_s,
        "simulated_duration_s": simulated_duration_s,
        "video_playback_duration_s": video_playback_duration_s,
        "time_multiplier": time_multiplier,
    }
def _generate_episode_video(run_root: str | Path, state: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    script_root = _hag4r_repo_root()
    script_path = _local_generate_simulation_mp4_script_path()
    if not script_path.is_file():
        raise FileNotFoundError(f"HAG4R MP4 script is missing: {script_path}")
    _materialize_episode_triptych_sequence_from_live_tool_results(state, episode)
    settings = _episode_video_settings(episode, Path("/"))
    episode_id = str(episode["episode_id"])
    generated_episode_dir = Path(str(episode["generated_episode_dir"])).expanduser().resolve()
    episode_log_dir = Path(str(episode["log_dir"])).expanduser().resolve()
    generated_episode_dir.mkdir(parents=True, exist_ok=True)
    episode_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = episode_log_dir / "generate_simulation_mp4.stdout.log"
    stderr_path = episode_log_dir / "generate_simulation_mp4.stderr.log"
    cmd_path = episode_log_dir / "generate_simulation_mp4.cmd.txt"
    encoder_output_root = episode_log_dir / "generate_simulation_mp4_output"
    source_mp4 = _expected_generate_simulation_mp4_path(encoder_output_root, settings["scene_path"])
    if settings["png_source_kind"] == "part_segmentation_triptych":
        destination_name = f"{episode_id}_part_segmentation_triptych_{settings['fps_label']}.mp4"
    elif settings["png_source_kind"] == "rgb_triptych":
        destination_name = f"{episode_id}_triptych_{settings['fps_label']}.mp4"
    else:
        destination_name = f"{episode_id}_{settings['fps_label']}.mp4"
    destination_mp4 = generated_episode_dir / destination_name
    if source_mp4.exists():
        source_mp4.unlink()
    cmd = [
        str(script_path),
        str(settings["scene_path"]),
        f"{settings['time_multiplier']:.12g}",
    ]
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["HAG4R_MP4_OUTPUT_ROOT"] = str(encoder_output_root)
    env["HAG4R_PYTHON_BIN"] = sys.executable
    env["HAG4R_LEGACY_SIM_ROOT"] = "/"
    completed = subprocess.run(
        cmd,
        cwd=script_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Genesis MP4 generation failed "
            f"(stdout: {stdout_path}, stderr: {stderr_path}, returncode: {completed.returncode})"
        )
    if not source_mp4.is_file() or source_mp4.stat().st_size <= 0:
        raise FileNotFoundError(f"Genesis MP4 script did not produce a non-empty file: {source_mp4}")
    shutil.copy2(source_mp4, destination_mp4)
    if not destination_mp4.is_file() or destination_mp4.stat().st_size <= 0:
        raise FileNotFoundError(f"diagnostic episode MP4 copy is missing or empty: {destination_mp4}")
    metadata = {
        "status": "success",
        "path": str(destination_mp4),
        "fps": settings["fps"],
        "video_fps": settings["video_fps"],
        "fps_label": settings["fps_label"],
        "duration_s": settings["duration_s"],
        "simulated_duration_s": settings["simulated_duration_s"],
        "video_playback_duration_s": settings["video_playback_duration_s"],
        "frame_count": settings["frame_count"],
        "min_frame_count": settings["min_frame_count"],
        "min_duration_s": settings["min_duration_s"],
        "png_dir": str(settings["png_dir"]),
        "png_source_kind": settings["png_source_kind"],
        "source_scene_path": str(settings["source_scene_path"]),
        "png_pattern": "frame_%06d.png",
        "capture_every_n_steps": settings["capture_every_n_steps"],
        "timestep": settings["timestep"],
        "time_multiplier": settings["time_multiplier"],
        "encoder_script": str(script_path),
        "encoder_workdir": str(script_root),
        "encoder_output_root": str(encoder_output_root),
        "encoder_output_path": str(source_mp4),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(cmd_path),
    }
    episode["video"] = metadata
    episode["video_path"] = str(destination_mp4)
    triptych_sequence = _episode_triptych_sequence_from_episode_frames(
        state,
        episode,
        legacy_settings=settings,
    )
    triptych_video = _generate_episode_triptych_video(
        run_root,
        state,
        episode,
        triptych_sequence=triptych_sequence,
        legacy_settings=settings,
    )
    media_validation = _episode_probe_target_media_validation(
        state,
        episode,
        triptych_sequence=triptych_sequence,
        triptych_video=triptych_video,
    )
    if media_validation is not None and media_validation["status"] == "error":
        raise ValueError(
            "diagnostic probe target media validation failed: "
            + ", ".join(str(error) for error in media_validation["hard_errors"])
        )
    metadata["triptych_png_sequence"] = triptych_sequence
    metadata["triptych_video"] = triptych_video
    if media_validation is not None:
        metadata["probe_target_media_validation"] = media_validation
    _write_episode_observations(episode, diagnostics)
    refresh_runtime_owned_diagnostic_files(
        run_root,
        state,
        provenance="video_generation",
        event="diagnostic_episode_video_generated",
        note=f"generated diagnostic video for {episode_id}",
        detail={**metadata, "episode_id": episode_id},
    )
    return metadata
def _episode_triptych_sequence_from_episode_frames(
    state: dict[str, Any],
    episode: dict[str, Any],
    *,
    legacy_settings: dict[str, Any],
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    generated_episode_dir = Path(str(episode["generated_episode_dir"])).expanduser().resolve()
    sequence_dir = Path(
        str(
            episode.get("png_part_segmentation_sequence_dir")
            or episode.get("png_rgb_triptych_dir")
            or generated_episode_dir / "live_output" / "png_rgb_triptych"
        )
    ).expanduser().resolve()
    source_view_root = Path(
        str(
            episode.get("png_part_segmentation_sequence_view_dir")
            or episode.get("png_rgb_triptych_view_dir")
            or generated_episode_dir / "live_output" / "png_rgb_triptych_views"
        )
    ).expanduser().resolve()
    frame_paths = sorted(sequence_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(
            f"cannot build episode triptych PNG sequence for {episode_id}: "
            f"no stitched top/ne_3q/sw_3q simulation frames were recorded in {sequence_dir}"
        )
    min_frame_count = int(legacy_settings.get("min_frame_count", 0))
    if len(frame_paths) < min_frame_count:
        raise ValueError(
            f"diagnostic episode triptych sequence is shorter than required: "
            f"frame_count={len(frame_paths)}, min_frame_count={min_frame_count}, png_dir={sequence_dir}"
        )
    frames: list[dict[str, Any]] = []
    for sequence_index, frame_path in enumerate(frame_paths):
        expected_frame_path = sequence_dir / f"frame_{sequence_index:06d}.png"
        if frame_path != expected_frame_path:
            raise ValueError(
                f"diagnostic triptych frames must be contiguous from frame_000000.png: "
                f"expected {expected_frame_path}, found {frame_path}"
            )
        views: dict[str, dict[str, Any]] = {}
        for view_name in TRIPLE_VIEW_PANEL_ORDER:
            source_view_path = source_view_root / view_name / f"frame_{sequence_index:06d}.png"
            if not source_view_path.is_file():
                raise FileNotFoundError(f"triptych source view PNG is missing: {source_view_path}")
            views[view_name] = {
                "source_png_path": str(source_view_path),
                "source_sha256": sha256_file(source_view_path),
            }
        frames.append(
            {
                "sequence_index": sequence_index,
                "triptych_png_path": str(frame_path),
                "triptych_sha256": sha256_file(frame_path),
                "views": views,
            }
        )
    evidence_id = _next_triple_view_evidence_id(state)
    manifest_path = sequence_dir / f"{episode_id}_triptych_sequence_manifest.json"
    is_part_segmentation = legacy_settings.get("png_source_kind") == "part_segmentation_triptych"
    source_extra = (
        {
            "part_segmentation_png_dir": str(legacy_settings.get("png_dir", "")),
            "part_segmentation_frame_count": int(legacy_settings.get("frame_count", 0)),
        }
        if is_part_segmentation
        else {
            "legacy_rgb_png_dir": str(legacy_settings.get("png_dir", "")),
            "legacy_rgb_frame_count": int(legacy_settings.get("frame_count", 0)),
        }
    )
    manifest = write_triple_view_manifest(
        manifest_path,
        evidence_id=evidence_id,
        source_kind=(
            "episode_part_segmentation_png_sequence" if is_part_segmentation else "episode_png_sequence"
        ),
        episode_id=episode_id,
        sequence_dir=sequence_dir,
        source_view_dirs={view_name: source_view_root / view_name for view_name in TRIPLE_VIEW_PANEL_ORDER},
        frames=frames,
        extra={
            "manifest_path": str(manifest_path),
            "source_strategy": "hag4r_live_resume_chunk_requested_camera_triptychs",
            **source_extra,
        },
    )
    manifest["manifest_path"] = str(manifest_path)
    record = _compact_triple_view_record(
        {
            **manifest,
            "triptych_png_path": str(sequence_dir),
            "triptych_dimensions": {},
            "triptych_sha256": "",
        }
    )
    _append_triple_view_evidence_once(state, record)
    metadata = {
        "status": "success",
        "source_kind": "episode_part_segmentation_png_sequence" if is_part_segmentation else "episode_png_sequence",
        "evidence_id": evidence_id,
        "sequence_dir": str(sequence_dir),
        "manifest_path": str(manifest_path),
        "frame_count": len(frames),
        "source_strategy": "hag4r_live_resume_chunk_requested_camera_triptychs",
        "source_view_root": str(source_view_root),
        "source_view_dirs": {
            view_name: str(source_view_root / view_name) for view_name in TRIPLE_VIEW_PANEL_ORDER
        },
        "frames": frames,
    }
    episode["triptych_png_sequence"] = metadata
    return metadata
def _triptych_scene_config_path(episode: dict[str, Any], triptych_sequence_dir: Path) -> Path:
    source_scene_path = Path(str(episode["scene_config_path"])).expanduser().resolve()
    scene = json.loads(source_scene_path.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError(f"diagnostic episode scene must be a JSON object: {source_scene_path}")
    scene["output_png"] = str(triptych_sequence_dir)
    render_png = scene.get("render_png")
    if isinstance(render_png, dict):
        render_png = dict(render_png)
        render_png["camera"] = {
            "source": "stitched_top_ne_3q_sw_3q_triptych_sequence",
            "real_cameras_recorded_in": episode.get("triptych_png_sequence", {}).get("manifest_path", ""),
        }
        scene["render_png"] = render_png
    sidecar_path = Path(str(episode["log_dir"])).expanduser().resolve() / "triptych_scene_for_mp4.json"
    _write_json(sidecar_path, scene)
    return sidecar_path
def _generate_episode_triptych_video(
    run_root: str | Path,
    state: dict[str, Any],
    episode: dict[str, Any],
    *,
    triptych_sequence: dict[str, Any],
    legacy_settings: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = _diagnostics(state)
    script_root = _hag4r_repo_root()
    script_path = _local_generate_simulation_mp4_script_path()
    if not script_path.is_file():
        raise FileNotFoundError(f"HAG4R MP4 script is missing: {script_path}")
    episode_id = str(episode["episode_id"])
    generated_episode_dir = Path(str(episode["generated_episode_dir"])).expanduser().resolve()
    episode_log_dir = Path(str(episode["log_dir"])).expanduser().resolve()
    triptych_sequence_dir = Path(str(triptych_sequence["sequence_dir"])).expanduser().resolve()
    scene_path = _triptych_scene_config_path(episode, triptych_sequence_dir)
    stdout_path = episode_log_dir / "generate_triptych_simulation_mp4.stdout.log"
    stderr_path = episode_log_dir / "generate_triptych_simulation_mp4.stderr.log"
    cmd_path = episode_log_dir / "generate_triptych_simulation_mp4.cmd.txt"
    encoder_output_root = episode_log_dir / "generate_triptych_simulation_mp4_output"
    source_mp4 = _expected_generate_simulation_mp4_path(encoder_output_root, scene_path)
    if legacy_settings.get("png_source_kind") == "part_segmentation_triptych":
        destination_name = f"{episode_id}_part_segmentation_triptych_{legacy_settings['fps_label']}.mp4"
    else:
        destination_name = f"{episode_id}_triptych_{legacy_settings['fps_label']}.mp4"
    destination_mp4 = generated_episode_dir / destination_name
    if source_mp4.exists():
        source_mp4.unlink()
    cmd = [str(script_path), str(scene_path), f"{legacy_settings['time_multiplier']:.12g}"]
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["HAG4R_MP4_OUTPUT_ROOT"] = str(encoder_output_root)
    env["HAG4R_PYTHON_BIN"] = sys.executable
    env["HAG4R_LEGACY_SIM_ROOT"] = "/"
    completed = subprocess.run(
        cmd,
        cwd=script_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Genesis triptych MP4 generation failed "
            f"(stdout: {stdout_path}, stderr: {stderr_path}, returncode: {completed.returncode})"
        )
    if not source_mp4.is_file() or source_mp4.stat().st_size <= 0:
        raise FileNotFoundError(f"Genesis MP4 script did not produce a non-empty triptych file: {source_mp4}")
    shutil.copy2(source_mp4, destination_mp4)
    if not destination_mp4.is_file() or destination_mp4.stat().st_size <= 0:
        raise FileNotFoundError(f"diagnostic triptych MP4 copy is missing or empty: {destination_mp4}")
    evidence_id = _next_triple_view_evidence_id(state)
    metadata = {
        "status": "success",
        "schema_version": TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION,
        "source_kind": (
            "episode_part_segmentation_triptych_mp4"
            if legacy_settings.get("png_source_kind") == "part_segmentation_triptych"
            else "episode_triptych_mp4"
        ),
        "evidence_id": evidence_id,
        "source_sequence_evidence_id": str(triptych_sequence["evidence_id"]),
        "triptych_png_sequence_dir": str(triptych_sequence_dir),
        "path": str(destination_mp4),
        "mp4_path": str(destination_mp4),
        "mp4_sha256": sha256_file(destination_mp4),
        "fps": legacy_settings["fps"],
        "frame_count": int(triptych_sequence["frame_count"]),
        "encoder_script": str(script_path),
        "encoder_workdir": str(script_root),
        "encoder_input_scene_path": str(scene_path),
        "encoder_output_root": str(encoder_output_root),
        "encoder_output_path": str(source_mp4),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(cmd_path),
        "time_multiplier": legacy_settings["time_multiplier"],
    }
    episode["triptych_video"] = metadata
    episode["triptych_video_path"] = str(destination_mp4)
    _append_triple_view_evidence_once(
        state,
        {
            "schema_version": TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "source_kind": metadata["source_kind"],
            "episode_id": episode_id,
            "triptych_png_path": str(destination_mp4),
            "triple_view_manifest_path": str(scene_path),
            "triptych_sha256": metadata["mp4_sha256"],
            "panel_order": list(TRIPLE_VIEW_PANEL_ORDER),
            "validation": {"status": "ok", "warning_codes": [], "hard_errors": []},
        },
    )
    _write_episode_observations(episode, diagnostics)
    refresh_runtime_owned_diagnostic_files(
        run_root,
        state,
        provenance="video_generation",
        event="diagnostic_episode_triptych_video_generated",
        note=f"generated diagnostic triptych video for {episode_id}",
        detail={**metadata, "episode_id": episode_id},
    )
    return metadata
def _episode_probe_target_media_validation(
    state: dict[str, Any],
    episode: dict[str, Any],
    *,
    triptych_sequence: dict[str, Any],
    triptych_video: dict[str, Any],
) -> dict[str, Any] | None:
    diagnostics = _diagnostics(state)
    episode_id = str(episode["episode_id"])
    validations = [
        validation
        for validation in diagnostics.get("probe_target_validations", [])
        if isinstance(validation, dict) and str(validation.get("episode_id", "")) == episode_id
    ]
    if not validations:
        return None
    validation = validations[-1]
    compile_id = str(validation.get("compile_id", ""))
    target_id = str(validation.get("target_id", ""))
    hard_errors: list[str] = []
    warning_codes: list[str] = []
    def append_warning(code: str) -> None:
        if code not in warning_codes:
            warning_codes.append(code)
    source_warning_codes = (
        [str(code) for code in validation.get("warning_codes", []) if str(code)]
        if isinstance(validation.get("warning_codes"), list)
        else []
    )
    source_hard_errors = (
        [str(error) for error in validation.get("hard_errors", []) if str(error)]
        if isinstance(validation.get("hard_errors"), list)
        else []
    )
    source_status = str(validation.get("status", ""))
    low_fraction_only_warning = (
        source_status == "warning"
        and not source_hard_errors
        and source_warning_codes == ["low_grabbed_selected_part_fraction"]
    )
    if source_status != "ok" or source_hard_errors:
        if low_fraction_only_warning:
            append_warning("grabbed_vertex_validation_not_ok")
        else:
            hard_errors.append("grabbed_vertex_validation_not_ok")
    if float(validation.get("grabbed_selected_part_fraction", 0.0) or 0.0) < MIN_GRABBED_SELECTED_PART_FRACTION:
        append_warning("low_grabbed_selected_part_fraction")
    static_triptychs = [
        evidence
        for evidence in diagnostics.get("triple_view_evidence", [])
        if isinstance(evidence, dict)
        and str(evidence.get("source_kind", "")) == "static_probe_preview"
        and str(evidence.get("compile_id", "")) == compile_id
        and str(evidence.get("target_id", "")) == target_id
    ]
    if not static_triptychs:
        append_warning("missing_static_probe_preview_triptych")
    live_triptychs = [
        evidence
        for evidence in diagnostics.get("triple_view_evidence", [])
        if isinstance(evidence, dict)
        and str(evidence.get("source_kind", ""))
        in {"live_pause_observation", "live_reset_observation", "live_simulate_observation"}
        and str(evidence.get("episode_id", "")) == episode_id
    ]
    if len(live_triptychs) < 2:
        append_warning("missing_pre_or_post_live_triptych")
    if int(triptych_sequence.get("frame_count", 0) or 0) < min_diagnostic_captured_frames():
        hard_errors.append("episode_triptych_sequence_too_short")
    completed_simulated_steps = _completed_simulated_steps_for_episode(diagnostics, episode)
    max_completed_simulate_steps = _max_completed_simulate_steps_for_episode(diagnostics, episode)
    if max_completed_simulate_steps < DEFAULT_DIAGNOSTIC_SIMULATE_STEPS:
        hard_errors.append("episode_simulated_steps_below_default_window")
    sequence_dir = Path(str(triptych_sequence.get("sequence_dir", ""))).expanduser()
    if not sequence_dir.is_dir() or not (sequence_dir / "frame_000000.png").is_file():
        hard_errors.append("missing_episode_triptych_png_sequence")
    mp4_path = Path(str(triptych_video.get("mp4_path") or triptych_video.get("path", ""))).expanduser()
    if not mp4_path.is_file() or mp4_path.stat().st_size <= 0:
        hard_errors.append("missing_episode_triptych_mp4")
    if Path(str(triptych_video.get("encoder_script", ""))).name != "generate_simulation_mp4.sh":
        hard_errors.append("triptych_mp4_not_generated_by_expected_script")
    if str(triptych_video.get("triptych_png_sequence_dir", "")) != str(sequence_dir):
        hard_errors.append("triptych_mp4_source_sequence_mismatch")
    status = "error" if hard_errors else ("warning" if warning_codes else "ok")
    record = {
        "schema_version": "hag4r-diagnostic-probe-target-media-validation-v1",
        "validation_index": len(diagnostics["probe_target_media_validations"]),
        "status": status,
        "target_id": target_id,
        "compile_id": compile_id,
        "episode_id": episode_id,
        "source_probe_target_validation_index": validation.get("validation_index"),
        "source_probe_target_tool_result_index": validation.get("tool_result_index"),
        "grabbed_selected_part_fraction": validation.get("grabbed_selected_part_fraction"),
        "static_preview_triple_view_evidence_ids": [
            str(evidence.get("evidence_id", "")) for evidence in static_triptychs
        ],
        "live_pause_triple_view_evidence_ids": [
            str(evidence.get("evidence_id", "")) for evidence in live_triptychs
        ],
        "triptych_png_sequence": {
            "evidence_id": str(triptych_sequence.get("evidence_id", "")),
            "sequence_dir": str(sequence_dir),
            "frame_count": int(triptych_sequence.get("frame_count", 0) or 0),
            "min_frame_count": min_diagnostic_captured_frames(),
        },
        "completed_simulated_steps": completed_simulated_steps,
        "max_completed_simulate_steps": max_completed_simulate_steps,
        "default_simulate_steps": DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
        "triptych_video": {
            "evidence_id": str(triptych_video.get("evidence_id", "")),
            "path": str(mp4_path),
            "encoder_script": str(triptych_video.get("encoder_script", "")),
            "triptych_png_sequence_dir": str(triptych_video.get("triptych_png_sequence_dir", "")),
        },
        "warning_codes": warning_codes,
        "hard_errors": hard_errors,
    }
    diagnostics["probe_target_media_validations"].append(record)
    episode["probe_target_media_validation"] = record
    return record
def _generate_diagnostic_episode_videos(run_root: str | Path, state: dict[str, Any]) -> None:
    diagnostics = _diagnostics(state)
    episodes = [
        episode
        for episode in diagnostics.get("episodes", [])
        if isinstance(episode, dict)
    ]
    if not episodes:
        raise ValueError("diagnostic success requires at least one episode suite and part segmentation triptych sequence")
    existing_media_complete = True
    for episode in episodes:
        triptych_video = episode.get("triptych_video")
        triptych_path = episode.get("triptych_video_path")
        if not isinstance(triptych_video, Mapping) or not isinstance(triptych_path, str) or not triptych_path:
            existing_media_complete = False
            break
        path = Path(triptych_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            existing_media_complete = False
            break
        frame_count = int(triptych_video.get("frame_count", 0) or 0)
        if frame_count < min_diagnostic_captured_frames():
            existing_media_complete = False
            break
    if existing_media_complete:
        return
    triptych_sequences: list[dict[str, Any]] = []
    triptych_videos: list[dict[str, Any]] = []
    strict_episode_media = (
        state.get("diagnostics_stage_only") is True
        or state.get("diagnostics_reroute_and_export") is True
    )
    sorted_episodes = sorted(episodes, key=lambda item: int(item.get("episode_index", 0)))
    for episode in sorted_episodes:
        metadata = _materialize_episode_triptych_sequence_from_live_tool_results(state, episode)
        if metadata is None:
            if strict_episode_media:
                raise ValueError(
                    "diagnostic success requires a part segmentation triptych sequence for every episode: "
                    f"{episode.get('episode_id') or episode.get('episode_index')}"
                )
            continue
        if int(metadata.get("frame_count", 0) or 0) <= 0:
            if strict_episode_media:
                raise ValueError(
                    "diagnostic success requires positive part segmentation triptych frame coverage for every episode: "
                    f"{episode.get('episode_id') or episode.get('episode_index')}"
                )
            continue
        settings = _episode_video_settings(episode, Path("/"))
        triptych_sequence = _episode_triptych_sequence_from_episode_frames(
            state,
            episode,
            legacy_settings=settings,
        )
        triptych_video = _generate_episode_triptych_video(
            run_root,
            state,
            episode,
            triptych_sequence=triptych_sequence,
            legacy_settings=settings,
        )
        episode["video"] = {"status": "not_generated", "reason": "Genesis diagnostics attach part segmentation triptych PNG sequences"}
        episode["video_path"] = ""
        _write_episode_observations(episode, diagnostics)
        triptych_sequences.append(triptych_sequence)
        triptych_videos.append(triptych_video)
    if not triptych_sequences:
        raise ValueError("diagnostic success requires at least one episode with a valid part segmentation triptych PNG sequence")
    diagnostics["videos"] = []
    diagnostics["video_paths"] = []
    diagnostics["triptych_videos"] = triptych_videos
    diagnostics["triptych_video_paths"] = [str(video.get("mp4_path") or video.get("path")) for video in triptych_videos]
    diagnostics["triptych_sequences"] = triptych_sequences
    refresh_runtime_owned_diagnostic_files(
        run_root,
        state,
        provenance="visual_sequence_materialization",
        event="diagnostic_episode_triptych_sequences_attached",
        note="attached Genesis diagnostic part segmentation triptych sequence metadata and MP4 previews",
        detail={"triptych_sequences": triptych_sequences, "triptych_videos": triptych_videos},
    )
def _write_diagnostic_artifacts(state: dict[str, Any], agent_record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = state["diagnostics"]
    terminal = diagnostics.get("terminal")
    if not isinstance(terminal, dict):
        require_v2_session_state(state)
        raise GenesisVlmSchemaError("diagnostic artifacts require a terminal recommendation")
    recommendation = terminal["recommendation"]
    is_accept = recommendation["recommendation"] == "accept"
    if is_accept:
        require_v2_session_state(state)
    coverage_summary = build_diagnostic_coverage_summary(state) if is_accept else None
    paths = state["paths"]
    report_path = Path(str(paths["sim_diagnostics_report_path"]))
    markdown_path = Path(str(paths["sim_diagnostics_markdown_report_path"]))
    cues_path = Path(str(paths["sim_diagnostics_cues_path"]))
    tool_results = list(diagnostics.get("tool_results", []))
    episodes = list(diagnostics.get("episodes", []))
    videos = list(diagnostics.get("videos", []))
    video_paths = list(diagnostics.get("video_paths", []))
    triptych_videos = list(diagnostics.get("triptych_videos", []))
    triptych_video_paths = list(diagnostics.get("triptych_video_paths", []))
    triple_view_evidence = list(diagnostics.get("triple_view_evidence", []))
    anchor_target_intents = [
        _anchor_target_intent_summary(intent)
        for intent in diagnostics.get("anchor_target_intents", [])
        if isinstance(intent, Mapping)
    ]
    compiled_anchor_targets = [
        _compiled_anchor_target_summary(compiled)
        for compiled in diagnostics.get("compiled_anchor_targets", [])
        if isinstance(compiled, Mapping)
    ]
    anchor_target_preview_artifacts = [
        _anchor_target_preview_summary(preview)
        for preview in diagnostics.get("anchor_target_preview_artifacts", [])
        if isinstance(preview, Mapping)
    ]
    runtime_owned_files = _runtime_file_refs(state)
    route_adjudication = diagnostics.get("route_adjudication", {})
    evidence_mode = terminal.get("evidence_mode") if is_accept else None
    terminal_evidence_key = (
        "live_probe_evidence" if is_accept else None
    )
    terminal_evidence = terminal.get(terminal_evidence_key, {}) if terminal_evidence_key else None
    episodes_by_index = {
        int(episode["episode_index"]): episode
        for episode in episodes
        if isinstance(episode, dict) and episode.get("episode_index") is not None
    }
    summary = {
        "schema_version": "hag4r-agentic-sim-diagnostics-react-v1",
        "run_id": state.get("run_id", ""),
        "agent_record": dict(agent_record),
        "recommended_route": recommendation["route"],
        "recommended_ready": is_accept,
        "forced_final_after_budget": False,
        "diagnostic_recommendation": recommendation,
        "tool_results": tool_results,
        "episodes": episodes,
        "active_episode_index": diagnostics.get("active_episode_index"),
        "videos": videos,
        "video_paths": video_paths,
        "triptych_videos": triptych_videos,
        "triptych_video_paths": triptych_video_paths,
        "expected_observations": list(diagnostics.get("expected_observations", [])),
        "actual_observations": list(diagnostics.get("actual_observations", [])),
        "observations": list(diagnostics.get("actual_observations", [])),
        "visual_evidence": list(diagnostics.get("visual_evidence", [])),
        "triple_view_evidence": triple_view_evidence,
        "anchor_target_intents": anchor_target_intents,
        "active_anchor_target_intent_by_anchor": dict(diagnostics.get("active_anchor_target_intent_by_anchor", {}))
        if isinstance(diagnostics.get("active_anchor_target_intent_by_anchor"), Mapping)
        else {},
        "compiled_anchor_targets": compiled_anchor_targets,
        "active_compiled_anchor_target_by_anchor": dict(diagnostics.get("active_compiled_anchor_target_by_anchor", {}))
        if isinstance(diagnostics.get("active_compiled_anchor_target_by_anchor"), Mapping)
        else {},
        "anchor_target_edit_trials": _redact_internal_anchor_target(list(diagnostics.get("anchor_target_edit_trials", []))),
        "anchor_target_preview_artifacts": anchor_target_preview_artifacts,
        "model_reflections": list(diagnostics.get("model_reflections", [])),
        "source_semantic_material_audits": list(diagnostics.get("source_semantic_material_audits", [])),
        "source_semantic_material_audit_scope": "constitutive material-plan evidence only; not realized structural compliance",
        "route_adjudication": route_adjudication,
        "runtime_owned_events": list(diagnostics.get("runtime_owned_events", [])),
        "runtime_owned_files": runtime_owned_files,
        "probe_report": {
            "diagnostic_run_index": 1,
            "episode_count": len(episodes),
            "route_adjudication": route_adjudication,
            "runtime_owned_files": runtime_owned_files,
            "triple_view_evidence": triple_view_evidence,
            "anchor_target_intents": anchor_target_intents,
            "compiled_anchor_targets": compiled_anchor_targets,
            "anchor_target_preview_artifacts": anchor_target_preview_artifacts,
            "report_refs": {
                "diagnostic_summary": str(report_path),
                "diagnostic_report": str(markdown_path),
                "diagnostic_cues": str(cues_path),
            },
            "probes": [
                {
                    "tool": result["tool"],
                    "expected_observation": result["expected_observation"],
                    "actual_observation": result["actual_observation"],
                    "result": result.get("result", {}).get("status", ""),
                    "episode_index": result.get("episode_index"),
                    "episode_id": result.get("episode_id", ""),
                    "episode_dir": result.get("episode_dir", ""),
                    "scene_config_path": result.get("scene_config_path", ""),
                    "observations_path": result.get("observations_path", ""),
                    "video_path": episodes_by_index.get(int(result.get("episode_index") or 0), {}).get("video_path", ""),
                    "video": episodes_by_index.get(int(result.get("episode_index") or 0), {}).get("video", {}),
                    "triptych_video_path": episodes_by_index.get(int(result.get("episode_index") or 0), {}).get("triptych_video_path", ""),
                    "triptych_video": episodes_by_index.get(int(result.get("episode_index") or 0), {}).get("triptych_video", {}),
                }
                for result in tool_results
            ],
            "recommended_route": recommendation["route"],
            "reason": recommendation.get("reason", ""),
            "diagnostic_cues": recommendation.get("diagnostic_cues", []),
        },
        "diagnostic_runs": [
            {
                "diagnostic_run_index": 1,
                "route": recommendation["route"],
                "ready": is_accept,
                "tool_results": tool_results,
                "episodes": episodes,
                "videos": videos,
                "video_paths": video_paths,
                "triptych_videos": triptych_videos,
                "triptych_video_paths": triptych_video_paths,
                "triple_view_evidence": triple_view_evidence,
                "anchor_target_intents": anchor_target_intents,
                "compiled_anchor_targets": compiled_anchor_targets,
                "anchor_target_preview_artifacts": anchor_target_preview_artifacts,
                "vlm_diagnostic_session_plans": list(diagnostics.get("diagnostic_session_plans", [])),
                "vlm_pre_episode_plans": list(diagnostics.get("pre_episode_plans", [])),
                "diagnostic_recommendation": recommendation,
                "route_adjudication": route_adjudication,
                "agent_attempts": [agent_record],
            }
        ],
    }
    if is_accept:
        summary.update(
            {
                "evidence_mode": evidence_mode,
                "live_probe_evidence": terminal_evidence,
                "coverage_summary": coverage_summary,
            }
        )
        summary["probe_report"].update(
            {
                "evidence_mode": evidence_mode,
                "live_probe_evidence": terminal_evidence,
                "coverage_summary": coverage_summary,
            }
        )
        summary["diagnostic_runs"][0].update(
            {
                "evidence_mode": evidence_mode,
                "live_probe_evidence": terminal_evidence,
                "coverage_summary": coverage_summary,
            }
        )
        cues = {
            "route": recommendation["route"],
            "ready": True,
            "evidence_mode": evidence_mode,
            "live_probe_evidence": terminal_evidence,
            "coverage_summary": coverage_summary,
            "source_semantic_material_audit_lineage": {
            "active_audit_id": diagnostics.get("active_source_semantic_material_audit_id"),
            "audits": [
                {
                    "audit_id": audit.get("audit_id", ""),
                    "active_revision_id": audit.get("active_revision_id", ""),
                    "status_counts": audit.get("status_counts", {}),
                }
                for audit in diagnostics.get("source_semantic_material_audits", [])
                if isinstance(audit, Mapping)
            ],
            "scope": "constitutive material-plan evidence only; not realized structural compliance",
            },
        }
    else:
        cues = {
            "route": recommendation["route"],
            "diagnostic_cues": list(recommendation["diagnostic_cues"]),
        }
    _write_json(report_path, summary)
    _write_json(cues_path, cues)
    _write_markdown(markdown_path, summary)
    return {
        "summary": str(report_path),
        "markdown": str(markdown_path),
        "cues": str(cues_path),
        "videos": video_paths,
        "triptych_videos": triptych_video_paths,
        "workspace_index": str(paths["diagnostic_workspace_index_path"]),
        "workspace_markdown_index": str(paths["diagnostic_workspace_markdown_index_path"]),
        "diagnostic_log": str(paths["diagnostic_log_path"]),
        "observations_digest": str(paths["diagnostic_observations_digest_path"]),
    }
_DIAGNOSTIC_BUSINESS_OUTCOME_STATUSES = frozenset(
    {"not_authored", "recommendation_valid", "business_halt"}
)
_DIAGNOSTIC_OPERATIONAL_OUTCOME_STATUSES = frozenset(
    {
        "pending",
        "complete",
        "cleanup_retryable",
        "lease_expired_retryable",
        "finalization_retryable",
        "artifact_incomplete",
    }
)


def _build_diagnostic_business_outcome(
    status: str,
    *,
    recommendation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in _DIAGNOSTIC_BUSINESS_OUTCOME_STATUSES:
        raise ValueError(f"unsupported diagnostic business outcome: {status}")
    if status == "recommendation_valid":
        if not isinstance(recommendation, Mapping) or recommendation.get("recommendation") not in {
            "accept",
            "revise",
        }:
            raise ValueError("recommendation_valid requires a validated accept or revise recommendation")
    elif recommendation is not None:
        raise ValueError(f"{status} cannot carry a business recommendation")
    return {
        "status": status,
        "recommendation": dict(recommendation) if recommendation is not None else None,
    }


def _build_diagnostic_operational_outcome(
    status: str,
    *,
    failure_kind: str = "",
    retryable: bool = False,
    error: str = "",
    cleanup_status: str = "pending",
    artifact_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in _DIAGNOSTIC_OPERATIONAL_OUTCOME_STATUSES:
        raise ValueError(f"unsupported diagnostic operational outcome: {status}")
    return {
        "status": status,
        "failure_kind": failure_kind,
        "retryable": bool(retryable),
        "error": error,
        "cleanup_status": cleanup_status,
        "artifact_status": dict(artifact_status or {}),
    }


def build_diagnostic_business_outcome(
    status: str,
    *,
    recommendation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable business half of a diagnostic terminal record."""
    return _build_diagnostic_business_outcome(status, recommendation=recommendation)


def build_diagnostic_operational_outcome(
    status: str,
    *,
    failure_kind: str = "",
    retryable: bool = False,
    error: str = "",
    cleanup_status: str = "pending",
    artifact_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable operational half of a diagnostic terminal record."""
    return _build_diagnostic_operational_outcome(
        status,
        failure_kind=failure_kind,
        retryable=retryable,
        error=error,
        cleanup_status=cleanup_status,
        artifact_status=artifact_status,
    )


def _project_diagnostic_terminal_status(
    business_outcome: Mapping[str, Any],
    operational_outcome: Mapping[str, Any],
) -> str:
    business_status = business_outcome.get("status")
    if business_status == "business_halt":
        return "halted"
    if business_status == "recommendation_valid":
        return "success"
    if operational_outcome.get("retryable") is True:
        return "retryable"
    return "running"


def materialize_diagnostic_terminal_artifacts(
    run_root: str | Path,
    agent_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Idempotently rebuild derived diagnostic media and reports from durable state."""
    state = _require_state(run_root)
    diagnostics = _diagnostics(state)
    terminal = diagnostics.get("terminal")
    if not isinstance(terminal, dict):
        raise GenesisVlmSchemaError("diagnostic artifact materialization requires terminal state")
    business = terminal.get("business_outcome")
    if not isinstance(business, Mapping) or business.get("status") != "recommendation_valid":
        raise GenesisVlmSchemaError("diagnostic artifact materialization requires a valid business recommendation")
    recommendation = terminal.get("recommendation")
    if not isinstance(recommendation, Mapping):
        raise GenesisVlmSchemaError("diagnostic artifact materialization requires recommendation state")

    statuses: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if recommendation.get("recommendation") == "accept":
        try:
            _generate_diagnostic_episode_videos(run_root, state)
            # The media generator updates the in-memory episode records while
            # refreshing the derived workspace files.  Persist that state
            # before reloading it so the subsequent coverage projection and
            # report materializer see the same triptych lineage as index.json.
            save_state(state, run_root)
            state = _require_state(run_root)
            diagnostics = _diagnostics(state)
            terminal = diagnostics["terminal"]
            terminal["coverage_summary"] = build_diagnostic_coverage_summary(state)
            statuses["media"] = {"status": "complete", "retryable": False}
        except Exception as exc:
            message = str(exc)
            errors.append(f"media: {message}")
            statuses["media"] = {"status": "incomplete", "retryable": True, "error": message}
    else:
        statuses["media"] = {"status": "not_required", "retryable": False}

    try:
        artifact_paths = _write_diagnostic_artifacts(state, agent_record or {})
        diagnostics["artifact_paths"] = artifact_paths
        summary_event_exists = any(
            isinstance(event, Mapping)
            and event.get("provenance") == "summary_report"
            and event.get("event") == "diagnostic_summary_materialized"
            for event in diagnostics.get("runtime_owned_events", [])
        )
        refresh_runtime_owned_diagnostic_files(
            run_root,
            state,
            provenance="summary_report",
            event=None if summary_event_exists else "diagnostic_summary_materialized",
            note="materialized diagnostic summary, markdown, and cue artifacts",
            detail={"artifact_paths": artifact_paths},
        )
        statuses["reports"] = {
            "status": "complete",
            "retryable": False,
            "paths": artifact_paths,
        }
    except Exception as exc:
        message = str(exc)
        errors.append(f"reports: {message}")
        artifact_paths = diagnostics.get("artifact_paths", {})
        statuses["reports"] = {"status": "incomplete", "retryable": True, "error": message}

    operational_status = "artifact_incomplete" if errors else "complete"
    prior_operational = terminal.get("operational_outcome", {})
    if not isinstance(prior_operational, Mapping):
        prior_operational = {}
    prior_status = str(prior_operational.get("status", ""))
    prior_cleanup_status = str(prior_operational.get("cleanup_status", "complete"))
    prior_failure_kind = str(prior_operational.get("failure_kind", ""))
    prior_error = str(prior_operational.get("error", ""))
    prior_retryable = bool(prior_operational.get("retryable", False))
    # Artifact materialization is a derived-output retry.  It must never
    # overwrite a still-unresolved cleanup or lease outcome recorded by the
    # MCP owner state machine.
    if prior_status in {"cleanup_retryable", "lease_expired_retryable"}:
        operational_status = prior_status
    else:
        operational_status = "artifact_incomplete" if errors else "complete"
    failure_kinds = [item for item in (prior_failure_kind, "artifact_materialization_failed" if errors else "") if item]
    error_messages = [item for item in (prior_error, *errors) if item]
    terminal["operational_outcome"] = _build_diagnostic_operational_outcome(
        operational_status,
        failure_kind="; ".join(dict.fromkeys(failure_kinds)),
        retryable=prior_retryable or bool(errors) or operational_status in {"cleanup_retryable", "lease_expired_retryable"},
        error="; ".join(dict.fromkeys(error_messages)),
        cleanup_status=prior_cleanup_status,
        artifact_status=statuses,
    )
    terminal["status"] = _project_diagnostic_terminal_status(
        terminal["business_outcome"], terminal["operational_outcome"]
    )
    terminal["validated"] = True
    terminal["error"] = ""
    save_state(state, run_root)
    return {
        "status": operational_status,
        "artifact_status": statuses,
        "artifact_paths": dict(artifact_paths) if isinstance(artifact_paths, Mapping) else {},
        "missing_artifacts": [name for name, item in statuses.items() if item["status"] == "incomplete"],
    }


def finalize_diagnostic_terminal(
    run_root: str | Path,
    *,
    status: str,
    agent_record: dict[str, Any],
    error: str = "",
) -> dict[str, Any] | None:
    state = _require_state(run_root)
    persisted_diagnostics = state.get("diagnostics")
    if not isinstance(persisted_diagnostics, dict) or not isinstance(
        persisted_diagnostics.get("terminal"), dict
    ):
        require_v2_session_state(state)
    diagnostics = _diagnostics(state)
    settled_episode_ids = {item.get("episode_id") for item in diagnostics.get("region_settlements", []) if isinstance(item, Mapping) and item.get("episode_id")}
    for episode in diagnostics.get("episodes", []):
        if not isinstance(episode, dict) or episode.get("episode_id") not in settled_episode_ids:
            continue
        if status == "success" and episode.get("status") not in {"superseded"}:
            episode["status"] = "completed"
        elif status in {"rejected", "failed", "halted"}:
            episode["status"] = status
        _write_episode_observations(episode, diagnostics)
    terminal = diagnostics.get("terminal")
    if not isinstance(terminal, dict):
        refresh_runtime_owned_diagnostic_files(
            run_root,
            state,
            provenance="runtime_event",
            event="diagnostic_terminal_missing",
            note="diagnostic terminal state was missing",
            detail={"status": status, "error": error},
        )
        save_state(state, run_root)
        return
    recommendation = terminal.get("recommendation")
    is_accept = isinstance(recommendation, Mapping) and recommendation.get("recommendation") == "accept"
    if is_accept:
        require_v2_session_state(state)
    if status == "success":
        if not isinstance(recommendation, Mapping):
            raise GenesisVlmSchemaError("successful diagnostic terminal requires recommendation")
        terminal["business_outcome"] = _build_diagnostic_business_outcome(
            "recommendation_valid", recommendation=recommendation
        )
        terminal["operational_outcome"] = _build_diagnostic_operational_outcome("pending")
        terminal["status"] = "success"
        terminal["validated"] = True
        terminal["error"] = ""
        route_state = state.setdefault("route_state", {})
        route_state["diagnostic_recommendation"] = dict(recommendation)
        route_state["diagnostic_route"] = {
            **_diagnostic_route_payload(
                state,
                str(terminal["route"]),
                bool(terminal["ready"]),
                evidence_mode="probe_window" if is_accept else None,
            ),
            "status": "success",
        }
    elif status == "halted":
        terminal["business_outcome"] = _build_diagnostic_business_outcome("business_halt")
        terminal["operational_outcome"] = _build_diagnostic_operational_outcome("complete")
        terminal["status"] = "halted"
        terminal["validated"] = False
        terminal["error"] = error
    else:
        terminal["business_outcome"] = _build_diagnostic_business_outcome("not_authored")
        terminal["operational_outcome"] = _build_diagnostic_operational_outcome(
            "finalization_retryable",
            failure_kind="terminal_finalization_failed",
            retryable=True,
            error=error,
        )
        terminal["status"] = status
        terminal["validated"] = False
        terminal["error"] = error
    refresh_runtime_owned_diagnostic_files(
        run_root,
        state,
        provenance="terminal_recommendation",
        event="diagnostic_terminal_status_finalized",
        note=error or f"diagnostic terminal status finalized: {status}",
        detail=terminal,
    )
    save_state(state, run_root)
    if terminal.get("tool_name") == "submit_diagnostic_recommendation" and status == "success":
        note = terminal["recommendation"].get("reason") or "; ".join(
            terminal["recommendation"].get("diagnostic_cues", [])
        )
        stage_outputs = {
            "route": terminal["route"],
            "ready": bool(terminal["ready"]),
        }
        if is_accept:
            stage_outputs["evidence_mode"] = "probe_window"
        record_stage(
            run_root,
            DIAGNOSTIC_STAGE_NAME,
            True,
            outputs=stage_outputs,
            artifacts={},
            allow_new=DIAGNOSTIC_STAGE_NAME not in state.get("stages", {}),
        )
        append_history(
            run_root,
            "submit_diagnostic_recommendation",
            note,
            event="diagnostic_recommendation_validated",
            detail=terminal,
        )
        return None
    if status == "rejected":
        state.setdefault("route_state", {})["diagnostic_route"] = {
            "route": terminal.get("route"),
            "original_route": terminal.get("route"),
            "effective_route": terminal.get("route"),
            "ready": False,
            "status": "rejected",
            "error": error,
            "recommended_stage_skill_paths": [],
            "force_rerun_stage_skill_paths": [],
            "reentry_stage_skill_paths": [],
        }
        refresh_runtime_owned_diagnostic_files(
            run_root,
            state,
            provenance="terminal_recommendation",
            event="diagnostic_recommendation_rejected",
            note=error or "diagnostic recommendation rejected",
            detail=terminal,
        )
        save_state(state, run_root)
        append_history(
            run_root,
            "submit_diagnostic_recommendation",
            error or "diagnostic recommendation rejected",
            event="diagnostic_recommendation_rejected",
            detail=terminal,
        )
DIAGNOSTIC_TOOLS = (
    record_diagnostic_session_plan,
    compute_geometry_context,
    compute_part_grounding_context,
    submit_diagnostic_anchor_target_intent,
    compile_diagnostic_anchor_target,
    preview_diagnostic_anchor_target,
    revise_diagnostic_anchor_target,
    submit_diagnostic_probe_target_intent,
    compile_diagnostic_probe_target,
    preview_diagnostic_probe_target,
    record_diagnostic_probe_target_reflection,
    record_diagnostic_episode_concern_disposition,
    revise_diagnostic_probe_target,
    preview_diagnostic_episode_setup,
    record_diagnostic_setup_reflection,
    define_episode,
    inspect_genesis_runtime_logs,
    simulation_reset,
    simulate,
    query_live_geometry_context,
    record_diagnostic_evidence,
    audit_source_semantic_material_invariants,
    submit_diagnostic_recommendation,
    halt_diagnostics,
)
__all__ = [
    "DIAGNOSTIC_STAGE_NAME",
    "DIAGNOSTIC_TERMINAL_TOOL_NAMES",
    "DIAGNOSTIC_TOOLS",
    "LIVE_TOOL_HANDLERS",
    "LiveToolHandler",
    "active_diagnostic_run_root",
    "active_live_tool_handlers",
    "audit_source_semantic_material_invariants",
    "archive_and_reset_diagnostic_execution_attempt",
    "bind_active_diagnostic_run",
    "bind_live_tool_handlers",
    "build_part_grounding_context_from_state",
    "build_diagnostic_business_outcome",
    "build_diagnostic_operational_outcome",
    "compile_diagnostic_anchor_target",
    "compile_diagnostic_probe_target",
    "compute_geometry_context",
    "compute_part_grounding_context",
    "create_diagnostic_episode_from_anchor",
    "create_diagnostic_episode_suite",
    "build_diagnostic_coverage_summary",
    "define_episode",
    "finalize_diagnostic_terminal",
    "materialize_diagnostic_terminal_artifacts",
    "is_fresh_unplanned_diagnostics_state",
    "get_contact_states",
    "get_deformation_states",
    "halt_diagnostics",
    "inspect_genesis_runtime_logs",
    "initialize_source_semantic_material_artifact_registration",
    "pause_and_observe",
    "pause_simulation",
    "preview_diagnostic_anchor_target",
    "preview_diagnostic_episode_setup",
    "preview_diagnostic_probe_target",
    "query_live_geometry_context",
    "record_diagnostic_evidence",
    "record_diagnostic_pre_episode_plan",
    "record_diagnostic_setup_reflection",
    "record_diagnostic_probe_target_reflection",
    "record_diagnostic_episode_concern_disposition",
    "record_diagnostic_session_plan",
    "refresh_runtime_owned_diagnostic_files",
    "render_part_grounding_markdown_table",
    "revise_diagnostic_anchor_target",
    "revise_diagnostic_probe_target",
    "resume_simulation",
    "set_material_params",
    "simulate",
    "simulation_reset",
    "submit_diagnostic_anchor_target_intent",
    "submit_diagnostic_probe_target_intent",
    "submit_diagnostic_recommendation",
    "settle_region_on_live_close",
    "require_v2_session_state",
    "verify_v2_anchor_probe_separation_records",
]
