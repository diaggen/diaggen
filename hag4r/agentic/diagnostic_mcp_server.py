from __future__ import annotations

import argparse
import base64
import fcntl
import inspect
import json
import os
import secrets
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from PIL import Image

from hag4r.agentic.diagnostic_session_registry import (
    DiagnosticSessionEvent,
    DiagnosticSessionRegistry,
    DiagnosticSessionState,
    NORMAL_CLOSE_STATES,
)
from hag4r.agentic.runtime_state import (
    load_state,
    record_stage,
    save_state,
    state_path,
    validate_single_cuda_visible_devices,
)
from hag4r.agentic.diagnostic_force_limited_controller import DIAGNOSTIC_FORCE_LIMITED_CAPABILITY
from hag4r.agentic.genesis_vlm_schemas import validate_vlm_final_recommendation
from hag4r.agentic.runtime_tools.diagnostics import (
    DIAGNOSTIC_STAGE_NAME,
    DIAGNOSTIC_TERMINAL_TOOL_NAMES,
    DIAGNOSTIC_TOOLS,
    bind_active_diagnostic_run,
    bind_live_tool_handlers,
    archive_and_reset_diagnostic_execution_attempt,
    build_diagnostic_business_outcome,
    build_diagnostic_operational_outcome,
    diagnostic_triple_view_cameras_from_state,
    finalize_diagnostic_terminal,
    materialize_diagnostic_terminal_artifacts,
    require_v2_session_state,
    settle_region_on_live_close,
)
from hag4r.tools.genesis.live_client import (
    COMMON_GENESIS_REQUIRED_CAPABILITIES,
    GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS,
    GENESIS_RUNTIME_LOG_SCHEMA_VERSION,
    GenesisLiveApiSession,
)
from hag4r.tools.genesis.live_protocol import DEFAULT_READY_TIMEOUT_S, GenesisLiveProtocolError, PROTOCOL_NAME

OWNER_KIND = "codex_diagnostic_subagent"
OWNER_LEASE_GRACE_S = 300
DEFAULT_OWNER_LEASE_TIMEOUT_S = 3900
DEFAULT_EPISODE_TIMEOUT_S = 3600
DIAGNOSTIC_FORBIDDEN_LIVE_CAPABILITIES = frozenset(
    {
        "deformable_textured_visual_overlay",
        "visual_overlay_depth_normal_triptych_telemetry",
        "visual_overlay_vertex_trace",
    }
)
LIFECYCLE_MCP_TOOL_NAMES = frozenset(
    {
        "attach_diagnostic_run",
        "renew_diagnostic_owner_lease",
        "create_genesis_live_session",
        "bind_genesis_live_handlers",
        "get_genesis_live_session_status",
        "close_genesis_live_session",
    }
)
DETERMINISTIC_MCP_TOOL_NAMES = frozenset(tool.__name__ for tool in DIAGNOSTIC_TOOLS)
LIVE_SESSION_ARGUMENT_TOOL_NAMES = frozenset(
    {"inspect_genesis_runtime_logs", "simulation_reset", "simulate", "query_live_geometry_context"}
)


def _validate_diagnostic_live_capability_evidence(
    payload: dict[str, Any],
    *,
    evidence_name: str,
) -> None:
    """Reject a diagnostic server that advertises a generic textured overlay."""
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise GenesisLiveProtocolError(
            f"diagnostic {evidence_name} did not report a capability list",
            code="genesis_live_missing_capabilities",
        )
    forbidden = sorted(DIAGNOSTIC_FORBIDDEN_LIVE_CAPABILITIES.intersection(capabilities))
    if forbidden:
        raise GenesisLiveProtocolError(
            f"diagnostic {evidence_name} advertised forbidden textured-overlay capabilities: "
            + ", ".join(forbidden),
            code="genesis_live_diagnostic_overlay_capability_violation",
            details={"forbidden_capabilities": forbidden, "reported_capabilities": capabilities},
        )
POST_RUNTIME_FAILURE_DENIED_TOOL_NAMES = frozenset(
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
VISUAL_RESULT_TOOL_NAMES = frozenset(
    {
        "preview_diagnostic_anchor_target",
        "preview_diagnostic_episode_setup",
        "preview_diagnostic_probe_target",
        "simulation_reset",
        "simulate",
    }
)
MAX_MODEL_VISIBLE_SIMULATE_IMAGES = 3
MAX_MODEL_VISIBLE_PREVIEW_IMAGES = 1
MAX_MODEL_VISIBLE_IMAGE_BYTES = 8 * 1024 * 1024

_ATTACHMENT_TERMINAL_STATUSES = frozenset(
    {"completed", "halted", "expired", "failed", "superseded"}
)
_SESSION_TERMINAL_STATES = frozenset({"closed", "closed_failed"})
_WATCHDOG_POLL_INTERVAL_S = 0.25
_OWNERSHIP_SCHEMA_VERSION = "hag4r-diagnostic-runtime-ownership-v1"
_OWNERSHIP_LOCK_FILENAME = ".diagnostic_mcp_ownership.lock"


class _RecoveryIdentityMismatch(RuntimeError):
    """Raised when an orphan endpoint cannot be proven to belong to its ledger."""


def _canonical_run_root(run_root: str | Path) -> Path:
    if isinstance(run_root, str) and not run_root.strip():
        raise ValueError("run_root must be a non-empty path")
    return Path(run_root).expanduser().resolve()


def _require_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _format_error(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


@dataclass
class DiagnosticAttachmentContext:
    run_root: Path
    run_lock: threading.RLock = field(repr=False)
    attachment_id: str
    agent_invocation_id: str
    owner_lease_token: str = field(repr=False)
    lease_id: str
    attached_at: str
    last_renewed_at: str
    expires_at: str
    monotonic_deadline: float
    owner_lease_timeout_s: int
    cuda_visible_devices: str
    status: str = "attached"
    in_flight_count: int = 0
    session_handles: list[str] = field(default_factory=list)
    watchdog_registered: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class DiagnosticMcpRuntime:
    def __init__(
        self,
        *,
        owner_lease_timeout_s: int = DEFAULT_OWNER_LEASE_TIMEOUT_S,
        session_registry: DiagnosticSessionRegistry | None = None,
        session_factory: Callable[..., GenesisLiveApiSession] = GenesisLiveApiSession,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        process_environ: Mapping[str, str] | None = None,
    ) -> None:
        if isinstance(owner_lease_timeout_s, bool) or not isinstance(owner_lease_timeout_s, int):
            raise TypeError("owner_lease_timeout_s must be an integer")
        if owner_lease_timeout_s <= 0:
            raise ValueError("owner_lease_timeout_s must be positive")
        self.owner_lease_timeout_s = owner_lease_timeout_s
        self.session_registry = session_registry or DiagnosticSessionRegistry()
        self.session_factory = session_factory
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self.monotonic_clock = monotonic_clock
        self.process_environ = dict(os.environ if process_environ is None else process_environ)
        self._attachment_map_lock = threading.RLock()
        self._current_attachment_by_run: dict[Path, DiagnosticAttachmentContext] = {}
        self._attachments_by_id: dict[str, DiagnosticAttachmentContext] = {}
        self._run_locks: dict[Path, threading.RLock] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._watchdog_failure_lock = threading.Lock()
        self._watchdog_failure_events: list[dict[str, str]] = []
        self._pending_stage_failures: dict[str, str] = {}

    @staticmethod
    def _transport_loss_failure_kind(error: BaseException) -> str | None:
        """Classify only typed Genesis transport loss, never arbitrary errors."""
        if isinstance(error, EOFError):
            return "genesis_live_socket_eof"
        if isinstance(error, (ConnectionError, BrokenPipeError)):
            return "genesis_live_connection_lost"
        if isinstance(error, GenesisLiveProtocolError):
            code = error.code
            if code in {"genesis_live_socket_eof", "genesis_live_connection_lost"}:
                return code
        return None

    @staticmethod
    def _terminal_transport_recovery_metadata(
        state: dict[str, Any],
        attachment_id: str,
    ) -> dict[str, str] | None:
        """Return an explicit, persisted successor cause for one attachment."""
        attachments = state.get("diagnostic_runtime_attachments", [])
        if not isinstance(attachments, list):
            raise ValueError("state diagnostic_runtime_attachments must be a list")
        attachment = next(
            (
                entry for entry in attachments
                if isinstance(entry, dict) and entry.get("attachment_id") == attachment_id
            ),
            None,
        )
        if not isinstance(attachment, dict) or attachment.get("status") not in _ATTACHMENT_TERMINAL_STATUSES:
            return None
        terminal = state.get("diagnostics", {}).get("terminal")
        if not isinstance(terminal, Mapping):
            return None
        business = terminal.get("business_outcome")
        operational = terminal.get("operational_outcome")
        if (
            business != {"status": "not_authored", "recommendation": None}
            or not isinstance(operational, Mapping)
            or operational.get("status") not in {"lease_expired_retryable", "cleanup_retryable"}
            or operational.get("retryable") is not True
            or operational.get("failure_kind")
            not in {"mcp_server_shutdown", "genesis_live_socket_eof", "genesis_live_connection_lost"}
        ):
            return None
        history = attachment.get("history", [])
        if not isinstance(history, list):
            raise ValueError("diagnostic attachment history must be a list")
        recovery_event = next(
            (
                event for event in reversed(history)
                if isinstance(event, Mapping)
                and event.get("event") == "diagnostic_transport_recovery_ready"
                and event.get("failure_kind") == operational.get("failure_kind")
            ),
            None,
        )
        if not isinstance(recovery_event, Mapping):
            return None
        event_id = recovery_event.get("recovery_event_id")
        if not isinstance(event_id, str) or not event_id:
            return None
        invocation = attachment.get("agent_invocation_id")
        if not isinstance(invocation, str) or not invocation:
            return None
        return {
            "cause": str(operational["failure_kind"]),
            "event_id": event_id,
            "source_attachment_id": attachment_id,
            "source_agent_invocation_id": invocation,
        }

    def _validated_cuda_visible_devices(self, state: dict[str, Any]) -> str:
        mcp_value = self.process_environ.get("CUDA_VISIBLE_DEVICES")
        if mcp_value is None:
            raise ValueError(
                "MCP process CUDA_VISIBLE_DEVICES is unset; configure one explicit token in "
                "the repository-local diagnostics MCP env and reload only that MCP process"
            )
        mcp_token = validate_single_cuda_visible_devices(
            mcp_value,
            source="MCP process CUDA_VISIBLE_DEVICES",
        )
        worker_isolation = state.get("worker_isolation")
        if not isinstance(worker_isolation, dict):
            raise ValueError("runtime state is missing worker_isolation")
        gpu_binding = worker_isolation.get("gpu_binding")
        if not isinstance(gpu_binding, dict):
            raise ValueError("worker_isolation.gpu_binding must be an object")
        if gpu_binding.get("source") != "env":
            raise ValueError("worker_isolation.gpu_binding.source must be 'env' for diagnostics")
        if gpu_binding.get("conflict_checked") is not True:
            raise ValueError("worker_isolation.gpu_binding.conflict_checked must be true")
        binding_value = gpu_binding.get("value")
        if binding_value is None:
            raise ValueError("worker_isolation.gpu_binding.value is required")
        binding_token = validate_single_cuda_visible_devices(
            binding_value,
            source="worker_isolation.gpu_binding.value",
        )
        recorded_value = gpu_binding.get("effective_cuda_visible_devices")
        if recorded_value is None:
            raise ValueError(
                "worker_isolation.gpu_binding.effective_cuda_visible_devices is required"
            )
        recorded_token = validate_single_cuda_visible_devices(
            recorded_value,
            source="worker_isolation.gpu_binding.effective_cuda_visible_devices",
        )
        if binding_token != recorded_token:
            raise ValueError(
                "worker_isolation.gpu_binding.value must match "
                "effective_cuda_visible_devices"
            )
        if recorded_token != mcp_token:
            raise ValueError(
                "worker GPU binding does not match the MCP process CUDA_VISIBLE_DEVICES: "
                f"recorded={recorded_token!r}, mcp={mcp_token!r}; reinitialize the affected "
                "run with the repository-configured MCP token"
            )
        return mcp_token

    def _now(self) -> datetime:
        now = self.wall_clock()
        if now.tzinfo is None:
            raise ValueError("wall_clock must return a timezone-aware datetime")
        return now

    def _lease_times(self) -> tuple[str, str, float]:
        now = self._now()
        return (
            now.isoformat(),
            (now + timedelta(seconds=self.owner_lease_timeout_s)).isoformat(),
            self.monotonic_clock() + self.owner_lease_timeout_s,
        )

    def _run_lock(self, run_root: Path) -> threading.RLock:
        with self._attachment_map_lock:
            return self._run_locks.setdefault(run_root, threading.RLock())

    @staticmethod
    @contextmanager
    def _interprocess_run_lock(run_root: Path) -> Iterator[None]:
        lock_path = run_root / _OWNERSHIP_LOCK_FILENAME
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _assert_lock_owned(lock: threading.RLock, *, label: str) -> None:
        is_owned = getattr(lock, "_is_owned", None)
        if is_owned is None or not is_owned():
            raise RuntimeError(f"{label} lock must be held")

    @staticmethod
    def _assert_attachment_lock_owned(attachment: DiagnosticAttachmentContext) -> None:
        is_owned = getattr(attachment.lock, "_is_owned", None)
        if is_owned is None or not is_owned():
            raise RuntimeError("attachment lock must be held for ownership mutation")

    @staticmethod
    def _attachment_entry(state: dict[str, Any], attachment_id: str) -> dict[str, Any]:
        entries = state.get("diagnostic_runtime_attachments")
        if not isinstance(entries, list):
            raise ValueError("state diagnostic_runtime_attachments must be a list")
        for entry in entries:
            if isinstance(entry, dict) and entry.get("attachment_id") == attachment_id:
                return entry
        raise ValueError(f"diagnostic attachment ledger entry is missing: {attachment_id}")

    @staticmethod
    def _session_entry(state: dict[str, Any], live_session_handle: str) -> dict[str, Any]:
        entries = state.get("diagnostic_runtime_sessions")
        if not isinstance(entries, list):
            raise ValueError("state diagnostic_runtime_sessions must be a list")
        for entry in entries:
            if isinstance(entry, dict) and entry.get("live_session_handle") == live_session_handle:
                return entry
        raise ValueError(f"diagnostic session ledger entry is missing: {live_session_handle}")

    @staticmethod
    def _ownership_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        attachments = state.get("diagnostic_runtime_attachments", [])
        sessions = state.get("diagnostic_runtime_sessions", [])
        if not isinstance(attachments, list) or not isinstance(sessions, list):
            raise ValueError("diagnostic runtime ownership ledgers must be lists")
        return {
            "schema_version": _OWNERSHIP_SCHEMA_VERSION,
            "diagnostic_runtime_attachments": attachments,
            "diagnostic_runtime_sessions": sessions,
        }

    @classmethod
    def _attachment_events(
        cls,
        state: dict[str, Any],
        attachment_id: str,
    ) -> list[dict[str, Any]]:
        attachment = cls._attachment_entry(state, attachment_id)
        events = [
            event
            for key in ("pre_session_tool_events", "terminal_tool_events")
            for event in attachment.get(key, [])
            if isinstance(event, dict)
        ]
        sessions = state.get("diagnostic_runtime_sessions", [])
        if not isinstance(sessions, list):
            raise ValueError("state diagnostic_runtime_sessions must be a list")
        events.extend(
            event
            for session in sessions
            if isinstance(session, dict) and session.get("attachment_id") == attachment_id
            for event in session.get("tool_events", [])
            if isinstance(event, dict)
        )
        return events

    @classmethod
    def _next_event_sequence_index(cls, state: dict[str, Any], attachment_id: str) -> int:
        indices = [int(event["sequence_index"]) for event in cls._attachment_events(state, attachment_id)]
        if len(indices) != len(set(indices)):
            raise RuntimeError("diagnostic ownership event sequence contains duplicate indices")
        return max(indices, default=-1) + 1

    @classmethod
    def _attachment_runtime_failure(
        cls, state: dict[str, Any], attachment_id: str
    ) -> dict[str, Any] | None:
        attachment = cls._attachment_entry(state, attachment_id)
        marker = attachment.get("runtime_failure")
        session_markers = [
            session.get("runtime_failure")
            for session in state.get("diagnostic_runtime_sessions", [])
            if isinstance(session, dict)
            and session.get("attachment_id") == attachment_id
            and isinstance(session.get("runtime_failure"), dict)
        ]
        if isinstance(marker, dict):
            if session_markers and marker != session_markers[-1]:
                raise RuntimeError("attachment and episode runtime_failure markers disagree")
            return marker
        if session_markers:
            raise RuntimeError("attachment runtime_failure marker is missing")
        return None

    @classmethod
    def _new_tool_event(
        cls,
        state: dict[str, Any],
        attachment_id: str,
        *,
        tool: str,
        timestamp: str,
        status: str,
        observation_kinds: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        episode_id: str | None = None,
        include_attachment_id: bool = False,
        live_session_handle: str | None = None,
        tool_result_index: int | None = None,
        reflection_index: int | None = None,
    ) -> dict[str, Any]:
        event = {
            "sequence_index": cls._next_event_sequence_index(state, attachment_id),
            "tool": tool,
            "timestamp": timestamp,
            "status": status,
            "observation_kinds": list(observation_kinds or []),
            "artifact_refs": list(artifact_refs or []),
        }
        if episode_id:
            event["episode_id"] = episode_id
        if include_attachment_id:
            event["attachment_id"] = attachment_id
        if live_session_handle is not None:
            event["live_session_handle"] = live_session_handle
        if tool_result_index is not None:
            event["tool_result_index"] = tool_result_index
        if reflection_index is not None:
            event["reflection_index"] = reflection_index
        return event

    @staticmethod
    def _validate_durable_tool_result_correlation(
        state: dict[str, Any],
        *,
        tool: str,
        tool_result_index: int,
        episode_id: str,
    ) -> None:
        diagnostics = state.get("diagnostics")
        tool_results = diagnostics.get("tool_results") if isinstance(diagnostics, dict) else None
        if not isinstance(tool_results, list):
            raise ValueError("state diagnostics.tool_results must be a list")
        if (
            isinstance(tool_result_index, bool)
            or not isinstance(tool_result_index, int)
            or tool_result_index < 0
            or tool_result_index >= len(tool_results)
        ):
            raise ValueError("diagnostic tool-result correlation index is missing or invalid")
        record = tool_results[tool_result_index]
        if not isinstance(record, dict):
            raise ValueError("correlated diagnostic tool result must be an object")
        if record.get("tool_result_index") != tool_result_index:
            raise ValueError("correlated diagnostic tool result stores a mismatched index")
        if record.get("tool") != tool:
            raise ValueError("correlated diagnostic tool result stores a mismatched tool")
        if record.get("episode_id") != episode_id:
            raise ValueError("correlated diagnostic tool result stores a mismatched episode")

    def _record_ownership_tool_event(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool: str,
        status: str,
        session_handle: str | None = None,
        terminal_phase: bool = False,
        episode_id: str | None = None,
        observation_kinds: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        tool_result_index: int | None = None,
        reflection_index: int | None = None,
    ) -> dict[str, Any]:
        timestamp = self._now().isoformat()
        recorded: dict[str, Any] = {}

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            correlate_runtime_log = tool == "inspect_genesis_runtime_logs"
            correlate_tool_result = tool_result_index is not None
            if correlate_runtime_log:
                if session_handle is None or episode_id is None or tool_result_index is None:
                    raise ValueError("Genesis runtime log event requires session, episode, and tool-result correlation")
                self._validate_durable_tool_result_correlation(
                    state,
                    tool=tool,
                    tool_result_index=tool_result_index,
                    episode_id=episode_id,
                )
            event = self._new_tool_event(
                state,
                attachment.attachment_id,
                tool=tool,
                timestamp=timestamp,
                status=status,
                observation_kinds=observation_kinds,
                artifact_refs=artifact_refs,
                episode_id=episode_id,
                include_attachment_id=correlate_tool_result or reflection_index is not None,
                live_session_handle=(
                    session_handle
                    if correlate_tool_result or reflection_index is not None
                    else None
                ),
                tool_result_index=tool_result_index,
                reflection_index=reflection_index,
            )
            if session_handle is not None:
                owner = self._session_entry(state, session_handle)
                owner["tool_sequence"].append(tool)
                owner["tool_events"].append(event)
            elif terminal_phase:
                entry["terminal_tool_events"].append(event)
            else:
                entry["pre_session_tool_sequence"].append(tool)
                entry["pre_session_tool_events"].append(event)
            recorded.update(event)

        self._mutate_ownership_ledgers(attachment, mutation)
        return recorded

    @staticmethod
    def _runtime_log_result_nonempty(record: dict[str, Any]) -> bool:
        wrapper = record.get("result")
        if not isinstance(wrapper, dict) or wrapper.get("status") != "ok":
            return False
        envelope = wrapper.get("result") if isinstance(wrapper, dict) else None
        if not isinstance(envelope, dict) or envelope.get("schema_version") != GENESIS_RUNTIME_LOG_SCHEMA_VERSION:
            return False
        streams = envelope.get("streams")
        return isinstance(streams, dict) and any(
            isinstance(stream, dict) and bool(stream.get("lines"))
            for stream in streams.values()
        )

    def _validate_runtime_failure_reflection(
        self,
        attachment: DiagnosticAttachmentContext,
        session_context: Any,
        evidence: dict[str, Any],
        state: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        if evidence.get("episode_id") != session_context.episode_id:
            raise PermissionError("runtime_failure evidence belongs to another episode")
        refs = evidence.get("artifact_refs")
        if not isinstance(refs, list) or len(refs) != 1:
            raise ValueError("runtime_failure requires exactly one raw log tool-result ref")
        ref = refs[0].get("ref") if isinstance(refs[0], dict) else None
        if not isinstance(ref, str) or not ref.startswith("tool_result:"):
            raise ValueError("runtime_failure requires a tool_result:<1-based-index> ref")
        tool_result_index = int(ref.removeprefix("tool_result:")) - 1
        self._validate_durable_tool_result_correlation(
            state,
            tool="inspect_genesis_runtime_logs",
            tool_result_index=tool_result_index,
            episode_id=session_context.episode_id,
        )
        record = state["diagnostics"]["tool_results"][tool_result_index]
        if not self._runtime_log_result_nonempty(record):
            raise ValueError("runtime_failure must cite a successful nonempty raw runtime-log result")
        matching = [
            event
            for event in self._attachment_events(state, attachment.attachment_id)
            if event.get("tool") == "inspect_genesis_runtime_logs"
            and event.get("status") == "ok"
            and event.get("tool_result_index") == tool_result_index
        ]
        if not matching:
            raise ValueError("runtime_failure raw log result lacks owned inspection-event correlation")
        event = matching[-1]
        if (
            event.get("attachment_id") != attachment.attachment_id
            or event.get("episode_id") != session_context.episode_id
            or event.get("live_session_handle") != session_context.handle
        ):
            raise PermissionError("runtime_failure raw log evidence belongs to another attachment, episode, or handle")
        return tool_result_index, event

    def _enrich_persisted_reflection(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        reflection_index: int,
        episode_id: str | None,
        session_handle: str | None,
        event: dict[str, Any],
        runtime_failure: dict[str, Any] | None = None,
    ) -> None:
        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            reflections = state.get("diagnostics", {}).get("model_reflections", [])
            if reflection_index < 0 or reflection_index >= len(reflections):
                raise ValueError("persisted diagnostic reflection index is missing")
            reflection = reflections[reflection_index]
            if not isinstance(reflection, dict) or reflection.get("reflection_index") != reflection_index:
                raise ValueError("persisted diagnostic reflection index is mismatched")
            reflection.update(
                {
                    "attachment_id": attachment.attachment_id,
                    "agent_invocation_id": attachment.agent_invocation_id,
                    "episode_id": episode_id or reflection.get("episode_id"),
                    "ownership_event_sequence_index": event["sequence_index"],
                    "active_revision_id": str(state.get("active_revision") or ""),
                }
            )
            if session_handle is not None:
                reflection["live_session_handle"] = session_handle
            if runtime_failure is not None:
                session = self._session_entry(state, session_handle or "")
                session["runtime_failure"] = runtime_failure
                entry["runtime_failure"] = dict(runtime_failure)

        self._mutate_ownership_ledgers(attachment, mutation)

    @staticmethod
    def _write_ownership_artifact(state: dict[str, Any], snapshot: dict[str, Any]) -> Path:
        paths = state.get("paths")
        if not isinstance(paths, dict) or not paths.get("diagnostic_workspace_dir"):
            raise ValueError("state.paths.diagnostic_workspace_dir is required")
        path = Path(str(paths["diagnostic_workspace_dir"])).expanduser().resolve() / "runtime_ownership.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".json.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    @staticmethod
    def _propagate_ownership_export(state: dict[str, Any], source: Path) -> None:
        paths = state.get("paths", {})
        destination_root = paths.get("final_export_sim_diagnostics_dir")
        if not destination_root:
            return
        destination = Path(str(destination_root)).expanduser().resolve() / "runtime_ownership.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            f".json.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
        manifest_value = paths.get("final_export_manifest_path")
        if manifest_value:
            manifest_path = Path(str(manifest_value)).expanduser().resolve()
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.setdefault("exported_artifacts", {})[
                    "diagnostic_runtime_ownership"
                ] = str(destination)
                manifest.setdefault("source_artifacts", {})[
                    "diagnostic_runtime_ownership"
                ] = str(source)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    def _mutate_ownership_ledgers(
        self,
        attachment: DiagnosticAttachmentContext,
        mutation: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> dict[str, Any]:
        self._assert_attachment_lock_owned(attachment)
        self._assert_lock_owned(attachment.run_lock, label="run ownership")
        state = load_state(attachment.run_root)
        entry = self._attachment_entry(state, attachment.attachment_id)
        mutation(entry, state)
        state["updated_at"] = self._now().isoformat()
        save_state(state, attachment.run_root)
        snapshot = self._ownership_snapshot(state)
        source = self._write_ownership_artifact(state, snapshot)
        self._propagate_ownership_export(state, source)
        return snapshot

    def _install_attachment_ledger(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        superseded_attachment_id: str | None,
        state: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        self._assert_attachment_lock_owned(attachment)
        self._assert_lock_owned(attachment.run_lock, label="run ownership")
        if state is None:
            state = load_state(attachment.run_root)
        attachments = state.setdefault("diagnostic_runtime_attachments", [])
        sessions = state.setdefault("diagnostic_runtime_sessions", [])
        if not isinstance(attachments, list) or not isinstance(sessions, list):
            raise ValueError("diagnostic runtime ownership ledgers must be lists")
        if superseded_attachment_id is not None:
            superseded = self._attachment_entry(state, superseded_attachment_id)
            if superseded.get("status") not in _ATTACHMENT_TERMINAL_STATUSES:
                raise RuntimeError("durable diagnostic attachment is not terminal")
            superseded_at = self._now().isoformat()
            superseded["status"] = "superseded"
            superseded["completed_at"] = superseded.get("completed_at") or superseded_at
            superseded["token_invalidated_at"] = superseded_at
            superseded["watchdog_registered"] = False
            superseded["owner_lease"]["status"] = "superseded"
            superseded["history"].append(
                {"event": "diagnostic_attachment_superseded", "timestamp": superseded_at}
            )
        attachments.append(
            {
                "attachment_id": attachment.attachment_id,
                "run_root": str(attachment.run_root),
                "agent_invocation_id": attachment.agent_invocation_id,
                "owner_kind": OWNER_KIND,
                "attached_at": attachment.attached_at,
                "completed_at": None,
                "status": "attached",
                "owner_lease": {
                    "lease_id": attachment.lease_id,
                    "last_renewed_at": attachment.last_renewed_at,
                    "expires_at": attachment.expires_at,
                    "timeout_s": attachment.owner_lease_timeout_s,
                    "in_flight_count": 0,
                    "status": "active",
                },
                "pre_session_tool_sequence": ["attach_diagnostic_run"],
                "pre_session_tool_events": [
                    {
                        "sequence_index": 0,
                        "tool": "attach_diagnostic_run",
                        "timestamp": attachment.attached_at,
                        "status": "ok",
                        "observation_kinds": [],
                        "artifact_refs": [],
                    }
                ],
                "terminal_tool_events": [],
                "session_handles": [],
                "token_invalidated_at": None,
                "watchdog_registered": True,
                "runtime_failure": None,
                "history": [
                    {
                        "event": "diagnostic_run_attached",
                        "timestamp": attachment.attached_at,
                        "agent_invocation_id": attachment.agent_invocation_id,
                        "attachment_id": attachment.attachment_id,
                    }
                ],
            }
        )
        if persist:
            state["updated_at"] = self._now().isoformat()
            save_state(state, attachment.run_root)
            snapshot = self._ownership_snapshot(state)
            source = self._write_ownership_artifact(state, snapshot)
            self._propagate_ownership_export(state, source)
        return state

    def _terminalize_failed_attachment_install(
        self,
        attachment: DiagnosticAttachmentContext,
        install_error: BaseException,
    ) -> None:
        self._assert_attachment_lock_owned(attachment)
        self._assert_lock_owned(attachment.run_lock, label="run ownership")
        state = load_state(attachment.run_root)
        entries = state.get("diagnostic_runtime_attachments", [])
        if not isinstance(entries, list):
            raise ValueError("state diagnostic_runtime_attachments must be a list")
        entry = next(
            (
                candidate
                for candidate in entries
                if isinstance(candidate, dict)
                and candidate.get("attachment_id") == attachment.attachment_id
            ),
            None,
        )
        if entry is None:
            return
        failed_at = self._now().isoformat()
        entry["status"] = "failed"
        entry["completed_at"] = failed_at
        entry["token_invalidated_at"] = failed_at
        entry["watchdog_registered"] = False
        entry["owner_lease"]["status"] = "closed"
        entry["owner_lease"]["in_flight_count"] = 0
        entry["history"].append(
            {
                "event": "diagnostic_attachment_install_failed",
                "timestamp": failed_at,
                "error": _format_error(install_error),
            }
        )
        state["updated_at"] = failed_at
        save_state(state, attachment.run_root)
        snapshot = self._ownership_snapshot(state)
        source = self._write_ownership_artifact(state, snapshot)
        self._propagate_ownership_export(state, source)

    @staticmethod
    def _durable_open_sessions(state: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], ...]:
        entries = state.get("diagnostic_runtime_sessions", [])
        if not isinstance(entries, list):
            raise ValueError("state diagnostic_runtime_sessions must be a list")
        return tuple(
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("run_root") == str(run_root)
            and entry.get("lifecycle_state") not in _SESSION_TERMINAL_STATES
        )

    def _renew_attachment(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        event: str,
    ) -> None:
        renewed_at, expires_at, monotonic_deadline = self._lease_times()
        attachment.last_renewed_at = renewed_at
        attachment.expires_at = expires_at
        attachment.monotonic_deadline = monotonic_deadline

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del state
            lease = entry["owner_lease"]
            lease.update(
                {
                    "last_renewed_at": renewed_at,
                    "expires_at": expires_at,
                    "in_flight_count": attachment.in_flight_count,
                    "status": "active",
                }
            )
            entry["history"].append(
                {
                    "event": event,
                    "timestamp": renewed_at,
                    "in_flight_count": attachment.in_flight_count,
                }
            )

        self._mutate_ownership_ledgers(attachment, mutation)

    def _reconcile_lease_state(self, attachment: DiagnosticAttachmentContext) -> None:
        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del state
            entry["owner_lease"].update(
                {
                    "last_renewed_at": attachment.last_renewed_at,
                    "expires_at": attachment.expires_at,
                    "in_flight_count": attachment.in_flight_count,
                    "status": (
                        "active"
                        if attachment.status not in _ATTACHMENT_TERMINAL_STATUSES
                        else entry["owner_lease"]["status"]
                    ),
                }
            )

        self._mutate_ownership_ledgers(attachment, mutation)

    def _current_attachment(self, run_root: Path) -> DiagnosticAttachmentContext | None:
        with self._attachment_map_lock:
            return self._current_attachment_by_run.get(run_root)

    @staticmethod
    def _durable_nonterminal_attachment(
        state: dict[str, Any],
        run_root: Path,
    ) -> dict[str, Any] | None:
        entries = state.get("diagnostic_runtime_attachments", [])
        if not isinstance(entries, list):
            raise ValueError("state diagnostic_runtime_attachments must be a list")
        nonterminal = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("run_root") == str(run_root)
            and entry.get("status") not in _ATTACHMENT_TERMINAL_STATUSES
        ]
        if len(nonterminal) > 1:
            raise RuntimeError(
                "multiple non-terminal durable diagnostic attachments are ambiguous"
            )
        return nonterminal[0] if nonterminal else None

    def _durable_lease_expired(self, entry: dict[str, Any]) -> tuple[bool, datetime]:
        lease = entry.get("owner_lease")
        if not isinstance(lease, dict):
            raise ValueError("durable diagnostic attachment owner_lease must be an object")
        expires_at = lease.get("expires_at")
        if not isinstance(expires_at, str) or not expires_at.strip():
            raise ValueError("durable diagnostic attachment owner_lease.expires_at is required")
        try:
            expiry = datetime.fromisoformat(expires_at)
        except ValueError as error:
            raise ValueError(
                "durable diagnostic attachment owner_lease.expires_at is malformed"
            ) from error
        if expiry.tzinfo is None:
            raise ValueError(
                "durable diagnostic attachment owner_lease.expires_at must be timezone-aware"
            )
        return self._now() > expiry, expiry

    @staticmethod
    def _same_resolved_path(left: Any, right: Any) -> bool:
        if not isinstance(left, str) or not left.strip():
            return False
        if not isinstance(right, str) or not right.strip():
            return False
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()

    def _validate_orphan_launch_identity(
        self,
        state: dict[str, Any],
        attachment_entry: dict[str, Any],
        session_entry: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
        attachment_id = _require_non_empty_string(
            attachment_entry.get("attachment_id"), field_name="attachment_id"
        )
        session_handle = _require_non_empty_string(
            session_entry.get("live_session_handle"), field_name="live_session_handle"
        )
        for key in ("run_root", "agent_invocation_id", "owner_kind"):
            if session_entry.get(key) != attachment_entry.get(key):
                raise _RecoveryIdentityMismatch(
                    f"orphan session {key} does not match its attachment"
                )
        if attachment_entry.get("owner_kind") != OWNER_KIND:
            raise _RecoveryIdentityMismatch("orphan attachment owner_kind is not supported")
        if session_entry.get("attachment_id") != attachment_id:
            raise _RecoveryIdentityMismatch("orphan session attachment_id does not match")
        handles = attachment_entry.get("session_handles")
        if not isinstance(handles, list) or session_handle not in handles:
            raise _RecoveryIdentityMismatch(
                "orphan session handle is absent from its attachment ledger"
            )
        episode_id = _require_non_empty_string(
            session_entry.get("episode_id"), field_name="episode_id"
        )
        episode = self._active_episode(state, episode_id)
        evidence_value = session_entry.get("launch_evidence_path")
        if not isinstance(evidence_value, str) or not evidence_value.strip():
            raise _RecoveryIdentityMismatch("orphan session launch_evidence_path is missing")
        evidence_path = Path(evidence_value).expanduser().resolve()
        expected_evidence_path = (
            Path(str(episode["log_dir"])).expanduser().resolve()
            / "genesis_live_launch.json"
        )
        if evidence_path != expected_evidence_path:
            raise _RecoveryIdentityMismatch(
                "orphan session launch_evidence_path is not the active episode's canonical "
                "genesis_live_launch.json"
            )
        if not evidence_path.is_file():
            raise _RecoveryIdentityMismatch(
                f"orphan session launch evidence is missing: {evidence_path}"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise _RecoveryIdentityMismatch("orphan launch evidence must be a JSON object")
        if evidence.get("schema_version") != "hag4r-genesis-live-launch-v1":
            raise _RecoveryIdentityMismatch("orphan launch evidence schema is unsupported")
        expected_identity = {
            "live_session_handle": session_handle,
            "episode_id": episode_id,
            "agent_invocation_id": attachment_entry.get("agent_invocation_id"),
            "owner_kind": OWNER_KIND,
        }
        for key, expected in expected_identity.items():
            if evidence.get(key) != expected:
                raise _RecoveryIdentityMismatch(
                    f"orphan launch evidence {key} does not match the durable ledger"
                )
        for evidence_key, episode_key in (
            ("ready_file_path", "ready_file_path"),
            ("scene_config_path", "scene_config_path"),
            ("output_dir", "live_output_dir"),
            ("log_dir", "log_dir"),
        ):
            if not self._same_resolved_path(evidence.get(evidence_key), episode.get(episode_key)):
                raise _RecoveryIdentityMismatch(
                    f"orphan launch evidence {evidence_key} does not match the active episode"
                )
        return episode, evidence, evidence_path

    def _recover_orphan_session(
        self,
        attachment_entry: dict[str, Any],
        session_entry: dict[str, Any],
        *,
        evidence: dict[str, Any],
        evidence_path: Path,
    ) -> dict[str, Any]:
        recovered_at = self._now().isoformat()
        ready_file = Path(str(evidence["ready_file_path"])).expanduser().resolve()
        result = {
            "episode_id": session_entry["episode_id"],
            "live_session_handle": session_entry["live_session_handle"],
            "launch_evidence_path": str(evidence_path),
            "ready_file_path": str(ready_file),
            "cleanup_status": "failed",
        }
        if not ready_file.is_file():
            message = f"expired owner recovery ready file is missing: {ready_file}"
            session_entry.update(
                {
                    "lifecycle_state": "closed_failed",
                    "closed_at": recovered_at,
                    "failed_at": recovered_at,
                    "failure": {
                        "kind": "expired_owner_recovery_unreachable",
                        "message": message,
                    },
                }
            )
            result["error"] = message
            return result
        try:
            ready = json.loads(ready_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise _RecoveryIdentityMismatch(
                f"orphan Genesis ready file cannot prove endpoint identity: {error}"
            ) from error
        if not isinstance(ready, dict) or ready.get("protocol") != PROTOCOL_NAME:
            raise _RecoveryIdentityMismatch("orphan Genesis ready file protocol is unsupported")
        if not self._same_resolved_path(
            ready.get("scene_config_path"), evidence.get("scene_config_path")
        ):
            raise _RecoveryIdentityMismatch(
                "orphan Genesis ready file scene_config_path does not match launch evidence"
            )
        session_token = ready.get("session_token")
        if not isinstance(session_token, str) or not session_token.strip():
            raise _RecoveryIdentityMismatch("orphan Genesis ready file lacks session_token")
        host = ready.get("host")
        port = ready.get("port")
        if not isinstance(host, str) or not host.strip() or isinstance(port, bool) or not isinstance(port, int):
            raise _RecoveryIdentityMismatch("orphan Genesis ready endpoint is malformed")
        recovery_session = self.session_factory(
            scene_config_path=None,
            genesis_root=None,
            genesis_env_path=None,
            host=host,
            port=port,
            ready_timeout_s=3.0,
        )
        try:
            recovery_session.connect()
            handshake = recovery_session.handshake_identity()
        except (ConnectionError, OSError, TimeoutError) as error:
            message = f"expired owner recovery endpoint is unreachable: {_format_error(error)}"
            session_entry.update(
                {
                    "lifecycle_state": "closed_failed",
                    "closed_at": recovered_at,
                    "failed_at": recovered_at,
                    "failure": {
                        "kind": "expired_owner_recovery_unreachable",
                        "message": message,
                    },
                }
            )
            result["error"] = message
            return result
        except GenesisLiveProtocolError as error:
            raise _RecoveryIdentityMismatch(
                f"orphan Genesis endpoint protocol identity failed: {error}"
            ) from error
        if not isinstance(handshake, dict) or handshake.get("session_id") != session_token:
            mismatch = _RecoveryIdentityMismatch(
                "orphan Genesis ready token does not match the live session identity"
            )
            try:
                recovery_session.detach_existing_endpoint()
            except Exception as detach_error:
                mismatch.add_note(
                    f"failed to detach identity-mismatched recovery socket: {detach_error}"
                )
            raise mismatch
        try:
            recovery_session.close(timeout_ms=3_000)
        except Exception as error:
            message = f"verified orphan Genesis endpoint close failed: {_format_error(error)}"
            session_entry.update(
                {
                    "lifecycle_state": "closed_failed",
                    "closed_at": recovered_at,
                    "failed_at": recovered_at,
                    "failure": {
                        "kind": "expired_owner_recovery_close_failed",
                        "message": message,
                    },
                }
            )
            result["error"] = message
            return result
        session_entry.update(
            {
                "lifecycle_state": "closed",
                "closed_at": recovered_at,
                "failed_at": None,
                "failure": None,
            }
        )
        result["cleanup_status"] = "verified_closed"
        return result

    def _persist_recovered_ownership(self, run_root: Path, state: dict[str, Any]) -> None:
        state["updated_at"] = self._now().isoformat()
        save_state(state, run_root)
        snapshot = self._ownership_snapshot(state)
        source = self._write_ownership_artifact(state, snapshot)
        self._propagate_ownership_export(state, source)

    def _recover_expired_attachment(
        self,
        run_root: Path,
        state: dict[str, Any],
        attachment_entry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._assert_lock_owned(self._run_lock(run_root), label="run ownership")
        attachment_id = str(attachment_entry["attachment_id"])
        sessions = state.get("diagnostic_runtime_sessions", [])
        if not isinstance(sessions, list):
            raise ValueError("state diagnostic_runtime_sessions must be a list")
        orphan_sessions = [
            entry
            for entry in sessions
            if isinstance(entry, dict)
            and entry.get("attachment_id") == attachment_id
            and entry.get("lifecycle_state") not in _SESSION_TERMINAL_STATES
        ]
        validated = [
            (
                session_entry,
                *self._validate_orphan_launch_identity(
                    state, attachment_entry, session_entry
                )[1:],
            )
            for session_entry in orphan_sessions
        ]
        cleanup_results = [
            self._recover_orphan_session(
                attachment_entry,
                session_entry,
                evidence=evidence,
                evidence_path=evidence_path,
            )
            for session_entry, evidence, evidence_path in validated
        ]
        if any(
            entry.get("lifecycle_state") not in _SESSION_TERMINAL_STATES
            for entry in orphan_sessions
        ):
            raise RuntimeError("expired owner recovery left a session non-terminal")
        recovered_at = self._now().isoformat()
        attachment_entry.update(
            {
                "status": "expired",
                "completed_at": recovered_at,
                "token_invalidated_at": recovered_at,
                "watchdog_registered": False,
            }
        )
        lease = attachment_entry.get("owner_lease")
        if not isinstance(lease, dict):
            raise ValueError("durable diagnostic attachment owner_lease must be an object")
        lease.update({"status": "expired", "in_flight_count": 0})
        history = attachment_entry.setdefault("history", [])
        if not isinstance(history, list):
            raise ValueError("durable diagnostic attachment history must be a list")
        history.append(
            {
                "event": "expired_owner_recovered",
                "timestamp": recovered_at,
                "attachment_id": attachment_id,
                "agent_invocation_id": attachment_entry.get("agent_invocation_id"),
                "cleanup_results": cleanup_results,
            }
        )
        self._persist_recovered_ownership(run_root, state)
        failed_results = [
            result for result in cleanup_results if result["cleanup_status"] == "failed"
        ]
        if failed_results:
            self._record_operational_cleanup_failure(
                run_root,
                (
                    "expired diagnostic owner recovery could not cleanly close every "
                    "Genesis session"
                ),
                failure_kind="owner_lease_expired_cleanup_failed",
                operational_status="cleanup_retryable",
                cleanup_status="retryable",
            )
        return cleanup_results

    @contextmanager
    def _authenticated_request(
        self,
        *,
        run_root: str | Path,
        agent_invocation_id: str,
        owner_lease_token: str,
    ) -> Iterator[DiagnosticAttachmentContext]:
        canonical = _canonical_run_root(run_root)
        invocation = _require_non_empty_string(
            agent_invocation_id,
            field_name="agent_invocation_id",
        )
        token = _require_non_empty_string(owner_lease_token, field_name="owner_lease_token")
        attachment = self._current_attachment(canonical)
        if attachment is None:
            raise PermissionError("no diagnostic attachment is current for this run")
        with attachment.run_lock:
            with attachment.lock:
                if attachment.agent_invocation_id != invocation:
                    raise PermissionError("diagnostic attachment invocation does not match")
                if not secrets.compare_digest(attachment.owner_lease_token, token):
                    raise PermissionError("diagnostic owner lease token does not match")
                if attachment.status in _ATTACHMENT_TERMINAL_STATUSES:
                    raise PermissionError(f"diagnostic attachment is stale: {attachment.status}")
                if self.monotonic_clock() >= attachment.monotonic_deadline:
                    raise PermissionError("diagnostic owner lease has expired")
                previous_lease = (
                    attachment.last_renewed_at,
                    attachment.expires_at,
                    attachment.monotonic_deadline,
                )
                attachment.in_flight_count += 1
                try:
                    self._renew_attachment(attachment, event="owner_lease_request_entered")
                except BaseException as entry_write_error:
                    attachment.in_flight_count -= 1
                    (
                        attachment.last_renewed_at,
                        attachment.expires_at,
                        attachment.monotonic_deadline,
                    ) = previous_lease
                    try:
                        self._reconcile_lease_state(attachment)
                    except BaseException as reconciliation_error:
                        entry_write_error.add_note(
                            f"failed to reconcile lease after request-entry write failure: {reconciliation_error}"
                        )
                    raise
                body_error: BaseException | None = None
                try:
                    yield attachment
                except BaseException as error:
                    body_error = error
                    raise
                finally:
                    attachment.in_flight_count -= 1
                    if attachment.status not in _ATTACHMENT_TERMINAL_STATUSES:
                        try:
                            self._renew_attachment(attachment, event="owner_lease_request_exited")
                        except BaseException as exit_write_error:
                            try:
                                self._reconcile_lease_state(attachment)
                            except BaseException as reconciliation_error:
                                exit_write_error.add_note(
                                    "failed to reconcile lease after request-exit write failure: "
                                    f"{reconciliation_error}"
                                )
                            if body_error is not None:
                                body_error.add_note(
                                    f"owner lease exit write failed: {exit_write_error}"
                                )
                            else:
                                raise

    def attach_diagnostic_run(self, run_root: str, agent_invocation_id: str) -> dict[str, Any]:
        canonical = _canonical_run_root(run_root)
        invocation = _require_non_empty_string(
            agent_invocation_id,
            field_name="agent_invocation_id",
        )
        run_lock = self._run_lock(canonical)
        if not state_path(canonical).is_file():
            raise FileNotFoundError(f"diagnostic runtime state is missing: {state_path(canonical)}")
        with run_lock, self._interprocess_run_lock(canonical):
            if not state_path(canonical).is_file():
                raise FileNotFoundError(
                    f"diagnostic runtime state disappeared while acquiring ownership: "
                    f"{state_path(canonical)}"
                )
            state = load_state(canonical)
            recorded_run_root = state.get("run_root")
            if not recorded_run_root or state_path(str(recorded_run_root)).parent.resolve() != canonical:
                raise ValueError("state.run_root does not identify the requested canonical run root")
            diagnostics = state.get("diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError("state.diagnostics must be an object")
            episode_timeout_s = int(diagnostics.get("episode_timeout_s", DEFAULT_EPISODE_TIMEOUT_S))
            minimum_timeout_s = episode_timeout_s + OWNER_LEASE_GRACE_S
            if self.owner_lease_timeout_s < minimum_timeout_s:
                raise ValueError(
                    "owner lease timeout is shorter than the diagnostic episode timeout plus grace: "
                    f"{self.owner_lease_timeout_s} < {episode_timeout_s} + {OWNER_LEASE_GRACE_S}"
                )
            cuda_visible_devices = self._validated_cuda_visible_devices(state)

            previous = self._current_attachment(canonical)
            if self.session_registry.has_open_session(run_root=canonical):
                raise RuntimeError("cannot replace a diagnostic owner while a live session is open")
            if previous is not None and previous.status not in _ATTACHMENT_TERMINAL_STATUSES:
                raise RuntimeError("cannot replace a non-terminal diagnostic attachment")
            durable_attachments = state.get("diagnostic_runtime_attachments", [])
            if not isinstance(durable_attachments, list):
                raise ValueError("state diagnostic_runtime_attachments must be a list")
            durable_for_run = [
                entry
                for entry in durable_attachments
                if isinstance(entry, dict) and entry.get("run_root") == str(canonical)
            ]
            durable_nonterminal = self._durable_nonterminal_attachment(state, canonical)
            recovered_attachment_id: str | None = None
            if durable_nonterminal is not None:
                expired, expiry = self._durable_lease_expired(durable_nonterminal)
                if not expired:
                    raise RuntimeError(
                        "cannot replace non-terminal durable diagnostic attachment before "
                        f"its owner lease expires at {expiry.isoformat()}"
                    )
                self._recover_expired_attachment(
                    canonical,
                    state,
                    durable_nonterminal,
                )
                recovered_attachment_id = str(durable_nonterminal["attachment_id"])
                state = load_state(canonical)
                if self._durable_nonterminal_attachment(state, canonical) is not None:
                    raise RuntimeError("expired owner recovery did not terminalize the attachment")
                if any(
                    entry.get("attachment_id") == recovered_attachment_id
                    for entry in self._durable_open_sessions(state, canonical)
                ):
                    raise RuntimeError("expired owner recovery did not terminalize every session")
                durable_attachments = state.get("diagnostic_runtime_attachments", [])
                if not isinstance(durable_attachments, list):
                    raise ValueError("state diagnostic_runtime_attachments must be a list")
                durable_for_run = [
                    entry
                    for entry in durable_attachments
                    if isinstance(entry, dict) and entry.get("run_root") == str(canonical)
                ]
            durable_current = durable_for_run[-1] if durable_for_run else None
            successor_recovery = None
            if durable_current is not None:
                successor_recovery = self._terminal_transport_recovery_metadata(
                    state,
                    str(durable_current.get("attachment_id", "")),
                )
            if successor_recovery is not None:
                if self._durable_open_sessions(state, canonical):
                    raise RuntimeError(
                        "cannot install diagnostic successor while a durable live session remains open"
                    )
                if self.session_registry.has_open_session(run_root=canonical):
                    raise RuntimeError(
                        "cannot install diagnostic successor while a registry live session remains open"
                    )
                archive_and_reset_diagnostic_execution_attempt(
                    state,
                    recovery=successor_recovery,
                )
                # The successor attachment/stage mutation below persists this
                # in-memory reset in the same durable write.
            # Fresh attachment is the only no-plan exception.  A typed
            # transport loss reaches that state only through the archive/reset
            # transition above; ordinary evidence-bearing states still fail.
            require_v2_session_state(state, allow_fresh_unplanned=True)
            superseded_attachment_id = (
                str(durable_current["attachment_id"])
                if durable_current is not None
                and str(durable_current["attachment_id"]) != recovered_attachment_id
                else None
            )

            attached_at, expires_at, monotonic_deadline = self._lease_times()
            attachment = DiagnosticAttachmentContext(
                run_root=canonical,
                run_lock=run_lock,
                attachment_id=secrets.token_urlsafe(32),
                agent_invocation_id=invocation,
                owner_lease_token=secrets.token_urlsafe(32),
                lease_id=secrets.token_urlsafe(32),
                attached_at=attached_at,
                last_renewed_at=attached_at,
                expires_at=expires_at,
                monotonic_deadline=monotonic_deadline,
                owner_lease_timeout_s=self.owner_lease_timeout_s,
                cuda_visible_devices=cuda_visible_devices,
            )
            with attachment.lock:
                try:
                    attached_state = self._install_attachment_ledger(
                        attachment,
                        superseded_attachment_id=superseded_attachment_id,
                        state=state,
                        persist=False,
                    )
                    stages = attached_state.setdefault("stages", {})
                    diagnostic_stage = stages.setdefault(
                        DIAGNOSTIC_STAGE_NAME,
                        {"ok": False, "status": "pending", "error": ""},
                    )
                    if not isinstance(diagnostic_stage, dict):
                        raise ValueError("diagnostic stage ledger must be an object at MCP attachment")
                    if diagnostic_stage.get("status") not in {
                        "pending",
                        "pending_retry",
                        "retryable",
                        "running",
                        "failed",
                    }:
                        raise RuntimeError(
                            "diagnostic MCP attachment cannot activate stage with status="
                            f"{diagnostic_stage.get('status')}"
                        )
                    diagnostic_stage.update(
                        {
                            "ok": False,
                            "status": "running",
                            "started_at": diagnostic_stage.get("started_at") or attached_at,
                            "finished_at": None,
                            "error": "",
                        }
                    )
                    attached_state.setdefault("runtime_events", []).append(
                        {
                            "timestamp": attached_at,
                            "event": "diagnostic_stage_attached",
                            "stage_name": DIAGNOSTIC_STAGE_NAME,
                            "status": "running",
                            "attachment_id": attachment.attachment_id,
                        }
                    )
                    attached_state["updated_at"] = attached_at
                    save_state(attached_state, canonical)
                    snapshot = self._ownership_snapshot(attached_state)
                    source = self._write_ownership_artifact(attached_state, snapshot)
                    self._propagate_ownership_export(attached_state, source)
                except BaseException as install_error:
                    try:
                        self._terminalize_failed_attachment_install(
                            attachment,
                            install_error,
                        )
                    except BaseException as terminalize_error:
                        install_error.add_note(
                            "failed to terminalize partially installed diagnostic attachment: "
                            f"{terminalize_error}"
                        )
                    raise
            if previous is not None:
                with previous.lock:
                    previous.status = "superseded"
                    previous.watchdog_registered = False
                    previous.owner_lease_token = secrets.token_urlsafe(32)
            with self._attachment_map_lock:
                current = self._current_attachment_by_run.get(canonical)
                if current is not previous:
                    raise RuntimeError("diagnostic attachment changed concurrently")
                self._current_attachment_by_run[canonical] = attachment
                self._attachments_by_id[attachment.attachment_id] = attachment
        return {
            "attachment_id": attachment.attachment_id,
            "run_root": str(canonical),
            "agent_invocation_id": invocation,
            "owner_kind": OWNER_KIND,
            "owner_lease_timeout_s": self.owner_lease_timeout_s,
            "owner_lease_expires_at": expires_at,
            "owner_lease_token": attachment.owner_lease_token,
        }

    def renew_diagnostic_owner_lease(
        self,
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
    ) -> dict[str, Any]:
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            all_open_contexts = self.session_registry.open_contexts(run_root=attachment.run_root)
            if any(
                context.agent_invocation_id != attachment.agent_invocation_id
                for context in all_open_contexts
            ):
                raise PermissionError("open diagnostic session belongs to another invocation")
            open_contexts = all_open_contexts
            session_handle = open_contexts[0].handle if len(open_contexts) == 1 else None
            self._record_ownership_tool_event(
                attachment,
                tool="renew_diagnostic_owner_lease",
                status="ok",
                session_handle=session_handle,
                episode_id=open_contexts[0].episode_id if len(open_contexts) == 1 else None,
            )
            return {
                "attachment_id": attachment.attachment_id,
                "status": attachment.status,
                "owner_lease_expires_at": attachment.expires_at,
                "owner_lease_timeout_s": attachment.owner_lease_timeout_s,
            }

    def require_no_open_session(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool_name: str,
    ) -> None:
        self._assert_attachment_lock_owned(attachment)
        state = load_state(attachment.run_root)
        if self.session_registry.has_open_session(run_root=attachment.run_root) or self._durable_open_sessions(
            state, attachment.run_root
        ):
            raise RuntimeError(f"{tool_name} requires the previous Genesis live session to be closed")

    @staticmethod
    def _active_episode(state: dict[str, Any], episode_id: str) -> dict[str, Any]:
        diagnostics = state["diagnostics"]
        active_index = diagnostics.get("active_episode_index")
        if active_index is None:
            raise ValueError("a diagnostic episode must be active before creating a live session")
        active = next(
            (
                episode
                for episode in diagnostics.get("episodes", [])
                if isinstance(episode, dict)
                and int(episode.get("episode_index", 0)) == int(active_index)
            ),
            None,
        )
        if active is None:
            raise ValueError(f"active diagnostic episode is not recorded: {active_index}")
        if str(active.get("episode_id", "")) != episode_id:
            raise ValueError(
                f"requested episode is not active: requested={episode_id}, active={active.get('episode_id')}"
            )
        return active

    @staticmethod
    def _append_session_event(
        state: dict[str, Any],
        attachment_id: str,
        entry: dict[str, Any],
        *,
        tool: str,
        timestamp: str,
        status: str,
    ) -> None:
        entry["tool_sequence"].append(tool)
        entry["tool_events"].append(
            DiagnosticMcpRuntime._new_tool_event(
                state,
                attachment_id,
                tool=tool,
                timestamp=timestamp,
                status=status,
                episode_id=str(entry.get("episode_id", "")) or None,
            )
        )

    @staticmethod
    def _ensure_session_event(
        state: dict[str, Any],
        attachment_id: str,
        entry: dict[str, Any],
        *,
        tool: str,
        timestamp: str,
        status: str,
    ) -> None:
        if not any(
            isinstance(event, dict)
            and event.get("tool") == tool
            and event.get("status") == status
            for event in entry["tool_events"]
        ):
            DiagnosticMcpRuntime._append_session_event(
                state,
                attachment_id,
                entry,
                tool=tool,
                timestamp=timestamp,
                status=status,
            )

    def _reconcile_session_ledger(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        context: Any,
        tool: str,
        status: str,
    ) -> None:
        timestamp = (
            context.bound_at
            if context.state is DiagnosticSessionState.BOUND
            else context.closed_at
        ) or self._now().isoformat()

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del entry
            session_entry = self._session_entry(state, context.handle)
            session_entry["lifecycle_state"] = context.state.value
            if context.bound_at is not None:
                session_entry["bound_at"] = context.bound_at
            if context.closed_at is not None:
                session_entry["closed_at"] = context.closed_at
            if context.state is DiagnosticSessionState.CLOSED_FAILED:
                session_entry["failed_at"] = context.close_started_at or timestamp
                failure_message = context.causal_error or "Genesis live session close failed"
                if context.cleanup_error:
                    failure_message += f"; cleanup error: {context.cleanup_error}"
                session_entry["failure"] = {
                    "kind": "session_close_failed",
                    "message": failure_message,
                }
                if isinstance(session_entry.get("runtime_failure"), dict):
                    require_v2_session_state(state)
                    settle_region_on_live_close(
                        state,
                        attachment_id=attachment.attachment_id,
                        live_session_handle=context.handle,
                        episode_id=context.episode_id,
                        close_outcome="runtime_failed",
                    )
            elif context.state is DiagnosticSessionState.CLOSED and status == "ok":
                require_v2_session_state(state)
                settle_region_on_live_close(
                    state,
                    attachment_id=attachment.attachment_id,
                    live_session_handle=context.handle,
                    episode_id=context.episode_id,
                    close_outcome=(
                        "runtime_failed"
                        if isinstance(session_entry.get("runtime_failure"), dict)
                        else "clean_closed"
                    ),
                )
            self._ensure_session_event(
                state,
                attachment.attachment_id,
                session_entry,
                tool=tool,
                timestamp=timestamp,
                status=status,
            )

        self._mutate_ownership_ledgers(attachment, mutation)

    def _reconcile_failed_session_create(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        context: Any,
        ledger_created_at: str,
        launch_evidence_path: Path,
        error: BaseException,
    ) -> None:
        failed_at = context.close_started_at or self._now().isoformat()
        closed_at = context.closed_at or failed_at
        failure_message = _format_error(error)
        if context.cleanup_error:
            failure_message += f"; cleanup error: {context.cleanup_error}"

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            sessions = state.setdefault("diagnostic_runtime_sessions", [])
            if not isinstance(sessions, list):
                raise ValueError("state diagnostic_runtime_sessions must be a list")
            same_episode_entries = [
                candidate
                for candidate in sessions
                if isinstance(candidate, dict)
                and candidate.get("attachment_id") == attachment.attachment_id
                and candidate.get("episode_id") == context.episode_id
            ]
            conflicting_entries = [
                candidate
                for candidate in same_episode_entries
                if candidate.get("live_session_handle") != context.handle
            ]
            if conflicting_entries:
                raise RuntimeError(
                    "cannot reconcile failed session create because this attachment/episode "
                    "already has a different durable session ledger"
                )
            session_entry = next(
                (
                    candidate
                    for candidate in same_episode_entries
                    if candidate.get("live_session_handle") == context.handle
                ),
                None,
            )
            if session_entry is None:
                session_entry = {
                    "attachment_id": attachment.attachment_id,
                    "live_session_handle": context.handle,
                    "run_root": str(attachment.run_root),
                        "episode_id": context.episode_id,
                        "agent_invocation_id": attachment.agent_invocation_id,
                        "active_revision_id": str(state.get("active_revision") or ""),
                    "owner_kind": OWNER_KIND,
                    "lifecycle_state": "closed_failed",
                    "ledger_created_at": ledger_created_at,
                    "connect_started_at": ledger_created_at,
                    "connected_at": None,
                    "connection_status": "failed",
                    "created_at": None,
                    "bound_at": None,
                    "closed_at": closed_at,
                    "failed_at": failed_at,
                    "failure": {"kind": "session_create_failed", "message": failure_message},
                    "launch_evidence_path": str(launch_evidence_path),
                    "tool_sequence": ["create_genesis_live_session"],
                    "tool_events": [],
                }
                sessions.append(session_entry)
            else:
                session_entry.update(
                    {
                        "lifecycle_state": "closed_failed",
                        "connection_status": "failed",
                        "created_at": None,
                        "closed_at": closed_at,
                        "failed_at": failed_at,
                        "failure": {
                            "kind": "session_create_failed",
                            "message": failure_message,
                        },
                    }
                )
            if context.handle not in entry["session_handles"]:
                entry["session_handles"].append(context.handle)
            if not any(
                isinstance(event, dict)
                and event.get("tool") == "create_genesis_live_session"
                and event.get("status") == "error"
                for event in session_entry["tool_events"]
            ):
                session_entry["tool_events"].append(
                    self._new_tool_event(
                        state,
                        attachment.attachment_id,
                        tool="create_genesis_live_session",
                        timestamp=closed_at,
                        status="error",
                        artifact_refs=[str(launch_evidence_path)],
                        episode_id=context.episode_id,
                    )
                )

        self._mutate_ownership_ledgers(attachment, mutation)

    def create_genesis_live_session(
        self,
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        episode_id: str,
    ) -> dict[str, Any]:
        episode_id = _require_non_empty_string(episode_id, field_name="episode_id")
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            state = load_state(attachment.run_root)
            diagnostics = state.get("diagnostics")
            if not isinstance(diagnostics, dict) or diagnostics.get("enabled") is not True:
                raise RuntimeError("Genesis diagnostics are disabled for this run")
            episode = self._active_episode(state, episode_id)
            self.require_no_open_session(attachment, tool_name="create_genesis_live_session")
            durable_sessions = state.get("diagnostic_runtime_sessions", [])
            if not isinstance(durable_sessions, list):
                raise ValueError("state diagnostic_runtime_sessions must be a list")
            if any(
                isinstance(candidate, dict)
                and candidate.get("attachment_id") == attachment.attachment_id
                and candidate.get("episode_id") == episode_id
                for candidate in durable_sessions
            ):
                raise RuntimeError(
                    "diagnostic attachment already has a durable session ledger for episode: "
                    f"{episode_id}"
                )

            required_capabilities = (*COMMON_GENESIS_REQUIRED_CAPABILITIES, DIAGNOSTIC_FORCE_LIMITED_CAPABILITY)
            log_dir = Path(str(episode["log_dir"])).expanduser().resolve()
            launch_evidence_path = log_dir / "genesis_live_launch.json"
            session = self.session_factory(
                scene_config_path=Path(str(episode["scene_config_path"])).expanduser().resolve(),
                genesis_root=(
                    Path(str(diagnostics["genesis_root"])).expanduser().resolve()
                    if diagnostics.get("genesis_root")
                    else None
                ),
                genesis_env_path=(
                    Path(str(diagnostics["genesis_env_path"])).expanduser().resolve()
                    if diagnostics.get("genesis_env_path")
                    else None
                ),
                genesis_live_command=str(diagnostics.get("genesis_live_command", "python -m genesis.live.server")),
                host=str(diagnostics.get("live_host", "127.0.0.1")),
                port=int(diagnostics.get("live_port", 0)),
                ready_file_path=Path(str(episode["ready_file_path"])).expanduser().resolve(),
                output_dir=Path(str(episode["live_output_dir"])).expanduser().resolve(),
                log_dir=log_dir,
                ready_timeout_s=float(diagnostics.get("live_ready_timeout_s", DEFAULT_READY_TIMEOUT_S)),
                heartbeat_ms=int(diagnostics.get("live_heartbeat_ms", 1000)),
                client_lease_timeout_ms=int(diagnostics.get("live_client_lease_timeout_ms", 30000)),
                start_paused=True,
                required_capabilities=required_capabilities,
                asset_root=Path(str(state["repo_root"])).expanduser().resolve(),
                process_env_overrides={
                    "CUDA_VISIBLE_DEVICES": attachment.cuda_visible_devices,
                },
                launch_evidence_path=launch_evidence_path,
            )
            context = self.session_registry.create(
                run_root=attachment.run_root,
                episode_id=episode_id,
                agent_invocation_id=attachment.agent_invocation_id,
                live_session=session,
            )
            session.launch_identity = {
                "live_session_handle": context.handle,
                "episode_id": episode_id,
                "agent_invocation_id": attachment.agent_invocation_id,
                "owner_kind": OWNER_KIND,
            }
            attachment.session_handles.append(context.handle)
            ledger_created_at = self._now().isoformat()

            def provisional(entry: dict[str, Any], current_state: dict[str, Any]) -> None:
                entry["session_handles"].append(context.handle)
                entry["history"].append(
                    {
                        "event": "genesis_live_session_creating",
                        "timestamp": ledger_created_at,
                        "episode_id": episode_id,
                        "live_session_handle": context.handle,
                    }
                )
                sessions = current_state.setdefault("diagnostic_runtime_sessions", [])
                if any(
                    isinstance(item, dict)
                    and item.get("attachment_id") == attachment.attachment_id
                    and item.get("episode_id") == episode_id
                    for item in sessions
                ):
                    raise ValueError("attachment already has a session ledger for this episode")
                sessions.append(
                    {
                        "attachment_id": attachment.attachment_id,
                        "live_session_handle": context.handle,
                        "run_root": str(attachment.run_root),
                        "episode_id": episode_id,
                        "agent_invocation_id": attachment.agent_invocation_id,
                        "active_revision_id": str(current_state.get("active_revision") or ""),
                        "owner_kind": OWNER_KIND,
                        "lifecycle_state": "creating",
                        "ledger_created_at": ledger_created_at,
                        "connect_started_at": ledger_created_at,
                        "connected_at": None,
                        "connection_status": "pending",
                        "created_at": None,
                        "bound_at": None,
                        "closed_at": None,
                        "failed_at": None,
                        "failure": None,
                        "launch_evidence_path": str(launch_evidence_path),
                        "tool_sequence": ["create_genesis_live_session"],
                        "tool_events": [],
                    }
                )

            try:
                with self.session_registry.locked_context(
                    handle=context.handle,
                    run_root=attachment.run_root,
                    allowed_states={DiagnosticSessionState.CREATING},
                ):
                    self._mutate_ownership_ledgers(attachment, provisional)
                    session.connect()
                    _validate_diagnostic_live_capability_evidence(
                        session.ready_identity(),
                        evidence_name="live_ready",
                    )
                    _validate_diagnostic_live_capability_evidence(
                        session.handshake_identity(),
                        evidence_name="handshake",
                    )
                    self.session_registry.transition(
                        context=context,
                        event=DiagnosticSessionEvent.CONNECT_SUCCEEDED,
                    )
                    connected_at = context.created_at or self._now().isoformat()

                    def connected(entry: dict[str, Any], current_state: dict[str, Any]) -> None:
                        del entry
                        session_entry = self._session_entry(current_state, context.handle)
                        session_entry.update(
                            {
                                "lifecycle_state": "created",
                                "connected_at": connected_at,
                                "connection_status": "ok",
                                "created_at": connected_at,
                            }
                        )
                        session_entry["tool_events"].append(
                            self._new_tool_event(
                                current_state,
                                attachment.attachment_id,
                                tool="create_genesis_live_session",
                                timestamp=connected_at,
                                status="ok",
                                artifact_refs=[str(launch_evidence_path)],
                                episode_id=episode_id,
                            )
                        )

                    self._mutate_ownership_ledgers(attachment, connected)
            except BaseException as error:
                failed_at = self._now().isoformat()

                def failed_closing(entry: dict[str, Any], current_state: dict[str, Any]) -> None:
                    del entry
                    session_entry = self._session_entry(current_state, context.handle)
                    session_entry.update(
                        {
                            "lifecycle_state": "failed_closing",
                            "connection_status": "failed",
                            "failed_at": failed_at,
                            "failure": {"kind": "session_create_failed", "message": _format_error(error)},
                        }
                    )

                evidence_errors: list[BaseException] = []
                try:
                    self._mutate_ownership_ledgers(attachment, failed_closing)
                except BaseException as evidence_error:
                    evidence_errors.append(evidence_error)
                try:
                    self.session_registry.fail_and_close(
                        handle=context.handle,
                        run_root=attachment.run_root,
                        causal_error=error,
                    )
                except BaseException as cleanup_error:
                    evidence_errors.append(cleanup_error)
                try:
                    self._reconcile_failed_session_create(
                        attachment,
                        context=context,
                        ledger_created_at=ledger_created_at,
                        launch_evidence_path=launch_evidence_path,
                        error=error,
                    )
                except BaseException as evidence_error:
                    evidence_errors.append(evidence_error)
                for evidence_error in evidence_errors:
                    error.add_note(
                        f"failed to persist or clean up session-create failure: {evidence_error}"
                    )
                raise
            return {
                "attachment_id": attachment.attachment_id,
                "live_session_handle": context.handle,
                "episode_id": episode_id,
                "agent_invocation_id": attachment.agent_invocation_id,
                "owner_kind": OWNER_KIND,
                "lifecycle_state": context.state.value,
                "connection_status": "ok",
            }

    def _require_owned_context(
        self,
        attachment: DiagnosticAttachmentContext,
        live_session_handle: str,
    ) -> None:
        if live_session_handle not in attachment.session_handles:
            raise PermissionError("live session handle does not belong to the authenticated attachment")

    def _live_handler(
        self,
        *,
        context: Any,
        tool_name: str,
    ) -> Callable[..., dict[str, Any]]:
        method = getattr(context.live_session, tool_name)

        def call(
            *,
            run_root: str,
            arguments: dict[str, Any],
            decision: dict[str, Any],
        ) -> dict[str, Any]:
            canonical = _canonical_run_root(run_root)
            if canonical != context.run_root:
                raise PermissionError("live handler run root does not match its session")
            if not isinstance(arguments, dict):
                raise TypeError("live handler arguments must be an object")
            if not isinstance(decision, dict):
                raise TypeError("live handler decision must be an object")
            method_arguments = dict(arguments)
            if tool_name in {"simulation_reset", "simulate"}:
                state = load_state(canonical)
                episode_id = context.episode_id
                episode = next(
                    (
                        candidate
                        for candidate in state.get("diagnostics", {}).get("episodes", [])
                        if isinstance(candidate, dict)
                        and str(candidate.get("episode_id", "")) == episode_id
                    ),
                    None,
                )
                if episode is None:
                    raise ValueError(f"diagnostic episode is missing for live handler: {episode_id}")
                decision_id = int(decision.get("decision_id", 0) or 0)
                method_arguments.update(
                    {
                        "output_root": Path(str(episode["live_output_dir"])).expanduser().resolve()
                        / "fast_visual"
                        / f"decision_{decision_id:04d}_{tool_name}",
                        "triple_view_cameras": diagnostic_triple_view_cameras_from_state(state),
                    }
                )
            return method(**method_arguments)

        return call

    def bind_genesis_live_handlers(
        self,
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        live_session_handle: str,
    ) -> dict[str, Any]:
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            self._require_owned_context(attachment, live_session_handle)
            with self.session_registry.locked_context(
                handle=live_session_handle,
                run_root=attachment.run_root,
                allowed_states={DiagnosticSessionState.CREATED},
            ) as context:
                if context.agent_invocation_id != attachment.agent_invocation_id:
                    raise PermissionError("live session invocation does not match attachment")
                context.handlers = {
                    tool_name: self._live_handler(context=context, tool_name=tool_name)
                    for tool_name in (
                        "inspect_genesis_runtime_logs",
                        "simulation_reset",
                        "simulate",
                        "query_live_geometry_context",
                        "register_probe_action",
                    )
                }
                self.session_registry.transition(
                    context=context,
                    event=DiagnosticSessionEvent.BIND_SUCCEEDED,
                )
                bound_at = context.bound_at or self._now().isoformat()

                def bound(entry: dict[str, Any], state: dict[str, Any]) -> None:
                    del entry
                    session_entry = self._session_entry(state, live_session_handle)
                    session_entry["lifecycle_state"] = "bound"
                    session_entry["bound_at"] = bound_at
                    self._ensure_session_event(
                        state,
                        attachment.attachment_id,
                        session_entry,
                        tool="bind_genesis_live_handlers",
                        timestamp=bound_at,
                        status="ok",
                    )

                try:
                    self._mutate_ownership_ledgers(attachment, bound)
                except BaseException as writer_error:
                    try:
                        self._reconcile_session_ledger(
                            attachment,
                            context=context,
                            tool="bind_genesis_live_handlers",
                            status="ok",
                        )
                    except BaseException as reconciliation_error:
                        writer_error.add_note(
                            f"failed to reconcile bound session ledger: {reconciliation_error}"
                        )
                    raise
                return {
                    "attachment_id": attachment.attachment_id,
                    "live_session_handle": live_session_handle,
                    "episode_id": context.episode_id,
                    "lifecycle_state": context.state.value,
                    "handler_names": sorted(context.handlers),
                }

    def get_genesis_live_session_status(
        self,
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        live_session_handle: str,
    ) -> dict[str, Any]:
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            self._require_owned_context(attachment, live_session_handle)
            with self.session_registry.locked_context(
                handle=live_session_handle,
                run_root=attachment.run_root,
            ) as context:
                if context.agent_invocation_id != attachment.agent_invocation_id:
                    raise PermissionError("live session invocation does not match attachment")
                state = load_state(attachment.run_root)
                session_entry = self._session_entry(state, live_session_handle)
                result = {
                    "attachment_id": attachment.attachment_id,
                    "agent_invocation_id": attachment.agent_invocation_id,
                    "owner_kind": OWNER_KIND,
                    "live_session_handle": live_session_handle,
                    "episode_id": context.episode_id,
                    "lifecycle_state": context.state.value,
                    "connection_status": session_entry["connection_status"],
                    "handler_names": sorted(context.handlers),
                    "owner_lease_status": "active",
                    "owner_lease_expires_at": attachment.expires_at,
                }
                self._record_ownership_tool_event(
                    attachment,
                    tool="get_genesis_live_session_status",
                    status="ok",
                    session_handle=live_session_handle,
                    episode_id=context.episode_id,
                )
                return result

    def close_genesis_live_session(
        self,
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        live_session_handle: str,
    ) -> dict[str, Any]:
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            self._require_owned_context(attachment, live_session_handle)
            with self.session_registry.locked_context(
                handle=live_session_handle,
                run_root=attachment.run_root,
                allowed_states=NORMAL_CLOSE_STATES,
            ) as context:
                if context.agent_invocation_id != attachment.agent_invocation_id:
                    raise PermissionError("live session invocation does not match attachment")
                try:
                    self.session_registry.close(
                        handle=live_session_handle,
                        run_root=attachment.run_root,
                    )
                except BaseException as error:
                    closed_at = context.closed_at or self._now().isoformat()

                    def close_failed(entry: dict[str, Any], state: dict[str, Any]) -> None:
                        del entry
                        session_entry = self._session_entry(state, live_session_handle)
                        session_entry.update(
                            {
                                "lifecycle_state": "closed_failed",
                                "closed_at": closed_at,
                                "failed_at": context.close_started_at or closed_at,
                                "failure": {
                                    "kind": "session_close_failed",
                                    "message": context.causal_error or _format_error(error),
                                },
                            }
                        )
                        if context.cleanup_error:
                            session_entry["failure"]["message"] += (
                                f"; cleanup error: {context.cleanup_error}"
                            )
                        if isinstance(session_entry.get("runtime_failure"), dict):
                            require_v2_session_state(state)
                            settle_region_on_live_close(
                                state,
                                attachment_id=attachment.attachment_id,
                                live_session_handle=live_session_handle,
                                episode_id=context.episode_id,
                                close_outcome="runtime_failed",
                            )
                        self._ensure_session_event(
                            state,
                            attachment.attachment_id,
                            session_entry,
                            tool="close_genesis_live_session",
                            timestamp=closed_at,
                            status="error",
                        )

                    try:
                        self._mutate_ownership_ledgers(attachment, close_failed)
                    except BaseException as writer_error:
                        try:
                            self._reconcile_session_ledger(
                                attachment,
                                context=context,
                                tool="close_genesis_live_session",
                                status="error",
                            )
                        except BaseException as reconciliation_error:
                            writer_error.add_note(
                                f"failed to reconcile failed-close session ledger: {reconciliation_error}"
                            )
                        error.add_note(f"failed to persist failed close evidence: {writer_error}")
                    raise
                closed_at = context.closed_at or self._now().isoformat()

                def closed(entry: dict[str, Any], state: dict[str, Any]) -> None:
                    require_v2_session_state(state)
                    entry["history"].append(
                        {
                            "event": "genesis_live_session_closed",
                            "timestamp": closed_at,
                            "episode_id": context.episode_id,
                            "live_session_handle": live_session_handle,
                        }
                    )
                    session_entry = self._session_entry(state, live_session_handle)
                    session_entry["lifecycle_state"] = "closed"
                    session_entry["closed_at"] = closed_at
                    # This runs under the same ownership-ledger mutation as
                    # close itself; a returned successful close cannot expose
                    # an un-settled region.
                    settle_region_on_live_close(
                        state,
                        attachment_id=attachment.attachment_id,
                        live_session_handle=live_session_handle,
                        episode_id=context.episode_id,
                        close_outcome=(
                            "runtime_failed"
                            if isinstance(session_entry.get("runtime_failure"), dict)
                            else "clean_closed"
                        ),
                    )
                    self._ensure_session_event(
                        state,
                        attachment.attachment_id,
                        session_entry,
                        tool="close_genesis_live_session",
                        timestamp=closed_at,
                        status="ok",
                    )

                try:
                    self._mutate_ownership_ledgers(attachment, closed)
                except BaseException as writer_error:
                    try:
                        self._reconcile_session_ledger(
                            attachment,
                            context=context,
                            tool="close_genesis_live_session",
                            status="ok",
                        )
                    except BaseException as reconciliation_error:
                        writer_error.add_note(
                            f"failed to reconcile closed session ledger: {reconciliation_error}"
                        )
                    raise
                return {
                    "attachment_id": attachment.attachment_id,
                    "live_session_handle": live_session_handle,
                    "episode_id": context.episode_id,
                    "lifecycle_state": context.state.value,
                    "connection_status": "ok",
                }

    @staticmethod
    def _domain_result_status(result: dict[str, Any]) -> str:
        current: Any = result
        for _ in range(4):
            if not isinstance(current, dict):
                break
            status = str(current.get("status", "")).strip()
            if status:
                return status
            current = current.get("result")
        return ""

    @staticmethod
    def _domain_result_error(result: dict[str, Any]) -> Any:
        current: Any = result
        for _ in range(4):
            if not isinstance(current, dict):
                break
            if current.get("error"):
                return current["error"]
            current = current.get("result")
        return None

    @classmethod
    def _result_successful(cls, result: dict[str, Any]) -> bool:
        status = cls._domain_result_status(result)
        if not status:
            return cls._domain_result_error(result) is None
        return status in {"ok", "success", "applied", "written", "pending_validation"}

    @staticmethod
    def _validate_terminal_pre_finalization(
        tool_name: str,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        terminal = state.get("diagnostics", {}).get("terminal")
        if not isinstance(terminal, dict) or terminal.get("tool_name") != tool_name:
            raise RuntimeError(f"{tool_name} did not persist its matching terminal state")
        if tool_name == "submit_diagnostic_recommendation":
            if (
                result.get("status") != "pending_validation"
                or terminal.get("status") != "pending_validation"
                or terminal.get("validated") is not False
            ):
                raise RuntimeError(
                    "submit_diagnostic_recommendation result/state mismatch before finalization"
                )
        elif (
            result.get("status") != "halted"
            or terminal.get("status") != "halted"
            or terminal.get("validated") is not True
        ):
            raise RuntimeError("halt_diagnostics result/state mismatch before finalization")

    def _require_clean_session_closure(
        self,
        attachment: DiagnosticAttachmentContext,
        state: dict[str, Any],
    ) -> None:
        for handle in attachment.session_handles:
            context = self.session_registry.inspect(
                handle=handle,
                run_root=attachment.run_root,
            )
            session = self._session_entry(state, handle)
            close_events = [
                event
                for event in session.get("tool_events", [])
                if isinstance(event, dict) and event.get("tool") == "close_genesis_live_session"
            ]
            if (
                context.state is not DiagnosticSessionState.CLOSED
                or context.causal_error is not None
                or context.cleanup_error is not None
                or session.get("lifecycle_state") != "closed"
                or session.get("failure") is not None
                or not close_events
                or close_events[-1].get("status") != "ok"
            ):
                raise RuntimeError(
                    "successful diagnostic recommendation requires every Genesis live session "
                    f"to close cleanly: {handle}"
                )

    def _cleanup_revision_sessions(
        self,
        attachment: DiagnosticAttachmentContext,
    ) -> None:
        """Close owned live sessions as server-owned revision terminal cleanup."""
        owned_open_contexts = self.session_registry.open_contexts(
            run_root=attachment.run_root,
            agent_invocation_id=attachment.agent_invocation_id,
        )
        handles = tuple(
            dict.fromkeys(
                (
                    *attachment.session_handles,
                    *(context.handle for context in owned_open_contexts),
                )
            )
        )
        for handle in handles:
            context = self.session_registry.inspect(handle=handle, run_root=attachment.run_root)
            if context.agent_invocation_id != attachment.agent_invocation_id:
                raise PermissionError("revision cleanup session belongs to another invocation")
            if context.state is DiagnosticSessionState.CLOSED_FAILED:
                if context.cleanup_error:
                    raise RuntimeError(
                        f"revision terminal cleanup failed for {handle}: {context.cleanup_error}"
                    )
                continue
            if context.state not in {DiagnosticSessionState.CLOSED, DiagnosticSessionState.CLOSED_FAILED}:
                try:
                    self.session_registry.close(handle=handle, run_root=attachment.run_root)
                except BaseException as error:
                    closed_at = context.closed_at or self._now().isoformat()

                    def failed(entry: dict[str, Any], state: dict[str, Any]) -> None:
                        del entry
                        session = self._session_entry(state, handle)
                        session.update(
                            {
                                "lifecycle_state": "closed_failed",
                                "closed_at": closed_at,
                                "failed_at": context.close_started_at or closed_at,
                                "failure": {
                                    "kind": "revision_terminal_cleanup_failed",
                                    "message": context.causal_error or _format_error(error),
                                },
                            }
                        )
                        if context.cleanup_error:
                            session["failure"]["message"] += f"; cleanup error: {context.cleanup_error}"
                        self._ensure_session_event(
                            state,
                            attachment.attachment_id,
                            session,
                            tool="server_revision_terminal_cleanup",
                            timestamp=closed_at,
                            status="error",
                        )

                    self._mutate_ownership_ledgers(attachment, failed)
                    raise
            closed_at = context.closed_at or self._now().isoformat()

            def closed(entry: dict[str, Any], state: dict[str, Any]) -> None:
                del entry
                session = self._session_entry(state, handle)
                session["lifecycle_state"] = "closed"
                session["closed_at"] = closed_at
                self._ensure_session_event(
                    state,
                    attachment.attachment_id,
                    session,
                    tool="server_revision_terminal_cleanup",
                    timestamp=closed_at,
                    status="ok",
                )

            self._mutate_ownership_ledgers(attachment, closed)
        if self.session_registry.has_open_session(run_root=attachment.run_root) or self._durable_open_sessions(
            load_state(attachment.run_root), attachment.run_root
        ):
            raise RuntimeError("revision terminal cleanup left a Genesis live session open")

    @staticmethod
    def _result_artifact_refs(result: dict[str, Any]) -> list[str]:
        refs: list[str] = []

        def visit(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    if child_key.endswith(("_path", "_paths")):
                        visit(child, child_key)
                    elif child_key in {"visual_evidence", "setup_preview", "artifact_paths"}:
                        visit(child, child_key)
            elif isinstance(value, list):
                for child in value:
                    visit(child, key)
            elif isinstance(value, str) and key.endswith(("_path", "_paths")) and value:
                refs.append(value)

        visit(result)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _active_episode_id(state: dict[str, Any]) -> str | None:
        diagnostics = state.get("diagnostics", {})
        active_index = diagnostics.get("active_episode_index") if isinstance(diagnostics, dict) else None
        if active_index is None:
            return None
        for episode in diagnostics.get("episodes", []):
            if isinstance(episode, dict) and int(episode.get("episode_index", 0)) == int(active_index):
                return str(episode.get("episode_id", "")) or None
        return None

    def _invoke_diagnostic_source(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool: Callable[..., dict[str, Any]],
        arguments: dict[str, Any],
        session_context: Any | None,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool, eval_str=True)
        with bind_active_diagnostic_run(str(attachment.run_root)):
            if session_context is None:
                if "run_root" in signature.parameters:
                    return tool(str(attachment.run_root), **arguments)
                return tool(**arguments)
            with bind_live_tool_handlers(session_context.handlers):
                if "run_root" in signature.parameters:
                    return tool(str(attachment.run_root), **arguments)
                return tool(**arguments)

    def _dispatch_with_session_lock(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool: Callable[..., dict[str, Any]],
        arguments: dict[str, Any],
        session_context: Any | None,
    ) -> dict[str, Any]:
        if session_context is None:
            return self._invoke_diagnostic_source(
                attachment,
                tool=tool,
                arguments=arguments,
                session_context=None,
            )
        name = tool.__name__
        allowed_states = None
        if name == "simulation_reset":
            allowed_states = {DiagnosticSessionState.BOUND}
        elif name == "simulate":
            allowed_states = {DiagnosticSessionState.RESET, DiagnosticSessionState.ACTIVE}
        elif name == "query_live_geometry_context":
            allowed_states = {
                DiagnosticSessionState.BOUND,
                DiagnosticSessionState.RESET,
                DiagnosticSessionState.ACTIVE,
            }
        elif name == "inspect_genesis_runtime_logs":
            allowed_states = {
                DiagnosticSessionState.BOUND,
                DiagnosticSessionState.RESET,
                DiagnosticSessionState.ACTIVE,
            }
        with self.session_registry.locked_context(
            handle=session_context.handle,
            run_root=attachment.run_root,
            allowed_states=allowed_states,
        ) as locked:
            if locked.agent_invocation_id != attachment.agent_invocation_id:
                raise PermissionError("diagnostic session belongs to another invocation")
            return self._invoke_diagnostic_source(
                attachment,
                tool=tool,
                arguments=arguments,
                session_context=locked,
            )

    def dispatch_diagnostic_tool(
        self,
        *,
        tool: Callable[..., dict[str, Any]],
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        live_session_handle: str | None,
        arguments: dict[str, Any],
    ) -> list[TextContent | ImageContent]:
        name = tool.__name__
        span_id = ""
        try:
            from hag4r.agentic.runtime_profiler import start_span

            span_id = start_span(
                run_root,
                stage="genesis_diagnostics",
                operation=name,
                metadata={"has_live_session": live_session_handle is not None},
            )
        except Exception:
            pass
        try:
            result = self._dispatch_diagnostic_tool_unprofiled(
                tool=tool,
                run_root=run_root,
                agent_invocation_id=agent_invocation_id,
                owner_lease_token=owner_lease_token,
                live_session_handle=live_session_handle,
                arguments=arguments,
            )
        except BaseException:
            try:
                from hag4r.agentic.runtime_profiler import finish_span, record_stage_terminal

                finish_span(
                    run_root,
                    span_id,
                    stage="genesis_diagnostics",
                    operation=name,
                    status="failed",
                )
                if name in DIAGNOSTIC_TERMINAL_TOOL_NAMES:
                    record_stage_terminal(
                        run_root,
                        stage="genesis_diagnostics",
                        operation=name,
                        status="failed",
                    )
            except Exception:
                pass
            raise
        try:
            from hag4r.agentic.runtime_profiler import finish_span, record_stage_terminal

            finish_span(
                run_root,
                span_id,
                stage="genesis_diagnostics",
                operation=name,
                status="success",
            )
            if name in DIAGNOSTIC_TERMINAL_TOOL_NAMES:
                terminal_status = "halted" if name == "halt_diagnostics" else "success"
                record_stage_terminal(
                    run_root,
                    stage="genesis_diagnostics",
                    operation=name,
                    status=terminal_status,
                )
        except Exception:
            pass
        return result

    def _dispatch_diagnostic_tool_unprofiled(
        self,
        *,
        tool: Callable[..., dict[str, Any]],
        run_root: str,
        agent_invocation_id: str,
        owner_lease_token: str,
        live_session_handle: str | None,
        arguments: dict[str, Any],
    ) -> list[TextContent | ImageContent]:
        name = tool.__name__
        if name not in DETERMINISTIC_MCP_TOOL_NAMES:
            raise ValueError(f"tool is not in the authoritative diagnostic tuple: {name}")
        with self._authenticated_request(
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
        ) as attachment:
            is_revision_terminal = False
            requires_closed_session = name in {
                "define_episode", "audit_source_semantic_material_invariants", "halt_diagnostics"
            }
            if name == "submit_diagnostic_recommendation":
                submitted_recommendation = validate_vlm_final_recommendation(arguments.get("recommendation"))
                requires_closed_session = submitted_recommendation["recommendation"] == "accept"
                is_revision_terminal = submitted_recommendation["recommendation"] == "revise"
            if requires_closed_session:
                self.require_no_open_session(attachment, tool_name=name)
            open_contexts = self.session_registry.open_contexts(run_root=attachment.run_root)
            if any(
                context.agent_invocation_id != attachment.agent_invocation_id
                for context in open_contexts
            ):
                raise PermissionError("open diagnostic session belongs to another invocation")
            if len(open_contexts) > 1 and not is_revision_terminal:
                raise RuntimeError("diagnostic attachment has ambiguous open sessions")
            session_context = (
                open_contexts[0]
                if len(open_contexts) == 1 and not is_revision_terminal
                else None
            )
            state_before = load_state(attachment.run_root)
            if name == "audit_source_semantic_material_invariants":
                submitted_audit = arguments.get("audit")
                reflection_index = submitted_audit.get("trigger_reflection_index") if isinstance(submitted_audit, dict) else None
                reflections = state_before.get("diagnostics", {}).get("model_reflections", [])
                if isinstance(reflection_index, int) and 0 <= reflection_index < len(reflections):
                    reflection = reflections[reflection_index]
                    if not isinstance(reflection, dict):
                        raise PermissionError("material audit trigger reflection is malformed")
                    if (
                        reflection.get("attachment_id") != attachment.attachment_id
                        or reflection.get("agent_invocation_id") != attachment.agent_invocation_id
                    ):
                        raise PermissionError("material audit trigger reflection belongs to another attachment or invocation")
            durable_runtime_failure = (
                None
                if is_revision_terminal
                else self._attachment_runtime_failure(state_before, attachment.attachment_id)
            )
            if (
                session_context is not None
                and session_context.state is DiagnosticSessionState.RUNTIME_FAILED
                and not is_revision_terminal
            ):
                raise RuntimeError(f"{name} is forbidden after runtime_failure; close the live session")
            if durable_runtime_failure is not None:
                evidence = arguments.get("evidence") if name == "record_diagnostic_evidence" else None
                is_sessionless_pre_terminal = (
                    session_context is None
                    and isinstance(evidence, dict)
                    and evidence.get("phase") == "pre_terminal"
                )
                if name not in DIAGNOSTIC_TERMINAL_TOOL_NAMES and not is_sessionless_pre_terminal:
                    raise RuntimeError(
                        f"{name} is forbidden after runtime_failure; only close, sessionless "
                        "pre_terminal evidence, and terminal submission are allowed"
                    )
            if name in LIVE_SESSION_ARGUMENT_TOOL_NAMES:
                if live_session_handle is None:
                    raise ValueError(f"{name} requires live_session_handle")
                self._require_owned_context(attachment, live_session_handle)
                session_context = self.session_registry.inspect(
                    handle=live_session_handle,
                    run_root=attachment.run_root,
                )
                if session_context.agent_invocation_id != attachment.agent_invocation_id:
                    raise PermissionError("live session handle belongs to another invocation")
            elif live_session_handle is not None:
                raise ValueError(f"{name} does not accept live_session_handle")

            episode_id = (
                session_context.episode_id
                if session_context is not None
                else self._active_episode_id(state_before)
            )
            terminal_phase = name in DIAGNOSTIC_TERMINAL_TOOL_NAMES or (
                name == "record_diagnostic_evidence"
                and session_context is None
                and bool(attachment.session_handles)
            )
            event_recorded = False
            correlated_tool_result_index: int | None = None
            reflection_index: int | None = None
            runtime_failure_summary: dict[str, Any] | None = None
            try:
                submitted_evidence = (
                    arguments.get("evidence")
                    if name == "record_diagnostic_evidence"
                    else None
                )
                if (
                    isinstance(submitted_evidence, dict)
                    and submitted_evidence.get("phase") == "runtime_failure"
                ):
                    if session_context is None:
                        raise RuntimeError("runtime_failure evidence requires an open Genesis live session")
                    self._validate_runtime_failure_reflection(
                        attachment,
                        session_context,
                        submitted_evidence,
                        state_before,
                    )
                result = self._dispatch_with_session_lock(
                    attachment,
                    tool=tool,
                    arguments=arguments,
                    session_context=session_context,
                )
                if not isinstance(result, dict):
                    raise TypeError(f"diagnostic tool {name} must return an object")
                artifact_refs = self._result_artifact_refs(result)
                domain_success = self._result_successful(result)
                if name == "record_diagnostic_evidence" and domain_success:
                    evidence = result.get("evidence")
                    if isinstance(evidence, dict):
                        raw_reflection_index = evidence.get("reflection_index")
                        if isinstance(raw_reflection_index, bool) or not isinstance(raw_reflection_index, int):
                            raise ValueError("record_diagnostic_evidence result is missing reflection_index")
                        reflection_index = raw_reflection_index
                    if isinstance(evidence, dict) and evidence.get("phase") == "runtime_failure":
                        if session_context is None:
                            raise RuntimeError("runtime_failure evidence requires an open Genesis live session")
                        cited_index, inspection_event = self._validate_runtime_failure_reflection(
                            attachment,
                            session_context,
                            evidence,
                            load_state(attachment.run_root),
                        )
                        self.session_registry.transition(
                            context=session_context,
                            event=DiagnosticSessionEvent.RUNTIME_FAILURE_RECORDED,
                        )
                        self._persist_runtime_session_lifecycle(attachment, session_context)
                        runtime_failure_summary = {
                            "reflection_index": reflection_index,
                            "cited_tool_result_index": cited_index,
                            "cited_tool_result_ref": f"tool_result:{cited_index + 1}",
                            "attachment_id": attachment.attachment_id,
                            "episode_id": session_context.episode_id,
                            "live_session_handle": session_context.handle,
                            "inspection_event_sequence_index": inspection_event["sequence_index"],
                        }
                if name == "simulation_reset" and domain_success:
                    assert session_context is not None
                    self.session_registry.transition(
                        context=session_context,
                        event=DiagnosticSessionEvent.RESET_SUCCEEDED,
                    )
                    self._persist_runtime_session_lifecycle(attachment, session_context)
                elif name == "simulate" and domain_success:
                    assert session_context is not None
                    self.session_registry.transition(
                        context=session_context,
                        event=DiagnosticSessionEvent.ACTIVITY_SUCCEEDED,
                    )
                    self._persist_runtime_session_lifecycle(attachment, session_context)
                if name in DIAGNOSTIC_TERMINAL_TOOL_NAMES:
                    self._validate_terminal_pre_finalization(
                        name,
                        result,
                        load_state(attachment.run_root),
                    )
                    self._record_ownership_tool_event(
                        attachment,
                        tool=name,
                        status="ok",
                        terminal_phase=True,
                        episode_id=episode_id,
                        artifact_refs=artifact_refs,
                    )
                    event_recorded = True
                    return self._finish_terminal_request(
                        attachment,
                        tool_name=name,
                        result=result,
                    )
                state_after = load_state(attachment.run_root)
                if name == "inspect_genesis_runtime_logs":
                    raw_index = result.get("tool_result_index")
                    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                        raise ValueError("Genesis runtime log result is missing tool_result_index")
                    if session_context is None:
                        raise RuntimeError("Genesis runtime log result is missing its live session")
                    self._validate_durable_tool_result_correlation(
                        state_after,
                        tool=name,
                        tool_result_index=raw_index,
                        episode_id=session_context.episode_id,
                    )
                    correlated_tool_result_index = raw_index
                elif domain_success:
                    raw_index = result.get("tool_result_index")
                    if isinstance(raw_index, int) and not isinstance(raw_index, bool) and session_context is not None:
                        self._validate_durable_tool_result_correlation(
                            state_after,
                            tool=name,
                            tool_result_index=raw_index,
                            episode_id=session_context.episode_id,
                        )
                        correlated_tool_result_index = raw_index
                formatted = self._format_diagnostic_mcp_result(
                    name,
                    result,
                    state_after,
                )
                has_image_blocks = any(isinstance(block, ImageContent) for block in formatted)
                event_episode_id = (
                    session_context.episode_id
                    if session_context is not None
                    else self._active_episode_id(load_state(attachment.run_root))
                )
                ownership_event = self._record_ownership_tool_event(
                    attachment,
                    tool=name,
                    status="ok" if domain_success else "error",
                    session_handle=session_context.handle if session_context is not None else None,
                    terminal_phase=terminal_phase,
                    episode_id=event_episode_id,
                    observation_kinds=["part_segmentation_triptych"] if domain_success and has_image_blocks else [],
                    artifact_refs=artifact_refs if domain_success else [],
                    tool_result_index=correlated_tool_result_index,
                    reflection_index=reflection_index,
                )
                if reflection_index is not None:
                    if runtime_failure_summary is not None:
                        runtime_failure_summary["event_sequence_index"] = ownership_event["sequence_index"]
                    self._enrich_persisted_reflection(
                        attachment,
                        reflection_index=reflection_index,
                        episode_id=event_episode_id,
                        session_handle=session_context.handle if session_context is not None else None,
                        event=ownership_event,
                        runtime_failure=runtime_failure_summary,
                    )
                event_recorded = True
                return formatted
            except BaseException as error:
                if name == "inspect_genesis_runtime_logs" and correlated_tool_result_index is None:
                    error_index = getattr(error, "diagnostic_tool_result_index", None)
                    if isinstance(error_index, int) and not isinstance(error_index, bool):
                        if session_context is None:
                            raise RuntimeError("Genesis runtime log failure lost its live session") from error
                        self._validate_durable_tool_result_correlation(
                            load_state(attachment.run_root),
                            tool=name,
                            tool_result_index=error_index,
                            episode_id=session_context.episode_id,
                        )
                        correlated_tool_result_index = error_index
                terminal_state = load_state(attachment.run_root).get("diagnostics", {}).get("terminal")
                terminal_commit_started = (
                    name in DIAGNOSTIC_TERMINAL_TOOL_NAMES
                    and isinstance(terminal_state, dict)
                    and terminal_state.get("status") in {"pending_validation", "running", "success", "rejected"}
                )
                if terminal_commit_started and attachment.status not in _ATTACHMENT_TERMINAL_STATUSES:
                    if not event_recorded:
                        try:
                            self._record_ownership_tool_event(
                                attachment,
                                tool=name,
                                status="error",
                                terminal_phase=True,
                                episode_id=episode_id,
                            )
                        except BaseException as evidence_error:
                            error.add_note(
                                f"failed to record terminal tool error evidence: {evidence_error}"
                            )
                    self._fail_terminal_request(attachment, tool_name=name, cause=error)
                elif not event_recorded and attachment.status not in _ATTACHMENT_TERMINAL_STATUSES:
                    try:
                        self._record_ownership_tool_event(
                            attachment,
                            tool=name,
                            status="error",
                            session_handle=session_context.handle if session_context is not None else None,
                            terminal_phase=terminal_phase,
                            episode_id=episode_id,
                            tool_result_index=correlated_tool_result_index,
                        )
                    except BaseException as evidence_error:
                        error.add_note(f"failed to record diagnostic tool error evidence: {evidence_error}")
                transport_failure_kind = (
                    self._transport_loss_failure_kind(error)
                    if name in LIVE_SESSION_ARGUMENT_TOOL_NAMES
                    else None
                )
                if (
                    transport_failure_kind is not None
                    and attachment.status not in _ATTACHMENT_TERMINAL_STATUSES
                ):
                    try:
                        self._terminalize_transport_loss(
                            attachment,
                            failure_kind=transport_failure_kind,
                            cause=error,
                        )
                    except BaseException as recovery_error:
                        error.add_note(
                            "failed to terminalize typed Genesis transport loss: "
                            f"{_format_error(recovery_error)}"
                        )
                raise

    def _persist_runtime_session_lifecycle(
        self,
        attachment: DiagnosticAttachmentContext,
        context: Any,
    ) -> None:
        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del entry
            session = self._session_entry(state, context.handle)
            session["lifecycle_state"] = context.state.value

        try:
            self._mutate_ownership_ledgers(attachment, mutation)
        except BaseException as writer_error:
            try:
                self._mutate_ownership_ledgers(attachment, mutation)
            except BaseException as reconciliation_error:
                writer_error.add_note(
                    f"failed to reconcile diagnostic live lifecycle: {reconciliation_error}"
                )
            raise

    def _build_terminal_agent_record(
        self,
        attachment: DiagnosticAttachmentContext,
        terminal_tool: str,
        status: str,
    ) -> dict[str, Any]:
        state = load_state(attachment.run_root)
        events = self._attachment_events(state, attachment.attachment_id)
        ordered = sorted(events, key=lambda event: int(event["sequence_index"]))
        indices = [int(event["sequence_index"]) for event in ordered]
        if indices != list(range(len(indices))):
            raise RuntimeError(f"diagnostic ownership event sequence is gapped or duplicated: {indices}")
        attachment_entry = self._attachment_entry(state, attachment.attachment_id)
        if (
            attachment_entry.get("agent_invocation_id") != attachment.agent_invocation_id
            or attachment_entry.get("run_root") != str(attachment.run_root)
        ):
            raise PermissionError("terminal ownership invocation mismatch")
        sessions = state.get("diagnostic_runtime_sessions", [])
        if any(
            isinstance(session, dict)
            and session.get("attachment_id") == attachment.attachment_id
            and (
                session.get("agent_invocation_id") != attachment.agent_invocation_id
                or session.get("run_root") != str(attachment.run_root)
            )
            for session in sessions
        ):
            raise PermissionError("terminal session ownership mismatch")
        return {
            "agent_kind": OWNER_KIND,
            "agent_invocation_id": attachment.agent_invocation_id,
            "attachment_id": attachment.attachment_id,
            "status": status,
            "terminal_tool": terminal_tool,
            "tool_call_sequence": [str(event["tool"]) for event in ordered],
            "tool_events": ordered,
        }

    def _terminalize_attachment(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        status: str,
        terminal_tool: str,
        artifact_refs: list[str],
        error: str = "",
        agent_record: dict[str, Any] | None = None,
        terminal_result: dict[str, Any] | None = None,
    ) -> None:
        timestamp = self._now().isoformat()

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            entry["completed_at"] = timestamp
            entry["status"] = status
            entry["token_invalidated_at"] = timestamp
            entry["watchdog_registered"] = False
            entry["owner_lease"]["status"] = "closed"
            entry["owner_lease"]["in_flight_count"] = 0
            finalization = {
                "event": "terminal_finalization",
                "timestamp": timestamp,
                "status": "error" if error else "ok",
                "terminal_tool": terminal_tool,
                "agent_invocation_id": attachment.agent_invocation_id,
                "artifact_refs": artifact_refs,
                "error": error,
            }
            if agent_record is not None:
                finalization["agent_record"] = agent_record
            entry["history"].append(finalization)
            existing_event = next(
                (
                    event
                    for event in entry["terminal_tool_events"]
                    if isinstance(event, dict)
                    and event.get("tool") == "terminal_finalization"
                    and event.get("terminal_tool") == terminal_tool
                ),
                None,
            )
            bounded_result = dict(terminal_result or {})
            bounded_result.update(
                {
                    "attachment_status": status,
                    "terminal_status": bounded_result.get(
                        "terminal_status",
                        "failed" if error else status,
                    ),
                }
            )
            if error:
                bounded_result["error"] = error
            if existing_event is None:
                existing_event = self._new_tool_event(
                    state,
                    attachment.attachment_id,
                    tool="terminal_finalization",
                    timestamp=timestamp,
                    status="error" if error else "ok",
                    artifact_refs=artifact_refs,
                    episode_id=self._active_episode_id(state),
                )
                entry["terminal_tool_events"].append(existing_event)
            else:
                existing_event.update(
                    {
                        "timestamp": timestamp,
                        "status": "error" if error else "ok",
                        "artifact_refs": list(artifact_refs),
                    }
                )
            existing_event.update(
                {
                    "agent_invocation_id": attachment.agent_invocation_id,
                    "terminal_tool": terminal_tool,
                    "result": bounded_result,
                }
            )

        self._mutate_ownership_ledgers(attachment, mutation)
        attachment.status = status
        attachment.watchdog_registered = False
        attachment.owner_lease_token = secrets.token_urlsafe(32)

    def _finish_terminal_request(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool_name: str,
        result: dict[str, Any],
    ) -> list[TextContent | ImageContent]:
        if tool_name == "submit_diagnostic_recommendation":
            if result.get("status") != "pending_validation":
                raise RuntimeError("submit_diagnostic_recommendation did not return pending_validation")
            pending_state = load_state(attachment.run_root)
            pending_terminal = pending_state.get("diagnostics", {}).get("terminal", {})
            pending_recommendation = pending_terminal.get("recommendation")
            if not isinstance(pending_recommendation, dict):
                raise RuntimeError("submit_diagnostic_recommendation terminal lacks its recommendation")
            normalized_recommendation = validate_vlm_final_recommendation(pending_recommendation)
            is_accept = normalized_recommendation["recommendation"] == "accept"
            agent_record = self._build_terminal_agent_record(attachment, tool_name, "success")
            finalize_diagnostic_terminal(
                attachment.run_root,
                status="success",
                agent_record=agent_record,
            )
            if is_accept:
                self._require_clean_session_closure(attachment, pending_state)
            else:
                self._cleanup_revision_sessions(attachment)
            cleanup_state = load_state(attachment.run_root)
            cleanup_terminal = cleanup_state.get("diagnostics", {}).get("terminal", {})
            cleanup_operational = (
                cleanup_terminal.get("operational_outcome")
                if isinstance(cleanup_terminal, Mapping)
                else None
            )
            if isinstance(cleanup_operational, dict) and cleanup_operational.get("status") == "pending":
                cleanup_operational["cleanup_status"] = "complete"
                save_state(cleanup_state, attachment.run_root)
            materialization = materialize_diagnostic_terminal_artifacts(
                attachment.run_root, agent_record
            )
            state = load_state(attachment.run_root)
            terminal = state.get("diagnostics", {}).get("terminal", {})
            stage = state.get("stages", {}).get(DIAGNOSTIC_STAGE_NAME, {})
            artifacts = state.get("diagnostics", {}).get("artifact_paths", {})
            if terminal.get("status") != "success" or terminal.get("validated") is not True:
                raise RuntimeError("diagnostic recommendation finalizer did not validate success")
            if stage.get("ok") is not True or not isinstance(artifacts, dict):
                raise RuntimeError("diagnostic recommendation finalizer did not preserve business stage success")
            required_artifact_names = ("summary", "markdown", "cues")
            missing_required_artifacts = [
                name
                for name in required_artifact_names
                if not isinstance(artifacts.get(name), str) or not artifacts[name]
            ]
            artifact_refs = [
                str(artifacts[name])
                for name in required_artifact_names
                if isinstance(artifacts.get(name), str) and artifacts[name]
            ]
            # Preserve existing derived media references for audit consumers,
            # but never make an absent optional artifact a terminal failure.
            for optional_name in ("videos", "triptych_videos"):
                optional_values = artifacts.get(optional_name, [])
                if isinstance(optional_values, list):
                    artifact_refs.extend(
                        str(path)
                        for path in optional_values
                        if isinstance(path, str)
                        and path
                        and Path(path).expanduser().is_file()
                    )
            artifact_refs = list(dict.fromkeys(artifact_refs))
            missing_artifacts = missing_required_artifacts + [
                ref for ref in artifact_refs if not Path(ref).expanduser().is_file()
            ]
            terminal_result = {
                "terminal_status": "success",
                "validated": True,
                "route": terminal.get("route"),
                "ready": bool(terminal.get("ready", False)),
                "artifact_paths": dict(artifacts) if isinstance(artifacts, Mapping) else {},
                "artifact_status": materialization["artifact_status"],
                "missing_artifacts": materialization["missing_artifacts"],
            }
            if is_accept:
                terminal_result["evidence_mode"] = "probe_window"
            self._terminalize_attachment(
                attachment,
                status="completed",
                terminal_tool=tool_name,
                artifact_refs=artifact_refs,
                agent_record=agent_record,
                terminal_result=terminal_result,
            )
            final_result = {
                "tool": tool_name,
                "status": "success",
                "validated": True,
                "attachment_status": "completed",
                "owner_lease_status": "closed",
                "watchdog_registered": False,
                "route": terminal.get("route"),
                "ready": terminal.get("ready"),
                "artifact_refs": artifact_refs,
                "artifact_paths": dict(artifacts) if isinstance(artifacts, Mapping) else {},
                "artifact_status": materialization["artifact_status"],
                "missing_artifacts": list(dict.fromkeys(missing_artifacts + materialization["missing_artifacts"])),
                "agent_record": agent_record,
            }
            if is_accept:
                final_result["evidence_mode"] = "probe_window"
        else:
            state = load_state(attachment.run_root)
            terminal = state.get("diagnostics", {}).get("terminal", {})
            halt_route = state.get("route_state", {}).get("diagnostic_route", {})
            if not isinstance(halt_route, dict):
                halt_route = {}
            if result.get("status") != "halted" or terminal.get("status") != "halted":
                raise RuntimeError("halt_diagnostics did not persist halted terminal state")
            agent_record = self._build_terminal_agent_record(attachment, tool_name, "halted")
            self._terminalize_attachment(
                attachment,
                status="halted",
                terminal_tool=tool_name,
                artifact_refs=[],
                agent_record=agent_record,
                terminal_result={
                    "terminal_status": "halted",
                    "validated": bool(terminal.get("validated", False)),
                    "route": terminal.get("route"),
                    "original_route": halt_route.get("original_route"),
                    "effective_route": halt_route.get("effective_route"),
                    "recommended_stage_skill_paths": list(
                        halt_route.get("recommended_stage_skill_paths", [])
                    ),
                    "reentry_stage_skill_paths": list(
                        halt_route.get("reentry_stage_skill_paths", [])
                    ),
                    "force_rerun_stage_skill_paths": list(
                        halt_route.get("force_rerun_stage_skill_paths", [])
                    ),
                    "ready": False,
                },
            )
            final_result = {
                "tool": tool_name,
                "status": "halted",
                "validated": bool(terminal.get("validated", False)),
                "attachment_status": "halted",
                "owner_lease_status": "closed",
                "watchdog_registered": False,
                "route": terminal.get("route"),
                "original_route": halt_route.get("original_route"),
                "effective_route": halt_route.get("effective_route"),
                "recommended_stage_skill_paths": list(
                    halt_route.get("recommended_stage_skill_paths", [])
                ),
                "reentry_stage_skill_paths": list(
                    halt_route.get("reentry_stage_skill_paths", [])
                ),
                "force_rerun_stage_skill_paths": list(
                    halt_route.get("force_rerun_stage_skill_paths", [])
                ),
                "ready": False,
                "error": terminal.get("error", ""),
                "agent_record": agent_record,
            }
        return self._format_diagnostic_mcp_result(
            tool_name,
            final_result,
            load_state(attachment.run_root),
        )

    def _fail_terminal_request(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        tool_name: str,
        cause: BaseException,
    ) -> None:
        message = _format_error(cause)
        secondary_errors: list[BaseException] = []
        try:
            agent_record = self._build_terminal_agent_record(attachment, tool_name, "failed")
        except BaseException as error:
            secondary_errors.append(error)
            agent_record = {
                "agent_kind": OWNER_KIND,
                "agent_invocation_id": attachment.agent_invocation_id,
                "attachment_id": attachment.attachment_id,
                "status": "failed",
                "terminal_tool": tool_name,
                "tool_call_sequence": [],
                "tool_events": [],
            }
        state = load_state(attachment.run_root)
        terminal = state.get("diagnostics", {}).get("terminal")
        business = terminal.get("business_outcome") if isinstance(terminal, dict) else None
        business_committed = (
            isinstance(business, Mapping) and business.get("status") == "recommendation_valid"
        )
        if business_committed:
            self._record_operational_cleanup_failure(
                attachment.run_root,
                message,
                failure_kind="terminal_cleanup_failed",
                operational_status="cleanup_retryable",
                cleanup_status="retryable",
                record_stage_on_failure=False,
            )
            try:
                self._terminalize_attachment(
                    attachment,
                    status="failed",
                    terminal_tool=tool_name,
                    artifact_refs=[],
                    error=message,
                    agent_record=agent_record,
                    terminal_result={
                        "terminal_status": "success",
                        "validated": True,
                        "operational_status": "cleanup_retryable",
                    },
                )
            except BaseException as error:
                secondary_errors.append(error)
            for secondary_error in secondary_errors:
                cause.add_note(f"secondary terminal failure: {_format_error(secondary_error)}")
            return
        if (
            tool_name == "submit_diagnostic_recommendation"
            and isinstance(terminal, dict)
            and terminal.get("status") in {"pending_validation", "running", "success", "rejected"}
        ):
            try:
                finalize_diagnostic_terminal(
                    attachment.run_root,
                    status="failed",
                    agent_record=agent_record,
                    error=message,
                )
            except BaseException as error:
                secondary_errors.append(error)
        try:
            record_stage(
                attachment.run_root,
                DIAGNOSTIC_STAGE_NAME,
                ok=False,
                status="failed",
                error=message,
                allow_new=False,
            )
        except BaseException as error:
            secondary_errors.append(error)
        try:
            self._terminalize_attachment(
                attachment,
                status="failed",
                terminal_tool=tool_name,
                artifact_refs=[],
                error=message,
                agent_record=agent_record,
                terminal_result={"terminal_status": "failed", "validated": False},
            )
        except BaseException as error:
            secondary_errors.append(error)
        for error in secondary_errors:
            cause.add_note(f"secondary terminal failure: {_format_error(error)}")

    @staticmethod
    def _compact_result_metadata(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "inspect_genesis_runtime_logs":
            wrapper = result.get("result")
            envelope = wrapper.get("result") if isinstance(wrapper, dict) else None
            if not isinstance(envelope, dict) or envelope.get("schema_version") != GENESIS_RUNTIME_LOG_SCHEMA_VERSION:
                raise ValueError("Genesis runtime log result is missing its raw v1 envelope")
            tool_result_index = result.get("tool_result_index")
            if isinstance(tool_result_index, bool) or not isinstance(tool_result_index, int):
                raise ValueError("Genesis runtime log result is missing tool_result_index")
            return {
                "tool": tool_name,
                "status": "ok" if DiagnosticMcpRuntime._result_successful(result) else "error",
                "tool_result_index": tool_result_index,
                "result": envelope,
            }
        allowed = {
            "tool",
            "status",
            "route",
            "original_route",
            "effective_route",
            "recommended_stage_skill_paths",
            "reentry_stage_skill_paths",
            "force_rerun_stage_skill_paths",
            "ready",
            "validated",
            "attachment_status",
            "owner_lease_status",
            "watchdog_registered",
            "error",
            "next_action",
            "episode_id",
            "anchor_id",
            "target_id",
            "compile_id",
            "semantic_group_part_ids",
            "semantic_group_part_grounding",
            "trial_id",
            "expected_observation",
            "actual_observation",
            "frame_ids",
            "count",
            "artifact_refs",
            "artifact_paths",
            "artifact_status",
            "missing_artifacts",
            "operational_status",
            "validation_status",
            "warning",
            "triptych_png_path",
            "triple_view_evidence_id",
            "evidence_mode",
        }
        compact = {key: result[key] for key in allowed if key in result}
        compact.setdefault("tool", tool_name)
        domain_status = DiagnosticMcpRuntime._domain_result_status(result)
        compact["status"] = domain_status or ("error" if result.get("error") else "success")
        if tool_name in {
            "compile_diagnostic_anchor_target",
            "compile_diagnostic_probe_target",
        } and DiagnosticMcpRuntime._result_successful(result):
            selected_part_id = result.get("selected_part_id")
            part_name = result.get("part_name")
            part_semantics = result.get("part_semantics")
            if (
                isinstance(selected_part_id, bool)
                or not isinstance(selected_part_id, int)
                or selected_part_id < 0
                or not isinstance(part_name, str)
                or not part_name
                or not isinstance(part_semantics, str)
                or not part_semantics
            ):
                raise ValueError("compiled target MCP result is missing its canonical part triple")
            compact.update(
                {
                    "selected_part_id": selected_part_id,
                    "part_name": part_name,
                    "part_semantics": part_semantics,
                }
            )
        part_grounding_context = result.get("part_grounding_context")
        if isinstance(part_grounding_context, dict):
            parts = part_grounding_context.get("parts")
            if not isinstance(parts, list):
                raise ValueError("part grounding MCP result is missing its canonical parts table")
            compact["part_grounding_context"] = {
                "parts": [
                    {
                        "part_id": part["part_id"],
                        "part_name": part["part_name"],
                        "part_semantics": part["part_semantics"],
                    }
                    for part in parts
                    if isinstance(part, dict)
                ]
            }
        nested_result = result.get("result") if isinstance(result.get("result"), dict) else {}
        nested_error = nested_result.get("error") if isinstance(nested_result.get("error"), dict) else None
        if nested_error is not None:
            compact["error"] = {
                key: nested_error[key]
                for key in ("code", "message")
                if key in nested_error
            }
        visual = result.get("visual_evidence")
        if isinstance(visual, dict):
            for key in ("evidence_id", "frame_ids", "count"):
                if key in visual:
                    compact[key] = visual[key]
        setup_preview = result.get("setup_preview")
        if isinstance(setup_preview, dict):
            compact["setup_preview"] = {
                key: setup_preview[key]
                for key in (
                    "trial_id",
                    "anchor_id",
                    "anchor_compile_id",
                    "anchor_preview_evidence_id",
                    "preview_image_path",
                    "validation_status",
                    "warning_codes",
                    "next_action",
                )
                if key in setup_preview
            }
        episode = result.get("episode")
        if isinstance(episode, dict) and episode.get("episode_id"):
            compact["episode_id"] = str(episode["episode_id"])
        probe_validation = result.get("probe_target_validation")
        if isinstance(probe_validation, dict):
            compact["probe_target_validation"] = {
                key: probe_validation[key]
                for key in (
                    "status",
                    "target_id",
                    "compile_id",
                    "semantic_group_part_ids",
                    "semantic_group_part_grounding",
                    "grabbed_selected_part_fraction",
                    "warning_codes",
                )
                if key in probe_validation
            }
        return compact

    @staticmethod
    def _nested(value: dict[str, Any], *keys: str) -> Any:
        current: Any = value
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _visual_paths(self, tool_name: str, result: dict[str, Any]) -> list[str]:
        if tool_name in {"preview_diagnostic_anchor_target", "preview_diagnostic_probe_target"}:
            values = [result.get("triptych_png_path")]
        elif tool_name == "preview_diagnostic_episode_setup":
            values = [self._nested(result, "setup_preview", "preview_image_path")]
        else:
            values = self._nested(result, "visual_evidence", "triptych_png_paths")
            if not isinstance(values, list):
                values = self._nested(result, "result", "triple_view_sequence", "triptych_png_paths")
            if not isinstance(values, list):
                values = self._nested(result, "result", "triptych_png_paths")
            if not isinstance(values, list):
                values = []
        paths = [str(value) for value in values if isinstance(value, (str, Path)) and str(value)]
        paths = list(dict.fromkeys(paths))
        if tool_name == "simulate" and len(paths) > MAX_MODEL_VISIBLE_SIMULATE_IMAGES:
            middle = len(paths) // 2
            paths = [paths[0], paths[middle], paths[-1]]
        elif paths:
            paths = paths[:MAX_MODEL_VISIBLE_PREVIEW_IMAGES]
        return paths

    def _validated_image_bytes(self, path_text: str, state: dict[str, Any]) -> bytes:
        path = Path(path_text).expanduser().resolve()
        roots = [
            Path(str(state["paths"][key])).expanduser().resolve()
            for key in ("diagnostic_workspace_dir", "diagnostic_generated_episodes_dir")
            if state.get("paths", {}).get(key)
        ]
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError(f"model-visible diagnostic image is outside runtime roots: {path}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"model-visible diagnostic image is missing or empty: {path}")
        if path.stat().st_size > MAX_MODEL_VISIBLE_IMAGE_BYTES:
            raise ValueError(f"model-visible diagnostic image exceeds byte limit: {path}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"model-visible diagnostic image is not PNG: {path}")
            image.verify()
        return path.read_bytes()

    def _format_diagnostic_mcp_result(
        self,
        tool_name: str,
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> list[TextContent | ImageContent]:
        compact = self._compact_result_metadata(tool_name, result)
        compact_text = json.dumps(compact, sort_keys=True, separators=(",", ":"))
        if tool_name == "inspect_genesis_runtime_logs" and len(compact_text) > GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS:
            raise ValueError("Genesis runtime log MCP text exceeds the model-visible character limit")
        blocks: list[TextContent | ImageContent] = [
            TextContent(
                type="text",
                text=compact_text,
            )
        ]
        if tool_name not in VISUAL_RESULT_TOOL_NAMES:
            return blocks
        if not self._result_successful(result):
            return blocks
        paths = self._visual_paths(tool_name, result)
        if not paths:
            raise ValueError(f"successful visual diagnostic tool returned no canonical PNG: {tool_name}")
        blocks.extend(
            ImageContent(
                type="image",
                data=base64.b64encode(self._validated_image_bytes(path, state)).decode("ascii"),
                mimeType="image/png",
            )
            for path in paths
        )
        return blocks

    def _cleanup_attachment(self, attachment: DiagnosticAttachmentContext, *, failure_kind: str) -> str | None:
        if attachment.status in _ATTACHMENT_TERMINAL_STATUSES:
            return None
        timestamp = self._now().isoformat()
        cleanup_results: list[dict[str, Any]] = []
        for live_session_handle in tuple(attachment.session_handles):
            state = load_state(attachment.run_root)
            session_entry = self._session_entry(state, live_session_handle)
            context = self.session_registry.inspect(
                handle=live_session_handle,
                run_root=attachment.run_root,
            )
            if context.agent_invocation_id != attachment.agent_invocation_id:
                raise PermissionError("attached session invocation changed during owner cleanup")
            if session_entry.get("lifecycle_state") in _SESSION_TERMINAL_STATES:
                cleanup_results.append(
                    {
                        "episode_id": context.episode_id,
                        "live_session_handle": context.handle,
                        "lifecycle_state": str(session_entry["lifecycle_state"]),
                        "cleanup_error": context.cleanup_error,
                    }
                )
                continue
            failure_message = (
                f"{failure_kind}: diagnostic owner {attachment.agent_invocation_id} lost ownership "
                f"of episode {context.episode_id}"
            )

            def failed_closing(entry: dict[str, Any], state: dict[str, Any]) -> None:
                del entry
                session_entry = self._session_entry(state, context.handle)
                session_entry.update(
                    {
                        "lifecycle_state": "failed_closing",
                        "failed_at": timestamp,
                        "failure": {"kind": failure_kind, "message": failure_message},
                    }
                )

            if context.state not in {
                DiagnosticSessionState.CLOSED,
                DiagnosticSessionState.CLOSED_FAILED,
            }:
                self._mutate_ownership_ledgers(attachment, failed_closing)
                if context.state is DiagnosticSessionState.FAILED_CLOSING:
                    with self.session_registry.locked_context(
                        handle=context.handle,
                        run_root=attachment.run_root,
                        allowed_states={DiagnosticSessionState.FAILED_CLOSING},
                    ):
                        cleanup_error: BaseException | None = None
                        try:
                            context.live_session.close()
                        except BaseException as error:
                            cleanup_error = error
                        self.session_registry.transition(
                            context=context,
                            event=DiagnosticSessionEvent.FAILURE_CLEANUP_FINISHED,
                            cleanup_error=cleanup_error,
                        )
                else:
                    self.session_registry.fail_and_close(
                        handle=context.handle,
                        run_root=attachment.run_root,
                        causal_error=failure_message,
                    )
            result = {
                "episode_id": context.episode_id,
                "live_session_handle": context.handle,
                "lifecycle_state": context.state.value,
                "cleanup_error": context.cleanup_error,
            }
            cleanup_results.append(result)

            def closed_failed(entry: dict[str, Any], state: dict[str, Any]) -> None:
                del entry
                session_entry = self._session_entry(state, context.handle)
                session_entry["lifecycle_state"] = context.state.value
                session_entry["closed_at"] = context.closed_at or timestamp
                if context.state is DiagnosticSessionState.CLOSED_FAILED:
                    message = failure_message
                    if context.cleanup_error:
                        message += f"; cleanup error: {context.cleanup_error}"
                    session_entry["failure"] = {"kind": failure_kind, "message": message}

            self._mutate_ownership_ledgers(attachment, closed_failed)

        final_state = load_state(attachment.run_root)
        nonterminal_handles = [
            handle
            for handle in attachment.session_handles
            if self._session_entry(final_state, handle).get("lifecycle_state")
            not in _SESSION_TERMINAL_STATES
        ]
        if nonterminal_handles:
            raise RuntimeError(
                "cannot terminalize diagnostic attachment while session ledgers remain non-terminal: "
                + ", ".join(nonterminal_handles)
            )

        terminal_status = "expired" if failure_kind == "owner_lease_expired" else "failed"

        def terminal(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del state
            entry["status"] = terminal_status
            entry["completed_at"] = timestamp
            entry["token_invalidated_at"] = timestamp
            entry["watchdog_registered"] = False
            lease = entry["owner_lease"]
            lease["status"] = "expired" if failure_kind == "owner_lease_expired" else "closed"
            lease["in_flight_count"] = 0
            entry["history"].append(
                {
                    "event": failure_kind,
                    "timestamp": timestamp,
                    "attachment_id": attachment.attachment_id,
                    "agent_invocation_id": attachment.agent_invocation_id,
                    "episode_ids": [result["episode_id"] for result in cleanup_results],
                    "cleanup_results": cleanup_results,
                }
            )

        self._mutate_ownership_ledgers(attachment, terminal)
        attachment.status = terminal_status
        attachment.watchdog_registered = False
        attachment.owner_lease_token = secrets.token_urlsafe(32)
        return (
            f"{failure_kind}: diagnostic owner lease ended for invocation "
            f"{attachment.agent_invocation_id} (attachment {attachment.attachment_id})"
        )

    def _record_operational_cleanup_failure(
        self,
        run_root: str | Path,
        message: str,
        *,
        failure_kind: str,
        operational_status: str = "lease_expired_retryable",
        cleanup_status: str = "complete",
        record_stage_on_failure: bool = True,
    ) -> None:
        state = load_state(run_root)
        diagnostics = state.setdefault("diagnostics", {})
        terminal = diagnostics.get("terminal")
        if not isinstance(terminal, dict):
            terminal = {}
            diagnostics["terminal"] = terminal
        business = terminal.get("business_outcome")
        recommendation_committed = (
            isinstance(business, Mapping) and business.get("status") == "recommendation_valid"
        )
        if not isinstance(business, Mapping):
            terminal["business_outcome"] = build_diagnostic_business_outcome("not_authored")
        prior_operational = terminal.get("operational_outcome", {})
        terminal["operational_outcome"] = build_diagnostic_operational_outcome(
            operational_status,
            failure_kind=failure_kind,
            retryable=True,
            error=message,
            cleanup_status=cleanup_status,
            artifact_status=(
                prior_operational.get("artifact_status", {})
                if isinstance(prior_operational, Mapping)
                else {}
            ),
        )
        terminal["status"] = "success" if recommendation_committed else "retryable"
        terminal["validated"] = bool(recommendation_committed)
        if recommendation_committed:
            terminal["error"] = ""
        else:
            terminal["error"] = message
        save_state(state, run_root)
        if recommendation_committed or not record_stage_on_failure:
            return
        record_stage(
            run_root,
            DIAGNOSTIC_STAGE_NAME,
            ok=False,
            status="pending_retry",
            error=message,
            allow_new=DIAGNOSTIC_STAGE_NAME not in state.get("stages", {}),
        )

    def _record_transport_recovery_ready(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        failure_kind: str,
        message: str,
    ) -> str:
        """Persist the narrow recovery fact after all owned sessions closed."""
        if failure_kind not in {
            "mcp_server_shutdown",
            "genesis_live_socket_eof",
            "genesis_live_connection_lost",
        }:
            raise ValueError(f"unsupported diagnostic transport recovery kind: {failure_kind}")
        state = load_state(attachment.run_root)
        entry = self._attachment_entry(state, attachment.attachment_id)
        history = entry.get("history")
        if not isinstance(history, list):
            raise ValueError("diagnostic attachment history must be a list")
        existing = next(
            (
                item for item in history
                if isinstance(item, Mapping)
                and item.get("event") == "diagnostic_transport_recovery_ready"
                and item.get("failure_kind") == failure_kind
            ),
            None,
        )
        if isinstance(existing, Mapping):
            event_id = existing.get("recovery_event_id")
            if isinstance(event_id, str) and event_id:
                return event_id
            raise ValueError("existing transport recovery event lacks recovery_event_id")
        event_id = secrets.token_urlsafe(24)
        timestamp = self._now().isoformat()

        def mutation(entry: dict[str, Any], state: dict[str, Any]) -> None:
            del state
            entry["history"].append(
                {
                    "event": "diagnostic_transport_recovery_ready",
                    "timestamp": timestamp,
                    "failure_kind": failure_kind,
                    "recovery_event_id": event_id,
                    "message": message,
                }
            )

        self._mutate_ownership_ledgers(attachment, mutation)
        return event_id

    def _terminalize_transport_loss(
        self,
        attachment: DiagnosticAttachmentContext,
        *,
        failure_kind: str,
        cause: BaseException | str,
    ) -> None:
        """Close/revoke one dead endpoint and leave a typed fresh-attempt retry."""
        message = _format_error(cause)
        cleanup_message = self._cleanup_attachment(attachment, failure_kind=failure_kind)
        if cleanup_message is None:
            raise RuntimeError("transport-loss attachment was not terminalized")
        self._record_transport_recovery_ready(
            attachment,
            failure_kind=failure_kind,
            message=message,
        )
        self._record_operational_cleanup_failure(
            attachment.run_root,
            message,
            failure_kind=failure_kind,
            operational_status="cleanup_retryable",
            cleanup_status="complete",
        )

    def _record_cleanup_stage_failure(self, attachment: DiagnosticAttachmentContext, message: str) -> None:
        self._record_operational_cleanup_failure(
            attachment.run_root,
            message,
            failure_kind="owner_lease_expired",
            operational_status="lease_expired_retryable",
            cleanup_status="complete",
        )

    def _record_watchdog_failure(
        self,
        error: BaseException,
        *,
        attachment_id: str | None,
    ) -> None:
        with self._watchdog_failure_lock:
            self._watchdog_failure_events.append(
                {
                    "timestamp": self._now().isoformat(),
                    "attachment_id": attachment_id or "",
                    "error": _format_error(error),
                }
            )

    def watchdog_failures(self) -> tuple[dict[str, str], ...]:
        with self._watchdog_failure_lock:
            return tuple(dict(event) for event in self._watchdog_failure_events)

    def _pending_stage_message(self, attachment: DiagnosticAttachmentContext) -> str | None:
        with self._watchdog_failure_lock:
            return self._pending_stage_failures.get(attachment.attachment_id)

    def _set_pending_stage_message(
        self,
        attachment: DiagnosticAttachmentContext,
        message: str,
    ) -> None:
        with self._watchdog_failure_lock:
            self._pending_stage_failures[attachment.attachment_id] = message

    def _clear_pending_stage_message(self, attachment: DiagnosticAttachmentContext) -> None:
        with self._watchdog_failure_lock:
            self._pending_stage_failures.pop(attachment.attachment_id, None)

    def expire_due_attachments_once(self) -> tuple[str, ...]:
        now = self.monotonic_clock()
        with self._attachment_map_lock:
            candidates = tuple(self._current_attachment_by_run.values())
        expired: list[str] = []
        for attachment in candidates:
            try:
                with attachment.run_lock:
                    message = self._pending_stage_message(attachment)
                    with attachment.lock:
                        if (
                            attachment.status == "attached"
                            and attachment.watchdog_registered
                            and attachment.in_flight_count == 0
                            and now >= attachment.monotonic_deadline
                        ):
                            message = self._cleanup_attachment(
                                attachment,
                                failure_kind="owner_lease_expired",
                            )
                            if message is not None:
                                self._set_pending_stage_message(attachment, message)
                                expired.append(attachment.attachment_id)
                    if message is not None:
                        self._record_cleanup_stage_failure(attachment, message)
                        self._clear_pending_stage_message(attachment)
            except BaseException as error:
                self._record_watchdog_failure(
                    error,
                    attachment_id=attachment.attachment_id,
                )
        return tuple(expired)

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_POLL_INTERVAL_S):
            try:
                self.expire_due_attachments_once()
            except BaseException as error:
                self._record_watchdog_failure(error, attachment_id=None)

    def start_watchdog(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                raise RuntimeError("diagnostic MCP runtime is shut down")
            if self._watchdog_thread is not None:
                return
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="hag4r-diagnostic-owner-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._watchdog_stop.set()
            watchdog = self._watchdog_thread
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=2.0)
        with self._attachment_map_lock:
            attachments = tuple(self._current_attachment_by_run.values())
        first_error: BaseException | None = None
        for attachment in attachments:
            try:
                with attachment.run_lock:
                    message = self._pending_stage_message(attachment)
                    with attachment.lock:
                        if message is None:
                            message = self._cleanup_attachment(
                                attachment,
                                failure_kind="mcp_server_shutdown",
                            )
                            if message is not None:
                                self._set_pending_stage_message(attachment, message)
                        if message is not None:
                            self._record_transport_recovery_ready(
                                attachment,
                                failure_kind="mcp_server_shutdown",
                                message=message,
                            )
                            self._record_operational_cleanup_failure(
                                attachment.run_root,
                                message,
                                failure_kind="mcp_server_shutdown",
                                operational_status="cleanup_retryable",
                                cleanup_status="complete",
                            )
                            self._clear_pending_stage_message(attachment)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(f"additional MCP shutdown failure: {error}")
        if first_error is not None:
            raise first_error
        with self._shutdown_lock:
            self._shutdown_complete = True


_RESERVED_DIAGNOSTIC_ARGUMENTS = frozenset(
    {"run_root", "agent_invocation_id", "owner_lease_token", "live_session_handle"}
)


def _authenticated_diagnostic_signature(tool: Callable[..., Any]) -> inspect.Signature:
    source = inspect.signature(tool, eval_str=True)
    parameters = []
    for name, annotation in (
        ("run_root", str),
        ("agent_invocation_id", str),
        ("owner_lease_token", str),
    ):
        parameters.append(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation)
        )
    if tool.__name__ in LIVE_SESSION_ARGUMENT_TOOL_NAMES:
        parameters.append(
            inspect.Parameter(
                "live_session_handle",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
            )
        )
    for parameter in source.parameters.values():
        if parameter.name == "run_root":
            continue
        if parameter.name in _RESERVED_DIAGNOSTIC_ARGUMENTS:
            raise TypeError(f"diagnostic tool uses reserved wrapper parameter: {parameter.name}")
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(f"diagnostic tool has unsupported parameter kind: {parameter}")
        parameters.append(
            parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        )
    return source.replace(parameters=parameters, return_annotation=list[TextContent | ImageContent])


def _build_authenticated_diagnostic_wrapper(
    runtime: DiagnosticMcpRuntime,
    tool: Callable[..., dict[str, Any]],
) -> Callable[..., list[TextContent | ImageContent]]:
    signature = _authenticated_diagnostic_signature(tool)

    def wrapper(**kwargs: Any) -> list[TextContent | ImageContent]:
        bound = signature.bind(**kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        run_root = values.pop("run_root")
        agent_invocation_id = values.pop("agent_invocation_id")
        owner_lease_token = values.pop("owner_lease_token")
        live_session_handle = values.pop("live_session_handle", None)
        return runtime.dispatch_diagnostic_tool(
            tool=tool,
            run_root=run_root,
            agent_invocation_id=agent_invocation_id,
            owner_lease_token=owner_lease_token,
            live_session_handle=live_session_handle,
            arguments=values,
        )

    wrapper.__name__ = tool.__name__
    wrapper.__doc__ = tool.__doc__
    wrapper.__module__ = tool.__module__
    wrapper.__signature__ = signature  # type: ignore[attr-defined]
    return wrapper


def build_mcp_server(runtime: DiagnosticMcpRuntime) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> Any:
        del server
        runtime.start_watchdog()
        try:
            yield runtime
        finally:
            runtime.shutdown()

    mcp = FastMCP("hag4r-genesis-diagnostics", lifespan=lifespan)

    mcp.tool(name="attach_diagnostic_run")(runtime.attach_diagnostic_run)
    mcp.tool(name="renew_diagnostic_owner_lease")(runtime.renew_diagnostic_owner_lease)
    mcp.tool(name="create_genesis_live_session")(runtime.create_genesis_live_session)
    mcp.tool(name="bind_genesis_live_handlers")(runtime.bind_genesis_live_handlers)
    mcp.tool(name="get_genesis_live_session_status")(runtime.get_genesis_live_session_status)
    mcp.tool(name="close_genesis_live_session")(runtime.close_genesis_live_session)
    if DETERMINISTIC_MCP_TOOL_NAMES & LIFECYCLE_MCP_TOOL_NAMES:
        raise RuntimeError("diagnostic deterministic and lifecycle MCP tool names overlap")
    registered_deterministic_names: set[str] = set()
    for tool in DIAGNOSTIC_TOOLS:
        wrapper = _build_authenticated_diagnostic_wrapper(runtime, tool)
        mcp.tool(name=tool.__name__, structured_output=False)(wrapper)
        registered_deterministic_names.add(tool.__name__)
    if registered_deterministic_names != {tool.__name__ for tool in DIAGNOSTIC_TOOLS}:
        raise RuntimeError("registered diagnostic MCP names do not match DIAGNOSTIC_TOOLS")
    return mcp


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HAG4R Genesis diagnostic lifecycle MCP server")
    parser.add_argument(
        "--owner-lease-timeout-s",
        type=_positive_int,
        default=DEFAULT_OWNER_LEASE_TIMEOUT_S,
        help="server-wide diagnostic owner lease timeout in seconds (default: 3900)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_argument_parser().parse_args(argv)
    runtime = DiagnosticMcpRuntime(owner_lease_timeout_s=arguments.owner_lease_timeout_s)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _shutdown_on_sigterm(signum: int, frame: Any) -> None:
        del frame
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _shutdown_on_sigterm)
    try:
        build_mcp_server(runtime).run(transport="stdio")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        runtime.shutdown()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_EPISODE_TIMEOUT_S",
    "DEFAULT_OWNER_LEASE_TIMEOUT_S",
    "DETERMINISTIC_MCP_TOOL_NAMES",
    "DiagnosticAttachmentContext",
    "DiagnosticMcpRuntime",
    "LIFECYCLE_MCP_TOOL_NAMES",
    "LIVE_SESSION_ARGUMENT_TOOL_NAMES",
    "MAX_MODEL_VISIBLE_IMAGE_BYTES",
    "MAX_MODEL_VISIBLE_PREVIEW_IMAGES",
    "MAX_MODEL_VISIBLE_SIMULATE_IMAGES",
    "OWNER_KIND",
    "OWNER_LEASE_GRACE_S",
    "VISUAL_RESULT_TOOL_NAMES",
    "_build_authenticated_diagnostic_wrapper",
    "build_argument_parser",
    "build_mcp_server",
    "main",
]
