from __future__ import annotations

import json
import hashlib
import math
import os
import shlex
import socket
import stat
import subprocess
import tempfile
import time
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hag4r.tools.common import _conda_run_argv
from hag4r.tools.genesis.config import DEFAULT_GENESIS_ROOT
from hag4r.tools.genesis.diagnostic_timing import (
    ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS,
    ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS,
    DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
    adaptive_compiled_probe_resume_steps,
)
from hag4r.tools.genesis.live_protocol import (
    DEFAULT_CLIENT_LEASE_TIMEOUT_MS,
    DEFAULT_HEARTBEAT_MS,
    DEFAULT_READY_TIMEOUT_S,
    GenesisLiveProtocolError,
    PROTOCOL_NAME,
    STATUS_ERROR,
    STATUS_OK,
    recv_json,
    send_json,
)
from hag4r.tools.triple_view_evidence import TRIPLE_VIEW_PANEL_ORDER

SERVER_TO_HAG4R_VIEW = {
    "top": "top",
    "northeast": "ne_3q",
    "southwest": "sw_3q",
}
HAG4R_TO_SERVER_VIEW = {value: key for key, value in SERVER_TO_HAG4R_VIEW.items()}
DEFAULT_GENESIS_LIVE_PREFLIGHT_TIMEOUT_S = 180
COMMON_GENESIS_REQUIRED_CAPABILITIES = (
    "part_segmentation_triptych_telemetry",
)
SURFACE_GENESIS_REQUIRED_CAPABILITIES = (
    "surface_mesh_import",
    "heterogeneous_surface_material_arrays",
    "surface_shell_diagnostics",
    "surface_static_box_anchors",
    "surface_live_box_controller_actions",
) + COMMON_GENESIS_REQUIRED_CAPABILITIES
ADAPTIVE_COMPILED_PROBE_RESUME_SCHEMA_VERSION = "hag4r-genesis-compiled-probe-resume-adaptation-v1"
GENESIS_RUNTIME_LOG_SCHEMA_VERSION = "hag4r-genesis-runtime-logs-v1"
GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM = 20_000
GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM = 500
GENESIS_RUNTIME_LOG_MAX_LINE_CHARS = 2_048
GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS = 12_000
GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS = 512
GENESIS_RUNTIME_LOG_STREAMS = ("stdout", "stderr")
GENESIS_LIVE_ASSET_ROOT_ENV = "GENESIS_LIVE_ASSET_ROOT"
GENESIS_LIVE_LAUNCH_IDENTITY_KEYS = frozenset(
    {
        "live_session_handle",
        "episode_id",
        "agent_invocation_id",
        "owner_kind",
    }
)


@dataclass(frozen=True)
class ProbeEndpoint:
    simulation_step: int
    vector_env_local_m: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        vector = list(self.vector_env_local_m)
        return {
            "simulation_step": self.simulation_step,
            "vector_env_local_m": vector,
            "magnitude_m": math.sqrt(sum(value * value for value in vector)),
        }


@dataclass(frozen=True)
class CompletedProbeMeasurement:
    dispatch_token: str
    identity: dict[str, Any]
    under_load: ProbeEndpoint
    post_release: ProbeEndpoint
    schedule: dict[str, Any] | None = None
    controller_policy: dict[str, Any] | None = None
    controller_telemetry: dict[str, Any] | None = None
    force_summary: dict[str, Any] | None = None


def _validated_launch_identity(identity: dict[str, str]) -> dict[str, str]:
    unknown_keys = set(identity) - GENESIS_LIVE_LAUNCH_IDENTITY_KEYS
    if unknown_keys:
        raise ValueError(
            "launch_identity contains unsupported keys: "
            + ", ".join(sorted(unknown_keys))
        )
    for key, value in identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"launch_identity.{key} must be a non-empty string")
    return dict(identity)


def _legacy_identity_terms() -> tuple[str, ...]:
    return (
        "real" + "sim-live-v1",
        "hag4r-" + "real" + "sim-live-tool-v1",
        "Real" + "SimLiveServer",
    )


def _reject_legacy_identity(payload: Any, *, context: str) -> None:
    text = json.dumps(payload, sort_keys=True) if isinstance(payload, (dict, list)) else str(payload)
    lowered = text.lower()
    for term in _legacy_identity_terms():
        if term.lower() in lowered:
            raise GenesisLiveProtocolError(f"{context} contains unsupported legacy simulator identity")


class GenesisLiveHandshakeError(GenesisLiveProtocolError):
    def __init__(self, response: dict[str, Any]):
        error = response.get("error") if isinstance(response.get("error"), dict) else {}
        details = error.get("details") if isinstance(error.get("details"), dict) else None
        super().__init__(
            str(error.get("message") or error or "Genesis live handshake failed"),
            code=str(error.get("code") or "genesis_live_handshake_error"),
            details=details,
            response=response,
        )


def _resolve_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _read_ready_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenesisLiveProtocolError(f"Genesis ready file must contain a JSON object: {path}")
    _reject_legacy_identity(payload, context="Genesis ready file")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise GenesisLiveProtocolError(f"Genesis ready file has unsupported protocol: {payload.get('protocol')}")
    return payload


def _reported_capabilities(payload: dict[str, Any]) -> tuple[str, ...] | None:
    for key in ("capabilities", "supported_capabilities", "server_capabilities"):
        value = payload.get(key)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    data = payload.get("data")
    if isinstance(data, dict):
        return _reported_capabilities(data)
    return None


def _require_capabilities(
    payload: dict[str, Any],
    *,
    required_capabilities: tuple[str, ...],
    context: str,
) -> None:
    if not required_capabilities:
        return
    reported = _reported_capabilities(payload)
    if reported is None:
        raise GenesisLiveProtocolError(
            f"{context} did not report Genesis live capabilities",
            code="genesis_live_missing_capabilities",
            details={"required_capabilities": list(required_capabilities)},
        )
    missing = sorted(set(required_capabilities) - set(reported))
    if missing:
        raise GenesisLiveProtocolError(
            f"{context} is missing required Genesis live capabilities: {', '.join(missing)}",
            code="genesis_live_missing_capabilities",
            details={
                "required_capabilities": list(required_capabilities),
                "reported_capabilities": list(reported),
                "missing_capabilities": missing,
            },
        )


def _canonical_fixed_rgb_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"mode", "render_every_steps", "views"}:
        raise GenesisLiveProtocolError("fixed_rgb_views request must contain exactly mode, render_every_steps, and views")
    if value.get("mode") != "fixed_rgb_views":
        raise GenesisLiveProtocolError("fixed_rgb_views request mode is invalid")
    interval = value.get("render_every_steps")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval != 10:
        raise GenesisLiveProtocolError("fixed_rgb_views render_every_steps must be 10")
    views = value.get("views")
    if not isinstance(views, list) or len(views) != 2:
        raise GenesisLiveProtocolError("fixed_rgb_views requires exactly two views")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(views):
        if not isinstance(raw, dict) or set(raw) != {"name", "position", "look_at", "up", "resolution", "fov_degrees"}:
            raise GenesisLiveProtocolError(f"fixed_rgb_views.views[{index}] has unexpected or missing fields")
        name = raw.get("name")
        if not isinstance(name, str) or name not in {"full", "context"} or name in names:
            raise GenesisLiveProtocolError("fixed_rgb_views view names must be unique full/context")
        names.add(name)
        resolution = raw.get("resolution")
        if resolution != [512, 512] or not isinstance(resolution, list):
            raise GenesisLiveProtocolError("fixed_rgb_views resolution must be [512, 512]")
        fov = raw.get("fov_degrees")
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not math.isfinite(float(fov)) or float(fov) != 40.0:
            raise GenesisLiveProtocolError("fixed_rgb_views fov_degrees must be 40.0")

        def vec3(field_name: str) -> list[float]:
            vector = raw.get(field_name)
            if not isinstance(vector, list) or len(vector) != 3:
                raise GenesisLiveProtocolError(f"fixed_rgb_views {name}.{field_name} must be a finite vec3")
            if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in vector):
                raise GenesisLiveProtocolError(f"fixed_rgb_views {name}.{field_name} must be a finite vec3")
            try:
                result = [struct.unpack("!f", struct.pack("!f", float(component)))[0] for component in vector]
            except (OverflowError, ValueError) as error:
                raise GenesisLiveProtocolError(
                    f"fixed_rgb_views {name}.{field_name} must be a finite vec3"
                ) from error
            if not all(math.isfinite(component) for component in result):
                raise GenesisLiveProtocolError(f"fixed_rgb_views {name}.{field_name} must be a finite vec3")
            return result

        position = vec3("position")
        look_at = vec3("look_at")
        up = vec3("up")
        direction = [look_at[i] - position[i] for i in range(3)]
        if math.sqrt(sum(component * component for component in direction)) <= 1.0e-8:
            raise GenesisLiveProtocolError(f"fixed_rgb_views {name} position and look_at are degenerate")
        if math.sqrt(sum(component * component for component in up)) <= 1.0e-8:
            raise GenesisLiveProtocolError(f"fixed_rgb_views {name}.up is degenerate")
        cross = [direction[1] * up[2] - direction[2] * up[1], direction[2] * up[0] - direction[0] * up[2], direction[0] * up[1] - direction[1] * up[0]]
        if math.sqrt(sum(component * component for component in cross)) <= 1.0e-8:
            raise GenesisLiveProtocolError(f"fixed_rgb_views {name}.up is collinear with the view direction")
        normalized.append({"name": name, "position": position, "look_at": look_at, "up": up, "resolution": [512, 512], "fov_degrees": 40.0})
    if names != {"full", "context"}:
        raise GenesisLiveProtocolError("fixed_rgb_views requires one full and one context view")
    normalized.sort(key=lambda item: ("full", "context").index(item["name"]))
    return {"mode": "fixed_rgb_views", "render_every_steps": int(interval), "views": normalized}


def _fixed_rgb_request_hash(value: Any) -> str:
    canonical = _canonical_fixed_rgb_request(value)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _fixed_rgb_up_matches_observed(value: Any, expected: Any) -> bool:
    """Compare Genesis's normalized up vector to its float32 request.

    Genesis normalizes camera basis vectors internally, so the observed
    ``camera.up`` can differ by a few float32 ulps from the bound request.
    The request-owned ``camera.pose`` remains exact; this tolerance applies
    only to the separately reported effective camera vectors.
    """

    if not isinstance(value, list) or not isinstance(expected, list) or len(value) != 3 or len(expected) != 3:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value + expected):
        return False
    return all(
        math.isfinite(float(observed))
        and math.isclose(float(observed), float(requested), rel_tol=0.0, abs_tol=2.0e-6)
        for observed, requested in zip(value, expected, strict=True)
    )


def _compiled_probe_motion_estimate_payload(applied_action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(applied_action, dict):
        return {
            "effective_steps": None,
            "estimated_motion_steps": None,
            "source": "probe.apply.probe.controller_state.estimated_motion_steps",
            "reason": "missing_probe_apply_result",
        }
    probe = applied_action.get("probe")
    source = "probe.apply.probe.controller_state.estimated_motion_steps"
    if isinstance(probe, dict):
        probe_status = str(probe.get("status", "")).strip()
        if probe_status in {"rejected", "error", "failed"} or isinstance(probe.get("error"), dict):
            return {
                "effective_steps": None,
                "estimated_motion_steps": None,
                "source": source,
                "reason": "probe_apply_status_not_applied",
                "probe_status": probe_status or None,
            }
        controller_state = probe.get("controller_state")
    else:
        source = "probe.apply.controller_state.estimated_motion_steps"
        probe_status = str(applied_action.get("status", "")).strip()
        if probe_status != "applied":
            return {
                "effective_steps": None,
                "estimated_motion_steps": None,
                "source": source,
                "reason": "probe_apply_status_not_applied",
                "probe_status": probe_status or None,
            }
        controller_state = applied_action.get("controller_state")
    if not isinstance(controller_state, dict):
        return {
            "effective_steps": None,
            "estimated_motion_steps": None,
            "source": source,
            "reason": "missing_controller_state_estimated_motion_steps",
        }
    value = controller_state.get("estimated_motion_steps")
    if value is None:
        return {
            "effective_steps": None,
            "estimated_motion_steps": None,
            "source": source,
            "reason": "missing_controller_state_estimated_motion_steps",
        }
    return {
        "effective_steps": adaptive_compiled_probe_resume_steps(value),
        "estimated_motion_steps": int(value),
        "source": source,
        "reason": None,
    }


@dataclass
class GenesisLiveApiSession:
    scene_config_path: Path | None = None
    genesis_root: Path | None = DEFAULT_GENESIS_ROOT
    genesis_env_path: Path | None = None
    genesis_live_command: str = "python -m genesis.live.server"
    host: str = "127.0.0.1"
    port: int = 0
    ready_file_path: Path | None = None
    output_dir: Path | None = None
    log_dir: Path | None = None
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S
    heartbeat_ms: int = DEFAULT_HEARTBEAT_MS
    client_lease_timeout_ms: int = DEFAULT_CLIENT_LEASE_TIMEOUT_MS
    start_paused: bool = True
    required_capabilities: tuple[str, ...] = ()
    asset_root: Path | None = None
    process_env_overrides: dict[str, str] = field(default_factory=dict)
    launch_evidence_path: Path | None = None
    launch_identity: dict[str, str] = field(default_factory=dict)
    process: subprocess.Popen[str] | None = field(default=None, init=False)
    stdout_log_path: Path | None = field(default=None, init=False)
    stderr_log_path: Path | None = field(default=None, init=False)
    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _ready_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _handshake_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _launched_once: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _fixed_rgb_visual_request: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _fixed_rgb_visual_request_hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.scene_config_path = _resolve_optional_path(self.scene_config_path)
        self.genesis_root = _resolve_optional_path(self.genesis_root)
        self.genesis_env_path = _resolve_optional_path(self.genesis_env_path)
        self.ready_file_path = _resolve_optional_path(self.ready_file_path)
        self.output_dir = _resolve_optional_path(self.output_dir)
        self.log_dir = _resolve_optional_path(self.log_dir)
        self.launch_evidence_path = _resolve_optional_path(self.launch_evidence_path)
        if self.asset_root is not None:
            asset_root = Path(self.asset_root).expanduser()
            if not asset_root.is_absolute():
                raise ValueError("asset_root must be an absolute path")
            asset_root = asset_root.resolve()
            if not asset_root.exists() or not asset_root.is_dir():
                raise ValueError("asset_root must identify an existing directory")
            self.asset_root = asset_root
        if GENESIS_LIVE_ASSET_ROOT_ENV in self.process_env_overrides:
            raise ValueError(
                f"process_env_overrides may not define {GENESIS_LIVE_ASSET_ROOT_ENV}; "
                "use asset_root instead"
            )
        self.launch_identity = _validated_launch_identity(self.launch_identity)
        self.required_capabilities = tuple(str(capability) for capability in self.required_capabilities)
        if not str(self.genesis_live_command).strip():
            raise ValueError("genesis_live_command must be a non-empty command")

    def _resolved_ready_file(self) -> Path:
        if self.ready_file_path is not None:
            return self.ready_file_path
        root = self.output_dir or self.log_dir or Path(tempfile.gettempdir()) / "hag4r_genesis_live"
        return root / "live_ready.json"

    def _resolved_log_dir(self, ready_file: Path) -> Path:
        return self.log_dir or ready_file.parent / "logs"

    def _resolved_output_dir(self, ready_file: Path) -> Path:
        return self.output_dir or ready_file.parent / "outputs"

    def _resolved_cwd(self) -> Path:
        if self.genesis_root is not None:
            return self.genesis_root
        if self.scene_config_path is not None:
            return self.scene_config_path.parent
        return Path.cwd()

    def _server_command(self, ready_file: Path, output_dir: Path) -> tuple[str, ...]:
        command = tuple(shlex.split(self.genesis_live_command))
        if not command:
            raise ValueError("genesis_live_command must include an executable")
        argv = (
            *command,
            "--host",
            self.host,
            "--port",
            str(int(self.port)),
            "--ready-file",
            str(ready_file),
            "--output-dir",
            str(output_dir),
            "--heartbeat-interval-s",
            f"{float(self.heartbeat_ms) / 1000.0:.6g}",
        )
        if self.scene_config_path is not None:
            argv = (*argv, "--scene-config", str(self.scene_config_path))
        if self.start_paused:
            argv = (*argv, "--start-paused")
        else:
            argv = (*argv, "--no-start-paused")
        return argv

    def _build_launch_argv(self, ready_file: Path, output_dir: Path) -> tuple[str, ...]:
        command = self._server_command(ready_file, output_dir)
        if self.genesis_env_path is None:
            return command
        conda_argv = _conda_run_argv(str(self.genesis_env_path), command, repo_root=self.genesis_root)
        return (*conda_argv[:2], "--no-capture-output", *conda_argv[2:])

    def _selected_env(self, process_env: dict[str, str]) -> dict[str, str | None]:
        return {
            "CUDA_VISIBLE_DEVICES": process_env.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONUNBUFFERED": process_env.get("PYTHONUNBUFFERED"),
            "PYTHONPATH": process_env.get("PYTHONPATH"),
        }

    def _build_process_env(self) -> dict[str, str]:
        process_env = os.environ.copy()
        process_env.update({key: str(value) for key, value in self.process_env_overrides.items()})
        process_env["PYTHONUNBUFFERED"] = "1"
        if self.asset_root is not None:
            process_env[GENESIS_LIVE_ASSET_ROOT_ENV] = str(self.asset_root)
        return process_env

    def _write_launch_evidence(
        self,
        *,
        evidence_path: Path,
        status: str,
        argv: tuple[str, ...],
        ready_file: Path,
        output_dir: Path,
        log_dir: Path,
        env: dict[str, str | None],
        pid: int | None = None,
    ) -> None:
        launch_identity = _validated_launch_identity(self.launch_identity)
        payload = {
            "schema_version": "hag4r-genesis-live-launch-v1",
            "status": status,
            "launch_mode": "conda" if self.genesis_env_path is not None else "direct",
            "launch_argv": list(argv),
            "cwd": str(self._resolved_cwd()),
            "genesis_root": str(self.genesis_root) if self.genesis_root is not None else None,
            "genesis_env_path": str(self.genesis_env_path) if self.genesis_env_path is not None else None,
            "genesis_live_command": self.genesis_live_command,
            "scene_config_path": str(self.scene_config_path) if self.scene_config_path is not None else None,
            "ready_file_path": str(ready_file),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "stdout_log_path": str(self.stdout_log_path) if self.stdout_log_path is not None else None,
            "stderr_log_path": str(self.stderr_log_path) if self.stderr_log_path is not None else None,
            "required_capabilities": list(self.required_capabilities),
            "env": env,
            "pid": pid,
            **launch_identity,
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _start_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None and self._launched_once:
            raise RuntimeError(f"server_exited: Genesis live server exited with code {self.process.returncode}")
        if self.scene_config_path is None:
            return

        ready_file = self._resolved_ready_file()
        output_dir = self._resolved_output_dir(ready_file)
        log_dir = self._resolved_log_dir(ready_file)
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        ready_file.unlink(missing_ok=True)
        self.stdout_log_path = log_dir / "genesis_live_stdout.log"
        self.stderr_log_path = log_dir / "genesis_live_stderr.log"
        argv = self._build_launch_argv(ready_file, output_dir)
        process_env = self._build_process_env()
        selected_env = self._selected_env(process_env)
        launch_evidence_path = self.launch_evidence_path or log_dir / "genesis_live_launch.json"
        self._write_launch_evidence(
            evidence_path=launch_evidence_path,
            status="prepared",
            argv=argv,
            ready_file=ready_file,
            output_dir=output_dir,
            log_dir=log_dir,
            env=selected_env,
        )
        stdout_handle = self.stdout_log_path.open("w", encoding="utf-8")
        stderr_handle = self.stderr_log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=self._resolved_cwd(),
                env=process_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        self._launched_once = True
        self._write_launch_evidence(
            evidence_path=launch_evidence_path,
            status="running",
            argv=argv,
            ready_file=ready_file,
            output_dir=output_dir,
            log_dir=log_dir,
            env=selected_env,
            pid=self.process.pid if self.process is not None else None,
        )

    def _wait_ready(self) -> dict[str, Any]:
        ready_file = self._resolved_ready_file()
        deadline = time.monotonic() + float(self.ready_timeout_s)
        while time.monotonic() < deadline:
            if ready_file.exists():
                return _read_ready_file(ready_file)
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Genesis live server exited before ready file: {self.process.returncode}")
            time.sleep(0.1)
        raise TimeoutError(f"Genesis live server did not write ready file within {self.ready_timeout_s:g}s: {ready_file}")

    def _ensure_socket(self) -> socket.socket:
        if self._closed:
            raise RuntimeError("Genesis live session is closed")
        if self._sock is not None:
            return self._sock
        self._start_process()
        ready = self._wait_ready() if self.scene_config_path is not None else {}
        if self.scene_config_path is not None:
            _require_capabilities(
                ready,
                required_capabilities=self.required_capabilities,
                context="Genesis ready file",
            )
            self._ready_result = dict(ready)
        host = str(ready.get("host", self.host))
        port = int(ready.get("port", self.port))
        sock = socket.create_connection((host, port), timeout=float(self.ready_timeout_s))
        sock.settimeout(None)
        self._sock = sock
        try:
            handshake = self._request("session.handshake", {})
            if handshake.get("protocol") != PROTOCOL_NAME:
                raise GenesisLiveHandshakeError(handshake)
            _require_capabilities(
                handshake,
                required_capabilities=self.required_capabilities,
                context="Genesis live handshake",
            )
            self._handshake_result = dict(handshake)
        except BaseException as error:
            self._sock = None
            self._ready_result = None
            self._handshake_result = None
            try:
                sock.close()
            except BaseException as cleanup_error:
                error.add_note(f"Failed to close rejected Genesis live socket: {cleanup_error}")
            raise
        return sock

    def _request_on_existing_socket(
        self,
        sock: socket.socket,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        send_json(sock, {"request_id": f"hag4r_{time.time_ns()}", "method": method, "params": params or {}})
        response = recv_json(sock)
        _reject_legacy_identity(response, context="Genesis live response")
        if response.get("protocol") != PROTOCOL_NAME:
            raise GenesisLiveProtocolError(
                f"unexpected Genesis live protocol: {response.get('protocol')}",
                response=response,
            )
        if response.get("status") != STATUS_OK:
            error = response.get("error")
            if isinstance(error, dict):
                details = error.get("details") if isinstance(error.get("details"), dict) else None
                raise GenesisLiveProtocolError(
                    str(error.get("message") or error),
                    code=str(error.get("code") or "genesis_live_protocol_error"),
                    details=details,
                    response=response,
                )
            raise GenesisLiveProtocolError(str(error or response), response=response)
        data = response.get("data")
        if not isinstance(data, dict):
            raise GenesisLiveProtocolError("Genesis live response data must be a JSON object", response=response)
        return data

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Genesis live session is closed")
        sock = self._sock
        if sock is None:
            if method == "session.handshake":
                raise RuntimeError("Genesis live socket is not connected")
            sock = self._ensure_socket()
        return self._request_on_existing_socket(sock, method, params)

    def connect(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Genesis live session is closed")
        self._ensure_socket()
        return {"tool": "session.connect", "status": STATUS_OK, "result": self.status()}

    def handshake_identity(self) -> dict[str, Any]:
        """Return the exact handshake retained when this socket connected."""
        if self._sock is None or self._handshake_result is None:
            raise RuntimeError("Genesis live session has no connected handshake identity")
        return dict(self._handshake_result)

    def ready_identity(self) -> dict[str, Any]:
        """Return the exact ready evidence accepted before this socket connected."""
        if self._sock is None or self._ready_result is None:
            raise RuntimeError("Genesis live session has no connected ready identity")
        return dict(self._ready_result)

    def detach_existing_endpoint(self) -> None:
        """Close only this client socket without sending session.close."""
        if self.process is not None or self.scene_config_path is not None:
            raise RuntimeError("local detach is only valid for a non-owning existing-endpoint client")
        sock = self._sock
        self._sock = None
        self._ready_result = None
        self._handshake_result = None
        self._closed = True
        if sock is not None:
            sock.close()

    def status(self) -> dict[str, Any]:
        return self._request("command.status", {})

    def close(self, timeout_ms: int = 3_000) -> dict[str, Any]:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if self._close_result is not None:
            return self._close_result
        if self._closed:
            raise RuntimeError("Genesis live session close previously failed")

        timeout_s = timeout_ms / 1_000.0
        self._closed = True
        sock = self._sock
        self._sock = None
        self._ready_result = None
        self._handshake_result = None
        request_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        result: dict[str, Any] = {"closed": True, "already_inactive": sock is None and self.process is None}

        def record_cleanup_error(error: BaseException) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(f"Additional Genesis live cleanup failure: {error}")

        if sock is not None:
            try:
                sock.settimeout(timeout_s)
                result = self._request_on_existing_socket(sock, "session.close", {})
            except BaseException as error:
                request_error = error
            try:
                sock.close()
            except BaseException as error:
                record_cleanup_error(error)

        process = self.process
        process_is_live = False
        if process is not None:
            try:
                process_is_live = process.poll() is None
            except BaseException as error:
                record_cleanup_error(error)

        if process_is_live:
            terminated = False
            if sock is None or request_error is not None:
                try:
                    process.terminate()
                    terminated = True
                except BaseException as error:
                    record_cleanup_error(error)
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if not terminated:
                    try:
                        process.terminate()
                        terminated = True
                    except BaseException as terminate_error:
                        record_cleanup_error(terminate_error)
                try:
                    process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except BaseException as kill_error:
                        record_cleanup_error(kill_error)
                    try:
                        process.wait(timeout=timeout_s)
                    except BaseException as wait_error:
                        record_cleanup_error(wait_error)
                except BaseException as wait_error:
                    record_cleanup_error(wait_error)
            except BaseException as error:
                record_cleanup_error(error)

        if request_error is not None:
            if cleanup_error is not None:
                request_error.add_note(f"Genesis live local cleanup also failed: {cleanup_error}")
                raise request_error from cleanup_error
            raise request_error
        if cleanup_error is not None:
            raise cleanup_error

        self._close_result = {"tool": "session.close", "status": STATUS_OK, "result": result}
        return self._close_result

    def _open_runtime_log(self, *, stream_name: str):
        if self._closed:
            raise RuntimeError("Genesis live session is closed")
        if stream_name not in GENESIS_RUNTIME_LOG_STREAMS:
            raise ValueError(f"unsupported Genesis runtime log stream: {stream_name}")
        path = self.stdout_log_path if stream_name == "stdout" else self.stderr_log_path
        if path is None:
            raise RuntimeError(f"Genesis live {stream_name} log path is not initialized")
        if self.log_dir is None:
            raise RuntimeError("Genesis live log directory is not initialized")
        expected_basename = f"genesis_live_{stream_name}.log"
        if path.name != expected_basename:
            raise ValueError(f"Genesis live {stream_name} log has an unexpected basename")
        expected_parent = self.log_dir.expanduser().resolve()
        if path.expanduser().resolve(strict=False).parent != expected_parent:
            raise ValueError(f"Genesis live {stream_name} log is outside the session log directory")
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Genesis live {stream_name} log must be a regular non-symlink file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            after = os.fstat(descriptor)
            if not stat.S_ISREG(after.st_mode):
                raise ValueError(f"Genesis live {stream_name} log must remain a regular file")
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise RuntimeError(f"Genesis live {stream_name} log changed while opening")
            return os.fdopen(descriptor, "rb"), int(after.st_size)
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _runtime_log_line(raw_line: bytes, *, line_number: int, query: str) -> tuple[dict[str, Any], bool]:
        decoded = raw_line.decode("utf-8", errors="replace").removesuffix("\n").removesuffix("\r")
        truncated = len(decoded) > GENESIS_RUNTIME_LOG_MAX_LINE_CHARS
        record = {
            "line_number": line_number,
            "text": decoded[:GENESIS_RUNTIME_LOG_MAX_LINE_CHARS],
            "truncated": truncated,
        }
        return record, bool(query and query.casefold() in decoded.casefold())

    def _read_runtime_log_stream(
        self,
        *,
        stream_name: str,
        requested_cursor: int,
        max_lines: int,
        contains: str,
        context_lines: int,
    ) -> dict[str, Any]:
        handle, snapshot_size = self._open_runtime_log(stream_name=stream_name)
        with handle:
            line_number = 0
            while line_number < requested_cursor - 1 and handle.tell() < snapshot_size:
                remaining = snapshot_size - handle.tell()
                if not handle.readline(remaining):
                    break
                line_number += 1

            actual_cursor = line_number + 1
            scanned: list[tuple[dict[str, Any], bool]] = []
            scan_limit = (
                GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM
                if contains
                else max_lines
            )
            while len(scanned) < scan_limit and handle.tell() < snapshot_size:
                remaining = snapshot_size - handle.tell()
                raw_line = handle.readline(remaining)
                if not raw_line:
                    break
                line_number += 1
                scanned.append(
                    self._runtime_log_line(raw_line, line_number=line_number, query=contains)
                )

            snapshot_remaining = handle.tell() < snapshot_size

        if contains:
            selected_indices: set[int] = set()
            for index, (_, matches) in enumerate(scanned):
                if matches:
                    selected_indices.update(
                        range(
                            max(0, index - context_lines),
                            min(len(scanned), index + context_lines + 1),
                        )
                    )
            qualifying = [scanned[index][0] for index in sorted(selected_indices)]
            lines = qualifying[:max_lines]
            output_truncated = len(qualifying) > max_lines
        else:
            lines = [record for record, _ in scanned]
            output_truncated = snapshot_remaining

        scan_end = line_number
        scan_truncated = (
            len(scanned) == GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM
            and snapshot_remaining
        )
        return {
            "requested_cursor": requested_cursor,
            "actual_cursor": actual_cursor,
            "scan_end": scan_end,
            "next_cursor": scan_end + 1,
            "eof": not snapshot_remaining,
            "scan_truncated": scan_truncated,
            "output_truncated": output_truncated,
            "returned_line_count": len(lines),
            "lines": lines,
        }

    @staticmethod
    def _runtime_log_model_payload_size(envelope: dict[str, Any]) -> int:
        compact = {
            "tool": "inspect_genesis_runtime_logs",
            "status": STATUS_OK,
            "tool_result_index": 9_999_999_999_999_999_999,
            "result": envelope,
        }
        return len(json.dumps(compact, sort_keys=True, separators=(",", ":")))

    @classmethod
    def _bound_runtime_log_envelope(cls, envelope: dict[str, Any]) -> dict[str, Any]:
        streams = envelope["streams"]
        for stream_name in reversed(GENESIS_RUNTIME_LOG_STREAMS):
            stream_result = streams.get(stream_name)
            if not isinstance(stream_result, dict):
                continue
            lines = stream_result["lines"]
            while lines and cls._runtime_log_model_payload_size(envelope) > GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS:
                lines.pop()
                stream_result["returned_line_count"] = len(lines)
                stream_result["output_truncated"] = True
        envelope["output_truncated"] = any(
            bool(result["output_truncated"])
            for result in streams.values()
        )
        if cls._runtime_log_model_payload_size(envelope) > GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS:
            raise RuntimeError("Genesis runtime log envelope metadata exceeds the model-visible character limit")
        return envelope

    def inspect_genesis_runtime_logs(
        self,
        *,
        stream: str = "both",
        cursors: dict[str, int] | None = None,
        max_lines: int = 120,
        contains: str = "",
        context_lines: int = 2,
    ) -> dict[str, Any]:
        if stream not in {"stdout", "stderr", "both"}:
            raise ValueError("stream must be stdout, stderr, or both")
        selected_streams = GENESIS_RUNTIME_LOG_STREAMS if stream == "both" else (stream,)
        cursor_values = {} if cursors is None else cursors
        if not isinstance(cursor_values, dict):
            raise TypeError("cursors must be null or an object")
        if set(cursor_values) - set(selected_streams):
            raise ValueError("cursors may contain only selected stream keys")
        for key, value in cursor_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"cursors.{key} must be a positive integer")
        if isinstance(max_lines, bool) or not isinstance(max_lines, int) or not 1 <= max_lines <= 500:
            raise ValueError("max_lines must be an integer in 1..500")
        if not isinstance(contains, str):
            raise TypeError("contains must be a string")
        if len(contains) > GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS:
            raise ValueError(f"contains must contain at most {GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS} characters")
        if (
            isinstance(context_lines, bool)
            or not isinstance(context_lines, int)
            or not 0 <= context_lines <= 5
        ):
            raise ValueError("context_lines must be an integer in 0..5")

        stream_results = {
            stream_name: self._read_runtime_log_stream(
                stream_name=stream_name,
                requested_cursor=cursor_values.get(stream_name, 1),
                max_lines=max_lines,
                contains=contains,
                context_lines=context_lines,
            )
            for stream_name in selected_streams
        }
        envelope = {
            "schema_version": GENESIS_RUNTIME_LOG_SCHEMA_VERSION,
            "stream": stream,
            "contains": contains,
            "context_lines": context_lines,
            "limits": {
                "max_scan_lines_per_stream": GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM,
                "max_returned_lines_per_stream": GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM,
                "max_line_chars": GENESIS_RUNTIME_LOG_MAX_LINE_CHARS,
                "max_model_visible_chars": GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS,
            },
            "output_truncated": any(result["output_truncated"] for result in stream_results.values()),
            "streams": stream_results,
        }
        return {
            "tool": "inspect_genesis_runtime_logs",
            "status": STATUS_OK,
            "result": self._bound_runtime_log_envelope(envelope),
        }

    def _normalize_visual_frame(
        self,
        visual: dict[str, Any],
        *,
        sequence_index: int,
        status: dict[str, Any],
    ) -> dict[str, Any] | None:
        frame_metadata = visual.get("frame_metadata") if isinstance(visual.get("frame_metadata"), list) else []
        views = visual.get("views") if isinstance(visual.get("views"), list) else []
        panel_records = views or [
            record
            for record in frame_metadata
            if isinstance(record, dict) and record.get("hag4r_label") in TRIPLE_VIEW_PANEL_ORDER
        ]
        paths_by_hag4r_view: dict[str, Path] = {}
        normalized_views: dict[str, dict[str, Any]] = {}
        for record in panel_records:
            if not isinstance(record, dict):
                continue
            server_label = str(record.get("label") or record.get("server_label") or "")
            hag4r_label = str(record.get("hag4r_label") or SERVER_TO_HAG4R_VIEW.get(server_label, ""))
            if hag4r_label not in TRIPLE_VIEW_PANEL_ORDER:
                continue
            source_path = str(record.get("path", "")).strip()
            if source_path:
                paths_by_hag4r_view[hag4r_label] = Path(source_path)
            normalized_views[hag4r_label] = {
                "source_png_path": source_path,
                "server_view_label": server_label,
                "label": server_label,
                "hag4r_label": hag4r_label,
                "width": record.get("width"),
                "height": record.get("height"),
                "byte_size": record.get("byte_size"),
                "sha256": record.get("sha256"),
                "frame_index": record.get("frame_index"),
                "simulation_step": record.get("simulation_step"),
                "camera": record.get("camera", {}) if isinstance(record.get("camera"), dict) else {},
            }
        stitched = visual.get("stitched") if isinstance(visual.get("stitched"), dict) else {}
        triptych_path = str(stitched.get("path", "")).strip()
        frame_id = int(stitched.get("frame_index", status.get("current_step", 0)) or 0)
        if set(paths_by_hag4r_view) == set(TRIPLE_VIEW_PANEL_ORDER) and triptych_path:
            return {
                "sequence_index": int(sequence_index),
                "frame_id": frame_id,
                "triptych_png_path": triptych_path,
                "views": normalized_views,
            }
        return None

    def _normalize_fixed_rgb_frame(
        self,
        visual: dict[str, Any],
        *,
        sequence_index: int,
        status: dict[str, Any],
    ) -> dict[str, Any] | None:
        records = visual.get("views")
        if not isinstance(records, list) or len(records) != 2:
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry must contain exactly two view records")
        if visual.get("view_order") != ["full", "context"]:
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry view order must be full, context")
        normalized_views: dict[str, dict[str, Any]] = {}
        frame_id: int | None = None
        expected_request = getattr(self, "_fixed_rgb_visual_request", None)
        expected_hash = getattr(self, "_fixed_rgb_visual_request_hash", None)
        actual_specs = visual.get("camera_specs")
        if expected_request is not None:
            if visual.get("camera_specs_hash") != expected_hash or actual_specs != expected_request["views"]:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera request/hash differs from the bound request")
        metadata_value = visual.get("metadata_path")
        if not isinstance(metadata_value, str) or not metadata_value.strip():
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry metadata file is missing")
        metadata_candidate = Path(metadata_value).expanduser()
        if metadata_candidate.is_symlink() or not metadata_candidate.is_file():
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry metadata file is missing or not regular")
        metadata_path = metadata_candidate.resolve()
        metadata_stat = metadata_path.stat()
        if not stat.S_ISREG(metadata_stat.st_mode):
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry metadata file is not regular")
        metadata_bytes = metadata_path.read_bytes()
        metadata_record = {
            "path": str(metadata_path),
            "byte_size": len(metadata_bytes),
            "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }
        for expected_label, record in zip(("full", "context"), records, strict=True):
            if not isinstance(record, dict) or record.get("label") != expected_label:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry labels/order are invalid")
            source_path = record.get("path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry view has no PNG path")
            width = record.get("width")
            height = record.get("height")
            if width != 512 or height != 512:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry PNG dimensions must be 512x512")
            byte_size = record.get("byte_size")
            digest = record.get("sha256")
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry byte_size is invalid")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry sha256 is invalid")
            raw_frame_id = record.get("frame_index")
            raw_step = record.get("simulation_step")
            if isinstance(raw_frame_id, bool) or not isinstance(raw_frame_id, int) or isinstance(raw_step, bool) or not isinstance(raw_step, int):
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry frame/step bindings are invalid")
            if raw_frame_id != raw_step:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry frame_index and simulation_step disagree")
            if frame_id is None:
                frame_id = raw_frame_id
            elif frame_id != raw_frame_id:
                raise GenesisLiveProtocolError("fixed_rgb_views frame records disagree on frame_index")
            renderer = record.get("renderer")
            camera = record.get("camera")
            if not isinstance(renderer, dict) or renderer.get("mode") != "fixed_rgb_views" or renderer.get("debug_camera") is not False:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry renderer metadata is not clean")
            if not isinstance(camera, dict) or camera.get("model") != "pinhole" or camera.get("res") != [512, 512]:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera metadata is invalid")
            camera_fov = camera.get("fov")
            if isinstance(camera_fov, bool) or not isinstance(camera_fov, (int, float)) or not math.isfinite(float(camera_fov)) or float(camera_fov) != 40.0:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera metadata is invalid")
            if camera.get("debug") is not False:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera must be non-debug")
            if not isinstance(record.get("camera_spec"), dict):
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry lacks the exact camera spec")
            expected_spec_index = 0 if expected_label == "full" else 1
            if expected_request is not None and record["camera_spec"] != expected_request["views"][expected_spec_index]:
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera spec differs from the bound request")
            spec = record["camera_spec"]
            pose = camera.get("pose")
            if (
                not isinstance(pose, dict)
                or pose.get("pos") != spec.get("position")
                or pose.get("lookat") != spec.get("look_at")
                or pose.get("up") != spec.get("up")
                or camera.get("pos") != spec.get("position")
                or camera.get("lookat") != spec.get("look_at")
                or not _fixed_rgb_up_matches_observed(camera.get("up"), spec.get("up"))
            ):
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera pose differs from the request")
            if record.get("camera_spec_hash") != visual.get("camera_specs_hash"):
                raise GenesisLiveProtocolError("fixed_rgb_views telemetry camera spec hash differs")
            normalized_views[expected_label] = {
                "source_png_path": source_path.strip(),
                "server_view_label": expected_label,
                "label": expected_label,
                "hag4r_label": expected_label,
                "width": width,
                "height": height,
                "byte_size": byte_size,
                "sha256": digest,
                "frame_index": raw_frame_id,
                "simulation_step": raw_step,
                "camera": camera,
                "camera_spec": record["camera_spec"],
                "camera_spec_hash": record["camera_spec_hash"],
                "renderer": renderer,
            }
        if visual.get("debug_markers") != [] or visual.get("overlays") != []:
            raise GenesisLiveProtocolError("fixed_rgb_views telemetry contains overlays or debug markers")
        return {
            "sequence_index": int(sequence_index),
            "frame_id": int(frame_id if frame_id is not None else status.get("current_step", 0)),
            "metadata": metadata_record,
            "metadata_path": metadata_record["path"],
            "views": normalized_views,
        }

    def _normalize_visual_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        visual = payload.get("visual_telemetry") if isinstance(payload.get("visual_telemetry"), dict) else {}
        if not visual or visual.get("requested") is False:
            return {
                "frame_ids": [],
                "count": 0,
                "part_segmentation_png_paths": [],
                "triptych_png_paths": [],
                "depth_png_paths": [],
                "von_mises_png_paths": [],
                "renderer": {"status": "not_requested", "reason": "Genesis part segmentation triptych was not requested"},
            }
        mode = str(visual.get("mode") or "")
        if mode not in {"part_segmentation_triptych", "rgb_triptych", "fixed_rgb_views"}:
            raise GenesisLiveProtocolError(f"Genesis returned unsupported visual telemetry mode: {mode!r}")
        status = payload.get("status", {}) if isinstance(payload.get("status"), dict) else {}
        visual_frames = visual.get("frames") if isinstance(visual.get("frames"), list) else [visual]
        sequence_frames: list[dict[str, Any]] = []
        for index, frame_visual in enumerate(visual_frames):
            if not isinstance(frame_visual, dict):
                continue
            if mode == "fixed_rgb_views":
                frame = self._normalize_fixed_rgb_frame(frame_visual, sequence_index=len(sequence_frames), status=status)
            else:
                frame = self._normalize_visual_frame(frame_visual, sequence_index=len(sequence_frames), status=status)
            if frame is not None:
                sequence_frames.append(frame)
        frame_ids = [int(frame["frame_id"]) for frame in sequence_frames]
        triptych_paths = [str(frame["triptych_png_path"]) for frame in sequence_frames if "triptych_png_path" in frame]
        render_every_steps = int(visual.get("render_every_steps", DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS))
        normalized = {
            "frame_ids": frame_ids,
            "count": len(sequence_frames),
            "triptych_png_paths": triptych_paths,
            "depth_png_paths": [],
            "von_mises_png_paths": [],
            "renderer": {
                "status": "ok" if sequence_frames else "unavailable",
                "mode": mode,
                "source": "genesis_live_visual_telemetry",
                "reason": f"Genesis {mode} capture every configured render step interval",
                "render_every_steps": render_every_steps,
            },
            "triple_view_sequence": {"frames": sequence_frames},
            "genesis_visual_telemetry": visual,
            "status": status,
        }
        if mode == "part_segmentation_triptych":
            normalized["part_segmentation_png_paths"] = triptych_paths
        elif mode == "rgb_triptych":
            normalized["rgb_png_paths"] = triptych_paths
        else:
            normalized["fixed_rgb_sequence"] = {"frames": sequence_frames}
            normalized["fixed_rgb_png_paths"] = [
                {label: str(view["source_png_path"]) for label, view in frame["views"].items()}
                for frame in sequence_frames
            ]
        return normalized

    def _decode_completed_probe_measurement(self, payload: Any) -> CompletedProbeMeasurement:
        """The sole HAG4R evidence-admission boundary for a released probe."""
        if not isinstance(payload, dict):
            raise GenesisLiveProtocolError("Genesis completed probe measurement must be an object")
        dispatch_token = str(payload.get("dispatch_token", "")).strip()
        identity_payload = payload.get("identity")
        if not dispatch_token or not isinstance(identity_payload, dict):
            raise GenesisLiveProtocolError("Genesis completed probe measurement lacks dispatch identity")

        def vec3(value: Any, *, field_name: str) -> tuple[float, float, float]:
            if not isinstance(value, list) or len(value) != 3:
                raise GenesisLiveProtocolError(f"Genesis {field_name} must be a finite vec3")
            if any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value):
                raise GenesisLiveProtocolError(f"Genesis {field_name} must be a finite vec3")
            vector = tuple(float(component) for component in value)
            if not all(math.isfinite(component) for component in vector):
                raise GenesisLiveProtocolError(f"Genesis {field_name} must be a finite vec3")
            return vector

        def vertices(value: Any, *, field_name: str) -> list[int]:
            if not isinstance(value, list) or not value or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise GenesisLiveProtocolError(f"Genesis {field_name} must be a non-empty integer vertex set")
            return list(value)

        entity = str(identity_payload.get("entity", "")).strip()
        controller_id = str(identity_payload.get("controller_id", "")).strip()
        anchor_id = str(identity_payload.get("anchor_id", "")).strip()
        if not entity or not controller_id or not anchor_id:
            raise GenesisLiveProtocolError("Genesis completed probe measurement has incomplete frozen identity")
        identity = {
            "entity": entity,
            "controller_id": controller_id,
            "anchor_id": anchor_id,
            "target_vertices": vertices(identity_payload.get("target_vertices"), field_name="target_vertices"),
            "anchor_vertices": vertices(identity_payload.get("anchor_vertices"), field_name="anchor_vertices"),
            "baseline_relative_env_local_m": list(vec3(identity_payload.get("baseline_relative_env_local_m"), field_name="baseline_relative_env_local_m")),
        }

        def endpoint(value: Any, *, field_name: str) -> ProbeEndpoint:
            if not isinstance(value, dict):
                raise GenesisLiveProtocolError(f"Genesis {field_name} endpoint must be an object")
            step = value.get("simulation_step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise GenesisLiveProtocolError(f"Genesis {field_name} endpoint has invalid simulation_step")
            return ProbeEndpoint(step, vec3(value.get("vector_env_local_m"), field_name=f"{field_name}.vector_env_local_m"))

        under_load = endpoint(payload.get("under_load"), field_name="under_load")
        post_release = endpoint(payload.get("post_release"), field_name="post_release")
        if under_load.simulation_step >= post_release.simulation_step:
            raise GenesisLiveProtocolError("Genesis completed probe measurement endpoint pair is not causally ordered")
        scheduled_fields = ("schedule", "controller_policy", "controller_telemetry", "force_summary")
        present_scheduled_fields = [field_name for field_name in scheduled_fields if field_name in payload]
        if not present_scheduled_fields:
            return CompletedProbeMeasurement(dispatch_token, identity, under_load, post_release)
        if len(present_scheduled_fields) != len(scheduled_fields):
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement has incomplete controller evidence")
        schedule = payload["schedule"]
        controller_policy = payload["controller_policy"]
        controller_telemetry = payload["controller_telemetry"]
        force_summary = payload["force_summary"]
        if not all(isinstance(value, dict) for value in (schedule, controller_policy, controller_telemetry, force_summary)):
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement fields must be objects")
        if set(schedule) != {"load_end_step", "release_step", "recovery_steps"}:
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement schedule has invalid fields")
        load_end_step = schedule["load_end_step"]
        release_step = schedule["release_step"]
        recovery_steps = schedule["recovery_steps"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (load_end_step, release_step, recovery_steps)):
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement schedule values must be integers")
        if load_end_step <= 0 or release_step != load_end_step or recovery_steps <= 0:
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement schedule is inconsistent")
        if under_load.simulation_step != load_end_step or post_release.simulation_step != release_step + recovery_steps:
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement endpoints disagree with its schedule")
        policy_hash = str(controller_policy.get("policy_hash", "")).strip()
        if not policy_hash or controller_telemetry.get("policy_hash") != policy_hash:
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement controller policy is inconsistent")
        telemetry_summary = controller_telemetry.get("summary")
        if not isinstance(telemetry_summary, dict) or telemetry_summary != force_summary:
            raise GenesisLiveProtocolError("Genesis completed scheduled probe measurement force summary is inconsistent")
        return CompletedProbeMeasurement(
            dispatch_token,
            identity,
            under_load,
            post_release,
            schedule=dict(schedule),
            controller_policy=dict(controller_policy),
            controller_telemetry=dict(controller_telemetry),
            force_summary=dict(force_summary),
        )

    def simulation_reset(self, **kwargs: Any) -> dict[str, Any]:
        visual_request = kwargs.pop("visual_request", None)
        # DiagnosticMcpRuntime binds these deterministic capture hints to every
        # live visual tool.  Genesis owns the actual diagnostic output location
        # and camera selection through the generated scene configuration, so
        # they are compatibility metadata rather than RPC fields.  Consume the
        # two known hints before enforcing the public client boundary; unknown
        # keywords remain a protocol error.
        kwargs.pop("output_root", None)
        kwargs.pop("triple_view_cameras", None)
        if kwargs:
            raise GenesisLiveProtocolError(f"unsupported simulation_reset keyword(s): {', '.join(sorted(kwargs))}")
        explicit_visual_request = visual_request is not None
        if visual_request is None:
            bound_request = getattr(self, "_fixed_rgb_visual_request", None)
            visual_request = dict(bound_request) if bound_request is not None else {
                "mode": "part_segmentation_triptych",
                "render_every_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            }
            explicit_visual_request = bound_request is not None
        if isinstance(visual_request, dict) and visual_request.get("mode") == "fixed_rgb_views":
            visual_request = _canonical_fixed_rgb_request(visual_request)
            request_hash = _fixed_rgb_request_hash(visual_request)
            existing_hash = getattr(self, "_fixed_rgb_visual_request_hash", None)
            if existing_hash is not None and existing_hash != request_hash:
                raise GenesisLiveProtocolError("fixed_rgb_views request changed after reset binding")
            self._fixed_rgb_visual_request = visual_request
            self._fixed_rgb_visual_request_hash = request_hash
        self._request("sim.reset", {"diagnostic_visual": visual_request} if explicit_visual_request else {})
        resumed = self._request(
            "sim.resume",
            {
                "steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
                "diagnostic_visual": visual_request,
            },
        )
        return {
            "tool": "simulation_reset",
            "status": STATUS_OK,
            "result": self._normalize_visual_result(resumed),
        }

    def register_probe_action(self, **kwargs: Any) -> dict[str, Any]:
        params = dict(kwargs)
        nested = params.get("probe_apply") if isinstance(params.get("probe_apply"), dict) else None
        if nested is None:
            nested = params.get("probe_release") if isinstance(params.get("probe_release"), dict) else None
        if nested is not None:
            flattened = dict(nested)
            flattened["action_id"] = params.get("action_id")
            flattened["metadata"] = params.get("metadata", {})
            flattened["action_type"] = params.get("action_type")
            controllers = flattened.get("controllers")
            if (
                flattened.get("action") == "probe_release"
                and isinstance(controllers, list)
                and controllers
                and isinstance(controllers[0], dict)
            ):
                flattened["controller_id"] = controllers[0].get("controller_id")
            params = flattened
        result = self._request("probe.action.register", params)
        return {"tool": "register_probe_action", "status": STATUS_OK, "result": result}

    def _apply_registered_action(self, action: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(action, dict):
            return None
        action_id = str(action.get("action_id", "")).strip()
        if not action_id:
            return None
        return self._request("probe.apply", {"action_id": action_id})

    def simulate(self, *, steps: int, action: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        visual_request = kwargs.pop("visual_request", None)
        kwargs.pop("output_root", None)
        kwargs.pop("triple_view_cameras", None)
        if kwargs:
            raise GenesisLiveProtocolError(f"unsupported simulate keyword(s): {', '.join(sorted(kwargs))}")
        if visual_request is None:
            bound_request = getattr(self, "_fixed_rgb_visual_request", None)
            visual_request = dict(bound_request) if bound_request is not None else {
                "mode": "part_segmentation_triptych",
                "render_every_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            }
        if isinstance(visual_request, dict) and visual_request.get("mode") == "fixed_rgb_views":
            visual_request = _canonical_fixed_rgb_request(visual_request)
            request_hash = _fixed_rgb_request_hash(visual_request)
            expected_hash = getattr(self, "_fixed_rgb_visual_request_hash", None)
            if expected_hash is None:
                raise GenesisLiveProtocolError("fixed_rgb_views simulate requires a prior fixed request reset")
            if request_hash != expected_hash:
                raise GenesisLiveProtocolError("fixed_rgb_views request changed after reset binding")
        requested_steps = int(steps)
        applied_action = self._apply_registered_action(action)
        effective_steps = requested_steps
        runtime_step_adaptation: dict[str, Any] | None = None
        action_type = str(action.get("type", "")) if isinstance(action, dict) else ""
        fixed_schedule = action.get("force_limited_schedule") if isinstance(action, dict) else None
        if action_type == "compiled_probe" and fixed_schedule is not None:
            if not isinstance(fixed_schedule, dict):
                raise GenesisLiveProtocolError("force-limited compiled_probe schedule must be an object")
            load_steps = fixed_schedule.get("load_steps")
            if isinstance(load_steps, bool) or not isinstance(load_steps, int) or load_steps != requested_steps:
                raise GenesisLiveProtocolError("force-limited compiled_probe must dispatch its exact frozen load schedule")
        elif action_type == "compiled_probe":
            motion_estimate = _compiled_probe_motion_estimate_payload(applied_action)
            if motion_estimate["effective_steps"] is None:
                runtime_step_adaptation = {
                    "schema_version": ADAPTIVE_COMPILED_PROBE_RESUME_SCHEMA_VERSION,
                    "action_type": action_type,
                    "requested_steps": requested_steps,
                    "effective_steps": requested_steps,
                    "estimated_motion_steps": None,
                    "min_steps": ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS,
                    "max_steps": ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS,
                    "source": motion_estimate["source"],
                    "applied": False,
                    "reason": motion_estimate["reason"],
                }
                if motion_estimate.get("probe_status") is not None:
                    runtime_step_adaptation["probe_status"] = motion_estimate["probe_status"]
            else:
                effective_steps = int(motion_estimate["effective_steps"])
                runtime_step_adaptation = {
                    "schema_version": ADAPTIVE_COMPILED_PROBE_RESUME_SCHEMA_VERSION,
                    "action_type": action_type,
                    "requested_steps": requested_steps,
                    "effective_steps": effective_steps,
                    "estimated_motion_steps": int(motion_estimate["estimated_motion_steps"]),
                    "min_steps": ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS,
                    "max_steps": ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS,
                    "source": motion_estimate["source"],
                    "applied": True,
                }
        resumed = self._request(
            "sim.resume",
            {
                "steps": effective_steps,
                "diagnostic_visual": visual_request,
            },
        )
        result = self._normalize_visual_result(resumed)
        measurement_payload = resumed.get("completed_probe_measurement")
        if action_type == "release_probe" and measurement_payload is not None:
            result["_completed_probe_measurement"] = self._decode_completed_probe_measurement(measurement_payload)
        # A successful synchronous sim.resume response is the runtime-owned
        # completion boundary for the requested window.  Preserve both the
        # agent-requested duration and the duration actually dispatched after
        # compiled-probe motion adaptation so downstream qualification can
        # audit the full window without inferring it from visual frame counts.
        result["steps_requested"] = requested_steps
        result["steps_completed"] = effective_steps
        if applied_action is not None:
            result["action"] = applied_action
        if runtime_step_adaptation is not None:
            result["runtime_step_adaptation"] = runtime_step_adaptation
        return {"tool": "simulate", "status": STATUS_OK, "result": result}

    def query_live_geometry_context(self, **kwargs: Any) -> dict[str, Any]:
        params = {}
        if "entity" in kwargs:
            params["entity"] = kwargs["entity"]
        result = self._request("geometry.context.get", params)
        return {"tool": "query_live_geometry_context", "status": STATUS_OK, "result": result}


def verify_genesis_live_runtime(
    *,
    genesis_root: str | Path | None,
    genesis_env_path: str | Path | None,
    genesis_live_command: str,
    process_env_overrides: dict[str, str] | None = None,
    preflight_timeout_s: float = DEFAULT_GENESIS_LIVE_PREFLIGHT_TIMEOUT_S,
    required_capabilities: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = _resolve_optional_path(genesis_root)
    env_path = _resolve_optional_path(genesis_env_path)
    if root is not None and not root.is_dir():
        raise FileNotFoundError(f"Genesis root is missing: {root}")
    if env_path is not None and not env_path.exists():
        raise FileNotFoundError(f"Genesis conda environment path is missing: {env_path}")
    if not str(genesis_live_command).strip():
        raise ValueError("genesis_live_command must be a non-empty command")
    process_env = os.environ.copy()
    process_env.update({key: str(value) for key, value in (process_env_overrides or {}).items()})
    command = tuple(shlex.split(genesis_live_command))
    if not command:
        raise ValueError("genesis_live_command must include an executable")
    required_capabilities = tuple(str(capability) for capability in required_capabilities)
    preflight_args: tuple[str, ...]
    preflight_mode: str
    if required_capabilities:
        preflight_mode = "capabilities"
        preflight_args = ("--print-capabilities",)
        for capability in required_capabilities:
            preflight_args = (*preflight_args, "--require-capability", capability)
    else:
        preflight_mode = "help"
        preflight_args = ("--help",)
    server_argv = (*command, *preflight_args)
    argv = server_argv if env_path is None else _conda_run_argv(str(env_path), server_argv, repo_root=root)
    completed = subprocess.run(
        argv,
        cwd=root or Path.cwd(),
        env=process_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=preflight_timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Genesis live server preflight failed "
            f"(returncode={completed.returncode}, stdout={completed.stdout[-1000:]}, stderr={completed.stderr[-1000:]})"
        )
    reported_capabilities: list[str] = []
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            reported_capabilities = [str(item) for item in payload]
        elif isinstance(payload, dict):
            reported = _reported_capabilities(payload)
            if reported is not None:
                reported_capabilities = list(reported)
    return {
        "status": "ok",
        "protocol": PROTOCOL_NAME,
        "preflight_mode": preflight_mode,
        "required_capabilities": list(required_capabilities),
        "reported_capabilities": reported_capabilities,
        "genesis_root": str(root) if root is not None else None,
        "genesis_env_path": str(env_path) if env_path is not None else None,
        "genesis_live_command": genesis_live_command,
    }


__all__ = [
    "COMMON_GENESIS_REQUIRED_CAPABILITIES",
    "DEFAULT_GENESIS_LIVE_PREFLIGHT_TIMEOUT_S",
    "GENESIS_RUNTIME_LOG_MAX_LINE_CHARS",
    "GENESIS_RUNTIME_LOG_MAX_MODEL_VISIBLE_CHARS",
    "GENESIS_RUNTIME_LOG_MAX_QUERY_CHARS",
    "GENESIS_RUNTIME_LOG_MAX_RETURNED_LINES_PER_STREAM",
    "GENESIS_RUNTIME_LOG_MAX_SCAN_LINES_PER_STREAM",
    "GENESIS_RUNTIME_LOG_SCHEMA_VERSION",
    "GenesisLiveApiSession",
    "GenesisLiveHandshakeError",
    "SERVER_TO_HAG4R_VIEW",
    "SURFACE_GENESIS_REQUIRED_CAPABILITIES",
    "verify_genesis_live_runtime",
]
