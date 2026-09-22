from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

PROTOCOL = "genesis-live-v1"
HEADER_STRUCT = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY = "anchor_relative_probe_measurement_telemetry"
FIXED_RGB_VIEW_CAPABILITY = "fixed_rgb_view_telemetry"

BASE_CAPABILITIES = (
    "pause_resume_reset",
    "deformable_textured_visual_overlay",
    "rgb_triptych_telemetry",
    "visual_overlay_depth_normal_triptych_telemetry",
    "visual_overlay_vertex_trace",
    "part_segmentation_triptych_telemetry",
    "part_shaded_triptych_telemetry",
    "diagnostic_visual_capture",
    FIXED_RGB_VIEW_CAPABILITY,
    "frame_metadata",
    "geometry_context",
    "fused_observation",
)

# Agentic diagnostics use the same live server, but their scene contract rejects
# visual_overlay.  Keep that unsupported feature set explicit so diagnostic
# evidence cannot imply a textured visual overlay influenced its observations.
TEXTURED_VISUAL_OVERLAY_CAPABILITIES = (
    "deformable_textured_visual_overlay",
    "visual_overlay_depth_normal_triptych_telemetry",
    "visual_overlay_vertex_trace",
)

DIAGNOSTIC_BASE_CAPABILITIES = tuple(
    capability for capability in BASE_CAPABILITIES if capability not in TEXTURED_VISUAL_OVERLAY_CAPABILITIES
)

VOLUMETRIC_CAPABILITIES = (
    "native_fem_force_limited_box_ee",
    "volumetric_mesh_import",
    "heterogeneous_fem_material_arrays",
    "static_box_anchors",
    "live_box_controller_actions",
    ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY,
)

SURFACE_CAPABILITIES = (
    "surface_mesh_import",
    "surface_shell_diagnostics",
)

SURFACE_POSITION_CONSTRAINT_CAPABILITIES = (
    "surface_static_box_anchors",
    "surface_live_box_controller_actions",
)

SURFACE_HETEROGENEOUS_CAPABILITIES = ("heterogeneous_surface_material_arrays",)

SURFACE_BACKEND_CAPABILITIES = (
    SURFACE_CAPABILITIES + SURFACE_POSITION_CONSTRAINT_CAPABILITIES + SURFACE_HETEROGENEOUS_CAPABILITIES
)

CAPABILITIES = VOLUMETRIC_CAPABILITIES + BASE_CAPABILITIES


def capabilities_for_report(
    surface_backend_available: bool,
    surface_heterogeneous_backend_available: bool = False,
    surface_position_constraint_available: bool = False,
    *,
    diagnostic_scene: bool = False,
) -> tuple[str, ...]:
    capabilities = list(VOLUMETRIC_CAPABILITIES)
    if surface_backend_available:
        capabilities.extend(SURFACE_CAPABILITIES)
    if surface_position_constraint_available:
        capabilities.extend(SURFACE_POSITION_CONSTRAINT_CAPABILITIES)
    if surface_heterogeneous_backend_available:
        capabilities.extend(SURFACE_HETEROGENEOUS_CAPABILITIES)
    capabilities.extend(DIAGNOSTIC_BASE_CAPABILITIES if diagnostic_scene else BASE_CAPABILITIES)
    return tuple(capabilities)


@dataclass
class GenesisLiveError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


def encode_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise GenesisLiveError("message_too_large", f"message is {len(body)} bytes")
    return HEADER_STRUCT.pack(len(body)) + body


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, HEADER_STRUCT.size)
    (length,) = HEADER_STRUCT.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise GenesisLiveError("message_too_large", f"message is {length} bytes")
    try:
        payload = json.loads(_recv_exact(sock, length).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise GenesisLiveError("invalid_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise GenesisLiveError("invalid_message", "framed JSON message must be an object")
    return payload


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_frame(payload))


def ok_response(request_id: str | int | None, data: dict[str, Any]) -> dict[str, Any]:
    return {"request_id": request_id, "status": "ok", "protocol": PROTOCOL, "data": data}


def error_response(request_id: str | int | None, error: GenesisLiveError) -> dict[str, Any]:
    return {"request_id": request_id, "status": "error", "protocol": PROTOCOL, "error": error.to_dict()}
