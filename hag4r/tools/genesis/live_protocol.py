from __future__ import annotations

import json
import socket
import struct
from typing import Any

PROTOCOL_NAME = "genesis-live-v1"
DEFAULT_READY_TIMEOUT_S = 300.0
DEFAULT_HANDSHAKE_TIMEOUT_MS = 30_000
DEFAULT_READ_TIMEOUT_MS = 30_000
DEFAULT_MUTATION_TIMEOUT_MS = 120_000
DEFAULT_HEARTBEAT_MS = 1_000
DEFAULT_CLIENT_LEASE_TIMEOUT_MS = 30_000
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_UNSUPPORTED = "unsupported"
STATUS_TIMEOUT = "timeout"

PUBLIC_LIVE_TOOL_NAMES = (
    "inspect_genesis_runtime_logs",
    "simulation_reset",
    "simulate",
    "query_live_geometry_context",
)

HEADER_STRUCT = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class GenesisLiveProtocolError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
    ):
        self.message = str(message)
        self.code = str(code or "genesis_live_protocol_error")
        self.details = details
        self.response = response
        super().__init__(self.message)


def encode_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise GenesisLiveProtocolError(f"Genesis live frame is too large: {len(body)} bytes")
    return HEADER_STRUCT.pack(len(body)) + body


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Genesis live socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, HEADER_STRUCT.size)
    (length,) = HEADER_STRUCT.unpack(header)
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise GenesisLiveProtocolError(f"Genesis live frame length is invalid: {length}")
    payload = json.loads(_recv_exact(sock, length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise GenesisLiveProtocolError("Genesis live frame payload must be a JSON object")
    return payload


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_frame(payload))


__all__ = [
    "DEFAULT_CLIENT_LEASE_TIMEOUT_MS",
    "DEFAULT_HANDSHAKE_TIMEOUT_MS",
    "DEFAULT_HEARTBEAT_MS",
    "DEFAULT_MUTATION_TIMEOUT_MS",
    "DEFAULT_READ_TIMEOUT_MS",
    "DEFAULT_READY_TIMEOUT_S",
    "GenesisLiveProtocolError",
    "PROTOCOL_NAME",
    "PUBLIC_LIVE_TOOL_NAMES",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_TIMEOUT",
    "STATUS_UNSUPPORTED",
    "recv_json",
    "send_json",
]
