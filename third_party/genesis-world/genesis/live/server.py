from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from contextlib import closing

from .capabilities import capability_report
from .protocol import SURFACE_BACKEND_CAPABILITIES, GenesisLiveError, recv_json, send_json
from .ready_file import ready_payload, write_ready_file
from .session import GenesisLiveSession


class GenesisLiveServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        ready_file: str,
        scene_config_path: str | None,
        start_paused: bool,
        output_dir: str | None = None,
        heartbeat_interval_s: float = 1.0,
    ):
        self.host = host
        self.port = int(port)
        self.ready_file = ready_file
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.session = GenesisLiveSession(
            scene_config_path=scene_config_path,
            start_paused=start_paused,
            output_dir=output_dir,
        )
        self._shutdown = threading.Event()

    def serve_forever(self) -> None:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(8)
            server_sock.settimeout(0.2)
            self.host, self.port = server_sock.getsockname()
            self.write_ready()

            while not self._shutdown.is_set():
                try:
                    client, _addr = server_sock.accept()
                except socket.timeout:
                    continue
                with closing(client):
                    client.settimeout(None)
                    self._serve_client(client)

    def write_ready(self) -> None:
        write_ready_file(
            self.ready_file,
            ready_payload(
                pid=os.getpid(),
                host=self.host,
                port=self.port,
                scene_config_path=self.session.scene_config_path,
                start_paused=self.session.start_paused,
                heartbeat_interval_s=self.heartbeat_interval_s,
                status=self.session.status(),
                capabilities=self.session.handshake()["capabilities"],
                backend_requirements=self.session.backend_requirements(),
                session_token=self.session.session_id,
            ),
        )

    def _serve_client(self, client: socket.socket) -> None:
        while not self._shutdown.is_set():
            try:
                request = recv_json(client)
            except EOFError:
                return
            except GenesisLiveError as exc:
                send_json(client, {"status": "error", "error": exc.to_dict()})
                continue

            response = self.session.handle_request(request)
            send_json(client, response)
            self.write_ready()
            if request.get("method") == "session.close":
                self._shutdown.set()
                return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Genesis live diagnostic server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--scene-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--heartbeat-interval-s", type=float, default=1.0)
    parser.add_argument("--print-capabilities", action="store_true")
    parser.add_argument("--require-capability", action="append", default=[])
    paused = parser.add_mutually_exclusive_group()
    paused.add_argument("--start-paused", dest="start_paused", action="store_true", default=True)
    paused.add_argument("--no-start-paused", dest="start_paused", action="store_false")
    args = parser.parse_args(argv)

    if args.print_capabilities:
        report = capability_report(args.require_capability)
        missing = report["missing_required_capabilities"]
        if missing:
            error_code = "missing_required_capability"
            if any(capability in SURFACE_BACKEND_CAPABILITIES for capability in missing):
                error_code = "unsupported_surface_backend"
            report["error"] = {"code": error_code, "missing_required_capabilities": missing}
            print(json_dumps(report), flush=True)
            return 1
        print(json_dumps(report), flush=True)
        return 0

    if args.ready_file is None:
        parser.error("--ready-file is required unless --print-capabilities is used")

    server = GenesisLiveServer(
        host=args.host,
        port=args.port,
        ready_file=args.ready_file,
        scene_config_path=args.scene_config,
        start_paused=args.start_paused,
        output_dir=args.output_dir,
        heartbeat_interval_s=args.heartbeat_interval_s,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
