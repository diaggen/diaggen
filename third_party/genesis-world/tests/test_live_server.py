import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import igl
import numpy as np
import pytest
from PIL import Image

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxAnchorRecord
from genesis.live.capabilities import capability_report
from genesis.live.protocol import (
    ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY,
    FIXED_RGB_VIEW_CAPABILITY,
    MAX_MESSAGE_BYTES,
    PROTOCOL,
    GenesisLiveError,
    capabilities_for_report,
    encode_frame,
    recv_json,
    send_json,
)
from genesis.live.server import GenesisLiveServer
from genesis.live.session import GenesisLiveSession
from genesis.live.visual_telemetry import (
    GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
    VisualTelemetry,
    canonical_fixed_rgb_request,
    fixed_rgb_request_hash,
)

_TWO_TET_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_TWO_TETS = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)


def _fixed_rgb_request():
    return {
        "mode": "fixed_rgb_views",
        "render_every_steps": 10,
        "views": [
            {
                "name": "full",
                "position": [1.0, 1.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "resolution": [512, 512],
                "fov_degrees": 40.0,
            },
            {
                "name": "context",
                "position": [2.0, 0.0, 1.0],
                "look_at": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "resolution": [512, 512],
                "fov_degrees": 40.0,
            },
        ],
    }


def _measurement_session_with_anchors(anchors: list[BoxAnchorRecord]) -> GenesisLiveSession:
    entity = object()
    session = GenesisLiveSession.__new__(GenesisLiveSession)
    session.entities = {"body": entity}
    session.anchor_records = {"body": anchors}
    session.anchor_by_id = {"body": {record.anchor_id: record for record in anchors}}
    session.active_measurement_by_controller = {}
    session.controllers = {}
    session.current_step = 0
    session.running = False
    session.paused = True
    session._entity_positions_for_validation = lambda *_args, **_kwargs: _TWO_TET_VERTS
    return session


def _measurement_anchor(anchor_id: str, selected_vertices: list[int]) -> BoxAnchorRecord:
    box = np.array([-0.1, -0.1, -0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    return BoxAnchorRecord(
        anchor_id=anchor_id,
        frame="env_local",
        source_box=box.copy(),
        env_local_box=box.copy(),
        selected_vertices=np.asarray(selected_vertices, dtype=np.int64),
    )


def test_probe_measurement_lifecycle_emits_one_release_pair_and_retires_active_record():
    anchor_id = "brim_midspan_from_front_band"
    session = _measurement_session_with_anchors([_measurement_anchor(anchor_id, [0])])

    active = session.prepare_probe_measurement(
        measurement={"dispatch_token": "runtime-token", "anchor_id": anchor_id},
        entity_name="body",
        controller_id="controller_1",
        target_vertices=np.array([1], dtype=np.int64),
    )
    session.scene = SimpleNamespace(step=lambda: None)
    session.status = dict
    session._validate_visual_request = lambda _visual: None
    session._visual_render_every_steps = lambda _visual: None
    session._validate_fem_state = lambda **_kwargs: None
    session._sync_visual_overlays = lambda **_kwargs: None
    session._visual_result_from_frames = lambda *_args, **_kwargs: {}
    session.publish_probe_measurement("controller_1", SimpleNamespace(prepare_step=lambda **_kwargs: None), active)
    compiled_resume = session.resume({"steps": 1})
    assert "completed_probe_measurement" not in compiled_resume
    assert active.under_load is not None
    session.release_probe_measurement("controller_1")
    release_resume = session.resume({"steps": 1})
    completed = release_resume["completed_probe_measurement"]

    assert completed["dispatch_token"] == "runtime-token"
    assert completed["under_load"]["simulation_step"] < completed["post_release"]["simulation_step"]
    assert session.active_measurement_by_controller == {}


def test_scheduled_probe_measurement_waits_for_load_and_recovery_endpoints():
    anchor_id = "scheduled_anchor"
    session = _measurement_session_with_anchors([_measurement_anchor(anchor_id, [0])])
    active = session.prepare_probe_measurement(
        measurement={
            "dispatch_token": "scheduled-token",
            "anchor_id": anchor_id,
            "schedule": {"load_steps": 2, "recovery_steps": 2},
        },
        entity_name="body",
        controller_id="controller_1",
        target_vertices=np.array([1], dtype=np.int64),
    )
    session.scene = SimpleNamespace(step=lambda: None)
    session.status = dict
    session._validate_visual_request = lambda _visual: None
    session._visual_render_every_steps = lambda _visual: None
    session._validate_fem_state = lambda **_kwargs: None
    session._sync_visual_overlays = lambda **_kwargs: None
    session._visual_result_from_frames = lambda *_args, **_kwargs: {}
    session.publish_probe_measurement("controller_1", SimpleNamespace(prepare_step=lambda **_kwargs: None), active)

    load = session.resume({"steps": 2})
    assert "completed_probe_measurement" not in load
    assert active.under_load.simulation_step == 2
    policy = {"policy_hash": "policy"}
    telemetry = {"summary": {"active_pre_step_count": 2}}
    session.release_probe_measurement(
        "controller_1",
        controller_state={"controller_policy": policy, "controller_telemetry": telemetry},
    )
    release = session.resume({"steps": 2})
    completed = release["completed_probe_measurement"]

    assert completed["schedule"] == {"load_end_step": 2, "release_step": 2, "recovery_steps": 2}
    assert completed["under_load"]["simulation_step"] == 2
    assert completed["post_release"]["simulation_step"] == 4
    assert completed["controller_policy"] == policy
    assert completed["controller_telemetry"] == telemetry
    assert completed["force_summary"] == telemetry["summary"]


def test_probe_measurement_rejects_invalid_physical_vertex_domain():
    anchor_id = "brim_midspan_from_front_band"
    session = _measurement_session_with_anchors([_measurement_anchor(anchor_id, [0])])

    with pytest.raises(GenesisLiveError, match="physical FEM domain") as exc_info:
        session.prepare_probe_measurement(
            measurement={"dispatch_token": "runtime-token", "anchor_id": anchor_id},
            entity_name="body",
            controller_id="controller_1",
            target_vertices=np.array([99], dtype=np.int64),
        )

    assert exc_info.value.code == "invalid_probe_measurement"


def test_non_y_live_apply_passes_runtime_selected_vertices_to_tracker(tmp_path):
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(tmp_path / "outputs")
    )
    response = session.dispatch(
        "probe.apply",
        {
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_steps": 12,
            "measurement": {"dispatch_token": "non_y_dispatch", "anchor_id": "bottom_pin"},
            "controllers": [
                {
                    "controller_id": "non_y_box",
                    "aabb_box": {"frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]},
                    "distance_scale": 0.5,
                    "motion_axis": "-X",
                }
            ],
        },
    )
    selected = response["probe"]["selected_vertices"]
    active = session.active_measurement_by_controller["non_y_box"]

    assert selected == [3]
    assert active.target_vertices.tolist() == selected
    assert session.controllers["non_y_box"].state.motion_axis == "-X"
    session.dispatch("sim.resume", {"steps": 12})
    assert active.under_load is not None


def test_probe_measurement_capability_is_reported_and_requireable():
    report = capability_report((ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY,), diagnostic_scene=True)

    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in report["capabilities"]
    assert report["missing_required_capabilities"] == []


def _write_scene_config(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_tets.mesh"
    igl.writeMESH(str(mesh_path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))
    config = {
        "sim_options": {"dt": 0.001},
        "fem_options": {"enable_vertex_constraints": True, "enable_floor": True},
        "entities": [
            {
                "name": "body",
                "morph": {"type": "tet_mesh", "file": str(mesh_path)},
                "material": {"type": "elastic", "E": 100000.0, "nu": 0.3, "rho": 1000.0},
                "anchors": [
                    {"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}
                ],
            }
        ],
    }
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _request(sock, method, params=None, request_id="req"):
    send_json(sock, {"request_id": request_id, "method": method, "params": params or {}})
    return recv_json(sock)


def _assert_png_record(record):
    path = Path(record["path"])
    assert path.exists()
    assert record["byte_size"] == path.stat().st_size
    assert record["byte_size"] > 0
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.size == (record["width"], record["height"])


def _wait_ready(path: Path, proc: subprocess.Popen, timeout_s: float = 90.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server exited early with {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        time.sleep(0.1)
    raise AssertionError("server did not write ready file")


def test_protocol_framing_round_trip():
    left, right = socket.socketpair()
    try:
        send_json(left, {"request_id": "a", "method": "session.handshake", "params": {}})
        assert recv_json(right) == {"request_id": "a", "method": "session.handshake", "params": {}}
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize(("requested_backend", "expected"), (("cpu", gs.cpu), ("cuda", gs.cuda)))
def test_live_session_initialization_honors_requested_volumetric_backend(
    monkeypatch, backend, requested_backend, expected
):
    del backend
    calls = []
    monkeypatch.setattr(gs, "_initialized", False)
    monkeypatch.setattr(gs, "init", lambda **kwargs: calls.append(kwargs))

    session = GenesisLiveSession.__new__(GenesisLiveSession)
    session._ensure_genesis_initialized(requested_backend, requires_surface_backend=False)

    assert calls == [{"backend": expected, "logging_level": "warning"}]


@pytest.mark.parametrize("backend", [None])
def test_live_session_rejects_initialized_volumetric_backend_mismatch(monkeypatch, backend):
    del backend
    monkeypatch.setattr(gs, "_initialized", True)
    monkeypatch.setattr(gs, "backend", gs.cpu)

    session = GenesisLiveSession.__new__(GenesisLiveSession)
    with pytest.raises(GenesisLiveError) as exc_info:
        session._ensure_genesis_initialized("cuda", requires_surface_backend=False)

    assert exc_info.value.code == "backend_mismatch"
    assert exc_info.value.details == {"current_backend": str(gs.cpu), "required_backend": "cuda"}


def test_live_session_lifecycle_and_box_action(tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )

    handshake = session.dispatch("session.handshake", {})
    assert handshake["protocol"] == PROTOCOL
    assert "live_box_controller_actions" in handshake["capabilities"]
    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in handshake["capabilities"]
    assert session.status()["current_step"] == 0
    assert session.status()["paused"]
    assert session.scene.fem_solver.enable_floor is False

    geometry = session.dispatch("geometry.context.get", {"entity": "body"})
    assert geometry["representation"] == "volumetric"
    assert geometry["primitive_kind"] == "tetrahedron"
    assert geometry["vertex_count"] == 5
    assert geometry["element_count"] == 2
    assert geometry["max_extent"] == pytest.approx(2.0)

    resumed = session.dispatch("sim.resume", {"steps": 3})
    assert resumed["status"]["current_step"] == 3
    assert resumed["status"]["paused"]
    assert session.dispatch("sim.pause", {})["paused"]

    action = session.dispatch(
        "probe.apply",
        {
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_steps": 12,
            "controllers": [
                {
                    "controller_id": "diag_box",
                    "aabb_box": {"frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]},
                    "distance_scale": 0.5,
                    "motion_axis": "+Y",
                }
            ],
        },
    )
    assert action["probe"]["selected_vertex_count"] == 1
    action_state = action["probe"]["controller_state"]
    assert action_state["speed"] == pytest.approx(0.6)
    assert action_state["motion_axis"] == "+Y"
    assert action_state["motion_active"] is True
    assert action_state["moved_distance"] == pytest.approx(0.0)
    assert action_state["displacement"] == pytest.approx([0.0, 0.0, 0.0])

    visual = session.dispatch("sim.resume", {"steps": 10, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]
    assert visual["rendered"]
    assert visual["count"] == 1
    assert visual["render_every_steps"] == 10
    assert visual["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert visual["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert visual["renderer"]["camera_model"] == "pinhole"
    assert visual["renderer"]["debug_camera"] is True
    assert visual["panel_order"] == ["top", "northeast", "southwest"]
    assert len(visual["views"]) == 3
    for record in visual["views"]:
        assert record["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["camera"]["model"] == "pinhole"
        assert record["camera"]["debug"] is True
    for record in visual["frame_metadata"]:
        _assert_png_record(record)
    live_overlay = next(
        record
        for record in visual["overlays"]
        if record.get("kind") == "live_box_controller" and record.get("controller_id") == "diag_box"
    )
    assert live_overlay["motion_active"] is True
    assert live_overlay["moved_distance"] > 0.0
    assert live_overlay["displacement"][1] == pytest.approx(live_overlay["moved_distance"])
    assert live_overlay["rendered_env_local_box"][1] > live_overlay["env_local_box"][1]
    live_marker = next(
        record
        for record in visual["debug_markers"]
        if record.get("kind") == "live_box_controller" and record.get("controller_id") == "diag_box"
    )
    assert live_marker["bounds"][0][1] == pytest.approx(live_overlay["rendered_env_local_box"][1])
    assert live_marker["bounds"][1][1] == pytest.approx(live_overlay["rendered_env_local_box"][4])
    assert session.dispatch("command.status", {})["last_rendered_frame_index"] == visual["stitched"]["frame_index"]
    assert visual["frames"][0]["stitched"]["frame_index"] == visual["stitched"]["frame_index"]

    fused = session.dispatch("observation.fused", {})
    assert fused["material"]["material_type"] == "Elastic"
    assert fused["material"]["heterogeneous"] is False

    release = session.dispatch("probe.apply", {"action": "probe_release", "controller_id": "diag_box"})
    assert not release["probe"]["controller_state"]["active"]
    assert session.dispatch("sim.reset", {})["current_step"] == 0


def test_live_session_forces_floor_off_and_allows_below_floor_downward_motion(tmp_path):
    config_path = _write_scene_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sim_options"]["gravity"] = [0.0, 0.0, 0.0]
    config["fem_options"]["enable_floor"] = True
    config["entities"][0]["anchors"] = []
    config["entities"][0]["morph"]["pos"] = [0.0, 0.0, -2.0]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    session = GenesisLiveSession(scene_config_path=str(config_path), start_paused=True)
    entity = session.entities["body"]
    assert session.scene.fem_solver.enable_floor is False
    entity.set_velocity((0.0, 0.0, -0.5))
    initial_z = float(entity.get_state().pos[..., 2].mean())
    session.scene.step(update_visualizer=False)
    final_z = float(entity.get_state().pos[..., 2].mean())
    assert final_z < initial_z - 1.0e-5


def test_live_session_rejects_unsupported_visual_modes(tmp_path):
    session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True)

    response = session.handle_request(
        {"request_id": "depth", "method": "sim.resume", "params": {"steps": 1, "diagnostic_visual": {"mode": "depth"}}}
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "unsupported_visual_mode"


def test_fixed_rgb_contract_is_advertised_and_rejects_camera_drift():
    request = _fixed_rgb_request()

    assert FIXED_RGB_VIEW_CAPABILITY in capabilities_for_report(False, diagnostic_scene=True)
    assert canonical_fixed_rgb_request(request) == request
    assert len(fixed_rgb_request_hash(request)) == 64

    drifted = json.loads(json.dumps(request))
    drifted["views"][0]["resolution"] = [256, 256]
    with pytest.raises(ValueError, match="resolution"):
        canonical_fixed_rgb_request(drifted)


def test_fixed_rgb_capture_emits_clean_request_bound_views(monkeypatch, tmp_path):
    class FakeCamera:
        model = "pinhole"
        debug = False
        res = (512, 512)
        fov = 40.0
        near = 0.1
        far = 20.0

        def set_pose(self, *, pos, lookat, up):
            self.pos = pos
            self.lookat = lookat
            self.up = up

    telemetry = VisualTelemetry(tmp_path)
    telemetry.fixed_rgb_cameras = {"full": FakeCamera(), "context": FakeCamera()}
    monkeypatch.setattr(
        "genesis.live.visual_telemetry._render_camera_rgb",
        lambda *_args, **_kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
    )
    request = _fixed_rgb_request()

    result = telemetry.capture_fixed_rgb_views(SimpleNamespace(current_step=10), visual=request, frame_index=10)

    assert result["view_order"] == ["full", "context"]
    assert result["camera_specs"] == request["views"]
    assert result["camera_specs_hash"] == fixed_rgb_request_hash(request)
    assert result["overlays"] == result["debug_markers"] == []
    assert [record["camera"]["debug"] for record in result["views"]] == [False, False]
    assert [record["width"] for record in result["views"]] == [512, 512]
    assert Path(result["metadata_path"]).is_file()


def test_resume_reports_invalid_fem_state_before_visual_capture(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )
    entity = session.entities["body"]
    bad_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [np.nan, 2.0, 3.0],
            [4.0, np.inf, 5.0],
            [-1.0, -2.0, -3.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(session.scene, "step", lambda: None)
    monkeypatch.setattr(entity, "get_state", lambda: SimpleNamespace(pos=bad_positions))

    response = session.handle_request(
        {
            "request_id": "bad",
            "method": "sim.resume",
            "params": {"steps": 1, "diagnostic_visual": {"mode": "rgb_triptych"}},
        }
    )

    assert response["status"] == "error"
    assert response["error"]["code"] == "invalid_simulation_state"
    details = response["error"]["details"]
    assert details["entity"] == "body"
    assert details["current_step"] == 1
    assert details["first_bad_step"] == 1
    assert details["checked_at"] == "after_step"
    assert details["shape"] == [5, 3]
    assert details["finite_vertex_count"] == 3
    assert details["nonfinite_vertex_count"] == 2
    assert details["finite_scalar_count"] == 13
    assert details["nonfinite_scalar_count"] == 2
    assert details["finite_min"] == pytest.approx(-3.0)
    assert details["finite_max"] == pytest.approx(5.0)
    assert details["first_nonfinite_vertex_index"] == 2
    status = session.status()
    assert status["paused"] is True
    assert status["running"] is False
    assert status["fatal_error"]["code"] == "invalid_simulation_state"
    assert status["last_rendered_frame_index"] is None
    assert not (output_dir / "png_rgb_triptych" / "frame_000001.png").exists()


def test_rgb_triptych_render_cadence_is_frame_not_step(tmp_path):
    output_dir = tmp_path / "outputs"
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)), start_paused=True, output_dir=str(output_dir)
    )

    visual = session.dispatch("sim.resume", {"steps": 20, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]

    assert visual["rendered"]
    assert visual["count"] == 2
    assert visual["render_every_steps"] == 10
    assert [frame["stitched"]["frame_index"] for frame in visual["frames"]] == [10, 20]
    assert session.dispatch("command.status", {})["current_step"] == 20
    assert session.dispatch("command.status", {})["last_rendered_frame_index"] == 20
    assert (output_dir / "png_rgb_triptych" / "frame_000010.png").is_file()
    assert (output_dir / "png_rgb_triptych" / "frame_000020.png").is_file()
    assert not (output_dir / "png_rgb_triptych" / "frame_000001.png").exists()


def test_live_server_subprocess_ready_handshake_status_and_close(tmp_path):
    config_path = _write_scene_config(tmp_path)
    ready_path = tmp_path / "ready.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "genesis.live.server",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_path),
            "--scene-config",
            str(config_path),
            "--start-paused",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _wait_ready(ready_path, proc)
        assert ready["protocol"] == PROTOCOL
        assert ready["server_pid"] == proc.pid
        assert ready["start_paused"] is True
        assert ready["scene_config_path"] == str(config_path)
        assert "live_box_controller_actions" in ready["capabilities"]
        assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in ready["capabilities"]

        with socket.create_connection((ready["host"], ready["port"]), timeout=30.0) as sock:
            handshake = _request(sock, "session.handshake", request_id="hello")
            assert handshake["status"] == "ok"
            assert handshake["data"]["protocol"] == PROTOCOL
            assert handshake["data"]["capabilities"] == ready["capabilities"]

            status = _request(sock, "command.status", request_id="status")
            assert status["data"]["paused"]
            assert status["data"]["current_step"] == 0

            resumed = _request(sock, "sim.resume", {"steps": 2}, request_id="resume")
            assert resumed["data"]["status"]["current_step"] == 2
            assert resumed["data"]["status"]["paused"]

            closed = _request(sock, "session.close", request_id="close")
            assert closed["status"] == "ok"

        proc.wait(timeout=30)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


def test_diagnostic_live_ready_and_handshake_hide_textured_overlay_capabilities(tmp_path):
    config_path = _write_scene_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["agentic_diagnostics"] = {"mode": "structured_live_diagnostic"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    ready_path = tmp_path / "diagnostic_ready.json"
    server = GenesisLiveServer(
        host="127.0.0.1",
        port=0,
        ready_file=str(ready_path),
        scene_config_path=str(config_path),
        start_paused=True,
    )

    server.write_ready()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    handshake = server.session.handshake()
    forbidden = {
        "deformable_textured_visual_overlay",
        "visual_overlay_depth_normal_triptych_telemetry",
        "visual_overlay_vertex_trace",
    }

    assert forbidden.isdisjoint(ready["capabilities"])
    assert forbidden.isdisjoint(handshake["capabilities"])
    assert ready["capabilities"] == handshake["capabilities"]
    assert "part_segmentation_triptych_telemetry" in ready["capabilities"]
    assert ANCHOR_RELATIVE_PROBE_MEASUREMENT_CAPABILITY in ready["capabilities"]


def test_live_package_does_not_contain_removed_server_protocol_names():
    live_dir = Path(__file__).resolve().parents[1] / "genesis" / "live"
    forbidden = ("RealSimLiveServer", "realsim-live-v1")
    for path in live_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
