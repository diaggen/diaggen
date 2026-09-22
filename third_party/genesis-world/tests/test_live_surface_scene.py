import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import genesis as gs
from genesis.live.capabilities import surface_backend_status
from genesis.live.protocol import PROTOCOL, GenesisLiveError, recv_json, send_json
from genesis.live.session import GenesisLiveSession
from genesis.live.triptych import HAG4R_LABELS, PANEL_ORDER
from genesis.live.visual_telemetry import GENESIS_NATIVE_DEBUG_CAMERA_RENDERER, PANEL_SIZE


_SURFACE_OBJ = """\
v -0.5 0.0 0.0
v  0.5 0.0 0.0
v -0.5 0.0 0.5
v  0.5 0.0 0.5
f 1 2 3
f 2 4 3
"""

_SOURCE_FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
_TWO_TRI_E_NU = np.array([[1.0e5, 0.25], [2.0e5, 0.40]], dtype=np.float64)
_TWO_TRI_DENSITY = np.array([1000.0, 2000.0], dtype=np.float64)
_TWO_TRI_THICKNESS = np.array([0.002, 0.003], dtype=np.float64)
_TWO_TRI_AREA_DENSITY = _TWO_TRI_DENSITY * _TWO_TRI_THICKNESS
_TWO_TRI_LABELS = np.array([3, 7], dtype=np.int32)


def _write_surface_obj(tmp_path: Path, obj_text: str = _SURFACE_OBJ) -> Path:
    mesh_path = tmp_path / "two_triangles.obj"
    mesh_path.write_text(obj_text, encoding="utf-8")
    return mesh_path


def _write_surface_material_npz(path: Path):
    np.savez(
        path,
        tri_E_nu=_TWO_TRI_E_NU,
        tri_density=_TWO_TRI_DENSITY,
        tri_thickness=_TWO_TRI_THICKNESS,
        tri_area_density=_TWO_TRI_AREA_DENSITY,
        tri_part_labels=_TWO_TRI_LABELS,
    )


def _write_surface_scene_config(
    tmp_path: Path,
    *,
    material_type: str = "surface_shell",
    heterogeneous: bool | dict = False,
    part_segmentation: bool = False,
    archived_palette: bool = False,
    obj_text: str = _SURFACE_OBJ,
) -> Path:
    material = {
        "type": material_type,
        "E": 10000.0,
        "nu": 0.45,
        "rho": 200.0,
        "thickness": 0.001,
        "bending_stiffness": 0.0,
        "friction_mu": 0.1,
    }
    material_path = None
    if heterogeneous:
        if heterogeneous is True:
            material_path = tmp_path / "surface_materials.npz"
            _write_surface_material_npz(material_path)
            material["heterogeneous"] = {"kind": "surface_triangles", "material_file": str(material_path)}
        else:
            material["heterogeneous"] = heterogeneous
    config = {
        "backend": "cuda",
        "sim_options": {"dt": 0.001, "gravity": [0.0, 0.0, 0.0]},
        "fem_options": {"enable_floor": True},
        "coupler_options": {
            "contact_enable": False,
            "enable_rigid_ground_contact": False,
            "enable_rigid_rigid_contact": False,
        },
        "entities": [
            {
                "name": "surface",
                "morph": {"type": "surface_mesh", "file": str(_write_surface_obj(tmp_path, obj_text=obj_text))},
                "material": material,
            }
        ],
    }
    if part_segmentation:
        if material_path is None:
            raise ValueError("surface part-segmentation fixture requires heterogeneous=True")
        part_colors = np.asarray(
            [[20 + 20 * index, 210 - 10 * index, 60 + index] for index in range(8)],
            dtype=np.uint8,
        )
        palette_path = tmp_path / "surface_part_palette.npz"
        np.savez(palette_path, part_colors=part_colors)
        context_palette = {
            "background": [0, 0, 0],
            "fixture": [41, 110, 255],
            "probe": [255, 89, 41],
        }
        if archived_palette:
            context_palette["ground"] = [96, 96, 96]
        config["entities"][0]["part_segmentation"] = {
            "primitive_labels_file": str(material_path),
            "primitive_labels_key": "tri_part_labels",
            "palette_file": str(palette_path),
            "palette_key": "part_colors",
            "parts": [
                {
                    "part_id": int(part_id),
                    "part_name": f"part_{part_id}",
                    "part_color_rgb": part_colors[part_id].tolist(),
                }
                for part_id in _TWO_TRI_LABELS
            ],
            "context_palette": context_palette,
        }
    config_path = tmp_path / "surface_scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _request(sock, method, params=None, request_id="req"):
    send_json(sock, {"request_id": request_id, "method": method, "params": params or {}})
    return recv_json(sock)


def _wait_ready(path: Path, proc: subprocess.Popen, timeout_s: float = 120.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server exited early with {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        time.sleep(0.1)
    raise AssertionError("server did not write ready file")


def _require_surface_backend():
    status = surface_backend_status()
    if not status["available"]:
        pytest.skip(f"surface backend unavailable: {status}")


def _assert_png_record_readable(record, expected_size=None):
    path = Path(record["path"])
    assert path.exists()
    assert record["byte_size"] == path.stat().st_size
    assert record["byte_size"] > 0
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        assert rgb.size == (record["width"], record["height"])
        if expected_size is not None:
            assert rgb.size == expected_size
        assert record["width"] > 0
        assert record["height"] > 0


def _assert_vec3(values):
    assert len(values) == 3
    assert all(np.isfinite(float(value)) for value in values)


def _assert_no_fallback_terms(payload):
    text = json.dumps(payload, sort_keys=True)
    for token in (
        "software_surface_rasterizer",
        "TetMesh",
        "TetGen",
        "tetgen",
        "tet_mesh",
        "proxy",
    ):
        assert token not in text


def _assert_panel_record_native(record, expected_step):
    _assert_png_record_readable(record, expected_size=PANEL_SIZE)
    assert record["frame_index"] == expected_step
    assert record["simulation_step"] == expected_step
    assert record["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert record["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert record["renderer"]["debug_camera"] is True
    assert record["renderer"]["native_camera_registered"] is True
    assert record["hag4r_label"] == HAG4R_LABELS[record["label"]]

    camera = record["camera"]
    assert camera["label"] == record["label"]
    assert camera["model"] == "pinhole"
    assert camera["debug"] is True
    assert camera["res"] == list(PANEL_SIZE)
    assert camera["fov"] > 0
    assert camera["near"] > 0
    assert camera["far"] > camera["near"]
    _assert_vec3(camera["pos"])
    _assert_vec3(camera["lookat"])
    _assert_vec3(camera["up"])
    _assert_vec3(camera["pose"]["pos"])
    _assert_vec3(camera["pose"]["lookat"])
    _assert_vec3(camera["pose"]["up"])


def _assert_triptych_metadata_complete(metadata, expected_step):
    metadata_path = Path(metadata["metadata_path"])
    assert metadata_path.exists()
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("views", "stitched", "panel_paths", "frame_metadata"):
        assert persisted[key] == metadata[key]

    assert metadata["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["debug_camera"] is True
    assert metadata["renderer"]["camera_model"] == "pinhole"
    assert metadata["panel_order"] == list(PANEL_ORDER)
    assert set(metadata["panel_paths"]) == set(PANEL_ORDER)
    assert len(metadata["views"]) == len(PANEL_ORDER)
    assert metadata["frame_metadata"] == metadata["views"] + [metadata["stitched"]]

    for record in metadata["views"]:
        assert record["label"] in PANEL_ORDER
        assert metadata["panel_paths"][record["label"]] == record["path"]
        _assert_panel_record_native(record, expected_step)

    stitched_size = (PANEL_SIZE[0] * len(PANEL_ORDER), PANEL_SIZE[1])
    stitched = metadata["stitched"]
    _assert_png_record_readable(stitched, expected_size=stitched_size)
    assert stitched["label"] == "triptych"
    assert stitched["hag4r_label"] == "triptych"
    assert stitched["frame_index"] == expected_step
    assert stitched["simulation_step"] == expected_step
    assert stitched["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert stitched["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert stitched["source_panel_count"] == len(PANEL_ORDER)
    return persisted


def test_print_capabilities_does_not_build_scene(monkeypatch, capsys):
    from genesis.live import server

    def fail_session(*_args, **_kwargs):
        raise AssertionError("GenesisLiveSession must not be constructed for --print-capabilities")

    monkeypatch.setattr(server, "GenesisLiveSession", fail_session)

    assert server.main(["--print-capabilities"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol"] == PROTOCOL
    assert "capabilities" in payload
    assert "backend_requirements" in payload
    assert "status" not in payload
    assert "server_pid" not in payload


def test_print_capabilities_reports_missing_surface_backend(monkeypatch, capsys):
    from genesis.live import server

    def fake_capability_report(required_capabilities=()):
        required = list(required_capabilities)
        capabilities = ["volumetric_mesh_import", "pause_resume_reset"]
        return {
            "protocol": PROTOCOL,
            "capabilities": capabilities,
            "backend_requirements": {
                "surface_shell": {
                    "available": False,
                    "code": "unsupported_surface_backend",
                    "reason": "missing_uipc",
                }
            },
            "missing_required_capabilities": [
                capability for capability in required if capability not in capabilities
            ],
        }

    monkeypatch.setattr(server, "capability_report", fake_capability_report)

    rc = server.main(["--print-capabilities", "--require-capability", "surface_mesh_import"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"]["code"] == "unsupported_surface_backend"
    assert payload["missing_required_capabilities"] == ["surface_mesh_import"]


@pytest.mark.parametrize("backend", [None])
def test_surface_mesh_rejects_elastic_material_before_volumetric_path(monkeypatch, tmp_path, backend):
    import genesis.engine.entities.fem_entity as fem_entity

    def fail_mesh_to_elements(*_args, **_kwargs):
        raise AssertionError("surface_mesh + elastic must fail before mesh_to_elements")

    def fail_tet_mesh(*_args, **_kwargs):
        raise AssertionError("surface_mesh + elastic must fail before TetMesh construction")

    monkeypatch.setattr(fem_entity.eu, "mesh_to_elements", fail_mesh_to_elements)
    monkeypatch.setattr(gs.morphs, "TetMesh", fail_tet_mesh)

    with pytest.raises(GenesisLiveError) as exc_info:
        GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path, material_type="elastic")))

    assert exc_info.value.code == "invalid_scene_config"
    assert "surface_mesh morph requires" in exc_info.value.message


@pytest.mark.parametrize("backend", [None])
def test_surface_mesh_rejects_elastic_heterogeneous_before_volumetric_path(monkeypatch, tmp_path, backend):
    import genesis.engine.entities.fem_entity as fem_entity

    def fail_mesh_to_elements(*_args, **_kwargs):
        raise AssertionError("surface_mesh + elastic + heterogeneous must fail before mesh_to_elements")

    def fail_tet_mesh(*_args, **_kwargs):
        raise AssertionError("surface_mesh + elastic + heterogeneous must fail before TetMesh construction")

    monkeypatch.setattr(fem_entity.eu, "mesh_to_elements", fail_mesh_to_elements)
    monkeypatch.setattr(gs.morphs, "TetMesh", fail_tet_mesh)

    with pytest.raises(GenesisLiveError) as exc_info:
        GenesisLiveSession(
            scene_config_path=str(
                _write_surface_scene_config(tmp_path, material_type="elastic", heterogeneous=True)
            )
        )

    assert exc_info.value.code == "invalid_scene_config"
    assert "surface_mesh morph requires" in exc_info.value.message


@pytest.mark.parametrize(
    "heterogeneous",
    [
        {"material_file": "legacy_file_without_kind.npz"},
        {"kind": "tetrahedra", "material_file": "surface_materials.npz"},
        {"kind": "surface_triangles", "file": "legacy_file_key.npz"},
        {"kind": "surface_triangles", "material_file": "surface_materials.npz", "extra": True},
    ],
)
def test_surface_scene_rejects_malformed_heterogeneous_material(tmp_path, heterogeneous):
    with pytest.raises(GenesisLiveError) as exc_info:
        GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path, heterogeneous=heterogeneous)))

    assert exc_info.value.code == "invalid_surface_heterogeneous_material"


@pytest.mark.parametrize("backend", [None])
def test_surface_scene_accepts_surface_triangle_heterogeneous_material(tmp_path, backend):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path, heterogeneous=True)))
        entity = session.entities["surface"]

        assert entity.material.__class__.__name__ == "Cloth"
        assert entity.surface_heterogeneous_material is not None
        assert entity.heterogeneous_material_metadata["heterogeneous_kind"] == "surface_triangles"
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
def test_surface_mesh_rejects_source_obj_non_triangle_faces(tmp_path, backend):
    quad_obj = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
"""

    with pytest.raises(GenesisLiveError) as exc_info:
        GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path, obj_text=quad_obj)))

    assert exc_info.value.code == "invalid_scene_config"
    assert "source OBJ triangle faces" in exc_info.value.message
    assert "expected 3" in exc_info.value.details["error"]


@pytest.mark.parametrize("backend", [None])
def test_surface_scene_does_not_call_volumetric_path(monkeypatch, tmp_path, backend):
    _require_surface_backend()
    import genesis.engine.entities.fem_entity as fem_entity

    def fail_mesh_to_elements(*_args, **_kwargs):
        raise AssertionError("surface_mesh live path must not call mesh_to_elements")

    def fail_tet_mesh(*_args, **_kwargs):
        raise AssertionError("surface_mesh live path must not construct TetMesh")

    monkeypatch.setattr(fem_entity.eu, "mesh_to_elements", fail_mesh_to_elements)
    monkeypatch.setattr(gs.morphs, "TetMesh", fail_tet_mesh)

    try:
        session = GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path)))

        entity = session.entities["surface"]
        assert entity.morph.__class__.__name__ == "Mesh"
        assert entity.material.__class__.__name__ == "Cloth"
        assert session.scene.fem_solver.enable_floor is False
        assert all(candidate.morph.__class__.__name__ != "Plane" for candidate in session.scene.entities)
        assert session.scene.sim.coupler._ipc_ground_contacts == {}
        session.scene.step()
        assert session.scene.sim.coupler._ipc_ground_contacts == {}
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
@pytest.mark.parametrize("archived_palette", [False, True])
def test_surface_part_segmentation_capture_has_no_visual_ground(tmp_path, archived_palette, backend):
    del backend
    _require_surface_backend()
    try:
        session = GenesisLiveSession(
            scene_config_path=str(
                _write_surface_scene_config(
                    tmp_path,
                    heterogeneous=True,
                    part_segmentation=True,
                    archived_palette=archived_palette,
                )
            ),
            output_dir=str(tmp_path / "outputs"),
        )
        expected_palette = {
            "background": [0, 0, 0],
            "fixture": [41, 110, 255],
            "probe": [255, 89, 41],
        }
        entity = session.entities["surface"]
        assert entity._part_segmentation_config["context_palette"] == expected_palette
        assert session.scene.fem_solver.enable_floor is False
        assert all(candidate.morph.__class__.__name__ != "Plane" for candidate in session.scene.entities)
        assert session.scene.sim.coupler._ipc_ground_contacts == {}

        context = session.scene.visualizer._context
        assert not hasattr(context, "part_segmentation_floor_nodes")
        assert all("ground" not in key and "floor" not in key for key in map(str, context.seg_color_map.key_map))
        metadata = session.visual_telemetry.capture_part_segmentation_triptych(session, frame_index=0)
        assert metadata["context_palette"] == expected_palette
        assert metadata["performance"]["active_context_node_count"] == 0
        for panel_path in metadata["panel_paths"].values():
            with Image.open(panel_path) as image:
                pixels = np.asarray(image.convert("RGB"))
            assert not np.any(np.all(pixels == np.asarray([96, 96, 96], dtype=np.uint8), axis=-1))
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
def test_surface_mesh_face_order_is_preserved_for_cloth(tmp_path, backend):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_surface_scene_config(tmp_path)))
        entity = session.entities["surface"]

        np.testing.assert_array_equal(entity.elems, _SOURCE_FACES)
        np.testing.assert_array_equal(entity._surface_tri_np, _SOURCE_FACES)
        assert entity.n_elements == 2
    finally:
        gs.destroy()


def test_live_homogeneous_surface_scene_load_render_advance(tmp_path):
    _require_surface_backend()
    config_path = _write_surface_scene_config(tmp_path)
    ready_file = tmp_path / "ready.json"
    output_dir = tmp_path / "outputs"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "genesis.live.server",
            "--ready-file",
            str(ready_file),
            "--scene-config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _wait_ready(ready_file, proc)
        assert ready["protocol"] == PROTOCOL
        assert "surface_mesh_import" in ready["capabilities"]
        assert "surface_shell_diagnostics" in ready["capabilities"]
        assert "heterogeneous_surface_material_arrays" in ready["capabilities"]

        with socket.create_connection((ready["host"], ready["port"]), timeout=30.0) as sock:
            handshake = _request(sock, "session.handshake", request_id="handshake")
            assert handshake["status"] == "ok"
            handshake_data = handshake["data"]
            assert "surface_mesh_import" in handshake_data["capabilities"]
            assert "surface_shell_diagnostics" in handshake_data["capabilities"]
            assert "heterogeneous_surface_material_arrays" in handshake_data["capabilities"]
            assert handshake_data["status"]["coupler_type"] == "IPCCoupler"
            assert handshake_data["status"]["surface_scene"] is True

            geometry = _request(sock, "geometry.context.get", request_id="geometry")["data"]
            assert geometry["representation"] == "surface"
            assert geometry["primitive_kind"] == "triangle"
            assert geometry["vertex_count"] == 4
            assert geometry["triangle_count"] == 2
            assert geometry["element_count"] == 2
            _assert_no_fallback_terms(geometry)

            resumed = _request(
                sock,
                "sim.resume",
                {
                    "steps": 5,
                    "diagnostic_visual": {"mode": "rgb_triptych", "render_every_steps": 5},
                },
                request_id="resume",
            )
            assert resumed["status"] == "ok"
            data = resumed["data"]
            assert data["status"]["current_step"] == 5
            assert data["status"]["coupler_type"] == "IPCCoupler"
            visual = data["visual_telemetry"]
            assert visual["rendered"] is True
            assert visual["count"] == 1
            assert visual["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
            assert visual["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
            assert visual["renderer"]["debug_camera"] is True
            assert visual["panel_order"] == list(PANEL_ORDER)
            assert set(visual["panel_paths"]) == set(PANEL_ORDER)
            assert Path(visual["metadata_path"]).exists()
            _assert_triptych_metadata_complete(visual, expected_step=5)
            assert len(visual["frames"]) == 1
            assert visual["frames"][0]["frame_sequence_index"] == 0
            assert visual["render_every_steps"] == 5
            assert visual["steps_requested"] == 5
            _assert_no_fallback_terms(visual)
            _assert_no_fallback_terms(data)

            close_response = _request(sock, "session.close", request_id="close")
            assert close_response["status"] == "ok"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
