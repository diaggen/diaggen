import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.parametrize("backend", [None])

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxEndEffectorController
from genesis.live.capabilities import capability_report, surface_backend_status
from genesis.live.protocol import GenesisLiveError
from genesis.live.session import GenesisLiveSession
from genesis.live.visual_telemetry import (
    ANCHOR_DEBUG_BOX_COLOR,
    CONTROLLER_DEBUG_BOX_COLOR,
    GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
)
from genesis.utils.misc import tensor_to_array


_SURFACE_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
f 2 4 3
"""


def _require_surface_backend():
    status = surface_backend_status()
    if not status["available"]:
        pytest.skip(f"surface backend unavailable: {status}")


def _write_surface_obj(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_triangles.obj"
    mesh_path.write_text(_SURFACE_OBJ, encoding="utf-8")
    return mesh_path


def _write_scene_config(tmp_path: Path, *, anchors=None) -> Path:
    config = {
        "backend": "cuda",
        "sim_options": {"dt": 0.001, "gravity": [0.0, 0.0, 0.0]},
        "coupler_options": {
            "contact_enable": False,
            "enable_rigid_ground_contact": False,
            "enable_rigid_rigid_contact": False,
        },
        "entities": [
            {
                "name": "surface",
                "morph": {"type": "surface_mesh", "file": str(_write_surface_obj(tmp_path))},
                "material": {
                    "type": "surface_shell",
                    "E": 10000.0,
                    "nu": 0.45,
                    "rho": 200.0,
                    "thickness": 0.001,
                    "bending_stiffness": 0.0,
                    "friction_mu": 0.1,
                },
                "anchors": list(anchors or []),
            }
        ],
    }
    config_path = tmp_path / "surface_scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _surface_entity(session: GenesisLiveSession):
    return session.entities["surface"]


def _surface_constraint_query(session: GenesisLiveSession, verts_idx):
    entity = _surface_entity(session)
    return entity.sim.coupler.query_surface_vertex_constraints(entity, verts_idx, envs_idx=None)


def _entity_positions(entity) -> np.ndarray:
    positions = tensor_to_array(entity.get_state().pos)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 3:
        positions = positions[0]
    return positions


def _dominant_pixel_count(path: str | Path, color: tuple[float, float, float, float]) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)

    target = np.asarray(color[:3], dtype=np.float32)
    dominant_channel = int(np.argmax(target))
    if dominant_channel == 2:
        mask = (
            (pixels[:, :, 2] >= 80)
            & (pixels[:, :, 2] > pixels[:, :, 0] + 20)
            & (pixels[:, :, 2] > pixels[:, :, 1] + 20)
        )
    elif dominant_channel == 0:
        mask = (
            (pixels[:, :, 0] >= 100)
            & (pixels[:, :, 0] > pixels[:, :, 2] + 30)
            & (pixels[:, :, 1] > pixels[:, :, 2] + 5)
            & (pixels[:, :, 0] > pixels[:, :, 1] + 10)
        )
    else:
        raise AssertionError(f"Unsupported overlay dominance color: {color}")
    return int(np.count_nonzero(mask))


def _assert_visible_overlay_pixels(paths: list[str | Path], color: tuple[float, float, float, float]):
    counts = [_dominant_pixel_count(path, color) for path in paths]
    assert max(counts) > 0, counts


def test_surface_position_constraint_capabilities_are_gated_by_backend_probe():
    _require_surface_backend()

    report = capability_report(
        required_capabilities=["surface_static_box_anchors", "surface_live_box_controller_actions"]
    )

    assert "surface_static_box_anchors" in report["capabilities"]
    assert "surface_live_box_controller_actions" in report["capabilities"]
    assert not report["missing_required_capabilities"]
    assert report["backend_requirements"]["surface_position_constraints"]["available"] is True


def test_surface_flat_patch_anchor_selects_expected_corner_vertices(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(
            scene_config_path=str(
                _write_scene_config(
                    tmp_path,
                    anchors=[
                        {
                            "anchor_id": "origin_corner",
                            "frame": "env_local",
                            "box": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
                        }
                    ],
                )
            )
        )

        record = session.anchor_records["surface"][0]
        assert record.selected_vertex_count == 1
        np.testing.assert_array_equal(record.selected_vertices, np.array([0], dtype=gs.np_int))

        query = _surface_constraint_query(session, [0, 1])
        np.testing.assert_array_equal(query["is_constrained"], np.array([True, False]))
        np.testing.assert_allclose(query["target_poss"][0], np.array([0.0, 0.0, 0.0]), atol=1e-8)
    finally:
        gs.destroy()


def test_surface_anchor_zero_selection_reports_box_bounds_counts_and_frame(tmp_path):
    _require_surface_backend()
    try:
        with pytest.raises(GenesisLiveError) as exc_info:
            GenesisLiveSession(
                scene_config_path=str(
                    _write_scene_config(
                        tmp_path,
                        anchors=[
                            {
                                "anchor_id": "miss",
                                "frame": "env_local",
                                "box": [10.0, 10.0, 10.0, 11.0, 11.0, 11.0],
                                "atol": 0.001,
                            }
                        ],
                    )
                )
            )
        assert exc_info.value.code == "invalid_scene_config"
        message = exc_info.value.details["error"]
        for token in (
            "source_box=",
            "env_local_box=",
            "entity_bbox_min=",
            "entity_bbox_max=",
            "vertex_count=4",
            "selected_count=0",
            "tolerance=0.001",
            "frame=env_local",
        ):
            assert token in message
    finally:
        gs.destroy()


def test_surface_box_ee_grasp_move_release_nonzero_selection(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))
        entity = _surface_entity(session)
        controller = BoxEndEffectorController(entity, controller_id="surface_box")

        state = controller.grasp([-0.01, -0.01, -0.01, 0.01, 0.01, 0.01])
        assert state.selected_vertex_count == 1
        np.testing.assert_array_equal(state.selected_vertices, np.array([0], dtype=gs.np_int))
        assert _surface_constraint_query(session, [0])["is_constrained"][0]

        controller.move_positive_y(distance_scale=1.0, duration_steps=5, speed=10.0)
        state = controller.advance_motion(steps=5)
        assert state.target_positions[0, 1] > 0.0
        query = _surface_constraint_query(session, [0])
        assert query["is_constrained"][0]
        np.testing.assert_allclose(query["target_poss"][0], state.target_positions[0], atol=1e-8)

        released = controller.release()
        assert not released.active
        assert not _surface_constraint_query(session, [0])["is_constrained"][0]
    finally:
        gs.destroy()


def test_surface_box_ee_move_changes_ipc_backend_geometry_after_step(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))
        entity = _surface_entity(session)
        controller = BoxEndEffectorController(entity, controller_id="surface_box")
        controller.grasp([-0.01, -0.01, -0.01, 0.01, 0.01, 0.01])
        controller.move_positive_y(distance_scale=1.0, duration_steps=5, speed=10.0)

        before = _entity_positions(entity).copy()
        for _ in range(10):
            controller.advance_motion(steps=1)
            session.scene.step()
        after = _entity_positions(entity)

        assert after[0, 1] > before[0, 1]
        assert not np.allclose(after[0], before[0])
    finally:
        gs.destroy()


def test_surface_release_preserves_overlapping_static_anchor(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(
            scene_config_path=str(
                _write_scene_config(
                    tmp_path,
                    anchors=[
                        {
                            "anchor_id": "origin_corner",
                            "frame": "env_local",
                            "box": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
                        }
                    ],
                )
            )
        )
        entity = _surface_entity(session)
        original = _surface_constraint_query(session, [0])
        controller = BoxEndEffectorController(entity, controller_id="surface_box")
        controller.grasp([-0.01, -0.01, -0.01, 0.01, 0.01, 0.01])
        controller.move_positive_y(distance_scale=1.0, duration_steps=5, speed=10.0)
        controller.advance_motion(steps=5)

        controller.release()

        restored = _surface_constraint_query(session, [0])
        assert restored["is_constrained"][0]
        np.testing.assert_allclose(restored["target_poss"][0], original["target_poss"][0], atol=1e-8)
    finally:
        gs.destroy()


def test_live_surface_probe_register_apply_resume_and_release(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))
        registered = session.dispatch(
            "probe.action.register",
            {
                "action": "box_ee_grasp_and_move",
                "entity": "surface",
                "duration_steps": 5,
                "controllers": [
                    {
                        "controller_id": "surface_box",
                        "aabb_box": {
                            "frame": "env_local",
                            "box": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
                        },
                        "distance_scale": 1.0,
                        "speed": 10.0,
                    }
                ],
            },
        )

        applied = session.dispatch("probe.apply", {"action_id": registered["action_id"]})
        assert applied["probe"]["selected_vertex_count"] == 1
        assert "surface_box" in session.controllers

        session.dispatch("sim.resume", {"steps": 5})
        assert session.controllers["surface_box"].state.moved_distance > 0.0

        released = session.dispatch("probe.apply", {"action": "probe_release", "controller_id": "surface_box"})
        assert released["probe"]["controller_state"]["active"] is False
        assert "surface_box" not in session.controllers
    finally:
        gs.destroy()


def test_surface_rgb_triptych_shows_static_anchor_and_live_controller_overlay_pixels(tmp_path):
    _require_surface_backend()
    anchor_box = [-0.06, -0.06, -0.06, 0.06, 0.06, 0.06]
    controller_box = [0.94, 0.94, -0.06, 1.06, 1.06, 0.06]
    try:
        session = GenesisLiveSession(
            scene_config_path=str(
                _write_scene_config(
                    tmp_path,
                    anchors=[
                        {
                            "anchor_id": "origin_corner",
                            "frame": "env_local",
                            "box": anchor_box,
                        }
                    ],
                )
            ),
            start_paused=True,
            output_dir=str(tmp_path / "outputs"),
        )

        applied = session.dispatch(
            "probe.apply",
            {
                "action": "box_ee_grasp_and_move",
                "entity": "surface",
                "duration_steps": 5,
                "controllers": [
                    {
                        "controller_id": "surface_box_overlay",
                        "aabb_box": {"frame": "env_local", "box": controller_box},
                        "distance_scale": 1.0,
                        "speed": 10.0,
                    }
                ],
            },
        )
        assert applied["probe"]["selected_vertex_count"] == 1

        visual = session.dispatch(
            "sim.resume",
            {
                "steps": 5,
                "diagnostic_visual": {"mode": "rgb_triptych", "render_every_steps": 5},
            },
        )["visual_telemetry"]

        assert visual["rendered"] is True
        assert visual["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert visual["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert visual["renderer"]["debug_camera"] is True

        overlays = visual["overlays"]
        static_overlay = next(record for record in overlays if record.get("kind") == "static_anchor")
        live_overlay = next(
            record
            for record in overlays
            if record.get("kind") == "live_box_controller"
            and record.get("controller_id") == "surface_box_overlay"
        )
        assert static_overlay["anchor_id"] == "origin_corner"
        np.testing.assert_allclose(static_overlay["env_local_box"], anchor_box, atol=1e-6)
        np.testing.assert_allclose(static_overlay["rendered_env_local_box"], anchor_box, atol=1e-6)

        source_box = np.asarray(live_overlay["env_local_box"], dtype=np.float32)
        displacement = np.asarray(live_overlay["displacement"], dtype=np.float32)
        expected_rendered_box = source_box + np.concatenate((displacement, displacement))
        np.testing.assert_allclose(source_box, np.asarray(controller_box, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(live_overlay["rendered_env_local_box"], expected_rendered_box, atol=1e-6)
        assert float(live_overlay["moved_distance"]) > 0.0

        markers_by_kind = {record["kind"]: record for record in visual["debug_markers"]}
        assert markers_by_kind["static_anchor"]["color"] == list(ANCHOR_DEBUG_BOX_COLOR)
        assert markers_by_kind["live_box_controller"]["color"] == list(CONTROLLER_DEBUG_BOX_COLOR)
        assert markers_by_kind["live_box_controller"]["controller_id"] == "surface_box_overlay"
        np.testing.assert_allclose(
            markers_by_kind["static_anchor"]["rendered_env_local_box"],
            anchor_box,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            markers_by_kind["live_box_controller"]["rendered_env_local_box"],
            expected_rendered_box,
            atol=1e-6,
        )

        # RGB triptych evidence proves geometry, native cameras, anchors, and live controller overlays.
        # Surface material evidence remains in fused observation and heterogeneous metadata, not RGB face colors.
        # Future stiffness visualization needs native per-face color rendering backed by tri_E_nu[:, 0].
        panel_paths = [record["path"] for record in visual["views"]]
        stitched_path = visual["stitched"]["path"]
        _assert_visible_overlay_pixels(panel_paths, ANCHOR_DEBUG_BOX_COLOR)
        _assert_visible_overlay_pixels(panel_paths, CONTROLLER_DEBUG_BOX_COLOR)
        assert _dominant_pixel_count(stitched_path, ANCHOR_DEBUG_BOX_COLOR) > 0
        assert _dominant_pixel_count(stitched_path, CONTROLLER_DEBUG_BOX_COLOR) > 0
    finally:
        gs.destroy()


def test_live_surface_probe_zero_selection_returns_protocol_error(tmp_path):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))
        response = session.handle_request(
            {
                "request_id": "miss",
                "method": "probe.apply",
                "params": {
                    "action": "box_ee_grasp_and_move",
                    "entity": "surface",
                    "duration_steps": 5,
                    "controllers": [
                        {
                            "controller_id": "surface_box",
                            "aabb_box": {
                                "frame": "env_local",
                                "box": [10.0, 10.0, 10.0, 11.0, 11.0, 11.0],
                                "atol": 0.001,
                            },
                            "distance_scale": 1.0,
                        }
                    ],
                },
            }
        )
        assert response["status"] == "error"
        assert response["error"]["code"] == "probe_selection_failed"
        message = response["error"]["message"]
        for token in (
            "source_box=",
            "env_local_box=",
            "entity_bbox_min=",
            "entity_bbox_max=",
            "vertex_count=4",
            "selected_count=0",
            "tolerance=0.001",
            "frame=env_local",
        ):
            assert token in message
    finally:
        gs.destroy()
