import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import igl
import numpy as np
import pytest
import torch
from PIL import Image

from genesis.ext import pyrender
from genesis.live import visual_telemetry
from genesis.live.session import GenesisLiveSession
from genesis.live.visual_telemetry import (
    ANCHOR_DEBUG_BOX_COLOR,
    CONTROLLER_DEBUG_BOX_COLOR,
    DEBUG_BOX_WIREFRAME_RADIUS,
    GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
    _entity_positions,
    _triptych_camera_pose,
)
from genesis.vis.rasterizer_context import RasterizerContext, SegmentationColorMap

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


def _write_scene_config(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_tets.mesh"
    igl.writeMESH(str(mesh_path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))
    config = {
        "sim_options": {"dt": 0.001},
        "entities": [
            {
                "name": "body",
                "morph": {"type": "tet_mesh", "file": str(mesh_path)},
                "material": {"type": "elastic"},
                "anchors": [
                    {"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}
                ],
            }
        ],
    }
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _write_part_segmentation_scene_config(tmp_path: Path, *, archived_palette: bool) -> Path:
    config_path = _write_scene_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels_path = tmp_path / "primitive_labels.npz"
    palette_path = tmp_path / "part_palette.npz"
    np.savez(
        labels_path,
        tet_E_nu=np.asarray([[1.0e5, 0.3], [1.0e5, 0.3]], dtype=np.float32),
        tet_density=np.asarray([1000.0, 1000.0], dtype=np.float32),
        tet_part_labels=np.asarray([0, 0], dtype=np.int32),
    )
    np.savez(palette_path, part_colors=np.asarray([[0, 255, 0]], dtype=np.uint8))
    context_palette = {
        "background": [0, 0, 0],
        "fixture": [41, 110, 255],
        "probe": [255, 89, 41],
    }
    if archived_palette:
        context_palette["ground"] = [96, 96, 96]
    config["entities"][0]["material"]["heterogeneous"] = {"file": str(labels_path)}
    config["entities"][0]["part_segmentation"] = {
        "primitive_labels_file": str(labels_path),
        "primitive_labels_key": "tet_part_labels",
        "palette_file": str(palette_path),
        "palette_key": "part_colors",
        "parts": [{"part_id": 0, "part_name": "body", "part_color_rgb": [0, 255, 0]}],
        "context_palette": context_palette,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _assert_png_record(record):
    path = Path(record["path"])
    assert path.exists()
    assert record["byte_size"] == path.stat().st_size
    assert record["byte_size"] > 0
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        assert image.size == (record["width"], record["height"])


def _assert_vec3(values):
    assert len(values) == 3
    assert all(np.isfinite(float(value)) for value in values)


def test_triptych_camera_poses_follow_physical_z_up_frame():
    bbox_min = np.array([-2.0, -1.0, -0.25], dtype=np.float32)
    bbox_max = np.array([2.0, 3.0, 1.25], dtype=np.float32)
    center = (bbox_min + bbox_max) * 0.5

    poses = {label: _triptych_camera_pose(label, bbox_min, bbox_max) for label in ("top", "northeast", "southwest")}
    positions = {label: np.asarray(pose["pos"], dtype=np.float32) for label, pose in poses.items()}

    profile_offset = positions["top"] - center
    assert profile_offset[1] > 0.0
    np.testing.assert_allclose(profile_offset[[0, 2]], (0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(poses["top"]["up"], (0.0, 0.0, -1.0), atol=1e-6)

    for label in ("northeast", "southwest"):
        assert positions[label][2] > bbox_max[2]

    northeast_xy = positions["northeast"][:2] - center[:2]
    southwest_xy = positions["southwest"][:2] - center[:2]
    np.testing.assert_allclose(northeast_xy, -southwest_xy, atol=1e-6)
    assert np.all(northeast_xy > 0.0)
    assert np.all(southwest_xy < 0.0)


def test_part_segmentation_context_boxes_reuse_stable_indexed_nodes():
    class FakeJit:
        def __init__(self):
            self.updates = []
            self.discarded = []

        def update_buffer(self, buffer_id, data, **kwargs):
            self.updates.append((buffer_id, np.asarray(data).copy(), kwargs))

        def discard_buffer_updates_for_node(self, node):
            self.discarded.append(node)

    context = RasterizerContext.__new__(RasterizerContext)
    context._scene = pyrender.Scene()
    context.jit = FakeJit()
    context.part_segmentation_context_nodes = {}
    context._part_segmentation_context_vertex_counts = {}
    context._part_segmentation_context_colors = {}
    context._part_segmentation_context_seg_keys = {}
    context.last_part_segmentation_context_update = {}
    context.segmentation_only_nodes = set()
    context.seg_node_map = {}
    context.seg_color_map = SegmentationColorMap()
    context.part_segmentation_palette_by_idxc = {}
    context.add_node = lambda mesh: context._scene.add(mesh)
    context.remove_node = context._scene.remove_node

    initial = [
        {"kind": "fixture", "controller_id": "anchor", "bounds": [[0, 0, 0], [1, 1, 1]], "color": [1, 2, 3]},
        {"kind": "probe", "controller_id": "probe", "bounds": [[2, 2, 2], [3, 3, 3]], "color": [4, 5, 6]},
    ]
    context.replace_part_segmentation_context_boxes(initial)
    nodes = dict(context.part_segmentation_context_nodes)
    old_positions = {
        key: node.mesh.primitives[0].positions.copy() for key, node in context.part_segmentation_context_nodes.items()
    }
    probe_seg_key = context._part_segmentation_context_seg_keys[("probe", "probe")]
    probe_seg_idxc = context.seg_color_map.key_map[probe_seg_key]
    assert context.last_part_segmentation_context_update == {
        "context_node_create_count": 2,
        "context_node_reuse_count": 0,
        "context_node_remove_count": 0,
        "context_position_upload_bytes": 0,
        "context_wireframe_vertex_count": 192,
        "context_wireframe_triangle_count": 288,
    }
    assert all(node.mesh.primitives[0].indices is not None for node in nodes.values())
    assert all(node.mesh.primitives[0].positions.shape == (96, 3) for node in nodes.values())
    assert all(node.mesh.primitives[0].indices.shape == (144, 3) for node in nodes.values())

    moved = [
        {**initial[0], "bounds": [[0.5, 0, 0], [1.5, 1, 1]]},
        {**initial[1], "bounds": [[2, 2.5, 2], [3, 3.5, 3]]},
    ]
    context.replace_part_segmentation_context_boxes(moved)
    assert context.part_segmentation_context_nodes == nodes
    assert context.last_part_segmentation_context_update == {
        "context_node_create_count": 0,
        "context_node_reuse_count": 2,
        "context_node_remove_count": 0,
        "context_position_upload_bytes": 2304,
        "context_wireframe_vertex_count": 192,
        "context_wireframe_triangle_count": 288,
    }
    assert len(context.jit.updates) == 2
    for key, node in nodes.items():
        assert not np.array_equal(node.mesh.primitives[0].positions, old_positions[key])
    assert {update[2]["node"] for update in context.jit.updates} == set(nodes.values())
    assert all(update[2]["buffer_name"] == "pos" for update in context.jit.updates)

    context.replace_part_segmentation_context_boxes(moved[:1])
    assert context.part_segmentation_context_nodes[("fixture", "anchor")] is nodes[("fixture", "anchor")]
    assert context.last_part_segmentation_context_update == {
        "context_node_create_count": 0,
        "context_node_reuse_count": 1,
        "context_node_remove_count": 1,
        "context_position_upload_bytes": 1152,
        "context_wireframe_vertex_count": 96,
        "context_wireframe_triangle_count": 144,
    }
    assert context.jit.discarded == [nodes[("probe", "probe")]]
    assert probe_seg_idxc not in context.part_segmentation_palette_by_idxc
    assert probe_seg_key not in context.seg_color_map.key_map


@pytest.mark.parametrize("archived_palette", [False, True])
def test_part_segmentation_is_floor_free_and_normalizes_archived_palette(tmp_path, archived_palette):
    session = GenesisLiveSession(
        scene_config_path=str(_write_part_segmentation_scene_config(tmp_path, archived_palette=archived_palette)),
        start_paused=True,
        output_dir=str(tmp_path / "outputs"),
    )
    expected_palette = {
        "background": [0, 0, 0],
        "fixture": [41, 110, 255],
        "probe": [255, 89, 41],
    }
    assert session.entities["body"]._part_segmentation_config["context_palette"] == expected_palette

    context = session.scene.visualizer._context
    assert not hasattr(context, "part_segmentation_floor_nodes")
    assert all("ground" not in key and "floor" not in key for key in map(str, context.seg_color_map.key_map))

    metadata = session.visual_telemetry.capture_part_segmentation_triptych(session, frame_index=0)
    assert metadata["context_palette"] == expected_palette
    assert metadata["performance"]["active_context_node_count"] == 1
    for panel_path in metadata["panel_paths"].values():
        with Image.open(panel_path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert not np.any(np.all(pixels == np.asarray([96, 96, 96], dtype=np.uint8), axis=-1))


def test_fem_state_default_tracks_but_telemetry_position_query_does_not(tmp_path):
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)),
        start_paused=True,
        output_dir=str(tmp_path / "outputs"),
    )
    entity = session.entities["body"]
    entity._queried_states.clear()

    tracked_state = entity.get_state()
    step = entity._sim.cur_step_global
    assert step in entity._queried_states
    assert tracked_state in entity._queried_states[step]

    entity._queried_states.clear()
    positions = _entity_positions(entity)
    assert positions.shape == (entity.n_vertices, 3)
    assert entity._queried_states.states == {}


def test_rgb_triptych_writes_panels_stitched_metadata_and_visible_overlays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    render_grad_states = []
    original_render_camera_rgb = visual_telemetry._render_camera_rgb

    def render_camera_rgb_without_grad(*args, **kwargs):
        render_grad_states.append(torch.is_grad_enabled())
        return original_render_camera_rgb(*args, **kwargs)

    monkeypatch.setattr(visual_telemetry, "_render_camera_rgb", render_camera_rgb_without_grad)
    session = GenesisLiveSession(
        scene_config_path=str(_write_scene_config(tmp_path)),
        start_paused=True,
        output_dir=str(tmp_path / "outputs"),
    )
    session.dispatch(
        "probe.action.register",
        {
            "action_id": "diag_box_action",
            "action": "box_ee_grasp_and_move",
            "entity": "body",
            "duration_steps": 1,
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
    session.dispatch("probe.apply", {"action_id": "diag_box_action"})

    metadata = session.dispatch("sim.resume", {"steps": 10, "diagnostic_visual": {"mode": "rgb_triptych"}})[
        "visual_telemetry"
    ]

    assert metadata["rendered"] is True
    assert metadata["count"] == 1
    assert metadata["render_every_steps"] == 10
    assert metadata["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["renderer"]["camera_model"] == "pinhole"
    assert metadata["renderer"]["debug_camera"] is True
    assert metadata["renderer"]["panel_camera_models"] == {
        "top": "pinhole",
        "northeast": "pinhole",
        "southwest": "pinhole",
    }
    assert metadata["renderer"]["panel_debug_cameras"] == {
        "top": True,
        "northeast": True,
        "southwest": True,
    }
    assert render_grad_states == [False, False, False]
    assert metadata["panel_order"] == ["top", "northeast", "southwest"]
    assert set(metadata["panel_paths"]) == {"top", "northeast", "southwest"}
    assert Path(metadata["metadata_path"]).exists()

    _assert_png_record(metadata["stitched"])
    assert metadata["stitched"]["width"] == 768
    assert metadata["stitched"]["height"] == 256
    assert metadata["stitched"]["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["stitched"]["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
    assert metadata["stitched"]["source_panel_count"] == 3

    assert len(metadata["views"]) == 3
    expected_hag4r_labels = {"top": "top", "northeast": "ne_3q", "southwest": "sw_3q"}
    for record in metadata["views"]:
        _assert_png_record(record)
        assert record["render_backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["backend"] == GENESIS_NATIVE_DEBUG_CAMERA_RENDERER
        assert record["renderer"]["debug_camera"] is True
        assert record["renderer"]["native_camera_registered"] is True
        assert record["hag4r_label"] == expected_hag4r_labels[record["label"]]
        camera = record["camera"]
        assert camera["label"] == record["label"]
        assert camera["model"] == "pinhole"
        assert camera["debug"] is True
        assert camera["res"] == [256, 256]
        assert camera["fov"] > 0
        assert camera["near"] > 0
        assert camera["far"] > camera["near"]
        _assert_vec3(camera["pos"])
        _assert_vec3(camera["lookat"])
        _assert_vec3(camera["up"])
        _assert_vec3(camera["pose"]["pos"])
        _assert_vec3(camera["pose"]["lookat"])
        _assert_vec3(camera["pose"]["up"])

    assert metadata["frame_metadata"] == metadata["views"] + [metadata["stitched"]]
    assert any(overlay.get("kind") == "static_anchor" for overlay in metadata["overlays"])
    assert any(
        overlay.get("kind") == "live_box_controller" and overlay.get("controller_id") == "diag_box"
        for overlay in metadata["overlays"]
    )
    markers_by_kind = {record["kind"]: record for record in metadata["debug_markers"]}
    assert markers_by_kind["static_anchor"]["color"] == list(ANCHOR_DEBUG_BOX_COLOR)
    assert markers_by_kind["live_box_controller"]["color"] == list(CONTROLLER_DEBUG_BOX_COLOR)
    live_overlay = next(
        record
        for record in metadata["overlays"]
        if record.get("kind") == "live_box_controller" and record.get("controller_id") == "diag_box"
    )
    source_box = np.asarray(live_overlay["env_local_box"], dtype=np.float32)
    displacement = np.asarray(live_overlay["displacement"], dtype=np.float32)
    expected_rendered_box = source_box + np.concatenate((displacement, displacement))
    np.testing.assert_allclose(live_overlay["rendered_env_local_box"], expected_rendered_box, atol=1e-6)
    assert float(live_overlay["moved_distance"]) > 0.0
    assert bool(live_overlay["motion_active"])
    np.testing.assert_allclose(
        np.asarray(markers_by_kind["live_box_controller"]["bounds"], dtype=np.float32),
        np.stack((expected_rendered_box[:3], expected_rendered_box[3:])),
        atol=1e-6,
    )
    static_overlay = next(record for record in metadata["overlays"] if record.get("kind") == "static_anchor")
    np.testing.assert_allclose(static_overlay["rendered_env_local_box"], static_overlay["env_local_box"], atol=1e-6)
    for marker in metadata["debug_markers"]:
        assert np.asarray(marker["bounds"], dtype=np.float32).shape == (2, 3)
        assert marker["wireframe"] is True
        assert marker["wireframe_radius"] == DEBUG_BOX_WIREFRAME_RADIUS

    serialized = Path(metadata["metadata_path"]).read_text(encoding="utf-8")
    assert GENESIS_NATIVE_DEBUG_CAMERA_RENDERER in serialized
    assert ("software" + "_surface_rasterizer") not in serialized


def test_visual_telemetry_source_has_no_software_triptych_renderer():
    live_dir = Path(__file__).resolve().parents[1] / "genesis" / "live"
    forbidden = (
        "software" + "_surface_rasterizer",
        "_" + "projection" + "_",
        "_" + "project_to_panel",
        "_" + "draw_box",
        "_" + "rasterize_surface_panel",
        "Image" + "Draw",
        "SCENE" + "_CONTEXT",
    )
    for path in live_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text
