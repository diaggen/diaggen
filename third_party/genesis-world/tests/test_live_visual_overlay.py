from __future__ import annotations

import json
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

import genesis.live.session as session_module
import genesis.live.visual_overlay as visual_overlay_module
import genesis.vis.rasterizer_context as rasterizer_context_module
from genesis.live.protocol import (
    TEXTURED_VISUAL_OVERLAY_CAPABILITIES,
    GenesisLiveError,
    capabilities_for_report,
)
from genesis.live.session import GenesisLiveSession, VisualOverlayRecord
from genesis.live.visual_overlay import (
    VISUAL_OVERLAY_ASSET_ROOT_ENV,
    VisualOverlayAssets,
    load_visual_overlay_assets,
    resolve_visual_overlay_asset_path,
    validate_visual_overlay_spec,
)
from genesis.vis.rasterizer_context import RasterizerContext


def _overlay_entity(**overlay_updates):
    overlay = {
        "schema_version": "hag4r-genesis-visual-overlay-v1",
        "mesh_file": "outputs/asset/appearance/visual_mesh.glb",
        "binding_file": "outputs/asset/appearance/visual_to_physics.npz",
        "binding_key": "physics_vertex_indices",
        "hide_physics_rgb": True,
    }
    overlay.update(overlay_updates)
    return {
        "name": "body",
        "morph": {"type": "tet_mesh", "file": "body.mesh"},
        "material": {"type": "elastic"},
        "visual_overlay": overlay,
    }


def _write_tiny_overlay_bundle(root):
    output = root / "outputs" / "asset" / "appearance"
    output.mkdir(parents=True)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (2, 2), (20, 40, 60)),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="visual_mesh", node_name="visual_mesh")
    mesh_path = output / "visual_mesh.glb"
    mesh_path.write_bytes(scene.export(file_type="glb"))
    binding_path = output / "visual_to_physics.npz"
    np.savez_compressed(
        binding_path,
        visual_rest_vertices_m=vertices,
        visual_faces=faces,
        visual_uv=uv,
        surface_vertex_to_physics_vertex=np.arange(3, dtype=np.int64),
        visual_vertex_to_surface_vertex=np.arange(3, dtype=np.int64),
        physics_vertex_indices=np.arange(3, dtype=np.int64),
        boundary_face_tet_indices=np.array([0], dtype=np.int64),
        boundary_face_part_ids=np.array([0], dtype=np.int32),
    )
    return mesh_path, binding_path, vertices, faces


def _binding_arrays(binding_path):
    with np.load(binding_path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _rewrite_glb_json(mesh_path, mutator):
    data = mesh_path.read_bytes()
    cursor = 12
    chunks = []
    mutated = False
    while cursor < len(data):
        chunk_length = int.from_bytes(data[cursor : cursor + 4], "little")
        chunk_type = data[cursor + 4 : cursor + 8]
        cursor += 8
        chunk = data[cursor : cursor + chunk_length]
        cursor += chunk_length
        if chunk_type == b"JSON":
            document = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            mutator(document)
            chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
            chunk += b" " * ((-len(chunk)) % 4)
            mutated = True
        chunks.append((chunk_type, chunk))
    assert mutated
    body = b"".join(
        len(chunk).to_bytes(4, "little") + chunk_type + chunk
        for chunk_type, chunk in chunks
    )
    mesh_path.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12 + len(body)).to_bytes(4, "little") + body)


def test_visual_overlay_schema_is_exact_and_owner_is_untransformed_tet():
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=3)
    assert spec is not None
    assert spec.entity_index == 3
    assert spec.entity_name == "body"

    invalid_entities = [
        _overlay_entity(extra=True),
        _overlay_entity(schema_version="wrong"),
        _overlay_entity(binding_key="wrong"),
        _overlay_entity(hide_physics_rgb=1),
        {**_overlay_entity(), "morph": {"type": "surface_mesh", "file": "body.obj"}},
        {**_overlay_entity(), "material": {"type": "cloth"}},
        {**_overlay_entity(), "morph": {"type": "tet_mesh", "file": "body.mesh", "scale": [1, 1, 1]}},
        {**_overlay_entity(), "morph": {"type": "tet_mesh", "file": "body.mesh", "pos": [0, 0, 0]}},
    ]
    for entity in invalid_entities:
        with pytest.raises(GenesisLiveError) as exc_info:
            validate_visual_overlay_spec(entity, entity_index=0)
        assert exc_info.value.code == "invalid_visual_overlay"


def test_diagnostic_overlay_rejection_precedes_all_contract_and_asset_io(monkeypatch):
    session = object.__new__(GenesisLiveSession)
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("lower-priority validation must not run")

    monkeypatch.setattr(session_module, "validate_visual_overlay_spec", unexpected)
    session._validate_part_segmentation_contract = unexpected
    config = {
        "agentic_diagnostics": {},
        "entities": [
            {
                **_overlay_entity(
                    mesh_file="outputs/missing.glb",
                    binding_file="outputs/missing.npz",
                ),
                "part_segmentation": {
                    "primitive_labels_file": "missing-labels.npz",
                    "palette_file": "missing-palette.npz",
                },
            }
        ],
    }
    with pytest.raises(GenesisLiveError) as exc_info:
        session._validate_scene_config_before_build(config)
    assert exc_info.value.code == "texture_overlay_not_supported_in_diagnostics"
    assert calls == []


def test_visual_overlay_asset_root_resolution_is_strict(tmp_path):
    mesh_path, _, _, _ = _write_tiny_overlay_bundle(tmp_path)
    environment = {VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())}
    assert (
        resolve_visual_overlay_asset_path(
            "outputs/asset/appearance/visual_mesh.glb",
            field="mesh_file",
            environ=environment,
        )
        == mesh_path.resolve()
    )
    for configured in (
        str(mesh_path.resolve()),
        "../visual_mesh.glb",
        "outputs/../visual_mesh.glb",
        "outputs/./visual_mesh.glb",
        "output/visual_mesh.glb",
        "outputs//visual_mesh.glb",
    ):
        with pytest.raises(GenesisLiveError, match="visual_overlay"):
            resolve_visual_overlay_asset_path(configured, field="mesh_file", environ=environment)
    with pytest.raises(GenesisLiveError):
        resolve_visual_overlay_asset_path(
            "outputs/asset/appearance/visual_mesh.glb",
            field="mesh_file",
            environ={},
        )
    with pytest.raises(GenesisLiveError):
        resolve_visual_overlay_asset_path(
            "outputs/asset/appearance/visual_mesh.glb",
            field="mesh_file",
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: "."},
        )
    file_root = tmp_path / "not-a-root"
    file_root.write_bytes(b"x")
    with pytest.raises(GenesisLiveError):
        resolve_visual_overlay_asset_path(
            "outputs/asset/appearance/visual_mesh.glb",
            field="mesh_file",
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(file_root.resolve())},
        )
    with pytest.raises(GenesisLiveError):
        resolve_visual_overlay_asset_path(
            "outputs/missing.glb",
            field="mesh_file",
            environ=environment,
        )
    directory_target = tmp_path / "outputs" / "directory.glb"
    directory_target.mkdir()
    with pytest.raises(GenesisLiveError, match="regular file"):
        resolve_visual_overlay_asset_path(
            "outputs/directory.glb",
            field="mesh_file",
            environ=environment,
        )

    outside = tmp_path.parent / f"{tmp_path.name}-outside.glb"
    outside.write_bytes(b"x")
    symlink = tmp_path / "outputs" / "escape.glb"
    symlink.symlink_to(outside)
    with pytest.raises(GenesisLiveError, match="escapes"):
        resolve_visual_overlay_asset_path(
            "outputs/escape.glb",
            field="mesh_file",
            environ=environment,
        )


def test_visual_overlay_asset_root_accepts_symlinked_outputs_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    storage_root = tmp_path / "storage"
    (storage_root / "outputs").mkdir(parents=True)
    (repo_root / "outputs").symlink_to(storage_root / "outputs", target_is_directory=True)
    mesh_path, _, _, _ = _write_tiny_overlay_bundle(repo_root)
    environment = {VISUAL_OVERLAY_ASSET_ROOT_ENV: str(repo_root.absolute())}

    assert (
        resolve_visual_overlay_asset_path(
            "outputs/asset/appearance/visual_mesh.glb",
            field="mesh_file",
            environ=environment,
        )
        == mesh_path.resolve()
    )

    outside = storage_root / "outside.glb"
    outside.write_bytes(b"x")
    (storage_root / "outputs" / "escape.glb").symlink_to(outside)
    with pytest.raises(GenesisLiveError, match="escapes"):
        resolve_visual_overlay_asset_path(
            "outputs/escape.glb",
            field="mesh_file",
            environ=environment,
        )


def test_visual_overlay_loader_revalidates_glb_and_exact_binding(tmp_path):
    _, _, vertices, faces = _write_tiny_overlay_bundle(tmp_path)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None
    assets = load_visual_overlay_assets(
        spec,
        environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
    )
    assert isinstance(assets.mesh, trimesh.Trimesh)
    assert np.allclose(assets.visual_rest_vertices_m, vertices, atol=1.0e-6, rtol=0.0)
    assert np.array_equal(assets.visual_faces, faces)
    assert assets.visual_rest_vertices_m.flags.c_contiguous
    assert not assets.visual_rest_vertices_m.flags.writeable
    assert assets.physics_vertex_indices.dtype == np.int64
    assert not assets.physics_vertex_indices.flags.writeable

    binding_path = tmp_path / "outputs" / "asset" / "appearance" / "visual_to_physics.npz"
    np.savez_compressed(
        binding_path,
        visual_rest_vertices_m=vertices,
        visual_faces=faces,
        visual_uv=np.zeros((3, 2), dtype=np.float32),
        surface_vertex_to_physics_vertex=np.arange(3, dtype=np.int64),
        visual_vertex_to_surface_vertex=np.arange(3, dtype=np.int64),
        physics_vertex_indices=np.array([0, 2, 1], dtype=np.int64),
        boundary_face_tet_indices=np.array([0], dtype=np.int64),
        boundary_face_part_ids=np.array([0], dtype=np.int32),
    )
    with pytest.raises(GenesisLiveError) as exc_info:
        load_visual_overlay_assets(
            spec,
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
        )
    assert exc_info.value.code == "invalid_visual_overlay"


@pytest.mark.parametrize(
    "case",
    (
        "missing_key",
        "extra_key",
        "wrong_dtype",
        "wrong_shape",
        "object_array",
        "nonfinite_rest",
        "nonfinite_uv",
        "bad_face_index",
    ),
)
def test_visual_overlay_binding_negative_matrix(tmp_path, case):
    _, binding_path, _, _ = _write_tiny_overlay_bundle(tmp_path)
    arrays = _binding_arrays(binding_path)
    if case == "missing_key":
        arrays.pop("boundary_face_part_ids")
    elif case == "extra_key":
        arrays["unexpected"] = np.array([1], dtype=np.int32)
    elif case == "wrong_dtype":
        arrays["physics_vertex_indices"] = arrays["physics_vertex_indices"].astype(np.int32)
    elif case == "wrong_shape":
        arrays["visual_uv"] = arrays["visual_uv"].reshape(-1)
    elif case == "object_array":
        arrays["visual_rest_vertices_m"] = arrays["visual_rest_vertices_m"].astype(object)
    elif case == "nonfinite_rest":
        arrays["visual_rest_vertices_m"][0, 0] = np.nan
    elif case == "nonfinite_uv":
        arrays["visual_uv"][0, 0] = np.inf
    elif case == "bad_face_index":
        arrays["visual_faces"][0, 2] = len(arrays["visual_rest_vertices_m"])
    else:
        raise AssertionError(case)
    np.savez_compressed(binding_path, **arrays)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None

    with pytest.raises(GenesisLiveError) as exc_info:
        load_visual_overlay_assets(
            spec,
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
        )

    assert exc_info.value.code == "invalid_visual_overlay"


@pytest.mark.parametrize("case", ("vertex_count", "face_order"))
def test_visual_overlay_loader_rejects_glb_binding_topology_mismatch(tmp_path, case):
    _, binding_path, _, _ = _write_tiny_overlay_bundle(tmp_path)
    arrays = _binding_arrays(binding_path)
    if case == "vertex_count":
        arrays["visual_rest_vertices_m"] = np.concatenate(
            (arrays["visual_rest_vertices_m"], np.array([[2.0, 2.0, 2.0]], dtype=np.float32))
        )
        arrays["visual_uv"] = np.concatenate(
            (arrays["visual_uv"], np.array([[0.5, 0.5]], dtype=np.float32))
        )
        arrays["surface_vertex_to_physics_vertex"] = np.arange(4, dtype=np.int64)
        arrays["visual_vertex_to_surface_vertex"] = np.arange(4, dtype=np.int64)
        arrays["physics_vertex_indices"] = np.arange(4, dtype=np.int64)
    elif case == "face_order":
        arrays["visual_faces"] = arrays["visual_faces"][:, ::-1].copy()
    else:
        raise AssertionError(case)
    np.savez_compressed(binding_path, **arrays)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None

    with pytest.raises(GenesisLiveError) as exc_info:
        load_visual_overlay_assets(
            spec,
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
        )

    assert exc_info.value.code == "invalid_visual_overlay"


@pytest.mark.parametrize(
    ("rest_offset_m", "passes"),
    (
        (0.999e-6, True),
        (1.0e-6, False),
        (1.001e-6, False),
    ),
)
def test_visual_overlay_loader_uses_strict_rest_error_boundary(
    monkeypatch,
    tmp_path,
    rest_offset_m,
    passes,
):
    _write_tiny_overlay_bundle(tmp_path)
    original_load = visual_overlay_module.trimesh.load

    def offset_load(*args, **kwargs):
        mesh = original_load(*args, **kwargs)
        shifted = np.array(mesh.vertices, dtype=np.float64, copy=True)
        shifted[0, 0] = rest_offset_m
        mesh.vertices = shifted
        return mesh

    monkeypatch.setattr(visual_overlay_module.trimesh, "load", offset_load)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None
    if passes:
        assets = load_visual_overlay_assets(
            spec,
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
        )
        assert assets.mesh.vertices[0, 0] == pytest.approx(rest_offset_m)
    else:
        with pytest.raises(GenesisLiveError) as exc_info:
            load_visual_overlay_assets(
                spec,
                environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
            )
        assert exc_info.value.code == "invalid_visual_overlay"


@pytest.mark.parametrize(
    "case",
    ("multi_mesh", "multi_primitive", "untextured", "null_texture_record"),
)
def test_visual_overlay_glb_container_negative_matrix(tmp_path, case):
    mesh_path, _, _, _ = _write_tiny_overlay_bundle(tmp_path)

    def mutate(document):
        if case == "multi_mesh":
            document["meshes"].append(deepcopy(document["meshes"][0]))
        elif case == "multi_primitive":
            document["meshes"][0]["primitives"].append(
                deepcopy(document["meshes"][0]["primitives"][0])
            )
        elif case == "untextured":
            material_index = document["meshes"][0]["primitives"][0]["material"]
            document["materials"][material_index]["pbrMetallicRoughness"].pop(
                "baseColorTexture"
            )
        elif case == "null_texture_record":
            document["textures"] = [None]
        else:
            raise AssertionError(case)

    _rewrite_glb_json(mesh_path, mutate)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None

    with pytest.raises(GenesisLiveError) as exc_info:
        load_visual_overlay_assets(
            spec,
            environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
        )

    assert exc_info.value.code == "invalid_visual_overlay"


class _FakeVisualEntity:
    def __init__(self):
        self.set_calls = []
        self.vertices = None

    def set_vverts(self, vertices):
        assert vertices is not None
        self.vertices = np.array(vertices, copy=True)
        self.set_calls.append(self.vertices)

    def get_vverts(self):
        return self.vertices[None, ...]


def test_visual_overlay_sync_uses_exact_indices_and_status_is_conditional():
    physical = SimpleNamespace(
        active=False,
        init_positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    visual = _FakeVisualEntity()
    session = object.__new__(GenesisLiveSession)
    session.current_step = 0
    session.visual_overlays = {
        "body": VisualOverlayRecord(
            physical_entity_name="body",
            physical_entity=physical,
            visual_entity=visual,
            physics_vertex_indices=np.array([0, 1, 1], dtype=np.int64),
            visual_rest_vertices_m=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        )
    }
    session._visual_overlay_last_sync_step = None
    session._visual_overlay_last_sync_seconds = None
    session._sync_visual_overlays(checked_at="after_build", require_rest_alignment=True)
    assert len(visual.set_calls) == 1
    assert np.array_equal(visual.vertices[1], visual.vertices[2])
    assert session._visual_overlay_last_sync_step == 0
    assert session._visual_overlay_last_sync_seconds >= 0.0

    session.scene = None
    session.session_id = "session"
    session.paused = True
    session.running = False
    session.last_request_id = None
    session.last_frame_index = None
    session.heartbeat_timestamp = 0.0
    session.fatal_error = None
    session.controllers = {}
    session._scene_config = {"entities": []}
    status_with_overlay = session.status()
    assert set(status_with_overlay).issuperset(
        {
            "visual_overlay_count",
            "visual_overlay_last_sync_step",
            "visual_overlay_last_sync_seconds",
        }
    )
    session.visual_overlays = {}
    status_without_overlay = session.status()
    assert not any(key.startswith("visual_overlay_") for key in status_without_overlay)


def test_visual_overlay_trace_reports_exact_binding_and_seam_duplicates():
    physical = SimpleNamespace(
        active=False,
        init_positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        ),
    )
    visual = _FakeVisualEntity()
    visual.set_vverts(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
    )
    session = object.__new__(GenesisLiveSession)
    session.current_step = 7
    session.visual_overlays = {
        "body": VisualOverlayRecord(
            physical_entity_name="body",
            physical_entity=physical,
            visual_entity=visual,
            physics_vertex_indices=np.array([0, 1, 1], dtype=np.int64),
            visual_rest_vertices_m=np.array(
                [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                dtype=np.float32,
            ),
        )
    }

    trace = session.visual_overlay_trace({})

    assert trace["schema_version"] == "genesis-live-visual-overlay-trace-v1"
    assert trace["current_step"] == 7
    assert trace["max_binding_error_m"] == 0.0
    assert trace["duplicate_physics_vertex_count"] == 1
    assert trace["duplicate_visual_vertex_count"] == 2
    assert trace["max_seam_duplicate_error_m"] == 0.0
    assert trace["duplicate_samples"] == [
        {
            "physics_vertex_index": 1,
            "visual_vertex_indices": [1, 2],
            "max_pairwise_error_m": 0.0,
        }
    ]


def test_depth_and_normal_triptychs_require_visual_overlay():
    session = object.__new__(GenesisLiveSession)
    session.entities = {}
    session.visual_overlays = {}
    for mode in ("depth", "depth_triptych", "normal", "normal_triptych"):
        with pytest.raises(GenesisLiveError) as exc_info:
            session._validate_visual_request({"mode": mode, "render_every_steps": 1})
        assert exc_info.value.code == "unsupported_visual_mode"

    session.visual_overlays = {"body": object()}
    for mode in ("depth", "depth_triptych", "normal", "normal_triptych"):
        session._validate_visual_request({"mode": mode, "render_every_steps": 1})


@pytest.mark.parametrize(
    "binding",
    (
        np.array([0, 1, 1], dtype=np.int32),
        np.array([0, -1, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    ),
    ids=("wrong_dtype", "negative", "length_mismatch", "physical_out_of_range"),
)
def test_visual_overlay_initial_sync_rejects_malformed_or_out_of_range_binding(binding):
    physical = SimpleNamespace(
        active=False,
        init_positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    session = object.__new__(GenesisLiveSession)
    session.current_step = 0
    session.visual_overlays = {
        "body": VisualOverlayRecord(
            physical_entity_name="body",
            physical_entity=physical,
            visual_entity=_FakeVisualEntity(),
            physics_vertex_indices=binding,
            visual_rest_vertices_m=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        )
    }

    with pytest.raises(GenesisLiveError) as exc_info:
        session._sync_visual_overlays(
            checked_at="after_build",
            require_rest_alignment=True,
        )

    assert exc_info.value.code == "invalid_visual_overlay"


def test_resume_orders_finite_validation_then_sync_then_capture():
    events = []
    session = object.__new__(GenesisLiveSession)
    session.controllers = {}
    session.scene = SimpleNamespace(step=lambda: events.append("step"))
    session.current_step = 0
    session.paused = True
    session.running = False
    session.fatal_error = None
    session._validate_visual_request = MethodType(lambda self, visual: None, session)
    session._visual_render_every_steps = MethodType(lambda self, visual: 1, session)
    session._validate_fem_state = MethodType(
        lambda self, *, checked_at: events.append(("finite", checked_at)),
        session,
    )
    session._sync_visual_overlays = MethodType(
        lambda self, *, checked_at: events.append(("sync", checked_at)),
        session,
    )
    session._capture_visual_request = MethodType(
        lambda self, *args, **kwargs: events.append("capture") or {},
        session,
    )
    session._visual_result_from_frames = MethodType(lambda self, *args, **kwargs: {}, session)
    session.status = MethodType(lambda self: {}, session)
    session.resume({"steps": 1, "diagnostic_visual": {"mode": "rgb_triptych"}})
    assert events == [
        "step",
        ("finite", "after_step"),
        ("sync", "after_step"),
        "capture",
    ]


def test_mocked_build_reuses_loaded_trimesh_and_keeps_overlay_out_of_physics(monkeypatch, tmp_path):
    _write_tiny_overlay_bundle(tmp_path)
    spec = validate_visual_overlay_spec(_overlay_entity(), entity_index=0)
    assert spec is not None
    assets = load_visual_overlay_assets(
        spec,
        environ={VISUAL_OVERLAY_ASSET_ROOT_ENV: str(tmp_path.resolve())},
    )
    meshset_calls = []
    kinematic_calls = []
    loaded_assets = []

    def fake_meshset(**kwargs):
        meshset_calls.append(kwargs)
        return SimpleNamespace(kind="visual_overlay_morph")

    def fake_kinematic(**kwargs):
        kinematic_calls.append(kwargs)
        return SimpleNamespace(kind="kinematic")

    class FakeScene:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.add_calls = []
            self.build_calls = 0
            self.destroy_calls = 0
            self.physical = SimpleNamespace(
                active=False,
                init_positions=np.array(assets.visual_rest_vertices_m, copy=True),
            )
            self.visual = _FakeVisualEntity()

        def add_entity(self, *, morph, material, name):
            self.add_calls.append((morph, material, name))
            return self.physical if morph == "physical_morph" else self.visual

        def build(self):
            self.build_calls += 1

        def destroy(self):
            self.destroy_calls += 1

    def fake_load(loaded_spec):
        loaded = VisualOverlayAssets(
            spec=loaded_spec,
            mesh=assets.mesh.copy(),
            visual_rest_vertices_m=assets.visual_rest_vertices_m,
            visual_faces=assets.visual_faces,
            physics_vertex_indices=assets.physics_vertex_indices,
        )
        loaded_assets.append(loaded)
        return loaded

    monkeypatch.setattr(session_module, "load_visual_overlay_assets", fake_load)
    monkeypatch.setattr(session_module.gs, "Scene", FakeScene)
    monkeypatch.setattr(session_module.gs.morphs, "MeshSet", fake_meshset)
    monkeypatch.setattr(session_module.gs.materials, "Kinematic", fake_kinematic)

    session = object.__new__(GenesisLiveSession)
    session._scene_config = {
        "entities": [_overlay_entity()],
    }
    session.scene = None
    session.controllers = {}
    session.actions = SimpleNamespace(clear=lambda: None)
    session.anchor_records = {}
    session.visual_telemetry = SimpleNamespace(
        reset_triptych_cameras=lambda: None,
        register_triptych_cameras=lambda current_session: None,
    )
    session.start_paused = True
    session._validated_visual_overlay_specs = {}
    session._ensure_genesis_initialized = MethodType(
        lambda self, backend, *, requires_surface_backend: None,
        session,
    )
    session._build_morph = MethodType(lambda self, config: "physical_morph", session)
    session._build_material = MethodType(lambda self, config: "physical_material", session)

    session.build_scene()

    assert len(meshset_calls) == 1
    assert meshset_calls[0] == {
        "files": (loaded_assets[0].mesh,),
        "fixed": True,
        "enable_custom_vverts": True,
    }
    assert meshset_calls[0]["files"][0] is loaded_assets[0].mesh
    assert kinematic_calls == [{"use_visual_raycasting": False}]
    first_scene = session.scene
    first_physical = first_scene.physical
    first_visual = first_scene.visual
    assert session.entities == {"body": first_physical}
    assert set(session.visual_overlays) == {"body"}
    assert session.visual_overlays["body"].visual_entity is first_visual
    assert first_physical._rgb_visualization_disabled is True
    assert len(first_visual.set_calls) == 1
    assert session.scene.build_calls == 1

    session.build_scene()

    assert len(loaded_assets) == 2
    assert len(meshset_calls) == 2
    assert meshset_calls[1]["files"][0] is loaded_assets[1].mesh
    assert loaded_assets[1].mesh is not loaded_assets[0].mesh
    assert first_scene.destroy_calls == 1
    assert session.scene is not first_scene
    assert session.visual_overlays["body"].visual_entity is session.scene.visual
    assert session.visual_overlays["body"].visual_entity is not first_visual
    assert len(session.scene.visual.set_calls) == 1


def test_no_overlay_build_preserves_legacy_order_and_does_not_read_fem_state(monkeypatch):
    events = []
    physical = SimpleNamespace()

    class FakeScene:
        def __init__(self, **kwargs):
            self.sim = None
            events.append("scene_init")

        def add_entity(self, *, morph, material, name):
            events.append("physical_add")
            return physical

        def build(self):
            events.append("scene_build")

    monkeypatch.setattr(session_module.gs, "Scene", FakeScene)
    monkeypatch.setattr(
        session_module,
        "apply_static_box_anchors",
        lambda entity, anchors, frame: events.append("anchors") or {"count": 1},
    )
    session = object.__new__(GenesisLiveSession)
    session._scene_config = {
        "entities": [
            {
                "name": "body",
                "morph": {"type": "tet_mesh", "file": "body.mesh"},
                "material": {"type": "elastic"},
                "anchors": [{"anchor_id": "pin"}],
            }
        ],
    }
    session.scene = None
    session.controllers = {}
    session.actions = SimpleNamespace(clear=lambda: None)
    session.anchor_records = {}
    session.visual_telemetry = SimpleNamespace(
        reset_triptych_cameras=lambda: events.append("telemetry_reset"),
        register_triptych_cameras=lambda current_session: events.append("camera_register"),
    )
    session.start_paused = True
    session.session_id = "legacy"
    session.last_request_id = None
    session.fatal_error = None
    session._validated_visual_overlay_specs = {}
    session._ensure_genesis_initialized = MethodType(
        lambda self, backend, *, requires_surface_backend: None,
        session,
    )
    session._build_morph = MethodType(lambda self, config: "physical_morph", session)
    session._build_material = MethodType(lambda self, config: "physical_material", session)
    session._validate_fem_state = MethodType(
        lambda self, *, checked_at: pytest.fail(
            "legacy no-overlay build must not validate or read FEM state"
        ),
        session,
    )

    session.build_scene()

    assert events == [
        "telemetry_reset",
        "scene_init",
        "physical_add",
        "camera_register",
        "scene_build",
        "anchors",
    ]
    assert session.entities == {"body": physical}
    assert session.visual_overlays == {}
    assert session.anchor_records == {"body": {"count": 1}}
    assert not any(key.startswith("visual_overlay_") for key in session.status())


def test_hidden_physics_rgb_update_skips_missing_rasterizer_node():
    entity = SimpleNamespace(
        _rgb_visualization_disabled=True,
        surface=SimpleNamespace(vis_mode="visual"),
    )

    class _RenderState:
        def to_numpy(self, dtype):
            return np.empty((0, 1, 3), dtype=dtype)

    context = SimpleNamespace(
        sim=SimpleNamespace(
            fem_solver=SimpleNamespace(
                is_active=True,
                entities=[entity],
                get_state_render=lambda substep: (_RenderState(), None, None),
            ),
            cur_substep_local=0,
        ),
    )
    RasterizerContext.update_fem(context)
    assert context.last_render_update_stats["active_rgb_node_count"] == 0
    assert context.last_render_update_stats["position_upload_bytes"] == 0


def test_hidden_physics_rgb_build_preserves_indexed_part_nodes(monkeypatch):
    vertices = np.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    entity = SimpleNamespace(
        uid=7,
        v_start=0,
        n_vertices=3,
        s_start=0,
        n_surfaces=1,
        n_surface_vertices=3,
        surface=SimpleNamespace(vis_mode="visual", smooth=True, double_sided=True),
        _rgb_visualization_disabled=True,
        _part_segmentation_config={
            "parts": [{"part_id": 0, "part_color_rgb": [10, 20, 30]}],
        },
        surface_primitive_part_labels=np.array([0], dtype=np.int64),
    )
    primitive = SimpleNamespace(
        positions=np.empty((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int64),
        vertex_mapping=None,
    )
    part_mesh = SimpleNamespace(primitives=[primitive])

    class _PartNode:
        def __init__(self, mesh):
            self.mesh = mesh

    part_node = _PartNode(part_mesh)
    monkeypatch.setattr(rasterizer_context_module, "qd_to_numpy", lambda value: value)
    monkeypatch.setattr(
        rasterizer_context_module.pyrender.Mesh,
        "from_trimesh",
        lambda *args, **kwargs: part_mesh,
    )
    added_static = []
    updated = []
    context = SimpleNamespace(
        sim=SimpleNamespace(
            fem_solver=SimpleNamespace(
                is_active=True,
                entities=[entity],
                get_state_render=lambda substep: (vertices, triangles, uvs),
            ),
            cur_substep_local=0,
        ),
        rendered_envs_idx=[0],
        _fem_surface_vertex_indices={},
        static_nodes={},
        rgb_only_nodes=set(),
        part_segmentation_nodes={},
        _part_segmentation_vertex_indices={},
        segmentation_only_nodes=set(),
        part_segmentation_indexed_counts={},
        part_segmentation_palette_by_idxc={},
        seg_color_map=SimpleNamespace(key_map={("hag4r_part", 7, 0): 11}),
        add_static_node=lambda *args, **kwargs: added_static.append((args, kwargs)),
        remove_node_seg=lambda node: pytest.fail("hidden RGB node must not exist"),
        add_node=lambda mesh: part_node,
        create_node_seg=lambda key, node: None,
    )

    RasterizerContext.on_fem(context)

    assert added_static == []
    assert context.static_nodes == {}
    assert context._fem_surface_vertex_indices == {}
    assert context.part_segmentation_nodes[(0, 7, 0)] is part_node
    assert part_node in context.segmentation_only_nodes
    assert np.array_equal(context.part_segmentation_palette_by_idxc[11], [10, 20, 30])

    class _RenderState:
        def to_numpy(self, dtype):
            return vertices.astype(dtype)

    context.sim.fem_solver.get_state_render = lambda substep: (_RenderState(), None, None)
    context._scene = SimpleNamespace(
        reorder_vertices=lambda node, values: values,
        get_buffer_id=lambda node, field: 1,
    )
    context.jit = SimpleNamespace(
        update_buffer=lambda buffer_id, data, **kwargs: updated.append(np.array(data, copy=True))
    )
    RasterizerContext.update_fem(context, render_pass="part_segmentation")
    assert len(updated) == 1
    assert updated[0].shape == (3, 3)
    assert context.last_part_segmentation_update["active_part_node_count"] == 1
    assert context.last_part_segmentation_update["rgb_update_node_count"] == 0


def test_flag_false_preserves_normal_physics_rgb_build_and_update(monkeypatch):
    vertices = np.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    entity = SimpleNamespace(
        uid=8,
        v_start=0,
        n_vertices=3,
        s_start=0,
        n_surfaces=1,
        n_surface_vertices=3,
        surface=SimpleNamespace(vis_mode="visual", smooth=True, double_sided=True),
        _rgb_visualization_disabled=False,
    )
    primitive = SimpleNamespace(
        positions=np.empty((3, 3), dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int64),
        vertex_mapping=None,
    )
    rgb_mesh = SimpleNamespace(primitives=[primitive])

    class _RgbNode:
        def __init__(self, mesh):
            self.mesh = mesh

    rgb_node = _RgbNode(rgb_mesh)
    monkeypatch.setattr(rasterizer_context_module, "qd_to_numpy", lambda value: value)
    monkeypatch.setattr(
        rasterizer_context_module.mu,
        "surface_uvs_to_trimesh_visual",
        lambda surface, *, uvs, n_verts: trimesh.visual.texture.TextureVisuals(uv=uvs),
    )
    monkeypatch.setattr(
        rasterizer_context_module.pyrender.Mesh,
        "from_trimesh",
        lambda *args, **kwargs: rgb_mesh,
    )
    uploaded = []
    context = SimpleNamespace(
        sim=SimpleNamespace(
            fem_solver=SimpleNamespace(
                is_active=True,
                entities=[entity],
                get_state_render=lambda substep: (vertices, triangles, uvs),
            ),
            cur_substep_local=0,
        ),
        rendered_envs_idx=[0],
        _fem_surface_vertex_indices={},
        static_nodes={},
        rgb_only_nodes=set(),
        part_segmentation_nodes={},
        _part_segmentation_vertex_indices={},
        segmentation_only_nodes=set(),
        part_segmentation_indexed_counts={},
        part_segmentation_palette_by_idxc={},
    )

    def add_static_node(fem_entity, mesh, *, i_b):
        assert fem_entity is entity
        assert mesh is rgb_mesh
        context.static_nodes[(i_b, fem_entity.uid)] = rgb_node

    context.add_static_node = add_static_node
    RasterizerContext.on_fem(context)
    assert context.static_nodes[(0, 8)] is rgb_node
    assert np.array_equal(context._fem_surface_vertex_indices[(0, 8)], [0, 1, 2])

    class _RenderState:
        def to_numpy(self, dtype):
            return vertices.astype(dtype)

    context.sim.fem_solver.get_state_render = lambda substep: (_RenderState(), None, None)
    context._scene = SimpleNamespace(
        reorder_vertices=lambda node, values: values,
        get_buffer_id=lambda node, field: 1,
    )
    context.jit = SimpleNamespace(
        update_buffer=lambda buffer_id, data, **kwargs: uploaded.append(np.array(data, copy=True)),
        update_normal=lambda node, data: None,
    )
    RasterizerContext.update_fem(context, render_pass="rgb")
    assert len(uploaded) == 1
    assert uploaded[0].shape == (3, 3)
    assert context.last_render_update_stats["active_rgb_node_count"] == 1
    assert context.last_render_update_stats["position_upload_bytes"] == uploaded[0].nbytes


def test_generic_capability_is_always_reported():
    capabilities = capabilities_for_report(False)
    assert set(TEXTURED_VISUAL_OVERLAY_CAPABILITIES).issubset(capabilities)


def test_diagnostic_capability_report_excludes_textured_visual_overlay_partition():
    capabilities = capabilities_for_report(False, diagnostic_scene=True)

    assert set(TEXTURED_VISUAL_OVERLAY_CAPABILITIES).isdisjoint(capabilities)
    assert "part_segmentation_triptych_telemetry" in capabilities


def test_diagnostic_session_handshake_uses_diagnostic_capability_report(monkeypatch):
    session = object.__new__(GenesisLiveSession)
    session._scene_config = {"agentic_diagnostics": {}}
    session.session_id = "diagnostic-session"
    session.status = lambda: {"mode": "diagnostic"}
    calls = []

    def report(*, diagnostic_scene=False):
        calls.append(diagnostic_scene)
        capabilities = capabilities_for_report(False, diagnostic_scene=diagnostic_scene)
        return {"capabilities": list(capabilities), "backend_requirements": {}}

    monkeypatch.setattr(session_module, "capability_report", report)

    handshake = session.handshake()

    assert calls == [True]
    assert set(TEXTURED_VISUAL_OVERLAY_CAPABILITIES).isdisjoint(handshake["capabilities"])
