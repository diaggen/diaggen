from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import genesis as gs
from genesis.engine.controllers.box_end_effector import apply_static_box_anchors
from genesis.utils.heterogeneous_materials import (
    SurfaceHeterogeneousMaterial,
    load_obj_triangle_faces,
    load_surface_heterogeneous_material,
    validate_surface_mesh_material_contract,
)

from .actions import ActionRegistry, apply_probe_action
from .capabilities import capability_report, surface_backend_status
from .protocol import PROTOCOL, GenesisLiveError, error_response, ok_response
from .snapshots import controller_snapshots, fused_observation, geometry_context
from .visual_overlay import (
    VISUAL_OVERLAY_REST_ATOL_M,
    VisualOverlaySpec,
    load_visual_overlay_assets,
    validate_visual_overlay_spec,
)
from .visual_telemetry import (
    GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
    VisualTelemetry,
    canonical_fixed_rgb_request,
    canonical_part_shaded_request,
    part_shaded_vis_options,
)

DEFAULT_RENDER_EVERY_STEPS = 10


def is_agentic_diagnostics_scene_config(scene_config: dict[str, Any]) -> bool:
    """Whether this config is governed by the agentic diagnostics contract."""
    return "agentic_diagnostics" in scene_config


@dataclass
class VisualOverlayRecord:
    physical_entity_name: str
    physical_entity: Any
    visual_entity: Any
    physics_vertex_indices: np.ndarray
    visual_rest_vertices_m: np.ndarray


@dataclass(frozen=True)
class ProbeEndpoint:
    simulation_step: int
    vector_env_local_m: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        vector = self.vector_env_local_m.astype(float).tolist()
        return {
            "simulation_step": int(self.simulation_step),
            "vector_env_local_m": vector,
            "magnitude_m": float(np.linalg.norm(self.vector_env_local_m)),
        }


@dataclass
class ActiveProbeMeasurement:
    """One runtime-owned measurement, kept only until release completes."""

    dispatch_token: str
    entity_name: str
    entity: Any
    controller_id: str
    anchor_id: str
    target_vertices: np.ndarray
    anchor_vertices: np.ndarray
    baseline_relative: np.ndarray
    load_end_step: int | None = None
    release_step: int | None = None
    recovery_steps: int | None = None
    controller_policy: dict[str, Any] | None = None
    controller_telemetry: dict[str, Any] | None = None
    force_summary: dict[str, Any] | None = None
    under_load: ProbeEndpoint | None = None
    released: bool = False


class GenesisLiveSession:
    def __init__(
        self,
        *,
        scene_config_path: str | None = None,
        start_paused: bool = True,
        output_dir: str | None = None,
    ):
        self.scene_config_path = scene_config_path
        self.start_paused = bool(start_paused)
        self.output_dir = output_dir
        self.session_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.actions = ActionRegistry()
        self.controllers = {}
        self.anchor_records = {}
        self.anchor_by_id: dict[str, dict[str, Any]] = {}
        self.active_measurement_by_controller: dict[str, ActiveProbeMeasurement] = {}
        self.last_request_id = None
        self.last_frame_index = None
        self.fatal_error = None
        self.heartbeat_timestamp = time.time()
        self.running = False
        self.paused = bool(start_paused)
        self.current_step = 0
        self._scene_config = self._load_scene_config(scene_config_path)
        self.scene = None
        self.entities = {}
        self.context_entities = {}
        self.visual_overlays: dict[str, VisualOverlayRecord] = {}
        self._visual_overlay_last_sync_step: int | None = None
        self._visual_overlay_last_sync_seconds: float | None = None
        self._validated_visual_overlay_specs: dict[int, VisualOverlaySpec] = {}
        self.visual_telemetry = VisualTelemetry(self._resolve_output_dir())
        self.build_scene()

    def _resolve_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        if self.scene_config_path is not None:
            return Path(self.scene_config_path).resolve().parent / "genesis_live_outputs" / self.session_id
        return Path.cwd() / "genesis_live_outputs" / self.session_id

    def _load_scene_config(self, path: str | None) -> dict[str, Any]:
        if path is None:
            return {"entities": []}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise GenesisLiveError("invalid_scene_config", "scene config must be a JSON object")
        return data

    def _scene_requires_surface_backend(self, scene_config: dict[str, Any]) -> bool:
        for entity_cfg in scene_config.get("entities", []):
            if not isinstance(entity_cfg, dict):
                continue
            morph_cfg = entity_cfg.get("morph", {})
            material_cfg = entity_cfg.get("material", {})
            if isinstance(morph_cfg, dict) and morph_cfg.get("type") == "surface_mesh":
                return True
            if isinstance(material_cfg, dict) and material_cfg.get("type") in {"surface_shell", "cloth"}:
                return True
        return False

    def _scene_requested_backend(self, scene_config: dict[str, Any]) -> Literal["cpu", "cuda"]:
        backend = scene_config.get("backend", "cpu")
        if backend not in {"cpu", "cuda"}:
            raise GenesisLiveError("invalid_scene_config", "scene backend must be 'cpu' or 'cuda'")
        return backend

    def _validate_scene_config_before_build(self, scene_config: dict[str, Any]) -> None:
        raw_entities = scene_config.get("entities", [])
        if is_agentic_diagnostics_scene_config(scene_config) and any(
            isinstance(entity_cfg, dict) and entity_cfg.get("visual_overlay") is not None for entity_cfg in raw_entities
        ):
            raise GenesisLiveError(
                "texture_overlay_not_supported_in_diagnostics",
                "visual_overlay is not supported in agentic diagnostics scenes",
            )

        visual_overlay_specs: dict[int, VisualOverlaySpec] = {}
        entity_name_counts: dict[str, int] = {}
        for index, entity_cfg in enumerate(raw_entities):
            if not isinstance(entity_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity entries must be JSON objects")
            morph_cfg = entity_cfg.get("morph", {})
            material_cfg = entity_cfg.get("material", {})
            if not isinstance(morph_cfg, dict) or not isinstance(material_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity morph and material entries must be JSON objects")
            morph_type = morph_cfg.get("type")
            material_type = material_cfg.get("type", "elastic")
            visual_overlay_spec = validate_visual_overlay_spec(entity_cfg, entity_index=index)
            if visual_overlay_spec is not None:
                visual_overlay_specs[index] = visual_overlay_spec
            entity_name = str(entity_cfg.get("name") or f"entity_{index}")
            entity_name_counts[entity_name] = entity_name_counts.get(entity_name, 0) + 1
            if entity_cfg.get("part_segmentation") is not None:
                entity_cfg["part_segmentation"] = self._validate_part_segmentation_contract(
                    entity_cfg["part_segmentation"], entity_index=index
                )
            if morph_type == "surface_mesh":
                if material_type not in {"surface_shell", "cloth"}:
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "surface_mesh morph requires material.type='surface_shell' or 'cloth'",
                        details={"entity_index": index, "material_type": material_type},
                    )
                if "file" not in morph_cfg:
                    raise GenesisLiveError("invalid_scene_config", "surface_mesh morph requires file")
                faces = self._validate_surface_obj_faces(morph_cfg["file"], entity_index=index)
                heterogeneous = material_cfg.get("heterogeneous")
                if heterogeneous is not None:
                    material_file = self._surface_heterogeneous_material_file(
                        heterogeneous,
                        entity_index=index,
                        mesh_file=morph_cfg["file"],
                    )
                    self._validate_surface_heterogeneous_material_contract(
                        mesh_file=morph_cfg["file"],
                        material_file=material_file,
                        source_triangle_count=int(faces.shape[0]),
                        entity_index=index,
                    )
            elif morph_type == "tet_mesh":
                if material_type != "elastic":
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "tet_mesh morph requires material.type='elastic' for this feature",
                        details={"entity_index": index, "material_type": material_type},
                    )
        duplicate_overlay_owner_names = sorted(
            spec.entity_name for spec in visual_overlay_specs.values() if entity_name_counts[spec.entity_name] > 1
        )
        if duplicate_overlay_owner_names:
            raise GenesisLiveError(
                "invalid_visual_overlay",
                "visual_overlay owner names must be unique",
                details={"entity_name": duplicate_overlay_owner_names[0]},
            )
        self._validated_visual_overlay_specs = visual_overlay_specs

    def _validate_part_segmentation_contract(self, config: Any, *, entity_index: int) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise GenesisLiveError(
                "invalid_part_segmentation",
                "each diagnostic entity requires a part_segmentation object",
                details={"entity_index": entity_index},
            )
        required = {
            "primitive_labels_file",
            "primitive_labels_key",
            "palette_file",
            "palette_key",
            "parts",
            "context_palette",
        }
        if set(config) != required:
            raise GenesisLiveError(
                "invalid_part_segmentation",
                "part_segmentation has unexpected or missing keys",
                details={"entity_index": entity_index, "keys": sorted(config), "required": sorted(required)},
            )
        labels_key = config["primitive_labels_key"]
        if labels_key not in {"tri_part_labels", "tet_part_labels"}:
            raise GenesisLiveError("invalid_part_segmentation", "unsupported primitive_labels_key")
        labels_path = Path(str(config["primitive_labels_file"]))
        palette_path = Path(str(config["palette_file"]))
        try:
            with np.load(labels_path, allow_pickle=False) as payload:
                labels = np.asarray(payload[labels_key])
            with np.load(palette_path, allow_pickle=False) as payload:
                colors = np.asarray(payload[str(config["palette_key"])])[:, :3]
        except (OSError, KeyError, ValueError, IndexError) as exc:
            raise GenesisLiveError(
                "invalid_part_segmentation",
                "failed to load part segmentation labels or palette",
                details={"entity_index": entity_index, "error": str(exc)},
            ) from exc
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise GenesisLiveError(
                "invalid_part_segmentation", "primitive labels must be a non-negative integer vector"
            )
        if colors.ndim != 2 or colors.shape[1] != 3 or not np.all(np.isfinite(colors)):
            raise GenesisLiveError("invalid_part_segmentation", "part palette must contain finite RGB rows")
        colors_u8 = np.rint(colors).astype(np.int64)
        if not np.allclose(colors, colors_u8, atol=1e-6, rtol=0.0) or np.any((colors_u8 < 0) | (colors_u8 > 255)):
            raise GenesisLiveError("invalid_part_segmentation", "part palette RGB values must be integers in [0, 255]")
        if (
            np.any(np.all(colors_u8 == 0, axis=1))
            or len({tuple(row) for row in colors_u8.tolist()}) != colors_u8.shape[0]
        ):
            raise GenesisLiveError("invalid_part_segmentation", "part palette colors must be unique and non-black")
        parts = config["parts"]
        if not isinstance(parts, list) or not parts:
            raise GenesisLiveError("invalid_part_segmentation", "part_segmentation.parts must be non-empty")
        part_ids = []
        for part in parts:
            if not isinstance(part, dict) or set(part) != {"part_id", "part_name", "part_color_rgb"}:
                raise GenesisLiveError("invalid_part_segmentation", "each part entry requires id, name, and RGB")
            part_id = int(part["part_id"])
            if not str(part["part_name"]).strip() or part_id < 0 or part_id >= colors_u8.shape[0]:
                raise GenesisLiveError("invalid_part_segmentation", "part entry has invalid id or empty name")
            if list(np.asarray(part["part_color_rgb"], dtype=np.int64)) != list(colors_u8[part_id]):
                raise GenesisLiveError("invalid_part_segmentation", "part entry color does not match palette")
            part_ids.append(part_id)
        if len(set(part_ids)) != len(part_ids) or set(np.unique(labels).tolist()) != set(part_ids):
            raise GenesisLiveError("invalid_part_segmentation", "part ids must exactly cover primitive labels")
        context_palette = config["context_palette"]
        active_context_keys = {"background", "fixture", "probe"}
        archived_context_keys = active_context_keys | {"ground"}
        if not isinstance(context_palette, dict):
            raise GenesisLiveError("invalid_part_segmentation", "context_palette has invalid categories")
        context_keys = set(context_palette)
        if context_keys != active_context_keys and context_keys != archived_context_keys:
            raise GenesisLiveError("invalid_part_segmentation", "context_palette has invalid categories")
        context_colors = {}
        for kind, raw_color in context_palette.items():
            color = np.asarray(raw_color)
            if color.shape != (3,) or not np.issubdtype(color.dtype, np.integer):
                raise GenesisLiveError(
                    "invalid_part_segmentation", f"context color {kind!r} must be an integer RGB triplet"
                )
            if np.any((color < 0) | (color > 255)):
                raise GenesisLiveError("invalid_part_segmentation", f"context color {kind!r} must be in [0, 255]")
            context_colors[kind] = tuple(int(value) for value in color)
        if context_colors["background"] != (0, 0, 0):
            raise GenesisLiveError("invalid_part_segmentation", "context background must be black")
        if len(set(context_colors.values())) != len(context_colors):
            raise GenesisLiveError("invalid_part_segmentation", "context palette colors must be unique")
        asset_colors = {tuple(int(value) for value in row) for row in colors_u8}
        conflicts = sorted(set(context_colors.values()).intersection(asset_colors))
        if conflicts:
            raise GenesisLiveError(
                "invalid_part_segmentation",
                "context palette colors conflict with asset part colors",
                details={"conflicts": conflicts},
            )
        normalized = dict(config)
        normalized["context_palette"] = {
            kind: list(context_colors[kind]) for kind in ("background", "fixture", "probe")
        }
        return normalized

    def _validate_surface_obj_faces(self, mesh_file: str | Path, *, entity_index: int) -> np.ndarray:
        path = Path(mesh_file)
        if path.suffix.lower() != ".obj":
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires a source OBJ mesh with triangle faces",
                details={"entity_index": entity_index, "file": str(path)},
            )
        try:
            faces = load_obj_triangle_faces(path)
        except gs.GenesisException as exc:
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires source OBJ triangle faces",
                details={"entity_index": entity_index, "file": str(path), "error": str(exc)},
            ) from exc
        if faces.shape[0] == 0:
            raise GenesisLiveError(
                "invalid_scene_config",
                "surface_mesh live path requires at least one source OBJ face",
                details={"entity_index": entity_index, "file": str(path)},
            )
        return faces

    def _surface_heterogeneous_material_file(
        self,
        heterogeneous: Any,
        *,
        entity_index: int,
        mesh_file: str | Path,
    ) -> str:
        details = {"entity_index": entity_index, "mesh_file": str(mesh_file)}
        if not isinstance(heterogeneous, dict):
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous must be an object",
                details=details,
            )
        allowed_keys = {"kind", "material_file"}
        if set(heterogeneous) != allowed_keys:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous must contain exactly kind='surface_triangles' and material_file",
                details={**details, "keys": sorted(heterogeneous), "allowed_keys": sorted(allowed_keys)},
            )
        if heterogeneous["kind"] != "surface_triangles":
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous.kind must be 'surface_triangles'",
                details={**details, "kind": heterogeneous["kind"]},
            )
        material_file = heterogeneous["material_file"]
        if not isinstance(material_file, str) or not material_file:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface material.heterogeneous.material_file must be a filesystem path string",
                details=details,
            )
        return material_file

    def _validate_surface_heterogeneous_material_contract(
        self,
        *,
        mesh_file: str | Path,
        material_file: str | Path,
        source_triangle_count: int,
        entity_index: int,
    ) -> None:
        try:
            material_data = load_surface_heterogeneous_material(
                SurfaceHeterogeneousMaterial(file=material_file),
                triangle_count=source_triangle_count,
            )
            validate_surface_mesh_material_contract(mesh_file, material_data)
        except gs.GenesisException as exc:
            raise GenesisLiveError(
                "invalid_surface_heterogeneous_material",
                "surface heterogeneous material does not match the source OBJ triangle contract",
                details={
                    "entity_index": entity_index,
                    "mesh_file": str(mesh_file),
                    "material_file": str(material_file),
                    "source_triangle_count": int(source_triangle_count),
                    "error": str(exc),
                },
            ) from exc

    def _ensure_genesis_initialized(self, backend: Literal["cpu", "cuda"], *, requires_surface_backend: bool) -> None:
        if requires_surface_backend:
            if backend != "cuda":
                raise GenesisLiveError(
                    "unsupported_surface_backend",
                    "surface shell diagnostics require scene backend='cuda'",
                    details={"requested_backend": backend},
                )
            status = surface_backend_status()
            if not status["available"]:
                raise GenesisLiveError(
                    "unsupported_surface_backend",
                    "surface shell diagnostics require uipc, CUDA, and IPCCoupler support",
                    details=status,
                )

        requested_backend = gs.cuda if backend == "cuda" else gs.cpu
        if gs._initialized:
            if gs.backend != requested_backend:
                error_code = "unsupported_surface_backend" if requires_surface_backend else "backend_mismatch"
                raise GenesisLiveError(
                    error_code,
                    "Genesis is already initialized on a different backend",
                    details={"current_backend": str(gs.backend), "required_backend": backend},
                )
            return
        gs.init(backend=requested_backend, logging_level="warning")

    def build_scene(self) -> None:
        self._validate_scene_config_before_build(self._scene_config)
        requires_surface_backend = self._scene_requires_surface_backend(self._scene_config)
        backend = self._scene_requested_backend(self._scene_config)
        self._ensure_genesis_initialized(backend, requires_surface_backend=requires_surface_backend)
        self.visual_overlays = {}
        self._visual_overlay_last_sync_step = None
        self._visual_overlay_last_sync_seconds = None
        self.visual_telemetry.reset_triptych_cameras()
        self.visual_telemetry.reset_fixed_rgb_cameras()
        if self.scene is not None:
            self.scene.destroy()
        self.controllers.clear()
        self.actions.clear()
        self.anchor_records.clear()
        self.anchor_by_id.clear()
        self.active_measurement_by_controller.clear()

        sim_options = self._scene_config.get("sim_options", {})
        fem_options = {"enable_vertex_constraints": True}
        fem_options.update(self._scene_config.get("fem_options", {}))
        fem_options["enable_floor"] = False
        scene_kwargs = {
            "show_viewer": False,
            "sim_options": gs.options.SimOptions(**sim_options),
            "fem_options": gs.options.FEMOptions(**fem_options),
        }
        if any(entity_cfg.get("part_segmentation") is not None for entity_cfg in self._scene_config.get("entities", [])):
            scene_kwargs["vis_options"] = part_shaded_vis_options()
        if requires_surface_backend:
            scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
                **self._scene_config.get("coupler_options", {})
            )
        self.scene = gs.Scene(**scene_kwargs)

        self.entities = {}
        self.context_entities = {}
        pending_anchors = []
        for index, entity_cfg in enumerate(self._scene_config.get("entities", [])):
            if not isinstance(entity_cfg, dict):
                raise GenesisLiveError("invalid_scene_config", "entity entries must be JSON objects")
            name = str(entity_cfg.get("name") or f"entity_{index}")
            morph_cfg = dict(entity_cfg.get("morph", {}))
            material_cfg = dict(entity_cfg.get("material", {}))
            morph = self._build_morph(morph_cfg)
            material = self._build_material(material_cfg)
            entity = self.scene.add_entity(morph=morph, material=material, name=name)
            if entity_cfg.get("part_segmentation") is not None:
                entity._part_segmentation_config = dict(entity_cfg["part_segmentation"])
            self.entities[name] = entity
            pending_anchors.append((entity, entity_cfg.get("anchors", [])))
            overlay_spec = self._validated_visual_overlay_specs.get(index)
            if overlay_spec is not None:
                assets = load_visual_overlay_assets(overlay_spec)
                entity._rgb_visualization_disabled = True
                try:
                    visual_entity = self.scene.add_entity(
                        morph=gs.morphs.MeshSet(
                            files=(assets.mesh,),
                            fixed=True,
                            enable_custom_vverts=True,
                        ),
                        material=gs.materials.Kinematic(use_visual_raycasting=False),
                        name=f"{name}__visual_overlay",
                    )
                except (gs.GenesisException, TypeError, ValueError) as exc:
                    raise GenesisLiveError(
                        "invalid_visual_overlay",
                        "failed to build visual_overlay entity",
                        details={
                            "entity_index": overlay_spec.entity_index,
                            "entity_name": overlay_spec.entity_name,
                        },
                    ) from exc
                self.visual_overlays[name] = VisualOverlayRecord(
                    physical_entity_name=name,
                    physical_entity=entity,
                    visual_entity=visual_entity,
                    physics_vertex_indices=assets.physics_vertex_indices,
                    visual_rest_vertices_m=assets.visual_rest_vertices_m,
                )

        self.visual_telemetry.register_triptych_cameras(self)
        self.visual_telemetry.register_fixed_rgb_cameras(self)
        try:
            self.scene.build()
        except gs.GenesisException as exc:
            if not self.visual_overlays:
                raise
            owner_name = next(iter(self.visual_overlays))
            spec = next(
                candidate
                for candidate in self._validated_visual_overlay_specs.values()
                if candidate.entity_name == owner_name
            )
            raise GenesisLiveError(
                "invalid_visual_overlay",
                "failed to build scene with visual_overlay",
                details={
                    "entity_index": spec.entity_index,
                    "entity_name": owner_name,
                },
            ) from exc
        for entity, anchors in pending_anchors:
            if anchors:
                name = next(name for name, candidate in self.entities.items() if candidate is entity)
                try:
                    records = apply_static_box_anchors(entity, anchors, frame="env_local")
                    self.anchor_records[name] = records
                    self.anchor_by_id[name] = {record.anchor_id: record for record in records}
                except gs.GenesisException as exc:
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        "static box anchor selection failed",
                        details={"entity": name, "error": str(exc)},
                    ) from exc
        self.current_step = 0
        if self.visual_overlays:
            self._validate_fem_state(checked_at="after_build")
            self._sync_visual_overlays(
                checked_at="after_build",
                require_rest_alignment=True,
            )
        self.paused = self.start_paused
        self.running = False
        self.last_frame_index = None
        self.heartbeat_timestamp = time.time()

    def _build_morph(self, morph_cfg: dict[str, Any]):
        morph_type = morph_cfg.get("type")
        if morph_type == "tet_mesh":
            if "file" not in morph_cfg:
                raise GenesisLiveError("invalid_scene_config", "tet_mesh morph requires file")
            kwargs = {"file": morph_cfg["file"]}
            if "scale" in morph_cfg:
                kwargs["scale"] = tuple(morph_cfg["scale"])
            if "pos" in morph_cfg:
                kwargs["pos"] = tuple(morph_cfg["pos"])
            return gs.morphs.TetMesh(**kwargs)
        if morph_type == "surface_mesh":
            if "file" not in morph_cfg:
                raise GenesisLiveError("invalid_scene_config", "surface_mesh morph requires file")
            for key in ("decimate", "convexify", "force_retet", "quality", "retet"):
                if bool(morph_cfg.get(key, False)):
                    raise GenesisLiveError(
                        "invalid_scene_config",
                        f"surface_mesh live path does not allow mesh rewriting option: {key}",
                    )
            kwargs = {"file": morph_cfg["file"], "convexify": False, "decimate": False}
            for key in ("scale", "pos", "euler", "quat"):
                if key in morph_cfg:
                    kwargs[key] = tuple(morph_cfg[key])
            if "file_meshes_are_zup" in morph_cfg:
                kwargs["file_meshes_are_zup"] = bool(morph_cfg["file_meshes_are_zup"])
            return gs.morphs.Mesh(**kwargs)
        raise GenesisLiveError(
            "invalid_scene_config",
            "only morph.type='tet_mesh' or morph.type='surface_mesh' is supported by this feature",
        )

    def _build_material(self, material_cfg: dict[str, Any]):
        material_type = material_cfg.get("type", "elastic")
        if material_type == "elastic":
            kwargs = {key: material_cfg[key] for key in ("E", "nu", "rho", "friction_mu") if key in material_cfg}
            heterogeneous = material_cfg.get("heterogeneous")
            if heterogeneous is not None:
                kwargs["heterogeneous"] = gs.materials.FEM.HeterogeneousMaterial(**heterogeneous)
            return gs.materials.FEM.Elastic(**kwargs)
        if material_type in {"surface_shell", "cloth"}:
            kwargs = {
                key: material_cfg[key]
                for key in ("E", "nu", "rho", "thickness", "bending_stiffness", "friction_mu", "contact_resistance")
                if key in material_cfg
            }
            heterogeneous = material_cfg.get("heterogeneous")
            if heterogeneous is not None:
                kwargs["heterogeneous"] = gs.materials.FEM.SurfaceHeterogeneousMaterial(
                    file=self._surface_heterogeneous_material_file(
                        heterogeneous,
                        entity_index=-1,
                        mesh_file="<material_build>",
                    )
                )
            return gs.materials.FEM.Cloth(**kwargs)
        raise GenesisLiveError(
            "invalid_scene_config",
            "only material.type='elastic', 'surface_shell', or 'cloth' is supported by this feature",
        )

    def default_entity(self):
        if not self.entities:
            raise GenesisLiveError("no_entity", "session has no diagnostic entity")
        name = next(iter(self.entities))
        return name, self.entities[name]

    def entity_by_name(self, name: str):
        entity = self.entities.get(name)
        if entity is None:
            raise GenesisLiveError("unknown_entity", f"unknown entity: {name}")
        return name, entity

    def status(self) -> dict[str, Any]:
        coupler_type = None
        if self.scene is not None and getattr(self.scene, "sim", None) is not None:
            coupler_type = self.scene.sim.coupler.__class__.__name__
        result = {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "paused": bool(self.paused),
            "running": bool(self.running),
            "current_step": int(self.current_step),
            "last_completed_request_id": self.last_request_id,
            "last_rendered_frame_index": self.last_frame_index,
            "heartbeat_timestamp": self.heartbeat_timestamp,
            "fatal_error": self.fatal_error,
            "controller_count": len(self.controllers),
            "backend": str(gs.backend) if gs._initialized else None,
            "coupler_type": coupler_type,
            "surface_scene": self._scene_requires_surface_backend(self._scene_config),
        }
        if self.visual_overlays:
            result.update(
                {
                    "visual_overlay_count": len(self.visual_overlays),
                    "visual_overlay_last_sync_step": self._visual_overlay_last_sync_step,
                    "visual_overlay_last_sync_seconds": self._visual_overlay_last_sync_seconds,
                }
            )
        return result

    def backend_requirements(self) -> dict[str, Any]:
        return self._scene_capability_report()["backend_requirements"]

    def _scene_capability_report(self) -> dict[str, Any]:
        return capability_report(
            diagnostic_scene=is_agentic_diagnostics_scene_config(self._scene_config),
        )

    def handshake(self) -> dict[str, Any]:
        report = self._scene_capability_report()
        return {
            "protocol": PROTOCOL,
            "session_id": self.session_id,
            "genesis_version": gs.__version__,
            "capabilities": report["capabilities"],
            "backend_requirements": report["backend_requirements"],
            "status": self.status(),
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        method = request.get("method")
        params = request.get("params", {})
        if not isinstance(method, str):
            return error_response(request_id, GenesisLiveError("invalid_request", "request method must be a string"))
        if not isinstance(params, dict):
            return error_response(request_id, GenesisLiveError("invalid_request", "request params must be an object"))

        with self.lock:
            try:
                data = self.dispatch(method, params)
                self.last_request_id = request_id
                self.heartbeat_timestamp = time.time()
                return ok_response(request_id, data)
            except GenesisLiveError as exc:
                self.heartbeat_timestamp = time.time()
                return error_response(request_id, exc)
            except Exception as exc:
                self.heartbeat_timestamp = time.time()
                return error_response(request_id, GenesisLiveError("internal_error", str(exc)))

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "session.handshake":
            return self.handshake()
        if method == "session.close":
            return {"closed": True, "status": self.status()}
        if method in {"command.status", "command.heartbeat"}:
            self.heartbeat_timestamp = time.time()
            return self.status()
        if method == "sim.pause":
            self.paused = True
            self.running = False
            return self.status()
        if method == "sim.reset":
            self.build_scene()
            return self.status()
        if method == "sim.resume":
            return self.resume(params)
        if method == "visual.capture":
            return self.capture_visual(params)
        if method == "geometry.context.get":
            entity_name, entity = (
                self.entity_by_name(str(params["entity"])) if "entity" in params else self.default_entity()
            )
            return geometry_context(entity, entity_name=entity_name)
        if method == "visual_overlay.trace.get":
            return self.visual_overlay_trace(params)
        if method == "probe.action.register":
            return self.actions.register(params)
        if method == "probe.apply":
            result = apply_probe_action(self, params)
            return {"probe": result, "status": self.status()}
        if method == "observation.fused":
            return fused_observation(self)
        raise GenesisLiveError("unknown_method", f"unknown live method: {method}")

    def prepare_probe_measurement(
        self, *, measurement: dict[str, Any], entity_name: str, controller_id: str, target_vertices: np.ndarray
    ) -> ActiveProbeMeasurement:
        dispatch_token = str(measurement.get("dispatch_token", "")).strip()
        anchor_id = str(measurement.get("anchor_id", "")).strip()
        if not dispatch_token or not anchor_id:
            raise GenesisLiveError("invalid_probe_measurement", "dispatch_token and anchor_id are required")
        entity = self.entities.get(entity_name)
        anchor = self.anchor_by_id.get(entity_name, {}).get(anchor_id)
        if anchor is None:
            raise GenesisLiveError(
                "invalid_probe_measurement",
                "measurement anchor is unavailable",
                details={"entity": entity_name, "anchor_id": anchor_id},
            )
        positions = self._entity_positions_for_validation(entity_name, entity, checked_at="probe_measurement_baseline")
        target_vertices = np.asarray(target_vertices, dtype=np.int64).reshape(-1)
        anchor_vertices = np.asarray(anchor.selected_vertices, dtype=np.int64).reshape(-1)
        if (
            target_vertices.size == 0
            or anchor_vertices.size == 0
            or np.any(target_vertices < 0)
            or np.any(anchor_vertices < 0)
            or np.any(target_vertices >= positions.shape[0])
            or np.any(anchor_vertices >= positions.shape[0])
        ):
            raise GenesisLiveError(
                "invalid_probe_measurement", "measurement vertex selections are outside the physical FEM domain"
            )
        baseline_relative = np.mean(positions[target_vertices], axis=0) - np.mean(positions[anchor_vertices], axis=0)
        if not np.all(np.isfinite(baseline_relative)):
            raise GenesisLiveError("invalid_probe_measurement", "measurement baseline relative vector is non-finite")
        schedule = measurement.get("schedule")
        load_end_step = None
        recovery_steps = None
        if schedule is not None:
            if not isinstance(schedule, dict) or set(schedule) != {"load_steps", "recovery_steps"}:
                raise GenesisLiveError(
                    "invalid_probe_measurement",
                    "scheduled measurement requires exactly load_steps and recovery_steps",
                )
            load_steps = schedule.get("load_steps")
            recovery_steps = schedule.get("recovery_steps")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (load_steps, recovery_steps)
            ):
                raise GenesisLiveError(
                    "invalid_probe_measurement",
                    "scheduled measurement steps must be positive integers",
                )
            load_end_step = int(self.current_step) + int(load_steps)
        return ActiveProbeMeasurement(
            dispatch_token=dispatch_token,
            entity_name=entity_name,
            entity=entity,
            controller_id=controller_id,
            anchor_id=anchor_id,
            target_vertices=target_vertices,
            anchor_vertices=anchor_vertices,
            baseline_relative=baseline_relative.copy(),
            load_end_step=load_end_step,
            recovery_steps=recovery_steps,
        )

    def publish_probe_measurement(
        self, controller_id: str, controller: Any, measurement: ActiveProbeMeasurement
    ) -> None:
        self.controllers[controller_id] = controller
        self.active_measurement_by_controller[controller_id] = measurement

    def release_probe_measurement(self, controller_id: str, *, controller_state: dict[str, Any] | None = None) -> None:
        active = self.active_measurement_by_controller.get(controller_id)
        if active is not None:
            if active.load_end_step is not None:
                if active.under_load is None or self.current_step != active.load_end_step:
                    raise GenesisLiveError(
                        "invalid_probe_measurement",
                        "scheduled probe release does not match its load endpoint",
                    )
                active.release_step = int(self.current_step)
                if not isinstance(controller_state, dict):
                    raise GenesisLiveError(
                        "invalid_probe_measurement",
                        "scheduled probe release lacks controller evidence",
                    )
                policy = controller_state.get("controller_policy")
                telemetry = controller_state.get("controller_telemetry")
                summary = telemetry.get("summary") if isinstance(telemetry, dict) else None
                if not all(isinstance(value, dict) for value in (policy, telemetry, summary)):
                    raise GenesisLiveError(
                        "invalid_probe_measurement",
                        "scheduled probe release lacks controller evidence",
                    )
                active.controller_policy = dict(policy)
                active.controller_telemetry = dict(telemetry)
                active.force_summary = dict(summary)
            active.released = True

    def _measurement_endpoint(self, active: ActiveProbeMeasurement) -> ProbeEndpoint:
        positions = self._entity_positions_for_validation(
            active.entity_name, active.entity, checked_at="probe_measurement_endpoint"
        )
        relative = np.mean(positions[active.target_vertices], axis=0) - np.mean(
            positions[active.anchor_vertices], axis=0
        )
        vector = relative - active.baseline_relative
        if not np.all(np.isfinite(vector)):
            raise GenesisLiveError("invalid_probe_measurement", "probe endpoint vector is non-finite")
        return ProbeEndpoint(
            simulation_step=int(self.current_step), vector_env_local_m=np.asarray(vector, dtype=np.float32)
        )

    def _completed_probe_measurement(
        self, active: ActiveProbeMeasurement, post_release: ProbeEndpoint
    ) -> dict[str, Any]:
        if active.under_load is None or post_release.simulation_step <= active.under_load.simulation_step:
            raise GenesisLiveError("invalid_probe_measurement", "probe endpoint pair is not causally ordered")
        result = {
            "dispatch_token": active.dispatch_token,
            "identity": {
                "entity": active.entity_name,
                "controller_id": active.controller_id,
                "anchor_id": active.anchor_id,
                "target_vertices": active.target_vertices.astype(int).tolist(),
                "anchor_vertices": active.anchor_vertices.astype(int).tolist(),
                "baseline_relative_env_local_m": active.baseline_relative.astype(float).tolist(),
            },
            "under_load": active.under_load.to_dict(),
            "post_release": post_release.to_dict(),
        }
        if active.load_end_step is not None:
            if (
                active.release_step is None
                or active.recovery_steps is None
                or active.controller_policy is None
                or active.controller_telemetry is None
                or active.force_summary is None
            ):
                raise GenesisLiveError("invalid_probe_measurement", "scheduled probe evidence is incomplete")
            result.update(
                {
                    "schedule": {
                        "load_end_step": int(active.load_end_step),
                        "release_step": int(active.release_step),
                        "recovery_steps": int(active.recovery_steps),
                    },
                    "controller_policy": dict(active.controller_policy),
                    "controller_telemetry": dict(active.controller_telemetry),
                    "force_summary": dict(active.force_summary),
                }
            )
        return result

    def _sample_active_measurements(self) -> dict[str, Any] | None:
        completed = None
        for controller_id, active in list(self.active_measurement_by_controller.items()):
            try:
                if active.load_end_step is None:
                    if active.released:
                        post_release = self._measurement_endpoint(active)
                        completed = self._completed_probe_measurement(active, post_release)
                        self.active_measurement_by_controller.pop(controller_id, None)
                    elif active.under_load is None:
                        active.under_load = self._measurement_endpoint(active)
                elif not active.released:
                    if self.current_step == active.load_end_step:
                        active.under_load = self._measurement_endpoint(active)
                    elif self.current_step > active.load_end_step and active.under_load is None:
                        raise GenesisLiveError(
                            "invalid_probe_measurement",
                            "scheduled probe passed its load endpoint without a measurement",
                        )
                else:
                    recovery_end_step = int(active.release_step or -1) + int(active.recovery_steps or -1)
                    if self.current_step == recovery_end_step:
                        post_release = self._measurement_endpoint(active)
                        completed = self._completed_probe_measurement(active, post_release)
                        self.active_measurement_by_controller.pop(controller_id, None)
                    elif self.current_step > recovery_end_step:
                        raise GenesisLiveError(
                            "invalid_probe_measurement",
                            "scheduled probe passed its recovery endpoint without a measurement",
                        )
            except GenesisLiveError:
                self.active_measurement_by_controller.pop(controller_id, None)
                raise
        return completed

    def resume(self, params: dict[str, Any]) -> dict[str, Any]:
        steps = int(params.get("steps", params.get("duration_steps", 1)))
        if steps <= 0:
            raise GenesisLiveError("invalid_resume", "sim.resume steps must be positive")
        visual = params.get("diagnostic_visual")
        self._validate_visual_request(visual)
        render_every_steps = self._visual_render_every_steps(visual)
        visual_frames = []
        completed_probe_measurement = None

        self.paused = False
        self.running = True
        try:
            for local_step_index in range(steps):
                for controller in list(self.controllers.values()):
                    controller.prepare_step()
                self.scene.step()
                self.current_step += 1
                self._validate_fem_state(checked_at="after_step")
                completed = self._sample_active_measurements()
                if completed is not None:
                    completed_probe_measurement = completed
                self._sync_visual_overlays(checked_at="after_step")
                if render_every_steps is not None and (local_step_index + 1) % render_every_steps == 0:
                    visual_frames.append(
                        self._capture_visual_request(
                            visual,
                            frame_index=int(self.current_step),
                            frame_sequence_index=len(visual_frames),
                            render_every_steps=render_every_steps,
                        )
                    )
            visual_result = self._visual_result_from_frames(
                visual,
                visual_frames,
                render_every_steps=render_every_steps,
                steps_requested=steps,
            )
        except GenesisLiveError as exc:
            if exc.code == "invalid_simulation_state":
                self.fatal_error = exc.to_dict()
            raise
        finally:
            self.running = False
            self.paused = True
        result = {"steps": steps, "status": self.status(), "visual_telemetry": visual_result}
        if completed_probe_measurement is not None:
            result["completed_probe_measurement"] = completed_probe_measurement
        return result

    def _entity_positions_for_validation(self, entity_name: str, entity, *, checked_at: str) -> np.ndarray:
        from genesis.utils.misc import tensor_to_array

        if getattr(entity, "active", False):
            positions = tensor_to_array(entity.get_state().pos)
        else:
            positions = tensor_to_array(entity.init_positions)

        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 3:
            positions = positions[0]
        if positions.ndim != 2 or positions.shape[1] != 3:
            details = {
                "entity": entity_name,
                "current_step": int(self.current_step),
                "first_bad_step": int(self.current_step),
                "checked_at": checked_at,
                "shape": [int(dim) for dim in positions.shape],
            }
            raise GenesisLiveError(
                "invalid_simulation_state",
                "FEM entity positions have invalid shape during sim.resume",
                details=details,
            )
        return positions

    def _invalid_fem_state_details(self, entity_name: str, positions: np.ndarray, *, checked_at: str) -> dict[str, Any]:
        finite_mask = np.isfinite(positions)
        finite_vertex_mask = np.all(finite_mask, axis=1)
        nonfinite_vertex_indices = np.flatnonzero(~finite_vertex_mask)
        finite_values = positions[finite_mask]
        return {
            "entity": entity_name,
            "current_step": int(self.current_step),
            "first_bad_step": int(self.current_step),
            "checked_at": checked_at,
            "shape": [int(dim) for dim in positions.shape],
            "finite_vertex_count": int(np.count_nonzero(finite_vertex_mask)),
            "nonfinite_vertex_count": int(np.count_nonzero(~finite_vertex_mask)),
            "finite_scalar_count": int(np.count_nonzero(finite_mask)),
            "nonfinite_scalar_count": int(positions.size - np.count_nonzero(finite_mask)),
            "finite_min": float(np.min(finite_values)) if finite_values.size else None,
            "finite_max": float(np.max(finite_values)) if finite_values.size else None,
            "first_nonfinite_vertex_index": int(nonfinite_vertex_indices[0]) if nonfinite_vertex_indices.size else None,
        }

    def _validate_fem_state(self, *, checked_at: str) -> None:
        for entity_name, entity in self.entities.items():
            positions = self._entity_positions_for_validation(entity_name, entity, checked_at=checked_at)
            if np.all(np.isfinite(positions)):
                continue
            raise GenesisLiveError(
                "invalid_simulation_state",
                "FEM entity positions became non-finite during sim.resume",
                details=self._invalid_fem_state_details(entity_name, positions, checked_at=checked_at),
            )

    def _sync_visual_overlays(
        self,
        *,
        checked_at: str,
        require_rest_alignment: bool = False,
    ) -> None:
        if not self.visual_overlays:
            return
        from genesis.utils.misc import tensor_to_array

        started_at = time.perf_counter()
        for record in self.visual_overlays.values():
            binding = record.physics_vertex_indices
            rest_vertices = record.visual_rest_vertices_m
            if (
                binding.dtype != np.dtype(np.int64)
                or binding.ndim != 1
                or len(binding) != len(rest_vertices)
                or np.any(binding < 0)
            ):
                raise GenesisLiveError(
                    "invalid_visual_overlay",
                    "visual_overlay binding is malformed during synchronization",
                    details={"entity_name": record.physical_entity_name, "checked_at": checked_at},
                )
            physical_positions = self._entity_positions_for_validation(
                record.physical_entity_name,
                record.physical_entity,
                checked_at=checked_at,
            )
            if np.any(binding >= len(physical_positions)):
                raise GenesisLiveError(
                    "invalid_visual_overlay",
                    "visual_overlay binding exceeds the physical FEM vertex range",
                    details={"entity_name": record.physical_entity_name, "checked_at": checked_at},
                )
            visual_positions = np.ascontiguousarray(physical_positions[binding], dtype=np.float32)
            if require_rest_alignment:
                max_error = float(np.max(np.abs(visual_positions - rest_vertices)))
                if not np.isfinite(max_error) or max_error >= VISUAL_OVERLAY_REST_ATOL_M:
                    raise GenesisLiveError(
                        "invalid_visual_overlay",
                        "visual_overlay rest vertices do not align with the physical FEM",
                        details={
                            "entity_name": record.physical_entity_name,
                            "checked_at": checked_at,
                            "max_error_m": max_error,
                        },
                    )
            try:
                record.visual_entity.set_vverts(visual_positions)
                if require_rest_alignment:
                    synced_vertices = np.asarray(tensor_to_array(record.visual_entity.get_vverts()), dtype=np.float32)
                    if synced_vertices.ndim == 3 and synced_vertices.shape[0] == 1:
                        synced_vertices = synced_vertices[0]
                    if synced_vertices.shape != rest_vertices.shape:
                        raise ValueError("visual entity returned an unexpected logical vertex shape")
                    max_readback_error = float(np.max(np.abs(synced_vertices - rest_vertices)))
                    if not np.isfinite(max_readback_error) or max_readback_error >= VISUAL_OVERLAY_REST_ATOL_M:
                        raise ValueError("visual entity rest-position readback is misaligned")
            except (gs.GenesisException, TypeError, ValueError) as exc:
                raise GenesisLiveError(
                    "invalid_visual_overlay",
                    "failed to synchronize visual_overlay vertices",
                    details={"entity_name": record.physical_entity_name, "checked_at": checked_at},
                ) from exc
        self._visual_overlay_last_sync_step = int(self.current_step)
        self._visual_overlay_last_sync_seconds = float(time.perf_counter() - started_at)

    def _validate_visual_request(self, visual):
        if visual is None:
            return
        if not isinstance(visual, dict):
            raise GenesisLiveError("invalid_visual_request", "diagnostic_visual must be an object")
        mode = visual.get("mode", "rgb_triptych")
        if mode in {"depth", "depth_triptych", "normal", "normal_triptych"}:
            if not self.visual_overlays:
                raise GenesisLiveError(
                    "unsupported_visual_mode",
                    f"{mode} requires a generic scene with visual_overlay",
                )
        elif mode in {"von_mises", "von_mises_triptych"}:
            raise GenesisLiveError("unsupported_visual_mode", f"{mode} is not supported by this migration feature")
        if mode not in {
            "rgb_triptych",
            "depth",
            "depth_triptych",
            "normal",
            "normal_triptych",
            "part_segmentation_triptych",
            "part_shaded_triptych",
            "fixed_rgb_views",
        }:
            raise GenesisLiveError("unsupported_visual_mode", f"unsupported diagnostic visual mode: {mode}")
        if mode in {"part_segmentation_triptych", "part_shaded_triptych"} and any(
            getattr(entity, "_part_segmentation_config", None) is None for entity in self.entities.values()
        ):
            raise GenesisLiveError(
                "invalid_visual_request",
                f"{mode} requires a part_segmentation contract on every diagnostic entity",
            )
        if mode == "part_shaded_triptych":
            try:
                canonical_part_shaded_request(visual)
            except ValueError as exc:
                raise GenesisLiveError("invalid_visual_request", str(exc)) from exc
        if mode == "fixed_rgb_views":
            try:
                canonical_fixed_rgb_request(visual)
            except ValueError as exc:
                raise GenesisLiveError("invalid_visual_request", str(exc)) from exc
        self._visual_render_every_steps(visual)

    def _visual_render_every_steps(self, visual) -> int | None:
        if visual is None:
            return None
        render_every_steps = visual.get(
            "render_every_steps", visual.get("capture_every_n_steps", DEFAULT_RENDER_EVERY_STEPS)
        )
        if isinstance(render_every_steps, bool) or not isinstance(render_every_steps, int) or render_every_steps <= 0:
            raise GenesisLiveError(
                "invalid_visual_request", "diagnostic_visual.render_every_steps must be a positive integer"
            )
        return int(render_every_steps)

    def _capture_visual_request(
        self,
        visual,
        *,
        frame_index: int,
        frame_sequence_index: int,
        render_every_steps: int,
    ):
        if visual is None:
            return {"requested": False}
        self._validate_fem_state(checked_at="before_visual_capture")
        mode = visual.get("mode", "rgb_triptych")
        handlers = {
            "rgb_triptych": self.visual_telemetry.capture_rgb_triptych,
            "depth": self.visual_telemetry.capture_depth_triptych,
            "depth_triptych": self.visual_telemetry.capture_depth_triptych,
            "normal": self.visual_telemetry.capture_normal_triptych,
            "normal_triptych": self.visual_telemetry.capture_normal_triptych,
            "part_segmentation_triptych": self.visual_telemetry.capture_part_segmentation_triptych,
            "part_shaded_triptych": self.visual_telemetry.capture_part_shaded_triptych,
        }
        if mode == "fixed_rgb_views":
            metadata = self.visual_telemetry.capture_fixed_rgb_views(
                self,
                visual=visual,
                frame_index=frame_index,
            )
        else:
            metadata = handlers[mode](self, frame_index=frame_index, visual=visual)
        metadata["frame_sequence_index"] = int(frame_sequence_index)
        metadata["sequence_index"] = int(frame_sequence_index)
        metadata["render_every_steps"] = int(render_every_steps)
        if "frame_index" in metadata:
            self.last_frame_index = int(metadata["frame_index"])
        else:
            self.last_frame_index = int(metadata["stitched"]["frame_index"])
        return metadata

    def capture_visual(self, params: dict[str, Any]) -> dict[str, Any]:
        visual = params.get("diagnostic_visual")
        self._validate_visual_request(visual)
        render_every_steps = self._visual_render_every_steps(visual)
        metadata = self._capture_visual_request(
            visual,
            frame_index=int(self.current_step),
            frame_sequence_index=0,
            render_every_steps=int(render_every_steps or DEFAULT_RENDER_EVERY_STEPS),
        )
        return {"visual_telemetry": metadata, "status": self.status()}

    def visual_overlay_trace(self, params: dict[str, Any]) -> dict[str, Any]:
        from genesis.utils.misc import tensor_to_array

        if not self.visual_overlays:
            raise GenesisLiveError(
                "visual_overlay_unavailable",
                "visual_overlay trace requires a generic scene with visual_overlay",
            )
        requested_entity = params.get("entity")
        if requested_entity is None:
            if len(self.visual_overlays) != 1:
                raise GenesisLiveError(
                    "invalid_visual_overlay_trace",
                    "visual_overlay trace entity is required when multiple overlays exist",
                )
            entity_name = next(iter(self.visual_overlays))
        else:
            entity_name = str(requested_entity)
        record = self.visual_overlays.get(entity_name)
        if record is None:
            raise GenesisLiveError(
                "invalid_visual_overlay_trace",
                f"unknown visual_overlay owner: {entity_name}",
            )

        physical_positions = self._entity_positions_for_validation(
            entity_name,
            record.physical_entity,
            checked_at="visual_overlay_trace",
        )
        visual_positions = np.asarray(
            tensor_to_array(record.visual_entity.get_vverts()),
            dtype=np.float32,
        )
        if visual_positions.ndim == 3 and visual_positions.shape[0] == 1:
            visual_positions = visual_positions[0]
        if visual_positions.shape != record.visual_rest_vertices_m.shape:
            raise GenesisLiveError(
                "invalid_visual_overlay",
                "visual_overlay trace readback has an unexpected shape",
                details={
                    "entity": entity_name,
                    "shape": list(visual_positions.shape),
                    "expected_shape": list(record.visual_rest_vertices_m.shape),
                },
            )
        binding = record.physics_vertex_indices
        expected = physical_positions[binding]
        binding_errors = np.max(np.abs(visual_positions - expected), axis=1)
        unique_indices, counts = np.unique(binding, return_counts=True)
        duplicate_physics_indices = unique_indices[counts > 1]
        max_duplicate_error = 0.0
        duplicate_visual_vertex_count = 0
        duplicate_samples = []
        for physics_index in duplicate_physics_indices:
            visual_indices = np.flatnonzero(binding == physics_index)
            duplicate_visual_vertex_count += len(visual_indices)
            positions = visual_positions[visual_indices]
            group_error = float(np.max(np.abs(positions - positions[0])))
            max_duplicate_error = max(max_duplicate_error, group_error)
            if len(duplicate_samples) < 16:
                duplicate_samples.append(
                    {
                        "physics_vertex_index": int(physics_index),
                        "visual_vertex_indices": visual_indices.astype(int).tolist(),
                        "max_pairwise_error_m": group_error,
                    }
                )
        return {
            "schema_version": "genesis-live-visual-overlay-trace-v1",
            "entity": entity_name,
            "current_step": int(self.current_step),
            "physics_vertex_count": int(len(physical_positions)),
            "visual_vertex_count": int(len(visual_positions)),
            "binding_count": int(len(binding)),
            "max_binding_error_m": float(np.max(binding_errors)) if len(binding_errors) else 0.0,
            "duplicate_physics_vertex_count": int(len(duplicate_physics_indices)),
            "duplicate_visual_vertex_count": int(duplicate_visual_vertex_count),
            "max_seam_duplicate_error_m": float(max_duplicate_error),
            "duplicate_samples": duplicate_samples,
        }

    def _visual_result_from_frames(
        self,
        visual,
        frames: list[dict[str, Any]],
        *,
        render_every_steps: int | None,
        steps_requested: int,
    ) -> dict[str, Any]:
        if visual is None:
            return {"requested": False}
        mode = visual.get("mode", "rgb_triptych")
        human_mode = {
            "rgb_triptych": "RGB triptych",
            "depth": "depth triptych",
            "depth_triptych": "depth triptych",
            "normal": "normal triptych",
            "normal_triptych": "normal triptych",
            "part_segmentation_triptych": "part segmentation triptych",
            "part_shaded_triptych": "part shaded triptych",
            "fixed_rgb_views": "fixed RGB views",
        }[mode]
        if not frames:
            return {
                "requested": True,
                "mode": mode,
                "rendered": False,
                "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                "renderer": {
                    "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "mode": mode,
                    "debug_camera": mode != "fixed_rgb_views",
                    "reason": f"no {human_mode} frame boundary reached during this resume",
                },
                "render_every_steps": int(render_every_steps or DEFAULT_RENDER_EVERY_STEPS),
                "steps_requested": int(steps_requested),
                "frames": [],
                "frame_metadata": [],
                "count": 0,
                "reason": f"no {human_mode} frame boundary reached during this resume",
            }
        result = dict(frames[-1])
        result["frames"] = frames
        result["count"] = len(frames)
        result["render_every_steps"] = int(render_every_steps or DEFAULT_RENDER_EVERY_STEPS)
        result["steps_requested"] = int(steps_requested)
        return result
