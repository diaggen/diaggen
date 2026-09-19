from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hag4r.tools.genesis.diagnostic_timing import (
    DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
    DIAGNOSTIC_SCENE_TIMESTEP_S,
)

GENESIS_TEST_SCHEMA_VERSION = "hag4r-agentic-genesis-diagnostic-scene-v1"
DEFAULT_GENESIS_ROOT = Path("third_party/genesis-world")
DEFAULT_GROUND_CLEARANCE_M = 0.02
PART_SEGMENTATION_CONTEXT_PALETTE = {
    "background": [0, 0, 0],
    "fixture": [41, 110, 255],
    "probe": [255, 89, 41],
}


@dataclass(frozen=True)
class AssetSceneFit:
    scale: float
    translation: tuple[float, float, float]
    camera_position: tuple[float, float, float]
    camera_target: tuple[float, float, float]


@dataclass(frozen=True)
class MeshVertexGeometry:
    path: Path
    mesh_format: str
    vertices: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class GenesisAgenticConfigBundle:
    run_id: str
    target_config_path: Path
    model_config_path: Path
    body_config_path: Path
    output_dir: Path
    config_payload: dict[str, Any]
    model_config_payload: dict[str, Any]
    body_config_payload: dict[str, Any]


@dataclass(frozen=True)
class GenesisConfigWriteResult:
    target_config_path: Path
    model_config_path: Path
    body_config_path: Path
    written_paths: tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_for_repo(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else _repo_root() / path


def read_mesh_vertex_geometry(path: Path) -> MeshVertexGeometry | None:
    resolved = _path_for_repo(path)
    assert resolved is not None
    if not resolved.exists():
        return None

    vertices: list[tuple[float, float, float]] = []
    suffix = resolved.suffix.lower()
    if suffix == ".mesh":
        with resolved.open("r", encoding="utf-8") as handle:
            lines = iter(handle)
            for line in lines:
                if line.strip() != "Vertices":
                    continue
                count = int(next(lines).strip())
                for _ in range(count):
                    parts = next(lines).split()
                    vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                break
    elif suffix == ".obj":
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not vertices:
        return None
    return MeshVertexGeometry(path=resolved, mesh_format=suffix, vertices=tuple(vertices))


def _bounds_from_vertices(
    vertices: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maxs = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return mins, maxs


def _read_asset_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    geometry = read_mesh_vertex_geometry(path)
    if geometry is None:
        return None
    return _bounds_from_vertices(geometry.vertices)


def fit_asset_for_diagnostic_scene(asset_mesh_path: Path) -> AssetSceneFit:
    bounds = _read_asset_bounds(asset_mesh_path)
    if bounds is None:
        return AssetSceneFit(
            scale=1.0,
            translation=(0.0, 0.0, 0.0),
            camera_position=(0.9, 0.7, 3.2),
            camera_target=(0.0, 0.5, 0.0),
        )

    mins, maxs = bounds
    extents = tuple(maxs[axis] - mins[axis] for axis in range(3))
    max_extent = max(extents)
    scale = 1.0
    raw_center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    translation = (
        -raw_center[0] * scale,
        DEFAULT_GROUND_CLEARANCE_M - mins[1] * scale,
        -raw_center[2] * scale,
    )
    fitted_center = (
        raw_center[0] * scale + translation[0],
        raw_center[1] * scale + translation[1],
        raw_center[2] * scale + translation[2],
    )
    fitted_extent = max(max_extent * scale, 0.1)
    camera_position = (
        fitted_center[0] + max(0.35, fitted_extent),
        fitted_center[1] + max(0.45, fitted_extent * 1.5),
        fitted_center[2] + max(1.2, fitted_extent * 4.0),
    )
    return AssetSceneFit(
        scale=scale,
        translation=translation,
        camera_position=camera_position,
        camera_target=fitted_center,
    )


def mesh_to_env_transform(asset_scene_fit: AssetSceneFit) -> dict[str, list[float] | str]:
    """Return the transform applied to a diagnostic Genesis mesh."""

    return {
        "source": "fit_asset_for_diagnostic_scene",
        "scale": [float(asset_scene_fit.scale)] * 3,
        "translation_m": [float(value) for value in asset_scene_fit.translation],
        "rotation": [0.0, 0.0, 0.0],
    }


def _validated_mesh_to_env_transform(
    transform: Mapping[str, Any] | None,
    asset_scene_fit: AssetSceneFit,
) -> dict[str, Any]:
    candidate = mesh_to_env_transform(asset_scene_fit) if transform is None else dict(transform)
    scale = candidate.get("scale")
    translation = candidate.get("translation_m", candidate.get("translation"))
    rotation = candidate.get("rotation", [0.0, 0.0, 0.0])
    if (
        not isinstance(scale, (list, tuple))
        or len(scale) != 3
        or not isinstance(translation, (list, tuple))
        or len(translation) != 3
        or not isinstance(rotation, (list, tuple))
        or len(rotation) != 3
    ):
        raise ValueError("mesh_to_env_transform requires scale, translation_m, and rotation vec3 values")
    values = [*scale, *translation, *rotation]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) for value in values):
        raise ValueError("mesh_to_env_transform values must be finite numbers")
    if any(float(value) <= 0.0 for value in scale):
        raise ValueError("mesh_to_env_transform scale values must be positive")
    if any(float(value) != 0.0 for value in rotation):
        raise ValueError("v2 mesh_to_env_transform rotation must be zero; Genesis TetMesh does not apply euler")
    return {
        "source": str(candidate.get("source", "persisted")),
        "scale": [float(value) for value in scale],
        "translation_m": [float(value) for value in translation],
        "rotation": [float(value) for value in rotation],
    }


def transform_points_to_env(points: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    """Apply the persisted Genesis mesh-to-environment transform to points."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("transform_points_to_env expects an (N, 3) point array")
    scale = np.asarray(transform["scale"], dtype=np.float64)
    translation = np.asarray(transform["translation_m"], dtype=np.float64)
    rotation = np.asarray(transform.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64)
    if scale.shape != (3,) or translation.shape != (3,) or rotation.shape != (3,) or not np.all(np.isfinite(np.concatenate([scale, translation, rotation]))) or np.any(scale <= 0.0):
        raise ValueError("mesh_to_env_transform is malformed")
    if np.any(rotation != 0.0):
        raise ValueError("v2 mesh_to_env_transform rotation must be zero; Genesis TetMesh does not apply euler")
    return array * scale + translation


def transform_box_to_env(box: list[float] | tuple[float, ...], transform: Mapping[str, Any]) -> list[float]:
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values[:3] >= values[3:]):
        raise ValueError("box must be a finite strict six-value AABB")
    corners = np.asarray(
        [[values[0] if x == 0 else values[3], values[1] if y == 0 else values[4], values[2] if z == 0 else values[5]] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.float64,
    )
    transformed = transform_points_to_env(corners, transform)
    return np.concatenate([transformed.min(axis=0), transformed.max(axis=0)]).astype(float).tolist()


def diagnostic_asset_geometry_payload(asset_mesh_path: Path) -> dict[str, Any]:
    resolved_asset_mesh_path = _path_for_repo(asset_mesh_path)
    assert resolved_asset_mesh_path is not None
    bounds = _read_asset_bounds(resolved_asset_mesh_path)
    asset_scene_fit = fit_asset_for_diagnostic_scene(resolved_asset_mesh_path)
    payload: dict[str, Any] = {
        "unit_system": "SI",
        "coordinate_frame": "hag4r_metric_mesh_meters",
        "units": {
            "time": "s",
            "length": "m",
            "surface_area": "m^2",
            "volume": "m^3",
            "density": "kg/m^3",
            "mass": "kg",
        },
        "metric_bounds": {"available": False, "unit": "m", "min": [], "max": [], "extents": []},
        "diagnostic_scale": [1.0, 1.0, 1.0],
        "diagnostic_transform": {
            "scale": [1.0, 1.0, 1.0],
            "translation_m": list(asset_scene_fit.translation),
        },
        "diagnostic_camera": {
            "position_m": list(asset_scene_fit.camera_position),
            "target_m": list(asset_scene_fit.camera_target),
        },
        "raw_bounds": {"available": False, "unit": "m", "min": [], "max": []},
        "scene_fit": {
            "scale": asset_scene_fit.scale,
            "translation": list(asset_scene_fit.translation),
            "camera_position": list(asset_scene_fit.camera_position),
            "camera_target": list(asset_scene_fit.camera_target),
        },
        "fitted_bounds": {"available": False, "unit": "m", "min": [], "max": []},
    }
    if bounds is None:
        return payload

    mins, maxs = bounds
    extents = [maxs[axis] - mins[axis] for axis in range(3)]
    fitted_mins = [
        mins[axis] * asset_scene_fit.scale + asset_scene_fit.translation[axis]
        for axis in range(3)
    ]
    fitted_maxs = [
        maxs[axis] * asset_scene_fit.scale + asset_scene_fit.translation[axis]
        for axis in range(3)
    ]
    payload["metric_bounds"] = {
        "available": True,
        "unit": "m",
        "min": list(mins),
        "max": list(maxs),
        "extents": extents,
    }
    payload["raw_bounds"] = {"available": True, "unit": "m", "min": list(mins), "max": list(maxs)}
    payload["fitted_bounds"] = {"available": True, "unit": "m", "min": fitted_mins, "max": fitted_maxs}
    return payload


def _fixed_anchor_payloads(
    fixed_pin_boxes: tuple[dict[str, Any], ...],
    *,
    mesh_to_env: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for index, pin_box in enumerate(fixed_pin_boxes):
        if pin_box.get("enabled", True) is False:
            continue
        box = pin_box.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 6:
            raise ValueError("fixed_pin_boxes entries must include a six-value box")
        if mesh_to_env is not None:
            box = transform_box_to_env(list(box), mesh_to_env)
        anchors.append(
            {
                "anchor_id": str(pin_box.get("anchor_id") or pin_box.get("name") or f"agent_fixed_anchor_{index}"),
                "name": str(pin_box.get("name") or f"agent_fixed_anchor_{index}"),
                "frame": "env_local",
                "box": [float(value) for value in box],
                "source": str(pin_box.get("source") or "hag4r_agentic_fixed_anchor_box"),
            }
        )
    return anchors


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    resolved = _path_for_repo(path)
    assert resolved is not None
    if not resolved.exists():
        raise FileNotFoundError(f"{name} JSON file does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {resolved}")
    return payload


def _numeric_stats(values: np.ndarray) -> dict[str, float]:
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
    }


def _part_segmentation_payload(
    *,
    primitive_labels_path: Path,
    part_labels_path: Path | None,
    inferred_params_path: Path | None,
) -> dict[str, Any]:
    if part_labels_path is None:
        raise ValueError("part_labels_path is required for Genesis part segmentation")
    if inferred_params_path is None:
        raise ValueError("inferred_params_path is required for Genesis part segmentation")
    for label, path in (
        ("primitive labels", primitive_labels_path),
        ("part palette", part_labels_path),
        ("inferred part names", inferred_params_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")

    primitive_labels_key = "tet_part_labels"
    with np.load(primitive_labels_path, allow_pickle=False) as payload:
        if primitive_labels_key not in payload.files:
            raise ValueError(f"{primitive_labels_path} is missing {primitive_labels_key}")
        primitive_labels = np.asarray(payload[primitive_labels_key])
    if primitive_labels.ndim != 1 or not np.issubdtype(primitive_labels.dtype, np.integer):
        raise ValueError(f"{primitive_labels_key} must be a one-dimensional integer array")
    if primitive_labels.size == 0 or np.any(primitive_labels < 0):
        raise ValueError(f"{primitive_labels_key} must be non-empty and non-negative")

    with np.load(part_labels_path, allow_pickle=False) as payload:
        if "part_colors" not in payload.files:
            raise ValueError(f"{part_labels_path} is missing part_colors")
        colors = np.asarray(payload["part_colors"])
    if colors.ndim != 2 or colors.shape[1] not in {3, 4} or colors.shape[0] == 0:
        raise ValueError(f"part_colors must have shape (N, 3) or (N, 4), got {colors.shape}")
    colors = colors[:, :3]
    if not np.issubdtype(colors.dtype, np.number) or not np.all(np.isfinite(colors)):
        raise ValueError("part_colors must contain finite numeric RGB values")
    rounded_colors = np.rint(colors).astype(np.int64)
    if not np.allclose(colors, rounded_colors, atol=1.0e-6, rtol=0.0) or np.any((rounded_colors < 0) | (rounded_colors > 255)):
        raise ValueError("part_colors must contain integer-valued RGB values in [0, 255]")
    if any(np.array_equal(color, np.zeros(3, dtype=np.int64)) for color in rounded_colors):
        raise ValueError("part_colors may not contain black because black is reserved for background")
    if len({tuple(int(value) for value in color) for color in rounded_colors}) != rounded_colors.shape[0]:
        raise ValueError("part_colors must be unique")

    inferred = _read_json_object(inferred_params_path, name="inferred params")
    predictions = inferred.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("inferred params must contain a non-empty predictions list for part names")
    parts_by_id: dict[int, dict[str, Any]] = {}
    for fallback_id, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"inferred prediction {fallback_id} must be an object")
        part_id = int(prediction.get("part_index", fallback_id))
        part_name = str(prediction.get("part_name", "")).strip()
        if not part_name:
            raise ValueError(f"inferred prediction {fallback_id} is missing part_name")
        if part_id in parts_by_id:
            raise ValueError(f"inferred params contains duplicate part_index {part_id}")
        parts_by_id[part_id] = prediction

    present_ids = sorted(int(value) for value in np.unique(primitive_labels))
    unknown_ids = [part_id for part_id in present_ids if part_id not in parts_by_id or part_id >= rounded_colors.shape[0]]
    if unknown_ids:
        raise ValueError(f"{primitive_labels_key} references unknown part ids: {unknown_ids}")
    parts = [
        {
            "part_id": part_id,
            "part_name": str(parts_by_id[part_id]["part_name"]),
            "part_color_rgb": [int(value) for value in rounded_colors[part_id]],
        }
        for part_id in present_ids
    ]
    context_colors: dict[str, list[int]] = {}
    for kind, raw_color in PART_SEGMENTATION_CONTEXT_PALETTE.items():
        color = np.asarray(raw_color)
        if color.shape != (3,) or not np.issubdtype(color.dtype, np.integer):
            raise ValueError(f"context palette color {kind!r} must be an integer RGB triplet")
        if np.any((color < 0) | (color > 255)):
            raise ValueError(f"context palette color {kind!r} must be in [0, 255]")
        context_colors[kind] = [int(value) for value in color]
    if context_colors["background"] != [0, 0, 0]:
        raise ValueError("context palette background must be black")
    context_tuples = {tuple(color) for color in context_colors.values()}
    if len(context_tuples) != len(context_colors):
        raise ValueError("context palette colors must be unique")
    asset_tuples = {tuple(int(value) for value in color) for color in rounded_colors}
    conflicts = sorted(context_tuples.intersection(asset_tuples))
    if conflicts:
        raise ValueError(f"context palette colors conflict with asset part colors: {conflicts}")
    return {
        "primitive_labels_file": str(primitive_labels_path),
        "primitive_labels_key": primitive_labels_key,
        "palette_file": str(part_labels_path),
        "palette_key": "part_colors",
        "parts": parts,
        "context_palette": context_colors,
    }


def validate_part_colors_unique(part_labels_path: Path) -> dict[str, Any]:
    """Validate Genesis's part identity color rule without enabling debug rendering."""

    path = Path(part_labels_path)
    if not path.exists():
        raise FileNotFoundError(f"part palette path does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "part_colors" not in payload.files:
            raise ValueError(f"{path} is missing part_colors")
        colors = np.asarray(payload["part_colors"])
    if colors.ndim != 2 or colors.shape[1] not in {3, 4} or colors.shape[0] == 0:
        raise ValueError(f"part_colors must have shape (N, 3) or (N, 4), got {colors.shape}")
    colors = colors[:, :3]
    if not np.issubdtype(colors.dtype, np.number) or not np.all(np.isfinite(colors)):
        raise ValueError("part_colors must contain finite numeric RGB values")
    rounded_colors = np.rint(colors).astype(np.int64)
    if not np.allclose(colors, rounded_colors, atol=1.0e-6, rtol=0.0) or np.any((rounded_colors < 0) | (rounded_colors > 255)):
        raise ValueError("part_colors must contain integer-valued RGB values in [0, 255]")
    if any(np.array_equal(color, np.zeros(3, dtype=np.int64)) for color in rounded_colors):
        raise ValueError("part_colors may not contain black because black is reserved for background")
    unique = {tuple(int(value) for value in color) for color in rounded_colors}
    if len(unique) != rounded_colors.shape[0]:
        raise ValueError("part_colors must be unique")
    return {
        "status": "valid",
        "part_color_count": int(rounded_colors.shape[0]),
        "unique_part_color_count": int(len(unique)),
    }


def build_agentic_genesis_config_bundle(
    *,
    run_id: str,
    asset_mesh_path: Path,
    material_params_path: Path,
    part_labels_path: Path | None = None,
    inferred_params_path: Path | None = None,
    volume_topology_path: Path | None = None,
    output_dir: Path | None = None,
    target_config_path: Path | None = None,
    diagnostic_cues: tuple[dict[str, Any], ...] = (),
    fixed_pin_boxes: tuple[dict[str, Any], ...] = (),
    include_part_segmentation: bool = True,
    mesh_to_env_transform: Mapping[str, Any] | None = None,
    box_ee_controller_policy: Mapping[str, Any] | None = None,
) -> GenesisAgenticConfigBundle:
    if not isinstance(include_part_segmentation, bool):
        raise TypeError("include_part_segmentation must be a boolean")
    resolved_asset_mesh_path = _path_for_repo(asset_mesh_path)
    resolved_material_params_path = _path_for_repo(material_params_path)
    resolved_part_labels_path = _path_for_repo(part_labels_path)
    resolved_inferred_params_path = _path_for_repo(inferred_params_path)
    resolved_volume_topology_path = _path_for_repo(volume_topology_path)
    assert resolved_asset_mesh_path is not None
    assert resolved_material_params_path is not None
    asset_scene_fit = fit_asset_for_diagnostic_scene(resolved_asset_mesh_path)
    applied_transform = _validated_mesh_to_env_transform(mesh_to_env_transform, asset_scene_fit) if not include_part_segmentation else None
    scene_fit_scale: Any = asset_scene_fit.scale if applied_transform is None else list(applied_transform["scale"])
    scene_fit_translation = list(asset_scene_fit.translation) if applied_transform is None else list(applied_transform["translation_m"])

    resolved_target_config_path = _path_for_repo(target_config_path)
    if resolved_target_config_path is None:
        raise ValueError("target_config_path is required for Genesis diagnostic config generation")
    config_dir = resolved_target_config_path.parent
    resolved_output_dir = _path_for_repo(output_dir) if output_dir is not None else config_dir / "live_output"
    assert resolved_output_dir is not None
    model_config_path = config_dir / "model_deformable.json"
    body_config_path = config_dir / "generated_asset.json"
    anchors = _fixed_anchor_payloads(fixed_pin_boxes, mesh_to_env=applied_transform)
    part_segmentation = (
        _part_segmentation_payload(
            primitive_labels_path=resolved_material_params_path,
            part_labels_path=resolved_part_labels_path,
            inferred_params_path=resolved_inferred_params_path,
        )
        if include_part_segmentation
        else None
    )
    topology_payload = (
        _read_json_object(resolved_volume_topology_path, name="volume_topology.json")
        if resolved_volume_topology_path is not None and resolved_volume_topology_path.exists()
        else {}
    )
    if box_ee_controller_policy is not None:
        required_policy_fields = {
            "mode",
            "policy_id",
            "total_stiffness_n_per_m",
            "max_net_spring_force_n",
        }
        policy = dict(box_ee_controller_policy)
        if set(policy) != required_policy_fields:
            raise ValueError("box_ee_controller_policy must contain exactly the Genesis force-limited policy fields")
        if policy["mode"] != "native_fem_force_limited_box_ee/v1" or not isinstance(policy["policy_id"], str) or not policy["policy_id"].strip():
            raise ValueError("box_ee_controller_policy mode and policy_id are invalid")
        for field_name in ("total_stiffness_n_per_m", "max_net_spring_force_n"):
            value = policy[field_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"box_ee_controller_policy.{field_name} must be finite and > 0")
        policy = {
            "mode": policy["mode"],
            "policy_id": policy["policy_id"],
            "total_stiffness_n_per_m": float(policy["total_stiffness_n_per_m"]),
            "max_net_spring_force_n": float(policy["max_net_spring_force_n"]),
        }
    else:
        policy = None

    body_config_payload: dict[str, Any] = {
        "schema_version": "hag4r-genesis-diagnostic-body-v1",
        "mesh": str(resolved_asset_mesh_path),
        "element_type": "TETRAHEDRON",
        "material_params_path": str(resolved_material_params_path),
        "part_labels_path": str(resolved_part_labels_path) if resolved_part_labels_path else "",
        "inferred_params_path": str(resolved_inferred_params_path) if resolved_inferred_params_path else "",
        "volume_topology_path": str(resolved_volume_topology_path) if resolved_volume_topology_path else "",
        "scene_fit": {
            "scale": scene_fit_scale,
            "translation": scene_fit_translation,
            "camera_position": list(asset_scene_fit.camera_position),
            "camera_target": list(asset_scene_fit.camera_target),
            "mesh_to_env_local": applied_transform,
        },
        "anchors": anchors,
    }
    model_config_payload = {
        "schema_version": "hag4r-genesis-diagnostic-model-v1",
        "name": f"{run_id}_tet_genesis_agentic",
        "bodies": [str(body_config_path)],
    }
    config_payload: dict[str, Any] = {
        "schema_version": GENESIS_TEST_SCHEMA_VERSION,
        "run_id": run_id,
        "backend": "cuda",
        "timestep": DIAGNOSTIC_SCENE_TIMESTEP_S,
        "sim_options": {"dt": DIAGNOSTIC_SCENE_TIMESTEP_S},
        "fem_options": {"enable_vertex_constraints": True, "use_implicit_solver": True},
        "render_png": {"capture_every_n_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS},
        "diagnostic_timing": {
            "step_dt_s": DIAGNOSTIC_SCENE_TIMESTEP_S,
            "render_every_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            "frame_dt_s": DIAGNOSTIC_SCENE_TIMESTEP_S * DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            "step_semantics": "smallest_simulation_stepping_unit",
            "frame_semantics": "smallest_rendering_unit",
        },
        "entities": [
            {
                "name": "body",
                "morph": {
                    "type": "tet_mesh",
                    "file": str(resolved_asset_mesh_path),
                    **(
                        {
                            "scale": list(applied_transform["scale"]),
                            "pos": list(applied_transform["translation_m"]),
                            "euler": list(applied_transform["rotation"]),
                        }
                        if applied_transform is not None
                        else {}
                    ),
                },
                "material": {
                    "type": "elastic",
                    "E": 1.0e6,
                    "nu": 0.3,
                    "rho": 1000.0,
                    "heterogeneous": {"file": str(resolved_material_params_path)},
                },
                "anchors": anchors,
                **({"part_segmentation": part_segmentation} if include_part_segmentation else {}),
            }
        ],
        "agentic_diagnostics": {
            "schema_version": GENESIS_TEST_SCHEMA_VERSION,
            "mode": "structured_live_diagnostic",
            "control_mode": "structured_live",
            "diagnostic_cues": list(diagnostic_cues),
            "asset": {
                "mesh_path": str(resolved_asset_mesh_path),
                "material_params_path": str(resolved_material_params_path),
                "part_labels_path": str(resolved_part_labels_path) if resolved_part_labels_path else "",
                "inferred_params_path": str(resolved_inferred_params_path) if resolved_inferred_params_path else "",
                "volume_topology_path": str(resolved_volume_topology_path) if resolved_volume_topology_path else "",
                "volume_topology": topology_payload,
                "part_identity_required": True,
                "mesh_to_env_local": applied_transform,
            },
            "outputs": {
                "directory": str(resolved_output_dir),
                "part_segmentation_triptych": bool(include_part_segmentation),
                "depth": False,
                "von_mises": False,
                "render_every_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            },
            "config_time_anchors": {
                "enabled": bool(anchors),
                "count": len(anchors),
                "names": [anchor["name"] for anchor in anchors],
                "boxes": anchors,
            },
        },
        "agentic_live": {
            "protocol": "genesis-live-v1",
            "requested_telemetry": [
                "geometry",
                "deformation",
                "controllers",
                "material",
                *(["part_segmentation_triptych"] if include_part_segmentation else ["fixed_rgb_views"]),
            ],
            "unsupported_telemetry": ["depth", "von_mises"],
            "step_dt_s": DIAGNOSTIC_SCENE_TIMESTEP_S,
            "render_every_steps": DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
            "probe_actions": [
                "box_ee_grasp_and_move",
                "probe_release",
            ],
            "box_ee_contract": {
                "allowed_actions": ["box_ee_grasp_and_move"],
                "grasp_behavior": "gentle_compliant_probe",
                "compliance_control": "backend_owned",
                "movement": "aabb_box_distance_scale",
                "box_frame": "env_local",
                "hardcoded_direction": [0.0, 1.0, 0.0],
                "hardcoded_speed_m_s": 0.6,
            },
            "diagnostics_only": True,
        },
    }
    if policy is not None:
        config_payload["box_ee_controller_policy"] = policy
    return GenesisAgenticConfigBundle(
        run_id=run_id,
        target_config_path=resolved_target_config_path,
        model_config_path=model_config_path,
        body_config_path=body_config_path,
        output_dir=resolved_output_dir,
        config_payload=config_payload,
        model_config_payload=model_config_payload,
        body_config_payload=body_config_payload,
    )


def write_agentic_genesis_config_bundle(
    bundle: GenesisAgenticConfigBundle,
) -> GenesisConfigWriteResult:
    written_paths = (bundle.target_config_path, bundle.model_config_path, bundle.body_config_path)
    for path, payload in (
        (bundle.target_config_path, bundle.config_payload),
        (bundle.model_config_path, bundle.model_config_payload),
        (bundle.body_config_path, bundle.body_config_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return GenesisConfigWriteResult(
        target_config_path=bundle.target_config_path,
        model_config_path=bundle.model_config_path,
        body_config_path=bundle.body_config_path,
        written_paths=written_paths,
    )


__all__ = [
    "DEFAULT_GENESIS_ROOT",
    "GENESIS_TEST_SCHEMA_VERSION",
    "GenesisAgenticConfigBundle",
    "GenesisConfigWriteResult",
    "build_agentic_genesis_config_bundle",
    "diagnostic_asset_geometry_payload",
    "fit_asset_for_diagnostic_scene",
    "mesh_to_env_transform",
    "transform_box_to_env",
    "transform_points_to_env",
    "validate_part_colors_unique",
    "read_mesh_vertex_geometry",
    "write_agentic_genesis_config_bundle",
]
