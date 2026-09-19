from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import igl
import numpy as np
import trimesh
from scipy import ndimage
from scipy import sparse
from scipy import spatial
from scipy.sparse import csgraph


VOLUME_MESH_FIDELITY_TABLE: dict[str, dict[str, float | int | str]] = {
    "high": {
        "agent_semantics": "Thin/hollow, contact-critical, small-feature, or diagnostics-detail-sensitive part.",
        "compression": "aggressive compression",
        "keep_ratio": 0.02,
        "absolute_cap": 10_000,
        "longest_axis_voxels": 96,
    },
    "medium": {
        "agent_semantics": "Ordinary structure or visible but non-contact-critical part.",
        "compression": "super-aggressive compression",
        "keep_ratio": 0.005,
        "absolute_cap": 5_000,
        "longest_axis_voxels": 96,
    },
    "low": {
        "agent_semantics": "Simple bulky, low-visible, low-interaction, or safely extreme-compression part.",
        "compression": "extreme-aggressive compression",
        "keep_ratio": 0.001,
        "absolute_cap": 2_000,
        "longest_axis_voxels": 96,
    },
}

HOLLOW_WALL_BAND_LAYERS = 2
TET_BUDGET_LIMIT = 40_000
DENSE_GRID_CELL_CAP = 128_000_000
VOXEL_GRID_PADDING_CELLS = HOLLOW_WALL_BAND_LAYERS + 2
VOLUME_TOPOLOGY_SCHEMA_VERSION = "hag4r-volume-topology-v3"
METRIC_MAX_DIMENSION_SCALING_SCHEMA_VERSION = "hag4r-metric-max-dimension-scaling-v1"
PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION = "hag4r-per-part-volumetric-meshing-request-v3"
MESH_PROCESSING_PLAN_SCHEMA_VERSION = "hag4r-mesh-processing-plan-v3"
MESH_PROCESSING_PLAN_CONTRACT = "per_part_filled_hollow_volumetric_meshing"
TET_COMPONENT_RESOLUTION_SCHEMA_VERSION = "tet-component-resolution-v1"

FILL_MODE_CODES = {"solid_fill": 0, "hollow_wall": 1}
SURFACE_COMBINATION_STRATEGIES = {"concatenate", "manifold_union"}
FORBIDDEN_AGENT_MESH_NUMERIC_KEYS = {
    "target_faces",
    "keep_ratio",
    "absolute_cap",
    "voxel_budget",
    "voxel_pitch",
    "sdf_grid_res",
    "edge_length_abs",
    "edge_length_fac",
    "wall_" + "thickness_m",
    "shell_" + "thickness_m",
}
FORBIDDEN_MATERIAL_KEYS = {
    "representation",
    "representation_decision",
    "shell_" + "thickness_m",
    "wall_" + "thickness_m",
}
LEGACY_ARRAY_PREFIX = "t" + "ri_"


@dataclass(frozen=True)
class MeshFidelityConfig:
    fidelity: str
    keep_ratio: float
    absolute_cap: int
    longest_axis_voxels: int


@dataclass(frozen=True)
class PartTopologySpec:
    part_index: int
    part_color_rgb: tuple[float, float, float]
    volume_fill_mode: str
    mesh_fidelity: str
    fidelity_rationale: str
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    friction_coefficient: float


@dataclass(frozen=True)
class VoxelGridSpec:
    origin: np.ndarray
    pitch_m: float
    dims: tuple[int, int, int]
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass
class PartOccupancy:
    spec: PartTopologySpec
    fidelity_config: MeshFidelityConfig
    source_mesh_path: Path
    source_mesh: trimesh.Trimesh
    grid: VoxelGridSpec
    occupancy: np.ndarray
    occupied_centers: np.ndarray
    input_faces: int = 0
    target_faces: int = 0
    output_faces: int = 0
    fill_operation: str = ""
    tet_count: int = 0


@dataclass(frozen=True)
class PerPartVolumetricMeshingRequest:
    input_mesh_dir: Path
    part_labels_path: Path
    inferred_params_path: Path
    partwise_params_path: Path
    mesh_processing_plan: dict[str, Any]
    output_mesh_path: Path
    heterogeneous_params_path: Path
    metric_mesh_scaling_path: Path
    volume_topology_path: Path
    repo_root: Path
    surface_combination_strategy: str = "manifold_union"
    request_path: Path | None = None


@dataclass(frozen=True)
class PerPartVolumetricMeshingResult:
    mesh_path: Path
    heterogeneous_params_path: Path
    metric_mesh_scaling_path: Path
    volume_topology_path: Path
    tet_count: int
    tet_budget_status: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TetComponentResolutionPaths:
    report_path: Path
    candidate_npz_path: Path


class SurfaceCombinationError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


class TetMeshValidationError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


class DisconnectedTetComponentsError(TetMeshValidationError):
    pass


def fidelity_config(fidelity: str) -> MeshFidelityConfig:
    key = str(fidelity).strip().lower()
    if key not in VOLUME_MESH_FIDELITY_TABLE:
        raise ValueError(f"mesh_fidelity must be one of {', '.join(VOLUME_MESH_FIDELITY_TABLE)}; got {fidelity!r}")
    raw = VOLUME_MESH_FIDELITY_TABLE[key]
    return MeshFidelityConfig(
        fidelity=key,
        keep_ratio=float(raw["keep_ratio"]),
        absolute_cap=int(raw["absolute_cap"]),
        longest_axis_voxels=int(raw["longest_axis_voxels"]),
    )


def target_faces_for_fidelity(input_faces: int, fidelity: str) -> int:
    count = int(input_faces)
    if count < 0:
        raise ValueError("input_faces must be non-negative")
    config = fidelity_config(fidelity)
    return min(count, max(32, min(int(math.ceil(count * config.keep_ratio)), config.absolute_cap)))


def tet_budget_status(tet_count: int) -> tuple[str, list[str]]:
    count = int(tet_count)
    if count <= TET_BUDGET_LIMIT:
        return "within_budget", []
    return "over_budget", [f"tet count {count} exceeds advisory budget {TET_BUDGET_LIMIT}; continuing"]


def derive_volume_topology(fill_modes: Sequence[str]) -> str:
    modes = {str(mode) for mode in fill_modes}
    if not modes:
        raise ValueError("cannot derive volume topology from an empty fill-mode list")
    if modes == {"solid_fill"}:
        return "filled"
    if modes == {"hollow_wall"}:
        return "hollow"
    if modes <= {"solid_fill", "hollow_wall"}:
        return "mixed"
    raise ValueError(f"unknown volume_fill_mode values: {sorted(modes)}")


def _as_color_tuple(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"part_color_rgb must contain exactly three values; got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _color_equal(a: Any, b: Any) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def _check_no_forbidden_prediction_keys(prediction: dict[str, Any]) -> None:
    for key in prediction:
        if key in FORBIDDEN_MATERIAL_KEYS or key.startswith(LEGACY_ARRAY_PREFIX):
            raise ValueError(f"legacy material field is forbidden in mesh processing payload: {key}")


def _load_part_colors(part_labels_path: Path) -> np.ndarray:
    labels = np.load(part_labels_path, allow_pickle=False)
    if "part_colors" not in labels:
        raise KeyError(f"{part_labels_path} is missing required array part_colors")
    colors = np.asarray(labels["part_colors"], dtype=float)
    if colors.ndim != 2 or colors.shape[1] not in (3, 4):
        raise ValueError("part_colors must have shape (N, 3) or (N, 4)")
    if colors.shape[1] == 4:
        if np.any(colors[:, 3] != 255):
            raise ValueError("RGBA part_colors must use alpha=255 for color-lock canonicalization")
        colors = colors[:, :3]
    return colors


def load_color_locked_material_predictions(
    inferred_params_path: Path,
    part_labels_path: Path,
) -> list[PartTopologySpec]:
    colors = _load_part_colors(part_labels_path)
    payload = json.loads(Path(inferred_params_path).read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("inferred_params.json must contain a predictions list")
    seen: set[int] = set()
    specs: list[PartTopologySpec] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("each material prediction must be an object")
        _check_no_forbidden_prediction_keys(prediction)
        idx = int(prediction["part_index"])
        if idx in seen:
            raise ValueError(f"duplicate material prediction for part_index {idx}")
        if idx < 0 or idx >= len(colors):
            raise ValueError(f"part_index {idx} is outside part_colors range")
        color = _as_color_tuple(prediction["part_color_rgb"])
        if not _color_equal(color, colors[idx]):
            raise ValueError(f"part_color_rgb mismatch for part_index {idx}")
        fill_mode = str(prediction["volume_fill_mode"])
        if fill_mode not in FILL_MODE_CODES:
            raise ValueError(f"volume_fill_mode must be solid_fill or hollow_wall for part_index {idx}")
        specs.append(
            PartTopologySpec(
                part_index=idx,
                part_color_rgb=color,
                volume_fill_mode=fill_mode,
                mesh_fidelity="",
                fidelity_rationale="",
                density_kg_m3=float(prediction["density_kg_m3"]),
                youngs_modulus_pa=float(prediction["youngs_modulus_pa"]),
                poisson_ratio=float(prediction["poisson_ratio"]),
                friction_coefficient=float(prediction["friction_coefficient"]),
            )
        )
        seen.add(idx)
    expected = set(range(len(colors)))
    if seen != expected:
        raise ValueError(f"material predictions must cover exact part indices {sorted(expected)}; got {sorted(seen)}")
    return sorted(specs, key=lambda spec: spec.part_index)


def validate_mesh_fidelity_payload(
    *,
    part_fidelities: Sequence[dict[str, Any]],
    indexed_parts: Sequence[PartTopologySpec],
) -> list[PartTopologySpec]:
    by_index = {part.part_index: part for part in indexed_parts}
    expected = set(by_index)
    seen: set[int] = set()
    resolved: list[PartTopologySpec] = []
    for entry in part_fidelities:
        if not isinstance(entry, dict):
            raise ValueError("each part fidelity entry must be an object")
        extra_numeric = FORBIDDEN_AGENT_MESH_NUMERIC_KEYS.intersection(entry)
        if extra_numeric:
            raise ValueError(f"agent-authored numeric mesh fields are forbidden: {sorted(extra_numeric)}")
        idx = int(entry["part_index"])
        if idx in seen:
            raise ValueError(f"duplicate mesh fidelity for part_index {idx}")
        if idx not in by_index:
            raise ValueError(f"mesh fidelity contains unknown part_index {idx}")
        color = _as_color_tuple(entry["part_color_rgb"])
        base = by_index[idx]
        if not _color_equal(color, base.part_color_rgb):
            raise ValueError(f"part_color_rgb mismatch for part_index {idx}")
        mesh_fidelity = str(entry["mesh_fidelity"]).strip().lower()
        fidelity_config(mesh_fidelity)
        rationale = str(entry.get("fidelity_rationale", "")).strip()
        if not rationale:
            raise ValueError(f"fidelity_rationale is required for part_index {idx}")
        resolved.append(
            PartTopologySpec(
                part_index=base.part_index,
                part_color_rgb=base.part_color_rgb,
                volume_fill_mode=base.volume_fill_mode,
                mesh_fidelity=mesh_fidelity,
                fidelity_rationale=rationale,
                density_kg_m3=base.density_kg_m3,
                youngs_modulus_pa=base.youngs_modulus_pa,
                poisson_ratio=base.poisson_ratio,
                friction_coefficient=base.friction_coefficient,
            )
        )
        seen.add(idx)
    if seen != expected:
        raise ValueError(f"mesh fidelities must cover exact part indices {sorted(expected)}; got {sorted(seen)}")
    return sorted(resolved, key=lambda spec: spec.part_index)


def build_mesh_processing_plan(
    *,
    part_fidelities: Sequence[dict[str, Any]],
    indexed_parts: Sequence[PartTopologySpec],
    target_max_dimension_m: float,
    estimate_rationale: str,
    object_semantics: str = "",
    diagnostic_cue_summary: str = "",
    diagnostic_cues: Sequence[dict[str, Any]] = (),
    tool_name: str = "select_mesh_processing_fidelities_stage",
) -> dict[str, Any]:
    target = float(target_max_dimension_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_max_dimension_m must be finite and > 0")
    rationale = estimate_rationale.strip()
    if not rationale:
        raise ValueError("estimate_rationale must be non-empty")
    resolved = validate_mesh_fidelity_payload(part_fidelities=part_fidelities, indexed_parts=indexed_parts)
    parts = []
    for spec in resolved:
        config = fidelity_config(spec.mesh_fidelity)
        parts.append(
            {
                "part_index": spec.part_index,
                "part_color_rgb": list(spec.part_color_rgb),
                "volume_fill_mode": spec.volume_fill_mode,
                "mesh_fidelity": spec.mesh_fidelity,
                "fidelity_rationale": spec.fidelity_rationale,
                "keep_ratio": config.keep_ratio,
                "absolute_cap": config.absolute_cap,
                "longest_axis_voxels": config.longest_axis_voxels,
            }
        )
    return {
        "schema_version": MESH_PROCESSING_PLAN_SCHEMA_VERSION,
        "contract": MESH_PROCESSING_PLAN_CONTRACT,
        "tool_name": tool_name,
        "target_max_dimension_m": target,
        "estimate_rationale": rationale,
        "hollow_wall_band_layers": HOLLOW_WALL_BAND_LAYERS,
        "tet_budget_limit": TET_BUDGET_LIMIT,
        "parts": parts,
        "object_semantics": object_semantics.strip(),
        "diagnostic_cue_summary": diagnostic_cue_summary.strip(),
        "diagnostic_cues": list(diagnostic_cues),
        "warnings": [],
    }


def load_mesh_fidelity_plan(plan: dict[str, Any], indexed_parts: Sequence[PartTopologySpec]) -> list[PartTopologySpec]:
    if plan.get("schema_version") != MESH_PROCESSING_PLAN_SCHEMA_VERSION:
        raise ValueError(f"mesh_processing_plan.schema_version must be {MESH_PROCESSING_PLAN_SCHEMA_VERSION}")
    if plan.get("contract") != MESH_PROCESSING_PLAN_CONTRACT:
        raise ValueError(f"mesh_processing_plan.contract must be {MESH_PROCESSING_PLAN_CONTRACT}")
    if int(plan.get("hollow_wall_band_layers", -1)) != HOLLOW_WALL_BAND_LAYERS:
        raise ValueError(f"hollow_wall_band_layers must be fixed at {HOLLOW_WALL_BAND_LAYERS}")
    by_index = {part.part_index: part for part in indexed_parts}
    expected = set(by_index)
    seen: set[int] = set()
    resolved: list[PartTopologySpec] = []
    for entry in plan.get("parts", ()):
        if not isinstance(entry, dict):
            raise ValueError("each mesh_processing_plan part entry must be an object")
        allowed_derived = {"keep_ratio", "absolute_cap"}
        extra_numeric = FORBIDDEN_AGENT_MESH_NUMERIC_KEYS.intersection(entry) - allowed_derived
        if extra_numeric:
            raise ValueError(f"mesh_processing_plan contains forbidden numeric mesh fields: {sorted(extra_numeric)}")
        idx = int(entry["part_index"])
        if idx in seen:
            raise ValueError(f"duplicate mesh fidelity for part_index {idx}")
        if idx not in by_index:
            raise ValueError(f"mesh fidelity contains unknown part_index {idx}")
        base = by_index[idx]
        if not _color_equal(entry["part_color_rgb"], base.part_color_rgb):
            raise ValueError(f"part_color_rgb mismatch for part_index {idx}")
        if str(entry["volume_fill_mode"]) != base.volume_fill_mode:
            raise ValueError(f"volume_fill_mode mismatch for part_index {idx}")
        mesh_fidelity = str(entry["mesh_fidelity"]).strip().lower()
        config = fidelity_config(mesh_fidelity)
        if float(entry["keep_ratio"]) != config.keep_ratio:
            raise ValueError(f"keep_ratio does not match frozen fidelity table for part_index {idx}")
        if int(entry["absolute_cap"]) != config.absolute_cap:
            raise ValueError(f"absolute_cap does not match frozen fidelity table for part_index {idx}")
        if int(entry["longest_axis_voxels"]) != config.longest_axis_voxels:
            raise ValueError(f"longest_axis_voxels does not match frozen fidelity table for part_index {idx}")
        rationale = str(entry.get("fidelity_rationale", "")).strip()
        if not rationale:
            raise ValueError(f"fidelity_rationale is required for part_index {idx}")
        resolved.append(
            PartTopologySpec(
                part_index=base.part_index,
                part_color_rgb=base.part_color_rgb,
                volume_fill_mode=base.volume_fill_mode,
                mesh_fidelity=mesh_fidelity,
                fidelity_rationale=rationale,
                density_kg_m3=base.density_kg_m3,
                youngs_modulus_pa=base.youngs_modulus_pa,
                poisson_ratio=base.poisson_ratio,
                friction_coefficient=base.friction_coefficient,
            )
        )
        seen.add(idx)
    if seen != expected:
        raise ValueError(f"mesh fidelities must cover exact part indices {sorted(expected)}; got {sorted(seen)}")
    return sorted(resolved, key=lambda spec: spec.part_index)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def tet_component_resolution_paths(output_mesh_path: Path) -> TetComponentResolutionPaths:
    return TetComponentResolutionPaths(
        report_path=output_mesh_path.parent / "component_resolution.json",
        candidate_npz_path=output_mesh_path.parent / "component_resolution_candidate.npz",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _bounds_payload(vertices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(vertices, dtype=np.float64)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    extent = maxs - mins
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extent": extent.tolist(),
        "max_extent": float(np.max(extent)),
    }


def _repo_relative(path: Path, repo_root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    repo_outputs = repo_root / "outputs"
    try:
        output_relative = candidate.resolve().relative_to(repo_outputs.resolve())
        return str(Path("outputs") / output_relative)
    except ValueError:
        pass
    try:
        return str(candidate.absolute().relative_to(repo_root.absolute()))
    except ValueError:
        pass
    resolved = candidate.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_request(path: Path, repo_root: Path) -> PerPartVolumetricMeshingRequest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"request schema_version must be {PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION}"
        )
    return PerPartVolumetricMeshingRequest(
        input_mesh_dir=_resolve_path(payload["input_mesh_dir"], repo_root),
        part_labels_path=_resolve_path(payload["part_labels_path"], repo_root),
        inferred_params_path=_resolve_path(payload["inferred_params_path"], repo_root),
        partwise_params_path=_resolve_path(payload["partwise_params_path"], repo_root),
        mesh_processing_plan=dict(payload["mesh_processing_plan"]),
        output_mesh_path=_resolve_path(payload["output_mesh_path"], repo_root),
        heterogeneous_params_path=_resolve_path(payload["heterogeneous_params_path"], repo_root),
        metric_mesh_scaling_path=_resolve_path(payload["metric_mesh_scaling_path"], repo_root),
        volume_topology_path=_resolve_path(payload["volume_topology_path"], repo_root),
        repo_root=repo_root,
        surface_combination_strategy=str(payload.get("surface_combination_strategy", "manifold_union")),
        request_path=path,
    )


def _sorted_part_paths(input_mesh_dir: Path) -> list[Path]:
    from hag4r.mesh import find_part_glbs

    return find_part_glbs(input_mesh_dir)


def _load_scaled_part_meshes(request: PerPartVolumetricMeshingRequest) -> tuple[list[Path], list[trimesh.Trimesh], dict[str, Any]]:
    from hag4r.mesh import load_part_mesh

    part_paths = _sorted_part_paths(request.input_mesh_dir)
    source_meshes = [load_part_mesh(path) for path in part_paths]
    vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in source_meshes], axis=0)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    extent = bounds_max - bounds_min
    max_extent = float(np.max(extent))
    if not math.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("combined source mesh maximum extent must be positive")
    target = float(request.mesh_processing_plan["target_max_dimension_m"])
    scale_factor = target / max_extent
    scaled_meshes = []
    for mesh in source_meshes:
        scaled_meshes.append(
            trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64) * scale_factor,
                faces=np.asarray(mesh.faces, dtype=np.int64),
                process=False,
            )
        )
    metric_vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in scaled_meshes], axis=0)
    metric_min = metric_vertices.min(axis=0)
    metric_max = metric_vertices.max(axis=0)
    scaling = {
        "schema_version": METRIC_MAX_DIMENSION_SCALING_SCHEMA_VERSION,
        "scaling_mode": "pre_monolithic_target_max_dimension",
        "target_max_dimension_m": target,
        "source_bounds": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
            "extent": extent.tolist(),
            "max_extent": max_extent,
        },
        "scale_factor": float(scale_factor),
        "metric_bounds": {
            "min": metric_min.tolist(),
            "max": metric_max.tolist(),
            "extent": (metric_max - metric_min).tolist(),
            "max_extent": float(np.max(metric_max - metric_min)),
        },
        "note": "Mesh was scaled before voxelization and tetrahedralization.",
    }
    return part_paths, scaled_meshes, scaling


def _unsigned_surface_distance(mesh: trimesh.Trimesh, points: np.ndarray, *, chunk_size: int = 200_000) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    out = np.empty((len(points),), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        sdf = igl.signed_distance(points[start:stop], verts, faces)[0]
        out[start:stop] = np.abs(np.asarray(sdf, dtype=np.float64))
    return out


def _grid_for_mesh(mesh: trimesh.Trimesh, config: MeshFidelityConfig) -> VoxelGridSpec:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    extent = np.maximum(bounds_max - bounds_min, 1e-12)
    longest = float(np.max(extent))
    budget = int(config.longest_axis_voxels)
    if budget < 4:
        raise ValueError("longest_axis_voxels must be at least 4")
    interior_axis_voxels = max(2, budget - 2 * VOXEL_GRID_PADDING_CELLS)
    pitch = longest / float(interior_axis_voxels)
    dims_arr = np.ceil(extent / pitch).astype(int) + 2 * VOXEL_GRID_PADDING_CELLS
    dims_arr = np.maximum(dims_arr, 3)
    cell_count = int(np.prod(dims_arr, dtype=np.int64))
    if cell_count > DENSE_GRID_CELL_CAP:
        raise MemoryError(f"per-part voxel grid has {cell_count} cells, exceeding cap {DENSE_GRID_CELL_CAP}")
    origin = bounds_min - VOXEL_GRID_PADDING_CELLS * pitch
    return VoxelGridSpec(
        origin=origin,
        pitch_m=float(pitch),
        dims=(int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2])),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _grid_centers(grid: VoxelGridSpec) -> np.ndarray:
    nx, ny, nz = grid.dims
    xs = grid.origin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * grid.pitch_m
    ys = grid.origin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * grid.pitch_m
    zs = grid.origin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * grid.pitch_m
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack((gx.ravel(order="C"), gy.ravel(order="C"), gz.ravel(order="C")))


def _occupied_centers(grid: VoxelGridSpec, occupancy: np.ndarray) -> np.ndarray:
    indices = np.argwhere(occupancy)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return grid.origin + (indices.astype(np.float64) + 0.5) * grid.pitch_m


def _build_part_occupancy(
    *,
    spec: PartTopologySpec,
    source_mesh_path: Path,
    mesh: trimesh.Trimesh,
) -> PartOccupancy:
    config = fidelity_config(spec.mesh_fidelity)
    grid = _grid_for_mesh(mesh, config)
    centers = _grid_centers(grid)
    distances = _unsigned_surface_distance(mesh, centers).reshape(grid.dims, order="C")
    if spec.volume_fill_mode == "hollow_wall":
        band_half_width = 0.5 * HOLLOW_WALL_BAND_LAYERS * grid.pitch_m
        occupancy = distances <= band_half_width
        fill_operation = "surface_voxel_band"
    elif spec.volume_fill_mode == "solid_fill":
        surface_mask = distances <= (0.75 * grid.pitch_m)
        occupancy = ndimage.binary_fill_holes(surface_mask)
        fill_operation = "binary_fill_holes"
    else:
        raise ValueError(f"unknown volume_fill_mode: {spec.volume_fill_mode}")
    occupied = _occupied_centers(grid, np.asarray(occupancy, dtype=bool))
    if len(occupied) == 0:
        raise RuntimeError(f"part {spec.part_index} produced no occupied voxels")
    return PartOccupancy(
        spec=spec,
        fidelity_config=config,
        source_mesh_path=source_mesh_path,
        source_mesh=mesh,
        grid=grid,
        occupancy=np.asarray(occupancy, dtype=bool),
        occupied_centers=occupied,
        fill_operation=fill_operation,
    )


def _extract_surface_for_part(part: PartOccupancy) -> trimesh.Trimesh:
    from hag4r.mesh import _marching_cubes_surface, simplify_trimesh_with_meshlab, vtkpoly_to_trimesh

    values = part.occupancy.transpose(2, 1, 0).astype(np.float64) - 0.5
    value_origin = tuple((part.grid.origin + 0.5 * part.grid.pitch_m).tolist())
    surface_poly, _ = _marching_cubes_surface(
        values,
        origin=value_origin,
        spacing=(part.grid.pitch_m, part.grid.pitch_m, part.grid.pitch_m),
    )
    raw_mesh = vtkpoly_to_trimesh(surface_poly)
    part.input_faces = int(len(raw_mesh.faces))
    part.target_faces = target_faces_for_fidelity(part.input_faces, part.spec.mesh_fidelity)
    simplified, info = simplify_trimesh_with_meshlab(raw_mesh, target_faces=part.target_faces)
    trimesh.repair.fix_normals(simplified, multibody=True)
    part.output_faces = int(info["output_face_count"])
    return simplified


def _triangle_face_components(faces: np.ndarray) -> tuple[int, np.ndarray]:
    face_arr = np.asarray(faces, dtype=np.int64)
    if face_arr.ndim != 2 or face_arr.shape[1:] != (3,):
        raise ValueError("triangle faces must have shape (N, 3)")
    if len(face_arr) == 0:
        return 0, np.empty(0, dtype=np.int64)
    edge_pattern = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    edges = np.sort(face_arr[:, edge_pattern].reshape((-1, 2)), axis=1)
    owners = np.repeat(np.arange(len(face_arr), dtype=np.int64), 3)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    shared_ids = np.flatnonzero(counts >= 2)
    rows = []
    cols = []
    for edge_id in shared_ids:
        edge_owners = owners[order[offsets[edge_id] : offsets[edge_id + 1]]]
        rows.extend(np.repeat(edge_owners[0], len(edge_owners) - 1).tolist())
        cols.extend(edge_owners[1:].tolist())
    graph = sparse.csr_matrix(
        (
            np.ones(2 * len(rows), dtype=np.uint8),
            (np.asarray(rows + cols, dtype=np.int64), np.asarray(cols + rows, dtype=np.int64)),
        ),
        shape=(len(face_arr), len(face_arr)),
    )
    component_count, component_labels = csgraph.connected_components(graph, directed=False, return_labels=True)
    return int(component_count), component_labels


def _surface_shell_stats(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    _, face_labels = _triangle_face_components(faces)
    shells = []
    for shell_id in np.unique(face_labels):
        shell = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=faces[face_labels == shell_id],
            process=False,
        )
        shell.remove_unreferenced_vertices()
        shells.append(shell)
    return [
        {
            "vertex_count": int(len(shell.vertices)),
            "face_count": int(len(shell.faces)),
            "signed_volume_m3": float(shell.volume),
            "absolute_volume_m3": float(abs(shell.volume)),
            "bounds_m": np.asarray(shell.bounds, dtype=np.float64).tolist(),
            "watertight": bool(shell.is_watertight),
            "winding_consistent": bool(shell.is_winding_consistent),
        }
        for shell in shells
    ]


def surface_validation_report(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    finite = bool(vertices.ndim == 2 and vertices.shape[1:] == (3,) and np.all(np.isfinite(vertices)))
    triangular = bool(faces.ndim == 2 and faces.shape[1:] == (3,))
    nonempty = bool(len(vertices) > 0 and len(faces) > 0)
    signed_volume = float(mesh.volume) if nonempty and finite and triangular else float("nan")
    report = {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "nonempty": nonempty,
        "finite": finite,
        "triangular": triangular,
        "watertight": bool(mesh.is_watertight) if nonempty and triangular else False,
        "winding_consistent": bool(mesh.is_winding_consistent) if nonempty and triangular else False,
        "signed_volume_m3": signed_volume,
        "absolute_volume_m3": float(abs(signed_volume)),
        "positive_volume": bool(math.isfinite(signed_volume) and signed_volume > 0.0),
        "bounds_m": np.asarray(mesh.bounds, dtype=np.float64).tolist() if nonempty and finite else None,
    }
    report["surface_shells"] = _surface_shell_stats(mesh) if nonempty and finite and triangular else []
    report["surface_shell_count"] = len(report["surface_shells"])
    report["valid"] = bool(
        report["nonempty"]
        and report["finite"]
        and report["triangular"]
        and report["watertight"]
        and report["winding_consistent"]
        and report["positive_volume"]
    )
    return report


def _require_valid_union_surface(mesh: trimesh.Trimesh, *, context: str, report: dict[str, Any]) -> None:
    if report["valid"]:
        return
    failed = [
        key
        for key in ("nonempty", "finite", "triangular", "watertight", "winding_consistent", "positive_volume")
        if not report[key]
    ]
    raise SurfaceCombinationError(f"{context} failed checked Boolean surface requirements: {', '.join(failed)}", report)


def combine_part_surfaces(
    meshes: Sequence[trimesh.Trimesh],
    *,
    strategy: str,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    if not meshes:
        raise ValueError("no per-part surfaces were generated")
    normalized = str(strategy).strip().lower()
    if normalized not in SURFACE_COMBINATION_STRATEGIES:
        raise ValueError(
            f"surface_combination_strategy must be one of {', '.join(sorted(SURFACE_COMBINATION_STRATEGIES))}; "
            f"got {strategy!r}"
        )
    started = time.perf_counter()
    input_reports = [surface_validation_report(mesh) for mesh in meshes]
    dependency_versions = {
        "trimesh": importlib.metadata.version("trimesh"),
        "manifold3d": importlib.metadata.version("manifold3d"),
    }
    base_report: dict[str, Any] = {
        "strategy": normalized,
        "input_count": len(meshes),
        "inputs": input_reports,
        "dependency_versions": dependency_versions,
    }
    if normalized == "concatenate":
        combined = meshes[0].copy() if len(meshes) == 1 else trimesh.util.concatenate([mesh.copy() for mesh in meshes])
    else:
        for index, (mesh, report) in enumerate(zip(meshes, input_reports, strict=True)):
            try:
                _require_valid_union_surface(mesh, context=f"Boolean input {index}", report=report)
            except SurfaceCombinationError as error:
                base_report["failure"] = str(error)
                base_report["runtime_s"] = float(time.perf_counter() - started)
                raise SurfaceCombinationError(str(error), base_report) from error
        try:
            combined = trimesh.boolean.union(
                [surface.copy() for surface in meshes],
                engine="manifold",
                check_volume=True,
            )
        except Exception as error:
            base_report["failure"] = f"{type(error).__name__}: {error}"
            base_report["runtime_s"] = float(time.perf_counter() - started)
            raise SurfaceCombinationError("Manifold Boolean union failed", base_report) from error
        if not isinstance(combined, trimesh.Trimesh):
            base_report["failure"] = f"unexpected Boolean result type: {type(combined).__name__}"
            base_report["runtime_s"] = float(time.perf_counter() - started)
            raise SurfaceCombinationError("Manifold Boolean union returned a non-mesh result", base_report)
    output_report = surface_validation_report(combined)
    report = {
        **base_report,
        "output": output_report,
        "runtime_s": float(time.perf_counter() - started),
    }
    if normalized == "manifold_union":
        try:
            _require_valid_union_surface(combined, context="Boolean output", report=output_report)
        except SurfaceCombinationError as error:
            report["failure"] = str(error)
            raise SurfaceCombinationError(str(error), report) from error
    return combined, report


def _merge_surfaces(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    merged, _ = combine_part_surfaces(meshes, strategy="concatenate")
    return merged


def _tetrahedralize_once(surface: trimesh.Trimesh) -> dict[str, Any]:
    from argparse import Namespace

    from hag4r.mesh import tetmesh_with_pytetwild, trimesh_to_vtkpoly

    args = Namespace(
        edge_length_fac=0.05,
        edge_length_abs=None,
        epsilon=1e-3,
        optimize=True,
        simplify=True,
        coarsen=False,
        stop_energy=None,
        num_opt_iter=None,
    )
    return tetmesh_with_pytetwild(trimesh_to_vtkpoly(surface), args)


def _structural_tet_arrays(tet_result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(tet_result["points"], dtype=np.float64)
    tets = np.asarray(tet_result["tets"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.all(np.isfinite(points)):
        raise RuntimeError("PyTetWild returned invalid or empty finite point array")
    if tets.ndim != 2 or tets.shape[1] != 4 or len(tets) == 0:
        raise RuntimeError("PyTetWild returned invalid or empty tetrahedron array")
    if int(tets.min()) < 0 or int(tets.max()) >= len(points):
        raise RuntimeError("PyTetWild returned tetrahedron indices out of bounds")
    return points, tets


def _tet_face_connectivity(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    face_pattern = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), dtype=np.int64)
    faces = np.sort(tets[:, face_pattern].reshape((-1, 3)), axis=1)
    owners = np.repeat(np.arange(len(tets), dtype=np.int64), 4)
    unique_faces, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    shared_ids = np.flatnonzero(counts == 2)
    if len(shared_ids):
        order = np.argsort(inverse, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        first = owners[order[offsets[shared_ids]]]
        second = owners[order[offsets[shared_ids] + 1]]
        rows = np.concatenate((first, second))
        cols = np.concatenate((second, first))
        graph = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(tets), len(tets)),
        )
    else:
        graph = sparse.csr_matrix((len(tets), len(tets)), dtype=np.uint8)
    component_count, component_labels = csgraph.connected_components(graph, directed=False, return_labels=True)
    return unique_faces, counts, component_labels, int(component_count)


def _tet_component_reports(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    component_labels: np.ndarray,
    tet_volumes: np.ndarray,
    tet_part_labels: np.ndarray | None,
) -> list[dict[str, Any]]:
    component_ids, tet_counts = np.unique(component_labels, return_counts=True)
    component_volumes = np.asarray(
        [float(np.sum(tet_volumes[component_labels == component_id])) for component_id in component_ids],
        dtype=np.float64,
    )
    max_volume = float(component_volumes.max())
    main_candidates = component_ids[np.isclose(component_volumes, max_volume, rtol=0.0, atol=1e-15)]
    main_id = int(main_candidates.min())
    main_vertices = np.unique(tets[component_labels == main_id])
    main_tree = spatial.cKDTree(points[main_vertices])
    reports = []
    for component_id, tet_count, component_volume in zip(component_ids, tet_counts, component_volumes, strict=True):
        mask = component_labels == component_id
        component_tets = tets[mask]
        vertex_indices = np.unique(component_tets)
        vertices = points[vertex_indices]
        volumes = tet_volumes[mask]
        tet_centroids = np.mean(points[component_tets], axis=1)
        volume = float(component_volume)
        centroid = (
            np.average(tet_centroids, axis=0, weights=volumes)
            if volume > 0.0
            else np.mean(tet_centroids, axis=0)
        )
        if int(component_id) == main_id:
            nearest_distance = 0.0
        else:
            nearest_distance = float(np.min(main_tree.query(vertices, k=1)[0]))
        label_distribution: dict[str, int] = {}
        if tet_part_labels is not None:
            labels, counts = np.unique(tet_part_labels[mask], return_counts=True)
            label_distribution = {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)}
        reports.append(
            {
                "component_id": int(component_id),
                "is_main_component": bool(int(component_id) == main_id),
                "tet_count": int(tet_count),
                "vertex_count": int(len(vertex_indices)),
                "volume_m3": volume,
                "bounds_m": np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist(),
                "centroid_m": np.asarray(centroid, dtype=np.float64).tolist(),
                "part_label_distribution": label_distribution,
                "nearest_distance_to_main_component_m": nearest_distance,
            }
        )
    return sorted(reports, key=lambda item: (-item["volume_m3"], item["component_id"]))


def analyze_tet_mesh(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray | None = None,
    expected_part_indices: Sequence[int] | None = None,
    material_coverage: str = "exact",
) -> dict[str, Any]:
    if material_coverage not in {"exact", "subset"}:
        raise ValueError("material_coverage must be 'exact' or 'subset'")
    point_arr = np.asarray(points, dtype=np.float64)
    tet_arr = np.asarray(tets, dtype=np.int64)
    if point_arr.ndim != 2 or point_arr.shape[1:] != (3,) or len(point_arr) == 0:
        raise ValueError("tet points must have nonempty shape (N, 3)")
    if tet_arr.ndim != 2 or tet_arr.shape[1:] != (4,) or len(tet_arr) == 0:
        raise ValueError("tetrahedra must have nonempty shape (M, 4)")

    finite_points = bool(np.all(np.isfinite(point_arr)))
    indices_in_range = bool(int(tet_arr.min()) >= 0 and int(tet_arr.max()) < len(point_arr))
    repeated_vertex_count = int(np.count_nonzero(np.any(np.diff(np.sort(tet_arr, axis=1), axis=1) == 0, axis=1)))
    duplicate_tet_count = int(len(tet_arr) - len(np.unique(np.sort(tet_arr, axis=1), axis=0)))
    labels = None if tet_part_labels is None else np.asarray(tet_part_labels, dtype=np.int64)
    label_length_valid = bool(labels is None or labels.shape == (len(tet_arr),))

    report: dict[str, Any] = {
        "point_count": int(len(point_arr)),
        "tet_count": int(len(tet_arr)),
        "finite_points": finite_points,
        "indices_in_range": indices_in_range,
        "repeated_vertex_tet_count": repeated_vertex_count,
        "duplicate_tet_count": duplicate_tet_count,
        "material_array_length_valid": label_length_valid,
        "failures": [],
        "warnings": [],
    }
    if not finite_points:
        report["failures"].append("non-finite point coordinates")
    if not indices_in_range:
        report["failures"].append("tetrahedron indices out of range")
    if repeated_vertex_count:
        report["failures"].append(f"{repeated_vertex_count} tetrahedra repeat a vertex")
    if duplicate_tet_count:
        report["failures"].append(f"{duplicate_tet_count} duplicate tetrahedra")
    if not label_length_valid:
        report["failures"].append("tet material-label length does not equal tet count")
    if report["failures"]:
        report["valid"] = False
        return report

    tet_points = point_arr[tet_arr]
    matrices = np.stack(
        (
            tet_points[:, 0] - tet_points[:, 3],
            tet_points[:, 1] - tet_points[:, 3],
            tet_points[:, 2] - tet_points[:, 3],
        ),
        axis=2,
    )
    determinants = np.linalg.det(matrices)
    bbox_diagonal = float(np.linalg.norm(point_arr.max(axis=0) - point_arr.min(axis=0)))
    genesis_epsilon = float(max(1e-12, 1e-12 * bbox_diagonal**3))
    genesis_invalid_count = int(np.count_nonzero(np.abs(determinants) <= genesis_epsilon))
    wrong_orientation_count = int(np.count_nonzero(determinants >= genesis_epsilon))
    unique_faces, face_counts, component_labels, component_count = _tet_face_connectivity(tet_arr)
    non_manifold_face_count = int(np.count_nonzero(face_counts > 2))
    boundary_faces = unique_faces[face_counts == 1]
    boundary_surface_component_count, _ = _triangle_face_components(boundary_faces)
    tet_volumes = np.abs(determinants) / 6.0

    material_coverage_valid = True
    expected_labels = [] if expected_part_indices is None else sorted(int(value) for value in expected_part_indices)
    present_labels = [] if labels is None else sorted(int(value) for value in np.unique(labels))
    missing_material_labels: list[int] = []
    unexpected_material_labels: list[int] = []
    if labels is not None and expected_part_indices is not None:
        expected_label_set = set(expected_labels)
        present_label_set = set(present_labels)
        missing_material_labels = sorted(expected_label_set - present_label_set)
        unexpected_material_labels = sorted(present_label_set - expected_label_set)
        if material_coverage == "exact":
            material_coverage_valid = not missing_material_labels and not unexpected_material_labels
        else:
            material_coverage_valid = bool(present_labels) and not unexpected_material_labels

    report.update(
        {
            "bbox_diagonal_m": bbox_diagonal,
            "genesis_determinant_threshold": genesis_epsilon,
            "determinant_min": float(determinants.min()),
            "determinant_max": float(determinants.max()),
            "absolute_determinant_min": float(np.abs(determinants).min()),
            "genesis_invalid_tet_count": genesis_invalid_count,
            "wrong_orientation_tet_count": wrong_orientation_count,
            "non_manifold_tet_face_count": non_manifold_face_count,
            "boundary_triangle_count": int(len(boundary_faces)),
            "boundary_surface_component_count": boundary_surface_component_count,
            "face_connected_component_count": component_count,
            "components": _tet_component_reports(
                points=point_arr,
                tets=tet_arr,
                component_labels=component_labels,
                tet_volumes=tet_volumes,
                tet_part_labels=labels,
            ),
            "expected_material_labels": expected_labels,
            "present_material_labels": present_labels,
            "missing_material_labels": missing_material_labels,
            "unexpected_material_labels": unexpected_material_labels,
            "material_label_coverage_mode": material_coverage,
            "material_label_coverage_valid": material_coverage_valid,
        }
    )
    if non_manifold_face_count:
        report["failures"].append(f"{non_manifold_face_count} tetrahedral faces have more than two owners")
    if wrong_orientation_count:
        report["failures"].append(f"{wrong_orientation_count} tetrahedra have non-Genesis orientation")
    if genesis_invalid_count:
        report["failures"].append(
            f"{genesis_invalid_count} tetrahedra are at or below the Genesis determinant threshold"
        )
    if component_count != 1:
        report["failures"].append(f"tetrahedral material domain has {component_count} face-connected components")
    if material_coverage == "exact" and missing_material_labels:
        report["warnings"].append(
            "material-label coverage incomplete: "
            f"missing expected labels {missing_material_labels}; expected {expected_labels}, present {present_labels}"
        )
    if unexpected_material_labels:
        report["failures"].append(
            "material labels include unexpected labels "
            f"{unexpected_material_labels}; expected {expected_labels}, present {present_labels}"
        )
    report["valid"] = not report["failures"]
    return report


def validate_tet_mesh(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray | None = None,
    expected_part_indices: Sequence[int] | None = None,
    material_coverage: str = "exact",
) -> dict[str, Any]:
    report = analyze_tet_mesh(
        points=points,
        tets=tets,
        tet_part_labels=tet_part_labels,
        expected_part_indices=expected_part_indices,
        material_coverage=material_coverage,
    )
    if not report["valid"]:
        raise TetMeshValidationError("tet mesh failed hard postconditions: " + "; ".join(report["failures"]), report)
    return report


def _point_owner_candidates(point: np.ndarray, parts: Sequence[PartOccupancy]) -> list[PartOccupancy]:
    candidates = []
    for part in parts:
        rel = (point - part.grid.origin) / part.grid.pitch_m
        ijk = np.floor(rel).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.asarray(part.grid.dims)):
            continue
        if bool(part.occupancy[tuple(ijk)]):
            candidates.append(part)
    return candidates


def _nearest_source_distance(part: PartOccupancy, point: np.ndarray) -> float:
    return float(_unsigned_surface_distance(part.source_mesh, point.reshape(1, 3))[0])


def assign_tet_part_labels(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    parts: Sequence[PartOccupancy],
) -> tuple[np.ndarray, dict[str, int]]:
    centroids = np.mean(points[tets], axis=1)
    labels = np.empty((len(tets),), dtype=np.int64)
    part_indices = np.asarray([part.spec.part_index for part in parts], dtype=np.int64)
    occupancy_membership = np.zeros((len(parts), len(centroids)), dtype=bool)
    for part_i, part in enumerate(parts):
        rel = (centroids - part.grid.origin) / part.grid.pitch_m
        ijk = np.floor(rel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(part.grid.dims, dtype=np.int64)), axis=1)
        inside_indices = np.flatnonzero(inside)
        if len(inside_indices):
            local_ijk = ijk[inside_indices]
            occupancy_membership[part_i, inside_indices] = part.occupancy[
                local_ijk[:, 0],
                local_ijk[:, 1],
                local_ijk[:, 2],
            ]
    membership_count = occupancy_membership.sum(axis=0)

    single_mask = membership_count == 1
    if np.any(single_mask):
        labels[single_mask] = part_indices[np.argmax(occupancy_membership[:, single_mask], axis=0)]

    fallback_mask = membership_count == 0
    all_centers = np.concatenate([part.occupied_centers for part in parts], axis=0)
    all_labels = np.concatenate(
        [np.full((len(part.occupied_centers),), part.spec.part_index, dtype=np.int64) for part in parts],
        axis=0,
    )
    tree_type = getattr(spatial, "cK" + "DTree")
    tree = tree_type(all_centers)
    fallback_count = int(np.count_nonzero(fallback_mask))
    if fallback_count:
        _, nearest_idx = tree.query(centroids[fallback_mask])
        labels[fallback_mask] = all_labels[np.asarray(nearest_idx, dtype=np.int64)]

    overlap_mask = membership_count > 1
    overlap_count = int(np.count_nonzero(overlap_mask))
    if overlap_count:
        overlap_indices = np.flatnonzero(overlap_mask)
        overlap_points = centroids[overlap_indices]
        best_dist = np.full((len(overlap_indices),), np.inf, dtype=np.float64)
        best_label = np.full((len(overlap_indices),), np.iinfo(np.int64).max, dtype=np.int64)
        for part_i, part in sorted(enumerate(parts), key=lambda item: item[1].spec.part_index):
            candidate_local = np.flatnonzero(occupancy_membership[part_i, overlap_indices])
            if len(candidate_local) == 0:
                continue
            distances = _unsigned_surface_distance(part.source_mesh, overlap_points[candidate_local])
            update = distances < best_dist[candidate_local]
            if np.any(update):
                local_update = candidate_local[update]
                best_dist[local_update] = distances[update]
                best_label[local_update] = part.spec.part_index
        if np.any(best_label == np.iinfo(np.int64).max):
            raise RuntimeError("overlap owner assignment failed to resolve every overlapped tetrahedron")
        labels[overlap_indices] = best_label
    return labels, {
        "nearest_occupied_voxel_fallback_count": int(fallback_count),
        "overlap_tie_break_count": int(overlap_count),
    }


def _write_partwise_final_npz(
    path: Path,
    specs: Sequence[PartTopologySpec],
    tet_part_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    part_E_nu = np.asarray(
        [[spec.youngs_modulus_pa, spec.poisson_ratio] for spec in specs],
        dtype=np.float64,
    )
    part_density = np.asarray([spec.density_kg_m3 for spec in specs], dtype=np.float64)
    part_colors = np.asarray([spec.part_color_rgb for spec in specs], dtype=np.float64)
    labels = np.asarray(tet_part_labels, dtype=np.int64)
    arrays = {
        "part_indices": np.asarray([spec.part_index for spec in specs], dtype=np.int64),
        "part_E_nu": part_E_nu,
        "part_density": part_density,
        "part_colors": part_colors,
        "part_volume_fill_mode_codes": np.asarray([FILL_MODE_CODES[spec.volume_fill_mode] for spec in specs], dtype=np.int64),
        "tet_E_nu": part_E_nu[labels],
        "tet_density": part_density[labels],
        "tet_part_labels": labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def mesh_processing_render_paths(
    *,
    output_mesh_path: Path,
    heterogeneous_params_path: Path,
) -> dict[str, Path]:
    """Return the four historical per-part mesh-processing render paths."""
    output_dir = heterogeneous_params_path.parent
    output_tag = output_mesh_path.stem
    return {
        "source_part": output_dir / f"{output_tag}_source_part.png",
        "target_part": output_dir / f"{output_tag}_target_part.png",
        "target_E": output_dir / f"{output_tag}_target_E.png",
        "target_density": output_dir / f"{output_tag}_target_density.png",
    }


def _build_pyvista_tet_grid(pv: Any, points: np.ndarray, tets: np.ndarray) -> Any:
    tet_array = np.asarray(tets, dtype=np.int64)
    cells = np.column_stack(
        (np.full((len(tet_array), 1), 4, dtype=np.int64), tet_array)
    ).ravel()
    cell_types = np.full(len(tet_array), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, np.asarray(points, dtype=np.float64))


def _get_render_scalars(
    grid: Any,
    *,
    scalar_name: str,
    scalar_values: np.ndarray,
    scalar_title: str,
    log_spread_threshold: float = 30.0,
) -> tuple[str, str]:
    values = np.asarray(scalar_values, dtype=np.float64).reshape(-1)
    if len(values) != grid.n_cells:
        raise ValueError(
            f"{scalar_name} length {len(values)} does not match rendered tetrahedron count {grid.n_cells}"
        )
    grid.cell_data[scalar_name] = values
    positive_values = np.unique(values[values > 0.0])
    value_spread = np.max(positive_values) / np.min(positive_values) if positive_values.size else 1.0
    if positive_values.size and positive_values.size == np.unique(values).size and value_spread > log_spread_threshold:
        log_scalar_name = f"{scalar_name}Log10"
        grid.cell_data[log_scalar_name] = np.log10(values)
        return log_scalar_name, f"{scalar_title} (log10)"
    return scalar_name, scalar_title


def _save_mesh_render(
    pv: Any,
    *,
    grid: Any,
    scalars: str,
    scalar_title: str,
    output_path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.add_mesh(
            grid,
            scalars=scalars,
            cmap="viridis",
            show_edges=False,
            scalar_bar_args={"title": scalar_title},
        )
        plotter.add_axes()
        plotter.view_isometric()
        plotter.screenshot(str(output_path))
    finally:
        plotter.close()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"PyVista did not write mesh-processing render: {output_path}")


def _render_mesh_processing_pngs(
    *,
    request: PerPartVolumetricMeshingRequest,
    points: np.ndarray,
    tets: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Path]:
    """Reproduce the historical monolithic source/target material renders."""
    import pyvista as pv

    from hag4r.tools.metric_mesh_scaling import read_medit_tet_mesh

    source_mesh = read_medit_tet_mesh(request.input_mesh_dir / "mesh_combined.mesh")
    with np.load(request.part_labels_path, allow_pickle=False) as part_labels:
        if "tet_part_labels" not in part_labels:
            raise KeyError(f"{request.part_labels_path} is missing tet_part_labels for source-part rendering")
        source_labels = np.asarray(part_labels["tet_part_labels"], dtype=np.int64)

    source_grid = _build_pyvista_tet_grid(pv, source_mesh.vertices, source_mesh.tets)
    target_grid = _build_pyvista_tet_grid(pv, points, tets)
    output_paths = mesh_processing_render_paths(
        output_mesh_path=request.output_mesh_path,
        heterogeneous_params_path=request.heterogeneous_params_path,
    )
    output_paths["source_part"].parent.mkdir(parents=True, exist_ok=True)

    source_scalars, source_title = _get_render_scalars(
        source_grid,
        scalar_name="PartIndex",
        scalar_values=source_labels,
        scalar_title="Source Part Index",
    )
    _save_mesh_render(
        pv,
        grid=source_grid,
        scalars=source_scalars,
        scalar_title=source_title,
        output_path=output_paths["source_part"],
    )
    target_part_scalars, target_part_title = _get_render_scalars(
        target_grid,
        scalar_name="PartIndex",
        scalar_values=arrays["tet_part_labels"],
        scalar_title="Part Index",
    )
    _save_mesh_render(
        pv,
        grid=target_grid,
        scalars=target_part_scalars,
        scalar_title=target_part_title,
        output_path=output_paths["target_part"],
    )
    youngs_modulus_scalars, youngs_modulus_title = _get_render_scalars(
        target_grid,
        scalar_name="YoungsModulus",
        scalar_values=arrays["tet_E_nu"][:, 0],
        scalar_title="Young's Modulus (E)",
    )
    _save_mesh_render(
        pv,
        grid=target_grid,
        scalars=youngs_modulus_scalars,
        scalar_title=youngs_modulus_title,
        output_path=output_paths["target_E"],
    )
    density_scalars, density_title = _get_render_scalars(
        target_grid,
        scalar_name="Density",
        scalar_values=arrays["tet_density"],
        scalar_title="Density (kg/m^3)",
    )
    _save_mesh_render(
        pv,
        grid=target_grid,
        scalars=density_scalars,
        scalar_title=density_title,
        output_path=output_paths["target_density"],
    )
    return output_paths


def _validate_material_arrays(arrays: dict[str, np.ndarray], *, tet_count: int) -> None:
    for name in ("tet_E_nu", "tet_density", "tet_part_labels"):
        if len(arrays[name]) != tet_count:
            raise RuntimeError(f"{name} length {len(arrays[name])} does not equal tet count {tet_count}")


def _part_topology_payload(
    *,
    part: PartOccupancy,
    repo_root: Path,
) -> dict[str, Any]:
    grid = part.grid
    payload: dict[str, Any] = {
        "part_index": part.spec.part_index,
        "part_color_rgb": list(part.spec.part_color_rgb),
        "volume_fill_mode": part.spec.volume_fill_mode,
        "mesh_fidelity": part.spec.mesh_fidelity,
        "fidelity_config": {
            "keep_ratio": part.fidelity_config.keep_ratio,
            "absolute_cap": part.fidelity_config.absolute_cap,
            "longest_axis_voxels": part.fidelity_config.longest_axis_voxels,
        },
        "voxel_grid": {
            "origin": grid.origin.tolist(),
            "pitch_m": grid.pitch_m,
            "dims": list(grid.dims),
            "cell_count": int(np.prod(grid.dims, dtype=np.int64)),
            "bounds_m": {"min": grid.bounds_min.tolist(), "max": grid.bounds_max.tolist()},
        },
        "derived_voxel_band_width_m": (
            HOLLOW_WALL_BAND_LAYERS * grid.pitch_m if part.spec.volume_fill_mode == "hollow_wall" else None
        ),
        "fill_operation": part.fill_operation,
        "input_faces": part.input_faces,
        "target_faces": part.target_faces,
        "output_faces": part.output_faces,
        "occupied_voxel_count": int(np.count_nonzero(part.occupancy)),
        "tet_count": part.tet_count,
        "retained": bool(part.tet_count > 0),
        "source_mesh_path": _repo_relative(part.source_mesh_path, repo_root),
    }
    return payload


def _material_specs_with_fidelities(
    *,
    inferred_params_path: Path,
    part_labels_path: Path,
    mesh_processing_plan: dict[str, Any],
) -> list[PartTopologySpec]:
    material_specs = load_color_locked_material_predictions(inferred_params_path, part_labels_path)
    return load_mesh_fidelity_plan(mesh_processing_plan, material_specs)


def _selected_component_id_from_report(report: dict[str, Any]) -> int:
    components = report.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("component report is missing components")
    return int(
        min(
            components,
            key=lambda item: (-float(item["volume_m3"]), int(item["component_id"])),
        )["component_id"]
    )


def _compact_tet_component(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    component_labels: np.ndarray,
    selected_component_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep_mask = np.asarray(component_labels, dtype=np.int64) == int(selected_component_id)
    if not np.any(keep_mask):
        raise ValueError(f"component_id {selected_component_id} is not present in candidate component labels")
    kept_tets = np.asarray(tets, dtype=np.int64)[keep_mask]
    kept_labels = np.asarray(tet_part_labels, dtype=np.int64)[keep_mask]
    used_vertices = np.unique(kept_tets)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    compact_tets = remap[kept_tets]
    if int(compact_tets.min()) < 0:
        raise RuntimeError("component compaction produced negative vertex indices")
    compact_points = np.asarray(points, dtype=np.float64)[used_vertices]
    return compact_points, compact_tets, kept_labels, used_vertices


def _write_candidate_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_component_resolution_candidate(
    *,
    request: PerPartVolumetricMeshingRequest,
    points: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    component_labels: np.ndarray,
    post_tetrahedralization_validation: dict[str, Any],
    pre_resolution_validation: dict[str, Any],
    surface_combination: dict[str, Any],
    owner_counts: dict[str, Any],
) -> dict[str, Any]:
    paths = tet_component_resolution_paths(request.output_mesh_path)
    _write_candidate_npz(
        paths.candidate_npz_path,
        points=np.asarray(points, dtype=np.float64),
        tets=np.asarray(tets, dtype=np.int64),
        tet_part_labels=np.asarray(tet_part_labels, dtype=np.int64),
        component_labels=np.asarray(component_labels, dtype=np.int64),
    )
    selected_component_id = _selected_component_id_from_report(pre_resolution_validation)
    present = set(map(int, pre_resolution_validation.get("present_material_labels", [])))
    selected = next(
        component
        for component in pre_resolution_validation["components"]
        if int(component["component_id"]) == selected_component_id
    )
    retained = set(map(int, selected.get("part_label_distribution", {}).keys()))
    payload = {
        "schema_version": TET_COMPONENT_RESOLUTION_SCHEMA_VERSION,
        "status": "needs_component_resolution",
        "selection_rule": "max_volume_m3_tie_min_component_id",
        "selected_component_id": selected_component_id,
        "dropped_component_ids": [
            int(component["component_id"])
            for component in pre_resolution_validation["components"]
            if int(component["component_id"]) != selected_component_id
        ],
        "dropped_part_indices": sorted(present - retained),
        "candidate_npz_path": _repo_relative(paths.candidate_npz_path, request.repo_root),
        "candidate_npz_sha256": _sha256_file(paths.candidate_npz_path),
        "request_path": (
            _repo_relative(request.request_path, request.repo_root)
            if request.request_path is not None
            else ""
        ),
        "request_sha256": _sha256_file(request.request_path) if request.request_path is not None else "",
        "output_mesh_path": _repo_relative(request.output_mesh_path, request.repo_root),
        "heterogeneous_params_path": _repo_relative(request.heterogeneous_params_path, request.repo_root),
        "metric_mesh_scaling_path": _repo_relative(request.metric_mesh_scaling_path, request.repo_root),
        "volume_topology_path": _repo_relative(request.volume_topology_path, request.repo_root),
        "surface_combination": surface_combination,
        "tet_validation": {
            "post_tetrahedralization": post_tetrahedralization_validation,
            "pre_resolution": pre_resolution_validation,
        },
        "component_count": int(pre_resolution_validation["face_connected_component_count"]),
        "components": pre_resolution_validation["components"],
        "owner_assignment_counts": owner_counts,
        "resolution": {
            "attempted": False,
            "agent_rationale": "",
        },
    }
    _write_json(paths.report_path, payload)
    return payload


def _write_validated_volumetric_outputs(
    *,
    request: PerPartVolumetricMeshingRequest,
    specs: Sequence[PartTopologySpec],
    parts: Sequence[PartOccupancy],
    points: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    scaling_payload: dict[str, Any],
    combination_report: dict[str, Any],
    owner_counts: dict[str, Any],
    post_tetrahedralization_validation: dict[str, Any],
    material_coverage: str,
    component_resolution: dict[str, Any] | None = None,
) -> PerPartVolumetricMeshingResult:
    from hag4r.mesh import write_medit_mesh

    expected_part_indices = [spec.part_index for spec in specs]
    pre_serialization_validation = validate_tet_mesh(
        points=points,
        tets=tets,
        tet_part_labels=tet_part_labels,
        expected_part_indices=expected_part_indices,
        material_coverage=material_coverage,
    )
    for part in parts:
        part.tet_count = int(np.count_nonzero(tet_part_labels == part.spec.part_index))

    request.output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{request.output_mesh_path.name}.",
        suffix=".tmp.mesh",
        dir=request.output_mesh_path.parent,
        delete=False,
    ) as handle:
        temporary_mesh_path = Path(handle.name)
    try:
        write_medit_mesh(temporary_mesh_path, points, tets)
        from hag4r.tools.metric_mesh_scaling import read_medit_tet_mesh

        reloaded = read_medit_tet_mesh(temporary_mesh_path)
        if not np.array_equal(reloaded.tets, tets):
            raise RuntimeError(".mesh serialization changed tetrahedron connectivity or ordering")
        if not np.array_equal(reloaded.vertices, points):
            raise RuntimeError(".mesh serialization changed vertex coordinates")
        post_serialization_validation = validate_tet_mesh(
            points=reloaded.vertices,
            tets=reloaded.tets,
            tet_part_labels=tet_part_labels,
            expected_part_indices=expected_part_indices,
            material_coverage=material_coverage,
        )
        temporary_mesh_path.replace(request.output_mesh_path)
    finally:
        temporary_mesh_path.unlink(missing_ok=True)
    arrays = _write_partwise_final_npz(request.heterogeneous_params_path, specs, tet_part_labels)
    _validate_material_arrays(arrays, tet_count=len(tets))
    budget_status, budget_warnings = tet_budget_status(len(tets))
    warnings = list(
        dict.fromkeys(
            [
                *budget_warnings,
                *pre_serialization_validation["warnings"],
                *post_serialization_validation["warnings"],
            ]
        )
    )

    scaling_payload = dict(scaling_payload)
    scaling_payload["pre_tetrahedralization_metric_bounds"] = scaling_payload["metric_bounds"]
    scaling_payload["metric_bounds"] = _bounds_payload(points)
    scaling_payload["output_mesh_path"] = _repo_relative(request.output_mesh_path, request.repo_root)
    scaling_payload["output_mesh_sha256"] = _sha256_file(request.output_mesh_path)
    _write_json(request.metric_mesh_scaling_path, scaling_payload)

    retained_fill_modes = [part.spec.volume_fill_mode for part in parts if part.tet_count > 0]
    topology_payload = {
        "schema_version": VOLUME_TOPOLOGY_SCHEMA_VERSION,
        "contract": MESH_PROCESSING_PLAN_CONTRACT,
        "volume_topology": derive_volume_topology(retained_fill_modes),
        "hollow_wall_band_layers": HOLLOW_WALL_BAND_LAYERS,
        "tet_budget_limit": TET_BUDGET_LIMIT,
        "tet_count": int(len(tets)),
        "tet_budget_status": budget_status,
        "warnings": warnings,
        "surface_combination": combination_report,
        "tet_validation": {
            "post_tetrahedralization": post_tetrahedralization_validation,
            "pre_serialization": pre_serialization_validation,
            "post_serialization": post_serialization_validation,
        },
        "mesh_path": _repo_relative(request.output_mesh_path, request.repo_root),
        "heterogeneous_params_path": _repo_relative(request.heterogeneous_params_path, request.repo_root),
        "metric_mesh_scaling_path": _repo_relative(request.metric_mesh_scaling_path, request.repo_root),
        "parts": [_part_topology_payload(part=part, repo_root=request.repo_root) for part in parts],
        "owner_assignment": {
            "method": "tet_centroid_occupancy_owner",
            "overlap_tie_break": "nearest_source_part_surface_then_smaller_part_index",
            "boundary_fallback": "nearest_occupied_voxel_only",
            **owner_counts,
        },
        "array_hashes": {
            "tet_part_labels_sha256": _sha256_array(arrays["tet_part_labels"]),
            "tet_E_nu_sha256": _sha256_array(arrays["tet_E_nu"]),
            "tet_density_sha256": _sha256_array(arrays["tet_density"]),
        },
    }
    if component_resolution is not None:
        topology_payload["component_resolution"] = component_resolution
    _write_json(request.volume_topology_path, topology_payload)
    _render_mesh_processing_pngs(
        request=request,
        points=points,
        tets=tets,
        arrays=arrays,
    )
    return PerPartVolumetricMeshingResult(
        mesh_path=request.output_mesh_path,
        heterogeneous_params_path=request.heterogeneous_params_path,
        metric_mesh_scaling_path=request.metric_mesh_scaling_path,
        volume_topology_path=request.volume_topology_path,
        tet_count=int(len(tets)),
        tet_budget_status=budget_status,
        warnings=tuple(warnings),
    )


def run_per_part_volumetric_meshing(request: PerPartVolumetricMeshingRequest) -> PerPartVolumetricMeshingResult:
    specs = _material_specs_with_fidelities(
        inferred_params_path=request.inferred_params_path,
        part_labels_path=request.part_labels_path,
        mesh_processing_plan=request.mesh_processing_plan,
    )
    part_paths, scaled_meshes, scaling_payload = _load_scaled_part_meshes(request)
    if len(part_paths) != len(specs):
        raise ValueError(f"part GLB count {len(part_paths)} does not match material part count {len(specs)}")
    parts = [
        _build_part_occupancy(spec=spec, source_mesh_path=part_paths[spec.part_index], mesh=scaled_meshes[spec.part_index])
        for spec in specs
    ]
    surfaces = [_extract_surface_for_part(part) for part in parts]
    merged_surface, combination_report = combine_part_surfaces(
        surfaces,
        strategy=request.surface_combination_strategy,
    )
    tet_result = _tetrahedralize_once(merged_surface)
    points, tets = _structural_tet_arrays(tet_result)
    post_tetrahedralization_validation = analyze_tet_mesh(
        points=points,
        tets=tets,
    )
    tet_part_labels, owner_counts = assign_tet_part_labels(points=points, tets=tets, parts=parts)
    expected_part_indices = [spec.part_index for spec in specs]
    pre_resolution_validation = analyze_tet_mesh(
        points=points,
        tets=tets,
        tet_part_labels=tet_part_labels,
        expected_part_indices=expected_part_indices,
    )
    if int(pre_resolution_validation.get("face_connected_component_count", 0)) > 1:
        _, _, component_labels, _ = _tet_face_connectivity(tets)
        component_report = _write_component_resolution_candidate(
            request=request,
            points=points,
            tets=tets,
            tet_part_labels=tet_part_labels,
            component_labels=component_labels,
            post_tetrahedralization_validation=post_tetrahedralization_validation,
            pre_resolution_validation=pre_resolution_validation,
            surface_combination=combination_report,
            owner_counts=owner_counts,
        )
        raise DisconnectedTetComponentsError(
            "tetrahedral material domain needs component resolution: "
            f"{component_report['component_count']} face-connected components",
            component_report,
        )
    if not post_tetrahedralization_validation["valid"]:
        raise TetMeshValidationError(
            "tet mesh failed hard postconditions: "
            + "; ".join(post_tetrahedralization_validation["failures"]),
            post_tetrahedralization_validation,
        )
    if not pre_resolution_validation["valid"]:
        raise TetMeshValidationError(
            "tet mesh failed hard postconditions: " + "; ".join(pre_resolution_validation["failures"]),
            pre_resolution_validation,
        )
    return _write_validated_volumetric_outputs(
        request=request,
        specs=specs,
        parts=parts,
        points=points,
        tets=tets,
        tet_part_labels=tet_part_labels,
        scaling_payload=scaling_payload,
        combination_report=combination_report,
        owner_counts=owner_counts,
        post_tetrahedralization_validation=post_tetrahedralization_validation,
        material_coverage="exact",
    )


def component_resolution_report_for_request(
    request_json_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    root = (repo_root or Path.cwd()).resolve()
    request = _load_request(Path(request_json_path), root)
    paths = tet_component_resolution_paths(request.output_mesh_path)
    if not paths.report_path.exists():
        return None
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != TET_COMPONENT_RESOLUTION_SCHEMA_VERSION:
        return None
    if str(report.get("request_sha256", "") or "") and report["request_sha256"] != _sha256_file(Path(request_json_path)):
        return None
    return report


def resolve_disconnected_tet_components_from_request_json(
    request_json_path: Path,
    *,
    rationale: str,
    repo_root: Path | None = None,
) -> PerPartVolumetricMeshingResult:
    root = (repo_root or Path.cwd()).resolve()
    request = _load_request(Path(request_json_path), root)
    reason = str(rationale or "").strip()
    if not reason:
        raise ValueError("rationale is required to resolve disconnected tet components")
    paths = tet_component_resolution_paths(request.output_mesh_path)
    if not paths.report_path.exists():
        raise FileNotFoundError(f"component resolution report does not exist: {paths.report_path}")
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != TET_COMPONENT_RESOLUTION_SCHEMA_VERSION:
        raise ValueError(f"component resolution report schema must be {TET_COMPONENT_RESOLUTION_SCHEMA_VERSION}")
    if report.get("status") != "needs_component_resolution":
        raise RuntimeError(f"component resolution status must be needs_component_resolution, got {report.get('status')!r}")
    resolution = report.get("resolution", {})
    if isinstance(resolution, dict) and resolution.get("attempted"):
        raise RuntimeError("component resolution has already been attempted")
    if str(report.get("request_sha256", "") or "") and report["request_sha256"] != _sha256_file(Path(request_json_path)):
        raise RuntimeError("component resolution report request hash does not match the active request")
    if not paths.candidate_npz_path.exists():
        raise FileNotFoundError(f"component resolution candidate does not exist: {paths.candidate_npz_path}")
    if str(report.get("candidate_npz_sha256", "") or "") != _sha256_file(paths.candidate_npz_path):
        raise RuntimeError("component resolution candidate hash mismatch")

    with np.load(paths.candidate_npz_path, allow_pickle=False) as archive:
        required = {"points", "tets", "tet_part_labels", "component_labels"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"component resolution candidate is missing array(s): {', '.join(missing)}")
        points = np.asarray(archive["points"], dtype=np.float64)
        tets = np.asarray(archive["tets"], dtype=np.int64)
        tet_part_labels = np.asarray(archive["tet_part_labels"], dtype=np.int64)
        component_labels = np.asarray(archive["component_labels"], dtype=np.int64)

    selected_component_id = _selected_component_id_from_report(report)
    trimmed_points, trimmed_tets, trimmed_labels, used_vertices = _compact_tet_component(
        points=points,
        tets=tets,
        tet_part_labels=tet_part_labels,
        component_labels=component_labels,
        selected_component_id=selected_component_id,
    )
    specs = _material_specs_with_fidelities(
        inferred_params_path=request.inferred_params_path,
        part_labels_path=request.part_labels_path,
        mesh_processing_plan=request.mesh_processing_plan,
    )
    part_paths, scaled_meshes, scaling_payload = _load_scaled_part_meshes(request)
    if len(part_paths) != len(specs):
        raise ValueError(f"part GLB count {len(part_paths)} does not match material part count {len(specs)}")
    parts = [
        _build_part_occupancy(spec=spec, source_mesh_path=part_paths[spec.part_index], mesh=scaled_meshes[spec.part_index])
        for spec in specs
    ]

    post_tetrahedralization_validation = validate_tet_mesh(points=trimmed_points, tets=trimmed_tets)
    retained_part_indices = sorted(int(value) for value in np.unique(trimmed_labels))
    component_resolution = {
        "schema_version": TET_COMPONENT_RESOLUTION_SCHEMA_VERSION,
        "report_path": _repo_relative(paths.report_path, request.repo_root),
        "candidate_npz_path": _repo_relative(paths.candidate_npz_path, request.repo_root),
        "selection_rule": "max_volume_m3_tie_min_component_id",
        "selected_component_id": selected_component_id,
        "dropped_component_ids": [
            int(component["component_id"])
            for component in report["components"]
            if int(component["component_id"]) != selected_component_id
        ],
        "retained_part_indices": retained_part_indices,
        "dropped_part_indices": sorted(set(map(int, report.get("dropped_part_indices", [])))),
        "agent_rationale": reason,
        "used_original_vertex_indices": used_vertices.tolist(),
    }
    result = _write_validated_volumetric_outputs(
        request=request,
        specs=specs,
        parts=parts,
        points=trimmed_points,
        tets=trimmed_tets,
        tet_part_labels=trimmed_labels,
        scaling_payload=scaling_payload,
        combination_report=dict(report["surface_combination"]),
        owner_counts=dict(report.get("owner_assignment_counts", {})),
        post_tetrahedralization_validation=post_tetrahedralization_validation,
        material_coverage="subset",
        component_resolution=component_resolution,
    )
    report["status"] = "resolved"
    report["resolution"] = {
        "attempted": True,
        "agent_rationale": reason,
        "selected_component_id": selected_component_id,
        "selection_rule": "max_volume_m3_tie_min_component_id",
        "retained_part_indices": retained_part_indices,
        "dropped_component_ids": component_resolution["dropped_component_ids"],
        "dropped_part_indices": component_resolution["dropped_part_indices"],
        "output_mesh_sha256": _sha256_file(request.output_mesh_path),
        "heterogeneous_params_sha256": _sha256_file(request.heterogeneous_params_path),
        "volume_topology_sha256": _sha256_file(request.volume_topology_path),
    }
    _write_json(paths.report_path, report)
    return result


def run_per_part_volumetric_meshing_from_request_json(
    request_json_path: Path,
    *,
    repo_root: Path | None = None,
) -> PerPartVolumetricMeshingResult:
    root = (repo_root or Path.cwd()).resolve()
    request = _load_request(Path(request_json_path), root)
    return run_per_part_volumetric_meshing(request)


__all__ = [
    "DENSE_GRID_CELL_CAP",
    "FILL_MODE_CODES",
    "FORBIDDEN_AGENT_MESH_NUMERIC_KEYS",
    "HOLLOW_WALL_BAND_LAYERS",
    "MESH_PROCESSING_PLAN_CONTRACT",
    "MESH_PROCESSING_PLAN_SCHEMA_VERSION",
    "METRIC_MAX_DIMENSION_SCALING_SCHEMA_VERSION",
    "PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION",
    "SURFACE_COMBINATION_STRATEGIES",
    "TET_COMPONENT_RESOLUTION_SCHEMA_VERSION",
    "TET_BUDGET_LIMIT",
    "VOLUME_MESH_FIDELITY_TABLE",
    "VOLUME_TOPOLOGY_SCHEMA_VERSION",
    "MeshFidelityConfig",
    "PartOccupancy",
    "PartTopologySpec",
    "PerPartVolumetricMeshingRequest",
    "PerPartVolumetricMeshingResult",
    "DisconnectedTetComponentsError",
    "SurfaceCombinationError",
    "TetComponentResolutionPaths",
    "TetMeshValidationError",
    "VoxelGridSpec",
    "assign_tet_part_labels",
    "analyze_tet_mesh",
    "build_mesh_processing_plan",
    "component_resolution_report_for_request",
    "combine_part_surfaces",
    "derive_volume_topology",
    "load_color_locked_material_predictions",
    "load_mesh_fidelity_plan",
    "mesh_processing_render_paths",
    "fidelity_config",
    "resolve_disconnected_tet_components_from_request_json",
    "run_per_part_volumetric_meshing",
    "run_per_part_volumetric_meshing_from_request_json",
    "surface_validation_report",
    "target_faces_for_fidelity",
    "tet_budget_status",
    "tet_component_resolution_paths",
    "validate_mesh_fidelity_payload",
    "validate_tet_mesh",
]
