from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hag4r.tools.metric_mesh_scaling import read_medit_tet_mesh, tetra_volume_m3
from hag4r.tools.genesis.config import fit_asset_for_diagnostic_scene


PART_GROUNDING_CONTEXT_SCHEMA_VERSION = "hag4r-diagnostic-part-grounding-context-v1"
FINAL_MESH_FRAME = "final_monolithic_mesh_m"
GENESIS_ENV_FRAME = "genesis_env_local_m"


@dataclass(frozen=True)
class FinalPrimitiveMesh:
    mesh_path: Path
    primitive_kind: str
    vertices: np.ndarray
    primitives: np.ndarray


@dataclass(frozen=True)
class VertexPartLabelMap:
    primary_labels: np.ndarray
    incident_label_counts: tuple[dict[int, int], ...]
    tied_vertex_indices: tuple[int, ...]
    isolated_vertex_indices: tuple[int, ...]


def _ensure_triangular_surface(polydata: Any) -> Any:
    import vtk

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    return tri.GetOutput()


def _append_polydata(poly_list: list[Any]) -> Any:
    if not poly_list:
        raise ValueError("No polygonal geometry found while reading surface mesh.")
    if len(poly_list) == 1:
        return poly_list[0]

    import vtk

    app = vtk.vtkAppendPolyData()
    for poly in poly_list:
        app.AddInputData(poly)
    app.Update()
    return app.GetOutput()


def _dataset_to_polydata(dataset: Any) -> Any:
    import vtk

    if dataset is None:
        raise ValueError("Input mesh reader returned no dataset.")

    if dataset.IsA("vtkPolyData"):
        return dataset

    if dataset.IsA("vtkMultiBlockDataSet"):
        polys = []
        it = dataset.NewIterator()
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            block = it.GetCurrentDataObject()
            if block is not None:
                try:
                    polys.append(_dataset_to_polydata(block))
                except ValueError:
                    pass
            it.GoToNextItem()
        if hasattr(it, "Delete"):
            it.Delete()
        return _append_polydata(polys)

    if dataset.IsA("vtkPartitionedDataSetCollection"):
        polys = []
        for i in range(dataset.GetNumberOfPartitionedDataSets()):
            partitioned = dataset.GetPartitionedDataSet(i)
            if partitioned is None:
                continue
            for j in range(partitioned.GetNumberOfPartitions()):
                block = partitioned.GetPartition(j)
                if block is None:
                    continue
                try:
                    polys.append(_dataset_to_polydata(block))
                except ValueError:
                    pass
        return _append_polydata(polys)

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(dataset)
    geom.Update()
    poly = geom.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not convert dataset type {dataset.GetClassName()} to vtkPolyData.")
    return poly


def load_surface_tri_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyvista as pv
        from vtkmodules.util import numpy_support
    except ImportError as exc:
        raise ImportError("surface mesh loading requires pyvista, vtk, and vtkmodules") from exc

    resolved = Path(mesh_path).expanduser().resolve()
    mesh_data = pv.read(str(resolved))
    vtk_obj = getattr(mesh_data, "vtk_obj", mesh_data)
    surface_poly = _ensure_triangular_surface(_dataset_to_polydata(vtk_obj))

    points_vtk = surface_poly.GetPoints()
    polys_vtk = surface_poly.GetPolys()
    if points_vtk is None or polys_vtk is None:
        raise ValueError(f"Surface mesh {resolved} does not contain polygonal faces.")

    points = numpy_support.vtk_to_numpy(points_vtk.GetData()).astype(np.float64, copy=False)
    faces_flat = numpy_support.vtk_to_numpy(polys_vtk.GetData())
    if faces_flat.size == 0:
        raise ValueError(f"Surface mesh {resolved} does not contain any faces.")

    faces = []
    i = 0
    while i < faces_flat.size:
        face_size = int(faces_flat[i])
        if face_size != 3:
            raise ValueError(f"Expected triangular surface mesh in {resolved}, got face size {face_size}.")
        faces.append(faces_flat[i + 1:i + 4])
        i += 4

    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 0:
        raise ValueError(f"Surface mesh {resolved} does not contain any triangles.")
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise ValueError(f"Surface mesh {resolved} contains out-of-range face indices.")
    return points, faces


def read_final_monolithic_mesh(
    monolithic_mesh_path: Path,
) -> FinalPrimitiveMesh:
    resolved = Path(monolithic_mesh_path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix != ".mesh":
        raise ValueError(f"canonical final mesh must be .mesh, got: {resolved}")
    mesh = read_medit_tet_mesh(resolved)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    primitives = np.asarray(mesh.tets, dtype=np.int64)
    primitive_kind = "tetrahedron"

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"final mesh vertices must have shape (N, 3), got {vertices.shape}")
    expected_width = 4 if primitive_kind == "tetrahedron" else 3
    if primitives.ndim != 2 or primitives.shape[1] != expected_width:
        raise ValueError(f"final mesh primitives must have shape (N, {expected_width}), got {primitives.shape}")
    if primitives.shape[0] == 0:
        raise ValueError(f"final mesh contains no {primitive_kind} primitives: {resolved}")
    if np.any(primitives < 0) or np.any(primitives >= vertices.shape[0]):
        raise ValueError("final mesh primitive connectivity contains out-of-range vertex indices")

    return FinalPrimitiveMesh(
        mesh_path=resolved,
        primitive_kind=primitive_kind,
        vertices=vertices,
        primitives=primitives,
    )


def read_final_primitive_labels(
    monolithic_params_path: Path,
    *,
    primitive_count: int,
) -> tuple[np.ndarray, str]:
    label_key = "tet_part_labels"

    resolved = Path(monolithic_params_path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as payload:
        if label_key not in payload:
            raise ValueError(f"{resolved} is missing expected primitive label array {label_key!r}")
        labels = np.asarray(payload[label_key])

    if labels.ndim != 1:
        raise ValueError(f"{label_key} must be a 1D array, got shape {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError(f"{label_key} must be non-empty")
    if labels.shape[0] != primitive_count:
        raise ValueError(
            f"{label_key} length ({labels.shape[0]}) does not match final primitive count ({primitive_count})"
        )
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"{label_key} contains non-finite labels")
    if np.issubdtype(labels.dtype, np.integer):
        int_labels = labels.astype(np.int64, copy=False)
    else:
        float_labels = labels.astype(np.float64, copy=False)
        if not np.all(float_labels == np.floor(float_labels)):
            raise ValueError(f"{label_key} must be integer-valued")
        int_labels = float_labels.astype(np.int64)
    if np.any(int_labels < 0):
        raise ValueError(f"{label_key} contains negative part labels")
    return int_labels, label_key


def derive_vertex_part_labels(
    *,
    vertex_count: int,
    primitives: np.ndarray,
    primitive_labels: np.ndarray,
) -> VertexPartLabelMap:
    if vertex_count < 0:
        raise ValueError(f"vertex_count must be nonnegative, got {vertex_count}")
    prims = np.asarray(primitives, dtype=np.int64)
    labels = np.asarray(primitive_labels, dtype=np.int64)
    if prims.ndim != 2:
        raise ValueError(f"primitives must be a 2D array, got shape {prims.shape}")
    if labels.ndim != 1 or labels.shape[0] != prims.shape[0]:
        raise ValueError(
            f"primitive_labels must be a 1D array matching primitive count, got {labels.shape} for {prims.shape[0]}"
        )
    if np.any(prims < 0) or np.any(prims >= vertex_count):
        raise ValueError("primitive connectivity contains out-of-range vertex indices")

    counters = [Counter() for _ in range(vertex_count)]
    for primitive, label in zip(prims, labels, strict=True):
        label_value = int(label)
        for vertex_index in primitive:
            counters[int(vertex_index)][label_value] += 1

    primary_labels = np.full(vertex_count, -1, dtype=np.int64)
    tied_vertices: list[int] = []
    isolated_vertices: list[int] = []
    incident_counts: list[dict[int, int]] = []
    for vertex_index, counter in enumerate(counters):
        counts = dict(sorted((int(label), int(count)) for label, count in counter.items()))
        incident_counts.append(counts)
        if not counts:
            isolated_vertices.append(vertex_index)
            continue
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        if len(winners) > 1:
            tied_vertices.append(vertex_index)
        primary_labels[vertex_index] = min(winners)

    return VertexPartLabelMap(
        primary_labels=primary_labels,
        incident_label_counts=tuple(incident_counts),
        tied_vertex_indices=tuple(tied_vertices),
        isolated_vertex_indices=tuple(isolated_vertices),
    )


def _load_material_inference(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"{path} must contain top-level predictions list")
    geometry_inference = payload.get("geometry_inference")
    if not isinstance(geometry_inference, dict) or not isinstance(geometry_inference.get("predictions"), list):
        raise ValueError(f"{path} must contain top-level geometry_inference.predictions list")
    return predictions, geometry_inference["predictions"], payload


def _validate_material_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {
        "part_index",
        "part_name",
        "part_semantics",
        "part_texture",
        "major_material_name",
        "part_color_rgb",
        "density_kg_m3",
        "volume_fill_mode",
        "fill_mode_rationale",
        "fill_mode_evidence",
        "youngs_modulus_pa",
        "poisson_ratio",
        "friction_coefficient",
    }
    normalized: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"material prediction {index} must be an object")
        missing = sorted(required.difference(prediction))
        if missing:
            raise ValueError(f"material prediction {index} is missing required fields: {missing}")
        part_index = int(prediction["part_index"])
        color = np.asarray(prediction["part_color_rgb"], dtype=np.float64)
        if color.shape != (3,) or not np.all(np.isfinite(color)):
            raise ValueError(f"material prediction {index} has invalid part_color_rgb")
        fill_mode = str(prediction["volume_fill_mode"])
        if fill_mode not in {"solid_fill", "hollow_wall"}:
            raise ValueError(
                f"material prediction {index} has invalid volume_fill_mode {fill_mode!r}"
            )
        evidence = prediction["fill_mode_evidence"]
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError(f"material prediction {index} has invalid fill_mode_evidence")
        normalized.append({**prediction, "part_index": part_index, "part_color_rgb": color.tolist()})

    part_indices = [part["part_index"] for part in normalized]
    expected = list(range(len(normalized)))
    if sorted(part_indices) != expected:
        raise ValueError(f"part_index values must be unique, zero-based, and contiguous: expected {expected}, got {part_indices}")
    return sorted(normalized, key=lambda item: int(item["part_index"]))


def _geometry_by_part_index(geometry_predictions: list[dict[str, Any]], num_parts: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for fallback_index, prediction in enumerate(geometry_predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"geometry inference prediction {fallback_index} must be an object")
        raw_index = prediction.get("part_index", fallback_index)
        part_index = int(raw_index)
        if part_index < 0 or part_index >= num_parts:
            raise ValueError(f"geometry inference references unknown part_index {part_index}")
        result[part_index] = dict(prediction)
    return result


def _validate_part_colors(part_labels_path: Path, material_parts: list[dict[str, Any]]) -> None:
    resolved = Path(part_labels_path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as payload:
        if "part_colors" not in payload:
            return
        part_colors = np.asarray(payload["part_colors"], dtype=np.float64)
    if part_colors.ndim != 2 or part_colors.shape[1] not in {3, 4}:
        raise ValueError(f"part_colors in {resolved} must have shape (N, 3) or (N, 4), got {part_colors.shape}")
    part_colors = part_colors[:, :3]
    if part_colors.shape[0] < len(material_parts):
        raise ValueError(
            f"part_colors in {resolved} has {part_colors.shape[0]} rows but material inference has {len(material_parts)} parts"
        )
    for part in material_parts:
        part_index = int(part["part_index"])
        expected = np.asarray(part["part_color_rgb"], dtype=np.float64)
        actual = part_colors[part_index]
        if not np.allclose(actual, expected, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                f"part_color_rgb mismatch for part_index {part_index}: material={expected.tolist()} part_labels={actual.tolist()}"
            )


def _bbox_payload(points: np.ndarray) -> tuple[list[float], list[float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return [*mins.tolist(), *maxs.tolist()], (maxs - mins).tolist()


def _triangle_area_m2(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri_points = vertices[triangles]
    cross = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
    return np.linalg.norm(cross, axis=1) * 0.5


def _json_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _generated_asset_transform(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload.get("body_config_payload"), dict):
        payload = payload["body_config_payload"]
    transform = payload.get("transformation")
    if not isinstance(transform, dict):
        raise ValueError(f"generated asset JSON is missing transformation: {path}")
    return transform


def _generated_asset_consistency(path: Path | None, expected_translation: tuple[float, float, float]) -> dict[str, Any]:
    if path is None:
        return {
            "checked": False,
            "path": "",
            "matches": None,
            "message": "no generated_asset_json_path supplied",
        }

    transform = _generated_asset_transform(path)
    expected = {
        "trans": list(expected_translation),
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "center": [0.0, 0.0, 0.0],
    }
    mismatches = []
    for key, expected_values in expected.items():
        actual_values = transform.get(key)
        if actual_values is None:
            mismatches.append(f"missing {key}")
            continue
        actual = np.asarray(actual_values, dtype=np.float64)
        wanted = np.asarray(expected_values, dtype=np.float64)
        if actual.shape != wanted.shape or not np.allclose(actual, wanted, atol=1.0e-9, rtol=0.0):
            mismatches.append(f"{key}: expected {wanted.tolist()}, got {actual.tolist()}")
    if mismatches:
        raise ValueError(f"generated asset transform mismatch for {path}: {'; '.join(mismatches)}")
    return {
        "checked": True,
        "path": str(Path(path).expanduser().resolve()),
        "matches": True,
        "message": "generated asset transform matches diagnostic scene fit",
    }


def build_part_grounding_context(
    *,
    run_id: str,
    object_name: str,
    inferred_params_path: Path,
    part_labels_path: Path,
    monolithic_mesh_path: Path,
    monolithic_params_path: Path,
    label_view_artifact_ids: tuple[str, ...] = (),
    generated_asset_json_path: Path | None = None,
) -> dict[str, Any]:
    final_mesh = read_final_monolithic_mesh(monolithic_mesh_path)
    primitive_labels, label_key = read_final_primitive_labels(
        monolithic_params_path,
        primitive_count=int(final_mesh.primitives.shape[0]),
    )
    predictions, geometry_predictions, inferred_payload = _load_material_inference(inferred_params_path)
    material_parts = _validate_material_predictions(predictions)
    num_parts = len(material_parts)
    geometry_by_part = _geometry_by_part_index(geometry_predictions, num_parts)
    _validate_part_colors(part_labels_path, material_parts)

    present_part_ids = sorted(int(label) for label in np.unique(primitive_labels).tolist())
    if present_part_ids and 0 not in present_part_ids and max(present_part_ids) == num_parts:
        raise ValueError(
            f"{label_key} appears to use likely one-indexed labels: present ids {present_part_ids}, material parts 0..{num_parts - 1}"
        )
    known_part_ids = set(range(num_parts))
    unknown_part_ids = sorted(set(present_part_ids).difference(known_part_ids))
    if unknown_part_ids:
        raise ValueError(f"{label_key} references unknown part id(s) absent from MaterialInference: {unknown_part_ids}")

    vertices = final_mesh.vertices
    primitives = final_mesh.primitives
    primitive_vertices = vertices[primitives]
    primitive_centroids = primitive_vertices.mean(axis=1)
    vertex_label_map = derive_vertex_part_labels(
        vertex_count=int(vertices.shape[0]),
        primitives=primitives,
        primitive_labels=primitive_labels,
    )

    if final_mesh.primitive_kind == "tetrahedron":
        primitive_volume = tetra_volume_m3(vertices, primitives)
        primitive_area = np.full(primitives.shape[0], np.nan, dtype=np.float64)
    else:
        primitive_volume = np.full(primitives.shape[0], np.nan, dtype=np.float64)
        primitive_area = _triangle_area_m2(vertices, primitives)

    mesh_bbox, mesh_extent = _bbox_payload(vertices)
    primitive_count = int(primitives.shape[0])
    primary_label_counts = Counter(int(label) for label in vertex_label_map.primary_labels if int(label) >= 0)
    parts: list[dict[str, Any]] = []
    warnings: list[str] = []
    visible_artifact_ids = list(label_view_artifact_ids)
    for part in material_parts:
        part_id = int(part["part_index"])
        part_mask = primitive_labels == part_id
        selected_primitives = primitives[part_mask]
        selected_vertices = np.unique(selected_primitives.reshape(-1)) if selected_primitives.size else np.asarray([], dtype=np.int64)
        part_warnings: list[str] = []
        if selected_primitives.shape[0] == 0:
            warning = f"part_id {part_id} has zero final primitives"
            warnings.append(warning)
            part_warnings.append(warning)
            final_geometry = {
                "coordinate_frame": FINAL_MESH_FRAME,
                "primitive_count": 0,
                "primitive_fraction": 0.0,
                "vertex_count": 0,
                "bbox_m": [],
                "centroid_m": [],
                "extent_m": [],
                "volume_m3": None,
                "surface_area_m2": None,
            }
        else:
            part_points = vertices[selected_vertices]
            part_bbox, part_extent = _bbox_payload(part_points)
            part_centroid = primitive_centroids[part_mask].mean(axis=0)
            final_geometry = {
                "coordinate_frame": FINAL_MESH_FRAME,
                "primitive_count": int(selected_primitives.shape[0]),
                "primitive_fraction": float(selected_primitives.shape[0] / primitive_count),
                "vertex_count": int(selected_vertices.shape[0]),
                "bbox_m": _json_float_list(part_bbox),
                "centroid_m": _json_float_list(part_centroid),
                "extent_m": _json_float_list(part_extent),
                "volume_m3": float(np.sum(primitive_volume[part_mask])) if final_mesh.primitive_kind == "tetrahedron" else None,
                "surface_area_m2": float(np.sum(primitive_area[part_mask])) if final_mesh.primitive_kind == "triangle" else None,
            }

        parts.append(
            {
                "part_id": part_id,
                "part_index": part_id,
                "part_name": str(part["part_name"]),
                "part_semantics": str(part["part_semantics"]),
                "part_texture": str(part["part_texture"]),
                "major_material_name": str(part["major_material_name"]),
                "part_color_rgb": _json_float_list(part["part_color_rgb"]),
                "material": {
                    "density_kg_m3": float(part["density_kg_m3"]),
                    "volume_fill_mode": str(part["volume_fill_mode"]),
                    "fill_mode_rationale": str(part["fill_mode_rationale"]),
                    "fill_mode_evidence": list(part["fill_mode_evidence"]),
                    "youngs_modulus_pa": float(part["youngs_modulus_pa"]),
                    "poisson_ratio": float(part["poisson_ratio"]),
                    "friction_coefficient": float(part["friction_coefficient"]),
                },
                "geometry_inference": geometry_by_part.get(part_id, {}),
                "final_geometry": final_geometry,
                "visibility": {
                    "label_view_evidence_available": bool(visible_artifact_ids),
                    "visible_label_view_artifact_ids": visible_artifact_ids,
                },
                "validation": {
                    "status": "warning" if part_warnings else "ok",
                    "warnings": part_warnings,
                },
            }
        )

    scene_fit = fit_asset_for_diagnostic_scene(Path(monolithic_mesh_path))
    consistency = _generated_asset_consistency(generated_asset_json_path, scene_fit.translation)
    return {
        "schema_version": PART_GROUNDING_CONTEXT_SCHEMA_VERSION,
        "run_id": str(run_id),
        "object_name": str(object_name),
        "coordinate_frames": {
            FINAL_MESH_FRAME: {
                "description": "vertices from monolithic_mesh_path after final metric scaling",
                "unit": "m",
            },
            GENESIS_ENV_FRAME: {
                "description": "Genesis diagnostic object-local scene frame used by aabb_box_v1",
                "unit": "m",
                "axes": "x/z horizontal, y up",
            },
        },
        "sources": {
            "inferred_params_path": str(Path(inferred_params_path).expanduser().resolve()),
            "part_labels_path": str(Path(part_labels_path).expanduser().resolve()),
            "monolithic_mesh_path": str(final_mesh.mesh_path),
            "monolithic_params_path": str(Path(monolithic_params_path).expanduser().resolve()),
            "label_array_key": label_key,
        },
        "transform": {
            "source": "hag4r.tools.genesis.config.fit_asset_for_diagnostic_scene",
            "mesh_to_env_local": {
                "scale": [float(scene_fit.scale), float(scene_fit.scale), float(scene_fit.scale)],
                "translation_m": _json_float_list(scene_fit.translation),
                "rotation": [0.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
            },
            "generated_asset_consistency": consistency,
        },
        "mesh": {
            "mesh_format": final_mesh.mesh_path.suffix.lower(),
            "primitive_kind": final_mesh.primitive_kind,
            "vertex_count": int(vertices.shape[0]),
            "primitive_count": primitive_count,
            "bbox_m": _json_float_list(mesh_bbox),
            "extent_m": _json_float_list(mesh_extent),
        },
        "labels": {
            "array_key": label_key,
            "label_count": int(primitive_labels.shape[0]),
            "present_part_ids": present_part_ids,
            "missing_material_part_ids": [],
            "label_index_base": 0,
        },
        "material_inference": {
            "inferred_object_name": str(inferred_payload.get("object_name", object_name)),
            "num_parts": num_parts,
            "fill_mode_counts": dict(Counter(str(part["volume_fill_mode"]) for part in material_parts)),
        },
        "visibility": {
            "omnipart_label_view_artifact_ids": visible_artifact_ids,
            "preview_visibility_score": None,
            "note": "static label views only; rendered target preview is Feature 2",
        },
        "parts": parts,
        "parts_by_id": {str(part["part_id"]): index for index, part in enumerate(parts)},
        "vertex_part_labels": {
            "primary_label_counts": {str(key): int(value) for key, value in sorted(primary_label_counts.items())},
            "tied_vertex_count": len(vertex_label_map.tied_vertex_indices),
            "tied_vertex_indices_sample": list(vertex_label_map.tied_vertex_indices[:16]),
            "isolated_vertex_count": len(vertex_label_map.isolated_vertex_indices),
            "isolated_vertex_indices_sample": list(vertex_label_map.isolated_vertex_indices[:16]),
        },
        "validation": {
            "status": "warning" if warnings else "ok",
            "errors": [],
            "warnings": warnings,
        },
    }


def compact_part_grounding_table(context: Mapping[str, Any]) -> dict[str, Any]:
    parts = context.get("parts", [])
    compact_parts = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            final_geometry = part.get("final_geometry") if isinstance(part.get("final_geometry"), Mapping) else {}
            compact_parts.append(
                {
                    "part_id": part.get("part_id"),
                    "part_name": part.get("part_name", ""),
                    "part_semantics": part.get("part_semantics", ""),
                    "major_material_name": part.get("major_material_name", ""),
                    "volume_fill_mode": part.get("material", {}).get("volume_fill_mode", "")
                    if isinstance(part.get("material"), Mapping)
                    else "",
                    "primitive_count": final_geometry.get("primitive_count", 0),
                    "vertex_count": final_geometry.get("vertex_count", 0),
                    "bbox_m": final_geometry.get("bbox_m", []),
                    "extent_m": final_geometry.get("extent_m", []),
                    "warnings": part.get("validation", {}).get("warnings", [])
                    if isinstance(part.get("validation"), Mapping)
                    else [],
                }
            )
    return {
        "schema_version": context.get("schema_version", PART_GROUNDING_CONTEXT_SCHEMA_VERSION),
        "run_id": context.get("run_id", ""),
        "object_name": context.get("object_name", ""),
        "sources": {
            "label_array_key": context.get("sources", {}).get("label_array_key", "")
            if isinstance(context.get("sources"), Mapping)
            else ""
        },
        "mesh": context.get("mesh", {}),
        "labels": context.get("labels", {}),
        "parts": compact_parts,
        "parts_by_id": context.get("parts_by_id", {}),
        "vertex_part_labels": context.get("vertex_part_labels", {}),
        "validation": context.get("validation", {}),
    }


def _format_bbox(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 6:
        return ""
    return "[" + ", ".join(f"{float(item):.6g}" for item in value) + "]"


def render_part_grounding_markdown_table(context: Mapping[str, Any]) -> str:
    compact = compact_part_grounding_table(context)
    lines = [
        "# Part Grounding Table",
        "",
        f"- schema_version: {compact.get('schema_version', '')}",
        f"- run_id: {compact.get('run_id', '')}",
        f"- object_name: {compact.get('object_name', '')}",
        f"- label_array_key: {compact.get('sources', {}).get('label_array_key', '')}",
        "",
        "| part_id | part_name | major_material | fill_mode | primitives | vertices | bbox_m | warnings |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for part in compact.get("parts", []):
        warnings = "; ".join(str(item) for item in part.get("warnings", []))
        lines.append(
            "| "
            f"{part.get('part_id', '')} | "
            f"{part.get('part_name', '')} | "
            f"{part.get('major_material_name', '')} | "
            f"{part.get('volume_fill_mode', '')} | "
            f"{part.get('primitive_count', 0)} | "
            f"{part.get('vertex_count', 0)} | "
            f"{_format_bbox(part.get('bbox_m', []))} | "
            f"{warnings} |"
        )
    validation = compact.get("validation", {})
    if isinstance(validation, Mapping) and validation.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in validation.get("warnings", []))
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "FINAL_MESH_FRAME",
    "GENESIS_ENV_FRAME",
    "PART_GROUNDING_CONTEXT_SCHEMA_VERSION",
    "FinalPrimitiveMesh",
    "VertexPartLabelMap",
    "build_part_grounding_context",
    "compact_part_grounding_table",
    "derive_vertex_part_labels",
    "load_surface_tri_mesh",
    "read_final_monolithic_mesh",
    "read_final_primitive_labels",
    "render_part_grounding_markdown_table",
]
