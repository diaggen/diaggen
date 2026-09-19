from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from hag4r.tools.part_grounding import (
    derive_vertex_part_labels,
    read_final_monolithic_mesh,
    read_final_primitive_labels,
)
from hag4r.tools.triple_view_evidence import (
    TRIPLE_VIEW_PANEL_ORDER,
    stitch_triptych,
    triple_view_camera,
    write_triple_view_manifest,
)


PROBE_TARGET_INTENT_SCHEMA_VERSION = "hag4r-diagnostic-probe-target-intent-v1"
PROBE_TARGET_INTENT_V2_SCHEMA_VERSION = "hag4r-diagnostic-probe-target-intent-v2"
PROBE_TARGET_COMPILE_SCHEMA_VERSION = "hag4r-diagnostic-probe-target-compile-v1"
PROBE_TARGET_EDIT_SCHEMA_VERSION = "hag4r-diagnostic-probe-target-edit-v2"
PROBE_TARGET_VALIDATION_SCHEMA_VERSION = "hag4r-diagnostic-probe-target-validation-v1"
PART_GROUNDED_BOX_COMPILE_SCHEMA_VERSION = "hag4r-diagnostic-part-grounded-box-compile-v1"
PART_GROUNDED_BOX_EDIT_SCHEMA_VERSION = "hag4r-diagnostic-part-grounded-box-edit-v1"
PART_GROUNDED_BOX_VALIDATION_SCHEMA_VERSION = "hag4r-diagnostic-part-grounded-box-validation-v1"
ANCHOR_TARGET_COMPILE_SCHEMA_VERSION = "hag4r-diagnostic-anchor-target-compile-v1"
ANCHOR_TARGET_EDIT_SCHEMA_VERSION = "hag4r-diagnostic-anchor-target-edit-v1"
ANCHOR_TARGET_VALIDATION_SCHEMA_VERSION = "hag4r-diagnostic-anchor-target-validation-v1"

ALLOWED_REGION_HINTS = ("full_part", "tip", "root", "edge_band")
ALLOWED_PART_GROUNDED_REGION_HINTS = (
    "full_part",
    "support_contact",
    "stable_base",
    "grip_root",
    "hinge_side",
    "root",
    "tip",
    "edge_band",
)
FACE_ADJUST_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
TRANSLATION_KEYS = ("x", "y", "z")
OBSERVED_MISMATCH_VALUES = (
    "misses_semantic_region",
    "too_large",
    "too_small",
    "pin_overlap",
    "occluded",
    "wrong_part",
    "acceptable_no_edit",
)

MIN_TARGET_BOX_EXTENT_M = 0.004
MAX_TARGET_TO_PART_VOLUME_RATIO = 8.0
MIN_PRIMITIVE_PURITY = 0.50
MIN_GRABBED_SELECTED_PART_FRACTION = 0.50
PIN_OVERLAP_WARNING_RATIO = 0.10

# Geometry-only local-patch policy.  These bounds are deliberately named and
# shared by every asset: a mechanics probe starts in the farthest 30% of a
# labelled part, then keeps one connected distal component.
MECHANICS_DISTAL_DISTANCE_QUANTILE = 0.70
MECHANICS_DISTAL_QUANTILE_INCREMENT = 0.02
MECHANICS_MAX_DISTAL_DISTANCE_QUANTILE = 0.98
MECHANICS_MAX_TARGET_VERTEX_FRACTION = 0.40
MECHANICS_MAX_TARGET_PRIMITIVE_FRACTION = 0.50
MECHANICS_MIN_TARGET_VERTEX_COUNT = 1
MECHANICS_MIN_ANCHOR_SEPARATION_M = 1.0e-6
ALLOWED_MECHANICS_PROBE_MODES = (
    "none", "bending", "compliance", "relative_structural_response", "stretch_tension",
)
ALLOWED_MOTION_AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

PREVIEW_VIEW_NAMES = TRIPLE_VIEW_PANEL_ORDER
TARGET_OVERLAY_RGB = (235, 32, 32)
ANCHOR_TARGET_OVERLAY_RGB = (255, 150, 0)
PIN_OVERLAY_RGB = (32, 92, 235)
SELECTED_OUTLINE_RGB = (16, 16, 16)
BACKGROUND_RGB = (248, 248, 248)
PART_REFERENCE_RENDERER_VERSION = "hag4r-static-part-reference-v2"
PART_REFERENCE_LABEL_RGB = (12, 12, 12)
PART_REFERENCE_LABEL_BACKING_RGB = (255, 255, 255)
PART_REFERENCE_LEADER_RGB = (48, 48, 48)


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _box_array(box: Any, context: str) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} must be a finite 6-vector")
    if np.any(arr[:3] >= arr[3:]):
        raise ValueError(f"{context} must have strict min < max on every axis")
    return arr


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("cannot compute bbox from empty or malformed points")
    return np.concatenate([np.min(points, axis=0), np.max(points, axis=0)]).astype(np.float64)


def _expand_to_min_extent(box: np.ndarray) -> np.ndarray:
    out = box.astype(np.float64, copy=True)
    spans = out[3:] - out[:3]
    for axis, span in enumerate(spans):
        if span >= MIN_TARGET_BOX_EXTENT_M:
            continue
        center = (out[axis] + out[axis + 3]) * 0.5
        half = MIN_TARGET_BOX_EXTENT_M * 0.5
        out[axis] = center - half
        out[axis + 3] = center + half
    return out


def _box_volume(box: np.ndarray) -> float:
    spans = np.maximum(box[3:] - box[:3], 0.0)
    return float(np.prod(spans))


def _box_overlap(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray | None, float]:
    mins = np.maximum(left[:3], right[:3])
    maxs = np.minimum(left[3:], right[3:])
    if np.any(mins >= maxs):
        return None, 0.0
    box = np.concatenate([mins, maxs]).astype(np.float64)
    return box, _box_volume(box)


def _points_inside_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return np.all((points >= box[:3]) & (points <= box[3:]), axis=1)


def _part_lookup(context: Mapping[str, Any], part_id: int) -> Mapping[str, Any]:
    parts_by_id = context.get("parts_by_id", {})
    parts = context.get("parts", [])
    key = str(part_id)
    if not isinstance(parts_by_id, Mapping) or key not in parts_by_id:
        raise ValueError(f"selected_part_id {part_id} is not present in part grounding context")
    index = int(parts_by_id[key])
    if not isinstance(parts, list) or index < 0 or index >= len(parts) or not isinstance(parts[index], Mapping):
        raise ValueError(f"part grounding context has invalid parts_by_id entry for part_id {part_id}")
    return parts[index]


def _load_compile_inputs(context: Mapping[str, Any]) -> dict[str, Any]:
    sources = context.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("part grounding context is missing sources")
    mesh = read_final_monolithic_mesh(Path(str(sources["monolithic_mesh_path"])))
    primitive_labels, label_key = read_final_primitive_labels(
        Path(str(sources["monolithic_params_path"])),
        primitive_count=int(mesh.primitives.shape[0]),
    )
    vertex_labels = derive_vertex_part_labels(
        vertex_count=int(mesh.vertices.shape[0]),
        primitives=mesh.primitives,
        primitive_labels=primitive_labels,
    )
    return {
        "mesh": mesh,
        "primitive_labels": primitive_labels,
        "label_key": label_key,
        "vertex_labels": vertex_labels,
        "primitive_centroids": mesh.vertices[mesh.primitives].mean(axis=1),
    }


def _cap_box(
    *,
    selected_points: np.ndarray,
    selected_centroids: np.ndarray,
    part_box: np.ndarray,
    region_hint: str,
    anchor_pin_box: np.ndarray | None,
    mesh_box: np.ndarray,
) -> np.ndarray:
    spans = part_box[3:] - part_box[:3]
    axis = int(np.argmax(spans))
    if spans[axis] <= 0:
        return part_box
    cap_width = max(spans[axis] * 0.35, MIN_TARGET_BOX_EXTENT_M)
    low_limit = part_box[axis] + cap_width
    high_limit = part_box[axis + 3] - cap_width
    low_mask = selected_points[:, axis] <= low_limit
    high_mask = selected_points[:, axis] >= high_limit
    low_points = selected_points[low_mask]
    high_points = selected_points[high_mask]
    if low_points.size == 0 or high_points.size == 0:
        return part_box

    def score(points: np.ndarray, reference: np.ndarray) -> float:
        return float(abs(float(points[:, axis].mean()) - float(reference[axis])))

    if anchor_pin_box is not None:
        reference = (anchor_pin_box[:3] + anchor_pin_box[3:]) * 0.5
    else:
        reference = (mesh_box[:3] + mesh_box[3:]) * 0.5
    low_score = score(low_points, reference)
    high_score = score(high_points, reference)
    if region_hint == "root":
        chosen = low_points if low_score <= high_score else high_points
    else:
        chosen = low_points if low_score > high_score else high_points
    return _bbox_from_points(chosen)


def _edge_band_box(selected_points: np.ndarray, part_box: np.ndarray) -> np.ndarray:
    spans = part_box[3:] - part_box[:3]
    axis = int(np.argmax(spans))
    if spans[axis] <= 0:
        return part_box
    band_width = max(spans[axis] * 0.25, MIN_TARGET_BOX_EXTENT_M)
    side_mask = selected_points[:, axis] >= (part_box[axis + 3] - band_width)
    side_points = selected_points[side_mask]
    if side_points.size == 0:
        return part_box
    return _bbox_from_points(side_points)


def _point_to_box_distances(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    below = np.maximum(box[:3] - points, 0.0)
    above = np.maximum(points - box[3:], 0.0)
    return np.linalg.norm(below + above, axis=1)


def _part_vertex_adjacency(selected_primitives: np.ndarray) -> dict[int, tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {}
    for primitive in selected_primitives:
        vertices = [int(value) for value in primitive]
        for vertex in vertices:
            adjacency.setdefault(vertex, set())
        for index, left in enumerate(vertices):
            for right in vertices[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    return {vertex: tuple(sorted(neighbors)) for vertex, neighbors in adjacency.items()}


def _boundary_vertices(selected_primitives: np.ndarray) -> set[int]:
    """Return labelled-part boundary vertices for triangles or tetrahedra."""
    faces: Counter[tuple[int, ...]] = Counter()
    width = int(selected_primitives.shape[1])
    if width not in {3, 4}:
        raise ValueError("mechanics edge_band requires triangle or tetrahedron primitives")
    for primitive in selected_primitives:
        vertices = [int(value) for value in primitive]
        for omit in range(width):
            faces[tuple(sorted(vertices[:omit] + vertices[omit + 1 :]))] += 1
    return {vertex for face, count in faces.items() if count == 1 for vertex in face}


def _anchor_distal_patch_box(
    *,
    selected_vertices: np.ndarray,
    selected_points: np.ndarray,
    selected_primitives: np.ndarray,
    selected_centroids: np.ndarray,
    anchor_box: np.ndarray,
    region_hint: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compile one anchor-relative distal component without axis heuristics."""
    adjacency = _part_vertex_adjacency(selected_primitives)
    candidates = np.ones(selected_vertices.shape[0], dtype=bool)
    strategy = "anchor_distal_cap"
    if region_hint == "edge_band":
        boundary = _boundary_vertices(selected_primitives)
        candidates = np.asarray([int(vertex) in boundary for vertex in selected_vertices], dtype=bool)
        if not np.any(candidates):
            raise ValueError("mechanics edge_band has no labelled-part boundary vertices")
        strategy = "anchor_distal_edge_band"
    distances = _point_to_box_distances(selected_points, anchor_box)
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size == 0:
        raise ValueError("mechanics local target has no eligible vertices")
    vertex_to_position = {int(vertex): index for index, vertex in enumerate(selected_vertices)}
    max_vertices = max(1, int(math.floor(selected_vertices.size * MECHANICS_MAX_TARGET_VERTEX_FRACTION)))
    quantiles = np.arange(
        MECHANICS_DISTAL_DISTANCE_QUANTILE,
        MECHANICS_MAX_DISTAL_DISTANCE_QUANTILE + MECHANICS_DISTAL_QUANTILE_INCREMENT * 0.5,
        MECHANICS_DISTAL_QUANTILE_INCREMENT,
    )
    for quantile in quantiles:
        threshold = float(np.quantile(distances[candidate_indices], quantile))
        distal_indices = candidate_indices[distances[candidate_indices] >= threshold]
        if distal_indices.size == 0:
            continue
        # lexsort makes the deterministic index tie-break explicit.
        farthest_index = int(distal_indices[np.lexsort((selected_vertices[distal_indices], -distances[distal_indices]))[0]])
        allowed_vertices = {int(selected_vertices[index]) for index in distal_indices}
        component: list[int] = []
        queue = [int(selected_vertices[farthest_index])]
        seen: set[int] = set()
        while queue:
            vertex = queue.pop(0)
            if vertex in seen or vertex not in allowed_vertices:
                continue
            seen.add(vertex)
            component.append(vertex)
            queue.extend(neighbor for neighbor in adjacency.get(vertex, ()) if neighbor in allowed_vertices and neighbor not in seen)
        if not component or len(component) > max_vertices:
            continue
        component_indices = np.asarray([vertex_to_position[vertex] for vertex in component], dtype=np.int64)
        component_points = selected_points[component_indices]
        component_set = set(component)
        incident_primitive_mask = np.asarray(
            [all(int(vertex) in component_set for vertex in primitive) for primitive in selected_primitives], dtype=bool
        )
        if not np.any(incident_primitive_mask):
            continue
        # Include wholly local primitive centroids so the existing purity check
        # evaluates the same cap; BoxEE identity still remains runtime-selected.
        box = _bbox_from_points(np.vstack([component_points, selected_centroids[incident_primitive_mask]]))
        actual_count = int(np.count_nonzero(_points_inside_box(selected_points, box)))
        actual_fraction = float(actual_count / selected_vertices.size)
        primitive_fraction = float(np.count_nonzero(incident_primitive_mask) / selected_primitives.shape[0])
        separation = float(np.min(_point_to_box_distances(component_points, anchor_box)))
        if (
            actual_count >= MECHANICS_MIN_TARGET_VERTEX_COUNT
            and actual_fraction <= MECHANICS_MAX_TARGET_VERTEX_FRACTION
            and primitive_fraction <= MECHANICS_MAX_TARGET_PRIMITIVE_FRACTION
            and separation > MECHANICS_MIN_ANCHOR_SEPARATION_M
        ):
            return box, {
                "locality_strategy": strategy,
                "locality_provenance": "farthest connected labelled-part component from owning static anchor",
                "distance_quantile": float(quantile),
                "target_vertex_count": int(component_indices.size),
                "selected_part_vertex_count": int(selected_vertices.size),
                "target_vertex_fraction": float(component_indices.size / selected_vertices.size),
                "incident_primitive_count": int(np.count_nonzero(incident_primitive_mask)),
                "incident_primitive_fraction": primitive_fraction,
                "actual_box_selected_vertex_count": actual_count,
                "actual_box_selected_vertex_fraction": actual_fraction,
                "anchor_separation_m": separation,
            }
    raise ValueError("mechanics local target cannot satisfy generic connected locality and primitive-purity bounds")


def _support_band_box(selected_points: np.ndarray, part_box: np.ndarray) -> np.ndarray:
    axis = 1
    span = float(part_box[axis + 3] - part_box[axis])
    if span <= 0.0:
        return part_box
    band_width = max(span * 0.30, MIN_TARGET_BOX_EXTENT_M)
    support_points = selected_points[selected_points[:, axis] <= part_box[axis] + band_width]
    if support_points.size == 0:
        return part_box
    return _bbox_from_points(support_points)


def validate_part_grounded_box_edit(
    edit: Mapping[str, Any],
    *,
    schema_version: str = PART_GROUNDED_BOX_EDIT_SCHEMA_VERSION,
    context_label: str = "part-grounded box",
) -> dict[str, Any]:
    if not isinstance(edit, Mapping):
        raise ValueError(f"{context_label} edit must be an object")
    data = dict(edit)
    expected = {
        "schema_version",
        "face_adjust_percent",
        "translation_fraction",
        "reason",
        "evidence_refs",
        "observed_mismatch",
        "view_basis",
    }
    missing = sorted(expected - set(data))
    extra = sorted(set(data) - expected)
    if missing:
        raise ValueError(f"{context_label} edit missing required field(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{context_label} edit has unknown field(s): {', '.join(extra)}")
    if data["schema_version"] != schema_version:
        raise ValueError(f"{context_label} edit schema_version must equal {schema_version}")
    evidence_refs = data["evidence_refs"]
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise ValueError(f"{context_label} edit evidence_refs must be a non-empty list")
    evidence_refs_out: list[str] = []
    for index, item in enumerate(evidence_refs):
        text = str(item).strip()
        if not text:
            raise ValueError(f"{context_label} edit evidence_refs[{index}] must be non-empty")
        evidence_refs_out.append(text)
    observed_mismatch = str(data["observed_mismatch"]).strip()
    if observed_mismatch not in OBSERVED_MISMATCH_VALUES:
        raise ValueError(
            f"{context_label} edit observed_mismatch must be one of: "
            + ", ".join(OBSERVED_MISMATCH_VALUES)
        )
    view_basis = data["view_basis"]
    if not isinstance(view_basis, list) or not view_basis:
        raise ValueError(f"{context_label} edit view_basis must be a non-empty list")
    view_basis_out: list[str] = []
    for index, item in enumerate(view_basis):
        text = str(item).strip()
        if text not in TRIPLE_VIEW_PANEL_ORDER:
            raise ValueError(
                f"{context_label} edit view_basis[{index}] must be one of: "
                + ", ".join(TRIPLE_VIEW_PANEL_ORDER)
            )
        if text not in view_basis_out:
            view_basis_out.append(text)
    face = data["face_adjust_percent"]
    trans = data["translation_fraction"]
    if not isinstance(face, Mapping):
        raise ValueError("face_adjust_percent must be an object")
    if not isinstance(trans, Mapping):
        raise ValueError("translation_fraction must be an object")
    face_out: dict[str, float] = {}
    for key, value in face.items():
        if key not in FACE_ADJUST_KEYS:
            raise ValueError(f"unsupported face_adjust_percent key: {key}")
        number = _finite_float(value, f"face_adjust_percent.{key}")
        if number < -50.0 or number > 50.0:
            raise ValueError(f"face_adjust_percent.{key} must be in [-50, 50]")
        face_out[str(key)] = number
    trans_out: dict[str, float] = {}
    for key, value in trans.items():
        if key not in TRANSLATION_KEYS:
            raise ValueError(f"unsupported translation_fraction key: {key}")
        number = _finite_float(value, f"translation_fraction.{key}")
        if number < -0.5 or number > 0.5:
            raise ValueError(f"translation_fraction.{key} must be in [-0.5, 0.5]")
        trans_out[str(key)] = number
    reason = str(data["reason"]).strip()
    if not reason:
        raise ValueError(f"{context_label} edit reason must be non-empty")
    if "{" in reason or "}" in reason or "aabb_box" in reason.lower():
        raise ValueError(f"{context_label} edit reason must not contain raw JSON or aabb_box payloads")
    return {
        "schema_version": schema_version,
        "face_adjust_percent": face_out,
        "translation_fraction": trans_out,
        "evidence_refs": evidence_refs_out,
        "observed_mismatch": observed_mismatch,
        "view_basis": view_basis_out,
        "reason": reason,
    }


def validate_probe_target_edit(edit: Mapping[str, Any]) -> dict[str, Any]:
    return validate_part_grounded_box_edit(
        edit,
        schema_version=PROBE_TARGET_EDIT_SCHEMA_VERSION,
        context_label="probe target",
    )


def validate_anchor_target_edit(edit: Mapping[str, Any]) -> dict[str, Any]:
    return validate_part_grounded_box_edit(
        edit,
        schema_version=ANCHOR_TARGET_EDIT_SCHEMA_VERSION,
        context_label="anchor target",
    )


def apply_part_grounded_box_edit(
    base_box: list[float] | np.ndarray,
    edit: Mapping[str, Any],
    *,
    schema_version: str = PART_GROUNDED_BOX_EDIT_SCHEMA_VERSION,
    context_label: str = "part-grounded box",
) -> list[float]:
    validated = validate_part_grounded_box_edit(edit, schema_version=schema_version, context_label=context_label)
    box = _box_array(base_box, "base_box")
    spans = box[3:] - box[:3]
    out = box.copy()
    face = validated["face_adjust_percent"]
    for axis, axis_name in enumerate(("x", "y", "z")):
        min_key = f"{axis_name}_min"
        max_key = f"{axis_name}_max"
        if min_key in face:
            out[axis] -= face[min_key] / 100.0 * spans[axis]
        if max_key in face:
            out[axis + 3] += face[max_key] / 100.0 * spans[axis]
    if np.any(out[:3] >= out[3:]):
        raise ValueError("probe target edit produced inverted or zero-volume box")
    edited_spans = out[3:] - out[:3]
    translation = np.zeros(3, dtype=np.float64)
    for axis, axis_name in enumerate(("x", "y", "z")):
        translation[axis] = validated["translation_fraction"].get(axis_name, 0.0) * edited_spans[axis]
    out[:3] += translation
    out[3:] += translation
    if np.any(out[:3] >= out[3:]):
        raise ValueError("probe target edit translation produced inverted or zero-volume box")
    return out.tolist()


def apply_probe_target_edit(base_box: list[float] | np.ndarray, edit: Mapping[str, Any]) -> list[float]:
    return apply_part_grounded_box_edit(
        base_box,
        edit,
        schema_version=PROBE_TARGET_EDIT_SCHEMA_VERSION,
        context_label="probe target",
    )


def apply_anchor_target_edit(base_box: list[float] | np.ndarray, edit: Mapping[str, Any]) -> list[float]:
    return apply_part_grounded_box_edit(
        base_box,
        edit,
        schema_version=ANCHOR_TARGET_EDIT_SCHEMA_VERSION,
        context_label="anchor target",
    )


def _transform_values(transform: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh_to_env = transform.get("mesh_to_env_local") if isinstance(transform.get("mesh_to_env_local"), Mapping) else transform
    if not isinstance(mesh_to_env, Mapping):
        raise ValueError("transform must contain mesh_to_env_local")
    scale = np.asarray(mesh_to_env.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.shape == ():
        scale = np.asarray([float(scale), float(scale), float(scale)], dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)):
        raise ValueError("mesh_to_env_local.scale must be a finite scalar or 3-vector")
    translation = np.asarray(mesh_to_env.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("mesh_to_env_local.translation_m must be a finite 3-vector")
    rotation = [float(value) for value in mesh_to_env.get("rotation", [0.0, 0.0, 0.0])]
    center = [float(value) for value in mesh_to_env.get("center", [0.0, 0.0, 0.0])]
    if any(abs(value) > 1.0e-12 for value in rotation) or any(abs(value) > 1.0e-12 for value in center):
        raise ValueError("probe target compiler only supports zero rotation and zero center transforms")
    return scale, translation


def transform_mesh_points_to_env(points: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    scale, translation = _transform_values(transform)
    return np.asarray(points, dtype=np.float64) * scale.reshape(1, 3) + translation.reshape(1, 3)


def mesh_box_to_env_aabb(mesh_box: list[float] | np.ndarray, transform: Mapping[str, Any]) -> dict[str, Any]:
    box = _box_array(mesh_box, "mesh_box")
    corners = np.asarray(
        [
            [x, y, z]
            for x in (box[0], box[3])
            for y in (box[1], box[4])
            for z in (box[2], box[5])
        ],
        dtype=np.float64,
    )
    env = transform_mesh_points_to_env(corners, transform)
    env_box = _bbox_from_points(env)
    return {"frame": "env_local", "box": env_box.tolist()}


def _primitive_purity(
    *,
    primitive_centroids: np.ndarray,
    primitive_labels: np.ndarray,
    selected_part_id: int,
    box: np.ndarray,
) -> dict[str, Any]:
    inside = _points_inside_box(primitive_centroids, box)
    inside_count = int(np.count_nonzero(inside))
    if inside_count == 0:
        return {"status": "error", "inside_primitive_count": 0, "selected_primitive_count": 0, "selected_part_fraction": 0.0}
    selected = int(np.count_nonzero(primitive_labels[inside] == selected_part_id))
    fraction = selected / inside_count
    return {
        "status": "ok" if fraction >= MIN_PRIMITIVE_PURITY else "warning",
        "inside_primitive_count": inside_count,
        "selected_primitive_count": selected,
        "selected_part_fraction": float(fraction),
    }


def compile_probe_target_from_context(
    context: Mapping[str, Any],
    *,
    selected_part_id: int,
    region_hint: str,
    edit: Mapping[str, Any] | None = None,
    anchor_pin_box: list[float] | None = None,
    mechanics_locality_required: bool = False,
    mechanics_probe_mode: str = "none",
    motion_axis: str = "+Y",
) -> dict[str, Any]:
    if mechanics_probe_mode not in ALLOWED_MECHANICS_PROBE_MODES:
        raise ValueError(f"mechanics_probe_mode must be one of {ALLOWED_MECHANICS_PROBE_MODES}")
    if motion_axis not in ALLOWED_MOTION_AXES:
        raise ValueError(f"motion_axis must be one of {ALLOWED_MOTION_AXES}")
    if mechanics_locality_required and region_hint not in {"tip", "edge_band"}:
        raise ValueError("mechanics-conditioned paired probe targets require region_hint tip or edge_band")
    if mechanics_locality_required and anchor_pin_box is None:
        raise ValueError("mechanics-conditioned probe target requires its owning static anchor box")
    compiled = compile_part_grounded_box_from_context(
        context,
        selected_part_id=selected_part_id,
        region_hint=region_hint,
        edit=edit,
        anchor_pin_box=anchor_pin_box,
        role="probe target",
        allowed_region_hints=ALLOWED_REGION_HINTS,
        compile_schema_version=PROBE_TARGET_COMPILE_SCHEMA_VERSION,
        edit_schema_version=PROBE_TARGET_EDIT_SCHEMA_VERSION,
        validation_schema_version=PROBE_TARGET_VALIDATION_SCHEMA_VERSION,
        include_reference_overlap=True,
        mechanics_locality_required=mechanics_locality_required,
    )
    compiled["mechanics_probe_mode"] = mechanics_probe_mode
    compiled["mechanics_locality_required"] = bool(mechanics_locality_required)
    compiled["motion_axis"] = motion_axis
    compiled["executable_probe_aabb_box"] = {
        "schema_version": "hag4r-executable-probe-aabb-v1",
        "frame": "env_local",
        "box": list(compiled["mesh_frame_box_m"]),
        "source_frame": "local_mesh",
        "source_field": "mesh_frame_box_m",
        "source_box": list(compiled["mesh_frame_box_m"]),
        "scene_encoded_mesh_transform": False,
        "scene_contract": "untransformed_genesis_tetmesh_env_local",
        "transformed_scene_fit_box": compiled["genesis_aabb_box"],
        "transformed_scene_fit_usage": "audit_only",
    }
    return compiled


def compile_part_grounded_box_from_context(
    context: Mapping[str, Any],
    *,
    selected_part_id: int,
    region_hint: str,
    edit: Mapping[str, Any] | None = None,
    anchor_pin_box: list[float] | None = None,
    role: str = "part-grounded box",
    allowed_region_hints: tuple[str, ...] = ALLOWED_PART_GROUNDED_REGION_HINTS,
    compile_schema_version: str = PART_GROUNDED_BOX_COMPILE_SCHEMA_VERSION,
    edit_schema_version: str = PART_GROUNDED_BOX_EDIT_SCHEMA_VERSION,
    validation_schema_version: str = PART_GROUNDED_BOX_VALIDATION_SCHEMA_VERSION,
    include_reference_overlap: bool = False,
    mechanics_locality_required: bool = False,
) -> dict[str, Any]:
    selected_part_id = int(selected_part_id)
    if region_hint not in allowed_region_hints:
        raise ValueError(f"region_hint must be one of {allowed_region_hints}")
    part = _part_lookup(context, selected_part_id)
    inputs = _load_compile_inputs(context)
    mesh = inputs["mesh"]
    primitive_labels = inputs["primitive_labels"]
    primitive_centroids = inputs["primitive_centroids"]
    selected_primitive_mask = primitive_labels == selected_part_id
    if not np.any(selected_primitive_mask):
        raise ValueError(f"selected_part_id {selected_part_id} has zero final primitives")
    selected_primitives = mesh.primitives[selected_primitive_mask]
    selected_vertices = np.unique(selected_primitives.reshape(-1)).astype(np.int64)
    selected_points = mesh.vertices[selected_vertices]
    selected_centroids = primitive_centroids[selected_primitive_mask]
    part_box = _bbox_from_points(selected_points)
    mesh_box = _bbox_from_points(mesh.vertices)
    anchor_box = _box_array(anchor_pin_box, "anchor_pin_box") if anchor_pin_box is not None else None

    locality: dict[str, Any] = {
        "locality_strategy": "ordinary_semantic_hint",
        "locality_provenance": "ordinary part-grounded selection",
    }
    if mechanics_locality_required:
        if anchor_box is None:
            raise ValueError("mechanics-conditioned part-grounded box requires anchor_pin_box")
        if region_hint == "full_part":
            raise ValueError("mechanics-conditioned part-grounded box cannot use full_part")
        if region_hint not in {"tip", "edge_band"}:
            raise ValueError("mechanics-conditioned part-grounded box requires tip or edge_band")
        base_box, locality = _anchor_distal_patch_box(
            selected_vertices=selected_vertices,
            selected_points=selected_points,
            selected_primitives=selected_primitives,
            selected_centroids=selected_centroids,
            anchor_box=anchor_box,
            region_hint=region_hint,
        )
        region_provenance = str(locality["locality_provenance"])
    elif region_hint == "full_part":
        base_box = part_box
        region_provenance = "selected part vertex bbox"
    elif region_hint in {"tip", "root", "grip_root"}:
        cap_hint = "root" if region_hint == "grip_root" else region_hint
        base_box = _cap_box(
            selected_points=selected_points,
            selected_centroids=selected_centroids,
            part_box=part_box,
            region_hint=cap_hint,
            anchor_pin_box=anchor_box,
            mesh_box=mesh_box,
        )
        region_provenance = f"{region_hint} cap along selected part longest axis"
    elif region_hint in {"support_contact", "stable_base"}:
        base_box = _support_band_box(selected_points, part_box)
        region_provenance = f"{region_hint} support band on selected part low y axis"
    elif region_hint in {"edge_band", "hinge_side"}:
        base_box = _edge_band_box(selected_points, part_box)
        region_provenance = "edge band on selected part longest-axis max side"
    else:
        raise ValueError(f"unsupported region_hint: {region_hint}")
    base_box = _expand_to_min_extent(base_box)
    edited_box = base_box
    validated_edit = None
    if edit is not None:
        validated_edit = validate_part_grounded_box_edit(
            edit,
            schema_version=edit_schema_version,
            context_label=role,
        )
        edited_box = np.asarray(
            apply_part_grounded_box_edit(
                base_box,
                validated_edit,
                schema_version=edit_schema_version,
                context_label=role,
            ),
            dtype=np.float64,
        )
        edited_box = _expand_to_min_extent(edited_box)

    purity = _primitive_purity(
        primitive_centroids=primitive_centroids,
        primitive_labels=primitive_labels,
        selected_part_id=selected_part_id,
        box=edited_box,
    )
    part_volume = max(_box_volume(part_box), 1.0e-18)
    target_volume = _box_volume(edited_box)
    actual_box_selected_vertex_count = int(np.count_nonzero(_points_inside_box(selected_points, edited_box)))
    actual_box_selected_vertex_fraction = float(actual_box_selected_vertex_count / selected_vertices.size)
    volume_ratio = target_volume / part_volume
    warning_codes: list[str] = []
    hard_errors: list[str] = []
    if purity["status"] == "warning":
        warning_codes.append("low_primitive_purity")
    if purity["status"] == "error":
        hard_errors.append("no_primitives_inside_target_box")
    if volume_ratio > MAX_TARGET_TO_PART_VOLUME_RATIO:
        warning_codes.append("high_target_to_part_volume_ratio")
    pin_overlap = {
        "available": anchor_box is not None,
        "overlap_volume_m3": 0.0,
        "target_overlap_ratio": 0.0,
        "pin_overlap_ratio": 0.0,
        "warning": False,
    }
    if include_reference_overlap and anchor_box is not None:
        _overlap_box, overlap_volume = _box_overlap(edited_box, anchor_box)
        pin_overlap = {
            "available": True,
            "overlap_volume_m3": float(overlap_volume),
            "target_overlap_ratio": float(overlap_volume / target_volume) if target_volume > 0.0 else 0.0,
            "pin_overlap_ratio": float(overlap_volume / _box_volume(anchor_box)) if _box_volume(anchor_box) > 0.0 else 0.0,
            "warning": bool(target_volume > 0.0 and overlap_volume / target_volume >= PIN_OVERLAP_WARNING_RATIO),
        }
        if pin_overlap["warning"]:
            warning_codes.append("pin_overlap")
    if mechanics_locality_required:
        locality["actual_box_selected_vertex_count"] = actual_box_selected_vertex_count
        locality["actual_box_selected_vertex_fraction"] = actual_box_selected_vertex_fraction
        if actual_box_selected_vertex_count < MECHANICS_MIN_TARGET_VERTEX_COUNT:
            hard_errors.append("mechanics_target_has_insufficient_selected_part_vertices")
        if actual_box_selected_vertex_fraction > MECHANICS_MAX_TARGET_VERTEX_FRACTION:
            hard_errors.append("mechanics_target_box_exceeds_locality_fraction_cap")
        if float(locality.get("incident_primitive_fraction", 1.0)) > MECHANICS_MAX_TARGET_PRIMITIVE_FRACTION:
            hard_errors.append("mechanics_target_primitives_exceed_locality_fraction_cap")
        if float(locality.get("anchor_separation_m", 0.0)) <= MECHANICS_MIN_ANCHOR_SEPARATION_M:
            hard_errors.append("mechanics_target_lacks_anchor_separation")
        if pin_overlap["warning"]:
            hard_errors.append("mechanics_target_overlaps_anchor")
    transform = context.get("transform", {})
    genesis_aabb = mesh_box_to_env_aabb(edited_box, transform)
    status = "error" if hard_errors else ("warning" if warning_codes else "ok")
    metrics = {
        "primitive_purity": purity,
        "target_volume_m3": float(target_volume),
        "selected_part_bbox_volume_m3": float(part_volume),
        "target_to_part_volume_ratio": float(volume_ratio),
        "actual_box_selected_vertex_count": actual_box_selected_vertex_count,
        "actual_box_selected_vertex_fraction": actual_box_selected_vertex_fraction,
    }
    if include_reference_overlap:
        metrics["pin_overlap"] = pin_overlap
    return {
        "schema_version": compile_schema_version,
        "selected_part_id": selected_part_id,
        "part_name": str(part.get("part_name", "")),
        "part_semantics": str(part.get("part_semantics", "")),
        "region_hint": region_hint,
        "region_provenance": region_provenance,
        "selected_vertex_indices": selected_vertices.astype(int).tolist(),
        "selected_primitive_count": int(selected_primitives.shape[0]),
        "base_box_m": base_box.tolist(),
        "mesh_frame_box_m": edited_box.tolist(),
        "genesis_aabb_box": genesis_aabb,
        "transform": transform,
        "metrics": metrics,
        "locality_strategy": str(locality["locality_strategy"]),
        "locality_provenance": str(locality["locality_provenance"]),
        "mechanics_locality": locality if mechanics_locality_required else None,
        "edit": validated_edit,
        "validation": {
            "schema_version": validation_schema_version,
            "status": status,
            "warning_codes": warning_codes,
            "hard_errors": hard_errors,
        },
    }


def compile_anchor_target_from_context(
    context: Mapping[str, Any],
    *,
    selected_part_id: int,
    region_hint: str,
    edit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    compiled = compile_part_grounded_box_from_context(
        context,
        selected_part_id=selected_part_id,
        region_hint=region_hint,
        edit=edit,
        role="anchor target",
        allowed_region_hints=ALLOWED_PART_GROUNDED_REGION_HINTS,
        compile_schema_version=ANCHOR_TARGET_COMPILE_SCHEMA_VERSION,
        edit_schema_version=ANCHOR_TARGET_EDIT_SCHEMA_VERSION,
        validation_schema_version=ANCHOR_TARGET_VALIDATION_SCHEMA_VERSION,
        include_reference_overlap=False,
    )
    compiled["anchor_region_box_m"] = list(compiled["mesh_frame_box_m"])
    return compiled


def _live_data(live_result: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(live_result.get("result"), Mapping):
        return live_result["result"]
    return live_result


def _candidate_object_local_vertices(data: Mapping[str, Any]) -> tuple[list[int], str]:
    for key in ("affected_object_local_vertices", "resolved_object_local_vertices"):
        value = data.get(key)
        if isinstance(value, list):
            return [int(item) for item in value], key
    controllers = data.get("controllers")
    if isinstance(controllers, list) and controllers:
        first = controllers[0]
        if isinstance(first, Mapping) and isinstance(first.get("resolved_object_local_vertices"), list):
            return [int(item) for item in first["resolved_object_local_vertices"]], "controllers[0].resolved_object_local_vertices"
    for key in ("affected_vertices", "resolved_vertices"):
        value = data.get(key)
        if isinstance(value, list):
            if data.get("object_local_vertex_index_space") == "object_local_deformable_vertex_index" or data.get("resolved_vertex_index_space") == "object_local_deformable_vertex_index":
                return [int(item) for item in value], key
            if int(data.get("mesh_offset", 0) or 0) == 0:
                return [int(item) for item in value], key
    if isinstance(controllers, list) and controllers:
        first = controllers[0]
        if isinstance(first, Mapping) and isinstance(first.get("resolved_vertices"), list):
            if first.get("object_local_vertex_index_space") == "object_local_deformable_vertex_index" or first.get("resolved_vertex_index_space") == "object_local_deformable_vertex_index":
                return [int(item) for item in first["resolved_vertices"]], "controllers[0].resolved_vertices"
            if int(first.get("mesh_offset", 0) or 0) == 0:
                return [int(item) for item in first["resolved_vertices"]], "controllers[0].resolved_vertices"
    probe = data.get("probe")
    if isinstance(probe, Mapping):
        controller_state = probe.get("controller_state")
        if isinstance(controller_state, Mapping) and isinstance(controller_state.get("selected_vertices"), list):
            return [int(item) for item in controller_state["selected_vertices"]], "probe.controller_state.selected_vertices"
        if isinstance(probe.get("selected_vertices"), list):
            return [int(item) for item in probe["selected_vertices"]], "probe.selected_vertices"
    return [], ""


def validate_grabbed_vertices_against_part(
    context: Mapping[str, Any],
    compiled_target: Mapping[str, Any],
    live_result: Mapping[str, Any],
) -> dict[str, Any]:
    selected_part_id = int(compiled_target["selected_part_id"])
    semantic_group_part_ids = compiled_target.get("semantic_group_part_ids", [selected_part_id])
    if not isinstance(semantic_group_part_ids, list) or not semantic_group_part_ids:
        raise ValueError("compiled probe target semantic_group_part_ids must be a non-empty list")
    if any(isinstance(part_id, bool) or not isinstance(part_id, int) or part_id < 0 for part_id in semantic_group_part_ids):
        raise ValueError("compiled probe target semantic_group_part_ids must contain non-negative integer part ids")
    semantic_group_part_ids = [int(part_id) for part_id in semantic_group_part_ids]
    if selected_part_id not in semantic_group_part_ids or len(set(semantic_group_part_ids)) != len(semantic_group_part_ids):
        raise ValueError("compiled probe target semantic_group_part_ids must uniquely include selected_part_id")
    inputs = _load_compile_inputs(context)
    vertex_labels = inputs["vertex_labels"]
    vertex_count = int(inputs["mesh"].vertices.shape[0])
    data = _live_data(live_result)
    vertices, source = _candidate_object_local_vertices(data)
    if not vertices:
        return {
            "schema_version": PROBE_TARGET_VALIDATION_SCHEMA_VERSION,
            "status": "error",
            "source": source,
            "selected_part_id": selected_part_id,
            "semantic_group_part_ids": semantic_group_part_ids,
            "grabbed_vertex_count": 0,
            "selected_part_vertex_count": 0,
            "grabbed_selected_part_fraction": 0.0,
            "invalid_vertex_indices": [],
            "warning_codes": [],
            "hard_errors": ["missing_grabbed_vertex_telemetry"],
        }
    invalid = [index for index in vertices if index < 0 or index >= vertex_count]
    if invalid:
        return {
            "schema_version": PROBE_TARGET_VALIDATION_SCHEMA_VERSION,
            "status": "error",
            "source": source,
            "selected_part_id": selected_part_id,
            "semantic_group_part_ids": semantic_group_part_ids,
            "grabbed_vertex_count": len(vertices),
            "selected_part_vertex_count": 0,
            "grabbed_selected_part_fraction": 0.0,
            "invalid_vertex_indices": invalid,
            "warning_codes": [],
            "hard_errors": ["invalid_grabbed_vertex_indices"],
        }
    labels = vertex_labels.primary_labels[np.asarray(vertices, dtype=np.int64)]
    selected = int(np.count_nonzero(np.isin(labels, semantic_group_part_ids)))
    fraction = selected / len(vertices)
    label_counts = Counter(int(label) for label in labels.tolist())
    warning_codes = [] if fraction >= MIN_GRABBED_SELECTED_PART_FRACTION else ["low_grabbed_selected_part_fraction"]
    return {
        "schema_version": PROBE_TARGET_VALIDATION_SCHEMA_VERSION,
        "status": "ok" if not warning_codes else "warning",
        "source": source,
        "selected_part_id": selected_part_id,
        "semantic_group_part_ids": semantic_group_part_ids,
        "grabbed_vertex_count": len(vertices),
        "selected_part_vertex_count": selected,
        "grabbed_selected_part_fraction": float(fraction),
        "grabbed_primary_label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "invalid_vertex_indices": [],
        "warning_codes": warning_codes,
        "hard_errors": [],
    }


def _box_corners(box: list[float] | np.ndarray) -> np.ndarray:
    arr = _box_array(box, "box")
    return np.asarray(
        [[x, y, z] for x in (arr[0], arr[3]) for y in (arr[1], arr[4]) for z in (arr[2], arr[5])],
        dtype=np.float64,
    )


_BOX_EDGE_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 4),
    (3, 1),
    (3, 2),
    (3, 7),
    (5, 1),
    (5, 4),
    (5, 7),
    (6, 2),
    (6, 4),
    (6, 7),
)


def _view_camera(mesh_box: np.ndarray, view_name: str) -> dict[str, Any]:
    camera = triple_view_camera(mesh_box, view_name)
    position = np.asarray(camera["position"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    return {
        "position": position,
        "target": target,
        "up": np.asarray(camera["up"], dtype=np.float64),
        "fly_to": bool(camera["fly_to"]),
        "distance": float(np.linalg.norm(position - target)),
    }


def _projector(camera: Mapping[str, Any], points_for_bounds: np.ndarray, size: int = 512):
    position = np.asarray(camera["position"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rel = points_for_bounds - target.reshape(1, 3)
    xy = np.column_stack([rel @ right, rel @ true_up])
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0e-6)
    scale = (size - 48) / float(np.max(span))
    center_xy = (min_xy + max_xy) * 0.5

    def project(points: np.ndarray) -> list[tuple[int, int]]:
        rel_points = points - target.reshape(1, 3)
        view_xy = np.column_stack([rel_points @ right, rel_points @ true_up])
        px = (view_xy[:, 0] - center_xy[0]) * scale + size * 0.5
        py = size * 0.5 - (view_xy[:, 1] - center_xy[1]) * scale
        return [(int(round(x)), int(round(y))) for x, y in zip(px, py, strict=True)]

    return project


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"part-reference input is not one regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": int(resolved.stat().st_size), "sha256": digest.hexdigest()}


def _part_reference_definition(context: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    mesh = inputs["mesh"]
    primitive_labels = np.asarray(inputs["primitive_labels"], dtype=np.int64)
    vertex_labels = np.asarray(inputs["vertex_labels"].primary_labels, dtype=np.int64)
    sources = context.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("part grounding context is missing sources")
    mesh_identity = _file_identity(Path(str(sources["monolithic_mesh_path"])))
    params_identity = _file_identity(Path(str(sources["monolithic_params_path"])))
    legend: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, part in enumerate(context.get("parts", [])):
        if not isinstance(part, Mapping):
            raise ValueError(f"part grounding context entry {index} is malformed")
        part_id = int(part.get("part_id", part.get("part_index", -1)))
        if part_id < 0 or part_id in seen_ids:
            raise ValueError(f"part grounding context has invalid or duplicate part_id {part_id}")
        seen_ids.add(part_id)
        primitive_count = int(np.count_nonzero(primitive_labels == part_id))
        vertex_count = int(np.count_nonzero(vertex_labels == part_id))
        rgb = [int(max(0, min(255, round(float(value))))) for value in part.get("part_color_rgb", [128, 128, 128])]
        if len(rgb) != 3:
            raise ValueError(f"part {part_id} has malformed part_color_rgb")
        legend.append(
            {
                "part_id": part_id,
                "part_name": str(part.get("part_name", "")),
                "part_semantics": str(part.get("part_semantics", "")),
                "part_color_rgb": rgb,
                "present_in_final_mesh": primitive_count > 0,
                "primitive_count": primitive_count,
                "vertex_count": vertex_count,
            }
        )
    if not legend:
        raise ValueError("part grounding context has no canonical parts")
    selectable = [record["part_id"] for record in legend if record["present_in_final_mesh"]]
    if not selectable:
        raise ValueError("part grounding context has no selectable final-mesh part")
    input_record = {
        "monolithic_mesh": mesh_identity,
        "monolithic_params": params_identity,
        "label_array_key": str(inputs["label_key"]),
        "mesh_vertex_count": int(mesh.vertices.shape[0]),
        "mesh_primitive_count": int(mesh.primitives.shape[0]),
        "mesh_primitive_kind": str(mesh.primitive_kind),
        "transform": context.get("transform", {}),
        "legend": legend,
    }
    return {
        "renderer_policy": PART_REFERENCE_RENDERER_VERSION,
        "view_order": list(PREVIEW_VIEW_NAMES),
        "monolithic_input_hash": _stable_hash(input_record),
        "legend": legend,
        "selectable_part_ids": selectable,
    }


def describe_part_reference(context: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the current monolithic part labels without rendering files."""

    inputs = _load_compile_inputs(context)
    return _part_reference_definition(context, inputs)


def render_part_reference(context: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Render three current-mesh views with explicit canonical part-ID labels."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = _load_compile_inputs(context)
    definition = _part_reference_definition(context, inputs)
    mesh = inputs["mesh"]
    primitive_labels = np.asarray(inputs["primitive_labels"], dtype=np.int64)
    vertex_labels = np.asarray(inputs["vertex_labels"].primary_labels, dtype=np.int64)
    env_vertices = transform_mesh_points_to_env(mesh.vertices, context.get("transform", {}))
    mesh_box = _bbox_from_points(env_vertices)
    colors = {record["part_id"]: tuple(record["part_color_rgb"]) for record in definition["legend"]}
    centroids = {
        part_id: np.mean(
            env_vertices[np.unique(mesh.primitives[primitive_labels == part_id].reshape(-1))],
            axis=0,
        )
        for part_id in definition["selectable_part_ids"]
    }
    views: dict[str, Any] = {}
    for view_name in PREVIEW_VIEW_NAMES:
        camera = _view_camera(mesh_box, view_name)
        image = Image.new("RGB", (512, 512), BACKGROUND_RGB)
        draw = ImageDraw.Draw(image)
        project = _projector(camera, env_vertices, size=512)
        for primitive, label in zip(mesh.primitives, primitive_labels, strict=True):
            points = project(env_vertices[primitive])
            color = colors.get(int(label), (160, 160, 160))
            if mesh.primitive_kind == "triangle":
                draw.polygon(points, fill=color)
                draw.line([*points, points[0]], fill=(72, 72, 72), width=1)
            else:
                for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                    draw.line([points[left], points[right]], fill=color, width=1)
        projected_centroids = {
            part_id: project(np.asarray([centroids[part_id]], dtype=np.float64))[0]
            for part_id in definition["selectable_part_ids"]
        }
        ordered_by_x = sorted(
            definition["selectable_part_ids"],
            key=lambda part_id: (projected_centroids[part_id][0], projected_centroids[part_id][1], part_id),
        )
        split = (len(ordered_by_x) + 1) // 2
        margin_parts = {"left": ordered_by_x[:split], "right": ordered_by_x[split:]}
        placements: dict[int, dict[str, Any]] = {}
        for margin_side, part_ids in margin_parts.items():
            part_ids = sorted(part_ids, key=lambda part_id: (projected_centroids[part_id][1], part_id))
            for row, part_id in enumerate(part_ids, start=1):
                anchor_pixel = projected_centroids[part_id]
                label_pixel = [4, int(round(row * 512 / (len(part_ids) + 1)))] if margin_side == "left" else [508, int(round(row * 512 / (len(part_ids) + 1)))]
                text_anchor = "lm" if margin_side == "left" else "rm"
                text_bbox = draw.textbbox(tuple(label_pixel), f"P{part_id}", anchor=text_anchor)
                label_bbox = [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1]
                leader_end = [label_bbox[2] + 2, label_pixel[1]] if margin_side == "left" else [label_bbox[0] - 2, label_pixel[1]]
                placements[part_id] = {
                    "anchor_pixel": list(anchor_pixel),
                    "label_pixel": label_pixel,
                    "label_bbox": label_bbox,
                    "margin_side": margin_side,
                    "text_anchor": text_anchor,
                    "leader_line": [list(anchor_pixel), leader_end],
                }
        placement_boxes = [placements[part_id]["label_bbox"] for part_id in definition["selectable_part_ids"]]
        for index, left_box in enumerate(placement_boxes):
            if left_box[0] < 0 or left_box[1] < 0 or left_box[2] > 512 or left_box[3] > 512:
                raise ValueError("part-reference margin label does not fit inside the 512px view")
            for right_box in placement_boxes[index + 1:]:
                if min(left_box[2], right_box[2]) > max(left_box[0], right_box[0]) and min(left_box[3], right_box[3]) > max(left_box[1], right_box[1]):
                    raise ValueError("part-reference margin label layout overlaps")
        for part_id in definition["selectable_part_ids"]:
            draw.line(placements[part_id]["leader_line"], fill=PART_REFERENCE_LEADER_RGB, width=1)
        labels: list[dict[str, Any]] = []
        for part_id in definition["selectable_part_ids"]:
            label = f"P{part_id}"
            placement = placements[part_id]
            draw.rectangle(placement["label_bbox"], fill=PART_REFERENCE_LABEL_BACKING_RGB)
            draw.text(tuple(placement["label_pixel"]), label, fill=PART_REFERENCE_LABEL_RGB, anchor=placement["text_anchor"])
            labels.append({"part_id": part_id, "label": label, **placement})
        path = destination / f"{view_name}.png"
        image.save(path)
        views[view_name] = {
            "path": str(path),
            "camera": {
                "position": [float(value) for value in camera["position"]],
                "target": [float(value) for value in camera["target"]],
                "up": [float(value) for value in camera["up"]],
                "fly_to": bool(camera.get("fly_to", False)),
            },
            "labels": labels,
        }
    return {**definition, "views": views}


def _draw_box(draw: ImageDraw.ImageDraw, project, box: list[float], color: tuple[int, int, int], width: int) -> None:
    points = project(_box_corners(box))
    for left, right in _BOX_EDGE_PAIRS:
        draw.line([points[left], points[right]], fill=color, width=width)


def _motion_axis_vector(axis: str) -> np.ndarray:
    vectors = {
        "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
    }
    if axis not in vectors:
        raise ValueError(f"motion_axis must be one of {ALLOWED_MOTION_AXES}")
    return np.asarray(vectors[axis], dtype=np.float64)


def _draw_motion_axis(draw: ImageDraw.ImageDraw, project, box: list[float], axis: str) -> None:
    center = (np.asarray(box[:3], dtype=np.float64) + np.asarray(box[3:], dtype=np.float64)) * 0.5
    extent = max(float(np.max(np.asarray(box[3:]) - np.asarray(box[:3]))), MIN_TARGET_BOX_EXTENT_M)
    end = center + _motion_axis_vector(axis) * (extent * 0.75)
    start_px, end_px = project(np.vstack([center, end]))
    draw.line([start_px, end_px], fill=TARGET_OVERLAY_RGB, width=4)
    draw.text((end_px[0] + 5, end_px[1] + 5), axis, fill=TARGET_OVERLAY_RGB)


def render_probe_target_preview(
    context: Mapping[str, Any],
    compiled_target: Mapping[str, Any],
    *,
    output_dir: Path,
    target_id: str,
    trial_id: str,
    anchor_pin_box: list[float] | None = None,
) -> dict[str, Any]:
    return render_part_grounded_box_preview(
        context,
        compiled_target,
        output_dir=output_dir,
        target_id=target_id,
        trial_id=trial_id,
        source_kind="static_probe_preview",
        primary_overlay_rgb=TARGET_OVERLAY_RGB,
        anchor_pin_box=anchor_pin_box,
        overlay_label="target_aabb_rgb",
    )


def render_part_grounded_box_preview(
    context: Mapping[str, Any],
    compiled_target: Mapping[str, Any],
    *,
    output_dir: Path,
    target_id: str,
    trial_id: str,
    source_kind: str,
    primary_overlay_rgb: tuple[int, int, int],
    anchor_pin_box: list[float] | None = None,
    overlay_label: str = "target_aabb_rgb",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_compile_inputs(context)
    mesh = inputs["mesh"]
    primitive_labels = inputs["primitive_labels"]
    transform = context.get("transform", {})
    env_vertices = transform_mesh_points_to_env(mesh.vertices, transform)
    mesh_box = _bbox_from_points(env_vertices)
    target_box = compiled_target["genesis_aabb_box"]["box"]
    bounds_points = [env_vertices, _box_corners(target_box)]
    motion_axis = compiled_target.get("motion_axis")
    if motion_axis is not None:
        motion_axis = str(motion_axis)
        center = (np.asarray(target_box[:3]) + np.asarray(target_box[3:])) * 0.5
        extent = max(float(np.max(np.asarray(target_box[3:]) - np.asarray(target_box[:3]))), MIN_TARGET_BOX_EXTENT_M)
        bounds_points.append(np.vstack([center, center + _motion_axis_vector(motion_axis) * (extent * 0.75)]))
    if anchor_pin_box is not None:
        bounds_points.append(_box_corners(anchor_pin_box))
    points_for_bounds = np.vstack(bounds_points)
    part_colors = {}
    for part in context.get("parts", []):
        if isinstance(part, Mapping):
            color = tuple(int(max(0, min(255, round(float(value))))) for value in part.get("part_color_rgb", [128, 128, 128]))
            part_colors[int(part.get("part_id", part.get("part_index", 0)))] = color
    selected_part_id = int(compiled_target["selected_part_id"])
    view_records: dict[str, Any] = {}
    source_png_paths_by_view: dict[str, Path] = {}
    cameras_by_view: dict[str, dict[str, Any]] = {}
    for view_name in PREVIEW_VIEW_NAMES:
        camera = _view_camera(mesh_box, view_name)
        image = Image.new("RGB", (512, 512), BACKGROUND_RGB)
        draw = ImageDraw.Draw(image)
        project = _projector(camera, points_for_bounds, size=512)
        for primitive, label in zip(mesh.primitives, primitive_labels, strict=True):
            primitive_points = env_vertices[primitive]
            pts = project(primitive_points)
            color = part_colors.get(int(label), (160, 160, 160))
            if mesh.primitive_kind == "triangle":
                draw.polygon(pts, fill=color)
                draw.line([*pts, pts[0]], fill=(72, 72, 72), width=1)
            else:
                edge_width = 2 if int(label) == selected_part_id else 1
                edge_color = SELECTED_OUTLINE_RGB if int(label) == selected_part_id else color
                for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                    draw.line([pts[left], pts[right]], fill=edge_color, width=edge_width)
        if anchor_pin_box is not None:
            _draw_box(draw, project, anchor_pin_box, PIN_OVERLAY_RGB, width=4)
        _draw_box(draw, project, target_box, primary_overlay_rgb, width=4)
        if motion_axis is not None:
            _draw_motion_axis(draw, project, target_box, motion_axis)
        png_path = output_dir / f"{trial_id}_{view_name}.png"
        image.save(png_path)
        source_png_paths_by_view[view_name] = png_path
        camera_record = {
            "position": [float(value) for value in camera["position"]],
            "target": [float(value) for value in camera["target"]],
            "up": [float(value) for value in camera["up"]],
            "fly_to": bool(camera.get("fly_to", False)),
        }
        cameras_by_view[view_name] = camera_record
        view_records[view_name] = {
            "path": str(png_path),
            "camera": camera_record,
        }
    triptych_png_path = output_dir / f"{trial_id}_triptych.png"
    triptych_metadata = stitch_triptych(source_png_paths_by_view, triptych_png_path)
    triple_view_evidence_id = f"{trial_id}_triple_view"
    triple_view_manifest_path = output_dir / f"{trial_id}_triple_view_manifest.json"
    triple_view_manifest = write_triple_view_manifest(
        triple_view_manifest_path,
        evidence_id=triple_view_evidence_id,
        source_kind=source_kind,
        target_id=target_id,
        compile_id=str(compiled_target.get("compile_id", "")),
        trial_id=trial_id,
        source_png_paths_by_view=source_png_paths_by_view,
        triptych_png_path=triptych_png_path,
        cameras_by_view=cameras_by_view,
        extra={"triptych": triptych_metadata, "motion_axis": motion_axis},
    )
    manifest = {
        "target_id": target_id,
        "compile_id": compiled_target.get("compile_id", ""),
        "trial_id": trial_id,
        "selected_part_id": selected_part_id,
        "region_hint": compiled_target.get("region_hint", ""),
        "motion_axis": motion_axis,
        "motion_axis_arrow": bool(motion_axis is not None),
        "views": view_records,
        "triptych_png_path": str(triptych_png_path),
        "triple_view_manifest_path": str(triple_view_manifest_path),
        "triple_view_evidence_id": triple_view_evidence_id,
        "panel_order": list(TRIPLE_VIEW_PANEL_ORDER),
        "overlay_colors": {
            overlay_label: list(primary_overlay_rgb),
            "anchor_pin_aabb_rgb": list(PIN_OVERLAY_RGB),
            "selected_outline_rgb": list(SELECTED_OUTLINE_RGB),
        },
    }
    manifest_path = output_dir / f"{trial_id}_preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "views": view_records,
        "manifest": manifest,
        "triptych_png_path": str(triptych_png_path),
        "triple_view_manifest_path": str(triple_view_manifest_path),
        "triple_view_evidence_id": triple_view_evidence_id,
        "panel_order": list(TRIPLE_VIEW_PANEL_ORDER),
        "triple_view_manifest": triple_view_manifest,
    }


def render_anchor_target_preview(
    context: Mapping[str, Any],
    compiled_target: Mapping[str, Any],
    *,
    output_dir: Path,
    anchor_id: str,
    trial_id: str,
) -> dict[str, Any]:
    return render_part_grounded_box_preview(
        context,
        compiled_target,
        output_dir=output_dir,
        target_id=anchor_id,
        trial_id=trial_id,
        source_kind="static_anchor_preview",
        primary_overlay_rgb=ANCHOR_TARGET_OVERLAY_RGB,
        overlay_label="anchor_target_aabb_rgb",
    )


__all__ = [
    "ALLOWED_REGION_HINTS",
    "ALLOWED_PART_GROUNDED_REGION_HINTS",
    "ANCHOR_TARGET_COMPILE_SCHEMA_VERSION",
    "ANCHOR_TARGET_EDIT_SCHEMA_VERSION",
    "ANCHOR_TARGET_OVERLAY_RGB",
    "ANCHOR_TARGET_VALIDATION_SCHEMA_VERSION",
    "FACE_ADJUST_KEYS",
    "MIN_GRABBED_SELECTED_PART_FRACTION",
    "MIN_PRIMITIVE_PURITY",
    "MIN_TARGET_BOX_EXTENT_M",
    "PIN_OVERLAP_WARNING_RATIO",
    "PART_REFERENCE_RENDERER_VERSION",
    "PART_GROUNDED_BOX_COMPILE_SCHEMA_VERSION",
    "PART_GROUNDED_BOX_EDIT_SCHEMA_VERSION",
    "PART_GROUNDED_BOX_VALIDATION_SCHEMA_VERSION",
    "PROBE_TARGET_COMPILE_SCHEMA_VERSION",
    "PROBE_TARGET_EDIT_SCHEMA_VERSION",
    "PROBE_TARGET_INTENT_SCHEMA_VERSION",
    "PROBE_TARGET_INTENT_V2_SCHEMA_VERSION",
    "PROBE_TARGET_VALIDATION_SCHEMA_VERSION",
    "OBSERVED_MISMATCH_VALUES",
    "TRANSLATION_KEYS",
    "apply_anchor_target_edit",
    "apply_part_grounded_box_edit",
    "apply_probe_target_edit",
    "compile_anchor_target_from_context",
    "compile_part_grounded_box_from_context",
    "compile_probe_target_from_context",
    "describe_part_reference",
    "mesh_box_to_env_aabb",
    "render_anchor_target_preview",
    "render_part_reference",
    "render_part_grounded_box_preview",
    "render_probe_target_preview",
    "transform_mesh_points_to_env",
    "validate_anchor_target_edit",
    "validate_grabbed_vertices_against_part",
    "validate_part_grounded_box_edit",
    "validate_probe_target_edit",
]
