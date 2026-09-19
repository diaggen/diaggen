from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from hag4r.tools.metric_mesh_scaling import read_medit_tet_mesh
from hag4r.agentic.state import ArtifactRole, Stage, StageRunResult
from hag4r.tools.common import (
    EnvName,
    _artifact,
    _conda_module_argv,
    _repo_root,
    _run_subprocess_stage,
    resolve_conda_env,
)


ATLAS_RESOLUTION = 1024
ATLAS_PADDING_PIXELS = 8
ATLAS_BILINEAR_PADDING = True
REST_ALIGNMENT_MAX_ERROR_M = 1.0e-6
APPEARANCE_SCHEMA_VERSION = "hag4r-omnipart-appearance-v1"
METRIC_GLTF_FRAME = "metric_gltf_y_up"
POST_MESH_TEXTURE_REQUEST_SCHEMA_VERSION = "hag4r-post-mesh-texture-request-v1"
POST_MESH_TEXTURE_VISUAL_MANIFEST_SCHEMA_VERSION = "hag4r-post-mesh-texture-visual-v1"
POST_MESH_TEXTURE_QA_REPORT_SCHEMA_VERSION = "hag4r-post-mesh-texture-qa-v1"

POST_MESH_TEXTURE_BUNDLE_INVENTORY = frozenset(
    {
        "visual_mesh.glb",
        "albedo.png",
        "visual_to_physics.npz",
        "visual_manifest.json",
        "post_mesh_texture_request.json",
        "qa/observed_coverage.png",
        "qa/fill_provenance.png",
        "qa/confidence.png",
        "qa/rest_pose_render.png",
        "qa/report.json",
    }
)

APPEARANCE_VIEW_COUNT = 64
APPEARANCE_VIEW_RESOLUTION = 512
REBAKE_TEXEL_CHUNK_SIZE = 262_144
MAX_SUBPIXEL_CANDIDATE_CELLS_PER_FACE = 262_144
MAX_SUBPIXEL_CANDIDATE_CELLS_TOTAL = 2_000_000
ALPHA_ACCEPT_THRESHOLD = 0.05
TARGET_FRONT_COSINE_THRESHOLD = 0.05
TARGET_SELF_DEPTH_EPSILON_FACTOR = 1.0e-4
SOURCE_TARGET_DEPTH_EPSILON_FACTOR = 2.0e-2
SOURCE_TARGET_NORMAL_DOT_THRESHOLD = 0.25

FILL_PROVENANCE_UNMAPPED = 0
FILL_PROVENANCE_OBSERVED = 1
FILL_PROVENANCE_NEAREST = 2
FILL_PROVENANCE_PART_COLOR = 3
FILL_PROVENANCE_GUTTER = 4

_APPEARANCE_VIEW_ARRAY_SPECS = {
    "rgb": (
        np.dtype(np.uint8),
        (
            APPEARANCE_VIEW_COUNT,
            APPEARANCE_VIEW_RESOLUTION,
            APPEARANCE_VIEW_RESOLUTION,
            3,
        ),
    ),
    "alpha": (
        np.dtype(np.uint8),
        (
            APPEARANCE_VIEW_COUNT,
            APPEARANCE_VIEW_RESOLUTION,
            APPEARANCE_VIEW_RESOLUTION,
        ),
    ),
    "source_depth": (
        np.dtype(np.float32),
        (
            APPEARANCE_VIEW_COUNT,
            APPEARANCE_VIEW_RESOLUTION,
            APPEARANCE_VIEW_RESOLUTION,
        ),
    ),
    "source_normal": (
        np.dtype(np.float16),
        (
            APPEARANCE_VIEW_COUNT,
            APPEARANCE_VIEW_RESOLUTION,
            APPEARANCE_VIEW_RESOLUTION,
            3,
        ),
    ),
    "source_part_id": (
        np.dtype(np.int16),
        (
            APPEARANCE_VIEW_COUNT,
            APPEARANCE_VIEW_RESOLUTION,
            APPEARANCE_VIEW_RESOLUTION,
        ),
    ),
    "extrinsics": (np.dtype(np.float32), (APPEARANCE_VIEW_COUNT, 4, 4)),
    "intrinsics": (np.dtype(np.float32), (APPEARANCE_VIEW_COUNT, 3, 3)),
}
_SOURCE_SURFACE_ARRAY_DTYPES = {
    "vertices": np.dtype(np.float32),
    "faces": np.dtype(np.int32),
    "face_part_id": np.dtype(np.int32),
}
_METRIC_SCALING_SCHEMA_VERSION = "hag4r-metric-max-dimension-scaling-v1"

VISUAL_TO_PHYSICS_ARRAY_KEYS = (
    "visual_rest_vertices_m",
    "visual_faces",
    "visual_uv",
    "surface_vertex_to_physics_vertex",
    "visual_vertex_to_surface_vertex",
    "physics_vertex_indices",
    "boundary_face_tet_indices",
    "boundary_face_part_ids",
)

_BINDING_METHOD = "exact_xatlas_vmapping_to_final_tet_vertex"
_LOCAL_TET_FACES = (
    (0, 1, 2, 3),
    (0, 3, 1, 2),
    (0, 2, 3, 1),
    (1, 3, 2, 0),
)


@dataclass(frozen=True)
class ExactBindingInputs:
    physics_vertices_m: np.ndarray
    tets: np.ndarray
    tet_part_labels: np.ndarray
    declared_part_count: int


@dataclass(frozen=True)
class ExactTetBoundary:
    surface_vertices_m: np.ndarray
    surface_faces: np.ndarray
    surface_vertex_to_physics_vertex: np.ndarray
    boundary_face_tet_indices: np.ndarray
    boundary_face_part_ids: np.ndarray
    internal_face_count: int


@dataclass(frozen=True)
class ExactVisualBinding:
    visual_rest_vertices_m: np.ndarray
    visual_faces: np.ndarray
    visual_uv: np.ndarray
    surface_vertex_to_physics_vertex: np.ndarray
    visual_vertex_to_surface_vertex: np.ndarray
    physics_vertex_indices: np.ndarray
    boundary_face_tet_indices: np.ndarray
    boundary_face_part_ids: np.ndarray


@dataclass(frozen=True)
class AppearanceRebakeEvidence:
    rgb: np.ndarray
    alpha: np.ndarray
    source_depth: np.ndarray
    source_normal: np.ndarray
    source_part_id: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    source_vertices: np.ndarray
    source_faces: np.ndarray
    source_face_part_id: np.ndarray
    internal_z_up_to_gltf_y_up: np.ndarray
    source_bbox_diagonal: float
    declared_part_count: int
    near: float
    far: float


@dataclass(frozen=True)
class RebakeStatistics:
    global_statistics: dict[str, int | float]
    per_part_statistics: tuple[dict[str, int | float], ...]


@dataclass(frozen=True)
class RebakeResult:
    albedo_srgb_uint8: np.ndarray
    chart_part_id: np.ndarray
    chart_face_id: np.ndarray
    observed_mask: np.ndarray
    confidence: np.ndarray
    fill_provenance: np.ndarray
    gutter_part_id: np.ndarray
    statistics: RebakeStatistics


@dataclass(frozen=True)
class PostMeshTextureRequest:
    appearance_manifest_path: Path
    monolithic_mesh_path: Path
    heterogeneous_params_path: Path
    metric_mesh_scaling_path: Path
    inferred_material_path: Path
    output_dir: Path


@dataclass(frozen=True)
class PostMeshTextureBundleResult:
    request_path: Path
    visual_mesh_path: Path
    albedo_path: Path
    binding_path: Path
    visual_manifest_path: Path
    texture_qa_report_path: Path
    timings_s: dict[str, float]


@dataclass(frozen=True)
class _UvTargetRaster:
    chart_face_id: Any
    chart_part_id: Any
    target_points_internal: Any
    target_normals_internal: Any


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any] | type[Any] | None = None) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _require_integer_array(name: str, value: np.ndarray, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail or len(array) == 0:
        expected = "(N" + "".join(f",{size}" for size in shape_tail) + ")"
        raise ValueError(f"{name} must be a nonempty {expected} array")
    return array


def _validate_physics_topology(
    *,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(physics_vertices_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("physics_vertices_m must be a finite nonempty (N,3) array")
    if not np.all(np.isfinite(points)):
        raise ValueError("physics_vertices_m contains non-finite values")

    elements_input = _require_integer_array("tets", np.asarray(tets), (4,))
    elements = np.ascontiguousarray(elements_input, dtype=np.int64)
    if np.any(elements < 0) or np.any(elements >= len(points)):
        raise ValueError("tets contains an out-of-range physics vertex index")
    repeated_vertex_tets = [
        index for index, tet in enumerate(elements) if len(set(map(int, tet))) != 4
    ]
    if repeated_vertex_tets:
        raise ValueError(
            "tets contains repeated vertices at tetrahedra: "
            + ",".join(map(str, repeated_vertex_tets))
        )
    canonical_tets = [tuple(sorted(map(int, tet))) for tet in elements]
    if len(set(canonical_tets)) != len(canonical_tets):
        raise ValueError("tets contains duplicate tetrahedra")

    if (
        isinstance(declared_part_count, bool)
        or not isinstance(declared_part_count, (int, np.integer))
        or int(declared_part_count) <= 0
    ):
        raise ValueError("declared_part_count must be a positive integer")
    part_count = int(declared_part_count)
    if part_count - 1 > np.iinfo(np.int32).max:
        raise ValueError("declared_part_count exceeds the int32 boundary part-ID domain")

    labels = np.asarray(tet_part_labels)
    if labels.dtype != np.dtype(np.int64):
        raise ValueError("tet_part_labels must have dtype int64")
    if labels.shape != (len(elements),):
        raise ValueError("tet_part_labels must have shape (tet_count,)")
    if np.any(labels < 0) or np.any(labels >= part_count):
        raise ValueError("tet_part_labels contains a value outside the declared part domain")

    tet_points = points[elements]
    signed_six_volumes = np.linalg.det(
        np.stack(
            (
                tet_points[:, 1] - tet_points[:, 0],
                tet_points[:, 2] - tet_points[:, 0],
                tet_points[:, 3] - tet_points[:, 0],
            ),
            axis=1,
        )
    )
    if not np.all(np.isfinite(signed_six_volumes)):
        raise ValueError("tet mesh has non-finite tetrahedron volume")
    zero_volume_indices = np.flatnonzero(signed_six_volumes == 0.0)
    if zero_volume_indices.size:
        raise ValueError(
            "tet mesh contains zero-volume tetrahedra: "
            + ",".join(str(int(index)) for index in zero_volume_indices)
        )
    return points, elements, np.ascontiguousarray(labels)


def _cyclic_equal(actual: np.ndarray, expected: np.ndarray) -> bool:
    return bool(
        np.array_equal(actual, expected)
        or np.array_equal(actual, np.roll(expected, -1))
        or np.array_equal(actual, np.roll(expected, -2))
    )


def _validate_boundary_against_physics(
    *,
    boundary: ExactTetBoundary,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
) -> None:
    surface_to_physics = boundary.surface_vertex_to_physics_vertex
    if np.any(boundary.surface_faces < 0) or np.any(
        boundary.surface_faces >= len(surface_to_physics)
    ):
        raise ValueError("surface_faces contains an out-of-range compact boundary vertex")
    if len({tuple(map(int, face)) for face in boundary.surface_faces}) != len(
        boundary.surface_faces
    ):
        raise ValueError("surface_faces contains duplicate oriented faces")

    for face_index, compact_face in enumerate(boundary.surface_faces):
        owner_tet_index = int(boundary.boundary_face_tet_indices[face_index])
        if owner_tet_index < 0 or owner_tet_index >= len(tets):
            raise ValueError("boundary_face_tet_indices contains an out-of-range owning tet")
        physics_face = surface_to_physics[compact_face]
        owner_tet = tets[owner_tet_index]
        if not set(map(int, physics_face)).issubset(set(map(int, owner_tet))):
            raise ValueError("boundary face is not a face of its owning tet")
        opposite_vertices = list(set(map(int, owner_tet)) - set(map(int, physics_face)))
        if len(opposite_vertices) != 1:
            raise ValueError("boundary face does not have exactly one owning-tet opposite vertex")
        a, b, c = physics_vertices_m[physics_face]
        opposite = physics_vertices_m[opposite_vertices[0]]
        orientation_dot = float(np.dot(np.cross(b - a, c - a), opposite - a))
        if not math.isfinite(orientation_dot) or orientation_dot >= 0.0:
            raise ValueError("boundary face winding is not outward from its owning tet")
        if int(boundary.boundary_face_part_ids[face_index]) != int(
            tet_part_labels[owner_tet_index]
        ):
            raise ValueError("boundary face part ID does not match its owning tet label")


def load_exact_binding_inputs(
    *,
    monolithic_mesh_path: Path,
    heterogeneous_params_path: Path,
    appearance_manifest_path: Path,
) -> ExactBindingInputs:
    mesh = read_medit_tet_mesh(monolithic_mesh_path)
    manifest = json.loads(Path(appearance_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("appearance manifest must be a JSON object")
    if manifest.get("schema_version") != APPEARANCE_SCHEMA_VERSION:
        raise ValueError(
            f"appearance manifest schema_version must be {APPEARANCE_SCHEMA_VERSION!r}"
        )
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("appearance manifest parts must be a nonempty list")
    for expected_index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError("every appearance manifest part must be an object")
        part_index = part.get("part_index")
        if (
            isinstance(part_index, bool)
            or not isinstance(part_index, int)
            or part_index != expected_index
        ):
            raise ValueError("appearance manifest part_index values must be contiguous from zero")
    coordinate_frames = manifest.get("coordinate_frames")
    if not isinstance(coordinate_frames, dict) or coordinate_frames.get("gltf") != "gltf_y_up":
        raise ValueError("appearance manifest must declare coordinate_frames.gltf='gltf_y_up'")

    part_count = len(parts)
    with np.load(heterogeneous_params_path, allow_pickle=False) as archive:
        required_keys = {"part_indices", "tet_part_labels"}
        missing = required_keys - set(archive.files)
        if missing:
            raise ValueError(
                "heterogeneous params is missing required arrays: " + ",".join(sorted(missing))
            )
        part_indices = np.asarray(archive["part_indices"])
        tet_part_labels = np.asarray(archive["tet_part_labels"])

    if part_indices.dtype != np.dtype(np.int64):
        raise ValueError("part_indices must have dtype int64")
    if tet_part_labels.dtype != np.dtype(np.int64):
        raise ValueError("tet_part_labels must have dtype int64")
    expected_parts = np.arange(part_count, dtype=np.int64)
    if part_indices.shape != expected_parts.shape or not np.array_equal(
        part_indices, expected_parts
    ):
        raise ValueError("part_indices must exactly match the appearance manifest part domain")
    if tet_part_labels.shape != (len(mesh.tets),):
        raise ValueError("tet_part_labels must have shape (tet_count,)")
    if np.any(tet_part_labels < 0) or np.any(tet_part_labels >= part_count):
        raise ValueError("tet_part_labels contains a value outside the manifest part domain")
    if not set(map(int, np.unique(tet_part_labels))).issubset(set(range(part_count))):
        raise ValueError("tet label set is not a subset of the appearance manifest part set")

    return ExactBindingInputs(
        physics_vertices_m=_readonly(mesh.vertices, dtype=np.float64),
        tets=_readonly(mesh.tets, dtype=np.int64),
        tet_part_labels=_readonly(tet_part_labels, dtype=np.int64),
        declared_part_count=part_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_affine_matrix(name: str, value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite (4,4) affine matrix")
    if not np.array_equal(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(f"{name} must have homogeneous row [0,0,0,1]")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError(f"{name} linear block must be invertible")
    return matrix


def _load_npz_exact(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"{path.name} array keys do not match the frozen contract")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_npz_manifest_record(
    *,
    appearance_dir: Path,
    record: Any,
    expected_name: str,
    arrays: dict[str, np.ndarray],
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{expected_name} manifest file record must be an object")
    if set(record) != {"path", "size_bytes", "sha256", "arrays"}:
        raise ValueError(f"{expected_name} manifest file record fields do not match")
    relative = Path(record["path"]) if isinstance(record.get("path"), str) else None
    if (
        relative is None
        or relative.as_posix() != expected_name
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(
            f"{expected_name} must be a safe manifest-sibling relative artifact"
        )
    resolved = appearance_dir / relative
    if not resolved.is_file():
        raise FileNotFoundError(f"appearance bundle file does not exist: {resolved}")
    if resolved.resolve().parent != appearance_dir.resolve():
        raise ValueError(f"{expected_name} must resolve inside the appearance directory")
    if (
        isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] != resolved.stat().st_size
    ):
        raise ValueError(f"{expected_name} size does not match its manifest record")
    if record.get("sha256") != _sha256_file(resolved):
        raise ValueError(f"{expected_name} hash does not match its manifest record")
    expected_metadata = {
        name: {"dtype": array.dtype.name, "shape": list(array.shape)}
        for name, array in sorted(arrays.items())
    }
    if record.get("arrays") != expected_metadata:
        raise ValueError(f"{expected_name} array metadata does not match")


def load_appearance_rebake_evidence(
    appearance_manifest_path: Path,
) -> AppearanceRebakeEvidence:
    manifest_path = Path(appearance_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("appearance manifest must be a JSON object")
    if manifest.get("schema_version") != APPEARANCE_SCHEMA_VERSION:
        raise ValueError(
            f"appearance manifest schema_version must be {APPEARANCE_SCHEMA_VERSION!r}"
        )

    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("appearance manifest parts must be a nonempty list")
    for expected_index, part in enumerate(parts):
        part_index = part.get("part_index") if isinstance(part, dict) else None
        if (
            not isinstance(part, dict)
            or isinstance(part_index, bool)
            or not isinstance(part_index, int)
            or part_index != expected_index
        ):
            raise ValueError(
                "appearance manifest part_index values must be contiguous from zero"
            )
    part_count = len(parts)

    coordinate_frames = manifest.get("coordinate_frames")
    if (
        not isinstance(coordinate_frames, dict)
        or coordinate_frames.get("source_surface") != "omnipart_internal_z_up"
        or coordinate_frames.get("gltf") != "gltf_y_up"
    ):
        raise ValueError("appearance manifest coordinate frames do not match")
    transform = _validated_affine_matrix(
        "omnipart_internal_z_up_to_gltf_y_up",
        coordinate_frames.get("omnipart_internal_z_up_to_gltf_y_up"),
    )

    capture = manifest.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("appearance manifest capture must be an object")
    if (
        capture.get("view_count") != APPEARANCE_VIEW_COUNT
        or capture.get("resolution")
        != [APPEARANCE_VIEW_RESOLUTION, APPEARANCE_VIEW_RESOLUTION]
        or capture.get("extrinsics_convention")
        != "world_to_camera_column_vector"
        or capture.get("intrinsics_convention") != "normalized_opencv_3x3"
        or capture.get("image_array_order") != "view_height_width_channel"
    ):
        raise ValueError("appearance manifest capture contract does not match")
    near = capture.get("near")
    far = capture.get("far")
    if (
        isinstance(near, bool)
        or isinstance(far, bool)
        or not isinstance(near, (int, float))
        or not isinstance(far, (int, float))
        or not math.isfinite(float(near))
        or not math.isfinite(float(far))
        or float(near) <= 0.0
        or float(far) <= float(near)
    ):
        raise ValueError("appearance capture near/far planes are invalid")

    encodings = manifest.get("encodings")
    expected_encodings = {
        "rgb": "straight_srgb_uint8",
        "alpha": "linear_coverage_uint8",
        "alpha_threshold": ALPHA_ACCEPT_THRESHOLD,
        "source_depth": "camera_space_positive_z_internal_units_background_zero",
        "source_normal": "internal_world_unit_face_normal_background_zero",
        "source_part_id": "zero_based_triangle_part_index_background_minus_one",
    }
    if encodings != expected_encodings:
        raise ValueError("appearance manifest encoding contract does not match")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "appearance_views",
        "source_surface",
    }:
        raise ValueError("appearance manifest file records do not match")
    appearance_dir = manifest_path.parent
    views_path = appearance_dir / "appearance_views.npz"
    surface_path = appearance_dir / "source_surface.npz"
    views = _load_npz_exact(views_path, set(_APPEARANCE_VIEW_ARRAY_SPECS))
    surface = _load_npz_exact(surface_path, set(_SOURCE_SURFACE_ARRAY_DTYPES))
    _validate_npz_manifest_record(
        appearance_dir=appearance_dir,
        record=files["appearance_views"],
        expected_name="appearance_views.npz",
        arrays=views,
    )
    _validate_npz_manifest_record(
        appearance_dir=appearance_dir,
        record=files["source_surface"],
        expected_name="source_surface.npz",
        arrays=surface,
    )

    for name, (dtype, shape) in _APPEARANCE_VIEW_ARRAY_SPECS.items():
        array = views[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"appearance view array {name} has the wrong contract")
        if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"appearance view array {name} contains non-finite values")
    for name, dtype in _SOURCE_SURFACE_ARRAY_DTYPES.items():
        if surface[name].dtype != dtype:
            raise ValueError(f"source surface array {name} has the wrong dtype")

    extrinsics = views["extrinsics"]
    expected_homogeneous_row = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.array_equal(
        extrinsics[:, 3, :],
        np.broadcast_to(expected_homogeneous_row, (APPEARANCE_VIEW_COUNT, 4)),
    ):
        raise ValueError("appearance extrinsics have invalid homogeneous rows")
    rotation_determinants = np.linalg.det(extrinsics[:, :3, :3])
    if not np.all(np.isfinite(rotation_determinants)) or np.any(
        rotation_determinants == 0.0
    ):
        raise ValueError("appearance extrinsic linear blocks must be invertible")
    intrinsics = views["intrinsics"]
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("appearance intrinsics focal lengths must be positive")

    source_part_id = views["source_part_id"]
    if np.any((source_part_id < -1) | (source_part_id >= part_count)):
        raise ValueError("source_part_id contains a value outside the manifest part domain")
    foreground = source_part_id >= 0
    background = ~foreground
    source_depth = views["source_depth"]
    source_normal = views["source_normal"]
    if np.any(source_depth[foreground] <= 0.0):
        raise ValueError("source foreground depth must be positive")
    if np.any(source_depth[background] != 0.0):
        raise ValueError("source background depth must be zero")
    if np.any(source_normal[background] != 0.0):
        raise ValueError("source background normal must be zero")
    foreground_normal_norm = np.linalg.norm(
        source_normal[foreground].astype(np.float32),
        axis=1,
    )
    if np.any(foreground_normal_norm <= 0.0) or not np.allclose(
        foreground_normal_norm,
        1.0,
        atol=5.0e-3,
        rtol=0.0,
    ):
        raise ValueError("source foreground normals must be nonzero unit vectors")
    if np.any(views["rgb"][views["alpha"] == 0] != 0):
        raise ValueError("zero-alpha Gaussian background RGB must be zero")

    vertices = surface["vertices"]
    faces = surface["faces"]
    face_part_id = surface["face_part_id"]
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("source vertices must be a nonempty (N,3) array")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("source vertices contain non-finite values")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError("source faces must be a nonempty (F,3) array")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("source faces contain an out-of-range vertex index")
    if face_part_id.shape != (len(faces),):
        raise ValueError("source_face_part_id must have shape (F,)")
    if np.any(face_part_id < 0) or np.any(face_part_id >= part_count):
        raise ValueError("source_face_part_id lies outside the manifest part domain")
    if not np.array_equal(np.unique(face_part_id), np.arange(part_count, dtype=np.int32)):
        raise ValueError("source_face_part_id does not cover the manifest part domain")
    source_triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        ),
        axis=1,
    )
    if not np.all(np.isfinite(doubled_areas)) or np.any(doubled_areas == 0.0):
        raise ValueError("source faces must have finite nonzero area")

    source_min = vertices.min(axis=0)
    source_max = vertices.max(axis=0)
    source_bbox_diagonal = float(np.linalg.norm(source_max - source_min))
    if not math.isfinite(source_bbox_diagonal) or source_bbox_diagonal <= 0.0:
        raise ValueError("source bounding-box diagonal must be finite and positive")
    source_bounds = manifest.get("source_bounds")
    if (
        not isinstance(source_bounds, dict)
        or source_bounds.get("min") != source_min.tolist()
        or source_bounds.get("max") != source_max.tolist()
    ):
        raise ValueError("appearance manifest source bounds do not match source vertices")

    vertex_cursor = 0
    for part_index, part in enumerate(parts):
        vertex_count = part.get("vertex_count")
        face_count = part.get("face_count")
        if (
            isinstance(vertex_count, bool)
            or isinstance(face_count, bool)
            or not isinstance(vertex_count, int)
            or not isinstance(face_count, int)
            or vertex_count <= 0
            or face_count <= 0
        ):
            raise ValueError("appearance manifest part mesh counts are invalid")
        part_vertices = vertices[vertex_cursor : vertex_cursor + vertex_count]
        if len(part_vertices) != vertex_count:
            raise ValueError("manifest part vertex counts do not cover source vertices")
        part_faces = faces[face_part_id == part_index]
        if len(part_faces) != face_count:
            raise ValueError("manifest part face count does not match source faces")
        if np.any(part_faces < vertex_cursor) or np.any(
            part_faces >= vertex_cursor + vertex_count
        ):
            raise ValueError("source part faces cross their manifest vertex segment")
        if (
            part.get("bounds_min") != part_vertices.min(axis=0).tolist()
            or part.get("bounds_max") != part_vertices.max(axis=0).tolist()
        ):
            raise ValueError("manifest part bounds do not match source vertices")
        vertex_cursor += vertex_count
    if vertex_cursor != len(vertices):
        raise ValueError("manifest part vertex counts do not cover source vertices")

    return AppearanceRebakeEvidence(
        rgb=_readonly(views["rgb"], dtype=np.uint8),
        alpha=_readonly(views["alpha"], dtype=np.uint8),
        source_depth=_readonly(source_depth, dtype=np.float32),
        source_normal=_readonly(source_normal, dtype=np.float16),
        source_part_id=_readonly(source_part_id, dtype=np.int16),
        extrinsics=_readonly(extrinsics, dtype=np.float32),
        intrinsics=_readonly(intrinsics, dtype=np.float32),
        source_vertices=_readonly(vertices, dtype=np.float32),
        source_faces=_readonly(faces, dtype=np.int32),
        source_face_part_id=_readonly(face_part_id, dtype=np.int32),
        internal_z_up_to_gltf_y_up=_readonly(transform, dtype=np.float64),
        source_bbox_diagonal=source_bbox_diagonal,
        declared_part_count=part_count,
        near=float(near),
        far=float(far),
    )


def load_rebake_part_colors(
    heterogeneous_params_path: Path,
    declared_part_count: int,
) -> np.ndarray:
    if (
        isinstance(declared_part_count, bool)
        or not isinstance(declared_part_count, (int, np.integer))
        or int(declared_part_count) <= 0
    ):
        raise ValueError("declared_part_count must be a positive integer")
    part_count = int(declared_part_count)
    with np.load(heterogeneous_params_path, allow_pickle=False) as archive:
        required = {"part_indices", "part_colors"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "heterogeneous params is missing required arrays: "
                + ",".join(sorted(missing))
            )
        part_indices = np.asarray(archive["part_indices"])
        colors = np.asarray(archive["part_colors"])
    expected_parts = np.arange(part_count, dtype=np.int64)
    if (
        part_indices.dtype != np.dtype(np.int64)
        or part_indices.shape != expected_parts.shape
        or not np.array_equal(part_indices, expected_parts)
    ):
        raise ValueError("part_indices must exactly match the declared part domain")
    if colors.shape != (part_count, 3) or not np.issubdtype(
        colors.dtype, np.number
    ):
        raise ValueError("part_colors must be a numeric array with shape (P,3)")
    numeric = np.asarray(colors, dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric < 0.0)
        or np.any(numeric > 255.0)
    ):
        raise ValueError("part_colors must contain integer-valued RGB in [0,255]")
    return _readonly(numeric, dtype=np.uint8)


def extract_exact_tet_boundary(
    *,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> ExactTetBoundary:
    points, elements, labels = _validate_physics_topology(
        physics_vertices_m=physics_vertices_m,
        tets=tets,
        tet_part_labels=tet_part_labels,
        declared_part_count=declared_part_count,
    )
    incidence: dict[
        tuple[int, int, int], list[tuple[tuple[int, int, int], int, int]]
    ] = defaultdict(list)
    for tet_index, tet in enumerate(elements):
        for i, j, k, opposite in _LOCAL_TET_FACES:
            oriented_face = (int(tet[i]), int(tet[j]), int(tet[k]))
            key = tuple(sorted(oriented_face))
            incidence[key].append((oriented_face, int(tet[opposite]), tet_index))
            if len(incidence[key]) > 2:
                raise ValueError(
                    "tet mesh has a non-manifold face with incidence greater than two"
                )

    records: list[tuple[tuple[int, int, int], int]] = []
    internal_face_count = 0
    for key in sorted(incidence):
        occurrences = incidence[key]
        if len(occurrences) == 2:
            internal_face_count += 1
            continue
        face, opposite_index, owner_tet_index = occurrences[0]
        a, b, c = points[list(face)]
        orientation_dot = float(
            np.dot(np.cross(b - a, c - a), points[opposite_index] - a)
        )
        if not math.isfinite(orientation_dot) or orientation_dot == 0.0:
            raise ValueError("tet boundary face orientation is undefined")
        if orientation_dot > 0.0:
            face = (face[0], face[2], face[1])

        a, b, c = points[list(face)]
        doubled_area = float(np.linalg.norm(np.cross(b - a, c - a)))
        if not math.isfinite(doubled_area):
            raise ValueError("tet boundary face area is non-finite")
        if doubled_area == 0.0:
            continue
        minimum_position = face.index(min(face))
        canonical_face = face[minimum_position:] + face[:minimum_position]
        records.append((canonical_face, owner_tet_index))

    if not records:
        raise ValueError("tet mesh has no nonzero-area boundary")
    referenced_physics_vertices = np.asarray(
        sorted({vertex for face, _ in records for vertex in face}),
        dtype=np.int64,
    )
    physics_to_surface = {
        int(physics_index): surface_index
        for surface_index, physics_index in enumerate(referenced_physics_vertices)
    }
    surface_faces = np.asarray(
        [
            tuple(physics_to_surface[physics_index] for physics_index in face)
            for face, _ in records
        ],
        dtype=np.int32,
    )
    owner_tets = np.asarray([owner for _, owner in records], dtype=np.int64)
    boundary = ExactTetBoundary(
        surface_vertices_m=_readonly(
            points[referenced_physics_vertices],
            dtype=np.float32,
        ),
        surface_faces=_readonly(surface_faces, dtype=np.int32),
        surface_vertex_to_physics_vertex=_readonly(
            referenced_physics_vertices,
            dtype=np.int64,
        ),
        boundary_face_tet_indices=_readonly(owner_tets, dtype=np.int64),
        boundary_face_part_ids=_readonly(labels[owner_tets], dtype=np.int32),
        internal_face_count=internal_face_count,
    )
    _validate_boundary_against_physics(
        boundary=boundary,
        physics_vertices_m=points,
        tets=elements,
        tet_part_labels=labels,
    )
    return boundary


def _canonicalize_xatlas_boundary_vertices(
    surface_vertices_m: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(surface_vertices_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError(
            "xatlas boundary vertices must be a nonempty (V,3) array"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("xatlas boundary vertices contain non-finite values")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
        raise ValueError("xatlas boundary extrema are non-finite")
    midpoint = (bbox_min + bbox_max) / 2.0
    if not np.all(np.isfinite(midpoint)):
        raise ValueError("xatlas boundary midpoint is non-finite")
    extent = float(np.max(bbox_max - bbox_min))
    if not math.isfinite(extent):
        raise ValueError("xatlas boundary maximum extent is non-finite")
    if extent <= 0.0:
        raise ValueError("xatlas boundary maximum extent must be positive")

    normalized = np.ascontiguousarray(
        (vertices - midpoint) / extent,
        dtype=np.float32,
    )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("canonical xatlas boundary vertices are non-finite")
    normalized_extent = float(
        np.max(normalized.max(axis=0) - normalized.min(axis=0))
    )
    float32_rounding_tolerance = 4.0 * float(np.finfo(np.float32).eps)
    if not math.isclose(
        normalized_extent,
        1.0,
        rel_tol=0.0,
        abs_tol=float32_rounding_tolerance,
    ):
        raise ValueError(
            "canonical xatlas boundary maximum extent is not one within "
            "float32 rounding"
        )
    return normalized


def unwrap_exact_tet_boundary(
    *,
    boundary: ExactTetBoundary,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> ExactVisualBinding:
    try:
        import xatlas
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "unwrap_exact_tet_boundary requires xatlas and must run in .conda/omnipart"
        ) from exc

    points, elements, labels = _validate_physics_topology(
        physics_vertices_m=physics_vertices_m,
        tets=tets,
        tet_part_labels=tet_part_labels,
        declared_part_count=declared_part_count,
    )
    expected_boundary = extract_exact_tet_boundary(
        physics_vertices_m=points,
        tets=elements,
        tet_part_labels=labels,
        declared_part_count=declared_part_count,
    )
    _require_exact_boundary_match(boundary, expected_boundary)

    atlas = xatlas.Atlas()
    canonical_vertices = _canonicalize_xatlas_boundary_vertices(
        boundary.surface_vertices_m
    )
    atlas.add_mesh(canonical_vertices, boundary.surface_faces)
    chart_options = xatlas.ChartOptions()
    chart_options.fix_winding = False
    pack_options = xatlas.PackOptions()
    pack_options.resolution = ATLAS_RESOLUTION
    pack_options.padding = ATLAS_PADDING_PIXELS
    pack_options.bilinear = ATLAS_BILINEAR_PADDING
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, atlas_faces, visual_uv = atlas[0]

    visual_to_surface = np.asarray(vmapping)
    if visual_to_surface.ndim != 1 or len(visual_to_surface) == 0:
        raise ValueError("xatlas vmapping must be a nonempty one-dimensional array")
    if not np.issubdtype(visual_to_surface.dtype, np.integer):
        raise ValueError("xatlas vmapping must have an integer dtype")
    visual_to_surface = np.ascontiguousarray(visual_to_surface, dtype=np.int64)
    if np.any(visual_to_surface < 0) or np.any(
        visual_to_surface >= len(boundary.surface_vertices_m)
    ):
        raise ValueError("xatlas vmapping contains an out-of-range surface vertex")
    if not np.array_equal(
        np.unique(visual_to_surface),
        np.arange(len(boundary.surface_vertices_m), dtype=np.int64),
    ):
        raise ValueError("xatlas vmapping does not cover every compact boundary vertex")

    visual_faces = np.ascontiguousarray(atlas_faces, dtype=np.int32)
    visual_uv_array = np.ascontiguousarray(visual_uv, dtype=np.float32)
    visual_rest_vertices = np.ascontiguousarray(
        boundary.surface_vertices_m[visual_to_surface],
        dtype=np.float32,
    )
    physics_vertex_indices = np.ascontiguousarray(
        boundary.surface_vertex_to_physics_vertex[visual_to_surface],
        dtype=np.int64,
    )
    binding = ExactVisualBinding(
        visual_rest_vertices_m=_readonly(visual_rest_vertices, dtype=np.float32),
        visual_faces=_readonly(visual_faces, dtype=np.int32),
        visual_uv=_readonly(visual_uv_array, dtype=np.float32),
        surface_vertex_to_physics_vertex=_readonly(
            boundary.surface_vertex_to_physics_vertex,
            dtype=np.int64,
        ),
        visual_vertex_to_surface_vertex=_readonly(visual_to_surface, dtype=np.int64),
        physics_vertex_indices=_readonly(physics_vertex_indices, dtype=np.int64),
        boundary_face_tet_indices=_readonly(
            boundary.boundary_face_tet_indices,
            dtype=np.int64,
        ),
        boundary_face_part_ids=_readonly(
            boundary.boundary_face_part_ids,
            dtype=np.int32,
        ),
    )
    validate_exact_visual_binding(
        binding=binding,
        physics_vertices_m=points,
        tets=elements,
        tet_part_labels=labels,
        declared_part_count=declared_part_count,
    )
    return binding


def _require_exact_boundary_match(
    boundary: ExactTetBoundary,
    expected: ExactTetBoundary,
) -> None:
    array_names = (
        "surface_vertices_m",
        "surface_faces",
        "surface_vertex_to_physics_vertex",
        "boundary_face_tet_indices",
        "boundary_face_part_ids",
    )
    for name in array_names:
        if not np.array_equal(getattr(boundary, name), getattr(expected, name)):
            raise ValueError(f"boundary {name} does not match the exact final tet boundary")
    if boundary.internal_face_count != expected.internal_face_count:
        raise ValueError("boundary internal_face_count does not match the final tet topology")


def _require_binding_array(
    binding: ExactVisualBinding,
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    array = getattr(binding, name)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype.name}")
    if array.ndim != len(shape) or any(
        expected is not None and array.shape[axis] != expected
        for axis, expected in enumerate(shape)
    ):
        raise ValueError(f"{name} has an invalid shape")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return array


def validate_exact_visual_binding(
    *,
    binding: ExactVisualBinding,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> dict[str, int | float | str | bool]:
    points, elements, labels = _validate_physics_topology(
        physics_vertices_m=physics_vertices_m,
        tets=tets,
        tet_part_labels=tet_part_labels,
        declared_part_count=declared_part_count,
    )
    expected_boundary = extract_exact_tet_boundary(
        physics_vertices_m=points,
        tets=elements,
        tet_part_labels=labels,
        declared_part_count=declared_part_count,
    )
    surface_count = len(expected_boundary.surface_vertices_m)
    face_count = len(expected_boundary.surface_faces)

    visual_rest_vertices = _require_binding_array(
        binding,
        "visual_rest_vertices_m",
        dtype=np.dtype(np.float32),
        shape=(None, 3),
    )
    visual_count = len(visual_rest_vertices)
    if visual_count == 0:
        raise ValueError("visual_rest_vertices_m must be nonempty")
    visual_faces = _require_binding_array(
        binding,
        "visual_faces",
        dtype=np.dtype(np.int32),
        shape=(face_count, 3),
    )
    visual_uv = _require_binding_array(
        binding,
        "visual_uv",
        dtype=np.dtype(np.float32),
        shape=(visual_count, 2),
    )
    surface_to_physics = _require_binding_array(
        binding,
        "surface_vertex_to_physics_vertex",
        dtype=np.dtype(np.int64),
        shape=(surface_count,),
    )
    visual_to_surface = _require_binding_array(
        binding,
        "visual_vertex_to_surface_vertex",
        dtype=np.dtype(np.int64),
        shape=(visual_count,),
    )
    physics_indices = _require_binding_array(
        binding,
        "physics_vertex_indices",
        dtype=np.dtype(np.int64),
        shape=(visual_count,),
    )
    owner_tets = _require_binding_array(
        binding,
        "boundary_face_tet_indices",
        dtype=np.dtype(np.int64),
        shape=(face_count,),
    )
    owner_parts = _require_binding_array(
        binding,
        "boundary_face_part_ids",
        dtype=np.dtype(np.int32),
        shape=(face_count,),
    )

    if not np.all(np.isfinite(visual_rest_vertices)):
        raise ValueError("visual_rest_vertices_m contains non-finite values")
    if not np.all(np.isfinite(visual_uv)):
        raise ValueError("visual_uv contains non-finite values")
    if np.any(visual_uv < 0.0) or np.any(visual_uv > 1.0):
        raise ValueError("visual_uv must lie within [0,1]")
    if np.any(visual_faces < 0) or np.any(visual_faces >= visual_count):
        raise ValueError("visual_faces contains an out-of-range visual vertex")
    uv_triangles = visual_uv[visual_faces].astype(np.float64)
    edge_1 = uv_triangles[:, 1] - uv_triangles[:, 0]
    edge_2 = uv_triangles[:, 2] - uv_triangles[:, 0]
    signed_doubled_area = (
        edge_1[:, 0] * edge_2[:, 1]
        - edge_1[:, 1] * edge_2[:, 0]
    )
    non_finite_area_faces = np.flatnonzero(
        ~np.isfinite(signed_doubled_area)
    )
    if len(non_finite_area_faces):
        raise ValueError(
            "visual UV face "
            f"{int(non_finite_area_faces[0])} has non-finite signed area"
        )
    zero_area_faces = np.flatnonzero(signed_doubled_area == 0.0)
    if len(zero_area_faces):
        raise ValueError(
            f"visual UV face {int(zero_area_faces[0])} has zero signed area"
        )
    if np.any(surface_to_physics < 0) or np.any(surface_to_physics >= len(points)):
        raise ValueError("surface_vertex_to_physics_vertex contains an out-of-range index")
    if not np.array_equal(
        surface_to_physics,
        expected_boundary.surface_vertex_to_physics_vertex,
    ):
        raise ValueError(
            "surface_vertex_to_physics_vertex does not match the exact compact boundary"
        )
    if np.any(visual_to_surface < 0) or np.any(visual_to_surface >= surface_count):
        raise ValueError("visual_vertex_to_surface_vertex contains an out-of-range index")
    if not np.array_equal(
        np.unique(visual_to_surface),
        np.arange(surface_count, dtype=np.int64),
    ):
        raise ValueError(
            "visual_vertex_to_surface_vertex does not cover every compact boundary vertex"
        )
    expected_physics_indices = surface_to_physics[visual_to_surface]
    if not np.array_equal(physics_indices, expected_physics_indices):
        raise ValueError("physics_vertex_indices is not the exact two-layer mapping composition")
    if np.any(physics_indices < 0) or np.any(physics_indices >= len(points)):
        raise ValueError("physics_vertex_indices contains an out-of-range physics vertex")
    if not np.array_equal(owner_tets, expected_boundary.boundary_face_tet_indices):
        raise ValueError("boundary_face_tet_indices does not match the exact boundary owners")
    if not np.array_equal(owner_parts, expected_boundary.boundary_face_part_ids):
        raise ValueError("boundary_face_part_ids does not match the owning tet labels")

    reconstructed_surface_faces = visual_to_surface[visual_faces]
    for face_index, reconstructed_face in enumerate(reconstructed_surface_faces):
        if not _cyclic_equal(reconstructed_face, expected_boundary.surface_faces[face_index]):
            raise ValueError(
                "visual face does not map rowwise and cyclically to its boundary face"
            )

    reconstructed_boundary = ExactTetBoundary(
        surface_vertices_m=expected_boundary.surface_vertices_m,
        surface_faces=_readonly(reconstructed_surface_faces, dtype=np.int32),
        surface_vertex_to_physics_vertex=_readonly(surface_to_physics, dtype=np.int64),
        boundary_face_tet_indices=_readonly(owner_tets, dtype=np.int64),
        boundary_face_part_ids=_readonly(owner_parts, dtype=np.int32),
        internal_face_count=expected_boundary.internal_face_count,
    )
    _validate_boundary_against_physics(
        boundary=reconstructed_boundary,
        physics_vertices_m=points,
        tets=elements,
        tet_part_labels=labels,
    )

    alignment_errors = np.abs(
        visual_rest_vertices.astype(np.float64) - points[physics_indices]
    )
    max_alignment_error = float(np.max(alignment_errors))
    if not math.isfinite(max_alignment_error) or max_alignment_error >= REST_ALIGNMENT_MAX_ERROR_M:
        raise ValueError(
            "visual rest vertices are not aligned with their exact physics vertices: "
            f"max_error_m={max_alignment_error}"
        )

    return {
        "coordinate_frame": METRIC_GLTF_FRAME,
        "binding_method": _BINDING_METHOD,
        "physics_vertex_count": int(len(points)),
        "surface_vertex_count": surface_count,
        "visual_vertex_count": visual_count,
        "tet_count": int(len(elements)),
        "boundary_face_count": face_count,
        "internal_face_count": int(expected_boundary.internal_face_count),
        "seam_duplicate_vertex_count": int(visual_count - surface_count),
        "atlas_resolution": ATLAS_RESOLUTION,
        "atlas_padding_pixels": ATLAS_PADDING_PIXELS,
        "atlas_bilinear_padding": ATLAS_BILINEAR_PADDING,
        "max_rest_alignment_error_m": max_alignment_error,
    }


def write_visual_to_physics_npz(
    *,
    output_path: Path,
    binding: ExactVisualBinding,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> Path:
    validate_exact_visual_binding(
        binding=binding,
        physics_vertices_m=physics_vertices_m,
        tets=tets,
        tet_part_labels=tet_part_labels,
        declared_part_count=declared_part_count,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.npz")
    arrays = {name: getattr(binding, name) for name in VISUAL_TO_PHYSICS_ARRAY_KEYS}
    try:
        np.savez_compressed(temporary, **arrays)
        load_visual_to_physics_npz(
            input_path=temporary,
            physics_vertices_m=physics_vertices_m,
            tets=tets,
            tet_part_labels=tet_part_labels,
            declared_part_count=declared_part_count,
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_visual_to_physics_npz(
    *,
    input_path: Path,
    physics_vertices_m: np.ndarray,
    tets: np.ndarray,
    tet_part_labels: np.ndarray,
    declared_part_count: int,
) -> ExactVisualBinding:
    with np.load(input_path, allow_pickle=False) as archive:
        if set(archive.files) != set(VISUAL_TO_PHYSICS_ARRAY_KEYS):
            raise ValueError(
                "visual_to_physics NPZ keys must exactly match the frozen eight-array schema"
            )
        arrays = {
            name: _readonly(np.asarray(archive[name]))
            for name in VISUAL_TO_PHYSICS_ARRAY_KEYS
        }
    binding = ExactVisualBinding(**arrays)
    validate_exact_visual_binding(
        binding=binding,
        physics_vertices_m=physics_vertices_m,
        tets=tets,
        tet_part_labels=tet_part_labels,
        declared_part_count=declared_part_count,
    )
    return binding


def exact_binding_manifest_fragment(
    *,
    boundary: ExactTetBoundary,
    binding: ExactVisualBinding,
    validation_summary: dict[str, int | float | str | bool],
) -> dict[str, object]:
    expected = {
        "coordinate_frame": METRIC_GLTF_FRAME,
        "binding_method": _BINDING_METHOD,
        "surface_vertex_count": int(len(boundary.surface_vertices_m)),
        "boundary_face_count": int(len(boundary.surface_faces)),
        "internal_face_count": int(boundary.internal_face_count),
        "visual_vertex_count": int(len(binding.visual_rest_vertices_m)),
        "seam_duplicate_vertex_count": int(
            len(binding.visual_rest_vertices_m) - len(boundary.surface_vertices_m)
        ),
        "atlas_resolution": ATLAS_RESOLUTION,
        "atlas_padding_pixels": ATLAS_PADDING_PIXELS,
        "atlas_bilinear_padding": ATLAS_BILINEAR_PADDING,
    }
    for key, value in expected.items():
        if validation_summary.get(key) != value:
            raise ValueError(f"validation_summary {key} does not match the exact binding")
    required_passthrough = (
        "physics_vertex_count",
        "tet_count",
        "max_rest_alignment_error_m",
    )
    for key in required_passthrough:
        if key not in validation_summary:
            raise ValueError(f"validation_summary is missing {key}")
    max_error = validation_summary["max_rest_alignment_error_m"]
    if (
        isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or not math.isfinite(float(max_error))
        or float(max_error) < 0.0
        or float(max_error) >= REST_ALIGNMENT_MAX_ERROR_M
    ):
        raise ValueError("validation_summary max_rest_alignment_error_m is invalid")

    return {
        "coordinate_frame": METRIC_GLTF_FRAME,
        "binding_method": _BINDING_METHOD,
        "physics_vertex_count": int(validation_summary["physics_vertex_count"]),
        "tet_count": int(validation_summary["tet_count"]),
        "surface_vertex_count": expected["surface_vertex_count"],
        "boundary_face_count": expected["boundary_face_count"],
        "internal_face_count": expected["internal_face_count"],
        "visual_vertex_count": expected["visual_vertex_count"],
        "seam_duplicate_vertex_count": expected["seam_duplicate_vertex_count"],
        "atlas_resolution": ATLAS_RESOLUTION,
        "atlas_padding_pixels": ATLAS_PADDING_PIXELS,
        "atlas_bilinear_padding": ATLAS_BILINEAR_PADDING,
        "atlas_fix_winding": False,
        "max_rest_alignment_error_m": float(max_error),
    }


def _read_metric_scale_factor(metric_mesh_scaling_path: Path) -> float:
    payload = json.loads(Path(metric_mesh_scaling_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metric_mesh_scaling.json must contain a JSON object")
    if payload.get("schema_version") != _METRIC_SCALING_SCHEMA_VERSION:
        raise ValueError(
            "metric_mesh_scaling.json schema_version must be "
            f"{_METRIC_SCALING_SCHEMA_VERSION!r}"
        )
    scale_factor = payload.get("scale_factor")
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, (int, float))
        or not math.isfinite(float(scale_factor))
        or float(scale_factor) <= 0.0
    ):
        raise ValueError("metric_mesh_scaling.json scale_factor must be finite and > 0")
    return float(scale_factor)


def _metric_gltf_geometry_to_internal(
    vertices_m: np.ndarray,
    face_normals_metric_gltf: np.ndarray,
    *,
    scale_factor: float,
    internal_z_up_to_gltf_y_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_m, dtype=np.float64)
    normals = np.asarray(face_normals_metric_gltf, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError("vertices_m must be a finite nonempty (N,3) array")
    if normals.ndim != 2 or normals.shape[1:] != (3,) or len(normals) == 0:
        raise ValueError(
            "face_normals_metric_gltf must be a finite nonempty (F,3) array"
        )
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
        raise ValueError("metric GLTF geometry contains non-finite values")
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, (int, float, np.integer, np.floating))
        or not math.isfinite(float(scale_factor))
        or float(scale_factor) <= 0.0
    ):
        raise ValueError("scale_factor must be finite and > 0")
    transform = _validated_affine_matrix(
        "internal_z_up_to_gltf_y_up",
        internal_z_up_to_gltf_y_up,
    )
    inverse_transform = np.linalg.inv(transform)
    unscaled_gltf = vertices / float(scale_factor)
    vertices_homogeneous = np.concatenate(
        (unscaled_gltf, np.ones((len(unscaled_gltf), 1), dtype=np.float64)),
        axis=1,
    )
    internal_vertices = vertices_homogeneous @ inverse_transform.T
    internal_normals = normals @ np.linalg.inv(transform[:3, :3]).T
    normal_lengths = np.linalg.norm(internal_normals, axis=1)
    if np.any(normal_lengths == 0.0) or not np.all(np.isfinite(normal_lengths)):
        raise ValueError("transformed target face normals must be finite and nonzero")
    internal_normals = internal_normals / normal_lengths[:, None]
    if (
        not np.all(np.isfinite(internal_vertices))
        or not np.all(np.isfinite(internal_normals))
        or not np.allclose(
            np.linalg.norm(internal_normals, axis=1),
            1.0,
            atol=1.0e-6,
            rtol=0.0,
        )
    ):
        raise ValueError("internal target geometry is non-finite or non-unit")
    return (
        np.ascontiguousarray(internal_vertices[:, :3], dtype=np.float32),
        np.ascontiguousarray(internal_normals, dtype=np.float32),
    )


def _require_rebake_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import nvdiffrast.torch as dr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "visibility-aware rebake requires Torch and nvdiffrast and must run "
            "in .conda/omnipart"
        ) from exc
    return torch, dr


def _require_distance_transform() -> Any:
    try:
        from scipy.ndimage import distance_transform_edt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "visibility-aware rebake requires SciPy and must run in .conda/omnipart"
        ) from exc
    return distance_transform_edt


def _validate_rebake_evidence(evidence: AppearanceRebakeEvidence) -> None:
    if not isinstance(evidence, AppearanceRebakeEvidence):
        raise TypeError("evidence must be an AppearanceRebakeEvidence")
    for name, (dtype, shape) in _APPEARANCE_VIEW_ARRAY_SPECS.items():
        array = getattr(evidence, name)
        if not isinstance(array, np.ndarray) or array.dtype != dtype or array.shape != shape:
            raise ValueError(f"evidence {name} has the wrong dtype or shape")
        if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"evidence {name} contains non-finite values")
    if (
        evidence.source_vertices.dtype != np.dtype(np.float32)
        or evidence.source_vertices.ndim != 2
        or evidence.source_vertices.shape[1:] != (3,)
        or len(evidence.source_vertices) == 0
        or not np.all(np.isfinite(evidence.source_vertices))
    ):
        raise ValueError("evidence source_vertices has the wrong contract")
    if (
        evidence.source_faces.dtype != np.dtype(np.int32)
        or evidence.source_faces.ndim != 2
        or evidence.source_faces.shape[1:] != (3,)
        or len(evidence.source_faces) == 0
    ):
        raise ValueError("evidence source_faces has the wrong contract")
    if np.any(evidence.source_faces < 0) or np.any(
        evidence.source_faces >= len(evidence.source_vertices)
    ):
        raise ValueError("evidence source_faces contains an out-of-range vertex index")
    if (
        evidence.source_face_part_id.dtype != np.dtype(np.int32)
        or evidence.source_face_part_id.shape != (len(evidence.source_faces),)
    ):
        raise ValueError("evidence source_face_part_id has the wrong contract")
    _validated_affine_matrix(
        "evidence internal_z_up_to_gltf_y_up",
        evidence.internal_z_up_to_gltf_y_up,
    )
    if (
        isinstance(evidence.declared_part_count, bool)
        or not isinstance(evidence.declared_part_count, (int, np.integer))
        or int(evidence.declared_part_count) <= 0
    ):
        raise ValueError("evidence declared_part_count must be a positive integer")
    part_count = int(evidence.declared_part_count)
    if np.any(evidence.source_face_part_id < 0) or np.any(
        evidence.source_face_part_id >= part_count
    ):
        raise ValueError("evidence source_face_part_id lies outside the part domain")
    if not np.array_equal(
        np.unique(evidence.source_face_part_id),
        np.arange(part_count, dtype=np.int32),
    ):
        raise ValueError("evidence source_face_part_id does not cover the part domain")
    triangles = evidence.source_vertices[evidence.source_faces]
    doubled_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    if not np.all(np.isfinite(doubled_areas)) or np.any(doubled_areas == 0.0):
        raise ValueError("evidence source faces must have finite nonzero area")
    foreground = evidence.source_part_id >= 0
    background = ~foreground
    if np.any(
        (evidence.source_part_id < -1)
        | (evidence.source_part_id >= part_count)
    ):
        raise ValueError("evidence source_part_id lies outside the part domain")
    if (
        np.any(evidence.source_depth[foreground] <= 0.0)
        or np.any(evidence.source_depth[background] != 0.0)
        or np.any(evidence.source_normal[background] != 0.0)
    ):
        raise ValueError("evidence source G-buffer foreground/background is inconsistent")
    foreground_normal_norm = np.linalg.norm(
        evidence.source_normal[foreground].astype(np.float32),
        axis=1,
    )
    if np.any(foreground_normal_norm <= 0.0) or not np.allclose(
        foreground_normal_norm,
        1.0,
        atol=5.0e-3,
        rtol=0.0,
    ):
        raise ValueError("evidence source foreground normals must be unit vectors")
    if np.any(evidence.rgb[evidence.alpha == 0] != 0):
        raise ValueError("evidence zero-alpha RGB must be zero")
    expected_homogeneous_row = np.asarray(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    if not np.array_equal(
        evidence.extrinsics[:, 3, :],
        np.broadcast_to(expected_homogeneous_row, (APPEARANCE_VIEW_COUNT, 4)),
    ):
        raise ValueError("evidence extrinsics have invalid homogeneous rows")
    determinants = np.linalg.det(evidence.extrinsics[:, :3, :3])
    if not np.all(np.isfinite(determinants)) or np.any(determinants == 0.0):
        raise ValueError("evidence extrinsic linear blocks must be invertible")
    if np.any(evidence.intrinsics[:, 0, 0] <= 0.0) or np.any(
        evidence.intrinsics[:, 1, 1] <= 0.0
    ):
        raise ValueError("evidence intrinsics focal lengths must be positive")
    if (
        not math.isfinite(float(evidence.source_bbox_diagonal))
        or float(evidence.source_bbox_diagonal) <= 0.0
    ):
        raise ValueError("evidence source_bbox_diagonal must be finite and positive")
    actual_diagonal = float(
        np.linalg.norm(
            evidence.source_vertices.max(axis=0)
            - evidence.source_vertices.min(axis=0)
        )
    )
    if actual_diagonal != float(evidence.source_bbox_diagonal):
        raise ValueError("evidence source_bbox_diagonal does not match source vertices")
    if (
        not math.isfinite(float(evidence.near))
        or not math.isfinite(float(evidence.far))
        or float(evidence.near) <= 0.0
        or float(evidence.far) <= float(evidence.near)
    ):
        raise ValueError("evidence near/far planes are invalid")


def _sampler_cell_bounds(index: int, resolution: int) -> tuple[float, float]:
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise ValueError("resolution must be an integer")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if (
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or int(index) < 0
        or int(index) >= resolution
    ):
        raise ValueError("sampler cell index lies outside the resolution")
    scale = float(resolution - 1)
    return (
        max(0.0, (int(index) - 0.5) / scale),
        min(1.0, (int(index) + 0.5) / scale),
    )


def _subpixel_diagnostic(
    *,
    face_index: int,
    x: int,
    y: int,
    message: str,
    clipped: object | None = None,
    fan_areas: object | None = None,
    representative_uv: object | None = None,
    weights: object | None = None,
) -> str:
    details = [
        f"subpixel coverage {message} for face {face_index}, cell (x={x}, y={y})"
    ]
    if clipped is not None:
        details.append(f"clipped={clipped}")
    if fan_areas is not None:
        details.append(f"fan_areas={fan_areas}")
    if representative_uv is not None:
        details.append(f"uv={representative_uv}")
    if weights is not None:
        details.append(f"weights={weights}")
    return "; ".join(details)


def _clip_weighted_polygon_longdouble(
    weighted_triangle: np.ndarray,
    *,
    x_low: np.longdouble,
    x_high: np.longdouble,
    y_low: np.longdouble,
    y_high: np.longdouble,
    face_index: int,
    x: int,
    y: int,
) -> np.ndarray:
    clipped = np.ascontiguousarray(weighted_triangle, dtype=np.longdouble)
    for axis, bound, keep_greater in (
        (0, x_low, True),
        (0, x_high, False),
        (1, y_low, True),
        (1, y_high, False),
    ):
        if len(clipped) == 0:
            break
        output: list[np.ndarray] = []
        start = clipped[-1]
        start_inside = (
            bool(start[axis] >= bound)
            if keep_greater
            else bool(start[axis] <= bound)
        )
        for end in clipped:
            end_inside = (
                bool(end[axis] >= bound)
                if keep_greater
                else bool(end[axis] <= bound)
            )
            if end_inside != start_inside:
                denominator = end[axis] - start[axis]
                if not bool(np.isfinite(denominator)) or denominator == 0:
                    raise RuntimeError(
                        _subpixel_diagnostic(
                            face_index=face_index,
                            x=x,
                            y=y,
                            message=(
                                "encountered an invalid clipping denominator "
                                f"on axis {axis} at bound {bound!r}: "
                                f"start={start.tolist()}, end={end.tolist()}"
                            ),
                        )
                    )
                fraction = (bound - start[axis]) / denominator
                if (
                    not bool(np.isfinite(fraction))
                    or fraction < 0
                    or fraction > 1
                ):
                    raise RuntimeError(
                        _subpixel_diagnostic(
                            face_index=face_index,
                            x=x,
                            y=y,
                            message=(
                                "encountered an invalid clipping fraction "
                                f"{fraction!r} on axis {axis} at bound {bound!r}: "
                                f"start={start.tolist()}, end={end.tolist()}"
                            ),
                        )
                    )
                intersection = (
                    (np.longdouble(1) - fraction) * start + fraction * end
                )
                intersection[axis] = bound
                output.append(intersection)
            if end_inside:
                output.append(end.copy())
            start = end
            start_inside = end_inside
        clipped = (
            np.ascontiguousarray(output, dtype=np.longdouble)
            if output
            else np.empty((0, 5), dtype=np.longdouble)
        )
    if not np.all(np.isfinite(clipped)):
        raise RuntimeError(
            _subpixel_diagnostic(
                face_index=face_index,
                x=x,
                y=y,
                message="produced non-finite weighted clip vertices",
                clipped=clipped.tolist(),
            )
        )
    return clipped


def _fraction_from_float64(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _fraction_to_longdouble(value: Fraction) -> np.longdouble:
    return np.longdouble(value.numerator) / np.longdouble(value.denominator)


def _weighted_representative_fraction(
    triangle: np.ndarray,
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    face_index: int,
    x: int,
    y: int,
) -> tuple[np.longdouble, np.ndarray, np.ndarray] | None:
    zero = Fraction(0)
    one = Fraction(1)
    triangle_fraction = [
        [_fraction_from_float64(value) for value in vertex] for vertex in triangle
    ]
    clipped: list[list[Fraction]] = [
        triangle_fraction[0] + [one, zero, zero],
        triangle_fraction[1] + [zero, one, zero],
        triangle_fraction[2] + [zero, zero, one],
    ]
    for axis, bound_float, keep_greater in (
        (0, x_low, True),
        (0, x_high, False),
        (1, y_low, True),
        (1, y_high, False),
    ):
        if not clipped:
            break
        bound = _fraction_from_float64(bound_float)
        output: list[list[Fraction]] = []
        start = clipped[-1]
        start_inside = start[axis] >= bound if keep_greater else start[axis] <= bound
        for end in clipped:
            end_inside = end[axis] >= bound if keep_greater else end[axis] <= bound
            if end_inside != start_inside:
                denominator = end[axis] - start[axis]
                if denominator == zero:
                    raise RuntimeError(
                        _subpixel_diagnostic(
                            face_index=face_index,
                            x=x,
                            y=y,
                            message=(
                                "exact clipping encountered an invalid parallel "
                                f"crossing on axis {axis} at bound {bound}"
                            ),
                        )
                    )
                fraction = (bound - start[axis]) / denominator
                if fraction < zero or fraction > one:
                    raise RuntimeError(
                        _subpixel_diagnostic(
                            face_index=face_index,
                            x=x,
                            y=y,
                            message=(
                                "exact clipping encountered invalid fraction "
                                f"{fraction} on axis {axis} at bound {bound}"
                            ),
                        )
                    )
                intersection = [
                    (one - fraction) * start[column] + fraction * end[column]
                    for column in range(5)
                ]
                intersection[axis] = bound
                output.append(intersection)
            if end_inside:
                output.append(end.copy())
            start = end
            start_inside = end_inside
        clipped = output
    if len(clipped) < 3:
        return None

    anchor = clipped[0]
    sign = 0
    fan_areas: list[Fraction] = []
    total_doubled_area = zero
    representative_numerator = [zero for _ in range(5)]
    for index in range(1, len(clipped) - 1):
        left = clipped[index]
        right = clipped[index + 1]
        cross = (
            (left[0] - anchor[0]) * (right[1] - anchor[1])
            - (left[1] - anchor[1]) * (right[0] - anchor[0])
        )
        if cross == zero:
            continue
        current_sign = 1 if cross > zero else -1
        if sign and current_sign != sign:
            raise RuntimeError(
                _subpixel_diagnostic(
                    face_index=face_index,
                    x=x,
                    y=y,
                    message="exact weighted clip is non-convex",
                    clipped=[[str(value) for value in row] for row in clipped],
                    fan_areas=[str(value) for value in fan_areas + [cross]],
                )
            )
        sign = current_sign
        magnitude = abs(cross)
        fan_areas.append(cross)
        total_doubled_area += magnitude
        for column in range(5):
            fan_centroid = (
                anchor[column] + left[column] + right[column]
            ) / 3
            representative_numerator[column] += magnitude * fan_centroid
    if total_doubled_area == zero:
        return None
    representative = np.asarray(
        [
            _fraction_to_longdouble(value / total_doubled_area)
            for value in representative_numerator
        ],
        dtype=np.longdouble,
    )
    return (
        _fraction_to_longdouble(total_doubled_area) / np.longdouble(2),
        representative[:2],
        representative[2:],
    )


def _weighted_clipped_triangle_representative(
    triangle: np.ndarray,
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    face_index: int,
    x: int,
    y: int,
) -> tuple[np.longdouble, np.ndarray, np.ndarray] | None:
    source_triangle = np.asarray(triangle, dtype=np.float64)
    if (
        source_triangle.shape != (3, 2)
        or not np.all(np.isfinite(source_triangle))
    ):
        raise ValueError("triangle must be a finite (3,2) array")
    bounds = np.asarray([x_low, x_high, y_low, y_high], dtype=np.float64)
    if not np.all(np.isfinite(bounds)) or x_low > x_high or y_low > y_high:
        raise ValueError("clipping box bounds must be finite and ordered")

    triangle_longdouble = source_triangle.astype(np.longdouble)
    weighted_triangle = np.concatenate(
        (triangle_longdouble, np.eye(3, dtype=np.longdouble)),
        axis=1,
    )
    clipped = _clip_weighted_polygon_longdouble(
        weighted_triangle,
        x_low=np.longdouble(x_low),
        x_high=np.longdouble(x_high),
        y_low=np.longdouble(y_low),
        y_high=np.longdouble(y_high),
        face_index=face_index,
        x=x,
        y=y,
    )
    if len(clipped) < 3:
        return None

    anchor = clipped[0]
    fan_areas: list[np.longdouble] = []
    fan_magnitudes: list[np.longdouble] = []
    ambiguous = False
    sign = 0
    epsilon = np.finfo(np.longdouble).eps
    gamma_three = (np.longdouble(3) * epsilon) / (
        np.longdouble(1) - np.longdouble(3) * epsilon
    )
    for index in range(1, len(clipped) - 1):
        left_delta = clipped[index, :2] - anchor[:2]
        right_delta = clipped[index + 1, :2] - anchor[:2]
        first_product = left_delta[0] * right_delta[1]
        second_product = left_delta[1] * right_delta[0]
        cross = first_product - second_product
        if cross == 0:
            if first_product != 0 or second_product != 0:
                ambiguous = True
            continue
        error_bound = gamma_three * (
            abs(first_product) + abs(second_product)
        )
        if abs(cross) <= error_bound:
            ambiguous = True
        current_sign = 1 if cross > 0 else -1
        if sign and current_sign != sign:
            ambiguous = True
        sign = current_sign if not sign else sign
        fan_areas.append(cross)
        fan_magnitudes.append(abs(cross))
    if ambiguous:
        exact_representative = _weighted_representative_fraction(
            source_triangle,
            x_low=x_low,
            x_high=x_high,
            y_low=y_low,
            y_high=y_high,
            face_index=face_index,
            x=x,
            y=y,
        )
        if exact_representative is None:
            return None
        exact_area, representative_uv, weights = exact_representative
        total_doubled_area = np.longdouble(2) * exact_area
    else:
        if not fan_areas:
            return None
        total_doubled_area = np.sum(
            np.asarray(fan_magnitudes, dtype=np.longdouble),
            dtype=np.longdouble,
        )
        representative = np.zeros(5, dtype=np.longdouble)
        fan_cursor = 0
        for index in range(1, len(clipped) - 1):
            left_delta = clipped[index, :2] - anchor[:2]
            right_delta = clipped[index + 1, :2] - anchor[:2]
            cross = (
                left_delta[0] * right_delta[1]
                - left_delta[1] * right_delta[0]
            )
            if cross == 0:
                continue
            magnitude = fan_magnitudes[fan_cursor]
            fan_cursor += 1
            representative += magnitude * (
                anchor + clipped[index] + clipped[index + 1]
            ) / np.longdouble(3)
        representative /= total_doubled_area
        representative_uv = representative[:2]
        weights = representative[2:]

    if (
        not np.all(np.isfinite(representative_uv))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise RuntimeError(
            _subpixel_diagnostic(
                face_index=face_index,
                x=x,
                y=y,
                message="produced an invalid weighted representative",
                clipped=clipped.tolist(),
                fan_areas=[float(value) for value in fan_areas],
                representative_uv=representative_uv.tolist(),
                weights=weights.tolist(),
            )
        )
    weight_total = np.sum(weights, dtype=np.longdouble)
    if not bool(np.isfinite(weight_total)) or weight_total <= 0:
        raise RuntimeError(
            _subpixel_diagnostic(
                face_index=face_index,
                x=x,
                y=y,
                message="produced a non-positive simplex-weight total",
                clipped=clipped.tolist(),
                fan_areas=[float(value) for value in fan_areas],
                representative_uv=representative_uv.tolist(),
                weights=weights.tolist(),
            )
        )
    weights = weights / weight_total
    reconstructed_uv = weights @ triangle_longdouble
    operation_count = 64 + 32 * len(clipped)
    gamma = (np.longdouble(operation_count) * epsilon) / (
        np.longdouble(1) - np.longdouble(operation_count) * epsilon
    )
    uv_scale = (
        np.sum(np.abs(weights[:, None] * triangle_longdouble), axis=0)
        + np.abs(representative_uv)
    )
    agreement_bound = gamma * uv_scale
    if np.any(np.abs(reconstructed_uv - representative_uv) > agreement_bound):
        raise RuntimeError(
            _subpixel_diagnostic(
                face_index=face_index,
                x=x,
                y=y,
                message="weighted representative UV reconstruction disagrees",
                clipped=clipped.tolist(),
                fan_areas=[float(value) for value in fan_areas],
                representative_uv=representative_uv.tolist(),
                weights=weights.tolist(),
            )
        )
    cell_bounds = np.asarray(
        [[x_low, x_high], [y_low, y_high]], dtype=np.longdouble
    )
    cell_bound_error = gamma * (
        np.abs(reconstructed_uv)
        + np.sum(np.abs(cell_bounds), axis=1)
    )
    if np.any(reconstructed_uv < cell_bounds[:, 0] - cell_bound_error) or np.any(
        reconstructed_uv > cell_bounds[:, 1] + cell_bound_error
    ):
        raise RuntimeError(
            _subpixel_diagnostic(
                face_index=face_index,
                x=x,
                y=y,
                message="weighted representative lies outside its sampler cell",
                clipped=clipped.tolist(),
                fan_areas=[float(value) for value in fan_areas],
                representative_uv=reconstructed_uv.tolist(),
                weights=weights.tolist(),
            )
        )
    return total_doubled_area / np.longdouble(2), representative_uv, weights


def _complete_sampler_coverage(
    *,
    visual_uv: np.ndarray,
    visual_faces: np.ndarray,
    visual_vertices_internal: np.ndarray,
    face_normals_internal: np.ndarray,
    boundary_face_part_ids: np.ndarray,
    chart_face_id: np.ndarray,
    chart_part_id: np.ndarray,
    target_points_internal: np.ndarray,
    target_normals_internal: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise ValueError("resolution must be an integer")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    uv = np.asarray(visual_uv, dtype=np.float64)
    faces = np.asarray(visual_faces)
    vertices = np.asarray(visual_vertices_internal)
    normals = np.asarray(face_normals_internal)
    part_ids = np.asarray(boundary_face_part_ids)
    face_map = np.asarray(chart_face_id)
    part_map = np.asarray(chart_part_id)
    points = np.asarray(target_points_internal)
    target_normals = np.asarray(target_normals_internal)
    face_count = len(faces)
    if uv.ndim != 2 or uv.shape[1:] != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError("visual_uv must be a finite (V,2) array")
    if (
        faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError("visual_faces must be an integer (F,3) array")
    if np.any(faces < 0) or np.any(faces >= len(uv)):
        raise ValueError("visual_faces contains an out-of-range UV vertex index")
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) != len(uv)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(
            "visual_vertices_internal must be a finite (V,3) array"
        )
    if (
        normals.shape != (face_count, 3)
        or not np.all(np.isfinite(normals))
    ):
        raise ValueError(
            "face_normals_internal must be a finite (F,3) array"
        )
    if (
        part_ids.shape != (face_count,)
        or not np.issubdtype(part_ids.dtype, np.integer)
    ):
        raise ValueError("boundary_face_part_ids must be an integer (F,) array")
    expected_hw = (resolution, resolution)
    if (
        face_map.shape != expected_hw
        or not np.issubdtype(face_map.dtype, np.integer)
        or part_map.shape != expected_hw
        or not np.issubdtype(part_map.dtype, np.integer)
        or points.shape != expected_hw + (3,)
        or target_normals.shape != expected_hw + (3,)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(target_normals))
    ):
        raise ValueError("native UV raster arrays have the wrong contract")
    if np.any((face_map < -1) | (face_map >= face_count)):
        raise ValueError("chart_face_id lies outside the face domain")

    uv_triangles = uv[faces]
    edge_1 = uv_triangles[:, 1] - uv_triangles[:, 0]
    edge_2 = uv_triangles[:, 2] - uv_triangles[:, 0]
    signed_doubled_area = (
        edge_1[:, 0] * edge_2[:, 1]
        - edge_1[:, 1] * edge_2[:, 0]
    )
    if not np.all(np.isfinite(signed_doubled_area)):
        raise ValueError("visual UV faces must have finite signed area")
    zero_area_faces = np.flatnonzero(signed_doubled_area == 0.0)
    if len(zero_area_faces):
        raise ValueError(
            f"visual UV face {int(zero_area_faces[0])} has zero signed area"
        )

    completed_face_map = np.ascontiguousarray(face_map.copy())
    completed_part_map = np.ascontiguousarray(part_map.copy())
    completed_points = np.ascontiguousarray(points.copy())
    completed_normals = np.ascontiguousarray(target_normals.copy())
    native_owned = face_map >= 0
    candidate_owner = np.full(expected_hw, -1, dtype=np.int64)
    candidate_weights = np.zeros(expected_hw + (3,), dtype=np.float64)
    total_candidate_cells = 0
    scale = resolution - 1

    for face_index in range(face_count):
        triangle = uv_triangles[face_index]
        minimum = triangle.min(axis=0)
        maximum = triangle.max(axis=0)
        x0 = int(np.clip(np.floor(minimum[0] * scale - 0.5), 0, resolution - 1))
        x1 = int(np.clip(np.ceil(maximum[0] * scale + 0.5), 0, resolution - 1))
        y0 = int(np.clip(np.floor(minimum[1] * scale - 0.5), 0, resolution - 1))
        y1 = int(np.clip(np.ceil(maximum[1] * scale + 0.5), 0, resolution - 1))
        local_y, local_x = np.nonzero(
            ~native_owned[y0 : y1 + 1, x0 : x1 + 1]
        )
        candidate_count = len(local_y)
        uv_bounds = (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        )
        if candidate_count > MAX_SUBPIXEL_CANDIDATE_CELLS_PER_FACE:
            raise ValueError(
                "subpixel candidate bound exceeded for "
                f"face {int(face_index)} with UV bounds {uv_bounds}, "
                f"resolution {resolution}, candidate count {candidate_count}"
            )
        if (
            total_candidate_cells + candidate_count
            > MAX_SUBPIXEL_CANDIDATE_CELLS_TOTAL
        ):
            raise ValueError(
                "total subpixel candidate bound exceeded at "
                f"face {int(face_index)} with UV bounds {uv_bounds}, "
                f"resolution {resolution}, candidate count {candidate_count}, "
                f"total candidate count "
                f"{total_candidate_cells + candidate_count}"
            )
        total_candidate_cells += candidate_count
        for candidate_index in range(candidate_count):
            y = y0 + int(local_y[candidate_index])
            x = x0 + int(local_x[candidate_index])
            y_low, y_high = _sampler_cell_bounds(y, resolution)
            x_low, x_high = _sampler_cell_bounds(x, resolution)
            representative = _weighted_clipped_triangle_representative(
                triangle,
                x_low=x_low,
                x_high=x_high,
                y_low=y_low,
                y_high=y_high,
                face_index=face_index,
                x=x,
                y=y,
            )
            if representative is None:
                continue
            _, _, weights_longdouble = representative
            current_owner = int(candidate_owner[y, x])
            if current_owner < 0 or int(face_index) < current_owner:
                candidate_owner[y, x] = int(face_index)
                candidate_weights[y, x] = weights_longdouble.astype(np.float64)

    candidate_y, candidate_x = np.nonzero(candidate_owner >= 0)
    for y, x in zip(candidate_y, candidate_x, strict=True):
        face_index = int(candidate_owner[y, x])
        weights = candidate_weights[y, x].copy()
        weight_total = float(np.sum(weights))
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not math.isfinite(weight_total)
            or weight_total <= 0
        ):
            raise RuntimeError(
                _subpixel_diagnostic(
                    face_index=face_index,
                    x=int(x),
                    y=int(y),
                    message="stored invalid simplex weights",
                    weights=weights.tolist(),
                )
            )
        weights /= weight_total
        target_point = (
            weights @ vertices[faces[face_index]].astype(np.float64)
        )
        if not np.all(np.isfinite(target_point)) or not np.all(
            np.isfinite(normals[face_index])
        ):
            raise RuntimeError(
                _subpixel_diagnostic(
                    face_index=face_index,
                    x=int(x),
                    y=int(y),
                    message="materialized a non-finite target point or normal",
                    weights=weights.tolist(),
                )
            )
        completed_face_map[y, x] = face_index
        completed_part_map[y, x] = part_ids[face_index]
        completed_points[y, x] = target_point
        completed_normals[y, x] = normals[face_index]

    return (
        completed_face_map,
        completed_part_map,
        completed_points,
        completed_normals,
    )


def _rasterize_uv_target(
    binding: ExactVisualBinding,
    visual_vertices_internal: np.ndarray,
    face_normals_internal: np.ndarray,
    *,
    raster_context: Any,
    device: Any,
    raster_resolution: int = ATLAS_RESOLUTION,
) -> _UvTargetRaster:
    torch, dr = _require_rebake_dependencies()
    if (
        isinstance(raster_resolution, bool)
        or not isinstance(raster_resolution, int)
        or raster_resolution <= 0
    ):
        raise ValueError("raster_resolution must be a positive integer")
    vertices = np.asarray(visual_vertices_internal, dtype=np.float32)
    normals = np.asarray(face_normals_internal, dtype=np.float32)
    if vertices.shape != binding.visual_rest_vertices_m.shape:
        raise ValueError("visual internal vertices do not match binding vertex shape")
    if normals.shape != (len(binding.visual_faces), 3):
        raise ValueError("target face normals do not match binding face shape")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
        raise ValueError("target UV raster attributes must be finite")

    uv = torch.as_tensor(binding.visual_uv, device=device, dtype=torch.float32)
    faces = torch.as_tensor(binding.visual_faces, device=device, dtype=torch.int32)
    positions = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    clip = torch.cat(
        (
            2.0 * uv - 1.0,
            torch.zeros((len(uv), 1), device=device, dtype=torch.float32),
            torch.ones((len(uv), 1), device=device, dtype=torch.float32),
        ),
        dim=1,
    )
    raster, _ = dr.rasterize(
        raster_context,
        clip.unsqueeze(0),
        faces,
        (raster_resolution, raster_resolution),
    )
    points = dr.interpolate(
        positions.unsqueeze(0).contiguous(),
        raster,
        faces,
    )[0][0]
    chart_face_id = raster[0, ..., 3].long() - 1
    interior = chart_face_id >= 0
    chart_part_id = torch.full_like(chart_face_id, -1)
    target_normals = torch.zeros(
        (raster_resolution, raster_resolution, 3),
        device=device,
        dtype=torch.float32,
    )
    part_ids = torch.as_tensor(
        binding.boundary_face_part_ids,
        device=device,
        dtype=torch.int64,
    )
    normal_tensor = torch.as_tensor(normals, device=device, dtype=torch.float32)
    chart_part_id[interior] = part_ids[chart_face_id[interior]]
    target_normals[interior] = normal_tensor[chart_face_id[interior]]
    points = torch.where(interior[..., None], points, torch.zeros_like(points))
    (
        completed_face_id,
        completed_part_id,
        completed_points,
        completed_normals,
    ) = _complete_sampler_coverage(
        visual_uv=binding.visual_uv,
        visual_faces=binding.visual_faces,
        visual_vertices_internal=vertices,
        face_normals_internal=normals,
        boundary_face_part_ids=binding.boundary_face_part_ids,
        chart_face_id=np.ascontiguousarray(
            chart_face_id.detach().cpu().numpy(),
            dtype=np.int64,
        ),
        chart_part_id=np.ascontiguousarray(
            chart_part_id.detach().cpu().numpy(),
            dtype=np.int64,
        ),
        target_points_internal=np.ascontiguousarray(
            points.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        target_normals_internal=np.ascontiguousarray(
            target_normals.detach().cpu().numpy(),
            dtype=np.float32,
        ),
        resolution=raster_resolution,
    )
    return _UvTargetRaster(
        chart_face_id=torch.as_tensor(
            completed_face_id,
            device=device,
            dtype=torch.long,
        ),
        chart_part_id=torch.as_tensor(
            completed_part_id,
            device=device,
            dtype=torch.long,
        ),
        target_points_internal=torch.as_tensor(
            completed_points,
            device=device,
            dtype=torch.float32,
        ),
        target_normals_internal=torch.as_tensor(
            completed_normals,
            device=device,
            dtype=torch.float32,
        ),
    )


def _intrinsics_to_projection(
    intrinsic: Any,
    *,
    near: float,
    far: float,
) -> Any:
    torch, _ = _require_rebake_dependencies()
    projection = torch.zeros(
        (4, 4),
        device=intrinsic.device,
        dtype=intrinsic.dtype,
    )
    projection[0, 0] = 2.0 * intrinsic[0, 0]
    projection[1, 1] = 2.0 * intrinsic[1, 1]
    projection[0, 2] = 2.0 * intrinsic[0, 2] - 1.0
    projection[1, 2] = -2.0 * intrinsic[1, 2] + 1.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = near * far / (near - far)
    projection[3, 2] = 1.0
    return projection


def _rasterize_target_camera_depth(
    visual_vertices_internal: Any,
    visual_faces: Any,
    extrinsic: Any,
    intrinsic: Any,
    *,
    raster_context: Any,
    near: float,
    far: float,
    raster_resolution: int = APPEARANCE_VIEW_RESOLUTION,
) -> Any:
    torch, dr = _require_rebake_dependencies()
    vertices = visual_vertices_internal
    faces = visual_faces
    vertices_homogeneous = torch.cat(
        (
            vertices,
            torch.ones(
                (len(vertices), 1),
                device=vertices.device,
                dtype=vertices.dtype,
            ),
        ),
        dim=1,
    )
    camera_vertices = vertices_homogeneous @ extrinsic.transpose(0, 1)
    projection = _intrinsics_to_projection(intrinsic, near=near, far=far)
    clip_vertices = vertices_homogeneous @ (projection @ extrinsic).transpose(0, 1)
    raster, _ = dr.rasterize(
        raster_context,
        clip_vertices.unsqueeze(0),
        faces,
        (raster_resolution, raster_resolution),
    )
    depth = dr.interpolate(
        camera_vertices[:, 2:3].unsqueeze(0).contiguous(),
        raster,
        faces,
    )[0][0, ..., 0]
    foreground = raster[0, ..., 3] > 0.0
    return torch.where(foreground, depth, torch.zeros_like(depth))


def _sample_texture(
    texture: Any,
    texture_coordinates: Any,
    *,
    filter_mode: str,
) -> Any:
    _, dr = _require_rebake_dependencies()
    sampled = dr.texture(
        texture.unsqueeze(0).contiguous(),
        texture_coordinates.reshape(1, -1, 1, 2).contiguous(),
        filter_mode=filter_mode,
        boundary_mode="zero",
    )
    return sampled[0, :, 0]


def _accumulate_view_observations(
    *,
    target_points_internal: Any,
    target_normals_internal: Any,
    target_part_ids: Any,
    evidence_view: dict[str, Any],
    target_self_depth: Any,
    self_visibility_epsilon: float,
    source_target_depth_epsilon: float,
    sum_rgb: Any,
    sum_weight: Any,
) -> None:
    torch, _ = _require_rebake_dependencies()
    extrinsic = evidence_view["extrinsic"]
    intrinsic = evidence_view["intrinsic"]
    points_homogeneous = torch.cat(
        (
            target_points_internal,
            torch.ones_like(target_points_internal[:, :1]),
        ),
        dim=1,
    )
    points_camera = points_homogeneous @ extrinsic.transpose(0, 1)
    camera_z = points_camera[:, 2]
    safe_z = torch.where(camera_z > 0.0, camera_z, torch.ones_like(camera_z))
    projected_u = intrinsic[0, 0] * points_camera[:, 0] / safe_z + intrinsic[0, 2]
    projected_v = intrinsic[1, 1] * points_camera[:, 1] / safe_z + intrinsic[1, 2]
    image_range_valid = (
        (camera_z > 0.0)
        & (projected_u >= 0.0)
        & (projected_u <= 1.0)
        & (projected_v >= 0.0)
        & (projected_v <= 1.0)
    )
    texture_coordinates = torch.stack((projected_u, projected_v), dim=1)

    sampled_rgb = _sample_texture(
        evidence_view["rgb"],
        texture_coordinates,
        filter_mode="linear",
    )
    sampled_alpha = _sample_texture(
        evidence_view["alpha"],
        texture_coordinates,
        filter_mode="linear",
    )[:, 0]
    sampled_source_depth = _sample_texture(
        evidence_view["source_depth"],
        texture_coordinates,
        filter_mode="linear",
    )[:, 0]
    sampled_source_normal = _sample_texture(
        evidence_view["source_normal"],
        texture_coordinates,
        filter_mode="linear",
    )
    sampled_source_part = _sample_texture(
        evidence_view["source_part_id"],
        texture_coordinates,
        filter_mode="nearest",
    )[:, 0]
    sampled_source_foreground = _sample_texture(
        evidence_view["source_foreground"],
        texture_coordinates,
        filter_mode="nearest",
    )[:, 0]
    sampled_target_depth = _sample_texture(
        target_self_depth[..., None],
        texture_coordinates,
        filter_mode="linear",
    )[:, 0]

    camera_center = torch.linalg.solve(
        extrinsic[:3, :3],
        -extrinsic[:3, 3],
    )
    target_to_camera = camera_center[None, :] - target_points_internal
    target_to_camera_length = torch.linalg.vector_norm(
        target_to_camera,
        dim=1,
    )
    safe_view_length = torch.where(
        target_to_camera_length > 0.0,
        target_to_camera_length,
        torch.ones_like(target_to_camera_length),
    )
    view_direction = target_to_camera / safe_view_length[:, None]
    front = torch.sum(target_normals_internal * view_direction, dim=1)

    source_normal_length = torch.linalg.vector_norm(
        sampled_source_normal,
        dim=1,
    )
    safe_source_normal_length = torch.where(
        source_normal_length > 0.0,
        source_normal_length,
        torch.ones_like(source_normal_length),
    )
    normalized_source_normal = (
        sampled_source_normal / safe_source_normal_length[:, None]
    )
    source_normal_dot_target = torch.sum(
        normalized_source_normal * target_normals_internal,
        dim=1,
    )
    sampled_source_part_integer = torch.round(sampled_source_part).long()
    valid = (
        image_range_valid
        & (sampled_alpha >= ALPHA_ACCEPT_THRESHOLD)
        & (front > TARGET_FRONT_COSINE_THRESHOLD)
        & (sampled_target_depth > 0.0)
        & (
            torch.abs(sampled_target_depth - camera_z)
            <= float(self_visibility_epsilon)
        )
        & (sampled_source_foreground > 0.5)
        & (sampled_source_part_integer == target_part_ids.long())
        & (sampled_source_depth > 0.0)
        & (
            torch.abs(sampled_source_depth - camera_z)
            <= float(source_target_depth_epsilon)
        )
        & (source_normal_length > 0.0)
        & (
            source_normal_dot_target
            >= SOURCE_TARGET_NORMAL_DOT_THRESHOLD
        )
    )
    weights = (
        sampled_alpha
        * torch.square(front)
        * torch.pow(torch.clamp(source_normal_dot_target, min=0.0), 4)
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    sum_rgb.add_(weights[:, None] * sampled_rgb)
    sum_weight.add_(weights)


def _fill_unobserved_chart_texels(
    *,
    albedo: np.ndarray,
    chart_part_id: np.ndarray,
    observed_mask: np.ndarray,
    part_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distance_transform_edt = _require_distance_transform()
    colors = np.asarray(albedo)
    part_map = np.asarray(chart_part_id)
    observed = np.asarray(observed_mask)
    fallback_colors = np.asarray(part_colors)
    if colors.dtype != np.dtype(np.uint8) or colors.ndim != 3 or colors.shape[2:] != (3,):
        raise ValueError("albedo must be a uint8 (H,W,3) array")
    if part_map.dtype != np.dtype(np.int32) or part_map.shape != colors.shape[:2]:
        raise ValueError("chart_part_id must be an int32 (H,W) array")
    if observed.dtype != np.dtype(bool) or observed.shape != colors.shape[:2]:
        raise ValueError("observed_mask must be a bool (H,W) array")
    if (
        fallback_colors.dtype != np.dtype(np.uint8)
        or fallback_colors.ndim != 2
        or fallback_colors.shape[1:] != (3,)
        or len(fallback_colors) == 0
    ):
        raise ValueError("part_colors must be a nonempty uint8 (P,3) array")
    if np.any((part_map < -1) | (part_map >= len(fallback_colors))):
        raise ValueError("chart_part_id lies outside the part domain")
    if np.any(observed & (part_map < 0)):
        raise ValueError("observed_mask may only mark chart-interior texels")

    filled = np.ascontiguousarray(colors.copy())
    provenance = np.full(
        part_map.shape,
        FILL_PROVENANCE_UNMAPPED,
        dtype=np.uint8,
    )
    provenance[observed] = FILL_PROVENANCE_OBSERVED
    for part_id in range(len(fallback_colors)):
        part_interior = part_map == part_id
        if not np.any(part_interior):
            continue
        part_observed = part_interior & observed
        unobserved = part_interior & ~observed
        if not np.any(part_observed):
            filled[part_interior] = fallback_colors[part_id]
            provenance[part_interior] = FILL_PROVENANCE_PART_COLOR
            continue
        _, nearest_indices = distance_transform_edt(
            ~part_observed,
            return_indices=True,
        )
        target_y, target_x = np.nonzero(unobserved)
        source_y = nearest_indices[0, target_y, target_x]
        source_x = nearest_indices[1, target_y, target_x]
        if np.any(part_map[source_y, source_x] != part_id):
            raise RuntimeError("same-part nearest fill selected a different part")
        filled[target_y, target_x] = filled[source_y, source_x]
        provenance[target_y, target_x] = FILL_PROVENANCE_NEAREST
    return filled, provenance


def _propagate_same_part_gutter_with_owner(
    *,
    filled_albedo: np.ndarray,
    chart_part_id: np.ndarray,
    fill_provenance: np.ndarray,
    padding_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_transform_edt = _require_distance_transform()
    colors = np.asarray(filled_albedo)
    part_map = np.asarray(chart_part_id)
    provenance = np.asarray(fill_provenance)
    if (
        colors.dtype != np.dtype(np.uint8)
        or colors.ndim != 3
        or colors.shape[2:] != (3,)
        or part_map.dtype != np.dtype(np.int32)
        or part_map.shape != colors.shape[:2]
        or provenance.dtype != np.dtype(np.uint8)
        or provenance.shape != colors.shape[:2]
    ):
        raise ValueError("gutter inputs have incompatible dtype or shape")
    if (
        isinstance(padding_pixels, bool)
        or not isinstance(padding_pixels, int)
        or padding_pixels < 0
    ):
        raise ValueError("padding_pixels must be a nonnegative integer")
    valid_interior_provenance = {
        FILL_PROVENANCE_OBSERVED,
        FILL_PROVENANCE_NEAREST,
        FILL_PROVENANCE_PART_COLOR,
    }
    interior = part_map >= 0
    if not set(map(int, np.unique(provenance[interior]))).issubset(
        valid_interior_provenance
    ):
        raise ValueError("chart interior must be completely filled before gutter")
    if np.any(provenance[~interior] != FILL_PROVENANCE_UNMAPPED):
        raise ValueError("non-chart pixels must be unmapped before gutter propagation")

    output_colors = np.ascontiguousarray(colors.copy())
    output_provenance = np.ascontiguousarray(provenance.copy())
    owner = np.full(part_map.shape, -1, dtype=np.int32)
    if padding_pixels == 0:
        return output_colors, output_provenance, owner
    best_distance = np.full(part_map.shape, np.inf, dtype=np.float64)
    best_source_y = np.full(part_map.shape, -1, dtype=np.int64)
    best_source_x = np.full(part_map.shape, -1, dtype=np.int64)
    background = part_map == -1
    part_ids = sorted(map(int, np.unique(part_map[interior])))
    for part_id in part_ids:
        part_interior = part_map == part_id
        distances, nearest_indices = distance_transform_edt(
            ~part_interior,
            return_indices=True,
        )
        candidates = background & (distances <= float(padding_pixels))
        closer = candidates & (distances < best_distance)
        tied_lower_part = (
            candidates
            & (distances == best_distance)
            & ((owner < 0) | (part_id < owner))
        )
        update = closer | tied_lower_part
        best_distance[update] = distances[update]
        owner[update] = part_id
        best_source_y[update] = nearest_indices[0][update]
        best_source_x[update] = nearest_indices[1][update]

    gutter = owner >= 0
    source_y = best_source_y[gutter]
    source_x = best_source_x[gutter]
    if np.any(part_map[source_y, source_x] != owner[gutter]):
        raise RuntimeError("same-part gutter selected a different part")
    output_colors[gutter] = colors[source_y, source_x]
    output_provenance[gutter] = FILL_PROVENANCE_GUTTER
    return output_colors, output_provenance, owner


def _propagate_same_part_gutter(
    *,
    filled_albedo: np.ndarray,
    chart_part_id: np.ndarray,
    fill_provenance: np.ndarray,
    padding_pixels: int = ATLAS_PADDING_PIXELS,
) -> tuple[np.ndarray, np.ndarray]:
    albedo, provenance, _ = _propagate_same_part_gutter_with_owner(
        filled_albedo=filled_albedo,
        chart_part_id=chart_part_id,
        fill_provenance=fill_provenance,
        padding_pixels=padding_pixels,
    )
    return albedo, provenance


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _statistics_record(
    *,
    part_id: int | None,
    chart_mask: np.ndarray,
    gutter_mask: np.ndarray,
    fill_provenance: np.ndarray,
) -> dict[str, int | float]:
    chart_count = int(np.count_nonzero(chart_mask))
    observed_count = int(
        np.count_nonzero(
            chart_mask & (fill_provenance == FILL_PROVENANCE_OBSERVED)
        )
    )
    nearest_count = int(
        np.count_nonzero(
            chart_mask & (fill_provenance == FILL_PROVENANCE_NEAREST)
        )
    )
    part_color_count = int(
        np.count_nonzero(
            chart_mask & (fill_provenance == FILL_PROVENANCE_PART_COLOR)
        )
    )
    gutter_count = int(np.count_nonzero(gutter_mask))
    footprint_count = chart_count + gutter_count
    record: dict[str, int | float] = {
        "chart_interior_texel_count": chart_count,
        "observed_texel_count": observed_count,
        "observed_fraction": _fraction(observed_count, chart_count),
        "nearest_filled_texel_count": nearest_count,
        "nearest_filled_fraction": _fraction(nearest_count, chart_count),
        "part_color_filled_texel_count": part_color_count,
        "part_color_filled_fraction": _fraction(part_color_count, chart_count),
        "gutter_texel_count": gutter_count,
        "gutter_fraction": _fraction(gutter_count, footprint_count),
    }
    if part_id is not None:
        record = {"part_id": part_id, **record}
    return record


def _compute_rebake_statistics(
    chart_part_id: np.ndarray,
    fill_provenance: np.ndarray,
    declared_part_count: int,
    *,
    gutter_part_id: np.ndarray | None = None,
) -> RebakeStatistics:
    part_map = np.asarray(chart_part_id)
    provenance = np.asarray(fill_provenance)
    if part_map.dtype != np.dtype(np.int32) or provenance.dtype != np.dtype(np.uint8):
        raise ValueError("statistics maps have invalid dtypes")
    if part_map.shape != provenance.shape:
        raise ValueError("statistics maps must have matching shapes")
    if (
        isinstance(declared_part_count, bool)
        or not isinstance(declared_part_count, (int, np.integer))
        or int(declared_part_count) <= 0
    ):
        raise ValueError("declared_part_count must be a positive integer")
    part_count = int(declared_part_count)
    if np.any((part_map < -1) | (part_map >= part_count)):
        raise ValueError("chart_part_id lies outside the declared part domain")
    provenance_domain = {
        FILL_PROVENANCE_UNMAPPED,
        FILL_PROVENANCE_OBSERVED,
        FILL_PROVENANCE_NEAREST,
        FILL_PROVENANCE_PART_COLOR,
        FILL_PROVENANCE_GUTTER,
    }
    if not set(map(int, np.unique(provenance))).issubset(provenance_domain):
        raise ValueError("fill_provenance contains an unknown category")
    gutter = provenance == FILL_PROVENANCE_GUTTER
    if gutter_part_id is None:
        if np.any(gutter):
            raise ValueError("per-part gutter statistics require gutter_part_id")
        gutter_owner = np.full(part_map.shape, -1, dtype=np.int32)
    else:
        gutter_owner = np.asarray(gutter_part_id)
        if gutter_owner.dtype != np.dtype(np.int32) or gutter_owner.shape != part_map.shape:
            raise ValueError("gutter_part_id must be an int32 map matching chart_part_id")
        if np.any(gutter & ((gutter_owner < 0) | (gutter_owner >= part_count))):
            raise ValueError("every gutter texel must have a valid part owner")
        if np.any(~gutter & (gutter_owner != -1)):
            raise ValueError("only gutter texels may have a gutter part owner")

    chart = part_map >= 0
    global_record = _statistics_record(
        part_id=None,
        chart_mask=chart,
        gutter_mask=gutter,
        fill_provenance=provenance,
    )
    per_part = tuple(
        _statistics_record(
            part_id=part_id,
            chart_mask=part_map == part_id,
            gutter_mask=gutter_owner == part_id,
            fill_provenance=provenance,
        )
        for part_id in range(part_count)
    )
    return RebakeStatistics(
        global_statistics=global_record,
        per_part_statistics=per_part,
    )


def _validated_part_colors(part_colors: np.ndarray, part_count: int) -> np.ndarray:
    colors = np.asarray(part_colors)
    if colors.dtype != np.dtype(np.uint8) or colors.shape != (part_count, 3):
        raise ValueError("part_colors must be uint8 with shape (declared_part_count,3)")
    return np.ascontiguousarray(colors)


def _validate_rebake_result(result: RebakeResult) -> None:
    expected_hw = (ATLAS_RESOLUTION, ATLAS_RESOLUTION)
    specifications = (
        ("albedo_srgb_uint8", np.dtype(np.uint8), expected_hw + (3,)),
        ("chart_part_id", np.dtype(np.int32), expected_hw),
        ("chart_face_id", np.dtype(np.int32), expected_hw),
        ("observed_mask", np.dtype(bool), expected_hw),
        ("confidence", np.dtype(np.float32), expected_hw),
        ("fill_provenance", np.dtype(np.uint8), expected_hw),
        ("gutter_part_id", np.dtype(np.int32), expected_hw),
    )
    for name, dtype, shape in specifications:
        array = getattr(result, name)
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"rebake result {name} has the wrong dtype or shape")
        if not array.flags.c_contiguous or array.flags.writeable:
            raise ValueError(f"rebake result {name} must be readonly and C-contiguous")
    if not np.all(np.isfinite(result.confidence)) or np.any(result.confidence < 0.0):
        raise ValueError("rebake result confidence must be finite and nonnegative")


def _orient_rebake_for_pil_gltf_serialization(
    rebake: RebakeResult,
) -> RebakeResult:
    """Convert UV-raster bottom-up rows to the top-down serialized image order."""
    _validate_rebake_result(rebake)
    serialized = RebakeResult(
        albedo_srgb_uint8=_readonly(np.flip(rebake.albedo_srgb_uint8, axis=0)),
        chart_part_id=_readonly(np.flip(rebake.chart_part_id, axis=0)),
        chart_face_id=_readonly(np.flip(rebake.chart_face_id, axis=0)),
        observed_mask=_readonly(np.flip(rebake.observed_mask, axis=0)),
        confidence=_readonly(np.flip(rebake.confidence, axis=0)),
        fill_provenance=_readonly(np.flip(rebake.fill_provenance, axis=0)),
        gutter_part_id=_readonly(np.flip(rebake.gutter_part_id, axis=0)),
        statistics=rebake.statistics,
    )
    _validate_rebake_result(serialized)
    return serialized


def rebake_visibility_aware_atlas(
    *,
    binding: ExactVisualBinding,
    binding_inputs: ExactBindingInputs,
    evidence: AppearanceRebakeEvidence,
    metric_mesh_scaling_path: Path,
    part_colors: np.ndarray,
    timings_s: dict[str, float] | None = None,
) -> RebakeResult:
    bake_started = time.perf_counter()
    if not isinstance(binding_inputs, ExactBindingInputs):
        raise TypeError("binding_inputs must be ExactBindingInputs")
    _validate_rebake_evidence(evidence)
    if binding_inputs.declared_part_count != evidence.declared_part_count:
        raise ValueError("binding inputs and appearance evidence part counts differ")
    colors = _validated_part_colors(
        part_colors,
        binding_inputs.declared_part_count,
    )
    validate_exact_visual_binding(
        binding=binding,
        physics_vertices_m=binding_inputs.physics_vertices_m,
        tets=binding_inputs.tets,
        tet_part_labels=binding_inputs.tet_part_labels,
        declared_part_count=binding_inputs.declared_part_count,
    )
    scale_factor = _read_metric_scale_factor(metric_mesh_scaling_path)

    metric_triangles = binding.visual_rest_vertices_m[binding.visual_faces]
    metric_face_normals = np.cross(
        metric_triangles[:, 1] - metric_triangles[:, 0],
        metric_triangles[:, 2] - metric_triangles[:, 0],
    )
    metric_face_normal_lengths = np.linalg.norm(metric_face_normals, axis=1)
    if np.any(metric_face_normal_lengths == 0.0) or not np.all(
        np.isfinite(metric_face_normal_lengths)
    ):
        raise ValueError("final target visual mesh contains a zero-area face")
    metric_face_normals /= metric_face_normal_lengths[:, None]
    visual_vertices_internal, face_normals_internal = (
        _metric_gltf_geometry_to_internal(
            binding.visual_rest_vertices_m,
            metric_face_normals,
            scale_factor=scale_factor,
            internal_z_up_to_gltf_y_up=evidence.internal_z_up_to_gltf_y_up,
        )
    )

    torch, dr = _require_rebake_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "visibility-aware rebake requires CUDA and must run in .conda/omnipart"
        )
    device = torch.device("cuda")
    raster_context = dr.RasterizeCudaContext(device=device)
    uv_target = _rasterize_uv_target(
        binding,
        visual_vertices_internal,
        face_normals_internal,
        raster_context=raster_context,
        device=device,
    )
    flat_interior = torch.nonzero(
        uv_target.chart_face_id.reshape(-1) >= 0,
        as_tuple=False,
    )[:, 0]
    if len(flat_interior) == 0:
        raise ValueError("UV target raster contains no chart-interior texels")
    target_points = uv_target.target_points_internal.reshape(-1, 3)[flat_interior]
    target_normals = uv_target.target_normals_internal.reshape(-1, 3)[flat_interior]
    target_parts = uv_target.chart_part_id.reshape(-1)[flat_interior]
    sum_rgb = torch.zeros(
        (len(flat_interior), 3),
        device=device,
        dtype=torch.float32,
    )
    sum_weight = torch.zeros(
        (len(flat_interior),),
        device=device,
        dtype=torch.float32,
    )
    vertices_tensor = torch.as_tensor(
        visual_vertices_internal,
        device=device,
        dtype=torch.float32,
    )
    faces_tensor = torch.as_tensor(
        binding.visual_faces,
        device=device,
        dtype=torch.int32,
    )
    self_visibility_epsilon = (
        TARGET_SELF_DEPTH_EPSILON_FACTOR * evidence.source_bbox_diagonal
    )
    source_target_depth_epsilon = (
        SOURCE_TARGET_DEPTH_EPSILON_FACTOR * evidence.source_bbox_diagonal
    )
    for view_index in range(APPEARANCE_VIEW_COUNT):
        extrinsic = torch.as_tensor(
            evidence.extrinsics[view_index],
            device=device,
            dtype=torch.float32,
        )
        intrinsic = torch.as_tensor(
            evidence.intrinsics[view_index],
            device=device,
            dtype=torch.float32,
        )
        target_self_depth = _rasterize_target_camera_depth(
            vertices_tensor,
            faces_tensor,
            extrinsic,
            intrinsic,
            raster_context=raster_context,
            near=evidence.near,
            far=evidence.far,
        )
        evidence_view = {
            "rgb": torch.as_tensor(
                evidence.rgb[view_index],
                device=device,
                dtype=torch.float32,
            )
            / 255.0,
            "alpha": torch.as_tensor(
                evidence.alpha[view_index],
                device=device,
                dtype=torch.float32,
            )[..., None]
            / 255.0,
            "source_depth": torch.as_tensor(
                evidence.source_depth[view_index],
                device=device,
                dtype=torch.float32,
            )[..., None],
            "source_normal": torch.as_tensor(
                evidence.source_normal[view_index],
                device=device,
                dtype=torch.float32,
            ),
            "source_part_id": torch.as_tensor(
                evidence.source_part_id[view_index],
                device=device,
                dtype=torch.float32,
            )[..., None],
            "source_foreground": torch.as_tensor(
                evidence.source_part_id[view_index] >= 0,
                device=device,
                dtype=torch.float32,
            )[..., None],
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }
        for start in range(0, len(flat_interior), REBAKE_TEXEL_CHUNK_SIZE):
            stop = min(start + REBAKE_TEXEL_CHUNK_SIZE, len(flat_interior))
            _accumulate_view_observations(
                target_points_internal=target_points[start:stop],
                target_normals_internal=target_normals[start:stop],
                target_part_ids=target_parts[start:stop],
                evidence_view=evidence_view,
                target_self_depth=target_self_depth,
                self_visibility_epsilon=self_visibility_epsilon,
                source_target_depth_epsilon=source_target_depth_epsilon,
                sum_rgb=sum_rgb[start:stop],
                sum_weight=sum_weight[start:stop],
            )

    chart_face_id = np.ascontiguousarray(
        uv_target.chart_face_id.detach().cpu().numpy(),
        dtype=np.int32,
    )
    chart_part_id = np.ascontiguousarray(
        uv_target.chart_part_id.detach().cpu().numpy(),
        dtype=np.int32,
    )
    flat_indices_cpu = flat_interior.detach().cpu().numpy()
    weights_cpu = sum_weight.detach().cpu().numpy()
    rgb_sums_cpu = sum_rgb.detach().cpu().numpy()
    observed_flat = weights_cpu > 0.0
    observed_mask = np.zeros(
        (ATLAS_RESOLUTION * ATLAS_RESOLUTION,),
        dtype=bool,
    )
    observed_mask[flat_indices_cpu] = observed_flat
    observed_mask = observed_mask.reshape(ATLAS_RESOLUTION, ATLAS_RESOLUTION)
    confidence = np.zeros(
        (ATLAS_RESOLUTION * ATLAS_RESOLUTION,),
        dtype=np.float32,
    )
    confidence[flat_indices_cpu] = weights_cpu
    confidence = confidence.reshape(ATLAS_RESOLUTION, ATLAS_RESOLUTION)
    albedo = np.zeros(
        (ATLAS_RESOLUTION * ATLAS_RESOLUTION, 3),
        dtype=np.uint8,
    )
    observed_rgb = np.zeros_like(rgb_sums_cpu)
    observed_rgb[observed_flat] = (
        rgb_sums_cpu[observed_flat] / weights_cpu[observed_flat, None]
    )
    albedo[flat_indices_cpu[observed_flat]] = np.rint(
        np.clip(observed_rgb[observed_flat], 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    albedo = albedo.reshape(ATLAS_RESOLUTION, ATLAS_RESOLUTION, 3)
    torch.cuda.synchronize()
    fill_started = time.perf_counter()
    if timings_s is not None:
        timings_s["bake"] = fill_started - bake_started
    filled_albedo, provenance = _fill_unobserved_chart_texels(
        albedo=albedo,
        chart_part_id=chart_part_id,
        observed_mask=observed_mask,
        part_colors=colors,
    )
    final_albedo, final_provenance, gutter_part_id = (
        _propagate_same_part_gutter_with_owner(
            filled_albedo=filled_albedo,
            chart_part_id=chart_part_id,
            fill_provenance=provenance,
            padding_pixels=ATLAS_PADDING_PIXELS,
        )
    )
    statistics = _compute_rebake_statistics(
        chart_part_id,
        final_provenance,
        binding_inputs.declared_part_count,
        gutter_part_id=gutter_part_id,
    )
    if timings_s is not None:
        timings_s["fill"] = time.perf_counter() - fill_started
    result = RebakeResult(
        albedo_srgb_uint8=_readonly(final_albedo, dtype=np.uint8),
        chart_part_id=_readonly(chart_part_id, dtype=np.int32),
        chart_face_id=_readonly(chart_face_id, dtype=np.int32),
        observed_mask=_readonly(observed_mask, dtype=bool),
        confidence=_readonly(confidence, dtype=np.float32),
        fill_provenance=_readonly(final_provenance, dtype=np.uint8),
        gutter_part_id=_readonly(gutter_part_id, dtype=np.int32),
        statistics=statistics,
    )
    _validate_rebake_result(result)
    return result


def _repo_relative_output_path(path: Path, *, repo_root: Path | None = None) -> str:
    root = (repo_root or _repo_root()).resolve()
    resolved = Path(path).expanduser().resolve()
    outputs_root = (root / "outputs").resolve()
    try:
        relative_to_outputs = resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"path must be inside the HAG4R outputs directory: {resolved}") from exc
    return (Path("outputs") / relative_to_outputs).as_posix()


def build_post_mesh_texture_request(
    *,
    appearance_manifest_path: Path,
    monolithic_mesh_path: Path,
    heterogeneous_params_path: Path,
    metric_mesh_scaling_path: Path,
    inferred_material_path: Path,
    output_dir: Path,
    repo_root: Path | None = None,
) -> PostMeshTextureRequest:
    root = (repo_root or _repo_root()).resolve()
    request = PostMeshTextureRequest(
        appearance_manifest_path=Path(appearance_manifest_path).expanduser().resolve(),
        monolithic_mesh_path=Path(monolithic_mesh_path).expanduser().resolve(),
        heterogeneous_params_path=Path(heterogeneous_params_path).expanduser().resolve(),
        metric_mesh_scaling_path=Path(metric_mesh_scaling_path).expanduser().resolve(),
        inferred_material_path=Path(inferred_material_path).expanduser().resolve(),
        output_dir=Path(output_dir).expanduser().resolve(),
    )
    for field_name in (
        "appearance_manifest_path",
        "monolithic_mesh_path",
        "heterogeneous_params_path",
        "metric_mesh_scaling_path",
        "inferred_material_path",
    ):
        value = getattr(request, field_name)
        if not value.is_file():
            raise FileNotFoundError(f"post-mesh texture input is missing: {field_name}={value}")
        _repo_relative_output_path(value, repo_root=root)
    _repo_relative_output_path(request.output_dir, repo_root=root)
    if request.output_dir == (root / "outputs").resolve():
        raise ValueError("post-mesh texture output_dir must not be the outputs root")
    return request


def _request_payload(
    request: PostMeshTextureRequest,
    *,
    repo_root: Path | None = None,
) -> dict[str, str]:
    root = repo_root or _repo_root()
    return {
        "schema_version": POST_MESH_TEXTURE_REQUEST_SCHEMA_VERSION,
        "appearance_manifest_path": _repo_relative_output_path(
            request.appearance_manifest_path, repo_root=root
        ),
        "monolithic_mesh_path": _repo_relative_output_path(
            request.monolithic_mesh_path, repo_root=root
        ),
        "heterogeneous_params_path": _repo_relative_output_path(
            request.heterogeneous_params_path, repo_root=root
        ),
        "metric_mesh_scaling_path": _repo_relative_output_path(
            request.metric_mesh_scaling_path, repo_root=root
        ),
        "inferred_material_path": _repo_relative_output_path(
            request.inferred_material_path, repo_root=root
        ),
        "output_dir": _repo_relative_output_path(request.output_dir, repo_root=root),
    }


def write_post_mesh_texture_request(
    request: PostMeshTextureRequest,
    path: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_request_payload(request, repo_root=repo_root), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _file_manifest_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _json_statistics(statistics: RebakeStatistics) -> dict[str, Any]:
    return {
        "global": dict(statistics.global_statistics),
        "per_part": [dict(record) for record in statistics.per_part_statistics],
    }


def _write_qa_images(staging_dir: Path, rebake: RebakeResult) -> tuple[Path, int, float]:
    from PIL import Image

    qa_dir = staging_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rebake.albedo_srgb_uint8, mode="RGB").save(staging_dir / "albedo.png")

    coverage = np.zeros((*rebake.observed_mask.shape, 3), dtype=np.uint8)
    coverage[rebake.observed_mask] = 255
    Image.fromarray(coverage, mode="RGB").save(qa_dir / "observed_coverage.png")

    palette = np.asarray(
        [
            [0, 0, 0],
            [0, 200, 0],
            [0, 96, 255],
            [255, 160, 0],
            [220, 0, 220],
        ],
        dtype=np.uint8,
    )
    if np.any(rebake.fill_provenance >= len(palette)):
        raise ValueError("fill provenance contains a value outside the frozen QA palette")
    provenance = palette[rebake.fill_provenance]
    Image.fromarray(provenance, mode="RGB").save(qa_dir / "fill_provenance.png")

    max_confidence = float(np.max(rebake.confidence))
    if max_confidence > 0.0:
        confidence_u8 = np.rint(
            255.0 * np.clip(rebake.confidence / max_confidence, 0.0, 1.0)
        ).astype(np.uint8)
    else:
        confidence_u8 = np.zeros(rebake.confidence.shape, dtype=np.uint8)
    confidence_rgb = np.repeat(confidence_u8[..., None], 3, axis=2)
    Image.fromarray(confidence_rgb, mode="RGB").save(qa_dir / "confidence.png")
    return qa_dir, int(np.count_nonzero(rebake.observed_mask)), max_confidence


def _render_rest_pose_preview(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    albedo: np.ndarray,
    output_path: Path,
) -> int:
    from PIL import Image

    image_size = 512
    view_direction = np.asarray([1.0, -1.0, 1.0], dtype=np.float64)
    view_direction /= np.linalg.norm(view_direction)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(up_hint, view_direction))) > 0.95:
        up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    screen_x = np.cross(up_hint, view_direction)
    screen_x /= np.linalg.norm(screen_x)
    screen_y = np.cross(view_direction, screen_x)
    centered = np.asarray(vertices, dtype=np.float64) - np.mean(vertices, axis=0)
    projected = np.stack(
        (centered @ screen_x, centered @ screen_y, centered @ view_direction),
        axis=1,
    )
    extent = np.ptp(projected[:, :2], axis=0)
    scale = float(max(extent))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("rest-pose preview requires non-degenerate projected bounds")
    xy = (projected[:, :2] / (scale * 1.10) + 0.5) * (image_size - 1)
    depth = projected[:, 2]
    image = np.full((image_size, image_size, 3), 32, dtype=np.uint8)
    z_buffer = np.full((image_size, image_size), -np.inf, dtype=np.float64)
    degenerate_count = 0
    texture_height, texture_width = albedo.shape[:2]
    for face in np.asarray(faces, dtype=np.int64):
        triangle = xy[face]
        signed_area = float(
            (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1]) * (triangle[2, 0] - triangle[0, 0])
        )
        if abs(signed_area) <= 1.0e-12:
            degenerate_count += 1
            continue
        min_x = max(0, int(math.floor(float(np.min(triangle[:, 0])))))
        max_x = min(image_size - 1, int(math.ceil(float(np.max(triangle[:, 0])))))
        min_y = max(0, int(math.floor(float(np.min(triangle[:, 1])))))
        max_y = min(image_size - 1, int(math.ceil(float(np.max(triangle[:, 1])))))
        if min_x > max_x or min_y > max_y:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
            np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
        )
        p0, p1, p2 = triangle
        denominator = (
            (p1[1] - p2[1]) * (p0[0] - p2[0])
            + (p2[0] - p1[0]) * (p0[1] - p2[1])
        )
        w0 = (
            (p1[1] - p2[1]) * (grid_x - p2[0])
            + (p2[0] - p1[0]) * (grid_y - p2[1])
        ) / denominator
        w1 = (
            (p2[1] - p0[1]) * (grid_x - p2[0])
            + (p0[0] - p2[0]) * (grid_y - p2[1])
        ) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1.0e-9) & (w1 >= -1.0e-9) & (w2 >= -1.0e-9)
        if not np.any(inside):
            continue
        interpolated_depth = w0 * depth[face[0]] + w1 * depth[face[1]] + w2 * depth[face[2]]
        local_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        visible = inside & (interpolated_depth > local_z)
        if not np.any(visible):
            continue
        interpolated_uv = (
            w0[..., None] * uv[face[0]]
            + w1[..., None] * uv[face[1]]
            + w2[..., None] * uv[face[2]]
        )
        texture_x = np.clip(
            np.rint(interpolated_uv[..., 0] * (texture_width - 1)),
            0,
            texture_width - 1,
        ).astype(np.int64)
        texture_y = np.clip(
            np.rint((1.0 - interpolated_uv[..., 1]) * (texture_height - 1)),
            0,
            texture_height - 1,
        ).astype(np.int64)
        local_image = image[min_y : max_y + 1, min_x : max_x + 1]
        local_image[visible] = albedo[texture_y[visible], texture_x[visible]]
        local_z[visible] = interpolated_depth[visible]
    Image.fromarray(image, mode="RGB").save(output_path)
    return degenerate_count


def _glb_json_document(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("visual mesh is not a binary GLB container")
    version = int.from_bytes(data[4:8], "little")
    declared_length = int.from_bytes(data[8:12], "little")
    if version != 2 or declared_length != len(data):
        raise ValueError("visual mesh GLB header is invalid")
    cursor = 12
    json_document: dict[str, Any] | None = None
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("visual mesh GLB chunk header is truncated")
        chunk_length = int.from_bytes(data[cursor : cursor + 4], "little")
        chunk_type = data[cursor + 4 : cursor + 8]
        cursor += 8
        chunk = data[cursor : cursor + chunk_length]
        if len(chunk) != chunk_length:
            raise ValueError("visual mesh GLB chunk is truncated")
        cursor += chunk_length
        if chunk_type == b"JSON":
            if json_document is not None:
                raise ValueError("visual mesh GLB contains multiple JSON chunks")
            parsed = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("visual mesh GLB JSON chunk must contain an object")
            json_document = parsed
    if cursor != len(data) or json_document is None:
        raise ValueError("visual mesh GLB does not contain one valid JSON chunk")
    return json_document


def validate_visual_mesh_glb(
    *,
    visual_mesh_path: Path,
    binding: ExactVisualBinding,
    binding_inputs: ExactBindingInputs,
) -> dict[str, int | float]:
    import trimesh

    document = _glb_json_document(Path(visual_mesh_path))
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise ValueError("visual GLB must contain exactly one mesh")
    primitives = meshes[0].get("primitives") if isinstance(meshes[0], dict) else None
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise ValueError("visual GLB must contain exactly one primitive")
    primitive = primitives[0]
    material_index = primitive.get("material") if isinstance(primitive, dict) else None
    materials = document.get("materials")
    if (
        isinstance(material_index, bool)
        or not isinstance(material_index, int)
        or not isinstance(materials, list)
        or material_index < 0
        or material_index >= len(materials)
    ):
        raise ValueError("visual GLB primitive must reference one material")
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness") if isinstance(material, dict) else None
    if (
        not isinstance(pbr, dict)
        or "baseColorTexture" not in pbr
        or float(pbr.get("metallicFactor", -1.0)) != 0.0
        or float(pbr.get("roughnessFactor", -1.0)) != 1.0
        or material.get("doubleSided") is not True
    ):
        raise ValueError("visual GLB PBR material does not match the frozen contract")
    textures = document.get("textures")
    images = document.get("images")
    texture_index = pbr["baseColorTexture"].get("index")
    if (
        isinstance(texture_index, bool)
        or not isinstance(texture_index, int)
        or not isinstance(textures, list)
        or texture_index < 0
        or texture_index >= len(textures)
    ):
        raise ValueError("visual GLB base-color texture reference is invalid")
    image_index = textures[texture_index].get("source")
    if (
        isinstance(image_index, bool)
        or not isinstance(image_index, int)
        or not isinstance(images, list)
        or image_index < 0
        or image_index >= len(images)
    ):
        raise ValueError("visual GLB embedded image reference is invalid")
    image_record = images[image_index]
    if not isinstance(image_record, dict) or (
        "bufferView" not in image_record
        and not str(image_record.get("uri", "")).startswith("data:")
    ):
        raise ValueError("visual GLB base-color image must be embedded")
    if "uri" in image_record and not str(image_record["uri"]).startswith("data:"):
        raise ValueError("visual GLB must not reference an external image")

    loaded = trimesh.load(str(visual_mesh_path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("visual GLB did not reload as one Trimesh")
    loaded_vertices = np.asarray(loaded.vertices, dtype=np.float64)
    loaded_faces = np.asarray(loaded.faces, dtype=np.int64)
    loaded_uv = np.asarray(getattr(loaded.visual, "uv", None), dtype=np.float64)
    if loaded_vertices.shape != binding.visual_rest_vertices_m.shape or not np.allclose(
        loaded_vertices,
        binding.visual_rest_vertices_m,
        atol=REST_ALIGNMENT_MAX_ERROR_M,
        rtol=0.0,
    ):
        raise ValueError("visual GLB vertex buffer was reordered or changed")
    if loaded_faces.shape != binding.visual_faces.shape or not np.array_equal(
        loaded_faces, binding.visual_faces
    ):
        raise ValueError("visual GLB face index buffer was reordered or changed")
    if loaded_uv.shape != binding.visual_uv.shape or not np.allclose(
        loaded_uv, binding.visual_uv, atol=1.0e-7, rtol=0.0
    ):
        raise ValueError("visual GLB UV buffer was reordered or changed")
    image = getattr(loaded.visual.material, "baseColorTexture", None)
    if image is None or tuple(image.size) != (ATLAS_RESOLUTION, ATLAS_RESOLUTION):
        raise ValueError("visual GLB base-color texture has the wrong resolution")
    reloaded_binding = ExactVisualBinding(
        visual_rest_vertices_m=_readonly(loaded_vertices, dtype=np.float32),
        visual_faces=_readonly(loaded_faces, dtype=np.int32),
        visual_uv=_readonly(loaded_uv, dtype=np.float32),
        surface_vertex_to_physics_vertex=binding.surface_vertex_to_physics_vertex,
        visual_vertex_to_surface_vertex=binding.visual_vertex_to_surface_vertex,
        physics_vertex_indices=binding.physics_vertex_indices,
        boundary_face_tet_indices=binding.boundary_face_tet_indices,
        boundary_face_part_ids=binding.boundary_face_part_ids,
    )
    metrics = validate_exact_visual_binding(
        binding=reloaded_binding,
        physics_vertices_m=binding_inputs.physics_vertices_m,
        tets=binding_inputs.tets,
        tet_part_labels=binding_inputs.tet_part_labels,
        declared_part_count=binding_inputs.declared_part_count,
    )
    return {
        "primitive_count": 1,
        "vertex_count": int(len(loaded_vertices)),
        "face_count": int(len(loaded_faces)),
        "uv_count": int(len(loaded_uv)),
        "texture_resolution": ATLAS_RESOLUTION,
        "max_rest_alignment_error_m": float(metrics["max_rest_alignment_error_m"]),
    }


def _write_visual_mesh_glb(
    *,
    output_path: Path,
    albedo_path: Path,
    binding: ExactVisualBinding,
    binding_inputs: ExactBindingInputs,
) -> dict[str, int | float]:
    import trimesh
    from PIL import Image

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.open(albedo_path).convert("RGB"),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    mesh = trimesh.Trimesh(
        vertices=binding.visual_rest_vertices_m,
        faces=binding.visual_faces,
        process=False,
        validate=False,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=binding.visual_uv,
        material=material,
    )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="visual_mesh", node_name="visual_mesh")
    output_path.write_bytes(scene.export(file_type="glb"))
    return validate_visual_mesh_glb(
        visual_mesh_path=output_path,
        binding=binding,
        binding_inputs=binding_inputs,
    )


def _validate_request_payload(
    payload: Any,
    *,
    expected_request: PostMeshTextureRequest | None = None,
    repo_root: Path | None = None,
) -> None:
    expected_keys = {
        "schema_version",
        "appearance_manifest_path",
        "monolithic_mesh_path",
        "heterogeneous_params_path",
        "metric_mesh_scaling_path",
        "inferred_material_path",
        "output_dir",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("post-mesh texture request fields do not match the frozen schema")
    if payload.get("schema_version") != POST_MESH_TEXTURE_REQUEST_SCHEMA_VERSION:
        raise ValueError("post-mesh texture request schema_version does not match")
    for key in expected_keys - {"schema_version"}:
        value = payload.get(key)
        relative = Path(value) if isinstance(value, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "outputs"
            or ".." in relative.parts
        ):
            raise ValueError(f"post-mesh texture request path must be repo-relative outputs/: {key}")
    if expected_request is not None and payload != _request_payload(
        expected_request, repo_root=repo_root
    ):
        raise ValueError("post-mesh texture request does not match the expected request")


def validate_post_mesh_texture_bundle(
    output_dir: Path,
    *,
    expected_request: PostMeshTextureRequest | None = None,
    validate_glb: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    bundle_dir = Path(output_dir)
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise NotADirectoryError(f"post-mesh texture bundle is not a regular directory: {bundle_dir}")
    inventory = frozenset(
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    )
    if inventory != POST_MESH_TEXTURE_BUNDLE_INVENTORY:
        raise ValueError(
            "post-mesh texture bundle inventory does not match: "
            f"missing={sorted(POST_MESH_TEXTURE_BUNDLE_INVENTORY - inventory)}, "
            f"extra={sorted(inventory - POST_MESH_TEXTURE_BUNDLE_INVENTORY)}"
        )
    if any(path.is_symlink() for path in bundle_dir.rglob("*")):
        raise ValueError("post-mesh texture bundle must not contain symlinks")
    request_payload = json.loads(
        (bundle_dir / "post_mesh_texture_request.json").read_text(encoding="utf-8")
    )
    _validate_request_payload(
        request_payload,
        expected_request=expected_request,
        repo_root=repo_root,
    )
    manifest = json.loads((bundle_dir / "visual_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        POST_MESH_TEXTURE_VISUAL_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("visual manifest schema_version does not match")
    qa_report = json.loads((bundle_dir / "qa/report.json").read_text(encoding="utf-8"))
    if not isinstance(qa_report, dict) or qa_report.get("schema_version") != (
        POST_MESH_TEXTURE_QA_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("texture QA report schema_version does not match")
    if any(key in qa_report for key in ("pass", "degraded", "export_gate", "approved")):
        raise ValueError("texture QA report must not define a quality gate")
    expected_bundle_records = {
        relative
        for relative in POST_MESH_TEXTURE_BUNDLE_INVENTORY
        if relative != "visual_manifest.json"
    }
    bundle_records = manifest.get("bundle_files")
    if not isinstance(bundle_records, dict) or set(bundle_records) != expected_bundle_records:
        raise ValueError("visual manifest bundle file records do not match exact inventory")
    for relative, record in bundle_records.items():
        path = bundle_dir / relative
        expected_record = _file_manifest_record(path, relative_to=bundle_dir)
        if record != expected_record:
            raise ValueError(f"visual manifest file record does not match: {relative}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "appearance_manifest_path",
        "monolithic_mesh_path",
        "heterogeneous_params_path",
        "metric_mesh_scaling_path",
        "inferred_material_path",
    }:
        raise ValueError("visual manifest input records do not match")
    root = (repo_root or _repo_root()).resolve()
    for key, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"visual manifest input record is malformed: {key}")
        relative = Path(str(record["path"]))
        path = root / relative
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "outputs"
            or ".." in relative.parts
            or not path.is_file()
            or record["size_bytes"] != path.stat().st_size
            or record["sha256"] != _sha256_file(path)
        ):
            raise ValueError(f"visual manifest input record does not match: {key}")
    binding_inputs = load_exact_binding_inputs(
        monolithic_mesh_path=root / inputs["monolithic_mesh_path"]["path"],
        heterogeneous_params_path=root / inputs["heterogeneous_params_path"]["path"],
        appearance_manifest_path=root / inputs["appearance_manifest_path"]["path"],
    )
    binding = load_visual_to_physics_npz(
        input_path=bundle_dir / "visual_to_physics.npz",
        physics_vertices_m=binding_inputs.physics_vertices_m,
        tets=binding_inputs.tets,
        tet_part_labels=binding_inputs.tet_part_labels,
        declared_part_count=binding_inputs.declared_part_count,
    )
    if validate_glb:
        validate_visual_mesh_glb(
            visual_mesh_path=bundle_dir / "visual_mesh.glb",
            binding=binding,
            binding_inputs=binding_inputs,
        )
    return manifest


def execute_post_mesh_texture(
    request: PostMeshTextureRequest,
    *,
    repo_root: Path | None = None,
) -> PostMeshTextureBundleResult:
    root = (repo_root or _repo_root()).resolve()
    request = build_post_mesh_texture_request(
        appearance_manifest_path=request.appearance_manifest_path,
        monolithic_mesh_path=request.monolithic_mesh_path,
        heterogeneous_params_path=request.heterogeneous_params_path,
        metric_mesh_scaling_path=request.metric_mesh_scaling_path,
        inferred_material_path=request.inferred_material_path,
        output_dir=request.output_dir,
        repo_root=root,
    )
    output_dir = request.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup.{uuid.uuid4().hex}")
    timings: dict[str, float] = {}
    try:
        write_post_mesh_texture_request(
            request,
            staging_dir / "post_mesh_texture_request.json",
            repo_root=root,
        )
        started = time.perf_counter()
        binding_inputs = load_exact_binding_inputs(
            monolithic_mesh_path=request.monolithic_mesh_path,
            heterogeneous_params_path=request.heterogeneous_params_path,
            appearance_manifest_path=request.appearance_manifest_path,
        )
        boundary = extract_exact_tet_boundary(
            physics_vertices_m=binding_inputs.physics_vertices_m,
            tets=binding_inputs.tets,
            tet_part_labels=binding_inputs.tet_part_labels,
            declared_part_count=binding_inputs.declared_part_count,
        )
        timings["boundary"] = time.perf_counter() - started

        started = time.perf_counter()
        binding = unwrap_exact_tet_boundary(
            boundary=boundary,
            physics_vertices_m=binding_inputs.physics_vertices_m,
            tets=binding_inputs.tets,
            tet_part_labels=binding_inputs.tet_part_labels,
            declared_part_count=binding_inputs.declared_part_count,
        )
        binding_metrics = validate_exact_visual_binding(
            binding=binding,
            physics_vertices_m=binding_inputs.physics_vertices_m,
            tets=binding_inputs.tets,
            tet_part_labels=binding_inputs.tet_part_labels,
            declared_part_count=binding_inputs.declared_part_count,
        )
        timings["unwrap"] = time.perf_counter() - started

        evidence = load_appearance_rebake_evidence(request.appearance_manifest_path)
        part_colors = load_rebake_part_colors(
            request.heterogeneous_params_path,
            binding_inputs.declared_part_count,
        )
        rebake_timings: dict[str, float] = {}
        rebake = rebake_visibility_aware_atlas(
            binding=binding,
            binding_inputs=binding_inputs,
            evidence=evidence,
            metric_mesh_scaling_path=request.metric_mesh_scaling_path,
            part_colors=part_colors,
            timings_s=rebake_timings,
        )
        timings["bake"] = rebake_timings.get("bake", 0.0)
        timings["fill"] = rebake_timings.get("fill", 0.0)
        rebake = _orient_rebake_for_pil_gltf_serialization(rebake)

        qa_dir, observed_texel_count, max_confidence = _write_qa_images(staging_dir, rebake)
        write_visual_to_physics_npz(
            output_path=staging_dir / "visual_to_physics.npz",
            binding=binding,
            physics_vertices_m=binding_inputs.physics_vertices_m,
            tets=binding_inputs.tets,
            tet_part_labels=binding_inputs.tet_part_labels,
            declared_part_count=binding_inputs.declared_part_count,
        )
        degenerate_preview_faces = _render_rest_pose_preview(
            vertices=binding.visual_rest_vertices_m,
            faces=binding.visual_faces,
            uv=binding.visual_uv,
            albedo=rebake.albedo_srgb_uint8,
            output_path=qa_dir / "rest_pose_render.png",
        )
        started = time.perf_counter()
        glb_metrics = _write_visual_mesh_glb(
            output_path=staging_dir / "visual_mesh.glb",
            albedo_path=staging_dir / "albedo.png",
            binding=binding,
            binding_inputs=binding_inputs,
        )
        timings["glb_export"] = time.perf_counter() - started

        statistics = _json_statistics(rebake.statistics)
        qa_report = {
            "schema_version": POST_MESH_TEXTURE_QA_REPORT_SCHEMA_VERSION,
            "statistics": statistics,
            "observed_texel_count": observed_texel_count,
            "max_confidence": max_confidence,
            "degenerate_projected_triangle_count": degenerate_preview_faces,
            "images": {
                "observed_coverage": "observed_coverage.png",
                "fill_provenance": "fill_provenance.png",
                "confidence": "confidence.png",
                "rest_pose_render": "rest_pose_render.png",
            },
            "fill_provenance_palette_rgb": {
                "unmapped": [0, 0, 0],
                "observed": [0, 200, 0],
                "nearest": [0, 96, 255],
                "part_color": [255, 160, 0],
                "gutter": [220, 0, 220],
            },
            "confidence_normalization": "round(255*clip(confidence/max_confidence,0,1)); all black when max_confidence=0",
        }
        (qa_dir / "report.json").write_text(
            json.dumps(qa_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        input_paths = {
            "appearance_manifest_path": request.appearance_manifest_path,
            "monolithic_mesh_path": request.monolithic_mesh_path,
            "heterogeneous_params_path": request.heterogeneous_params_path,
            "metric_mesh_scaling_path": request.metric_mesh_scaling_path,
            "inferred_material_path": request.inferred_material_path,
        }
        bundle_records = {
            relative: _file_manifest_record(staging_dir / relative, relative_to=staging_dir)
            for relative in sorted(
                POST_MESH_TEXTURE_BUNDLE_INVENTORY - {"visual_manifest.json"}
            )
        }
        manifest = {
            "schema_version": POST_MESH_TEXTURE_VISUAL_MANIFEST_SCHEMA_VERSION,
            "request": {
                "schema_version": POST_MESH_TEXTURE_REQUEST_SCHEMA_VERSION,
                "path": "post_mesh_texture_request.json",
            },
            "inputs": {
                key: {
                    "path": _repo_relative_output_path(path, repo_root=root),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for key, path in input_paths.items()
            },
            "bundle_files": bundle_records,
            "exact_binding": exact_binding_manifest_fragment(
                boundary=boundary,
                binding=binding,
                validation_summary=binding_metrics,
            ),
            "statistics": statistics,
            "timings_s": timings,
            "glb": glb_metrics,
            "pbr": {
                "base_color_texture": "albedo.png",
                "metallic": 0.0,
                "roughness": 1.0,
                "double_sided": True,
            },
            "binding_validation": binding_metrics,
        }
        (staging_dir / "visual_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_post_mesh_texture_bundle(
            staging_dir,
            expected_request=request,
            validate_glb=True,
            repo_root=root,
        )

        if output_dir.exists():
            if not output_dir.is_dir() or output_dir.is_symlink():
                raise NotADirectoryError(
                    f"post-mesh texture output exists but is not a regular directory: {output_dir}"
                )
            output_dir.rename(backup_dir)
        try:
            staging_dir.rename(output_dir)
        except BaseException:
            if backup_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return PostMeshTextureBundleResult(
            request_path=output_dir / "post_mesh_texture_request.json",
            visual_mesh_path=output_dir / "visual_mesh.glb",
            albedo_path=output_dir / "albedo.png",
            binding_path=output_dir / "visual_to_physics.npz",
            visual_manifest_path=output_dir / "visual_manifest.json",
            texture_qa_report_path=output_dir / "qa/report.json",
            timings_s=dict(timings),
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_dir.exists():
            if not output_dir.exists():
                backup_dir.rename(output_dir)
            else:
                shutil.rmtree(backup_dir)


def run_post_mesh_texture(
    *,
    appearance_manifest_path: Path,
    monolithic_mesh_path: Path,
    heterogeneous_params_path: Path,
    metric_mesh_scaling_path: Path,
    inferred_material_path: Path,
    output_dir: Path,
    env: dict[str, str] | None = None,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
) -> StageRunResult:
    root = (repo_root or _repo_root()).resolve()
    request = build_post_mesh_texture_request(
        appearance_manifest_path=appearance_manifest_path,
        monolithic_mesh_path=monolithic_mesh_path,
        heterogeneous_params_path=heterogeneous_params_path,
        metric_mesh_scaling_path=metric_mesh_scaling_path,
        inferred_material_path=inferred_material_path,
        output_dir=output_dir,
        repo_root=root,
    )
    args = (
        "--appearance_manifest_path",
        request.appearance_manifest_path,
        "--monolithic_mesh_path",
        request.monolithic_mesh_path,
        "--heterogeneous_params_path",
        request.heterogeneous_params_path,
        "--metric_mesh_scaling_path",
        request.metric_mesh_scaling_path,
        "--inferred_material_path",
        request.inferred_material_path,
        "--output_dir",
        request.output_dir,
    )
    artifacts = (
        _artifact(
            ArtifactRole.TEXTURED_VISUAL_MESH,
            request.output_dir / "visual_mesh.glb",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.VISUAL_TO_PHYSICS_BINDING,
            request.output_dir / "visual_to_physics.npz",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.VISUAL_MANIFEST,
            request.output_dir / "visual_manifest.json",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.TEXTURE_QA_REPORT,
            request.output_dir / "qa/report.json",
            Stage.POST_MESH_TEXTURE,
        ),
    )
    result = _run_subprocess_stage(
        name="hag4r_post_mesh_texture",
        stage=Stage.POST_MESH_TEXTURE,
        argv=_conda_module_argv(
            EnvName.OMNIPART,
            "hag4r.tools.post_mesh_texture",
            args,
            repo_root=root,
        ),
        cwd=root,
        expected_artifacts=artifacts,
        read_paths=(
            request.appearance_manifest_path,
            request.monolithic_mesh_path,
            request.heterogeneous_params_path,
            request.metric_mesh_scaling_path,
            request.inferred_material_path,
        ),
        write_paths=(request.output_dir,),
        conda_env=resolve_conda_env(EnvName.OMNIPART, repo_root=root),
        env=env,
        log_dir=log_dir,
        repo_root=root,
    )
    if result.success:
        validate_post_mesh_texture_bundle(
            request.output_dir,
            expected_request=request,
            validate_glb=True,
            repo_root=root,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one exact post-mesh texture bundle.")
    parser.add_argument("--appearance_manifest_path", required=True)
    parser.add_argument("--monolithic_mesh_path", required=True)
    parser.add_argument("--heterogeneous_params_path", required=True)
    parser.add_argument("--metric_mesh_scaling_path", required=True)
    parser.add_argument("--inferred_material_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = build_post_mesh_texture_request(
        appearance_manifest_path=Path(args.appearance_manifest_path),
        monolithic_mesh_path=Path(args.monolithic_mesh_path),
        heterogeneous_params_path=Path(args.heterogeneous_params_path),
        metric_mesh_scaling_path=Path(args.metric_mesh_scaling_path),
        inferred_material_path=Path(args.inferred_material_path),
        output_dir=Path(args.output_dir),
    )
    result = execute_post_mesh_texture(request)
    print(
        json.dumps(
            {
                "status": "success",
                "visual_manifest_path": str(result.visual_manifest_path),
                "timings_s": result.timings_s,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
