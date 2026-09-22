from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nvdiffrast.torch as dr
import numpy as np
import torch
from PIL import Image

from .renderers.gaussian_render import GaussianRenderer
from .renderers.mesh_renderer import intrinsics_to_projection
from .representations.gaussian.gaussian_model import Gaussian
from .utils.random_utils import sphere_hammersley_sequence
from .utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


APPEARANCE_SCHEMA_VERSION = "hag4r-omnipart-appearance-v1"
BUNDLE_VALIDATION_SCHEMA_VERSION = "hag4r-omnipart-appearance-validation-v1"
VIEW_COUNT = 64
RESOLUTION = 512
FOV_DEGREES = 40.0
CAMERA_RADIUS = 2.0
CAPTURE_DEPTH_MARGIN_ABSOLUTE = 0.1
CAPTURE_DEPTH_MARGIN_FRACTION = 0.1
ALPHA_THRESHOLD = 0.05
INTERNAL_Z_UP_TO_GLTF_Y_UP = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_APPEARANCE_VIEW_SPECS = {
    "rgb": (np.dtype("uint8"), (VIEW_COUNT, RESOLUTION, RESOLUTION, 3)),
    "alpha": (np.dtype("uint8"), (VIEW_COUNT, RESOLUTION, RESOLUTION)),
    "source_depth": (np.dtype("float32"), (VIEW_COUNT, RESOLUTION, RESOLUTION)),
    "source_normal": (np.dtype("float16"), (VIEW_COUNT, RESOLUTION, RESOLUTION, 3)),
    "source_part_id": (np.dtype("int16"), (VIEW_COUNT, RESOLUTION, RESOLUTION)),
    "extrinsics": (np.dtype("float32"), (VIEW_COUNT, 4, 4)),
    "intrinsics": (np.dtype("float32"), (VIEW_COUNT, 3, 3)),
}
_SOURCE_SURFACE_SPECS = {
    "vertices": np.dtype("float32"),
    "faces": np.dtype("int32"),
    "face_part_id": np.dtype("int32"),
}
_MANAGED_RELATIVE_PATHS = (
    "appearance_manifest.json",
    "appearance_views.npz",
    "source_surface.npz",
    "qa/appearance_contact_sheet.png",
    "qa/bundle_validation.json",
)


@dataclass(frozen=True)
class AppearanceSourcePart:
    part_index: int
    source_output_index: int
    source_glb_name: str
    vertices: np.ndarray
    faces: np.ndarray


def _unpremultiply_to_straight_srgb(
    premultiplied_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    premultiplied_rgb = np.asarray(premultiplied_rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if premultiplied_rgb.ndim < 1 or premultiplied_rgb.shape[-1] != 3:
        raise ValueError("premultiplied_rgb must end in three RGB channels")
    if alpha.shape != premultiplied_rgb.shape[:-1]:
        raise ValueError("alpha shape must match the RGB image dimensions")
    if not np.isfinite(premultiplied_rgb).all() or not np.isfinite(alpha).all():
        raise ValueError("RGB and alpha must be finite")

    straight_rgb = np.zeros_like(premultiplied_rgb, dtype=np.float32)
    foreground = alpha >= ALPHA_THRESHOLD
    straight_rgb[foreground] = (
        premultiplied_rgb[foreground] / alpha[foreground, None]
    )
    return np.clip(straight_rgb, 0.0, 1.0)


def _validate_source_part(part: AppearanceSourcePart, expected_index: int) -> None:
    if not isinstance(part, AppearanceSourcePart):
        raise TypeError("source_parts must contain AppearanceSourcePart instances")
    if part.part_index != expected_index:
        raise ValueError("part_index must be continuous and match source part ordering")
    if not isinstance(part.source_output_index, int) or isinstance(
        part.source_output_index, bool
    ):
        raise TypeError("source_output_index must be an integer")
    if part.source_output_index <= 0:
        raise ValueError("source_output_index must refer to a non-overall output")
    if (
        not isinstance(part.source_glb_name, str)
        or not part.source_glb_name
        or Path(part.source_glb_name).name != part.source_glb_name
        or Path(part.source_glb_name).suffix.lower() != ".glb"
    ):
        raise ValueError("source_glb_name must be a GLB basename")
    if not isinstance(part.vertices, np.ndarray) or part.vertices.dtype != np.float32:
        raise TypeError("source vertices must be a float32 numpy array")
    if not isinstance(part.faces, np.ndarray) or part.faces.dtype != np.int32:
        raise TypeError("source faces must be an int32 numpy array")
    if (
        part.vertices.ndim != 2
        or part.vertices.shape[1] != 3
        or part.vertices.shape[0] == 0
    ):
        raise ValueError("source vertices must have shape [N, 3] with N > 0")
    if part.faces.ndim != 2 or part.faces.shape[1] != 3 or part.faces.shape[0] == 0:
        raise ValueError("source faces must have shape [F, 3] with F > 0")
    if not np.isfinite(part.vertices).all():
        raise ValueError("source vertices must be finite")
    if part.faces.min() < 0 or part.faces.max() >= part.vertices.shape[0]:
        raise ValueError("source face indices are out of range")
    triangles = part.vertices[part.faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(doubled_area == 0):
        raise ValueError("source faces must have nonzero area")


def _combine_source_surface(
    source_parts: Sequence[AppearanceSourcePart],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_parts = tuple(source_parts)
    if not source_parts:
        raise ValueError("source_parts must contain at least one successful part")

    vertices = []
    faces = []
    face_part_id = []
    vertex_offset = 0
    source_output_indices = set()
    for expected_index, part in enumerate(source_parts):
        _validate_source_part(part, expected_index)
        if part.source_output_index in source_output_indices:
            raise ValueError("source_output_index values must be unique")
        source_output_indices.add(part.source_output_index)
        vertices.append(np.ascontiguousarray(part.vertices))
        faces.append(np.ascontiguousarray(part.faces + vertex_offset, dtype=np.int32))
        face_part_id.append(
            np.full(part.faces.shape[0], part.part_index, dtype=np.int32)
        )
        vertex_offset += part.vertices.shape[0]

    combined_vertices = np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32)
    combined_faces = np.ascontiguousarray(np.concatenate(faces), dtype=np.int32)
    combined_face_part_id = np.ascontiguousarray(
        np.concatenate(face_part_id), dtype=np.int32
    )
    if sorted(np.unique(combined_face_part_id).tolist()) != list(
        range(len(source_parts))
    ):
        raise ValueError("combined face_part_id does not cover the source part set")
    return combined_vertices, combined_faces, combined_face_part_id


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _array_metadata(array: np.ndarray) -> dict:
    return {"dtype": array.dtype.name, "shape": list(array.shape)}


def _npz_file_record(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "arrays": {
            name: _array_metadata(array) for name, array in sorted(arrays.items())
        },
    }


def _omnipart_commit() -> str:
    omnipart_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=omnipart_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("OmniPart git HEAD must be a 40-character lowercase SHA-1")
    return commit


def _generate_cameras() -> tuple[torch.Tensor, torch.Tensor]:
    cameras = [sphere_hammersley_sequence(i, VIEW_COUNT) for i in range(VIEW_COUNT)]
    yaws = [camera[0] for camera in cameras]
    pitches = [camera[1] for camera in cameras]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws,
        pitches,
        CAMERA_RADIUS,
        FOV_DEGREES,
    )
    extrinsics = torch.stack(extrinsics).to(dtype=torch.float32)
    intrinsics = torch.stack(intrinsics).to(dtype=torch.float32)
    if extrinsics.shape != (VIEW_COUNT, 4, 4):
        raise ValueError("camera extrinsics have the wrong shape")
    if intrinsics.shape != (VIEW_COUNT, 3, 3):
        raise ValueError("camera intrinsics have the wrong shape")
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("camera matrices must be finite")
    expected_last_row = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=extrinsics.device,
        dtype=extrinsics.dtype,
    )
    if not torch.equal(
        extrinsics[:, 3, :],
        expected_last_row.expand(VIEW_COUNT, -1),
    ):
        raise ValueError("camera extrinsics must use homogeneous world-to-camera form")
    return extrinsics, intrinsics


def _camera_depth_extrema(
    vertices: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[float, float]:
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] == 0
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("capture-plane source vertices must be finite [N,3]")
    if (
        extrinsics.shape != (VIEW_COUNT, 4, 4)
        or not np.isfinite(extrinsics).all()
    ):
        raise ValueError("capture-plane extrinsics must be finite [64,4,4]")
    vertices_homogeneous = np.concatenate(
        [
            vertices.astype(np.float64, copy=False),
            np.ones((len(vertices), 1), dtype=np.float64),
        ],
        axis=1,
    )
    minimum = np.inf
    maximum = -np.inf
    for extrinsic in extrinsics.astype(np.float64, copy=False):
        depth = vertices_homogeneous @ extrinsic[2]
        minimum = min(minimum, float(depth.min()))
        maximum = max(maximum, float(depth.max()))
    return minimum, maximum


def _derive_capture_planes(
    vertices: np.ndarray,
    extrinsics: torch.Tensor,
) -> tuple[float, float]:
    minimum, maximum = _camera_depth_extrema(
        vertices,
        extrinsics.detach().cpu().numpy(),
    )
    if minimum <= 0.0:
        raise ValueError("source surface crosses or lies behind a capture camera")
    margin = max(
        CAPTURE_DEPTH_MARGIN_ABSOLUTE,
        CAPTURE_DEPTH_MARGIN_FRACTION * (maximum - minimum),
    )
    near = max(minimum * 0.5, minimum - margin)
    far = maximum + margin
    if not (0.0 < near < minimum <= maximum < far):
        raise ValueError("failed to derive strict source-surface capture planes")
    return near, far


@torch.no_grad()
def _render_gaussian_views(
    merged_gaussian: Gaussian,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray]:
    renderer = GaussianRenderer(
        {
            "resolution": RESOLUTION,
            "near": near,
            "far": far,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
        }
    )
    white_override = torch.ones_like(merged_gaussian.get_xyz)
    premultiplied_views = []
    alpha_views = []
    for view_index in range(VIEW_COUNT):
        premultiplied = (
            renderer.render(
                merged_gaussian,
                extrinsics[view_index],
                intrinsics[view_index],
            )["color"]
            .permute(1, 2, 0)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        white_render = (
            renderer.render(
                merged_gaussian,
                extrinsics[view_index],
                intrinsics[view_index],
                colors_overwrite=white_override,
            )["color"]
            .permute(1, 2, 0)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        if not np.isfinite(premultiplied).all() or not np.isfinite(white_render).all():
            raise ValueError("Gaussian RGB and alpha renders must be finite")
        channel_spread = np.max(white_render, axis=-1) - np.min(
            white_render, axis=-1
        )
        if float(channel_spread.max()) > 1e-5:
            raise ValueError("white Gaussian override did not produce scalar coverage")
        premultiplied_views.append(premultiplied)
        alpha_views.append(np.mean(white_render, axis=-1, dtype=np.float32))

    premultiplied_rgb = np.ascontiguousarray(
        np.stack(premultiplied_views), dtype=np.float32
    )
    alpha = np.clip(
        np.ascontiguousarray(np.stack(alpha_views), dtype=np.float32),
        0.0,
        1.0,
    )
    straight_rgb = _unpremultiply_to_straight_srgb(premultiplied_rgb, alpha)
    rgb_uint8 = np.rint(straight_rgb * 255.0).astype(np.uint8)
    alpha_uint8 = np.rint(alpha * 255.0).astype(np.uint8)
    return rgb_uint8, alpha_uint8


@torch.no_grad()
def _render_source_gbuffers(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_part_id: np.ndarray,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = extrinsics.device
    vertices_tensor = torch.from_numpy(vertices).to(device=device)
    faces_tensor = torch.from_numpy(faces).to(device=device)
    face_part_id_tensor = torch.from_numpy(face_part_id).to(device=device)
    vertices_homogeneous = torch.cat(
        [vertices_tensor, torch.ones_like(vertices_tensor[:, :1])],
        dim=1,
    ).unsqueeze(0)
    triangles = vertices_tensor[faces_tensor.long()]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    face_normals = face_normals / torch.linalg.vector_norm(
        face_normals, dim=1, keepdim=True
    )
    if not torch.isfinite(face_normals).all():
        raise ValueError("source face normals must be finite")

    raster_context = dr.RasterizeCudaContext(device=device)
    depth_views = []
    normal_views = []
    part_id_views = []
    for view_index in range(VIEW_COUNT):
        extrinsic = extrinsics[view_index]
        projection = intrinsics_to_projection(
            intrinsics[view_index],
            near,
            far,
        )
        camera_vertices = torch.matmul(
            vertices_homogeneous,
            extrinsic.transpose(0, 1),
        )
        clip_vertices = torch.matmul(
            vertices_homogeneous,
            (projection @ extrinsic).transpose(0, 1),
        )
        raster, _ = dr.rasterize(
            raster_context,
            clip_vertices,
            faces_tensor,
            (RESOLUTION, RESOLUTION),
        )
        triangle_id = raster[0, ..., 3].long() - 1
        foreground = triangle_id >= 0
        depth = dr.interpolate(
            camera_vertices[..., 2:3].contiguous(),
            raster,
            faces_tensor,
        )[0][0, ..., 0]
        depth = torch.where(foreground, depth, torch.zeros_like(depth))
        if not torch.isfinite(depth).all() or torch.any(depth[foreground] <= 0):
            raise ValueError("source foreground depth must be finite and positive")

        normal = torch.zeros(
            (RESOLUTION, RESOLUTION, 3),
            device=device,
            dtype=torch.float32,
        )
        normal[foreground] = face_normals[triangle_id[foreground]]
        part_id = torch.full(
            (RESOLUTION, RESOLUTION),
            -1,
            device=device,
            dtype=torch.int16,
        )
        part_id[foreground] = face_part_id_tensor[
            triangle_id[foreground]
        ].to(torch.int16)
        depth_views.append(depth.cpu().numpy())
        normal_views.append(normal.cpu().numpy())
        part_id_views.append(part_id.cpu().numpy())

    return (
        np.ascontiguousarray(np.stack(depth_views), dtype=np.float32),
        np.ascontiguousarray(np.stack(normal_views), dtype=np.float32),
        np.ascontiguousarray(np.stack(part_id_views), dtype=np.int16),
    )


def _part_visibility(
    source_part_id: np.ndarray,
    part_count: int,
) -> list[dict]:
    return [
        {
            "part_index": part_index,
            "visible_pixel_count": int(
                np.count_nonzero(source_part_id == part_index)
            ),
            "visible_view_count": int(
                np.count_nonzero(
                    np.any(source_part_id == part_index, axis=(1, 2))
                )
            ),
        }
        for part_index in range(part_count)
    ]


def _qa_statistics(
    alpha: np.ndarray,
    source_part_id: np.ndarray,
    part_count: int,
) -> dict:
    alpha_float = alpha.astype(np.float32) / 255.0
    total_pixels = int(alpha.size)
    nonzero = int(np.count_nonzero(alpha))
    at_threshold = int(np.count_nonzero(alpha_float >= ALPHA_THRESHOLD))
    alpha_foreground = alpha_float >= ALPHA_THRESHOLD
    geometry_foreground = source_part_id >= 0
    overlap = int(np.count_nonzero(alpha_foreground & geometry_foreground))
    geometry_count = int(np.count_nonzero(geometry_foreground))
    return {
        "alpha_nonzero_pixel_count": nonzero,
        "alpha_nonzero_fraction": nonzero / total_pixels,
        "alpha_at_least_threshold_pixel_count": at_threshold,
        "alpha_at_least_threshold_fraction": at_threshold / total_pixels,
        "geometry_foreground_pixel_count": geometry_count,
        "alpha_geometry_overlap_pixel_count": overlap,
        "geometry_recall_against_alpha": overlap / at_threshold,
        "geometry_precision_against_alpha": overlap / geometry_count,
        "part_visibility": _part_visibility(source_part_id, part_count),
    }


def _write_contact_sheet(rgb: np.ndarray, path: Path) -> None:
    contact_sheet = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    for view_index in range(VIEW_COUNT):
        image = Image.fromarray(rgb[view_index], mode="RGB").resize(
            (128, 128),
            resample=Image.Resampling.BILINEAR,
        )
        contact_sheet.paste(
            image,
            ((view_index % 8) * 128, (view_index // 8) * 128),
        )
    contact_sheet.save(path)


def _validate_relative_file_record(record: dict, expected_path: str) -> None:
    path = record.get("path")
    if path != expected_path:
        raise ValueError(f"bundle file path must be {expected_path}")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bundle file paths must be safe appearance-relative paths")


def _validate_npz_record(
    *,
    appearance_dir: Path,
    record: dict,
    expected_path: str,
    arrays: dict[str, np.ndarray],
) -> None:
    _validate_relative_file_record(record, expected_path)
    path = appearance_dir / expected_path
    if record.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"{expected_path} size does not match its manifest record")
    if record.get("sha256") != _sha256(path):
        raise ValueError(f"{expected_path} hash does not match its manifest record")
    expected_metadata = {
        name: _array_metadata(array) for name, array in sorted(arrays.items())
    }
    if record.get("arrays") != expected_metadata:
        raise ValueError(f"{expected_path} array metadata does not match")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _validate_manifest_and_bundle(
    appearance_dir: str | os.PathLike[str],
) -> dict:
    appearance_dir = Path(appearance_dir)
    manifest_path = appearance_dir / "appearance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != APPEARANCE_SCHEMA_VERSION:
        raise ValueError("appearance bundle schema version does not match")
    if set(manifest) != {
        "schema_version",
        "producer",
        "capture",
        "coordinate_frames",
        "encodings",
        "source_bounds",
        "parts",
        "qa_statistics",
        "files",
    }:
        raise ValueError("appearance manifest top-level fields do not match")

    producer = manifest["producer"]
    if producer.get("module") != "modules.part_synthesis.appearance_evidence":
        raise ValueError("appearance producer module does not match")
    if re.fullmatch(r"[0-9a-f]{40}", str(producer.get("omnipart_commit"))) is None:
        raise ValueError("appearance producer commit is invalid")
    if not isinstance(producer.get("random_seed"), int) or isinstance(
        producer.get("random_seed"), bool
    ):
        raise ValueError("appearance random seed must be an integer")

    capture = manifest["capture"]
    if set(capture) != {
        "view_count",
        "resolution",
        "sampling",
        "fov_degrees",
        "radius",
        "near",
        "far",
        "extrinsics_convention",
        "intrinsics_convention",
        "image_array_order",
    }:
        raise ValueError("appearance capture fields do not match")
    expected_capture = {
        "view_count": VIEW_COUNT,
        "resolution": [RESOLUTION, RESOLUTION],
        "sampling": "hammersley_sphere",
        "fov_degrees": FOV_DEGREES,
        "radius": CAMERA_RADIUS,
        "extrinsics_convention": "world_to_camera_column_vector",
        "intrinsics_convention": "normalized_opencv_3x3",
        "image_array_order": "view_height_width_channel",
    }
    if any(capture.get(key) != value for key, value in expected_capture.items()):
        raise ValueError("appearance capture contract does not match")
    near = capture.get("near")
    far = capture.get("far")
    if (
        isinstance(near, bool)
        or isinstance(far, bool)
        or not isinstance(near, (int, float))
        or not isinstance(far, (int, float))
        or not np.isfinite(near)
        or not np.isfinite(far)
        or not 0.0 < near < far
    ):
        raise ValueError("appearance capture planes are invalid")
    expected_coordinate_frames = {
        "source_surface": "omnipart_internal_z_up",
        "gltf": "gltf_y_up",
        "omnipart_internal_z_up_to_gltf_y_up": (
            INTERNAL_Z_UP_TO_GLTF_Y_UP.tolist()
        ),
    }
    if manifest["coordinate_frames"] != expected_coordinate_frames:
        raise ValueError("appearance coordinate-frame contract does not match")
    expected_encodings = {
        "rgb": "straight_srgb_uint8",
        "alpha": "linear_coverage_uint8",
        "alpha_threshold": ALPHA_THRESHOLD,
        "source_depth": "camera_space_positive_z_internal_units_background_zero",
        "source_normal": "internal_world_unit_face_normal_background_zero",
        "source_part_id": "zero_based_triangle_part_index_background_minus_one",
    }
    if manifest["encodings"] != expected_encodings:
        raise ValueError("appearance encoding contract does not match")
    if set(manifest["files"]) != {"appearance_views", "source_surface"}:
        raise ValueError("appearance manifest file records do not match")

    appearance_views_path = appearance_dir / "appearance_views.npz"
    source_surface_path = appearance_dir / "source_surface.npz"
    appearance_views = _load_npz(appearance_views_path)
    source_surface = _load_npz(source_surface_path)
    if set(appearance_views) != set(_APPEARANCE_VIEW_SPECS):
        raise ValueError("appearance view array keys do not match")
    if set(source_surface) != set(_SOURCE_SURFACE_SPECS):
        raise ValueError("source surface array keys do not match")
    for name, (dtype, shape) in _APPEARANCE_VIEW_SPECS.items():
        array = appearance_views[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"appearance view array {name} has the wrong contract")
    for name, dtype in _SOURCE_SURFACE_SPECS.items():
        if source_surface[name].dtype != dtype:
            raise ValueError(f"source surface array {name} has the wrong dtype")

    vertices = source_surface["vertices"]
    faces = source_surface["faces"]
    face_part_id = source_surface["face_part_id"]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError("source vertices have the wrong shape")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("source faces have the wrong shape")
    if face_part_id.shape != (faces.shape[0],):
        raise ValueError("source face_part_id has the wrong shape")
    if not np.isfinite(vertices).all():
        raise ValueError("source vertices must be finite")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise ValueError("source face indices are out of range")
    triangles = vertices[faces]
    if np.any(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        == 0
    ):
        raise ValueError("source faces must have nonzero area")

    parts = manifest["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("appearance manifest must contain at least one part")
    part_count = len(parts)
    if sorted(np.unique(face_part_id).tolist()) != list(range(part_count)):
        raise ValueError("source face_part_id does not match the manifest part set")

    vertex_cursor = 0
    source_output_indices = set()
    for expected_index, part in enumerate(parts):
        if set(part) != {
            "part_index",
            "source_output_index",
            "source_glb_name",
            "vertex_count",
            "face_count",
            "bounds_min",
            "bounds_max",
            "visible_pixel_count",
            "visible_view_count",
        }:
            raise ValueError("manifest part fields do not match")
        if part.get("part_index") != expected_index:
            raise ValueError("manifest part_index ordering is not continuous")
        if (
            not isinstance(part.get("source_output_index"), int)
            or isinstance(part.get("source_output_index"), bool)
            or part["source_output_index"] <= 0
        ):
            raise ValueError("manifest source_output_index is invalid")
        if part["source_output_index"] in source_output_indices:
            raise ValueError("manifest source_output_index values must be unique")
        source_output_indices.add(part["source_output_index"])
        source_glb_name = part.get("source_glb_name")
        if (
            not isinstance(source_glb_name, str)
            or Path(source_glb_name).name != source_glb_name
            or Path(source_glb_name).suffix.lower() != ".glb"
        ):
            raise ValueError("manifest source_glb_name is invalid")
        vertex_count = part.get("vertex_count")
        face_count = part.get("face_count")
        if (
            not isinstance(vertex_count, int)
            or vertex_count <= 0
            or not isinstance(face_count, int)
            or face_count <= 0
        ):
            raise ValueError("manifest part mesh counts are invalid")
        part_vertices = vertices[vertex_cursor : vertex_cursor + vertex_count]
        if part_vertices.shape[0] != vertex_count:
            raise ValueError("manifest part vertex counts do not cover source vertices")
        if int(np.count_nonzero(face_part_id == expected_index)) != face_count:
            raise ValueError("manifest part face count does not match source faces")
        part_faces = faces[face_part_id == expected_index]
        if (
            part_faces.min() < vertex_cursor
            or part_faces.max() >= vertex_cursor + vertex_count
        ):
            raise ValueError("source part faces cross their manifest vertex segment")
        if part.get("bounds_min") != part_vertices.min(axis=0).tolist():
            raise ValueError("manifest part bounds_min does not match")
        if part.get("bounds_max") != part_vertices.max(axis=0).tolist():
            raise ValueError("manifest part bounds_max does not match")
        vertex_cursor += vertex_count
    if vertex_cursor != vertices.shape[0]:
        raise ValueError("manifest part vertex counts do not cover source vertices")

    rgb = appearance_views["rgb"]
    alpha = appearance_views["alpha"]
    source_depth = appearance_views["source_depth"]
    source_normal = appearance_views["source_normal"]
    source_part_id = appearance_views["source_part_id"]
    extrinsics = appearance_views["extrinsics"]
    intrinsics = appearance_views["intrinsics"]
    if not (
        np.isfinite(source_depth).all()
        and np.isfinite(source_normal).all()
        and np.isfinite(extrinsics).all()
        and np.isfinite(intrinsics).all()
    ):
        raise ValueError("appearance view floating-point arrays must be finite")
    if not np.array_equal(
        extrinsics[:, 3, :],
        np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (VIEW_COUNT, 4),
        ),
    ):
        raise ValueError("appearance extrinsics have invalid homogeneous rows")
    if np.any((source_part_id < -1) | (source_part_id >= part_count)):
        raise ValueError("source_part_id contains an invalid label")
    foreground = source_part_id >= 0
    background = ~foreground
    if np.any(source_depth[foreground] <= 0):
        raise ValueError("source foreground depth must be positive")
    if np.any(source_depth[background] != 0):
        raise ValueError("source background depth must be zero")
    if np.any(source_normal[background] != 0):
        raise ValueError("source background normal must be zero")
    if np.any(rgb[alpha == 0] != 0):
        raise ValueError("zero-alpha Gaussian background RGB must be zero")
    alpha_foreground = alpha.astype(np.float32) / 255.0 >= ALPHA_THRESHOLD
    geometry_foreground = source_part_id >= 0
    if not np.any(alpha_foreground):
        raise ValueError("appearance capture has no Gaussian alpha foreground")
    if not np.all(np.any(geometry_foreground, axis=(1, 2))):
        raise ValueError("every appearance view must contain source G-buffer foreground")
    if not np.any(alpha_foreground & geometry_foreground):
        raise ValueError("Gaussian alpha and source G-buffer foreground do not overlap")
    minimum_depth, maximum_depth = _camera_depth_extrema(vertices, extrinsics)
    if not float(near) < minimum_depth <= maximum_depth < float(far):
        raise ValueError(
            "appearance capture planes do not strictly contain all source vertices"
        )
    expected_near, expected_far = _derive_capture_planes(
        vertices,
        torch.from_numpy(extrinsics),
    )
    if float(near) != expected_near or float(far) != expected_far:
        raise ValueError("appearance capture planes do not match source geometry")

    expected_source_bounds = {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
    }
    if manifest["source_bounds"] != expected_source_bounds:
        raise ValueError("source bounds do not match source vertices")

    visibility = _part_visibility(source_part_id, part_count)
    for part, expected_visibility in zip(parts, visibility):
        if part.get("visible_pixel_count") != expected_visibility[
            "visible_pixel_count"
        ]:
            raise ValueError("manifest part visible_pixel_count does not match")
        if part.get("visible_view_count") != expected_visibility[
            "visible_view_count"
        ]:
            raise ValueError("manifest part visible_view_count does not match")
    if manifest["qa_statistics"] != _qa_statistics(
        alpha,
        source_part_id,
        part_count,
    ):
        raise ValueError("appearance QA statistics do not match bundle arrays")

    _validate_npz_record(
        appearance_dir=appearance_dir,
        record=manifest["files"]["appearance_views"],
        expected_path="appearance_views.npz",
        arrays=appearance_views,
    )
    _validate_npz_record(
        appearance_dir=appearance_dir,
        record=manifest["files"]["source_surface"],
        expected_path="source_surface.npz",
        arrays=source_surface,
    )
    contact_sheet_path = appearance_dir / "qa" / "appearance_contact_sheet.png"
    with Image.open(contact_sheet_path) as contact_sheet:
        if contact_sheet.mode != "RGB" or contact_sheet.size != (1024, 1024):
            raise ValueError("appearance contact sheet must be RGB 1024x1024")

    return {
        "schema_version": BUNDLE_VALIDATION_SCHEMA_VERSION,
        "bundle_schema_version": APPEARANCE_SCHEMA_VERSION,
        "valid": True,
        "validated_files": [
            "appearance_manifest.json",
            "appearance_views.npz",
            "source_surface.npz",
            "qa/appearance_contact_sheet.png",
        ],
    }


def write_appearance_bundle(
    *,
    output_dir: str | os.PathLike[str],
    merged_gaussian: Gaussian,
    source_parts: Sequence[AppearanceSourcePart],
    seed: int,
) -> Path:
    """Write and validate the complete HAG4R OmniPart appearance bundle."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be the integer used for OmniPart inference")
    source_parts = tuple(source_parts)
    vertices, faces, face_part_id = _combine_source_surface(source_parts)

    appearance_dir = Path(output_dir) / "appearance"
    qa_dir = appearance_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in _MANAGED_RELATIVE_PATHS:
        target = appearance_dir / relative_path
        if target.exists():
            target.unlink()

    extrinsics_tensor, intrinsics_tensor = _generate_cameras()
    near, far = _derive_capture_planes(vertices, extrinsics_tensor)
    rgb, alpha = _render_gaussian_views(
        merged_gaussian,
        extrinsics_tensor,
        intrinsics_tensor,
        near,
        far,
    )
    source_depth, source_normal_float32, source_part_id = _render_source_gbuffers(
        vertices,
        faces,
        face_part_id,
        extrinsics_tensor,
        intrinsics_tensor,
        near,
        far,
    )
    foreground_norms = np.linalg.norm(
        source_normal_float32[source_part_id >= 0],
        axis=1,
    )
    if (
        foreground_norms.size > 0
        and float(np.max(np.abs(foreground_norms - 1.0))) > 1e-5
    ):
        raise ValueError("source foreground normals must be unit length")

    appearance_views = {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "alpha": np.ascontiguousarray(alpha, dtype=np.uint8),
        "source_depth": np.ascontiguousarray(source_depth, dtype=np.float32),
        "source_normal": np.ascontiguousarray(
            source_normal_float32, dtype=np.float16
        ),
        "source_part_id": np.ascontiguousarray(source_part_id, dtype=np.int16),
        "extrinsics": np.ascontiguousarray(
            extrinsics_tensor.detach().cpu().numpy(), dtype=np.float32
        ),
        "intrinsics": np.ascontiguousarray(
            intrinsics_tensor.detach().cpu().numpy(), dtype=np.float32
        ),
    }
    for name, (dtype, shape) in _APPEARANCE_VIEW_SPECS.items():
        array = appearance_views[name]
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"appearance view array {name} has the wrong contract")
    source_surface = {
        "vertices": vertices,
        "faces": faces,
        "face_part_id": face_part_id,
    }

    appearance_views_path = appearance_dir / "appearance_views.npz"
    source_surface_path = appearance_dir / "source_surface.npz"
    contact_sheet_path = qa_dir / "appearance_contact_sheet.png"
    np.savez_compressed(appearance_views_path, **appearance_views)
    np.savez_compressed(source_surface_path, **source_surface)
    _write_contact_sheet(rgb, contact_sheet_path)

    visibility = _part_visibility(source_part_id, len(source_parts))
    manifest_parts = []
    for part, part_visibility in zip(source_parts, visibility):
        manifest_parts.append(
            {
                "part_index": part.part_index,
                "source_output_index": part.source_output_index,
                "source_glb_name": part.source_glb_name,
                "vertex_count": int(part.vertices.shape[0]),
                "face_count": int(part.faces.shape[0]),
                "bounds_min": part.vertices.min(axis=0).tolist(),
                "bounds_max": part.vertices.max(axis=0).tolist(),
                "visible_pixel_count": part_visibility["visible_pixel_count"],
                "visible_view_count": part_visibility["visible_view_count"],
            }
        )

    manifest = {
        "schema_version": APPEARANCE_SCHEMA_VERSION,
        "producer": {
            "module": "modules.part_synthesis.appearance_evidence",
            "omnipart_commit": _omnipart_commit(),
            "random_seed": seed,
        },
        "capture": {
            "view_count": VIEW_COUNT,
            "resolution": [RESOLUTION, RESOLUTION],
            "sampling": "hammersley_sphere",
            "fov_degrees": FOV_DEGREES,
            "radius": CAMERA_RADIUS,
            "near": near,
            "far": far,
            "extrinsics_convention": "world_to_camera_column_vector",
            "intrinsics_convention": "normalized_opencv_3x3",
            "image_array_order": "view_height_width_channel",
        },
        "coordinate_frames": {
            "source_surface": "omnipart_internal_z_up",
            "gltf": "gltf_y_up",
            "omnipart_internal_z_up_to_gltf_y_up": (
                INTERNAL_Z_UP_TO_GLTF_Y_UP.tolist()
            ),
        },
        "encodings": {
            "rgb": "straight_srgb_uint8",
            "alpha": "linear_coverage_uint8",
            "alpha_threshold": ALPHA_THRESHOLD,
            "source_depth": (
                "camera_space_positive_z_internal_units_background_zero"
            ),
            "source_normal": "internal_world_unit_face_normal_background_zero",
            "source_part_id": (
                "zero_based_triangle_part_index_background_minus_one"
            ),
        },
        "source_bounds": {
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        },
        "parts": manifest_parts,
        "qa_statistics": _qa_statistics(
            alpha,
            source_part_id,
            len(source_parts),
        ),
        "files": {
            "appearance_views": _npz_file_record(
                appearance_views_path,
                appearance_views,
            ),
            "source_surface": _npz_file_record(
                source_surface_path,
                source_surface,
            ),
        },
    }
    manifest_path = appearance_dir / "appearance_manifest.json"
    _json_write(manifest_path, manifest)
    validation = _validate_manifest_and_bundle(appearance_dir)
    _json_write(qa_dir / "bundle_validation.json", validation)
    return manifest_path


__all__ = [
    "ALPHA_THRESHOLD",
    "APPEARANCE_SCHEMA_VERSION",
    "AppearanceSourcePart",
    "BUNDLE_VALIDATION_SCHEMA_VERSION",
    "CAMERA_RADIUS",
    "CAPTURE_DEPTH_MARGIN_ABSOLUTE",
    "CAPTURE_DEPTH_MARGIN_FRACTION",
    "FOV_DEGREES",
    "INTERNAL_Z_UP_TO_GLTF_Y_UP",
    "RESOLUTION",
    "VIEW_COUNT",
    "write_appearance_bundle",
]
