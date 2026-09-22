from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import genesis as gs

from .overlay_state import anchor_overlay_records, controller_overlay_records
from .triptych import HAG4R_LABELS, PANEL_ORDER, png_record, stitch_triptych

PANEL_SIZE = (768, 768)
TRIPTYCH_FOV_DEGREES = 35.0
TRIPTYCH_TARGET_FILL_FRACTION = 0.75
PART_SHADED_METALLIC = 0.0
PART_SHADED_ROUGHNESS = 0.65
PART_SHADED_BACKGROUND_RGB = (0.18, 0.18, 0.18)
PART_SHADED_AMBIENT_RGB = (0.3, 0.3, 0.3)
PART_SHADED_LIGHT_DIRECTION = (-1.0, -1.0, -1.0)
PART_SHADED_LIGHT_COLOR = (1.0, 1.0, 1.0)
PART_SHADED_LIGHT_INTENSITY = 3.0
FIXED_RGB_VIEW_SIZE = (512, 512)
FIXED_RGB_VIEW_ORDER = ("full", "context")
FIXED_RGB_FOV_DEGREES = 40.0
GENESIS_NATIVE_DEBUG_CAMERA_RENDERER = "genesis_native_debug_camera_renderer"
ANCHOR_DEBUG_BOX_COLOR = (0.16, 0.43, 1.0, 1.0)
CONTROLLER_DEBUG_BOX_COLOR = (1.0, 0.35, 0.16, 1.0)
CONTROLLER_DEBUG_AXIS_COLOR = (1.0, 0.35, 0.16, 1.0)
DEBUG_BOX_WIREFRAME_RADIUS = 0.004


def _entity_positions(entity) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    if getattr(entity, "active", False):
        positions = tensor_to_array(entity.get_state(track_grad=False).pos)
    else:
        positions = tensor_to_array(entity.init_positions)

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 3:
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"FEM entity positions must have shape (N, 3); got {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("FEM entity positions must be finite")
    return positions


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError(f"Cannot construct RGB triptych camera with degenerate {label} vector")
    return (vector / norm).astype(np.float32)


def _triptych_world_bounds(session, boxes: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for entity in session.entities.values():
        positions = _entity_positions(entity)
        mins.append(positions.min(axis=0))
        maxs.append(positions.max(axis=0))

    for box in boxes or []:
        box = np.asarray(box, dtype=np.float32)
        if box.shape != (6,):
            raise ValueError(f"RGB triptych overlay boxes must have shape (6,); got {box.shape}")
        if not np.all(np.isfinite(box)):
            raise ValueError("RGB triptych overlay boxes must be finite")
        if np.any(box[:3] > box[3:]):
            raise ValueError(f"RGB triptych overlay box has inverted min/max bounds: {box.tolist()}")
        mins.append(box[:3])
        maxs.append(box[3:])

    if not mins:
        mins.append(np.array([-0.5, -0.5, -0.5], dtype=np.float32))
        maxs.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    bbox_min = np.min(np.stack(mins), axis=0)
    bbox_max = np.max(np.stack(maxs), axis=0)
    return bbox_min, bbox_max


def _bounds_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float32,
    )


def _triptych_view_basis(label: str) -> tuple[np.ndarray, np.ndarray]:
    if label == "top":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)
    if label == "northeast":
        return _normalize(np.array([-1.0, -1.0, -0.75], dtype=np.float32), "view direction"), np.array(
            [0.0, 0.0, 1.0], dtype=np.float32
        )
    if label == "southwest":
        return _normalize(np.array([1.0, 1.0, -0.75], dtype=np.float32), "view direction"), np.array(
            [0.0, 0.0, 1.0], dtype=np.float32
        )
    raise ValueError(f"Unsupported RGB triptych panel label: {label}")


def _triptych_camera_pose(
    label: str,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    fov_degrees: float = TRIPTYCH_FOV_DEGREES,
    target_fill_fraction: float = TRIPTYCH_TARGET_FILL_FRACTION,
) -> dict[str, Any]:
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    view_dir, nominal_up = _triptych_view_basis(label)
    right = _normalize(np.cross(view_dir, nominal_up), "right")
    up = _normalize(np.cross(right, view_dir), "up")
    relative_corners = _bounds_corners(bbox_min, bbox_max) - center
    tangent = math.tan(math.radians(fov_degrees) * 0.5) * target_fill_fraction
    along_view = relative_corners @ view_dir
    horizontal = np.abs(relative_corners @ right)
    vertical = np.abs(relative_corners @ up)
    distance = float(np.max(np.maximum(horizontal, vertical) / tangent - along_view))
    extent = float(np.max(bbox_max - bbox_min))
    distance = max(distance, max(extent, 1.0e-4))
    position = center - view_dir * distance
    radius = float(np.max(np.linalg.norm(relative_corners, axis=1)))
    near = max(1.0e-5, distance - radius * 1.25)
    far = distance + radius * 1.25
    return {
        "pos": tuple(float(value) for value in position),
        "lookat": tuple(float(value) for value in center),
        "up": tuple(float(value) for value in up),
        "near": float(near),
        "far": float(far),
    }


def _validate_overlay_box(box: Any) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (6,):
        raise ValueError(f"RGB triptych overlay box must have shape (6,); got {box.shape}")
    if not np.all(np.isfinite(box)):
        raise ValueError("RGB triptych overlay box must be finite")
    if np.any(box[:3] > box[3:]):
        raise ValueError(f"RGB triptych overlay box has inverted min/max bounds: {box.tolist()}")
    return box


def _overlay_source_box_values(record: dict[str, Any]) -> np.ndarray:
    box = record.get("env_local_box")
    if box is None:
        box = record.get("source_box")
    if box is None:
        raise ValueError(f"RGB triptych overlay {record.get('kind')!r} is missing box bounds")
    return _validate_overlay_box(box)


def _overlay_displacement(record: dict[str, Any]) -> np.ndarray:
    displacement = np.asarray(record.get("displacement", [0.0, 0.0, 0.0]), dtype=np.float32)
    if displacement.shape != (3,):
        raise ValueError(f"RGB triptych live controller displacement must have shape (3,); got {displacement.shape}")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("RGB triptych live controller displacement must be finite")
    return displacement


def _overlay_render_box_values(record: dict[str, Any]) -> np.ndarray:
    if record.get("rendered_env_local_box") is not None:
        return _validate_overlay_box(record["rendered_env_local_box"])

    box = _overlay_source_box_values(record)
    if record.get("kind") == "live_box_controller":
        displacement = _overlay_displacement(record)
        box = box + np.concatenate((displacement, displacement))
    return _validate_overlay_box(box)


def _render_overlay_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_records = []
    for record in records:
        data = dict(record)
        data["rendered_env_local_box"] = _overlay_render_box_values(record).tolist()
        rendered_records.append(data)
    return rendered_records


def _boxes_from_overlays(records: list[dict[str, Any]]) -> list[np.ndarray]:
    boxes = []
    for record in records:
        kind = record.get("kind")
        if kind not in {"static_anchor", "live_box_controller"}:
            raise ValueError(f"Unsupported RGB triptych overlay kind: {kind!r}")
        boxes.append(_overlay_render_box_values(record))
    return boxes


def _overlay_box_bounds(record: dict[str, Any]) -> np.ndarray:
    box = _overlay_render_box_values(record)
    return np.stack((box[:3], box[3:])).astype(np.float32)


def _overlay_debug_color(record: dict[str, Any]) -> tuple[float, float, float, float]:
    if record.get("kind") == "static_anchor":
        return ANCHOR_DEBUG_BOX_COLOR
    if record.get("kind") == "live_box_controller":
        return CONTROLLER_DEBUG_BOX_COLOR
    raise ValueError(f"Unsupported RGB triptych overlay kind: {record.get('kind')!r}")


def _normalize_camera_rgb(rgb: Any, *, label: str, camera: Any) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    array = np.asarray(tensor_to_array(rgb))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Genesis debug camera {label!r} returned RGB with invalid shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]

    expected_width, expected_height = camera.res
    if array.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"Genesis debug camera {label!r} returned RGB shape {array.shape[:2]}, "
            f"expected {(expected_height, expected_width)}"
        )

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _render_camera_rgb(
    camera: Any, *, label: str, force_render: bool, render_pass: str = "rgb"
) -> np.ndarray:
    rgb, _depth, _seg, _normal = camera.render(
        rgb=True,
        depth=False,
        segmentation=False,
        normal=False,
        force_render=force_render,
        render_pass=render_pass,
    )
    if rgb is None:
        raise ValueError(f"Genesis debug camera {label!r} did not return RGB data")
    return _normalize_camera_rgb(rgb, label=label, camera=camera)


def _camera_scalar_array(value: Any, *, label: str, camera: Any) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    array = np.asarray(tensor_to_array(value), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    expected_width, expected_height = camera.res
    if array.shape != (expected_height, expected_width):
        raise ValueError(
            f"Genesis debug camera {label!r} returned scalar shape {array.shape}, "
            f"expected {(expected_height, expected_width)}"
        )
    return array


def _render_camera_depth(camera: Any, *, label: str, force_render: bool) -> np.ndarray:
    _rgb, depth, _seg, _normal = camera.render(
        rgb=False,
        depth=True,
        segmentation=False,
        normal=False,
        force_render=force_render,
        render_pass="rgb",
    )
    if depth is None:
        raise ValueError(f"Genesis debug camera {label!r} did not return depth data")
    array = _camera_scalar_array(depth, label=label, camera=camera)
    valid = np.isfinite(array) & (array > float(camera.near)) & (array < float(camera.far) * (1.0 - 1.0e-3))
    image = np.zeros((*array.shape, 3), dtype=np.uint8)
    if np.any(valid):
        minimum = float(np.min(array[valid]))
        maximum = float(np.max(array[valid]))
        if maximum > minimum:
            normalized = 1.0 - (array[valid] - minimum) / (maximum - minimum)
        else:
            normalized = np.ones(int(np.count_nonzero(valid)), dtype=np.float32)
        values = np.rint(255.0 * np.clip(normalized, 0.0, 1.0)).astype(np.uint8)
        image[valid] = values[:, None]
    return image


def _render_camera_normal(camera: Any, *, label: str, force_render: bool) -> np.ndarray:
    from genesis.utils.misc import tensor_to_array

    _rgb, depth, _seg, normal = camera.render(
        rgb=False,
        depth=True,
        segmentation=False,
        normal=True,
        force_render=force_render,
        render_pass="rgb",
    )
    if depth is None or normal is None:
        raise ValueError(f"Genesis debug camera {label!r} did not return depth/normal data")
    depth_array = _camera_scalar_array(depth, label=label, camera=camera)
    array = np.asarray(tensor_to_array(normal))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    expected_width, expected_height = camera.res
    if array.shape != (expected_height, expected_width, 3):
        raise ValueError(
            f"Genesis debug camera {label!r} returned normal shape {array.shape}, "
            f"expected {(expected_height, expected_width, 3)}"
        )
    if np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float32)
        if float(np.nanmin(array)) < 0.0:
            array = 0.5 * (array + 1.0)
        image = np.rint(255.0 * np.clip(array, 0.0, 1.0)).astype(np.uint8)
    else:
        image = np.clip(array, 0, 255).astype(np.uint8)
    valid = (
        np.isfinite(depth_array)
        & (depth_array > float(camera.near))
        & (depth_array < float(camera.far) * (1.0 - 1.0e-3))
    )
    image[~valid] = 0
    return np.ascontiguousarray(image)


def _render_camera_part_segmentation(
    camera: Any,
    *,
    label: str,
    force_render: bool,
    context: Any,
    render_pass: str = "part_segmentation",
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    _rgb, _depth, segmentation, _normal = camera.render(
        rgb=False,
        depth=False,
        segmentation=True,
        colorize_seg=False,
        normal=False,
        force_render=force_render,
        render_pass=render_pass,
    )
    if segmentation is None:
        raise ValueError(f"Genesis debug camera {label!r} did not return segmentation data")
    from genesis.utils.misc import tensor_to_array

    indices = np.asarray(tensor_to_array(segmentation))
    if indices.ndim == 3 and indices.shape[0] == 1:
        indices = indices[0]
    expected_width, expected_height = camera.res
    if indices.shape != (expected_height, expected_width):
        raise ValueError(
            f"Genesis debug camera {label!r} returned segmentation shape {indices.shape}, "
            f"expected {(expected_height, expected_width)}"
        )
    image = np.zeros((*indices.shape, 3), dtype=np.uint8)
    known = np.zeros(indices.shape, dtype=bool)
    known[indices == 0] = True
    for seg_idxc, color in context.part_segmentation_palette_by_idxc.items():
        mask = indices == int(seg_idxc)
        image[mask] = np.asarray(color, dtype=np.uint8)
        known |= mask
    if not np.all(known):
        unknown = sorted(int(value) for value in np.unique(indices[~known]))
        raise ValueError(f"Part segmentation render returned unmapped indices: {unknown}")
    image = np.ascontiguousarray(image)
    if return_indices:
        return image, indices
    return image


def _json_vec3(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"RGB triptych camera vector must have shape (3,); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("RGB triptych camera vector must be finite")
    return [float(component) for component in array]


def _camera_metadata(camera: Any, label: str, pose: dict[str, tuple[float, float, float]]) -> dict[str, Any]:
    position = _json_vec3(camera.pos)
    target = _json_vec3(camera.lookat)
    up = _json_vec3(camera.up)
    return {
        "label": label,
        "model": camera.model,
        "debug": bool(camera.debug),
        "res": [int(camera.res[0]), int(camera.res[1])],
        "resolution": [int(camera.res[0]), int(camera.res[1])],
        "fov": float(camera.fov),
        "fov_degrees": float(camera.fov),
        "near": float(camera.near),
        "far": float(camera.far),
        "pos": position,
        "position": position,
        "lookat": target,
        "target": target,
        "up": up,
        "pose": {
            "pos": [float(value) for value in pose["pos"]],
            "lookat": [float(value) for value in pose["lookat"]],
            "up": [float(value) for value in pose["up"]],
        },
    }


def canonical_fixed_rgb_request(value: Any) -> dict[str, Any]:
    """Validate the immutable PAIP camera request and return its canonical form."""
    if not isinstance(value, dict) or set(value) != {"mode", "render_every_steps", "views"}:
        raise ValueError("fixed_rgb_views request must contain exactly mode, render_every_steps, and views")
    if value.get("mode") != "fixed_rgb_views" or value.get("render_every_steps") != 10:
        raise ValueError("fixed_rgb_views requires mode=fixed_rgb_views and render_every_steps=10")
    raw_views = value.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != 2:
        raise ValueError("fixed_rgb_views requires exactly two views")

    views = []
    for index, (expected_name, raw) in enumerate(zip(FIXED_RGB_VIEW_ORDER, raw_views, strict=True)):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "position",
            "look_at",
            "up",
            "resolution",
            "fov_degrees",
        }:
            raise ValueError(f"fixed_rgb_views.views[{index}] has unexpected or missing fields")
        if raw.get("name") != expected_name:
            raise ValueError("fixed_rgb_views view order must be full, context")
        if raw.get("resolution") != list(FIXED_RGB_VIEW_SIZE):
            raise ValueError("fixed_rgb_views resolution must be [512, 512]")
        fov = raw.get("fov_degrees")
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or float(fov) != FIXED_RGB_FOV_DEGREES:
            raise ValueError("fixed_rgb_views fov_degrees must be 40.0")

        vectors = {}
        for field in ("position", "look_at", "up"):
            vector = raw.get(field)
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in vector)
            ):
                raise ValueError(f"fixed_rgb_views {expected_name}.{field} must be a finite vec3")
            canonical = [float(component) for component in vector]
            if not all(math.isfinite(component) for component in canonical):
                raise ValueError(f"fixed_rgb_views {expected_name}.{field} must be a finite vec3")
            vectors[field] = canonical
        direction = np.asarray(vectors["look_at"]) - np.asarray(vectors["position"])
        up = np.asarray(vectors["up"])
        if np.linalg.norm(direction) <= 1.0e-8 or np.linalg.norm(up) <= 1.0e-8:
            raise ValueError(f"fixed_rgb_views {expected_name} has a degenerate camera vector")
        if np.linalg.norm(np.cross(direction, up)) <= 1.0e-8:
            raise ValueError(f"fixed_rgb_views {expected_name}.up is collinear with the view direction")
        views.append(
            {
                "name": expected_name,
                "position": vectors["position"],
                "look_at": vectors["look_at"],
                "up": vectors["up"],
                "resolution": list(FIXED_RGB_VIEW_SIZE),
                "fov_degrees": FIXED_RGB_FOV_DEGREES,
            }
        )
    return {"mode": "fixed_rgb_views", "render_every_steps": 10, "views": views}


def fixed_rgb_request_hash(value: Any) -> str:
    canonical = canonical_fixed_rgb_request(value)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_part_shaded_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("mode") != "part_shaded_triptych":
        raise ValueError("part shaded rendering requires mode=part_shaded_triptych")
    expected = {
        "resolution": list(PANEL_SIZE),
        "fov_deg": TRIPTYCH_FOV_DEGREES,
        "target_fill_fraction": TRIPTYCH_TARGET_FILL_FRACTION,
        "surface": {"metallic": PART_SHADED_METALLIC, "roughness": PART_SHADED_ROUGHNESS},
        "lighting": {
            "background_rgb": list(PART_SHADED_BACKGROUND_RGB),
            "ambient_rgb": list(PART_SHADED_AMBIENT_RGB),
            "directional": {
                "direction": list(PART_SHADED_LIGHT_DIRECTION),
                "color": list(PART_SHADED_LIGHT_COLOR),
                "intensity": PART_SHADED_LIGHT_INTENSITY,
            },
        },
        "shadows": True,
        "part_id_pass": True,
    }
    request = dict(value)
    for key, default in expected.items():
        request.setdefault(key, default)
        if request[key] != default:
            raise ValueError(f"part_shaded_triptych requires {key}={default!r}")
    roi = dict(request.get("roi", {}))
    roi.setdefault("enabled", request.get("target_box") is not None)
    roi["enabled"] = bool(roi["enabled"] and request.get("target_box") is not None)
    roi.setdefault("resolution", list(PANEL_SIZE))
    roi.setdefault("target_fill_fraction", TRIPTYCH_TARGET_FILL_FRACTION)
    if roi["resolution"] != list(PANEL_SIZE):
        raise ValueError(f"part_shaded_triptych ROI resolution must be {list(PANEL_SIZE)}")
    if float(roi["target_fill_fraction"]) != TRIPTYCH_TARGET_FILL_FRACTION:
        raise ValueError(
            f"part_shaded_triptych ROI target_fill_fraction must be {TRIPTYCH_TARGET_FILL_FRACTION}"
        )
    request["roi"] = roi
    if roi["enabled"]:
        request["target_box"] = _validate_overlay_box(request["target_box"]).astype(float).tolist()
    capture_id = str(request.get("capture_id", "")).strip()
    if capture_id and not all(character.isalnum() or character in "_-" for character in capture_id):
        raise ValueError("diagnostic_visual.capture_id must contain only letters, digits, underscores, or hyphens")
    request["capture_id"] = capture_id or None
    request["action_phase"] = str(request.get("action_phase", "periodic"))
    if request.get("target_part_id") is not None:
        request["target_part_id"] = int(request["target_part_id"])
    return request


def part_shaded_vis_options() -> gs.options.VisOptions:
    return gs.options.VisOptions(
        background_color=PART_SHADED_BACKGROUND_RGB,
        ambient_light=PART_SHADED_AMBIENT_RGB,
        shadow=True,
        lights=(
            gs.options.vis.DirectionalLight(
                dir=PART_SHADED_LIGHT_DIRECTION,
                color=PART_SHADED_LIGHT_COLOR,
                intensity=PART_SHADED_LIGHT_INTENSITY,
            ),
        ),
    )


class VisualTelemetry:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.triptych_cameras: dict[str, Any] = {}
        self.triptych_camera_poses: dict[str, dict[str, Any]] = {}
        self.triptych_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.roi_camera: Any | None = None
        self.fixed_rgb_cameras: dict[str, Any] = {}
        self._triptych_debug_marker_handles: list[Any] = []

    def reset_triptych_cameras(self) -> None:
        self.triptych_cameras.clear()
        self.triptych_camera_poses.clear()
        self.triptych_bounds = None
        self.roi_camera = None
        self._triptych_debug_marker_handles.clear()

    def reset_fixed_rgb_cameras(self) -> None:
        self.fixed_rgb_cameras.clear()

    def register_fixed_rgb_cameras(self, session) -> None:
        self.reset_fixed_rgb_cameras()
        if session.scene is None:
            raise ValueError("Cannot register fixed RGB cameras before the Genesis scene exists")
        for index, label in enumerate(FIXED_RGB_VIEW_ORDER):
            camera = session.scene.add_camera(
                model="pinhole",
                res=FIXED_RGB_VIEW_SIZE,
                pos=(1.0 + index, 1.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov=FIXED_RGB_FOV_DEGREES,
                GUI=False,
                debug=False,
            )
            self.fixed_rgb_cameras[label] = camera
        if tuple(self.fixed_rgb_cameras) != FIXED_RGB_VIEW_ORDER:
            raise ValueError("Fixed RGB camera registration did not produce full/context order")

    def register_triptych_cameras(self, session) -> None:
        self.reset_triptych_cameras()
        if session.scene is None:
            raise ValueError("Cannot register RGB triptych cameras before the Genesis scene exists")
        bbox_min, bbox_max = _triptych_world_bounds(session)
        self.triptych_bounds = (bbox_min.copy(), bbox_max.copy())
        for label in PANEL_ORDER:
            pose = _triptych_camera_pose(label, bbox_min, bbox_max)
            camera = session.scene.add_camera(
                model="pinhole",
                res=PANEL_SIZE,
                pos=pose["pos"],
                lookat=pose["lookat"],
                up=pose["up"],
                fov=TRIPTYCH_FOV_DEGREES,
                near=pose["near"],
                far=pose["far"],
                GUI=False,
                debug=True,
            )
            self.triptych_cameras[label] = camera
            self.triptych_camera_poses[label] = pose
        if tuple(self.triptych_cameras) != PANEL_ORDER:
            raise ValueError("RGB triptych camera registration did not produce the expected panel order")
        roi_pose = self.triptych_camera_poses[PANEL_ORDER[0]]
        self.roi_camera = session.scene.add_camera(
            model="pinhole",
            res=PANEL_SIZE,
            pos=roi_pose["pos"],
            lookat=roi_pose["lookat"],
            up=roi_pose["up"],
            fov=TRIPTYCH_FOV_DEGREES,
            near=roi_pose["near"],
            far=roi_pose["far"],
            GUI=False,
            debug=True,
        )

    def triptych_camera(self, label: str):
        if label not in self.triptych_cameras:
            raise ValueError(f"RGB triptych camera is not registered for panel {label!r}")
        return self.triptych_cameras[label]

    def _clear_triptych_debug_markers(self, session) -> None:
        if session.scene is None:
            self._triptych_debug_marker_handles.clear()
            return
        while self._triptych_debug_marker_handles:
            handle = self._triptych_debug_marker_handles.pop()
            session.scene.clear_debug_object(handle)

    def _draw_triptych_debug_markers(self, session, overlays: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if session.scene is None:
            raise ValueError("Cannot draw RGB triptych debug markers before the Genesis scene exists")
        self._clear_triptych_debug_markers(session)
        marker_records = []
        for index, record in enumerate(overlays):
            bounds = _overlay_box_bounds(record)
            color = _overlay_debug_color(record)
            handle = session.scene.draw_debug_box(
                bounds,
                color=color,
                wireframe=True,
                wireframe_radius=DEBUG_BOX_WIREFRAME_RADIUS,
            )
            session.scene.visualizer._context.mark_part_shaded_overlay(handle)
            self._triptych_debug_marker_handles.append(handle)
            motion_axis = record.get("motion_axis")
            if record.get("kind") == "live_box_controller" and motion_axis in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
                axis_vectors = {
                    "+X": (1.0, 0.0, 0.0),
                    "-X": (-1.0, 0.0, 0.0),
                    "+Y": (0.0, 1.0, 0.0),
                    "-Y": (0.0, -1.0, 0.0),
                    "+Z": (0.0, 0.0, 1.0),
                    "-Z": (0.0, 0.0, -1.0),
                }
                center = bounds.mean(axis=0)
                length = max(float(np.max(bounds[1] - bounds[0])) * 0.30, 0.01)
                axis_handle = session.scene.draw_debug_arrow(
                    center,
                    np.asarray(axis_vectors[motion_axis], dtype=np.float32) * length,
                    radius=DEBUG_BOX_WIREFRAME_RADIUS,
                    color=CONTROLLER_DEBUG_AXIS_COLOR,
                )
                session.scene.visualizer._context.mark_part_shaded_overlay(axis_handle)
                self._triptych_debug_marker_handles.append(axis_handle)
            marker_records.append(
                {
                    "index": int(index),
                    "kind": record.get("kind"),
                    "controller_id": record.get("controller_id"),
                    "bounds": bounds.tolist(),
                    "rendered_env_local_box": record.get("rendered_env_local_box", bounds.reshape(6).tolist()),
                    "env_local_box": record.get("env_local_box"),
                    "displacement": record.get("displacement"),
                    "moved_distance": record.get("moved_distance"),
                    "motion_active": record.get("motion_active"),
                    "motion_axis": motion_axis,
                    "motion_axis_label": motion_axis,
                    "color": list(color),
                    "wireframe": True,
                    "wireframe_radius": DEBUG_BOX_WIREFRAME_RADIUS,
                }
            )
        return marker_records

    @torch.no_grad()
    def capture_triptych(
        self,
        session,
        *,
        mode: str,
        frame_index: int | None = None,
        visual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode not in {
            "rgb_triptych",
            "depth_triptych",
            "normal_triptych",
            "part_segmentation_triptych",
            "part_shaded_triptych",
        }:
            raise ValueError(f"Unsupported diagnostic triptych mode: {mode}")
        request = canonical_part_shaded_request(visual or {"mode": mode}) if mode == "part_shaded_triptych" else {}
        if frame_index is None:
            frame_index = int(session.current_step)
        capture_id = request.get("capture_id")
        stem = f"frame_{frame_index:06d}" + (f"__{capture_id}" if capture_id else "")
        overlays = _render_overlay_records(
            anchor_overlay_records(session.anchor_records) + controller_overlay_records(session.controllers)
        )
        if self.triptych_bounds is None:
            raise ValueError("RGB triptych cameras have no frozen episode bounds")
        bbox_min, bbox_max = self.triptych_bounds
        context = session.scene.visualizer._context
        part_mode = mode in {"part_segmentation_triptych", "part_shaded_triptych"}
        if part_mode:
            contracts = [entity._part_segmentation_config for entity in session.entities.values()]
            palette = contracts[0]["context_palette"]
            context.replace_part_segmentation_context_boxes(
                [
                    {
                        "kind": "fixture" if record["kind"] == "static_anchor" else "probe",
                        "controller_id": record.get("controller_id") or record.get("anchor_id") or index,
                        "bounds": _overlay_box_bounds(record),
                        "color": palette["fixture" if record["kind"] == "static_anchor" else "probe"],
                    }
                    for index, record in enumerate(overlays)
                ]
            )
        else:
            contracts = []

        marker_records = []
        panel_records = []
        panel_paths = []
        panel_render_seconds = []
        png_encode_write_seconds = 0.0
        capture_started = time.perf_counter()
        prefix = {
            "rgb_triptych": "rgb",
            "depth_triptych": "depth",
            "normal_triptych": "normal",
            "part_segmentation_triptych": "part_segmentation",
            "part_shaded_triptych": "part_shaded",
        }[mode]

        def write_panel(image, *, label, hag4r_label, directory, pose, camera, renderer_mode):
            nonlocal png_encode_write_seconds
            path = self.output_dir / directory / label / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            png_started = time.perf_counter()
            Image.fromarray(image, mode="RGB").save(path)
            png_encode_write_seconds += time.perf_counter() - png_started
            record = png_record(
                path,
                label=label,
                hag4r_label=hag4r_label,
                frame_index=frame_index,
                simulation_step=session.current_step,
            )
            record.update(
                {
                    "capture_id": capture_id,
                    "action_phase": request.get("action_phase", "periodic"),
                    "simulation_time_s": float(session.current_step * session.scene.dt),
                    "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "renderer": {
                        "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                        "mode": renderer_mode,
                        "debug_camera": True,
                        "native_camera_registered": True,
                    },
                    "camera": _camera_metadata(camera, label, pose),
                }
            )
            return path, record

        cuda_memory_available = bool(torch.cuda.is_available() and gs.backend == gs.cuda)
        if cuda_memory_available:
            torch.cuda.reset_peak_memory_stats()
        try:
            if mode in {"rgb_triptych", "part_shaded_triptych"}:
                marker_records = self._draw_triptych_debug_markers(session, overlays)
            for panel_index, label in enumerate(PANEL_ORDER):
                camera = self.triptych_camera(label)
                pose = self.triptych_camera_poses[label]
                camera.set_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
                render_started = time.perf_counter()
                if mode == "rgb_triptych":
                    image = _render_camera_rgb(camera, label=label, force_render=panel_index == 0)
                elif mode == "depth_triptych":
                    image = _render_camera_depth(camera, label=label, force_render=panel_index == 0)
                elif mode == "normal_triptych":
                    image = _render_camera_normal(camera, label=label, force_render=panel_index == 0)
                elif mode == "part_segmentation_triptych":
                    image = _render_camera_part_segmentation(
                        camera, label=label, force_render=panel_index == 0, context=context
                    )
                else:
                    image = _render_camera_rgb(
                        camera,
                        label=label,
                        force_render=panel_index == 0,
                        render_pass="part_shaded",
                    )
                panel_render_seconds.append(time.perf_counter() - render_started)
                path, record = write_panel(
                    image,
                    label=label,
                    hag4r_label=HAG4R_LABELS[label],
                    directory=f"png_{prefix}_panels",
                    pose=pose,
                    camera=camera,
                    renderer_mode=mode,
                )
                panel_paths.append(path)
                panel_records.append(record)

            update_stats = dict(context.last_render_update_stats)
            actual_upload_stats = dict(context.jit.last_buffer_upload_stats)

            part_id_views = []
            part_id_panel_paths = {}
            part_id_stitched = None
            target_visible_pixels = {}
            if mode == "part_shaded_triptych" and request["part_id_pass"]:
                target_part_id = request.get("target_part_id")
                target_segmentation_indices = sorted(
                    {
                        int(context.seg_color_map.key_map[("hag4r_part", entity_uid, part_id)])
                        for _env_index, entity_uid, part_id in context.part_segmentation_nodes
                        if part_id == target_part_id
                    }
                )
                id_paths = []
                for label in PANEL_ORDER:
                    camera = self.triptych_camera(label)
                    pose = self.triptych_camera_poses[label]
                    image, indices = _render_camera_part_segmentation(
                        camera,
                        label=label,
                        force_render=False,
                        context=context,
                        render_pass="part_segmentation_current",
                        return_indices=True,
                    )
                    target_visible_pixels[label] = int(
                        np.count_nonzero(np.isin(indices, target_segmentation_indices))
                    )
                    path, record = write_panel(
                        image,
                        label=label,
                        hag4r_label=HAG4R_LABELS[label],
                        directory="png_part_id_panels",
                        pose=pose,
                        camera=camera,
                        renderer_mode="part_segmentation_triptych",
                    )
                    record["target_part_visible_pixel_count"] = target_visible_pixels[label]
                    id_paths.append(path)
                    part_id_views.append(record)
                    part_id_panel_paths[label] = str(path)
                id_stitched_path = self.output_dir / "png_part_id_triptych" / f"{stem}.png"
                stitch_triptych(id_paths, id_stitched_path)
                part_id_stitched = png_record(
                    id_stitched_path,
                    label="triptych",
                    hag4r_label="triptych",
                    frame_index=frame_index,
                    simulation_step=session.current_step,
                )
                part_id_stitched["renderer"] = {
                    "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "mode": "part_segmentation_triptych",
                }

            roi_view = None
            roi_path = None
            roi_camera_id = None
            best_overview_label = None
            if mode == "part_shaded_triptych" and request["roi"]["enabled"]:
                target_box = np.asarray(request["target_box"], dtype=np.float32)
                target_min, target_max = target_box[:3], target_box[3:]
                best_overview_label = max(PANEL_ORDER, key=target_visible_pixels.__getitem__)
                roi_pose = _triptych_camera_pose(best_overview_label, target_min, target_max)
                roi_camera_id = hashlib.sha256(
                    json.dumps(
                        {"source": best_overview_label, "target_box": request["target_box"]},
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:16]
                self.roi_camera._near = roi_pose["near"]
                self.roi_camera._far = roi_pose["far"]
                self.roi_camera.set_pose(pos=roi_pose["pos"], lookat=roi_pose["lookat"], up=roi_pose["up"])
                render_started = time.perf_counter()
                roi_image = _render_camera_rgb(
                    self.roi_camera,
                    label="roi",
                    force_render=False,
                    render_pass="part_shaded",
                )
                panel_render_seconds.append(time.perf_counter() - render_started)
                roi_path_obj, roi_view = write_panel(
                    roi_image,
                    label="roi",
                    hag4r_label="roi",
                    directory="png_part_shaded_roi",
                    pose=roi_pose,
                    camera=self.roi_camera,
                    renderer_mode=mode,
                )
                roi_path = str(roi_path_obj)
                roi_view.update(
                    {
                        "roi_camera_id": roi_camera_id,
                        "source_view": best_overview_label,
                        "source_overview_label": best_overview_label,
                        "source_view_target_part_visible_pixel_count": target_visible_pixels[
                            best_overview_label
                        ],
                        "target_box": request["target_box"],
                        "target_part_id": request.get("target_part_id"),
                    }
                )

            triptych_path = self.output_dir / f"png_{prefix}_triptych" / f"{stem}.png"
            stitch_started = time.perf_counter()
            stitch_triptych(panel_paths, triptych_path)
            stitch_seconds = time.perf_counter() - stitch_started
            png_encode_write_seconds += stitch_seconds
            triptych_record = png_record(
                triptych_path,
                label="triptych",
                hag4r_label="triptych",
                frame_index=frame_index,
                simulation_step=session.current_step,
            )
            triptych_record.update(
                {
                    "capture_id": capture_id,
                    "action_phase": request.get("action_phase", "periodic"),
                    "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "renderer": {"backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER, "mode": mode},
                    "source_panel_count": len(PANEL_ORDER),
                }
            )

            indexed_parts = [
                {"env_index": int(key[0]), "entity_uid": str(key[1]), "part_id": int(key[2]), **counts}
                for key, counts in sorted(context.part_segmentation_indexed_counts.items())
            ]
            if cuda_memory_available:
                torch.cuda.synchronize()
                peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
                current_gpu_memory_allocated_bytes = int(torch.cuda.memory_allocated())
            else:
                peak_gpu_memory_bytes = None
                current_gpu_memory_allocated_bytes = None
            metadata_path = self.output_dir / f"{prefix}_triptych_metadata" / f"{stem}.json"
            frame_metadata = panel_records + [triptych_record] + ([roi_view] if roi_view is not None else [])
            metadata = {
                "requested": True,
                "mode": mode,
                "rendered": True,
                "capture_id": capture_id,
                "action_phase": request.get("action_phase", "periodic"),
                "simulation_time_s": float(session.current_step * session.scene.dt),
                "target_part_id": request.get("target_part_id"),
                "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                "renderer": {
                    "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "mode": mode,
                    "camera_model": "pinhole",
                    "debug_camera": True,
                    "panel_size": list(PANEL_SIZE),
                    "fov_degrees": TRIPTYCH_FOV_DEGREES,
                    "target_fill_fraction": TRIPTYCH_TARGET_FILL_FRACTION,
                    "surface": request.get("surface"),
                    "lighting": request.get("lighting"),
                    "shadows": request.get("shadows"),
                },
                "geometry": {
                    "entity_count": len(session.entities),
                    "bounds": {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
                    "frozen_episode_bounds": {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
                },
                "panel_order": list(PANEL_ORDER),
                "panel_paths": {record["label"]: record["path"] for record in panel_records},
                "panels": {record["hag4r_label"]: record for record in panel_records},
                "views": panel_records,
                "stitched": triptych_record,
                "roi_view": roi_view,
                "roi_path": roi_path,
                "roi_camera_id": roi_camera_id,
                "best_overview_label": best_overview_label,
                "part_id_views": part_id_views,
                "part_id_panel_paths": part_id_panel_paths,
                "part_id_stitched": part_id_stitched,
                "frame_metadata": frame_metadata,
                "overlays": overlays,
                "debug_markers": marker_records,
                "part_legend": [part for contract in contracts for part in contract["parts"]],
                "context_palette": dict(contracts[0]["context_palette"]) if contracts else {},
                "indexed_part_nodes": indexed_parts,
                "performance": {
                    **update_stats,
                    **actual_upload_stats,
                    **context.last_part_segmentation_context_update,
                    "active_context_node_count": len(context.part_segmentation_context_nodes) if part_mode else 0,
                    "total_active_draw_update_node_count": (
                        len(context.part_segmentation_nodes) + len(context.part_segmentation_context_nodes)
                        if part_mode
                        else 0
                    ),
                    "rgb_render_count": (
                        0
                        if mode == "part_segmentation_triptych"
                        else len(PANEL_ORDER) + int(roi_view is not None)
                    ),
                    "segmentation_render_count": (
                        len(PANEL_ORDER) if mode == "part_segmentation_triptych" else len(part_id_views)
                    ),
                    "panel_raster_seconds": panel_render_seconds,
                    "stitch_seconds": stitch_seconds,
                    "png_encode_write_seconds": png_encode_write_seconds,
                    "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                    "current_gpu_memory_allocated_bytes": current_gpu_memory_allocated_bytes,
                    "peak_gpu_memory_available": cuda_memory_available,
                    "peak_gpu_memory_unavailable_reason": (
                        None
                        if cuda_memory_available
                        else "genesis_backend_is_not_cuda" if torch.cuda.is_available() else "cuda_unavailable"
                    ),
                    "capture_seconds": time.perf_counter() - capture_started,
                },
                "capture_time": time.time(),
                "metadata_path": str(metadata_path),
            }
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return metadata
        finally:
            self._clear_triptych_debug_markers(session)

    def capture_rgb_triptych(
        self, session, *, frame_index: int | None = None, visual: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.capture_triptych(session, mode="rgb_triptych", frame_index=frame_index, visual=visual)

    def capture_depth_triptych(
        self, session, *, frame_index: int | None = None, visual: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.capture_triptych(session, mode="depth_triptych", frame_index=frame_index, visual=visual)

    def capture_normal_triptych(
        self, session, *, frame_index: int | None = None, visual: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.capture_triptych(session, mode="normal_triptych", frame_index=frame_index, visual=visual)

    def capture_part_segmentation_triptych(
        self, session, *, frame_index: int | None = None, visual: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.capture_triptych(session, mode="part_segmentation_triptych", frame_index=frame_index, visual=visual)

    def capture_part_shaded_triptych(
        self, session, *, frame_index: int | None = None, visual: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.capture_triptych(session, mode="part_shaded_triptych", frame_index=frame_index, visual=visual)

    @torch.no_grad()
    def capture_fixed_rgb_views(
        self,
        session,
        *,
        visual: dict[str, Any],
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        request = canonical_fixed_rgb_request(visual)
        request_hash = fixed_rgb_request_hash(request)
        if frame_index is None:
            frame_index = int(session.current_step)
        if tuple(self.fixed_rgb_cameras) != FIXED_RGB_VIEW_ORDER:
            raise ValueError("Fixed RGB cameras are not registered")

        panel_records = []
        panel_render_seconds = []
        png_encode_write_seconds = 0.0
        capture_started = time.perf_counter()
        for index, spec in enumerate(request["views"]):
            label = spec["name"]
            camera = self.fixed_rgb_cameras[label]
            if bool(camera.debug):
                raise ValueError(f"Fixed RGB camera {label!r} must be non-debug")
            pose = {
                "pos": tuple(spec["position"]),
                "lookat": tuple(spec["look_at"]),
                "up": tuple(spec["up"]),
            }
            camera.set_pose(pos=pose["pos"], lookat=pose["lookat"], up=pose["up"])
            render_started = time.perf_counter()
            image = _render_camera_rgb(camera, label=label, force_render=index == 0)
            panel_render_seconds.append(time.perf_counter() - render_started)
            path = self.output_dir / "png_fixed_rgb_views" / label / f"frame_{frame_index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            png_started = time.perf_counter()
            Image.fromarray(image, mode="RGB").save(path)
            png_encode_write_seconds += time.perf_counter() - png_started
            record = png_record(
                path,
                label=label,
                hag4r_label=label,
                frame_index=frame_index,
                simulation_step=session.current_step,
            )
            record.update(
                {
                    "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                    "renderer": {
                        "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                        "mode": "fixed_rgb_views",
                        "debug_camera": False,
                        "native_camera_registered": True,
                    },
                    "camera": _camera_metadata(camera, label, pose),
                    "camera_spec": spec,
                    "camera_spec_hash": request_hash,
                }
            )
            panel_records.append(record)

        metadata_path = self.output_dir / "fixed_rgb_views_metadata" / f"frame_{frame_index:06d}.json"
        metadata = {
            "requested": True,
            "mode": "fixed_rgb_views",
            "rendered": True,
            "render_backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
            "renderer": {
                "backend": GENESIS_NATIVE_DEBUG_CAMERA_RENDERER,
                "mode": "fixed_rgb_views",
                "camera_model": "pinhole",
                "debug_camera": False,
                "debug_markers": False,
                "panel_size": list(FIXED_RGB_VIEW_SIZE),
            },
            "camera_specs": request["views"],
            "camera_specs_hash": request_hash,
            "view_order": list(FIXED_RGB_VIEW_ORDER),
            "views": panel_records,
            "frame_metadata": panel_records,
            "overlays": [],
            "debug_markers": [],
            "frame_id": int(frame_index),
            "frame_index": int(frame_index),
            "simulation_step": int(session.current_step),
            "performance": {
                "rgb_render_count": len(FIXED_RGB_VIEW_ORDER),
                "panel_raster_seconds": panel_render_seconds,
                "png_encode_write_seconds": png_encode_write_seconds,
                "capture_seconds": time.perf_counter() - capture_started,
            },
            "capture_time": time.time(),
            "metadata_path": str(metadata_path),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
