from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION = "hag4r-diagnostic-triple-view-evidence-v1"
TRIPLE_VIEW_PANEL_ORDER = ("top", "ne_3q", "sw_3q")
TRIPLE_VIEW_SOURCE_KINDS = frozenset(
    {
        "static_probe_preview",
        "static_anchor_preview",
        "live_pause_observation",
        "live_reset_observation",
        "live_simulate_observation",
        "live_reset_part_segmentation_observation",
        "live_simulate_part_segmentation_observation",
        "episode_png_sequence",
        "episode_part_segmentation_png_sequence",
        "episode_triptych_mp4",
        "episode_part_segmentation_triptych_mp4",
    }
)


def _mesh_box_array(mesh_box: Any) -> np.ndarray:
    arr = np.asarray(mesh_box, dtype=np.float64)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError("mesh_box must be a finite 6-vector")
    if np.any(arr[:3] >= arr[3:]):
        raise ValueError("mesh_box must have strict min < max on every axis")
    return arr


def _camera_json(camera: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "position": [float(value) for value in camera["position"]],
        "target": [float(value) for value in camera["target"]],
        "up": [float(value) for value in camera["up"]],
        "fly_to": bool(camera.get("fly_to", False)),
    }


def triple_view_camera(mesh_box: Any, view_name: str, *, fly_to: bool = False) -> dict[str, Any]:
    box = _mesh_box_array(mesh_box)
    center = (box[:3] + box[3:]) * 0.5
    max_extent = float(np.max(box[3:] - box[:3]))
    d = max(0.35, 2.5 * max_extent)
    if view_name == "top":
        position = center + np.asarray([0.0, d, 0.0])
        up = np.asarray([0.0, 0.0, -1.0])
    elif view_name == "ne_3q":
        position = center + np.asarray([d, 0.75 * d, d])
        up = np.asarray([0.0, 1.0, 0.0])
    elif view_name == "sw_3q":
        position = center + np.asarray([-d, 0.75 * d, -d])
        up = np.asarray([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"unsupported triple-view panel: {view_name}")
    return {
        "position": [float(value) for value in position],
        "target": [float(value) for value in center],
        "up": [float(value) for value in up],
        "fly_to": bool(fly_to),
    }


def triple_view_cameras(mesh_box: Any, *, fly_to: bool = False) -> dict[str, dict[str, Any]]:
    return {view_name: triple_view_camera(mesh_box, view_name, fly_to=fly_to) for view_name in TRIPLE_VIEW_PANEL_ORDER}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_panel_metadata(
    path: str | Path,
    *,
    background_rgb: tuple[int, int, int] | None = None,
    allow_blank: bool = False,
) -> dict[str, Any]:
    png_path = Path(path).expanduser()
    with Image.open(png_path) as image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        height, width = int(arr.shape[0]), int(arr.shape[1])
        unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
        unique_color_count = int(unique_colors.shape[0])
        alpha_nonzero_ratio = float(np.count_nonzero(alpha) / alpha.size)
        visible = alpha > 0
        if background_rgb is None:
            nonblank = visible & np.any(rgb != 255, axis=2) & np.any(rgb != 0, axis=2)
        else:
            background = np.asarray(background_rgb, dtype=np.uint8).reshape(1, 1, 3)
            nonblank = visible & np.any(rgb != background, axis=2)
        nonblank_ratio = float(np.count_nonzero(nonblank) / nonblank.size)
        mode = image.mode

    solid_white = unique_color_count == 1 and bool(np.all(unique_colors[0] == 255))
    solid_black = unique_color_count == 1 and bool(np.all(unique_colors[0] == 0))
    fully_transparent = alpha_nonzero_ratio == 0.0
    one_color = unique_color_count <= 1
    invalid = fully_transparent or solid_white or solid_black or one_color or nonblank_ratio <= 0.0
    if invalid and not allow_blank:
        raise ValueError(
            f"triple-view panel is blank or invalid: path={png_path}, "
            f"unique_color_count={unique_color_count}, alpha_nonzero_ratio={alpha_nonzero_ratio:.6f}, "
            f"nonblank_ratio={nonblank_ratio:.6f}"
        )
    return {
        "path": str(png_path),
        "mode": mode,
        "width": width,
        "height": height,
        "sha256": sha256_file(png_path),
        "unique_color_count": unique_color_count,
        "alpha_nonzero_ratio": alpha_nonzero_ratio,
        "nonblank_ratio": nonblank_ratio,
    }


def _validate_panel_keys(source_png_paths_by_view: Mapping[str, str | Path], panel_order: tuple[str, ...]) -> None:
    expected = set(panel_order)
    actual = set(source_png_paths_by_view)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"triple-view panels must be exactly {list(panel_order)}; missing={missing}, extra={extra}")


def stitch_triptych(
    source_png_paths_by_view: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    panel_order: tuple[str, ...] = TRIPLE_VIEW_PANEL_ORDER,
) -> dict[str, Any]:
    _validate_panel_keys(source_png_paths_by_view, panel_order)
    panels: list[Image.Image] = []
    hashes: dict[str, str] = {}
    dimensions: dict[str, dict[str, int]] = {}
    for view_name in panel_order:
        path = Path(source_png_paths_by_view[view_name]).expanduser()
        panel = Image.open(path).convert("RGB")
        panels.append(panel)
        hashes[view_name] = sha256_file(path)
        dimensions[view_name] = {"width": panel.width, "height": panel.height}
    widths = {panel.width for panel in panels}
    heights = {panel.height for panel in panels}
    if len(widths) != 1 or len(heights) != 1:
        for panel in panels:
            panel.close()
        raise ValueError(f"triple-view panels must have equal dimensions: {dimensions}")
    if len(set(hashes.values())) == 1:
        for panel in panels:
            panel.close()
        raise ValueError("triple-view panels are all byte-identical")
    if len(set(hashes.values())) == 2:
        warnings.warn("two triple-view panels are byte-identical", RuntimeWarning, stacklevel=2)

    panel_width = panels[0].width
    panel_height = panels[0].height
    triptych = Image.new("RGB", (panel_width * len(panel_order), panel_height))
    for index, panel in enumerate(panels):
        triptych.paste(panel, (index * panel_width, 0))
        panel.close()
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triptych.save(out_path)
    triptych.close()
    return {
        "triptych_png_path": str(out_path),
        "triptych_sha256": sha256_file(out_path),
        "triptych_dimensions": {"width": panel_width * len(panel_order), "height": panel_height},
        "panel_order": list(panel_order),
        "panel_hashes": hashes,
        "panel_dimensions": dimensions,
    }


def write_triple_view_manifest(
    output_path: str | Path,
    *,
    evidence_id: str,
    source_kind: str,
    target_id: str = "",
    compile_id: str = "",
    trial_id: str = "",
    episode_id: str = "",
    tool_result_index: int | None = None,
    source_png_paths_by_view: Mapping[str, str | Path] | None = None,
    triptych_png_path: str | Path | None = None,
    cameras_by_view: Mapping[str, Mapping[str, Any]] | None = None,
    views_extra_by_view: Mapping[str, Mapping[str, Any]] | None = None,
    frames: list[dict[str, Any]] | None = None,
    sequence_dir: str | Path | None = None,
    source_view_dirs: Mapping[str, str | Path] | None = None,
    validation: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source_kind not in TRIPLE_VIEW_SOURCE_KINDS:
        raise ValueError(f"unsupported triple-view source_kind: {source_kind}")
    manifest: dict[str, Any] = {
        "schema_version": TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": str(evidence_id),
        "source_kind": str(source_kind),
        "target_id": str(target_id),
        "compile_id": str(compile_id),
        "trial_id": str(trial_id),
        "episode_id": str(episode_id),
        "panel_order": list(TRIPLE_VIEW_PANEL_ORDER),
        "validation": {"status": "ok", "warning_codes": [], "hard_errors": []},
    }
    if tool_result_index is not None:
        manifest["tool_result_index"] = int(tool_result_index)
    if triptych_png_path is not None:
        triptych_path = Path(triptych_png_path).expanduser()
        with Image.open(triptych_path) as image:
            manifest["triptych_dimensions"] = {"width": int(image.width), "height": int(image.height)}
        manifest["triptych_png_path"] = str(triptych_path)
        manifest["triptych_sha256"] = sha256_file(triptych_path)
    if source_png_paths_by_view is not None:
        _validate_panel_keys(source_png_paths_by_view, TRIPLE_VIEW_PANEL_ORDER)
        views = []
        for index, view_name in enumerate(TRIPLE_VIEW_PANEL_ORDER):
            path = Path(source_png_paths_by_view[view_name]).expanduser()
            panel = image_panel_metadata(path)
            extra_view = dict((views_extra_by_view or {}).get(view_name, {}))
            camera = dict((cameras_by_view or {}).get(view_name, {}))
            record = {
                "view_name": view_name,
                "panel_index": index,
                "source_png_path": str(path),
                "source_sha256": panel["sha256"],
                "dimensions": {"width": panel["width"], "height": panel["height"]},
                "camera": _camera_json(camera) if camera else {},
                "metrics": {
                    "unique_color_count": panel["unique_color_count"],
                    "nonblank_ratio": panel["nonblank_ratio"],
                    "alpha_nonzero_ratio": panel["alpha_nonzero_ratio"],
                },
                "validation_status": "ok",
            }
            record.update(extra_view)
            views.append(record)
        manifest["views"] = views
    if sequence_dir is not None:
        manifest["sequence_dir"] = str(Path(sequence_dir).expanduser())
    if source_view_dirs is not None:
        manifest["source_view_dirs"] = {
            view_name: str(Path(source_view_dirs[view_name]).expanduser())
            for view_name in TRIPLE_VIEW_PANEL_ORDER
            if view_name in source_view_dirs
        }
    if frames is not None:
        manifest["frames"] = frames
        manifest["frame_count"] = len(frames)
    if validation:
        manifest["validation"] = dict(validation)
    if extra:
        manifest.update(dict(extra))
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "TRIPLE_VIEW_EVIDENCE_SCHEMA_VERSION",
    "TRIPLE_VIEW_PANEL_ORDER",
    "TRIPLE_VIEW_SOURCE_KINDS",
    "image_panel_metadata",
    "sha256_file",
    "stitch_triptych",
    "triple_view_camera",
    "triple_view_cameras",
    "write_triple_view_manifest",
]
