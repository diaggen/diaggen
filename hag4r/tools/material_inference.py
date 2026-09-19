"""Infer primitive-wise constitutive parameters from segmented multi-view images."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hag4r.tools.object_description import object_description_text
from hag4r.agentic.state import to_json_dict


DEFAULT_MATERIAL_MAX_ATTEMPTS = 3
DEFAULT_MATERIAL_MAX_CALLS = DEFAULT_MATERIAL_MAX_ATTEMPTS * 4
MATERIAL_INFERENCE_SCHEMA_VERSION = "hag4r-agentic-gpt-staged-material-inference-v1"
MATERIAL_FILL_MODE_DECISION_SCHEMA_VERSION = "hag4r-material-fill-mode-decision-v1"
VIEW_NAMES = ["top", "bottom", "front", "back", "left", "right"]
VOLUME_FILL_MODES = ("solid_fill", "hollow_wall")
LEGACY_MATERIAL_FIELDS = (
    "representation",
    "expected_representation",
    "representation_decision",
    "shell_" + "thickness_m",
    "wall_" + "thickness_m",
)
POSITIVE_MATERIAL_FIELDS = (
    "density_kg_m3",
    "youngs_modulus_pa",
)
GEOMETRY_REQUIRED_TEXT_FIELDS = (
    "geometric_category",
    "geometry_description",
    "relative_size",
    "attachment_pattern",
)
MATERIAL_REQUIRED_TEXT_FIELDS = (
    "part_name",
    "part_semantics",
    "part_texture",
    "major_material_name",
)
MATERIAL_REQUIRED_NUMERIC_FIELDS = (
    "density_kg_m3",
    "youngs_modulus_pa",
    "poisson_ratio",
    "friction_coefficient",
)
GEOMETRY_ALLOWED_ENTRY_FIELDS = ("part_index", "part_color_rgb", *GEOMETRY_REQUIRED_TEXT_FIELDS)
FILL_MODE_ALLOWED_ENTRY_FIELDS = (
    "part_index",
    "part_color_rgb",
    "volume_fill_mode",
    "fill_mode_rationale",
    "fill_mode_evidence",
)
MATERIAL_ALLOWED_ENTRY_FIELDS = (
    "part_index",
    *MATERIAL_REQUIRED_TEXT_FIELDS,
    "part_color_rgb",
    *MATERIAL_REQUIRED_NUMERIC_FIELDS,
)
NUMERIC_LITERAL_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PROFILE_TUPLE_BINDING_RE = re.compile(
    r"profile=(?P<profile>[A-Za-z0-9_-]+); tuple "
    rf"density_kg_m3=(?P<density_kg_m3>{NUMERIC_LITERAL_PATTERN}), "
    rf"youngs_modulus_pa=(?P<youngs_modulus_pa>{NUMERIC_LITERAL_PATTERN}), "
    rf"poisson_ratio=(?P<poisson_ratio>{NUMERIC_LITERAL_PATTERN}), "
    rf"friction_coefficient=(?P<friction_coefficient>{NUMERIC_LITERAL_PATTERN})\s*$"
)
NON_PHYSICAL_MATERIAL_NAMES = {"", "void", "none", "unknown", "unobserved", "artifact", "air"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def infer_image_mime_type(image_path: str | Path) -> str:
    ext = os.path.splitext(str(image_path))[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def build_indexed_parts(part_colors_list):
    indexed_parts = []
    for i, color in enumerate(part_colors_list):
        indexed_parts.append({
            "part_index": int(i),
            "part_color_rgb": [float(color[0]), float(color[1]), float(color[2])],
        })
    return indexed_parts


def _part_color_schema(rgb: list[float]) -> dict[str, Any]:
    return {
        "type": "array",
        "prefixItems": [
            {"type": "number", "const": rgb[0]},
            {"type": "number", "const": rgb[1]},
            {"type": "number", "const": rgb[2]},
        ],
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }


def _reject_legacy_material_fields(
    payload: dict[str, Any],
    *,
    phase: str,
    entries_key: str = "parts",
) -> None:
    if not isinstance(payload, dict):
        return
    for key in payload:
        if key in LEGACY_MATERIAL_FIELDS or key.startswith("tri_"):
            raise ValueError(f"{phase} payload contains forbidden legacy field `{key}`.")
    entries = payload.get(entries_key)
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        part_index = entry.get("part_index", "?")
        for key in entry:
            if key in LEGACY_MATERIAL_FIELDS or key.startswith("tri_"):
                raise ValueError(
                    f"{phase} payload part {part_index} contains forbidden legacy field `{key}`."
                )


def _reject_unexpected_payload_fields(
    payload: dict[str, Any],
    *,
    phase: str,
    top_level_fields: tuple[str, ...],
    entry_fields: tuple[str, ...],
    entries_key: str = "parts",
) -> None:
    unexpected_top = sorted(set(payload) - set(top_level_fields))
    if unexpected_top:
        raise ValueError(f"{phase} payload contains unexpected field `{unexpected_top[0]}`.")
    entries = payload.get(entries_key)
    if not isinstance(entries, list):
        return
    allowed = set(entry_fields)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        unexpected = sorted(set(entry) - allowed)
        if unexpected:
            part_index = entry.get("part_index", "?")
            raise ValueError(
                f"{phase} payload part {part_index} contains unexpected field `{unexpected[0]}`."
            )


def build_index_locked_schema(indexed_parts):
    any_of_items = []
    for part in indexed_parts:
        idx = part["part_index"]
        rgb = part["part_color_rgb"]
        any_of_items.append({
            "type": "object",
            "properties": {
                "part_index": {"type": "integer", "const": idx},
                "part_name": {"type": "string"},
                "part_semantics": {"type": "string"},
                "part_texture": {"type": "string"},
                "major_material_name": {"type": "string"},
                "part_color_rgb": _part_color_schema(rgb),
                "density_kg_m3": {"type": "number", "exclusiveMinimum": 0},
                "youngs_modulus_pa": {"type": "number", "exclusiveMinimum": 0},
                "poisson_ratio": {
                    "type": "number",
                    "exclusiveMinimum": -1,
                    "exclusiveMaximum": 0.5,
                },
                "friction_coefficient": {"type": "number", "minimum": 0},
            },
            "required": [
                "part_index",
                "part_name",
                "part_semantics",
                "part_texture",
                "major_material_name",
                "part_color_rgb",
                "density_kg_m3",
                "youngs_modulus_pa",
                "poisson_ratio",
                "friction_coefficient",
            ],
            "additionalProperties": False,
        })

    return {
        "type": "object",
        "properties": {
            "inferred_object_name": {"type": "string"},
            "parts": {
                "type": "array",
                "minItems": len(indexed_parts),
                "maxItems": len(indexed_parts),
                "items": {"anyOf": any_of_items},
            },
        },
        "required": ["inferred_object_name", "parts"],
        "additionalProperties": False,
    }


def build_fill_mode_locked_schema(indexed_parts) -> dict[str, Any]:
    any_of_items = []
    for part in indexed_parts:
        idx = part["part_index"]
        rgb = part["part_color_rgb"]
        any_of_items.append({
            "type": "object",
            "properties": {
                "part_index": {"type": "integer", "const": idx},
                "part_color_rgb": _part_color_schema(rgb),
                "volume_fill_mode": {"type": "string", "enum": list(VOLUME_FILL_MODES)},
                "fill_mode_rationale": {"type": "string", "minLength": 1},
                "fill_mode_evidence": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
            },
            "required": [
                "part_index",
                "part_color_rgb",
                "volume_fill_mode",
                "fill_mode_rationale",
                "fill_mode_evidence",
            ],
            "additionalProperties": False,
        })

    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": [MATERIAL_FILL_MODE_DECISION_SCHEMA_VERSION],
            },
            "inferred_object_name": {"type": "string"},
            "parts": {
                "type": "array",
                "minItems": len(indexed_parts),
                "maxItems": len(indexed_parts),
                "items": {"anyOf": any_of_items},
            },
        },
        "required": ["schema_version", "inferred_object_name", "parts"],
        "additionalProperties": False,
    }


def build_geometry_locked_schema(indexed_parts):
    any_of_items = []
    for part in indexed_parts:
        idx = part["part_index"]
        rgb = part["part_color_rgb"]
        any_of_items.append({
            "type": "object",
            "properties": {
                "part_index": {"type": "integer", "const": idx},
                "part_color_rgb": _part_color_schema(rgb),
                "geometric_category": {"type": "string"},
                "geometry_description": {"type": "string"},
                "relative_size": {"type": "string"},
                "attachment_pattern": {"type": "string"},
            },
            "required": [
                "part_index",
                "part_color_rgb",
                "geometric_category",
                "geometry_description",
                "relative_size",
                "attachment_pattern",
            ],
            "additionalProperties": False,
        })

    return {
        "type": "object",
        "properties": {
            "inferred_object_name": {"type": "string"},
            "parts": {
                "type": "array",
                "minItems": len(indexed_parts),
                "maxItems": len(indexed_parts),
                "items": {"anyOf": any_of_items},
            },
        },
        "required": ["inferred_object_name", "parts"],
        "additionalProperties": False,
    }


def validate_and_canonicalize_indexed_entries(result, indexed_parts, entries_key):
    if not isinstance(result, dict):
        raise ValueError("Response is not a JSON object.")
    if entries_key not in result:
        raise ValueError(f"Missing required field `{entries_key}`.")
    if "inferred_object_name" not in result:
        raise ValueError("Missing required field `inferred_object_name`.")

    entries = result[entries_key]
    if not isinstance(entries, list):
        raise ValueError(f"Field `{entries_key}` is not a list.")

    num_parts = len(indexed_parts)
    if len(entries) != num_parts:
        raise ValueError(
            f"Expected exactly {num_parts} entries in `{entries_key}`, but got {len(entries)}."
        )

    expected_colors = {
        part["part_index"]: part["part_color_rgb"] for part in indexed_parts
    }
    canonical_entries = [None] * num_parts
    seen_indices = set()
    mismatches = []

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Each entry in `{entries_key}` must be an object.")

        idx = entry.get("part_index")
        if not isinstance(idx, int):
            raise ValueError(f"Invalid part_index value: {idx}")
        if idx < 0 or idx >= num_parts:
            raise ValueError(f"part_index out of range: {idx}")
        if idx in seen_indices:
            raise ValueError(f"Duplicate part_index found: {idx}")
        seen_indices.add(idx)

        predicted_rgb = entry.get("part_color_rgb")
        if not isinstance(predicted_rgb, list) or len(predicted_rgb) != 3:
            raise ValueError(f"Invalid part_color_rgb for index {idx}: {predicted_rgb}")
        try:
            predicted_rgb = [float(predicted_rgb[0]), float(predicted_rgb[1]), float(predicted_rgb[2])]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric part_color_rgb for index {idx}: {predicted_rgb}"
            ) from exc

        expected_rgb = expected_colors[idx]
        if predicted_rgb != expected_rgb:
            mismatches.append(
                f"index {idx}: expected {expected_rgb}, got {predicted_rgb}"
            )

        canonical_entry = dict(entry)
        canonical_entry["part_color_rgb"] = list(expected_rgb)
        canonical_entries[idx] = canonical_entry

    missing_indices = [i for i in range(num_parts) if i not in seen_indices]
    if missing_indices:
        raise ValueError(f"Missing part indices: {missing_indices}")

    if mismatches:
        raise ValueError(
            "part_index and part_color_rgb mismatch:\n" + "\n".join(mismatches)
        )

    return canonical_entries


def validate_and_canonicalize_parts(result, indexed_parts):
    _reject_legacy_material_fields(result, phase="material")
    _reject_unexpected_payload_fields(
        result,
        phase="material",
        top_level_fields=("inferred_object_name", "parts"),
        entry_fields=MATERIAL_ALLOWED_ENTRY_FIELDS,
    )
    parts = validate_and_canonicalize_indexed_entries(
        result, indexed_parts, entries_key="parts")
    _validate_material_required_fields(parts)
    _validate_profile_tuple_bindings(parts)
    validate_constitutive_parameters(parts)
    return parts


def validate_and_canonicalize_fill_modes(result, indexed_parts) -> list[dict[str, Any]]:
    _reject_legacy_material_fields(result, phase="fill_mode")
    _reject_unexpected_payload_fields(
        result,
        phase="fill_mode",
        top_level_fields=("schema_version", "inferred_object_name", "parts"),
        entry_fields=FILL_MODE_ALLOWED_ENTRY_FIELDS,
    )
    if result.get("schema_version") != MATERIAL_FILL_MODE_DECISION_SCHEMA_VERSION:
        raise ValueError("material fill-mode decision has invalid schema_version")
    parts = validate_and_canonicalize_indexed_entries(
        result, indexed_parts, entries_key="parts")
    for entry in parts:
        part_index = int(entry["part_index"])
        mode = entry.get("volume_fill_mode")
        if mode not in VOLUME_FILL_MODES:
            raise ValueError(
                f"part {part_index} field `volume_fill_mode` must be one of {VOLUME_FILL_MODES}, got {mode!r}."
            )
        _require_non_empty_text_field(entry, "fill_mode_rationale", part_index=part_index)
        evidence = entry.get("fill_mode_evidence")
        if not isinstance(evidence, list) or not any(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError(
                f"part {part_index} field `fill_mode_evidence` must contain at least one non-empty string."
            )
    return parts


def _finite_number(value, *, field_name, part_index):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"part {part_index} field `{field_name}` must be a finite number, got {value!r}."
        )
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(
            f"part {part_index} field `{field_name}` must be finite, got {value!r}."
        )
    return numeric


def validate_constitutive_parameters(parts):
    for part in parts:
        idx = part["part_index"]
        for field_name in ("density_kg_m3", "youngs_modulus_pa"):
            value = _finite_number(part.get(field_name), field_name=field_name, part_index=idx)
            if value <= 0.0:
                raise ValueError(
                    f"part {idx} field `{field_name}` must be > 0. "
                    "Unobserved or ambiguous parts still need a guessed positive material."
                )

        poisson = _finite_number(part.get("poisson_ratio"), field_name="poisson_ratio", part_index=idx)
        if poisson <= -1.0 or poisson >= 0.5:
            raise ValueError(
                f"part {idx} field `poisson_ratio` must be > -1 and < 0.5, got {poisson}."
            )

        friction = _finite_number(
            part.get("friction_coefficient"),
            field_name="friction_coefficient",
            part_index=idx,
        )
        if friction < 0.0:
            raise ValueError(
                f"part {idx} field `friction_coefficient` must be >= 0, got {friction}."
            )

        material_name = str(part.get("major_material_name", "")).strip().lower()
        if material_name in NON_PHYSICAL_MATERIAL_NAMES:
            raise ValueError(
                f"part {idx} field `major_material_name` must be a guessed physical material, "
                f"got {part.get('major_material_name')!r}."
            )


def validate_and_canonicalize_geometry(result, indexed_parts):
    _reject_legacy_material_fields(result, phase="geometry")
    _reject_unexpected_payload_fields(
        result,
        phase="geometry",
        top_level_fields=("inferred_object_name", "parts"),
        entry_fields=GEOMETRY_ALLOWED_ENTRY_FIELDS,
    )
    return validate_and_canonicalize_indexed_entries(
        result, indexed_parts, entries_key="parts")


def _require_non_empty_text_field(entry: dict[str, Any], field_name: str, *, part_index: int) -> None:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"part {part_index} field `{field_name}` must be a non-empty string.")


def _validate_geometry_required_fields(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        part_index = int(entry["part_index"])
        for field_name in GEOMETRY_REQUIRED_TEXT_FIELDS:
            _require_non_empty_text_field(entry, field_name, part_index=part_index)


def _validate_material_required_fields(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        part_index = int(entry["part_index"])
        for field_name in MATERIAL_REQUIRED_TEXT_FIELDS:
            _require_non_empty_text_field(entry, field_name, part_index=part_index)
        for field_name in MATERIAL_REQUIRED_NUMERIC_FIELDS:
            if field_name not in entry:
                raise ValueError(f"part {part_index} is missing required numeric field `{field_name}`.")
            _finite_number(entry[field_name], field_name=field_name, part_index=part_index)


def _validate_profile_tuple_bindings(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        part_index = int(entry["part_index"])
        semantics = str(entry.get("part_semantics", ""))
        match = PROFILE_TUPLE_BINDING_RE.search(semantics)
        if match is None:
            raise ValueError(
                f"part {part_index} field `part_semantics` must end with "
                "`profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, "
                "poisson_ratio=<nu>, friction_coefficient=<mu>`."
            )
        if not match.group("profile").strip():
            raise ValueError(f"part {part_index} profile tuple binding has an empty profile slug.")
        for field_name in MATERIAL_REQUIRED_NUMERIC_FIELDS:
            tuple_value = float(match.group(field_name))
            field_value = _finite_number(entry.get(field_name), field_name=field_name, part_index=part_index)
            if not np.isclose(field_value, tuple_value, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"part {part_index} field `{field_name}` must match the profile tuple binding: "
                    f"field={field_value}, tuple={tuple_value}."
                )


def build_view_image_paths(image_dir: str | Path):
    image_dir = Path(image_dir)
    view_image_paths = []
    for view_name in VIEW_NAMES:
        image_path = os.path.join(
            image_dir, f"mesh_combined_part_labels_{view_name}.png")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Missing segmented image for view `{view_name}`: {image_path}"
            )
        view_image_paths.append((view_name, image_path))
    return view_image_paths


def build_processed_image_path(image_dir: str | Path):
    image_dir = Path(image_dir)
    image_name = os.path.basename(os.path.normpath(str(image_dir)))
    image_path = os.path.join(image_dir, f"{image_name}_processed.png")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Missing processed reference image: {image_path}"
        )
    return image_path


def build_view_image_inputs(view_image_paths):
    view_image_inputs = []
    for view_name, image_path in view_image_paths:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        mime_type = infer_image_mime_type(image_path)
        data_url = f"data:{mime_type};base64,{b64}"
        view_image_inputs.append({
            "view_name": view_name,
            "image_path": image_path,
            "data_url": data_url,
            "display_name": f"Segmented reconstructed view: {view_name}",
        })
    return view_image_inputs


def build_processed_image_input(image_path: str | Path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = infer_image_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{b64}"
    return {
        "view_name": "processed_reference",
        "image_path": image_path,
        "data_url": data_url,
        "display_name": (
            "Processed original reference image: "
            f"{os.path.basename(image_path)}"
        ),
    }


def infer_shape_hint(bbox_width, bbox_height, fill_ratio, area_fraction):
    aspect_ratio = float(bbox_width) / float(max(bbox_height, 1))
    if area_fraction >= 0.15:
        if aspect_ratio >= 3.0:
            return "large horizontal panel"
        if aspect_ratio <= 0.33:
            return "large vertical panel"
        return "large broad panel"
    if aspect_ratio >= 2.5:
        return "wide horizontal strip"
    if aspect_ratio <= 0.45:
        return "tall vertical post"
    if fill_ratio >= 0.7:
        return "compact solid feature"
    return "thin irregular patch"


def compute_mask_geometry(mask):
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return None

    ys, xs = np.where(mask)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    bbox_area = bbox_width * bbox_height

    return {
        "pixel_count": pixel_count,
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "bbox_wh": [bbox_width, bbox_height],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "fill_ratio": float(pixel_count) / float(max(bbox_area, 1)),
    }


def summarize_part_visibility(part_colors, view_image_paths):
    part_rgb_colors = np.asarray(part_colors)[:, :3].astype(np.uint8)
    visibility_summary = []
    for i, color in enumerate(part_rgb_colors):
        per_view_pixels = {}
        total_visible_pixels = 0
        visible_views = []
        for view_name, image_path in view_image_paths:
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            pixel_count = int((image_rgb == color).all(axis=-1).sum())
            per_view_pixels[view_name] = pixel_count
            total_visible_pixels += pixel_count
            if pixel_count > 0:
                visible_views.append(view_name)

        visibility_summary.append({
            "part_index": int(i),
            "part_color_rgb": [float(color[0]), float(color[1]), float(color[2])],
            "visible_views": visible_views,
            "total_visible_pixels": total_visible_pixels,
            "per_view_pixels": per_view_pixels,
        })
    return visibility_summary


def summarize_part_geometry(part_colors, view_image_paths):
    part_rgb_colors = np.asarray(part_colors)[:, :3].astype(np.uint8)
    geometry_summary = []
    for i, color in enumerate(part_rgb_colors):
        per_view_geometry = {}
        total_visible_pixels = 0
        visible_views = []
        dominant_view_name = None
        dominant_view_pixels = 0
        for view_name, image_path in view_image_paths:
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            image_height, image_width = image_rgb.shape[:2]
            image_area = image_height * image_width
            mask = (image_rgb == color).all(axis=-1)
            geometry = compute_mask_geometry(mask)
            if geometry is None:
                continue

            pixel_count = geometry["pixel_count"]
            bbox_width, bbox_height = geometry["bbox_wh"]
            area_fraction = float(pixel_count) / float(image_area)
            geometry["area_fraction"] = area_fraction
            geometry["shape_hint"] = infer_shape_hint(
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                fill_ratio=geometry["fill_ratio"],
                area_fraction=area_fraction,
            )
            per_view_geometry[view_name] = geometry
            visible_views.append(view_name)
            total_visible_pixels += pixel_count
            if pixel_count > dominant_view_pixels:
                dominant_view_pixels = pixel_count
                dominant_view_name = view_name

        geometry_summary.append({
            "part_index": int(i),
            "part_color_rgb": [float(color[0]), float(color[1]), float(color[2])],
            "visible_views": visible_views,
            "total_visible_pixels": total_visible_pixels,
            "dominant_view": dominant_view_name,
            "per_view_geometry": per_view_geometry,
        })
    return geometry_summary


def summarize_pairwise_part_relations(part_geometry_summary):
    pairwise_summary = []
    num_parts = len(part_geometry_summary)
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            part_i = part_geometry_summary[i]
            part_j = part_geometry_summary[j]
            per_view_relations = {}
            shared_views = sorted(
                set(part_i["per_view_geometry"].keys()) &
                set(part_j["per_view_geometry"].keys())
            )
            if not shared_views:
                continue

            i_contains_j_views = []
            j_contains_i_views = []
            for view_name in shared_views:
                geom_i = part_i["per_view_geometry"][view_name]
                geom_j = part_j["per_view_geometry"][view_name]

                ix0, iy0, ix1, iy1 = geom_i["bbox_xyxy"]
                jx0, jy0, jx1, jy1 = geom_j["bbox_xyxy"]
                i_contains_j = ix0 <= jx0 and iy0 <= jy0 and ix1 >= jx1 and iy1 >= jy1
                j_contains_i = jx0 <= ix0 and jy0 <= iy0 and jx1 >= ix1 and jy1 >= iy1
                if i_contains_j:
                    i_contains_j_views.append(view_name)
                if j_contains_i:
                    j_contains_i_views.append(view_name)

                cx_i, cy_i = geom_i["centroid_xy"]
                cx_j, cy_j = geom_j["centroid_xy"]
                per_view_relations[view_name] = {
                    "bbox_i_contains_j": i_contains_j,
                    "bbox_j_contains_i": j_contains_i,
                    "centroid_offset_j_minus_i_xy": [
                        float(cx_j - cx_i), float(cy_j - cy_i)
                    ],
                    "pixel_ratio_j_over_i": (
                        float(geom_j["pixel_count"]) / float(max(geom_i["pixel_count"], 1))
                    ),
                }

            observations = []
            if len(i_contains_j_views) == len(shared_views):
                observations.append(
                    f"part {i} encloses part {j} in every shared view"
                )
            if len(j_contains_i_views) == len(shared_views):
                observations.append(
                    f"part {j} encloses part {i} in every shared view"
                )

            pairwise_summary.append({
                "part_i_index": int(i),
                "part_j_index": int(j),
                "shared_views": shared_views,
                "observations": observations,
                "per_view_relations": per_view_relations,
            })
    return pairwise_summary


def _resolve_object_description_context(
    object_description: str,
    object_description_path: Path | None,
    *,
    fallback_name: str,
) -> str:
    if object_description.strip():
        return object_description.strip()
    if object_description_path is not None:
        try:
            description = object_description_text(object_description_path)
            if description.strip():
                return description.strip()
        except Exception:
            text = object_description_path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return fallback_name


@dataclass(frozen=True)
class GPTStagedMaterialInferenceRequest:
    object_description: str
    image_dir: Path
    part_labels_path: Path
    output_path: Path
    object_description_path: Path | None
    processed_image_path: Path
    diagnostic_cues: tuple[dict[str, Any], ...]
    request_payload: dict[str, Any]
    skill_markdown: str = ""
    skill_metadata: dict[str, str] | None = None
    diagnostic_hint_text: str = ""


@dataclass(frozen=True)
class MaterialInferenceContext:
    request: GPTStagedMaterialInferenceRequest
    part_colors: list[list[float]]
    indexed_parts: list[dict[str, Any]]
    view_image_paths: list[tuple[str, str]]
    processed_image_path: Path
    part_visibility_summary: list[dict[str, Any]]
    part_geometry_summary: list[dict[str, Any]]
    pairwise_part_relations: list[dict[str, Any]]


def material_scratch_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.material_inference_scratch.json")


def build_gpt_staged_material_inference_request(
    *,
    image_dir: Path,
    part_labels_path: Path,
    output_path: Path | None,
    object_description: str = "",
    object_description_path: Path | None = None,
    processed_image_path: Path | None = None,
    diagnostic_cues: tuple[dict[str, Any], ...] = (),
    skill_markdown: str = "",
    skill_metadata: dict[str, str] | None = None,
    diagnostic_hint_text: str = "",
) -> GPTStagedMaterialInferenceRequest:
    resolved_image_dir = _path_for_repo(image_dir)
    resolved_part_labels_path = _path_for_repo(part_labels_path)
    resolved_output_path = (
        _path_for_repo(output_path)
        if output_path is not None
        else _path_for_repo(Path("outputs/infer_params") / resolved_image_dir.name / f"{resolved_image_dir.name}.json")
    )
    resolved_object_description_path = (
        _path_for_repo(object_description_path) if object_description_path is not None else None
    )
    resolved_processed_image_path = (
        _path_for_repo(processed_image_path)
        if processed_image_path is not None
        else Path(build_processed_image_path(resolved_image_dir))
    )
    resolved_object_description = _resolve_object_description_context(
        object_description,
        resolved_object_description_path,
        fallback_name=resolved_image_dir.name,
    )
    request_payload: dict[str, Any] = {
        "schema_version": MATERIAL_INFERENCE_SCHEMA_VERSION,
        "image_dir": str(resolved_image_dir),
        "part_labels_path": str(resolved_part_labels_path),
        "output_path": str(resolved_output_path),
        "object_description": resolved_object_description,
        "object_description_path": str(resolved_object_description_path) if resolved_object_description_path else "",
        "processed_image_path": str(resolved_processed_image_path),
        "diagnostic_cues": list(diagnostic_cues),
        "diagnostic_hint_text": diagnostic_hint_text,
        "skill_metadata": dict(skill_metadata) if skill_metadata is not None else {},
    }
    return GPTStagedMaterialInferenceRequest(
        object_description=resolved_object_description,
        image_dir=resolved_image_dir,
        part_labels_path=resolved_part_labels_path,
        output_path=resolved_output_path,
        object_description_path=resolved_object_description_path,
        processed_image_path=resolved_processed_image_path,
        diagnostic_cues=diagnostic_cues,
        request_payload=request_payload,
        skill_markdown=skill_markdown,
        skill_metadata=dict(skill_metadata) if skill_metadata is not None else None,
        diagnostic_hint_text=diagnostic_hint_text,
    )


def load_material_part_labels(request: GPTStagedMaterialInferenceRequest) -> dict[str, Any]:
    part_correspond_dict = np.load(request.part_labels_path, allow_pickle=True)
    if "part_colors" not in part_correspond_dict:
        raise KeyError(f"part labels file is missing `part_colors`: {request.part_labels_path}")
    part_colors = np.asarray(part_correspond_dict["part_colors"])
    if part_colors.ndim != 2 or part_colors.shape[0] == 0 or part_colors.shape[1] < 3:
        raise ValueError(
            f"`part_colors` must be a non-empty Nx3/Nx4 array, got shape {part_colors.shape}"
        )
    for i in range(part_colors.shape[0]):
        print(f"part index: {i}, color (RGB): {part_colors[i]}")
    return {
        "part_colors": part_colors.astype(float).tolist(),
        "part_colors_shape": [int(dim) for dim in part_colors.shape],
        "part_labels_path": str(request.part_labels_path),
    }


def build_material_index_context(
    request: GPTStagedMaterialInferenceRequest,
    part_label_payload: dict[str, Any],
) -> MaterialInferenceContext:
    part_colors = np.asarray(part_label_payload["part_colors"], dtype=float)
    indexed_parts = build_indexed_parts(part_colors.tolist())
    view_image_paths = build_view_image_paths(request.image_dir)
    if not request.processed_image_path.is_file():
        raise FileNotFoundError(
            f"Missing processed reference image: {request.processed_image_path}"
        )
    part_visibility_summary = summarize_part_visibility(part_colors, view_image_paths)
    part_geometry_summary = summarize_part_geometry(part_colors, view_image_paths)
    pairwise_part_relations = summarize_pairwise_part_relations(part_geometry_summary)
    return MaterialInferenceContext(
        request=request,
        part_colors=part_colors.tolist(),
        indexed_parts=indexed_parts,
        view_image_paths=[(str(name), str(path)) for name, path in view_image_paths],
        processed_image_path=request.processed_image_path,
        part_visibility_summary=part_visibility_summary,
        part_geometry_summary=part_geometry_summary,
        pairwise_part_relations=pairwise_part_relations,
    )


def _diagnostic_context(request: GPTStagedMaterialInferenceRequest) -> str:
    diagnostic_context = ""
    if request.diagnostic_cues:
        diagnostic_context = (
            "\nGenesis diagnostic cues to address in this material inference pass: "
            + json.dumps(list(request.diagnostic_cues), indent=2)
            + "\n"
        )
    if request.diagnostic_hint_text.strip():
        diagnostic_context += (
            "\nSelected material_inference diagnostic hints:\n"
            + request.diagnostic_hint_text.strip()
            + "\n"
        )
    return diagnostic_context


def _skill_context(*skill_markdowns: str) -> str:
    chunks = [markdown.strip() for markdown in skill_markdowns if markdown.strip()]
    if not chunks:
        return ""
    return "\nMaterial inference skill markdown:\n" + "\n\n".join(chunks) + "\n"


def build_material_fill_mode_prompt(
    context: MaterialInferenceContext,
    *,
    geometry_result: dict[str, Any],
    canonical_geometry: list[dict[str, Any]],
    skill_markdown: str = "",
) -> str:
    request = context.request
    return (
        "You are deciding per-part volumetric topology for HAG4R material inference after geometry analysis and before material semantics or numeric parameter estimation.\n"
        f"{_skill_context(skill_markdown)}"
        f"Object-level context: {request.object_description}.\n"
        f"{_diagnostic_context(request)}"
        f"Indexed part mapping: {json.dumps(context.indexed_parts)}.\n"
        f"Observed part visibility: {json.dumps(context.part_visibility_summary)}.\n"
        f"Observed geometry summary: {json.dumps(context.part_geometry_summary)}.\n"
        f"Observed pairwise part relations: {json.dumps(context.pairwise_part_relations)}.\n"
        f"Prior geometry analysis is: {json.dumps({'inferred_object_name': geometry_result['inferred_object_name'], 'parts': canonical_geometry})}.\n"
        "The pipeline is volumetric-only. Decide `volume_fill_mode` separately for every indexed part.\n"
        "Choose `solid_fill` when the part should be represented as filled material in a tetrahedral volume.\n"
        "Choose `hollow_wall` when the part should remain a hollow volumetric wall or shell-like structure with an empty cavity or open interior.\n"
        "Do not estimate thickness. Hollow geometry is derived later from fixed voxel-band layers and voxel pitch.\n"
        "Preserve the exact `part_index` and `part_color_rgb` from the indexed mapping.\n"
        "Return only the fill-mode JSON with schema_version, inferred_object_name, and parts.\n"
        "Do not infer final material semantics, density, Young's modulus, Poisson ratio, friction, or material names in this phase.\n"
    )


def material_validation_attempt(
    *,
    status: str,
    error: str = "",
    stage: str = "material_validation",
    attempt_index: int | None = None,
) -> dict[str, Any]:
    attempt = {
        "status": str(status),
        "stage": str(stage),
        "error": str(error),
    }
    if attempt_index is not None:
        attempt["attempt_index"] = int(attempt_index)
    return attempt


def validate_agent_material_fill_mode_payload(
    payload: dict[str, Any],
    context: MaterialInferenceContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("agent material fill-mode payload must be a JSON object")
    canonical_fill_modes = validate_and_canonicalize_fill_modes(payload, context.indexed_parts)
    normalized = dict(payload)
    normalized["parts"] = canonical_fill_modes
    return normalized, canonical_fill_modes


def validate_agent_material_geometry_payload(
    payload: dict[str, Any],
    context: MaterialInferenceContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("agent material geometry payload must be a JSON object")
    canonical_geometry = validate_and_canonicalize_geometry(payload, context.indexed_parts)
    _validate_geometry_required_fields(canonical_geometry)
    normalized = dict(payload)
    normalized["parts"] = canonical_geometry
    return normalized, canonical_geometry


def validate_agent_material_semantics_payload(
    payload: dict[str, Any],
    context: MaterialInferenceContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("agent material semantics payload must be a JSON object")
    canonical_parts = validate_and_canonicalize_parts(payload, context.indexed_parts)
    normalized = dict(payload)
    normalized["parts"] = canonical_parts
    return normalized, canonical_parts


def validate_agent_material_repair_payload(
    payload: dict[str, Any],
    context: MaterialInferenceContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("agent material repair payload must be a JSON object")
    canonical_parts = validate_and_canonicalize_parts(payload, context.indexed_parts)
    normalized = dict(payload)
    normalized["parts"] = canonical_parts
    return normalized, canonical_parts


def _material_parameter_worked_example() -> str:
    return (
        "Worked material-parameter example for output style only; do not copy the object name, part count, or values unless the material and geometry match.\n"
        "Example object: toilet plunger with two real indexed parts.\n"
        "Valid example output:\n"
        "{\n"
        '  "inferred_object_name": "toilet plunger",\n'
        '  "parts": [\n'
        "    {\n"
        '      "part_index": 0,\n'
        '      "part_name": "painted_wood_handle",\n'
        '      "part_semantics": "rigid long handle used to push and pull the plunger",\n'
        '      "part_texture": "smooth painted wood with light grip wear",\n'
        '      "major_material_name": "wood",\n'
        '      "part_color_rgb": [141, 211, 199],\n'
        '      "density_kg_m3": 650,\n'
        '      "youngs_modulus_pa": 9000000000,\n'
        '      "poisson_ratio": 0.32,\n'
        '      "friction_coefficient": 0.45\n'
        "    },\n"
        "    {\n"
        '      "part_index": 1,\n'
        '      "part_name": "rubber_suction_cup",\n'
        '      "part_semantics": "flexible cup that seals against a surface and deforms during plunging",\n'
        '      "part_texture": "matte slightly tacky rubber",\n'
        '      "major_material_name": "rubber",\n'
        '      "part_color_rgb": [255, 255, 179],\n'
        '      "density_kg_m3": 1100,\n'
        '      "youngs_modulus_pa": 8000000,\n'
        '      "poisson_ratio": 0.49,\n'
        '      "friction_coefficient": 1.05\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Pattern to copy: every real indexed part has a physical material, positive density, positive Young's modulus, a dot-decimal Poisson ratio below 0.5, and material-consistent friction.\n"
    )


def _material_profile_bank_text() -> str:
    return (
        "Material Profile Bank for numeric fields. When uncertain, first choose the closest positive profile row, then copy all four numeric fields from that row into the JSON before adjusting only if visual evidence strongly supports a different positive value. Zero is not an uncertainty marker.\n"
        "- `thin_plastic_sheet`: density_kg_m3 1050, youngs_modulus_pa 2000000000, poisson_ratio 0.38, friction_coefficient 0.35.\n"
        "- `rigid_plastic_hub`: density_kg_m3 1100, youngs_modulus_pa 2200000000, poisson_ratio 0.37, friction_coefficient 0.35.\n"
        "- `wooden_stick`: density_kg_m3 650, youngs_modulus_pa 9000000000, poisson_ratio 0.35, friction_coefficient 0.45.\n"
        "- `paper_cardboard_sheet`: density_kg_m3 700, youngs_modulus_pa 3000000000, poisson_ratio 0.30, friction_coefficient 0.55.\n"
        "- `rubber_flexible`: density_kg_m3 1100, youngs_modulus_pa 5000000, poisson_ratio 0.49, friction_coefficient 0.90.\n"
        "- `metal_connector`: density_kg_m3 7850, youngs_modulus_pa 200000000000, poisson_ratio 0.30, friction_coefficient 0.40.\n"
        "Do not free-form invent numeric values when uncertain. Select the closest profile row and copy the full positive tuple for density, Young's modulus, Poisson ratio, and friction.\n"
    )


def _profile_tuple_binding_instruction_text() -> str:
    return (
        "Schema-compatible tuple binding requirement: for every part, end the `part_semantics` string with a binding clause of the form "
        "`profile=<profile_slug>; tuple density_kg_m3=<d>, youngs_modulus_pa=<E>, poisson_ratio=<nu>, friction_coefficient=<mu>`.\n"
        "Then copy those exact tuple values into that part's numeric fields. Do not write a positive tuple in `part_semantics` while leaving any numeric field at zero.\n"
        "Example schema-valid binding: `curved thin pinwheel blade panel; profile=thin_plastic_sheet; tuple density_kg_m3=1050, youngs_modulus_pa=2000000000, poisson_ratio=0.38, friction_coefficient=0.35`.\n"
        "No extra output outside the JSON is allowed; the binding clause must live inside the existing `part_semantics` string.\n"
    )


def build_material_geometry_prompt(
    context: MaterialInferenceContext,
    *,
    skill_markdown: str = "",
) -> str:
    request = context.request
    image_name = request.image_dir.name
    return (
        "You are a geometry analyst for segmented multi-view object images.\n"
        f"{_skill_context(skill_markdown)}"
        f"Object-level context: {request.object_description}.\n"
        f"{_diagnostic_context(request)}"
        f"Given six segmented object images of {image_name} from views {', '.join(VIEW_NAMES)}, and an indexed part-color mapping, infer only the visible geometry of each indexed part.\n"
        f"The attached images are ordered exactly as: {', '.join(VIEW_NAMES)}.\n"
        f"The required indexed mapping is: {json.dumps(context.indexed_parts)}.\n"
        f"Observed part visibility across the six images is: {json.dumps(context.part_visibility_summary)}.\n"
        f"Observed part geometry from the segmentation masks is: {json.dumps(context.part_geometry_summary)}.\n"
        f"Observed pairwise relations between parts is: {json.dumps(context.pairwise_part_relations)}.\n"
        "You must output exactly one `parts` item for each mapping entry, and each item must preserve the exact `part_index` and `part_color_rgb` from that mapping.\n"
        "The six images are different segmented views of the same object. Use information accumulated across all views when identifying each part.\n"
        "Some parts may be broad panels, plates, strips, posts, stems, shells, loops, caps, or compact solids. Infer the geometry from the visible masks and the relative placement only.\n"
        "Do not infer material properties yet. Do not assign object semantics yet beyond geometric attachment or support patterns.\n"
        "Return the likely object name in `inferred_object_name`, but focus each part entry on geometry only.\n"
        "For each part, infer:\n"
        "- `geometric_category`: concise geometry label\n"
        "- `geometry_description`: short description of shape and placement across views\n"
        "- `relative_size`: dominant / medium / small / tiny, or another concise size descriptor\n"
        "- `attachment_pattern`: how the part sits relative to the other parts (for example flush panel, raised cap on panel, vertical post under cap, side strap, nested insert)\n"
    )


def build_material_semantics_prompt(
    context: MaterialInferenceContext,
    *,
    geometry_result: dict[str, Any],
    canonical_geometry: list[dict[str, Any]],
    fill_mode_result: dict[str, Any],
    canonical_fill_modes: list[dict[str, Any]],
    material_semantics_skill: str = "",
    physical_parameter_skill: str = "",
) -> str:
    request = context.request
    image_name = request.image_dir.name
    return (
        "You are a material-property estimator for robotics simulation.\n"
        f"{_skill_context(material_semantics_skill, physical_parameter_skill)}"
        f"Object-level context: {request.object_description}.\n"
        f"{_diagnostic_context(request)}"
        f"Given six segmented reconstructed object images of {image_name} from views {', '.join(VIEW_NAMES)}, one processed original reference image, an indexed part-color mapping, and a prior geometry analysis, infer semantics and constitutive parameters for each part.\n"
        f"The first six attached images are ordered exactly as: {', '.join(VIEW_NAMES)}.\n"
        f"The final attached image is {os.path.basename(str(context.processed_image_path))}, which is a background-cropped single view of the original object rather than the reconstructed object.\n"
        f"The required indexed mapping is: {json.dumps(context.indexed_parts)}.\n"
        f"Observed part visibility across the six images is: {json.dumps(context.part_visibility_summary)}.\n"
        f"Observed part geometry from the segmentation masks is: {json.dumps(context.part_geometry_summary)}.\n"
        f"Observed pairwise relations between parts is: {json.dumps(context.pairwise_part_relations)}.\n"
        f"Prior geometry analysis is: {json.dumps({'inferred_object_name': geometry_result['inferred_object_name'], 'parts': canonical_geometry})}.\n"
        f"Fixed fill-mode decision is: {json.dumps({'inferred_object_name': fill_mode_result['inferred_object_name'], 'parts': canonical_fill_modes})}.\n"
        "You must output exactly one `parts` item for each mapping entry, and each item must preserve the exact `part_index` and `part_color_rgb` from that mapping.\n"
        "The fill-mode decision is fixed context. Do not output, edit, or reinterpret `volume_fill_mode`, `fill_mode_rationale`, or `fill_mode_evidence` in this material payload.\n"
        "Do not estimate thickness. Hollow geometry is derived later from fixed voxel-band layers and voxel pitch, not from material inference.\n"
        "Use the six segmented reconstructed views and the prior geometry analysis as the primary evidence for part coverage, geometry, and relative placement.\n"
        "Use the processed original reference image as appearance evidence only. It may omit some reconstructed parts, show only one side of the object, or merge multiple reconstructed parts into a single visible surface.\n"
        "For each part, infer a reasonable major surface texture in `part_texture` from the processed original reference image when visible; when the part is missing or ambiguous in that image, extrapolate conservatively from visible object surfaces, part role, and object-level style.\n"
        "Infer part semantics from the combination of multi-view geometry, relative placement, inferred texture, color, and object-level context.\n"
        "If a part is described as a cap/head/plate and another part is described as the post/stem/support beneath it, keep the higher-level exposed component as the main semantic head and the smaller support as the subordinate connector component.\n"
        "When a part is tiny, unobserved, or not clearly visible, do not label it as void/artifact/air/none and do not use zero-valued properties. Treat it as a real indexed mesh part that will receive volumetric tetrahedra downstream, guess the most plausible physical material from neighboring visible parts, object-level context, geometry, and attachment pattern, then infer material parameters from that guessed material.\n"
        "For unobserved reconstructed parts, first decide the likely role and major material by analogy to the nearest or parent visible object surface; if uncertain, choose the dominant compatible material of the object rather than emitting an unknown material.\n"
        "After inferring semantics and texture for a part, infer the single major material used to compose that part in `major_material_name` from the combination of part semantics, inferred texture, and geometry.\n"
        "Then infer the constitutive parameters from the chosen material name, while keeping the values physically plausible for that material and compatible with the part geometry.\n"
        "This is a combined material semantics and numeric estimation call: even if standalone semantics instructions avoid final numbers, this tool must return final numeric candidate values for every indexed part.\n"
        "Every indexed mapping entry is a real mesh part for this stage. Tiny, hidden, merged, ambiguous, or weakly visible parts still need a guessed physical material and valid positive numeric values.\n"
        "Express uncertainty through conservative material choice and existing text fields in the schema, not by zeros, invalid sentinels, unknown/void/artifact/air/none material names, or placeholder numerics.\n"
        "All indexed parts must have strictly positive `youngs_modulus_pa` and `density_kg_m3`; Genesis treats zero or negative values as invalid. "
        "`poisson_ratio` must be greater than -1 and less than 0.5. `friction_coefficient` must be non-negative.\n"
        "Before returning JSON, silently run this pre-submit numeric check on every part: `density_kg_m3 > 0`, `youngs_modulus_pa > 0`, `-1 < poisson_ratio < 0.5`, and `friction_coefficient >= 0`; for real contact surfaces, prefer `friction_coefficient > 0`.\n"
        "Do not output a separate checklist, table, or extra fields; the schema only allows the requested JSON fields.\n"
        "`poisson_ratio: 0.5` is invalid. For near-incompressible rubber or silicone, choose a valid value below `0.5`, such as `0.48` or `0.49`.\n"
        "Use JSON numeric literals with period decimals only: `0.35`, not `0,35`; `poisson_ratio: 0,5` is invalid JSON and must never appear.\n"
        "Forbidden placeholder cluster: do not output `major_material_name: plastic` with `density_kg_m3: 1050`, `youngs_modulus_pa: 0`, `poisson_ratio: 0,5`, and `friction_coefficient: 0` for any real indexed part; replace the entire cluster with material-consistent JSON-valid values.\n"
        "When a draft resembles that cluster, do not patch one field at a time; reselect the material hypothesis for the part and replace density, Young's modulus, Poisson ratio, and friction together.\n"
        f"{_material_profile_bank_text()}"
        f"{_profile_tuple_binding_instruction_text()}"
        f"{_material_parameter_worked_example()}"
        "Examples: paper/cardboard thin panels, including paper pinwheel blades, need positive density, positive Young's modulus, valid Poisson ratio, and nonzero contact friction; fabric/foam or soft compliant parts need positive density and modulus with no zero stiffness; plastic visible bodies, caps, and panels need positive plastic-like values; hidden or ambiguous connector and central support parts should be guessed as plastic or metal unless evidence says otherwise, with positive modulus.\n"
        "For thin folded paper pinwheel blades, prefer paper/cardboard ranges unless the processed reference appearance clearly supports molded plastic; never use plastic-plus-zero fields as an uncertainty shortcut.\n"
        "Return the object name in `inferred_object_name`.\n"
        "Return realistic values for:\n"
        "- part_texture\n"
        "- density_kg_m3\n"
        "- youngs_modulus_pa\n"
        "- poisson_ratio\n"
        "- friction_coefficient\n"
    )


def build_material_repair_prompt(
    context: MaterialInferenceContext,
    *,
    validation_error: str,
    material_result: dict[str, Any],
    geometry_result: dict[str, Any],
    canonical_geometry: list[dict[str, Any]],
    fill_mode_result: dict[str, Any],
    canonical_fill_modes: list[dict[str, Any]],
    material_validation_repair_skill: str = "",
    physical_parameter_skill: str = "",
) -> str:
    return (
        "You are repairing a schema-shaped HAG4R material inference JSON object that failed physical validation.\n"
        f"{_skill_context(material_validation_repair_skill, physical_parameter_skill)}"
        f"Object-level context: {context.request.object_description}.\n"
        f"Validation error: {validation_error}\n"
        f"Required indexed mapping, exact and immutable: {json.dumps(context.indexed_parts)}\n"
        f"Geometry analysis: {json.dumps({'inferred_object_name': geometry_result.get('inferred_object_name', ''), 'parts': canonical_geometry})}\n"
        f"Fixed fill-mode decision: {json.dumps({'inferred_object_name': fill_mode_result.get('inferred_object_name', ''), 'parts': canonical_fill_modes})}\n"
        f"Previous material output to repair: {json.dumps(material_result)}\n"
        "Return a full replacement JSON object with the same schema as the failed material output.\n"
        "Preserve every `part_index` and `part_color_rgb` exactly. Do not add, remove, merge, split, or reorder indexed parts.\n"
        "Do not output, edit, or reinterpret `volume_fill_mode`, `fill_mode_rationale`, or `fill_mode_evidence`; repair cannot change fill mode.\n"
        "Do not estimate thickness. Hollow geometry is derived later from fixed voxel-band layers and voxel pitch.\n"
        "This is the last agent-side chance to return a valid replacement; preserve the exact part index/color mapping while fixing the physical values now.\n"
        "For every invalid physical value, choose a physically plausible value from the guessed material and part role. Never preserve zero for density or Young's modulus.\n"
        "Do not preserve placeholder zeros, negative values, or boundary Poisson values such as `poisson_ratio: 0.5`.\n"
        "Use JSON numeric literals with period decimals only: `0.35`, not `0,35`; `poisson_ratio: 0,5` is invalid JSON and must never appear.\n"
        "Forbidden placeholder cluster: do not output `major_material_name: plastic` with `density_kg_m3: 1050`, `youngs_modulus_pa: 0`, `poisson_ratio: 0,5`, and `friction_coefficient: 0`; replace the entire cluster with material-consistent JSON-valid values.\n"
        "When repairing a draft that resembles that cluster, do not keep `plastic` plus zero physical fields; reselect a coherent material hypothesis and replace density, Young's modulus, Poisson ratio, and friction together.\n"
        f"{_material_profile_bank_text()}"
        f"{_profile_tuple_binding_instruction_text()}"
        f"{_material_parameter_worked_example()}"
        "When repairing any invalid numeric field, replace both the `part_semantics` binding clause and all four numeric fields from one positive profile row.\n"
        "If a material name is void, artifact, unknown, air, or none, replace it with a guessed physical material using geometry, neighboring parts, object context, and appearance.\n"
        "Before returning, silently run the same numeric preflight on every part and revise any failing field.\n"
        "After repair, every part must pass these constraints: density_kg_m3 > 0, youngs_modulus_pa > 0, -1 < poisson_ratio < 0.5, friction_coefficient >= 0.\n"
    )


def validate_repair_material_predictions(
    context: MaterialInferenceContext,
    *,
    material_result: dict[str, Any],
    repair_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        canonical_parts = validate_and_canonicalize_parts(material_result, context.indexed_parts)
        return material_result, canonical_parts, [
            material_validation_attempt(status="success", stage="material_validation")
        ]
    except Exception as first_err:
        attempts = [
            material_validation_attempt(status="failed", stage="material_validation", error=str(first_err))
        ]
        if repair_payload is None:
            raise ValueError(
                "Material output failed validation and no Codex-authored repair payload was provided. "
                f"Validation error: {first_err}"
            ) from first_err
        repaired, canonical_parts = validate_agent_material_repair_payload(repair_payload, context)
        attempts.append(
            material_validation_attempt(status="success", stage="constitutive_parameters_repair", attempt_index=1)
        )
        return repaired, canonical_parts, attempts


def build_material_output_payload(
    context: MaterialInferenceContext,
    *,
    geometry_result: dict[str, Any],
    canonical_geometry: list[dict[str, Any]],
    geometry_stage_notes: list[Any],
    geometry_stage_attempts: list[dict[str, Any]],
    fill_mode_result: dict[str, Any],
    canonical_fill_modes: list[dict[str, Any]],
    fill_mode_stage_notes: list[Any],
    fill_mode_stage_attempts: list[dict[str, Any]],
    material_result: dict[str, Any],
    canonical_parts: list[dict[str, Any]],
    semantics_stage_notes: list[Any],
    semantics_stage_attempts: list[dict[str, Any]],
    repair_stage_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(canonical_parts) != len(canonical_fill_modes):
        raise ValueError("final material output requires aligned material and fill-mode entries")
    merged_predictions = []
    for material_part, fill_mode_part in zip(canonical_parts, canonical_fill_modes, strict=True):
        if material_part["part_index"] != fill_mode_part["part_index"]:
            raise ValueError(
                "final material output material/fill-mode part_index mismatch: "
                f"{material_part['part_index']} != {fill_mode_part['part_index']}"
            )
        merged = dict(material_part)
        merged["volume_fill_mode"] = fill_mode_part["volume_fill_mode"]
        merged["fill_mode_rationale"] = fill_mode_part["fill_mode_rationale"]
        merged["fill_mode_evidence"] = list(fill_mode_part["fill_mode_evidence"])
        merged_predictions.append(merged)
    top_image_path = context.view_image_paths[0][1]
    stage_attempts: dict[str, Any] = {
        "geometry": geometry_stage_attempts,
        "fill_mode": fill_mode_stage_attempts,
        "semantics": semantics_stage_attempts,
    }
    if repair_stage_attempts:
        stage_attempts["repair"] = repair_stage_attempts
    return {
        "schema_version": MATERIAL_INFERENCE_SCHEMA_VERSION,
        "image_path": str(top_image_path),
        "processed_image_path": str(context.processed_image_path),
        "part_colors_path": str(context.request.part_labels_path),
        "num_parts": int(len(context.indexed_parts)),
        "inferred_object_name": material_result["inferred_object_name"],
        "geometry_inference": {
            "inferred_object_name": geometry_result["inferred_object_name"],
            "predictions": canonical_geometry,
        },
        "fill_mode_decision": {
            "schema_version": fill_mode_result["schema_version"],
            "inferred_object_name": fill_mode_result["inferred_object_name"],
            "predictions": canonical_fill_modes,
        },
        "geometry_stage_notes": geometry_stage_notes,
        "fill_mode_stage_notes": fill_mode_stage_notes,
        "semantics_stage_notes": semantics_stage_notes,
        "stage_attempts": stage_attempts,
        "predictions": merged_predictions,
    }


def write_material_output_payload(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"output_path": str(output_path), "num_parts": int(payload.get("num_parts", 0))}


def material_inference_request_payload(
    *,
    image_dir: str,
    part_labels_path: str,
    output_path: str,
    object_description: str = "",
    object_description_path: str | None = None,
    processed_image_path: str | None = None,
    diagnostic_cues: tuple[dict[str, Any], ...] = (),
    skill_markdown: str = "",
    skill_metadata: dict[str, str] | None = None,
    diagnostic_hint_text: str = "",
) -> dict[str, Any]:
    request = build_gpt_staged_material_inference_request(
        image_dir=Path(image_dir),
        part_labels_path=Path(part_labels_path),
        output_path=Path(output_path),
        object_description=object_description,
        object_description_path=Path(object_description_path) if object_description_path else None,
        processed_image_path=Path(processed_image_path) if processed_image_path else None,
        diagnostic_cues=diagnostic_cues,
        skill_markdown=skill_markdown,
        skill_metadata=skill_metadata,
        diagnostic_hint_text=diagnostic_hint_text,
    )
    payload = to_json_dict(request)
    payload["status"] = "ready"
    return payload


__all__ = [
    "GPTStagedMaterialInferenceRequest",
    "MATERIAL_FILL_MODE_DECISION_SCHEMA_VERSION",
    "MATERIAL_INFERENCE_SCHEMA_VERSION",
    "MaterialInferenceContext",
    "VOLUME_FILL_MODES",
    "build_fill_mode_locked_schema",
    "build_gpt_staged_material_inference_request",
    "build_material_index_context",
    "build_material_fill_mode_prompt",
    "build_material_output_payload",
    "load_material_part_labels",
    "material_validation_attempt",
    "material_inference_request_payload",
    "material_scratch_path_for_output",
    "validate_agent_material_fill_mode_payload",
    "validate_agent_material_geometry_payload",
    "validate_agent_material_repair_payload",
    "validate_agent_material_semantics_payload",
    "validate_and_canonicalize_fill_modes",
    "validate_repair_material_predictions",
    "write_material_output_payload",
]
