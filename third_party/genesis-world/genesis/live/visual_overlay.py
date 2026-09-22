from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .protocol import GenesisLiveError


VISUAL_OVERLAY_SCHEMA_VERSION = "hag4r-genesis-visual-overlay-v1"
VISUAL_OVERLAY_ASSET_ROOT_ENV = "GENESIS_LIVE_ASSET_ROOT"
VISUAL_OVERLAY_BINDING_KEY = "physics_vertex_indices"
VISUAL_OVERLAY_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "mesh_file",
        "binding_file",
        "binding_key",
        "hide_physics_rgb",
    }
)
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
VISUAL_OVERLAY_REST_ATOL_M = 1.0e-6


@dataclass(frozen=True)
class VisualOverlaySpec:
    entity_index: int
    entity_name: str
    mesh_file: str
    binding_file: str


@dataclass(frozen=True)
class VisualOverlayAssets:
    spec: VisualOverlaySpec
    mesh: trimesh.Trimesh
    visual_rest_vertices_m: np.ndarray
    visual_faces: np.ndarray
    physics_vertex_indices: np.ndarray


def _overlay_error(
    message: str,
    *,
    entity_index: int,
    entity_name: str,
    field: str | None = None,
) -> GenesisLiveError:
    details: dict[str, Any] = {
        "entity_index": int(entity_index),
        "entity_name": entity_name,
    }
    if field is not None:
        details["field"] = field
    return GenesisLiveError("invalid_visual_overlay", message, details=details)


def validate_visual_overlay_spec(
    entity_cfg: dict[str, Any],
    *,
    entity_index: int,
) -> VisualOverlaySpec | None:
    overlay_cfg = entity_cfg.get("visual_overlay")
    if overlay_cfg is None:
        return None

    entity_name = str(entity_cfg.get("name") or f"entity_{entity_index}")
    if not isinstance(overlay_cfg, dict):
        raise _overlay_error(
            "visual_overlay must be an object",
            entity_index=entity_index,
            entity_name=entity_name,
        )
    if set(overlay_cfg) != VISUAL_OVERLAY_CONFIG_KEYS:
        raise _overlay_error(
            "visual_overlay fields do not match the frozen schema",
            entity_index=entity_index,
            entity_name=entity_name,
        )
    if overlay_cfg["schema_version"] != VISUAL_OVERLAY_SCHEMA_VERSION:
        raise _overlay_error(
            "visual_overlay schema_version is unsupported",
            entity_index=entity_index,
            entity_name=entity_name,
            field="schema_version",
        )
    if overlay_cfg["binding_key"] != VISUAL_OVERLAY_BINDING_KEY:
        raise _overlay_error(
            "visual_overlay binding_key is unsupported",
            entity_index=entity_index,
            entity_name=entity_name,
            field="binding_key",
        )
    if overlay_cfg["hide_physics_rgb"] is not True:
        raise _overlay_error(
            "visual_overlay hide_physics_rgb must be true",
            entity_index=entity_index,
            entity_name=entity_name,
            field="hide_physics_rgb",
        )

    morph_cfg = entity_cfg.get("morph", {})
    material_cfg = entity_cfg.get("material", {})
    if not isinstance(morph_cfg, dict) or morph_cfg.get("type") != "tet_mesh":
        raise _overlay_error(
            "visual_overlay owner must use morph.type='tet_mesh'",
            entity_index=entity_index,
            entity_name=entity_name,
        )
    if not isinstance(material_cfg, dict) or material_cfg.get("type", "elastic") != "elastic":
        raise _overlay_error(
            "visual_overlay owner must use material.type='elastic'",
            entity_index=entity_index,
            entity_name=entity_name,
        )
    forbidden_transforms = sorted({"scale", "pos"}.intersection(morph_cfg))
    if forbidden_transforms:
        raise _overlay_error(
            "visual_overlay owner morph may not define scale or pos",
            entity_index=entity_index,
            entity_name=entity_name,
        )

    configured_paths: dict[str, str] = {}
    for field in ("mesh_file", "binding_file"):
        value = overlay_cfg[field]
        if not isinstance(value, str) or not value:
            raise _overlay_error(
                f"visual_overlay {field} must be a non-empty string",
                entity_index=entity_index,
                entity_name=entity_name,
                field=field,
            )
        configured_paths[field] = value
    return VisualOverlaySpec(
        entity_index=int(entity_index),
        entity_name=entity_name,
        mesh_file=configured_paths["mesh_file"],
        binding_file=configured_paths["binding_file"],
    )


def resolve_visual_overlay_asset_path(
    configured_path: str,
    *,
    field: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    if not isinstance(configured_path, str) or not configured_path:
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"visual_overlay {field} must be a non-empty string",
            details={"field": field},
        )
    raw_parts = configured_path.split("/")
    configured = Path(configured_path)
    if (
        configured.is_absolute()
        or not raw_parts
        or raw_parts[0] != "outputs"
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"visual_overlay {field} must be a repo-relative outputs/... path",
            details={"field": field},
        )

    environment = os.environ if environ is None else environ
    root_value = environment.get(VISUAL_OVERLAY_ASSET_ROOT_ENV)
    if not isinstance(root_value, str) or not root_value:
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"{VISUAL_OVERLAY_ASSET_ROOT_ENV} must identify an absolute asset directory",
            details={"field": field},
        )
    root = Path(root_value)
    if not root.is_absolute() or not root.exists() or not root.is_dir():
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"{VISUAL_OVERLAY_ASSET_ROOT_ENV} must identify an absolute asset directory",
            details={"field": field},
        )
    resolved_root = root.resolve()
    lexical_outputs_root = resolved_root / "outputs"
    try:
        resolved_outputs_root = lexical_outputs_root.resolve(strict=True)
        resolved_target = (resolved_root / configured).resolve(strict=True)
    except OSError as exc:
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"visual_overlay {field} does not identify an existing file",
            details={"field": field},
        ) from exc
    if not resolved_outputs_root.is_dir():
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"{VISUAL_OVERLAY_ASSET_ROOT_ENV}/outputs must identify a directory",
            details={"field": field},
        )
    if not resolved_target.is_relative_to(resolved_outputs_root):
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"visual_overlay {field} escapes the configured asset outputs root",
            details={"field": field},
        )
    if not resolved_target.is_file():
        raise GenesisLiveError(
            "invalid_visual_overlay",
            f"visual_overlay {field} must identify a regular file",
            details={"field": field},
        )
    return resolved_target


def _glb_json_document(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError("visual overlay mesh is not a binary GLB container")
    version = int.from_bytes(data[4:8], "little")
    declared_length = int.from_bytes(data[8:12], "little")
    if version != 2 or declared_length != len(data):
        raise ValueError("visual overlay GLB header is invalid")
    cursor = 12
    json_document: dict[str, Any] | None = None
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("visual overlay GLB chunk header is truncated")
        chunk_length = int.from_bytes(data[cursor : cursor + 4], "little")
        chunk_type = data[cursor + 4 : cursor + 8]
        cursor += 8
        chunk = data[cursor : cursor + chunk_length]
        if len(chunk) != chunk_length:
            raise ValueError("visual overlay GLB chunk is truncated")
        cursor += chunk_length
        if chunk_type == b"JSON":
            if json_document is not None:
                raise ValueError("visual overlay GLB contains multiple JSON chunks")
            parsed = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("visual overlay GLB JSON chunk must contain an object")
            json_document = parsed
    if cursor != len(data) or json_document is None:
        raise ValueError("visual overlay GLB does not contain one valid JSON chunk")
    return json_document


def _validate_textured_single_primitive_glb(document: dict[str, Any]) -> None:
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise ValueError("visual overlay GLB must contain exactly one mesh")
    primitives = meshes[0].get("primitives") if isinstance(meshes[0], dict) else None
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise ValueError("visual overlay GLB must contain exactly one primitive")
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
        raise ValueError("visual overlay GLB primitive must reference a material")
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness") if isinstance(material, dict) else None
    base_color_texture = pbr.get("baseColorTexture") if isinstance(pbr, dict) else None
    if not isinstance(base_color_texture, dict):
        raise ValueError("visual overlay GLB material must have a base-color texture")
    texture_index = base_color_texture.get("index")
    textures = document.get("textures")
    if (
        isinstance(texture_index, bool)
        or not isinstance(texture_index, int)
        or not isinstance(textures, list)
        or texture_index < 0
        or texture_index >= len(textures)
    ):
        raise ValueError("visual overlay GLB base-color texture reference is invalid")
    texture_record = textures[texture_index]
    if not isinstance(texture_record, dict):
        raise ValueError("visual overlay GLB texture record must be an object")
    image_index = texture_record.get("source")
    images = document.get("images")
    if (
        isinstance(image_index, bool)
        or not isinstance(image_index, int)
        or not isinstance(images, list)
        or image_index < 0
        or image_index >= len(images)
    ):
        raise ValueError("visual overlay GLB image reference is invalid")
    image_record = images[image_index]
    if not isinstance(image_record, dict) or (
        "bufferView" not in image_record and not str(image_record.get("uri", "")).startswith("data:")
    ):
        raise ValueError("visual overlay GLB image must be embedded")
    if "uri" in image_record and not str(image_record["uri"]).startswith("data:"):
        raise ValueError("visual overlay GLB may not reference an external image")


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype,
    trailing_shape: tuple[int, ...],
    length: int | None = None,
) -> np.ndarray:
    array = arrays[name]
    if array.dtype != dtype or array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape:
        raise ValueError(f"visual overlay binding array {name} has the wrong dtype or shape")
    if array.shape[0] == 0:
        raise ValueError(f"visual overlay binding array {name} must be non-empty")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"visual overlay binding array {name} has the wrong length")
    return array


def _readonly_c_array(array: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _load_and_validate_binding(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(VISUAL_TO_PHYSICS_ARRAY_KEYS):
            raise ValueError("visual overlay binding keys do not match the frozen eight-array schema")
        arrays = {name: np.asarray(archive[name]) for name in VISUAL_TO_PHYSICS_ARRAY_KEYS}

    rest = _require_array(
        arrays,
        "visual_rest_vertices_m",
        dtype=np.dtype(np.float32),
        trailing_shape=(3,),
    )
    visual_count = len(rest)
    faces = _require_array(
        arrays,
        "visual_faces",
        dtype=np.dtype(np.int32),
        trailing_shape=(3,),
    )
    face_count = len(faces)
    uv = _require_array(
        arrays,
        "visual_uv",
        dtype=np.dtype(np.float32),
        trailing_shape=(2,),
        length=visual_count,
    )
    surface_to_physics = _require_array(
        arrays,
        "surface_vertex_to_physics_vertex",
        dtype=np.dtype(np.int64),
        trailing_shape=(),
    )
    visual_to_surface = _require_array(
        arrays,
        "visual_vertex_to_surface_vertex",
        dtype=np.dtype(np.int64),
        trailing_shape=(),
        length=visual_count,
    )
    physics_indices = _require_array(
        arrays,
        "physics_vertex_indices",
        dtype=np.dtype(np.int64),
        trailing_shape=(),
        length=visual_count,
    )
    boundary_tets = _require_array(
        arrays,
        "boundary_face_tet_indices",
        dtype=np.dtype(np.int64),
        trailing_shape=(),
        length=face_count,
    )
    boundary_parts = _require_array(
        arrays,
        "boundary_face_part_ids",
        dtype=np.dtype(np.int32),
        trailing_shape=(),
        length=face_count,
    )

    if not np.all(np.isfinite(rest)) or not np.all(np.isfinite(uv)):
        raise ValueError("visual overlay binding contains non-finite floating-point values")
    if np.any(faces < 0) or np.any(faces >= visual_count):
        raise ValueError("visual overlay binding faces contain an out-of-range vertex index")
    if (
        np.any(surface_to_physics < 0)
        or np.any(visual_to_surface < 0)
        or np.any(visual_to_surface >= len(surface_to_physics))
        or np.any(physics_indices < 0)
        or np.any(boundary_tets < 0)
        or np.any(boundary_parts < 0)
    ):
        raise ValueError("visual overlay binding contains an out-of-range index")
    if not np.array_equal(physics_indices, surface_to_physics[visual_to_surface]):
        raise ValueError("visual overlay physics indices do not match the exact composed binding")
    return arrays


def load_visual_overlay_assets(
    spec: VisualOverlaySpec,
    *,
    environ: Mapping[str, str] | None = None,
) -> VisualOverlayAssets:
    try:
        mesh_path = resolve_visual_overlay_asset_path(
            spec.mesh_file,
            field="mesh_file",
            environ=environ,
        )
        binding_path = resolve_visual_overlay_asset_path(
            spec.binding_file,
            field="binding_file",
            environ=environ,
        )
        document = _glb_json_document(mesh_path)
        _validate_textured_single_primitive_glb(document)
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("visual overlay GLB did not load as one Trimesh")
        loaded_vertices = np.asarray(loaded.vertices)
        loaded_faces = np.asarray(loaded.faces)
        loaded_uv = np.asarray(getattr(getattr(loaded, "visual", None), "uv", None))
        if (
            loaded_vertices.ndim != 2
            or loaded_vertices.shape[1:] != (3,)
            or len(loaded_vertices) == 0
            or not np.all(np.isfinite(loaded_vertices))
        ):
            raise ValueError("visual overlay GLB vertices are empty, malformed, or non-finite")
        if (
            loaded_faces.ndim != 2
            or loaded_faces.shape[1:] != (3,)
            or len(loaded_faces) == 0
            or np.any(loaded_faces < 0)
            or np.any(loaded_faces >= len(loaded_vertices))
        ):
            raise ValueError("visual overlay GLB faces are empty, malformed, or out of range")
        if (
            loaded_uv.shape != (len(loaded_vertices), 2)
            or not np.issubdtype(loaded_uv.dtype, np.floating)
            or not np.all(np.isfinite(loaded_uv))
        ):
            raise ValueError("visual overlay GLB does not have valid per-vertex texture coordinates")
        if getattr(getattr(loaded.visual, "material", None), "baseColorTexture", None) is None:
            raise ValueError("visual overlay GLB does not have a base-color texture")

        arrays = _load_and_validate_binding(binding_path)
        rest = arrays["visual_rest_vertices_m"]
        faces = arrays["visual_faces"]
        if loaded_vertices.shape != rest.shape:
            raise ValueError("visual overlay GLB vertex buffer does not match the binding")
        max_rest_error_m = float(np.max(np.abs(loaded_vertices - rest)))
        if not np.isfinite(max_rest_error_m) or max_rest_error_m >= VISUAL_OVERLAY_REST_ATOL_M:
            raise ValueError("visual overlay GLB vertex buffer does not match the binding")
        if loaded_faces.shape != faces.shape or not np.array_equal(loaded_faces, faces):
            raise ValueError("visual overlay GLB face buffer does not match the binding")
    except GenesisLiveError as exc:
        if exc.code == "invalid_visual_overlay":
            details = dict(exc.details or {})
            details.setdefault("entity_index", spec.entity_index)
            details.setdefault("entity_name", spec.entity_name)
            raise GenesisLiveError(exc.code, exc.message, details=details) from exc
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise _overlay_error(
            "failed to load or validate visual_overlay assets",
            entity_index=spec.entity_index,
            entity_name=spec.entity_name,
        ) from exc

    return VisualOverlayAssets(
        spec=spec,
        mesh=loaded,
        visual_rest_vertices_m=_readonly_c_array(rest, dtype=np.dtype(np.float32)),
        visual_faces=_readonly_c_array(faces, dtype=np.dtype(np.int32)),
        physics_vertex_indices=_readonly_c_array(
            arrays["physics_vertex_indices"],
            dtype=np.dtype(np.int64),
        ),
    )
