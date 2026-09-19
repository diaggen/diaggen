from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


OBJECT_DESCRIPTION_SCHEMA_VERSION = "hag4r-agentic-object-description-v1"
OBJECT_DESCRIPTION_REQUIRED_KEYS = (
    "inferred_object_name",
    "object_description",
)
OBJECT_DESCRIPTION_LIST_KEYS: tuple[str, ...] = ()
OBJECT_DESCRIPTION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OBJECT_DESCRIPTION_REQUIRED_KEYS),
    "properties": {
        "inferred_object_name": {"type": "string"},
        "object_description": {"type": "string"},
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_prefix = handle.read(1)
                if not marker_prefix:
                    return None
                if marker_prefix != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    segment_length = struct.unpack(">H", handle.read(2))[0]
                    if segment_length < 7:
                        return None
                    handle.read(1)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return int(width), int(height)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                segment_length_bytes = handle.read(2)
                if len(segment_length_bytes) != 2:
                    return None
                segment_length = struct.unpack(">H", segment_length_bytes)[0]
                if segment_length < 2:
                    return None
                handle.seek(segment_length - 2, 1)
    return None


def _image_within_budget(path: Path, max_image_edge_px: int) -> bool:
    if not path.exists():
        return True
    dimensions = _image_dimensions(path)
    return dimensions is None or max(dimensions) <= max_image_edge_px


def _select_object_description_images(
    source_image: Path,
    context_image_paths: tuple[Path, ...],
    *,
    max_image_count: int,
    max_image_edge_px: int,
) -> tuple[Path, ...]:
    selected: list[Path] = []
    for image_path in (*context_image_paths, source_image):
        if image_path in selected:
            continue
        if _image_within_budget(image_path, max_image_edge_px):
            selected.append(image_path)
        if len(selected) >= max_image_count:
            break
    return tuple(selected) if selected else (source_image,)


def _parse_json_response(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Object-description response must be a JSON object.")
    return payload


def _missing_required_object_description_keys(response: dict[str, Any]) -> list[str]:
    return [key for key in OBJECT_DESCRIPTION_REQUIRED_KEYS if key not in response]


def _missing_list_object_description_keys(response: dict[str, Any]) -> list[str]:
    return []


def normalize_object_description_response(
    response: dict[str, Any],
    *,
    source_image: Path,
    image_paths: tuple[Path, ...],
    generator: str,
    object_name_hint: str = "",
    user_hints: str = "",
    diagnostic_cues: tuple[dict[str, Any], ...] = (),
    allow_missing_list_fields: bool = True,
    repair_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing = _missing_required_object_description_keys(response)
    missing_non_list = [key for key in missing if key not in OBJECT_DESCRIPTION_LIST_KEYS]
    if missing_non_list:
        raise ValueError("VLM object-description response is missing required field(s): " + ", ".join(missing_non_list))
    missing_list_fields = _missing_list_object_description_keys(response)
    if missing_list_fields and not allow_missing_list_fields:
        raise ValueError(
            "VLM object-description response is missing required list field(s): "
            + ", ".join(missing_list_fields)
        )
    description = str(response.get("object_description", "")).strip()
    inferred_name = str(response.get("inferred_object_name", "")).strip()
    if not description:
        raise ValueError("VLM object-description response must include `object_description`.")
    normalized = {
        "schema_version": OBJECT_DESCRIPTION_SCHEMA_VERSION,
        "inferred_object_name": inferred_name or object_name_hint or source_image.stem.replace("_", " "),
        "object_description": description,
        "source_image": str(source_image),
        "image_paths": [str(path) for path in image_paths],
        "generator": generator,
        "user_hints": user_hints,
        "diagnostic_cues": list(diagnostic_cues),
    }
    if repair_provenance:
        normalized["repair_provenance"] = dict(repair_provenance)
    return normalized


def load_object_description_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(_path_for_repo(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Object-description payload must be a JSON object: {path}")
    schema_version = payload.get("schema_version")
    if schema_version != OBJECT_DESCRIPTION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported object-description schema_version at {path}: {schema_version!r}")
    return payload


def object_description_text(payload_or_path: dict[str, Any] | Path | str) -> str:
    payload = (
        load_object_description_payload(Path(payload_or_path))
        if isinstance(payload_or_path, str | Path)
        else payload_or_path
    )
    description = str(payload.get("object_description", "")).strip()
    if description:
        return description
    inferred_name = str(payload.get("inferred_object_name", "")).strip()
    if inferred_name:
        return inferred_name
    raise ValueError("Object-description payload is missing `object_description`.")

__all__ = [
    "OBJECT_DESCRIPTION_OUTPUT_SCHEMA",
    "OBJECT_DESCRIPTION_SCHEMA_VERSION",
    "load_object_description_payload",
    "normalize_object_description_response",
    "object_description_text",
]
