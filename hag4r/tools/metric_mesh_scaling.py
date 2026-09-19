from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeditTetMesh:
    vertices: np.ndarray
    vertex_refs: np.ndarray
    tets: np.ndarray
    tet_refs: np.ndarray


def _next_nonempty(lines: list[str], index: int) -> tuple[str, int]:
    while index < len(lines):
        text = lines[index].strip()
        index += 1
        if text and not text.startswith("#"):
            return text, index
    raise ValueError("unexpected end of .mesh file")


def _read_count(lines: list[str], index: int, section: str) -> tuple[int, int]:
    text, index = _next_nonempty(lines, index)
    try:
        count = int(text.split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{section} count is malformed: {text!r}") from exc
    if count < 1:
        raise ValueError(f"{section} count must be positive, got {count}")
    return count, index


def read_medit_tet_mesh(path: Path) -> MeditTetMesh:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".mesh":
        raise ValueError(f"expected a .mesh file, got: {resolved}")
    lines = resolved.read_text(encoding="utf-8").splitlines()
    vertices: list[list[float]] | None = None
    vertex_refs: list[int] | None = None
    tets: list[list[int]] | None = None
    tet_refs: list[int] | None = None
    index = 0
    while index < len(lines):
        text = lines[index].strip()
        index += 1
        if not text or text.startswith("#"):
            continue
        key = text.split()[0]
        if key == "Vertices":
            count, index = _read_count(lines, index, "Vertices")
            vertices = []
            vertex_refs = []
            for _ in range(count):
                row, index = _next_nonempty(lines, index)
                fields = row.split()
                if len(fields) < 3:
                    raise ValueError(f"vertex row is malformed: {row!r}")
                vertices.append([float(fields[0]), float(fields[1]), float(fields[2])])
                vertex_refs.append(int(fields[3]) if len(fields) > 3 else 1)
        elif key == "Tetrahedra":
            count, index = _read_count(lines, index, "Tetrahedra")
            tets = []
            tet_refs = []
            for _ in range(count):
                row, index = _next_nonempty(lines, index)
                fields = row.split()
                if len(fields) < 4:
                    raise ValueError(f"tetrahedron row is malformed: {row!r}")
                tets.append([int(fields[0]) - 1, int(fields[1]) - 1, int(fields[2]) - 1, int(fields[3]) - 1])
                tet_refs.append(int(fields[4]) if len(fields) > 4 else 1)
        elif key == "End":
            break

    if vertices is None or vertex_refs is None:
        raise ValueError(f"{resolved} does not contain a Vertices section")
    if tets is None or tet_refs is None:
        raise ValueError(f"{resolved} does not contain a Tetrahedra section")

    vertex_array = np.asarray(vertices, dtype=np.float64)
    tet_array = np.asarray(tets, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if tet_array.ndim != 2 or tet_array.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (N, 4)")
    if np.any(tet_array < 0) or np.any(tet_array >= vertex_array.shape[0]):
        raise ValueError("tetrahedron connectivity contains out-of-range vertex indices")
    return MeditTetMesh(
        vertices=vertex_array,
        vertex_refs=np.asarray(vertex_refs, dtype=np.int64),
        tets=tet_array,
        tet_refs=np.asarray(tet_refs, dtype=np.int64),
    )


def write_medit_tet_mesh(path: Path, mesh: MeditTetMesh) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")
        f.write("Vertices\n")
        f.write(f"{mesh.vertices.shape[0]}\n")
        for vertex, ref in zip(mesh.vertices, mesh.vertex_refs, strict=True):
            f.write(f"{vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g} {int(ref)}\n")
        f.write("\nTetrahedra\n")
        f.write(f"{mesh.tets.shape[0]}\n")
        for tet, ref in zip(mesh.tets, mesh.tet_refs, strict=True):
            a, b, c, d = (tet + 1).tolist()
            f.write(f"{a} {b} {c} {d} {int(ref)}\n")
        f.write("\nEnd\n")


def tetra_volume_m3(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    tet_array = np.asarray(tets, dtype=np.int64)
    tet_points = points[tet_array]
    matrices = np.stack(
        (
            tet_points[:, 1] - tet_points[:, 0],
            tet_points[:, 2] - tet_points[:, 0],
            tet_points[:, 3] - tet_points[:, 0],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def _bounds_payload(vertices: np.ndarray) -> dict[str, Any]:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    extent = maxs - mins
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "extent": extent.tolist(),
        "max_extent": float(np.max(extent)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    dtype_bytes = values.dtype.str.encode("utf-8")
    shape_bytes = json.dumps(list(values.shape), separators=(",", ":")).encode("utf-8")
    content_bytes = np.ascontiguousarray(values).tobytes(order="C")
    digest = hashlib.sha256()
    for field in (dtype_bytes, shape_bytes, content_bytes):
        digest.update(len(field).to_bytes(8, byteorder="big", signed=False))
        digest.update(field)
    return digest.hexdigest()


def mesh_volume_metadata(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    mesh = read_medit_tet_mesh(resolved)
    volumes = tetra_volume_m3(mesh.vertices, mesh.tets)
    total_volume = float(np.sum(volumes))
    if not math.isfinite(total_volume) or total_volume <= 0.0:
        raise ValueError(f"mesh volume must be finite and positive, got {total_volume}")
    return {
        "schema_version": "hag4r-unscaled-monolithic-mesh-volume-v1",
        "stage_name": "hag4r_scale_monolithic_mesh_to_metric",
        "input_mesh_path": str(resolved),
        "unscaled_volume_m3": total_volume,
        "bounds_m": _bounds_payload(mesh.vertices),
        "tetra_count": int(mesh.tets.shape[0]),
        "vertex_count": int(mesh.vertices.shape[0]),
        "tet_volume_min_m3": float(np.min(volumes)),
        "tet_volume_max_m3": float(np.max(volumes)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scale_medit_tet_mesh(input_path: Path, output_path: Path, scale_factor: float) -> dict[str, Any]:
    if not math.isfinite(float(scale_factor)) or float(scale_factor) <= 0.0:
        raise ValueError(f"scale_factor must be finite and > 0, got {scale_factor}")
    resolved_input = Path(input_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    mesh = read_medit_tet_mesh(resolved_input)
    scaled_mesh = MeditTetMesh(
        vertices=mesh.vertices * float(scale_factor),
        vertex_refs=mesh.vertex_refs,
        tets=mesh.tets,
        tet_refs=mesh.tet_refs,
    )
    write_medit_tet_mesh(resolved_output, scaled_mesh)
    return {
        "input_mesh_path": str(resolved_input),
        "output_mesh_path": str(resolved_output),
        "scale_factor": float(scale_factor),
        "unscaled_bounds_m": _bounds_payload(mesh.vertices),
        "scaled_bounds_m": _bounds_payload(scaled_mesh.vertices),
        "tetra_count": int(mesh.tets.shape[0]),
        "vertex_count": int(mesh.vertices.shape[0]),
    }


def build_metric_scaling_metadata(
    *,
    input_mesh_path: Path,
    output_mesh_path: Path,
    volume_metadata_path: Path,
    volume_metadata: dict[str, Any],
    target_real_world_volume_m3: float,
    estimate_rationale: str,
    object_context: str = "",
) -> dict[str, Any]:
    target = float(target_real_world_volume_m3)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_real_world_volume_m3 must be finite and > 0, got {target_real_world_volume_m3}")
    if not estimate_rationale.strip():
        raise ValueError("estimate_rationale must be non-empty")
    unscaled_volume = float(volume_metadata["unscaled_volume_m3"])
    if not math.isfinite(unscaled_volume) or unscaled_volume <= 0.0:
        raise ValueError(f"unscaled_volume_m3 must be finite and > 0, got {unscaled_volume}")
    scale_factor = float((target / unscaled_volume) ** (1.0 / 3.0))
    scale_payload = scale_medit_tet_mesh(input_mesh_path, output_mesh_path, scale_factor)
    scaled_volume = float(unscaled_volume * scale_factor**3)
    return {
        "schema_version": "hag4r-metric-mesh-scaling-v1",
        "stage_name": "hag4r_scale_monolithic_mesh_to_metric",
        "input_mesh_path": str(Path(input_mesh_path).expanduser().resolve()),
        "output_mesh_path": str(Path(output_mesh_path).expanduser().resolve()),
        "volume_metadata_path": str(Path(volume_metadata_path).expanduser().resolve()),
        "unscaled_volume_m3": unscaled_volume,
        "target_real_world_volume_m3": target,
        "scale_factor": scale_factor,
        "scaled_volume_m3": scaled_volume,
        "unscaled_bounds_m": scale_payload["unscaled_bounds_m"],
        "scaled_bounds_m": scale_payload["scaled_bounds_m"],
        "tetra_count": int(scale_payload["tetra_count"]),
        "vertex_count": int(scale_payload["vertex_count"]),
        "estimate_rationale": estimate_rationale,
        "object_context": object_context,
    }


__all__ = [
    "MeditTetMesh",
    "build_metric_scaling_metadata",
    "mesh_volume_metadata",
    "read_medit_tet_mesh",
    "scale_medit_tet_mesh",
    "tetra_volume_m3",
    "write_json",
    "write_medit_tet_mesh",
]
