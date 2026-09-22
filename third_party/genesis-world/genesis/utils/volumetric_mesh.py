from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import igl
import numpy as np

import genesis as gs


_TET_FACE_VERTICES = np.array(
    [
        [0, 2, 1],
        [1, 2, 3],
        [0, 1, 3],
        [0, 3, 2],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class VolumetricMeshData:
    verts: np.ndarray
    tets: np.ndarray
    surface_triangles: np.ndarray
    surface_triangle_tet_indices: np.ndarray
    boundary_vertex_indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    metadata: dict[str, object]


def _float_dtype():
    return gs.np_float if gs._initialized else np.float32


def _int_dtype():
    return gs.np_int if gs._initialized else np.int32


def _raise(path, message):
    gs.raise_exception(f"Invalid volumetric mesh '{path}': {message}")


def _validate_index_dtype(indices, path, name):
    if indices.size and not np.issubdtype(indices.dtype, np.integer):
        _raise(path, f"{name} indices must be integers")


def _tet_faces(tets):
    faces = tets[:, _TET_FACE_VERTICES].reshape((-1, 3))
    owners = np.repeat(np.arange(tets.shape[0], dtype=np.int64), 4)
    return faces, owners


def _boundary_face_lookup(tets, path="<array>"):
    all_faces, owners = _tet_faces(np.asarray(tets, dtype=np.int64))
    face_keys = np.sort(all_faces, axis=1)
    unique_keys, unique_idcs, counts = np.unique(face_keys, axis=0, return_index=True, return_counts=True)

    if np.any(counts > 2):
        _raise(path, "non-manifold tetrahedral face shared by more than two tetrahedra")

    boundary_mask = counts == 1
    boundary_keys = unique_keys[boundary_mask]
    boundary_face_idcs = unique_idcs[boundary_mask]

    lookup = {tuple(key.tolist()): int(owners[idx]) for key, idx in zip(boundary_keys, boundary_face_idcs, strict=True)}
    return all_faces, owners, face_keys, lookup


def derive_surface_triangles(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tets = np.asarray(tets)
    if tets.ndim != 2 or tets.shape[1] != 4:
        gs.raise_exception("Tetrahedron array must have shape (n_tets, 4).")

    all_faces, owners = _tet_faces(tets.astype(np.int64, copy=False))
    face_keys = np.sort(all_faces, axis=1)
    unique_keys, counts = np.unique(face_keys, axis=0, return_counts=True)
    if np.any(counts > 2):
        gs.raise_exception("Non-manifold tetrahedral face shared by more than two tetrahedra.")
    count_by_key = {tuple(key.tolist()): int(count) for key, count in zip(unique_keys, counts, strict=True)}

    surface_faces = []
    surface_owners = []
    for face, owner, key in zip(all_faces, owners, face_keys, strict=True):
        if count_by_key[tuple(key.tolist())] == 1:
            surface_faces.append(face)
            surface_owners.append(owner)

    return (
        np.asarray(surface_faces, dtype=_int_dtype()).reshape((-1, 3)),
        np.asarray(surface_owners, dtype=_int_dtype()).reshape((-1,)),
    )


def validate_volumetric_mesh(
    verts: np.ndarray,
    tets: np.ndarray,
    surface_triangles: np.ndarray | None,
    path: str | PathLike = "<array>",
) -> VolumetricMeshData:
    path = str(path)
    verts = np.asarray(verts)
    tets = np.asarray(tets)
    surface_triangles = (
        np.empty((0, 3), dtype=np.int64) if surface_triangles is None else np.asarray(surface_triangles)
    )

    if verts.ndim != 2 or verts.shape[1] != 3:
        _raise(path, "vertex array must have shape (n_vertices, 3)")
    if verts.shape[0] == 0:
        _raise(path, "missing vertices")
    if tets.ndim != 2 or tets.shape[1] != 4:
        _raise(path, "tetrahedron array must have shape (n_tets, 4)")
    if tets.shape[0] == 0:
        _raise(path, "missing tetrahedra")
    if surface_triangles.size != 0 and (surface_triangles.ndim != 2 or surface_triangles.shape[1] != 3):
        _raise(path, "surface triangle array must be empty or have shape (n_surface_triangles, 3)")

    _validate_index_dtype(tets, path, "tetrahedron")
    _validate_index_dtype(surface_triangles, path, "surface triangle")

    if not np.all(np.isfinite(verts)):
        _raise(path, "non-finite vertex coordinates")

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)

    if np.any(tets < 0) or np.any(tets >= verts.shape[0]):
        _raise(path, "tetrahedron indices are out of range")
    if surface_triangles.size and (np.any(surface_triangles < 0) or np.any(surface_triangles >= verts.shape[0])):
        _raise(path, "surface triangle indices are out of range")

    verts = verts.astype(_float_dtype(), copy=False)
    tets = tets.astype(_int_dtype(), copy=False)
    surface_triangles = surface_triangles.reshape((-1, 3)).astype(_int_dtype(), copy=False)

    if np.any([np.unique(tet).size != 4 for tet in tets]):
        _raise(path, "tetrahedron repeats a vertex")
    if surface_triangles.size and np.any([np.unique(tri).size != 3 for tri in surface_triangles]):
        _raise(path, "surface triangle repeats a vertex")

    sorted_tets = np.sort(tets, axis=1)
    if np.unique(sorted_tets, axis=0).shape[0] != sorted_tets.shape[0]:
        _raise(path, "duplicate tetrahedra")

    a = verts[tets[:, 0]] - verts[tets[:, 3]]
    b = verts[tets[:, 1]] - verts[tets[:, 3]]
    c = verts[tets[:, 2]] - verts[tets[:, 3]]
    det = np.linalg.det(np.stack([a, b, c], axis=2))
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    eps = max(1e-12, 1e-12 * bbox_diag**3)
    if np.any(np.abs(det) <= eps):
        _raise(path, "degenerate tetrahedron with near-zero signed volume")
    if np.any(det >= eps):
        _raise(path, "inverted tetrahedron orientation")

    _, _, _, boundary_lookup = _boundary_face_lookup(tets, path)

    if surface_triangles.size:
        sorted_surface = np.sort(surface_triangles, axis=1)
        if np.unique(sorted_surface, axis=0).shape[0] != sorted_surface.shape[0]:
            _raise(path, "duplicate explicit surface triangles")

        surface_triangle_tet_indices = []
        for tri_key in sorted_surface:
            key = tuple(tri_key.tolist())
            if key not in boundary_lookup:
                _raise(path, "explicit surface triangle is not a boundary face of exactly one tetrahedron")
            surface_triangle_tet_indices.append(boundary_lookup[key])

        surface_source = "file"
        surface_triangle_tet_indices = np.asarray(surface_triangle_tet_indices, dtype=_int_dtype())
    else:
        surface_source = "derived"
        surface_triangles, surface_triangle_tet_indices = derive_surface_triangles(tets)

    boundary_vertex_indices = np.unique(surface_triangles).astype(_int_dtype(), copy=False)
    metadata = {
        "mesh_path": path,
        "vertex_count": int(verts.shape[0]),
        "tet_count": int(tets.shape[0]),
        "surface_triangle_count": int(surface_triangles.shape[0]),
        "surface_source": surface_source,
        "bbox_min": bbox_min.copy(),
        "bbox_max": bbox_max.copy(),
        "index_base": 0,
        "mesh_format": "mesh",
        "ignored_sections": [],
    }

    return VolumetricMeshData(
        verts=verts,
        tets=tets,
        surface_triangles=surface_triangles,
        surface_triangle_tet_indices=surface_triangle_tet_indices,
        boundary_vertex_indices=boundary_vertex_indices,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        metadata=metadata,
    )


def load_tet_mesh(path: str | PathLike) -> VolumetricMeshData:
    if not isinstance(path, (str, PathLike)):
        gs.raise_exception("Tet mesh path must be a filesystem path.")

    mesh_path = Path(path)
    if mesh_path.suffix.lower() != ".mesh":
        gs.raise_exception(f"Expected `.mesh` extension for volumetric mesh file: {path}")

    try:
        verts, tets, faces = igl.readMESH(str(mesh_path))
    except Exception as exc:
        gs.raise_exception_from(f"Failed to parse volumetric mesh file '{mesh_path}'.", exc)

    return validate_volumetric_mesh(verts, tets, faces, mesh_path)
