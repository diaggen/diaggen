from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

import genesis as gs


@dataclass(frozen=True)
class HeterogeneousMaterial:
    file: str | PathLike
    e_nu_key: str = "tet_E_nu"
    density_key: str = "tet_density"
    labels_key: str = "tet_part_labels"


@dataclass(frozen=True)
class HeterogeneousMaterialData:
    youngs_modulus: np.ndarray
    poisson_ratio: np.ndarray
    density: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class SurfaceHeterogeneousMaterial:
    file: str | PathLike
    e_nu_key: str = "tri_E_nu"
    density_key: str = "tri_density"
    thickness_key: str = "tri_thickness"
    area_density_key: str = "tri_area_density"
    labels_key: str = "tri_part_labels"


@dataclass(frozen=True)
class SurfaceHeterogeneousMaterialData:
    youngs_modulus: np.ndarray
    poisson_ratio: np.ndarray
    density: np.ndarray
    thickness: np.ndarray
    area_density: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    labels: np.ndarray | None
    metadata: dict[str, object]


def _float_dtype():
    return gs.np_float if gs._initialized else np.float32


def _int_dtype():
    return gs.np_int if gs._initialized else np.int32


def _raise(path, message):
    gs.raise_exception(f"Invalid heterogeneous material '{path}': {message}")


def _raise_surface(path, message):
    gs.raise_exception(f"Invalid surface heterogeneous material '{path}': {message}")


def _stats(array: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _surface_lame2d_from_e_nu(youngs_modulus: np.ndarray, poisson_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / (1.0 - poisson_ratio**2)
    return mu, lam


def _require_key(npz, key: str, path: str, raise_func=_raise):
    if key not in npz.files:
        raise_func(path, f"missing required key '{key}'")
    return npz[key]


def _require_surface_vector(array: np.ndarray, *, triangle_count: int, key: str, path: str) -> np.ndarray:
    if array.shape == (triangle_count, 1):
        return array.reshape((triangle_count,))
    if array.shape != (triangle_count,):
        _raise_surface(path, f"'{key}' must have shape ({triangle_count},) or ({triangle_count}, 1), got {array.shape}")
    return array


def validate_heterogeneous_material_arrays(
    e_nu: np.ndarray,
    density: np.ndarray,
    *,
    tet_count: int,
    labels: np.ndarray | None = None,
    path: str | PathLike = "<array>",
    e_nu_key: str = "tet_E_nu",
    density_key: str = "tet_density",
    labels_key: str = "tet_part_labels",
) -> HeterogeneousMaterialData:
    path = str(path)
    e_nu = np.asarray(e_nu)
    density = np.asarray(density)
    labels = None if labels is None else np.asarray(labels)

    if tet_count <= 0:
        _raise(path, f"tet_count must be positive, got {tet_count}")
    if e_nu.shape != (tet_count, 2):
        _raise(path, f"'{e_nu_key}' must have shape ({tet_count}, 2), got {e_nu.shape}")
    if density.shape == (tet_count, 1):
        density = density.reshape((tet_count,))
    elif density.shape != (tet_count,):
        _raise(path, f"'{density_key}' must have shape ({tet_count},) or ({tet_count}, 1), got {density.shape}")
    if labels is not None and labels.shape != (tet_count,):
        _raise(path, f"'{labels_key}' must have shape ({tet_count},), got {labels.shape}")

    if not np.all(np.isfinite(e_nu)):
        _raise(path, f"'{e_nu_key}' contains non-finite values")
    if not np.all(np.isfinite(density)):
        _raise(path, f"'{density_key}' contains non-finite values")

    youngs_modulus = e_nu[:, 0]
    poisson_ratio = e_nu[:, 1]

    if np.any(youngs_modulus <= 0.0):
        _raise(path, f"'{e_nu_key}' Young's modulus values must be positive")
    if np.any((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5)):
        _raise(path, f"'{e_nu_key}' Poisson ratio values must satisfy -1 < nu < 0.5")
    if np.any(density <= 0.0):
        _raise(path, f"'{density_key}' values must be positive")

    youngs_modulus = youngs_modulus.astype(_float_dtype(), copy=False)
    poisson_ratio = poisson_ratio.astype(_float_dtype(), copy=False)
    density = density.astype(_float_dtype(), copy=False)
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    labels = None if labels is None else labels.astype(_int_dtype(), copy=False)

    metadata = {
        "material_path": path,
        "row_count": int(tet_count),
        "e_nu_key": e_nu_key,
        "density_key": density_key,
        "labels_key": labels_key if labels is not None else None,
        "has_labels": labels is not None,
        "youngs_modulus": _stats(youngs_modulus),
        "density": _stats(density),
        "mu": _stats(mu),
        "lambda": _stats(lam),
    }
    if labels is not None:
        metadata["labels"] = {
            "min": int(np.min(labels)),
            "max": int(np.max(labels)),
            "count": int(np.unique(labels).size),
        }

    return HeterogeneousMaterialData(
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        mu=mu.astype(_float_dtype(), copy=False),
        lam=lam.astype(_float_dtype(), copy=False),
        labels=labels,
        metadata=metadata,
    )


def load_heterogeneous_material(spec: HeterogeneousMaterial, tet_count: int) -> HeterogeneousMaterialData:
    if not isinstance(spec.file, (str, PathLike)):
        gs.raise_exception("Heterogeneous material path must be a filesystem path.")

    material_path = Path(spec.file)
    try:
        with np.load(material_path, allow_pickle=False) as npz:
            e_nu = _require_key(npz, spec.e_nu_key, str(material_path))
            density = _require_key(npz, spec.density_key, str(material_path))
            labels = npz[spec.labels_key] if spec.labels_key in npz.files else None
            return validate_heterogeneous_material_arrays(
                e_nu,
                density,
                tet_count=tet_count,
                labels=labels,
                path=material_path,
                e_nu_key=spec.e_nu_key,
                density_key=spec.density_key,
                labels_key=spec.labels_key,
            )
    except Exception as exc:
        if isinstance(exc, gs.GenesisException):
            raise
        gs.raise_exception_from(f"Failed to load heterogeneous material file '{material_path}'.", exc)


def validate_surface_heterogeneous_material_arrays(
    e_nu: np.ndarray,
    density: np.ndarray,
    thickness: np.ndarray,
    area_density: np.ndarray,
    *,
    triangle_count: int,
    labels: np.ndarray | None = None,
    path: str | PathLike = "<array>",
    e_nu_key: str = "tri_E_nu",
    density_key: str = "tri_density",
    thickness_key: str = "tri_thickness",
    area_density_key: str = "tri_area_density",
    labels_key: str = "tri_part_labels",
) -> SurfaceHeterogeneousMaterialData:
    path = str(path)
    e_nu = np.asarray(e_nu)
    density = np.asarray(density)
    thickness = np.asarray(thickness)
    area_density = np.asarray(area_density)
    labels = None if labels is None else np.asarray(labels)

    if triangle_count <= 0:
        _raise_surface(path, f"triangle_count must be positive, got {triangle_count}")
    if e_nu.shape != (triangle_count, 2):
        _raise_surface(path, f"'{e_nu_key}' must have shape ({triangle_count}, 2), got {e_nu.shape}")

    density = _require_surface_vector(density, triangle_count=triangle_count, key=density_key, path=path)
    thickness = _require_surface_vector(thickness, triangle_count=triangle_count, key=thickness_key, path=path)
    area_density = _require_surface_vector(
        area_density,
        triangle_count=triangle_count,
        key=area_density_key,
        path=path,
    )
    if labels is not None and labels.shape != (triangle_count,):
        _raise_surface(path, f"'{labels_key}' must have shape ({triangle_count},), got {labels.shape}")

    for key, array in (
        (e_nu_key, e_nu),
        (density_key, density),
        (thickness_key, thickness),
        (area_density_key, area_density),
    ):
        if not np.issubdtype(array.dtype, np.number):
            _raise_surface(path, f"'{key}' must contain numeric values")
        if not np.all(np.isfinite(array)):
            _raise_surface(path, f"'{key}' contains non-finite values")

    youngs_modulus = e_nu[:, 0]
    poisson_ratio = e_nu[:, 1]

    if np.any(youngs_modulus <= 0.0):
        _raise_surface(path, f"'{e_nu_key}' Young's modulus values must be positive")
    if np.any((poisson_ratio <= -1.0) | (poisson_ratio >= 0.5)):
        _raise_surface(path, f"'{e_nu_key}' Poisson ratio values must satisfy -1 < nu < 0.5")
    if np.any(density <= 0.0):
        _raise_surface(path, f"'{density_key}' values must be positive")
    if np.any(thickness <= 0.0):
        _raise_surface(path, f"'{thickness_key}' values must be positive")
    if np.any(area_density <= 0.0):
        _raise_surface(path, f"'{area_density_key}' values must be positive")
    if not np.allclose(area_density, density * thickness, rtol=1e-6, atol=1e-9):
        _raise_surface(path, f"'{area_density_key}' must match '{density_key}' * '{thickness_key}'")

    if labels is not None:
        if not np.issubdtype(labels.dtype, np.number):
            _raise_surface(path, f"'{labels_key}' must contain numeric values")
        if not np.all(np.isfinite(labels)):
            _raise_surface(path, f"'{labels_key}' contains non-finite values")
        if not np.all(labels == np.round(labels)):
            _raise_surface(path, f"'{labels_key}' values must be integer-compatible")

    youngs_modulus = youngs_modulus.astype(_float_dtype(), copy=False)
    poisson_ratio = poisson_ratio.astype(_float_dtype(), copy=False)
    density = density.astype(_float_dtype(), copy=False)
    thickness = thickness.astype(_float_dtype(), copy=False)
    area_density = area_density.astype(_float_dtype(), copy=False)
    mu, lam = _surface_lame2d_from_e_nu(youngs_modulus, poisson_ratio)
    labels = None if labels is None else labels.astype(_int_dtype(), copy=False)

    metadata = {
        "material_path": path,
        "representation": "surface",
        "primitive_kind": "triangle",
        "row_count": int(triangle_count),
        "e_nu_key": e_nu_key,
        "density_key": density_key,
        "thickness_key": thickness_key,
        "area_density_key": area_density_key,
        "labels_key": labels_key if labels is not None else None,
        "has_labels": labels is not None,
        "youngs_modulus": _stats(youngs_modulus),
        "poisson_ratio": _stats(poisson_ratio),
        "density": _stats(density),
        "thickness": _stats(thickness),
        "area_density": _stats(area_density),
        "mu": _stats(mu),
        "lambda": _stats(lam),
        "area_density_rtol": 1e-6,
        "area_density_atol": 1e-9,
    }
    if labels is not None:
        metadata["labels"] = {
            "min": int(np.min(labels)),
            "max": int(np.max(labels)),
            "count": int(np.unique(labels).size),
        }

    return SurfaceHeterogeneousMaterialData(
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        thickness=thickness,
        area_density=area_density,
        mu=mu.astype(_float_dtype(), copy=False),
        lam=lam.astype(_float_dtype(), copy=False),
        labels=labels,
        metadata=metadata,
    )


def load_surface_heterogeneous_material(
    spec: SurfaceHeterogeneousMaterial,
    triangle_count: int,
) -> SurfaceHeterogeneousMaterialData:
    if not isinstance(spec.file, (str, PathLike)):
        gs.raise_exception("Surface heterogeneous material path must be a filesystem path.")

    material_path = Path(spec.file)
    try:
        with np.load(material_path, allow_pickle=False) as npz:
            e_nu = _require_key(npz, spec.e_nu_key, str(material_path), raise_func=_raise_surface)
            density = _require_key(npz, spec.density_key, str(material_path), raise_func=_raise_surface)
            thickness = _require_key(npz, spec.thickness_key, str(material_path), raise_func=_raise_surface)
            area_density = _require_key(npz, spec.area_density_key, str(material_path), raise_func=_raise_surface)
            labels = npz[spec.labels_key] if spec.labels_key in npz.files else None
            return validate_surface_heterogeneous_material_arrays(
                e_nu,
                density,
                thickness,
                area_density,
                triangle_count=triangle_count,
                labels=labels,
                path=material_path,
                e_nu_key=spec.e_nu_key,
                density_key=spec.density_key,
                thickness_key=spec.thickness_key,
                area_density_key=spec.area_density_key,
                labels_key=spec.labels_key,
            )
    except Exception as exc:
        if isinstance(exc, gs.GenesisException):
            raise
        gs.raise_exception_from(f"Failed to load surface heterogeneous material file '{material_path}'.", exc)


def load_obj_triangle_faces(mesh_file: str | PathLike) -> np.ndarray:
    if not isinstance(mesh_file, (str, PathLike)):
        gs.raise_exception("Surface mesh path must be a filesystem path.")

    mesh_path = Path(mesh_file)
    vertex_count = 0
    face_records: list[tuple[int, int, list[str]]] = []
    with open(mesh_path, encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if tokens[0] == "v":
                vertex_count += 1
            elif tokens[0] == "f":
                face_records.append((line_number, vertex_count, tokens[1:]))

    faces = []
    for line_number, face_vertex_count, tokens in face_records:
        if len(tokens) != 3:
            _raise_surface(mesh_path, f"OBJ face on line {line_number} has {len(tokens)} vertices; expected 3")

        face = []
        for token in tokens:
            vertex_token = token.split("/", 1)[0]
            if vertex_token == "":
                _raise_surface(mesh_path, f"OBJ face on line {line_number} has an empty vertex index")
            try:
                vertex_index = int(vertex_token)
            except ValueError:
                _raise_surface(mesh_path, f"OBJ face on line {line_number} has invalid vertex index '{vertex_token}'")
            if vertex_index == 0:
                _raise_surface(mesh_path, f"OBJ face on line {line_number} uses zero as a vertex index")
            if vertex_index < 0:
                zero_based = face_vertex_count + vertex_index
            else:
                zero_based = vertex_index - 1
            if zero_based < 0 or zero_based >= face_vertex_count:
                _raise_surface(mesh_path, f"OBJ face on line {line_number} references vertex {vertex_index} out of range")
            face.append(zero_based)
        faces.append(face)

    return np.asarray(faces, dtype=_int_dtype()).reshape((-1, 3))


def validate_surface_mesh_material_contract(
    mesh_file: str | PathLike,
    material_data: SurfaceHeterogeneousMaterialData,
) -> dict[str, object]:
    faces = load_obj_triangle_faces(mesh_file)
    mesh_path = Path(mesh_file)
    triangle_count = int(faces.shape[0])
    material_row_count = material_data.metadata["row_count"]
    if triangle_count != material_row_count:
        _raise_surface(
            mesh_path,
            f"OBJ triangle count {triangle_count} does not match material row_count {material_row_count}",
        )
    return {
        "mesh_path": str(mesh_path),
        "triangle_count": triangle_count,
        "face_order": "obj_source_order",
    }


def validate_surface_face_remap(
    face_remap: np.ndarray,
    *,
    source_triangle_count: int,
    backend_triangle_count: int,
    path: str | PathLike = "<array>",
) -> np.ndarray:
    path = str(path)
    face_remap = np.asarray(face_remap)

    if source_triangle_count <= 0:
        _raise_surface(path, f"source_triangle_count must be positive, got {source_triangle_count}")
    if backend_triangle_count <= 0:
        _raise_surface(path, f"backend_triangle_count must be positive, got {backend_triangle_count}")
    if face_remap.shape != (backend_triangle_count,):
        _raise_surface(path, f"face_remap must have shape ({backend_triangle_count},), got {face_remap.shape}")
    if not np.issubdtype(face_remap.dtype, np.number):
        _raise_surface(path, "face_remap must contain numeric values")
    if not np.all(np.isfinite(face_remap)):
        _raise_surface(path, "face_remap contains non-finite values")
    if not np.all(face_remap == np.round(face_remap)):
        _raise_surface(path, "face_remap values must be integer-compatible")

    face_remap = face_remap.astype(_int_dtype(), copy=False)
    if np.any((face_remap < 0) | (face_remap >= source_triangle_count)):
        _raise_surface(path, f"face_remap values must be in [0, {source_triangle_count})")
    if source_triangle_count == backend_triangle_count and np.unique(face_remap).size != source_triangle_count:
        _raise_surface(path, "face_remap must be a one-to-one permutation when triangle counts match")
    return face_remap
