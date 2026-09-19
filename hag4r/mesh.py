#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util import numpy_support
import trimesh
import igl

try:
    import pytetwild
except ImportError:
    pytetwild = None

try:
    import pymeshlab as ml
except ImportError:
    ml = None


SURFACE_EXTRACTION_METHOD = "sdf_marching_cubes"


FIDELITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "part_reduction": 0.90,
        "global_reduction": 0.90,
        "sdf_grid_res": 128,
    },
    "medium": {
        "part_reduction": 0.75,
        "global_reduction": 0.75,
        "sdf_grid_res": 128,
    },
    "high": {
        "part_reduction": 0.50,
        "global_reduction": 0.50,
        "sdf_grid_res": 128,
    },
}


@dataclass(frozen=True)
class MonolithicMeshRequest:
    input_mesh_dir: Path
    output_mesh: Path
    representation: str
    fidelity: str = "medium"
    mesh_extra_args: tuple[str, ...] = ()


class MeshProcessingProtocol:
    """Protocol for monolithizing and cleaning OmniPart part meshes."""

    def run_monolithic_mesh(self, request: MonolithicMeshRequest) -> dict[str, Any]:
        return run_monolithic_mesh(
            input_mesh_dir=request.input_mesh_dir,
            output_mesh=request.output_mesh,
            fidelity=request.fidelity,
            mesh_extra_args=request.mesh_extra_args,
        )


def _ensure_triangular_surface(polydata: Any) -> Any:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    return tri.GetOutput()


def _append_polydata(poly_list: list[Any]) -> Any:
    if not poly_list:
        raise ValueError("No polygonal geometry found while converting input to vtkPolyData.")
    if len(poly_list) == 1:
        return poly_list[0]
    app = vtk.vtkAppendPolyData()
    for poly in poly_list:
        app.AddInputData(poly)
    app.Update()
    return app.GetOutput()


def _dataset_to_polydata(dataset: Any) -> Any:
    if dataset is None:
        raise ValueError("Input mesh reader returned None.")

    if dataset.IsA("vtkPolyData"):
        return dataset

    if dataset.IsA("vtkMultiBlockDataSet"):
        polys = []
        it = dataset.NewIterator()
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            block = it.GetCurrentDataObject()
            if block is not None:
                try:
                    poly = _dataset_to_polydata(block)
                    polys.append(poly)
                except ValueError:
                    pass
            it.GoToNextItem()
        # Some VTK Python wrappers expose Delete(), others rely on Python GC only.
        if hasattr(it, "Delete"):
            it.Delete()
        return _append_polydata(polys)

    if dataset.IsA("vtkPartitionedDataSetCollection"):
        polys = []
        for i in range(dataset.GetNumberOfPartitionedDataSets()):
            pds = dataset.GetPartitionedDataSet(i)
            if pds is None:
                continue
            for j in range(pds.GetNumberOfPartitions()):
                block = pds.GetPartition(j)
                if block is None:
                    continue
                try:
                    polys.append(_dataset_to_polydata(block))
                except ValueError:
                    pass
        return _append_polydata(polys)

    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(dataset)
    geom.Update()
    poly = geom.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not convert dataset type {dataset.GetClassName()} to vtkPolyData.")
    return poly


# Legacy single-mesh loader kept commented out for rollback to the pre-part-GLB CLI.
# def load_input(input_path: Path) -> Any:
#     """Load a surface mesh and return vtkPolyData."""
#     suffix = input_path.suffix.lower()
#
#     # Prefer PyVista for GLB/GLTF because it handles VTK scene readers and multiblock outputs.
#     if suffix in {".glb", ".gltf"}:
#         data = pv.read(str(input_path))
#         vtk_obj = getattr(data, "vtk_obj", data)
#         poly = _dataset_to_polydata(vtk_obj)
#         return _ensure_triangular_surface(poly)
#
#     reader: Any
#     if suffix == ".ply":
#         reader = vtk.vtkPLYReader()
#     elif suffix == ".stl":
#         reader = vtk.vtkSTLReader()
#     elif suffix == ".obj":
#         reader = vtk.vtkOBJReader()
#     elif suffix == ".vtp":
#         reader = vtk.vtkXMLPolyDataReader()
#     else:
#         # Fallback through PyVista for formats VTK readers are not wired here.
#         data = pv.read(str(input_path))
#         vtk_obj = getattr(data, "vtk_obj", data)
#         poly = _dataset_to_polydata(vtk_obj)
#         return _ensure_triangular_surface(poly)
#
#     reader.SetFileName(str(input_path))
#     reader.Update()
#     poly = _dataset_to_polydata(reader.GetOutput())
#     return _ensure_triangular_surface(poly)


def _compute_voxel_grid(bounds: Tuple[float, float, float, float, float, float], voxel_spacing: Optional[float], resolution: Optional[int]) -> Tuple[Tuple[float, float, float], Tuple[int, int, int], Tuple[float, float, float, float, float, float]]:
    if voxel_spacing is None and resolution is None:
        raise ValueError("Either --voxel_spacing or --resolution must be provided.")

    x0, x1, y0, y1, z0, z1 = bounds
    dx = max(x1 - x0, 1e-12)
    dy = max(y1 - y0, 1e-12)
    dz = max(z1 - z0, 1e-12)
    longest = max(dx, dy, dz)

    if voxel_spacing is None:
        if resolution is None or resolution <= 1:
            raise ValueError("--resolution must be > 1.")
        spacing = longest / float(resolution)
    else:
        spacing = float(voxel_spacing)
        if spacing <= 0:
            raise ValueError("--voxel_spacing must be positive.")

    pad_vox = 2.0
    padded_bounds = (
        x0 - pad_vox * spacing,
        x1 + pad_vox * spacing,
        y0 - pad_vox * spacing,
        y1 + pad_vox * spacing,
        z0 - pad_vox * spacing,
        z1 + pad_vox * spacing,
    )

    px = padded_bounds[1] - padded_bounds[0]
    py = padded_bounds[3] - padded_bounds[2]
    pz = padded_bounds[5] - padded_bounds[4]

    nx = max(2, int(round(px / spacing)) + 1)
    ny = max(2, int(round(py / spacing)) + 1)
    nz = max(2, int(round(pz / spacing)) + 1)

    return (spacing, spacing, spacing), (nx, ny, nz), padded_bounds


def extract_surface_via_voxels(
    input_poly: Any,
    voxel_spacing: Optional[float],
    resolution: Optional[int],
    iso_value: float,
    use_flying_edges: bool = True,
) -> tuple[Any, Dict[str, Any]]:
    bounds = tuple(float(v) for v in input_poly.GetBounds())
    spacing_xyz, dims, padded_bounds = _compute_voxel_grid(bounds, voxel_spacing, resolution)

    voxel = vtk.vtkVoxelModeller()
    voxel.SetInputData(input_poly)
    voxel.SetModelBounds(*padded_bounds)
    voxel.SetSampleDimensions(*dims)
    # Important: bit output can break contouring filters.
    if hasattr(voxel, "SetScalarTypeToFloat"):
        voxel.SetScalarTypeToFloat()
    # Use roughly one voxel distance as max influence distance.
    if hasattr(voxel, "SetMaximumDistance"):
        voxel.SetMaximumDistance(max(spacing_xyz))
    voxel.Update()

    volume = voxel.GetOutput()

    surface = None
    if use_flying_edges:
        try:
            surface = vtk.vtkFlyingEdges3D()
        except AttributeError:
            surface = vtk.vtkMarchingCubes()
    else:
        surface = vtk.vtkMarchingCubes()
    surface.SetInputData(volume)
    surface.SetValue(0, float(iso_value))
    if hasattr(surface, "ComputeNormalsOff"):
        surface.ComputeNormalsOff()
    surface.Update()
    out_poly = surface.GetOutput()

    info = {
        "input_bounds": bounds,
        "padded_bounds": padded_bounds,
        "voxel_spacing": spacing_xyz,
        "voxel_dims": dims,
    }
    return clean_surface(out_poly), info


# Legacy VTK -> Trimesh bridge kept commented out for rollback to the single-mesh path.
# def vtkpoly_to_trimesh(polydata: Any) -> trimesh.Trimesh:
#     verts, faces = vtkpoly_to_numpy_faces(polydata)
#     return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _part_glb_index(path: Path) -> int:
    match = re.fullmatch(r"part(\d+)\.glb", path.name)
    if match is None:
        raise ValueError(f"Expected part mesh name like part1.glb, got: {path.name}")
    return int(match.group(1))


def find_part_glbs(input_mesh_dir: Path) -> list[Path]:
    part_paths = [
        path
        for path in input_mesh_dir.iterdir()
        if path.is_file() and re.fullmatch(r"part(\d+)\.glb", path.name) is not None
    ]
    if not part_paths:
        raise ValueError(f"No part GLBs matching part*.glb were found in: {input_mesh_dir}")
    return sorted(part_paths, key=_part_glb_index)


def load_part_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        geometries = [geom.copy() for geom in loaded.geometry.values() if len(geom.vertices) > 0 and len(geom.faces) > 0]
        if not geometries:
            raise ValueError(f"Part GLB contains no triangle geometry: {path}")
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            mesh = trimesh.util.concatenate(geometries)
    else:
        raise TypeError(f"Unsupported trimesh load result for {path}: {type(loaded).__name__}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Part GLB contains no mesh faces: {path}")
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)


def load_part_meshes(input_mesh_dir: Path) -> list[trimesh.Trimesh]:
    return [load_part_mesh(path) for path in find_part_glbs(input_mesh_dir)]


def validate_input_mesh(mesh: trimesh.Trimesh, mesh_path: Optional[Path] = None) -> None:
    mesh_label = str(mesh_path) if mesh_path is not None else "<input mesh>"

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if vertices.size == 0 or len(vertices) == 0:
        raise ValueError(f"Input mesh has empty vertices: {mesh_label}")
    if faces.size == 0 or len(faces) == 0:
        raise ValueError(f"Input mesh has empty faces: {mesh_label}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Input mesh vertices must have shape (N, 3): {mesh_label}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Input mesh faces must have shape (N, 3): {mesh_label}")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError(f"Input mesh face indices are out of range: {mesh_label}")


def load_trimesh_surface(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            geometries = [
                geom.copy()
                for geom in loaded.dump(concatenate=False)
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0
            ]
            if not geometries:
                raise ValueError(f"Mesh scene contains no triangle geometry: {path}")
            mesh = trimesh.util.concatenate(geometries)
    else:
        raise TypeError(f"Unsupported trimesh load result for {path}: {type(loaded).__name__}")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )
    validate_input_mesh(mesh, path)
    return mesh


def split_disconnected_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    components = [component.copy() for component in mesh.split(only_watertight=False)]
    if not components:
        validate_input_mesh(mesh)
        return [mesh.copy()]
    for component in components:
        validate_input_mesh(component)
    return components


def validate_input_meshes(input_mesh_dir: Path, input_meshes: list[trimesh.Trimesh]) -> None:
    part_paths = find_part_glbs(input_mesh_dir)
    if len(part_paths) != len(input_meshes):
        raise ValueError(
            f"Part path count {len(part_paths)} does not match loaded mesh count {len(input_meshes)} "
            f"for {input_mesh_dir}"
        )
    for path, mesh in zip(part_paths, input_meshes):
        validate_input_mesh(mesh, path)


def trimesh_to_meshlab_mesh(mesh: trimesh.Trimesh) -> Any:
    if ml is None:
        raise RuntimeError("pymeshlab is required for MeshLab-based cleanup and simplification.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return ml.Mesh(vertex_matrix=vertices, face_matrix=faces)


def meshlab_current_mesh_to_trimesh(ms: Any) -> trimesh.Trimesh:
    current = ms.current_mesh()
    vertices = np.asarray(current.vertex_matrix(), dtype=np.float64)
    faces = np.asarray(current.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _build_meshlab_meshset(mesh: trimesh.Trimesh) -> Any:
    ms = ml.MeshSet()
    ms.add_mesh(trimesh_to_meshlab_mesh(mesh), "mesh")
    return ms


def _drop_tiny_components_with_trimesh(mesh: trimesh.Trimesh, min_component_faces: int) -> tuple[trimesh.Trimesh, int]:
    components = list(mesh.split(only_watertight=False))
    kept: list[trimesh.Trimesh] = []
    dropped = 0
    for component in components:
        validate_input_mesh(component)
        if len(component.faces) < int(min_component_faces):
            dropped += 1
            continue
        kept.append(component)

    if not kept:
        return mesh.copy(), dropped
    if len(kept) == 1:
        return kept[0].copy(), dropped
    return trimesh.util.concatenate(kept), dropped


def _remove_tiny_components_by_face_number(ms: Any, min_component_faces: int) -> int:
    if min_component_faces <= 0:
        return 0

    before = int(ms.current_mesh().face_number())
    try:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=int(min_component_faces),
            removeunref=True,
        )
        after = int(ms.current_mesh().face_number())
        return max(0, before - after)
    except Exception:
        mesh = meshlab_current_mesh_to_trimesh(ms)
        mesh, dropped = _drop_tiny_components_with_trimesh(mesh, min_component_faces)
        ms.clear()
        ms.add_mesh(trimesh_to_meshlab_mesh(mesh), "mesh")
        return dropped


def _remove_small_floaters(ms: Any, part_floater_face_ratio: float) -> int:
    ms.set_selection_none()
    try:
        ms.compute_selection_by_small_disconnected_components_per_face(
            nbfaceratio=float(part_floater_face_ratio),
        )
    except Exception:
        mesh = meshlab_current_mesh_to_trimesh(ms)
        threshold = max(4, int(round(len(mesh.faces) * float(part_floater_face_ratio))))
        mesh, dropped = _drop_tiny_components_with_trimesh(mesh, threshold)
        ms.clear()
        ms.add_mesh(trimesh_to_meshlab_mesh(mesh), "mesh")
        return int(dropped)

    selected_faces = int(ms.current_mesh().selected_face_number())
    if selected_faces > 0:
        ms.meshing_remove_selected_faces()
        ms.meshing_remove_unreferenced_vertices()
    return max(0, selected_faces)


def _simplify_with_meshlab(ms: Any, target_faces: int) -> int:
    current_face_count = int(ms.current_mesh().face_number())
    if current_face_count <= 4 or target_faces >= current_face_count:
        return current_face_count

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        targetperc=0.0,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3.0,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=False,
        qualityweight=False,
        autoclean=True,
        selected=False,
    )
    return int(ms.current_mesh().face_number())


def _cleanup_meshlab_mesh(ms: Any) -> None:
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()


def _current_mesh_bounds(mesh: trimesh.Trimesh) -> tuple[float, float, float, float, float, float]:
    if len(mesh.vertices) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in mesh.bounds.reshape(-1))


def _normalize_mesh_to_unit_box(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = mesh.bounds
    mins = np.asarray(bounds[0], dtype=np.float64)
    maxs = np.asarray(bounds[1], dtype=np.float64)
    scale = float(np.max(np.maximum(maxs - mins, 1e-12)) * 1.01)
    normalized_vertices = (vertices - mins) / scale
    normalized = trimesh.Trimesh(vertices=normalized_vertices, faces=np.asarray(mesh.faces), process=False)
    info = {
        "bounds_min": mins,
        "bounds_max": maxs,
        "scale": scale,
    }
    return normalized, info


def _denormalize_vertices(vertices: np.ndarray, transform_info: Dict[str, Any]) -> np.ndarray:
    mins = np.asarray(transform_info["bounds_min"], dtype=np.float64)
    scale = float(transform_info["scale"])
    return np.asarray(vertices, dtype=np.float64) * scale + mins


def _sample_signed_distance(mesh: trimesh.Trimesh, points: np.ndarray, sign_method: str) -> tuple[np.ndarray, str]:
    points = np.asarray(points, dtype=np.float64)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if sign_method == "fast_winding":
        sign_type = igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
    else:
        sign_type = igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    sdf_result = igl.signed_distance(points, v, f, sign_type=sign_type)
    sdf = np.asarray(sdf_result[0], dtype=np.float64)
    return sdf, "libigl"


def _marching_cubes_surface(
    values: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[Any, str]:
    nz, ny, nx = values.shape
    xs = origin[0] + np.arange(nx, dtype=np.float64) * spacing[0]
    ys = origin[1] + np.arange(ny, dtype=np.float64) * spacing[1]
    zs = origin[2] + np.arange(nz, dtype=np.float64) * spacing[2]
    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack((grid_x.ravel(order="C"), grid_y.ravel(order="C"), grid_z.ravel(order="C")))
    marching_cubes_result = igl.marching_cubes(
        np.asarray(values, dtype=np.float64).ravel(order="C"),
        points,
        nx,
        ny,
        nz,
        0.0,
    )
    verts = marching_cubes_result[0]
    faces = marching_cubes_result[1]
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return clean_surface(trimesh_to_vtkpoly(mesh)), "libigl"


def _target_faces_from_reduction(current_face_count: int, reduction: float) -> int:
    reduction = max(0.0, min(1.0, float(reduction)))
    return max(4, int(round(current_face_count * (1.0 - reduction))))


def target_faces_from_reduction(current_face_count: int, reduction: float) -> int:
    return _target_faces_from_reduction(current_face_count, reduction)


def simplify_trimesh_with_meshlab(
    mesh: trimesh.Trimesh,
    target_faces: Optional[int] = None,
    reduction: Optional[float] = None,
    preclean: bool = False,
    postclean: bool = False,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    validate_input_mesh(mesh)
    input_face_count = int(len(mesh.faces))
    if target_faces is None:
        if reduction is None:
            target_faces = input_face_count
        else:
            target_faces = target_faces_from_reduction(input_face_count, float(reduction))
    target_faces = max(4, int(target_faces))

    if ml is None:
        if target_faces < input_face_count and input_face_count > 4:
            reduction_value = 1.0 - float(target_faces) / float(input_face_count)
            simplified = vtkpoly_to_trimesh(decimate_surface(trimesh_to_vtkpoly(mesh), reduction_value))
        else:
            simplified = mesh.copy()
        return simplified, {
            "input_face_count": input_face_count,
            "output_face_count": int(len(simplified.faces)),
            "target_faces": int(target_faces),
            "simplification_backend": "pyvista_fallback",
        }

    ms = _build_meshlab_meshset(mesh)
    if preclean:
        _cleanup_meshlab_mesh(ms)
    output_face_count = _simplify_with_meshlab(ms, target_faces)
    if postclean:
        _cleanup_meshlab_mesh(ms)
    simplified = meshlab_current_mesh_to_trimesh(ms)
    validate_input_mesh(simplified)
    return simplified, {
        "input_face_count": input_face_count,
        "output_face_count": int(len(simplified.faces)),
        "target_faces": int(target_faces),
        "simplification_backend": "pymeshlab",
        "meshlab_reported_face_count": int(output_face_count),
    }


def _resolve_fidelity(args: argparse.Namespace) -> Dict[str, Any]:
    if args.fidelity not in FIDELITY_PRESETS:
        raise ValueError(
            f"Unknown mesh fidelity preset {args.fidelity!r}; "
            f"expected one of {', '.join(sorted(FIDELITY_PRESETS))}."
        )
    preset = dict(FIDELITY_PRESETS[args.fidelity])
    if args.part_reduction is None:
        args.part_reduction = preset["part_reduction"]
    if args.global_target_faces is None and args.decimate_target_reduction is None:
        args.decimate_target_reduction = preset["global_reduction"]
    if args.sdf_grid_res is None:
        args.sdf_grid_res = preset["sdf_grid_res"]
    return preset


def _surface_component_summary(polydata: Any) -> Dict[str, Any]:
    mesh = vtkpoly_to_trimesh(polydata)
    components = list(mesh.split(only_watertight=False))
    if not components:
        return {
            "component_count": 0,
            "dominant_component_index": -1,
            "dominant_component_faces": 0,
            "dominant_component_volume": 0.0,
            "dominant_component_area": 0.0,
        }

    component_scores: list[tuple[float, float, float]] = []
    for component in components:
        volume = float(abs(component.volume)) if component.is_watertight else 0.0
        area = float(component.area)
        score = volume if volume > 0.0 else area
        component_scores.append((score, volume, area))

    dominant_component_index = int(np.argmax([score for score, _, _ in component_scores]))
    dominant = components[dominant_component_index]
    return {
        "component_count": len(components),
        "dominant_component_index": dominant_component_index,
        "dominant_component_faces": int(len(dominant.faces)),
        "dominant_component_volume": float(abs(dominant.volume)) if dominant.is_watertight else 0.0,
        "dominant_component_area": float(dominant.area),
    }


def clean_simplify_part_mesh(mesh: trimesh.Trimesh, args: argparse.Namespace) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    input_face_count = int(len(mesh.faces))
    input_vertex_count = int(len(mesh.vertices))
    input_bounds = tuple(float(v) for v in mesh.bounds.reshape(-1))

    if ml is None:
        cleaned = mesh.copy()
        cleaned.remove_duplicate_faces()
        cleaned.remove_degenerate_faces()
        cleaned.remove_unreferenced_vertices()
        cleaned_face_count = int(len(cleaned.faces))
        target_faces = cleaned_face_count
        removed_tiny_components = 0
        removed_floaters = 0
        cleaned, removed_tiny_components = _drop_tiny_components_with_trimesh(cleaned, int(args.part_min_component_faces))
        cleaned_face_count = int(len(cleaned.faces))
        if cleaned_face_count == 0:
            raise RuntimeError("Partwise cleanup removed all faces from a mesh.")
        floater_threshold = max(
            int(args.part_min_component_faces),
            int(round(cleaned_face_count * float(args.part_floater_face_ratio))),
        )
        cleaned, removed_floaters = _drop_tiny_components_with_trimesh(cleaned, floater_threshold)
        cleaned_face_count = int(len(cleaned.faces))
        if cleaned_face_count == 0:
            raise RuntimeError("Partwise cleanup removed all faces from a mesh.")
        if args.part_target_faces is not None:
            target_faces = max(4, int(args.part_target_faces))
        elif args.part_reduction is not None:
            target_faces = _target_faces_from_reduction(cleaned_face_count, float(args.part_reduction))
        if target_faces < cleaned_face_count and cleaned_face_count > 4:
            cleaned = vtkpoly_to_trimesh(
                decimate_surface(trimesh_to_vtkpoly(cleaned), 1.0 - float(target_faces) / float(cleaned_face_count))
            )
        cleaned_face_count = int(len(cleaned.faces))
        if cleaned_face_count == 0:
            raise RuntimeError("Partwise cleanup removed all faces from a mesh.")
        return cleaned, {
            "input_face_count": input_face_count,
            "input_vertex_count": input_vertex_count,
            "input_bounds": input_bounds,
            "removed_tiny_components": int(removed_tiny_components),
            "removed_floaters": int(removed_floaters),
            "simplified_face_count": cleaned_face_count,
            "output_face_count": cleaned_face_count,
            "output_vertex_count": int(len(cleaned.vertices)),
            "output_bounds": tuple(float(v) for v in cleaned.bounds.reshape(-1)),
            "cleanup_backend": "trimesh_fallback",
        }

    ms = _build_meshlab_meshset(mesh)
    _cleanup_meshlab_mesh(ms)
    removed_tiny_components = _remove_tiny_components_by_face_number(ms, int(args.part_min_component_faces))
    removed_floaters = _remove_small_floaters(ms, float(args.part_floater_face_ratio))

    current_face_count = int(ms.current_mesh().face_number())
    target_faces = current_face_count
    if args.part_target_faces is not None:
        target_faces = max(4, int(args.part_target_faces))
    elif args.part_reduction is not None:
        target_faces = _target_faces_from_reduction(current_face_count, float(args.part_reduction))
    simplified_face_count = _simplify_with_meshlab(ms, target_faces)
    _cleanup_meshlab_mesh(ms)
    cleaned = meshlab_current_mesh_to_trimesh(ms)
    if len(cleaned.faces) == 0:
        raise RuntimeError("Partwise MeshLab cleanup removed all faces from a mesh.")

    return cleaned, {
        "input_face_count": input_face_count,
        "input_vertex_count": input_vertex_count,
        "input_bounds": input_bounds,
        "removed_tiny_components": int(removed_tiny_components),
        "removed_floaters": int(removed_floaters),
        "simplified_face_count": int(simplified_face_count),
        "output_face_count": int(len(cleaned.faces)),
        "output_vertex_count": int(len(cleaned.vertices)),
        "output_bounds": tuple(float(v) for v in cleaned.bounds.reshape(-1)),
        "cleanup_backend": "pymeshlab",
    }


def preprocess_part_meshes(input_meshes: list[trimesh.Trimesh], args: argparse.Namespace) -> tuple[list[trimesh.Trimesh], list[Dict[str, Any]]]:
    cleaned_meshes: list[trimesh.Trimesh] = []
    stats: list[Dict[str, Any]] = []
    for part_index, mesh in enumerate(input_meshes, start=1):
        cleaned, info = clean_simplify_part_mesh(mesh, args)
        info["part_index"] = part_index
        cleaned_meshes.append(cleaned)
        stats.append(info)

    if not cleaned_meshes:
        raise RuntimeError("No geometry remains after partwise cleanup and simplification.")
    return cleaned_meshes, stats


def concatenate_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("No meshes were provided for concatenation.")
    if len(meshes) == 1:
        return meshes[0].copy()
    return trimesh.util.concatenate(meshes)


def reconstruct_watertight_surface_sdf(input_meshes: list[trimesh.Trimesh], args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    merged_mesh = concatenate_meshes(input_meshes)
    normalized_mesh, transform_info = _normalize_mesh_to_unit_box(merged_mesh)

    grid_res = int(args.sdf_grid_res)
    if grid_res <= 1:
        raise ValueError("--sdf_grid_res must be greater than 1.")
    padding = float(args.sdf_bbox_padding)
    xmin, ymin, zmin = -padding, -padding, -padding
    xmax, ymax, zmax = 1.0 + padding, 1.0 + padding, 1.0 + padding

    xs = np.linspace(xmin, xmax, grid_res, dtype=np.float64)
    ys = np.linspace(ymin, ymax, grid_res, dtype=np.float64)
    zs = np.linspace(zmin, zmax, grid_res, dtype=np.float64)
    epsilon = float(args.sdf_epsilon_voxels) / float(grid_res)

    step_x = float(xs[1] - xs[0]) if grid_res > 1 else 1.0
    step_y = float(ys[1] - ys[0]) if grid_res > 1 else 1.0
    step_z = float(zs[1] - zs[0]) if grid_res > 1 else 1.0

    grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing="ij")
    sample_points = np.column_stack((grid_x.ravel(order="C"), grid_y.ravel(order="C"), grid_z.ravel(order="C")))
    signed_distance, sdf_backend = _sample_signed_distance(normalized_mesh, sample_points, args.sdf_sign_method)
    signed_distance = signed_distance.reshape((grid_res, grid_res, grid_res), order="C")
    shell_field = epsilon - np.abs(signed_distance)

    surface_poly, marching_cubes_backend = _marching_cubes_surface(
        shell_field,
        origin=(xmin, ymin, zmin),
        spacing=(step_x, step_y, step_z),
    )

    surface_mesh = vtkpoly_to_trimesh(surface_poly)
    denormalized_vertices = _denormalize_vertices(np.asarray(surface_mesh.vertices, dtype=np.float64), transform_info)
    denormalized_mesh = trimesh.Trimesh(vertices=denormalized_vertices, faces=np.asarray(surface_mesh.faces, dtype=np.int64), process=False)
    if len(denormalized_mesh.faces) == 0:
        raise RuntimeError("SDF marching cubes returned an empty surface.")

    return trimesh_to_vtkpoly(denormalized_mesh), {
        "input_mesh_count": len(input_meshes),
        "merged_face_count": int(len(merged_mesh.faces)),
        "normalized_bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
        "grid_res": grid_res,
        "epsilon": epsilon,
        "sign_method": args.sdf_sign_method,
        "sdf_backend": sdf_backend,
        "marching_cubes_backend": marching_cubes_backend,
        "transform_info": transform_info,
        "method": "sdf_marching_cubes",
    }


def simplify_surface_with_meshlab(polydata: Any, args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    mesh = vtkpoly_to_trimesh(polydata)
    input_face_count = int(len(mesh.faces))
    target_faces = input_face_count
    if args.global_target_faces is not None:
        target_faces = max(4, int(args.global_target_faces))
    elif args.decimate_target_reduction is not None:
        target_faces = _target_faces_from_reduction(input_face_count, float(args.decimate_target_reduction))

    if ml is None:
        if target_faces < input_face_count and input_face_count > 4:
            reduction = 1.0 - float(target_faces) / float(input_face_count)
            reduced = decimate_surface(polydata, reduction)
            return reduced, {
                "input_face_count": input_face_count,
                "output_face_count": int(vtkpoly_to_trimesh(reduced).faces.shape[0]),
                "target_faces": target_faces,
                "cleanup_backend": "pyvista_fallback",
            }
        return polydata, {
            "input_face_count": input_face_count,
            "output_face_count": input_face_count,
            "target_faces": target_faces,
            "cleanup_backend": "pyvista_fallback",
        }

    if target_faces >= input_face_count or input_face_count <= 4:
        return polydata, {
            "input_face_count": input_face_count,
            "output_face_count": input_face_count,
            "target_faces": target_faces,
            "cleanup_backend": "none",
        }

    ms = _build_meshlab_meshset(mesh)
    _cleanup_meshlab_mesh(ms)
    output_face_count = _simplify_with_meshlab(ms, target_faces)
    _cleanup_meshlab_mesh(ms)
    simplified = meshlab_current_mesh_to_trimesh(ms)
    return trimesh_to_vtkpoly(simplified), {
        "input_face_count": input_face_count,
        "output_face_count": int(output_face_count),
        "target_faces": int(target_faces),
        "cleanup_backend": "pymeshlab",
    }


def validate_final_surface(polydata: Any) -> Dict[str, Any]:
    if polydata is None or polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
        raise RuntimeError("Final surface is empty.")

    region_count = count_connected_regions(polydata)
    mesh = vtkpoly_to_trimesh(polydata)
    watertight = bool(mesh.is_watertight)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    boundary = igl.boundary_loop(faces)
    boundary_loop_count = 0 if len(boundary) == 0 else 1
    edge_manifold_result = igl.is_edge_manifold(faces)
    edge_manifold = bool(edge_manifold_result[0] if isinstance(edge_manifold_result, tuple) else edge_manifold_result)
    vertex_manifold_result = igl.is_vertex_manifold(faces)
    if isinstance(vertex_manifold_result, tuple):
        vertex_manifold = bool(vertex_manifold_result[0])
    else:
        vertex_manifold = bool(np.all(vertex_manifold_result))
    validation_backend = "libigl"

    if region_count != 1:
        raise RuntimeError(f"Final surface is not monolithic: found {region_count} connected components.")
    if not watertight:
        raise RuntimeError("Final surface is not watertight.")
    if boundary_loop_count != 0:
        raise RuntimeError(f"Final surface has {boundary_loop_count} boundary loops.")
    if not edge_manifold:
        raise RuntimeError("Final surface is not edge manifold.")
    if not vertex_manifold:
        raise RuntimeError("Final surface is not vertex manifold.")

    return {
        "connected_regions": region_count,
        "watertight": watertight,
        "boundary_loops": boundary_loop_count,
        "edge_manifold": edge_manifold,
        "vertex_manifold": vertex_manifold,
        "validation_backend": validation_backend,
    }


def validate_tetmesh(points: Any, tets: Any, zero_volume_threshold: float = 1e-14) -> Dict[str, Any]:
    point_arr = np.asarray(points, dtype=np.float64)
    tet_arr = np.asarray(tets, dtype=np.int64)

    if point_arr.ndim != 2 or point_arr.shape[1] != 3 or len(point_arr) == 0:
        raise RuntimeError("TetWild returned invalid or empty tet vertices.")
    if tet_arr.ndim != 2 or tet_arr.shape[1] != 4 or len(tet_arr) == 0:
        raise RuntimeError("TetWild returned no tetrahedra.")
    if int(tet_arr.min()) < 0 or int(tet_arr.max()) >= len(point_arr):
        raise RuntimeError("TetWild returned tetrahedra with vertex indices out of bounds.")

    a = point_arr[tet_arr[:, 0]]
    b = point_arr[tet_arr[:, 1]]
    c = point_arr[tet_arr[:, 2]]
    d = point_arr[tet_arr[:, 3]]
    signed_volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    abs_volumes = np.abs(signed_volumes)
    zero_volume_count = int(np.sum(abs_volumes < float(zero_volume_threshold)))
    if zero_volume_count:
        raise RuntimeError(
            f"TetWild returned {zero_volume_count} near-zero-volume tetrahedra "
            f"under threshold {zero_volume_threshold:g}."
        )

    return {
        "tet_vertices": int(len(point_arr)),
        "tetrahedra": int(len(tet_arr)),
        "tet_indices_in_bounds": True,
        "tet_abs_volume_min": float(abs_volumes.min()),
        "tet_abs_volume_max": float(abs_volumes.max()),
        "tet_zero_volume_count": zero_volume_count,
        "tet_zero_volume_threshold": float(zero_volume_threshold),
    }


def trimesh_to_vtkpoly(mesh: trimesh.Trimesh) -> Any:
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Trimesh vertices with shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected triangular Trimesh faces with shape (M, 3).")

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

    faces_flat = np.empty((faces.shape[0], 4), dtype=np.int64)
    faces_flat[:, 0] = 3
    faces_flat[:, 1:] = faces
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetCells(
        int(faces.shape[0]),
        numpy_support.numpy_to_vtkIdTypeArray(faces_flat.reshape(-1), deep=True),
    )

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(vtk_faces)
    return poly


def clean_surface(polydata: Any) -> Any:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    current = tri.GetOutput()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(current)
    clean.Update()
    return clean.GetOutput()


def smooth_surface(polydata: Any, iterations: int, pass_band: float) -> Any:
    if iterations <= 0:
        return polydata
    if pass_band <= 0.0 or pass_band >= 2.0:
        raise ValueError("--smooth_pass_band must be in the range (0, 2).")

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(int(iterations))
    smoother.SetPassBand(float(pass_band))
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return clean_surface(smoother.GetOutput())


def orient_surface_normals(polydata: Any) -> Any:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def decimate_surface(polydata: Any, target_reduction: float) -> Any:
    if target_reduction <= 0.0:
        return polydata

    triangulated = clean_surface(polydata)
    decimated = vtkpoly_to_pyvista(triangulated).decimate(target_reduction=float(target_reduction))
    return clean_surface(decimated)


def count_connected_regions(polydata: Any) -> int:
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(polydata)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def keep_only_dominant_component_with_stats(polydata: Any) -> tuple[Any, Dict[str, Any]]:
    if polydata.GetNumberOfPolys() == 0:
        return polydata, {
            "component_count": 0,
            "kept_component_index": -1,
            "dropped_component_count": 0,
            "ranked_by": "none",
        }

    mesh = vtkpoly_to_trimesh(polydata)
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        return clean_surface(polydata), {
            "component_count": len(components),
            "kept_component_index": 0 if components else -1,
            "dropped_component_count": 0,
            "ranked_by": "volume_or_area",
        }

    scores: list[float] = []
    for component in components:
        score = float(abs(component.volume)) if component.is_watertight and abs(component.volume) > 0.0 else float(component.area)
        scores.append(score)

    kept_index = int(np.argmax(scores))
    kept_component = components[kept_index]
    return trimesh_to_vtkpoly(kept_component), {
        "component_count": len(components),
        "kept_component_index": kept_index,
        "dropped_component_count": len(components) - 1,
        "ranked_by": "volume_or_area",
        "kept_component_faces": int(len(kept_component.faces)),
        "kept_component_score": float(scores[kept_index]),
    }


def keep_only_dominant_component(polydata: Any) -> Any:
    kept, _ = keep_only_dominant_component_with_stats(polydata)
    return kept


def validate_monolithic_surface(polydata: Any, args: argparse.Namespace) -> None:
    region_count = count_connected_regions(polydata)
    if region_count != 1:
        raise RuntimeError(
            "Extracted surface is not monolithic: "
            f"found {region_count} connected regions after {SURFACE_EXTRACTION_METHOD}/cleanup."
        )


def save_polydata(polydata: Any, out_path: Path) -> None:
    suffix = out_path.suffix.lower()
    if suffix == ".ply":
        writer = vtk.vtkPLYWriter()
    elif suffix == ".stl":
        writer = vtk.vtkSTLWriter()
    elif suffix == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    elif suffix == ".obj":
        writer = vtk.vtkOBJWriter()
    else:
        raise ValueError(f"Unsupported surface output format: {out_path.suffix} (use .ply/.stl/.vtp/.obj)")
    writer.SetFileName(str(out_path))
    writer.SetInputData(polydata)
    writer.Write()


def vtkpoly_to_pyvista(polydata: Any) -> Any:
    return pv.wrap(polydata)


def vtkpoly_to_numpy_faces(polydata: Any) -> tuple[Any, Any]:
    points_vtk = polydata.GetPoints()
    polys_vtk = polydata.GetPolys()
    if points_vtk is None or polys_vtk is None:
        raise ValueError("vtkPolyData has no points/polys.")

    verts = numpy_support.vtk_to_numpy(points_vtk.GetData()).astype(np.float64, copy=False)
    faces_flat = numpy_support.vtk_to_numpy(polys_vtk.GetData())
    if faces_flat.size == 0:
        raise ValueError("vtkPolyData contains no polygon faces.")

    faces = []
    i = 0
    n = int(faces_flat.size)
    while i < n:
        k = int(faces_flat[i])
        if k < 3:
            i += 1 + max(k, 0)
            continue
        cell = faces_flat[i + 1 : i + 1 + k]
        if k == 3:
            faces.append(cell)
        else:
            # Fan triangulation (TriangleFilter should have already triangulated, but keep safe fallback).
            for j in range(1, k - 1):
                faces.append([cell[0], cell[j], cell[j + 1]])
        i += 1 + k

    if not faces:
        raise ValueError("No triangular faces extracted from vtkPolyData.")
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def vtkpoly_to_trimesh(polydata: Any) -> trimesh.Trimesh:
    verts, faces = vtkpoly_to_numpy_faces(clean_surface(polydata))
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _build_pytetwild_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "edge_length_fac": args.edge_length_fac,
        "epsilon": args.epsilon,
        "optimize": args.optimize,
        "simplify": args.simplify,
    }
    if args.edge_length_abs is not None:
        kwargs["edge_length_abs"] = args.edge_length_abs
    if args.coarsen:
        kwargs["coarsen"] = True
    if args.stop_energy is not None:
        kwargs["stop_energy"] = args.stop_energy
    if args.num_opt_iter is not None:
        kwargs["num_opt_iter"] = args.num_opt_iter
    return kwargs


def tetmesh_with_pytetwild(surface_poly: Any, args: argparse.Namespace) -> Dict[str, Any]:
    if pytetwild is None:
        raise RuntimeError("pytetwild is required when --output_mesh ends with .mesh.")

    kwargs = _build_pytetwild_kwargs(args)

    # Prefer PyVista route when available.
    if hasattr(pytetwild, "tetrahedralize_pv"):
        pv_surface = vtkpoly_to_pyvista(surface_poly)
        tet_grid = pytetwild.tetrahedralize_pv(pv_surface, **kwargs)
        points = np.asarray(tet_grid.points)

        if hasattr(tet_grid, "cells_dict") and tet_grid.cells_dict:
            # VTK tetra type code = 10; pyvista maps to dict keys using vtk cell enum ints.
            if 10 in tet_grid.cells_dict:
                tets = np.asarray(tet_grid.cells_dict[10], dtype=np.int64)
            else:
                # Fallback: first 4-node cell block.
                tets = None
                for _, cell_block in tet_grid.cells_dict.items():
                    arr = np.asarray(cell_block)
                    if arr.ndim == 2 and arr.shape[1] == 4:
                        tets = arr.astype(np.int64, copy=False)
                        break
                if tets is None:
                    raise RuntimeError("Could not extract tetra connectivity from PyVista UnstructuredGrid.")
        else:
            # Older PyVista fallback using cell array layout [4, i, j, k, l, 4, ...]
            celltypes = np.asarray(tet_grid.celltypes)
            mask = celltypes == 10
            if not np.any(mask):
                raise RuntimeError("PyVista tetrahedralize_pv returned no tetrahedral cells.")
            # This path is brittle if mixed cells are present; prefer cells_dict above.
            cells = np.asarray(tet_grid.cells)
            cells = cells.reshape((-1, 5))
            tets = cells[mask, 1:5].astype(np.int64, copy=False)

        return {"points": points, "tets": tets, "grid": tet_grid, "route": "pyvista"}

    if hasattr(pytetwild, "tetrahedralize"):
        verts, faces = vtkpoly_to_numpy_faces(surface_poly)
        tet_verts, tet_tets = pytetwild.tetrahedralize(verts, faces, **kwargs)
        return {
            "points": np.asarray(tet_verts, dtype=np.float64),
            "tets": np.asarray(tet_tets, dtype=np.int64),
            "grid": None,
            "route": "numpy",
        }

    raise RuntimeError("pytetwild module does not expose tetrahedralize_pv or tetrahedralize.")


def write_medit_mesh(output_path: Path, points: Any, tets: Any) -> None:
    pts = np.asarray(points, dtype=float)
    tet_arr = np.asarray(tets, dtype=np.int64)
    if tet_arr.ndim != 2 or tet_arr.shape[1] != 4:
        raise ValueError("Expected tetrahedra connectivity with shape (N, 4).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")
        f.write("Vertices\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1\n")
        f.write("\n")
        f.write("Tetrahedra\n")
        f.write(f"{len(tet_arr)}\n")
        # MEDIT is 1-based indexing.
        for tet in tet_arr:
            a, b, c, d = (tet + 1).tolist()
            f.write(f"{a} {b} {c} {d} 1\n")
        f.write("\nEnd\n")


def infer_output_mode(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".mesh":
        return "tetrahedral"
    if suffix == ".obj":
        return "surface"
    raise ValueError(f"Unsupported output extension: {output_path.suffix} (use .mesh or .obj)")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-part GLBs from an input directory into either a monolithic "
            "surface .obj or a monolithic tetrahedral .mesh using partwise MeshLab "
            "cleanup and SDF marching-cubes reconstruction."
        )
    )
    parser.add_argument(
        "--per_part_volume_meshing_request",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input_mesh_dir",
        type=Path,
        required=False,
        help="Input directory containing per-part GLBs named part1.glb, part2.glb, ...",
    )
    parser.add_argument(
        "--output_mesh",
        type=Path,
        required=False,
        help="Output mesh path (.mesh for tetrahedral output, .obj for surface output).",
    )
    parser.add_argument(
        "--surface_out",
        type=Path,
        default=None,
        help="Optional intermediate extracted surface output for .mesh mode (.ply recommended).",
    )

    parser.add_argument(
        "--fidelity",
        choices=sorted(FIDELITY_PRESETS.keys()),
        default="medium",
        help="Preset controlling default part simplification and SDF resolution (default: medium).",
    )
    parser.add_argument(
        "--part_target_faces",
        type=int,
        default=None,
        help="Absolute face target per part after cleanup and simplification.",
    )
    parser.add_argument(
        "--part_reduction",
        type=float,
        default=None,
        help="Fraction of part faces to remove before monolithic reconstruction.",
    )
    parser.add_argument(
        "--part_floater_face_ratio",
        type=float,
        default=0.005,
        help="MeshLab small-component face ratio used to remove part floaters (default: 0.005).",
    )
    parser.add_argument(
        "--part_min_component_faces",
        type=int,
        default=4,
        help="Drop part connected components with fewer than this many faces (default: 4).",
    )
    parser.add_argument(
        "--global_target_faces",
        type=int,
        default=None,
        help="Absolute face target for the final monolithic surface after reconstruction.",
    )
    parser.add_argument(
        "--decimate_target_reduction",
        type=float,
        default=None,
        help="Fraction of final monolithic surface detail to remove before meshing.",
    )
    parser.add_argument(
        "--sdf_grid_res",
        type=int,
        default=None,
        help="Uniform SDF grid resolution used for marching-cubes reconstruction.",
    )
    parser.add_argument(
        "--sdf_bbox_padding",
        type=float,
        default=0.05,
        help="Padding ratio around the normalized bounding box before SDF sampling (default: 0.05).",
    )
    parser.add_argument(
        "--sdf_epsilon_voxels",
        type=float,
        default=2.0,
        help="Shell half-thickness in voxel units for SDF marching cubes (default: 2.0).",
    )
    parser.add_argument(
        "--sdf_sign_method",
        choices=["pseudonormal", "fast_winding"],
        default="pseudonormal",
        help="Signed-distance sign method used for SDF reconstruction.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Retained for CLI compatibility; the extracted surface is always cleaned before output.",
    )

    parser.add_argument("--edge_length_fac", type=float, default=0.05, help="pytetwild edge length factor.")
    parser.add_argument("--edge_length_abs", type=float, default=None, help="pytetwild absolute edge length.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="pytetwild epsilon.")
    parser.add_argument("--optimize", dest="optimize", action="store_true", help="Enable pytetwild optimization.")
    parser.add_argument("--no_optimize", dest="optimize", action="store_false", help="Disable pytetwild optimization.")
    parser.set_defaults(optimize=True)
    parser.add_argument("--simplify", dest="simplify", action="store_true", help="Enable pytetwild simplification.")
    parser.add_argument("--no_simplify", dest="simplify", action="store_false", help="Disable pytetwild simplification.")
    parser.set_defaults(simplify=True)
    parser.add_argument("--coarsen", action="store_true", help="Enable pytetwild coarsening.")
    parser.add_argument("--stop_energy", type=float, default=None, help="pytetwild stop_energy.")
    parser.add_argument("--num_opt_iter", type=int, default=None, help="pytetwild num_opt_iter.")

    return parser


def _bbox_str(bounds: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"x=[{bounds[0]:.6g},{bounds[1]:.6g}] "
        f"y=[{bounds[2]:.6g},{bounds[3]:.6g}] "
        f"z=[{bounds[4]:.6g},{bounds[5]:.6g}]"
    )


def _build_monolithic_mesh_args(
    input_mesh_dir: Path,
    output_mesh: Path,
    *,
    fidelity: str = "medium",
    mesh_extra_args: tuple[str, ...] = (),
) -> argparse.Namespace:
    parser = _make_parser()
    args = parser.parse_args(
        [
            "--input_mesh_dir",
            str(input_mesh_dir),
            "--output_mesh",
            str(output_mesh),
            "--fidelity",
            fidelity,
            *mesh_extra_args,
        ]
    )
    _resolve_fidelity(args)
    return args


def _print_monolithic_mesh_summary(summary: dict[str, Any]) -> None:
    print(f"bbox: {_bbox_str(summary['bbox'])}")
    print("surface_repair_method: sdf")
    print(f"surface_extraction_method: {summary['surface_extraction_method']}")
    print(f"input_mesh_count: {summary['input_mesh_count']}")
    print(f"cleaned_mesh_count: {summary['cleaned_mesh_count']}")
    print(f"input_faces: {summary['input_faces']}")
    print(f"input_vertices: {summary['input_vertices']}")
    print(f"cleaned_faces: {summary['cleaned_faces']}")
    print(f"cleaned_vertices: {summary['cleaned_vertices']}")
    print(f"removed_floaters: {summary['removed_floaters']}")
    print(f"removed_tiny_components: {summary['removed_tiny_components']}")
    print(f"part_target_faces: {summary['part_target_faces']}")
    print(f"part_reduction: {summary['part_reduction']}")
    print(f"sdf_grid_res: {summary['sdf_grid_res']}")
    print(f"sdf_bbox_padding: {summary['sdf_bbox_padding']}")
    print(f"sdf_epsilon_voxels: {summary['sdf_epsilon_voxels']}")
    print(f"sdf_sign_method: {summary['sdf_sign_method']}")
    print(f"dominant_component_count: {summary['dominant_component_count']}")
    print(f"dominant_component_kept_index: {summary['dominant_component_kept_index']}")
    print(f"dominant_component_dropped_count: {summary['dominant_component_dropped_count']}")
    print(f"final_validation_connected_regions: {summary['final_validation_connected_regions']}")
    print(f"final_validation_watertight: {summary['final_validation_watertight']}")
    print(f"final_validation_boundary_loops: {summary['final_validation_boundary_loops']}")
    print(f"final_validation_edge_manifold: {summary['final_validation_edge_manifold']}")
    print(f"final_validation_vertex_manifold: {summary['final_validation_vertex_manifold']}")
    print(f"final_validation_backend: {summary['final_validation_backend']}")
    print(f"output_mode: {summary['output_mode']}")
    print(f"surface_tris: {summary['surface_tris']}")
    print(f"output_mesh: {summary['output_mesh']}")
    print(f"final_simplification_target_faces: {summary['final_simplification_target_faces']}")
    print(f"final_simplification_backend: {summary['final_simplification_backend']}")
    if summary.get("tets") is not None:
        print(f"tets: {summary['tets']}")
        print(f"tet_route: {summary['tet_route']}")
        print(f"tet_vertices: {summary['tet_vertices']}")
        print(f"tet_indices_in_bounds: {summary['tet_indices_in_bounds']}")
        print(f"tet_abs_volume_min: {summary['tet_abs_volume_min']:.12g}")
        print(f"tet_abs_volume_max: {summary['tet_abs_volume_max']:.12g}")
        print(f"tet_zero_volume_count: {summary['tet_zero_volume_count']}")
        print(f"tet_zero_volume_threshold: {summary['tet_zero_volume_threshold']:.12g}")
    if summary.get("surface_out"):
        print(f"surface_out: {summary['surface_out']}")
    print(f"reconstruction_method: {summary['reconstruction_method']}")
    if "sdf_backend" in summary:
        print(f"sdf_backend: {summary['sdf_backend']}")
    if "marching_cubes_backend" in summary:
        print(f"marching_cubes_backend: {summary['marching_cubes_backend']}")


def _run_monolithic_mesh_from_args(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_mesh_dir
    output_path = args.output_mesh
    if not input_path.exists():
        raise FileNotFoundError(f"--input_mesh_dir does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"--input_mesh_dir is not a directory: {input_path}")
    output_mode = infer_output_mode(output_path)
    if output_mode == "surface" and args.surface_out is not None:
        raise ValueError("--surface_out is only supported when --output_mesh ends with .mesh")

    input_meshes = load_part_meshes(input_path)
    validate_input_meshes(input_path, input_meshes)
    cleaned_parts, part_stats = preprocess_part_meshes(input_meshes, args)

    surface_extraction_method = SURFACE_EXTRACTION_METHOD
    surface_poly, surface_info = reconstruct_watertight_surface_sdf(cleaned_parts, args)

    surface_poly, dominant_info = keep_only_dominant_component_with_stats(surface_poly)
    surface_poly, simplify_info = simplify_surface_with_meshlab(surface_poly, args)
    validation_info = validate_final_surface(surface_poly)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_surface_normals = output_mode == "surface" or args.surface_out is not None
    if save_surface_normals:
        surface_poly = orient_surface_normals(surface_poly)

    if output_mode == "tetrahedral" and args.surface_out is not None:
        args.surface_out.parent.mkdir(parents=True, exist_ok=True)
        save_polydata(surface_poly, args.surface_out)

    tet_result: Optional[Dict[str, Any]] = None
    tet_validation_info: Optional[Dict[str, Any]] = None
    if output_mode == "surface":
        save_polydata(surface_poly, output_path)
    else:
        tet_result = tetmesh_with_pytetwild(surface_poly, args)
        tet_validation_info = validate_tetmesh(tet_result["points"], tet_result["tets"])
        write_medit_mesh(output_path, tet_result["points"], tet_result["tets"])

    bbox = tuple(float(v) for v in surface_poly.GetBounds())
    ntris = int(surface_poly.GetNumberOfPolys())
    total_input_faces = int(sum(stat["input_face_count"] for stat in part_stats))
    total_input_vertices = int(sum(stat["input_vertex_count"] for stat in part_stats))
    total_output_faces = int(sum(stat["output_face_count"] for stat in part_stats))
    total_output_vertices = int(sum(stat["output_vertex_count"] for stat in part_stats))
    removed_floaters = int(sum(stat["removed_floaters"] for stat in part_stats))
    removed_tiny_components = int(sum(stat["removed_tiny_components"] for stat in part_stats))
    summary: dict[str, Any] = {
        "bbox": bbox,
        "surface_repair_method": "sdf",
        "surface_extraction_method": surface_extraction_method,
        "input_mesh_count": len(input_meshes),
        "cleaned_mesh_count": len(cleaned_parts),
        "input_faces": total_input_faces,
        "input_vertices": total_input_vertices,
        "cleaned_faces": total_output_faces,
        "cleaned_vertices": total_output_vertices,
        "removed_floaters": removed_floaters,
        "removed_tiny_components": removed_tiny_components,
        "part_target_faces": args.part_target_faces,
        "part_reduction": args.part_reduction,
        "sdf_grid_res": args.sdf_grid_res,
        "sdf_bbox_padding": args.sdf_bbox_padding,
        "sdf_epsilon_voxels": args.sdf_epsilon_voxels,
        "sdf_sign_method": args.sdf_sign_method,
        "dominant_component_count": dominant_info["component_count"],
        "dominant_component_kept_index": dominant_info["kept_component_index"],
        "dominant_component_dropped_count": dominant_info["dropped_component_count"],
        "final_validation_connected_regions": validation_info["connected_regions"],
        "final_validation_watertight": validation_info["watertight"],
        "final_validation_boundary_loops": validation_info["boundary_loops"],
        "final_validation_edge_manifold": validation_info["edge_manifold"],
        "final_validation_vertex_manifold": validation_info["vertex_manifold"],
        "final_validation_backend": validation_info["validation_backend"],
        "output_mode": output_mode,
        "surface_tris": ntris,
        "output_mesh": str(output_path),
        "final_simplification_target_faces": simplify_info["target_faces"],
        "final_simplification_backend": simplify_info["cleanup_backend"],
        "reconstruction_method": surface_info.get("method", surface_extraction_method),
    }
    if tet_result is not None:
        summary["tets"] = int(len(tet_result["tets"]))
        summary["tet_route"] = tet_result["route"]
        if tet_validation_info is not None:
            summary.update(tet_validation_info)
    if output_mode == "tetrahedral" and args.surface_out is not None:
        summary["surface_out"] = str(args.surface_out)
    if "sdf_backend" in surface_info:
        summary["sdf_backend"] = surface_info["sdf_backend"]
    if "marching_cubes_backend" in surface_info:
        summary["marching_cubes_backend"] = surface_info["marching_cubes_backend"]
    return summary


def run_monolithic_mesh(
    *,
    input_mesh_dir: Path,
    output_mesh: Path,
    fidelity: str = "medium",
    mesh_extra_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    args = _build_monolithic_mesh_args(
        input_mesh_dir,
        output_mesh,
        fidelity=fidelity,
        mesh_extra_args=mesh_extra_args,
    )
    return _run_monolithic_mesh_from_args(args)


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    if args.per_part_volume_meshing_request is not None:
        from hag4r.tools.volumetric_meshing import run_per_part_volumetric_meshing_from_request_json

        result = run_per_part_volumetric_meshing_from_request_json(args.per_part_volume_meshing_request)
        print("per_part_volume_meshing: ok")
        print(f"output_mesh: {result.mesh_path}")
        print(f"heterogeneous_params: {result.heterogeneous_params_path}")
        print(f"metric_mesh_scaling: {result.metric_mesh_scaling_path}")
        print(f"volume_topology: {result.volume_topology_path}")
        print(f"tets: {result.tet_count}")
        print(f"tet_budget_status: {result.tet_budget_status}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    if args.input_mesh_dir is None:
        parser.error("--input_mesh_dir is required unless --per_part_volume_meshing_request is used")
    if args.output_mesh is None:
        parser.error("--output_mesh is required unless --per_part_volume_meshing_request is used")
    _resolve_fidelity(args)
    try:
        summary = _run_monolithic_mesh_from_args(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
    _print_monolithic_mesh_summary(summary)

    return 0


__all__ = [
    "FIDELITY_PRESETS",
    "MeshProcessingProtocol",
    "MonolithicMeshRequest",
    "clean_surface",
    "concatenate_meshes",
    "count_connected_regions",
    "decimate_surface",
    "load_part_meshes",
    "load_trimesh_surface",
    "run_monolithic_mesh",
    "simplify_trimesh_with_meshlab",
    "split_disconnected_components",
    "target_faces_from_reduction",
    "tetmesh_with_pytetwild",
    "trimesh_to_vtkpoly",
    "validate_input_mesh",
    "validate_tetmesh",
    "vtkpoly_to_trimesh",
    "write_medit_mesh",
]


if __name__ == "__main__":
    sys.exit(main())
