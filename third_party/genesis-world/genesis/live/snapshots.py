from __future__ import annotations

import time
from typing import Any

import numpy as np

from genesis.utils.misc import tensor_to_array


def _entity_positions(entity) -> np.ndarray:
    if getattr(entity, "active", False):
        positions = tensor_to_array(entity.get_state().pos)
        if positions.ndim == 3:
            positions = positions[0]
        return np.asarray(positions, dtype=np.float32)
    return np.asarray(tensor_to_array(entity.init_positions), dtype=np.float32)


def geometry_context(entity, *, entity_name: str, frame: str = "env_local") -> dict[str, Any]:
    from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

    positions = _entity_positions(entity)
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    extents = bbox_max - bbox_min
    is_cloth = isinstance(entity.material, ClothMaterial)
    surface_triangles = getattr(entity, "_surface_tri_np", None)
    element_count = int(getattr(entity, "n_elements", 0))
    payload = {
        "entity": entity_name,
        "frame": frame,
        "representation": "surface" if is_cloth else "volumetric",
        "primitive_kind": "triangle" if is_cloth else "tetrahedron",
        "vertex_count": int(getattr(entity, "n_vertices", positions.shape[0])),
        "element_count": element_count,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": extents.tolist(),
        "max_extent": float(np.max(extents)),
        "timestamp": time.time(),
    }
    if is_cloth:
        payload["triangle_count"] = int(len(surface_triangles)) if surface_triangles is not None else element_count
        metadata = getattr(entity, "heterogeneous_material_metadata", None)
        if metadata is not None:
            if "total_area" in metadata:
                payload["surface_total_area"] = float(metadata["total_area"])
            if "total_mass" in metadata:
                payload["surface_total_mass"] = float(metadata["total_mass"])
    return payload


def deformation_snapshot(entity, *, entity_name: str) -> dict[str, Any]:
    positions = _entity_positions(entity)
    initial = np.asarray(tensor_to_array(entity.init_positions), dtype=np.float32)
    if initial.ndim == 3:
        initial = initial[0]
    displacement = positions - initial
    norms = np.linalg.norm(displacement, axis=1)
    return {
        "entity": entity_name,
        "available": True,
        "max_displacement": float(np.max(norms)) if norms.size else 0.0,
        "mean_displacement": float(np.mean(norms)) if norms.size else 0.0,
    }


def material_snapshot(entity, *, entity_name: str) -> dict[str, Any]:
    material = entity.material
    heterogeneous = getattr(entity, "heterogeneous_material_metadata", None)
    payload = {
        "entity": entity_name,
        "material_type": material.__class__.__name__,
        "E": float(getattr(material, "E", 0.0)),
        "nu": float(getattr(material, "nu", 0.0)),
        "rho": float(getattr(material, "rho", 0.0)),
        "friction_mu": float(getattr(material, "friction_mu", 0.0)),
        "heterogeneous": heterogeneous is not None,
    }
    if hasattr(material, "thickness"):
        payload["thickness"] = float(material.thickness)
    if hasattr(material, "bending_stiffness"):
        bending_stiffness = material.bending_stiffness
        payload["bending_stiffness"] = None if bending_stiffness is None else float(bending_stiffness)
    if heterogeneous is not None:
        payload["heterogeneous_kind"] = heterogeneous.get("heterogeneous_kind")
        payload["heterogeneous_metadata"] = heterogeneous
    return payload


def controller_snapshots(controllers: dict[str, Any]) -> list[dict[str, Any]]:
    return [controller.snapshot() for controller in controllers.values()]


def fused_observation(session) -> dict[str, Any]:
    entity_name, entity = session.default_entity()
    return {
        "step": int(session.current_step),
        "paused": bool(session.paused),
        "geometry": geometry_context(entity, entity_name=entity_name),
        "material": material_snapshot(entity, entity_name=entity_name),
        "deformation": deformation_snapshot(entity, entity_name=entity_name),
        "controllers": controller_snapshots(session.controllers),
        "contacts": {"available": False, "reason": "contact snapshot is not implemented in this server feature"},
        "energy": {"available": False, "reason": "energy snapshot is not implemented in this server feature"},
        "timestamp": time.time(),
    }
