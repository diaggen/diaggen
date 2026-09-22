from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .protocol import PROTOCOL, capabilities_for_report


SURFACE_HETEROGENEOUS_REQUIRED_ATTRS = (
    ("triangles", "lambda", 1, np.array([1.5], dtype=np.float64)),
    ("triangles", "mu", 1, np.array([2.5], dtype=np.float64)),
    ("vertices", "thickness", 3, np.array([0.001, 0.002, 0.003], dtype=np.float64)),
    ("vertices", "volume", 3, np.array([1.0, 2.0, 3.0], dtype=np.float64)),
    ("meta", "mass_density", 1, np.array([1.0], dtype=np.float64)),
)

SURFACE_POSITION_CONSTRAINT_REQUIRED_ATTRS = (
    ("vertices", "is_constrained", 3, np.array([1, 0, 1], dtype=np.int32)),
    (
        "vertices",
        "aim_position",
        9,
        np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]], dtype=np.float64),
    ),
    ("vertices", "strength_ratio", 3, np.array([100.0, 200.0, 300.0], dtype=np.float32)),
)


def _missing_status(reason: str, message: str, *, requirement: str, exc: BaseException | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "code": "unsupported_surface_backend",
        "reason": reason,
        "message": message,
        "missing": [requirement],
    }
    if exc is not None:
        payload["exception"] = f"{exc.__class__.__name__}: {exc}"
    return payload


def surface_backend_status() -> dict[str, Any]:
    if importlib.util.find_spec("uipc") is None:
        return _missing_status(
            "missing_uipc",
            "surface shell diagnostics require the uipc Python module",
            requirement="uipc",
        )
    try:
        import uipc  # noqa: F401
    except Exception as exc:
        return _missing_status(
            "broken_uipc",
            "surface shell diagnostics require an importable uipc Python module",
            requirement="uipc",
            exc=exc,
        )
    if not torch.cuda.is_available():
        return _missing_status(
            "missing_cuda",
            "surface shell diagnostics require a CUDA-capable torch runtime",
            requirement="cuda",
        )
    try:
        from genesis.engine.couplers import IPCCoupler  # noqa: F401
        from genesis.options import IPCCouplerOptions  # noqa: F401
    except Exception as exc:
        return _missing_status(
            "missing_ipc_coupler",
            "surface shell diagnostics require Genesis IPCCoupler support",
            requirement="ipc_coupler",
            exc=exc,
        )
    return {
        "available": True,
        "requirements": {
            "uipc": True,
            "cuda": True,
            "ipc_coupler": True,
        },
    }


def _surface_heterogeneous_attr_probe() -> dict[str, Any]:
    import uipc
    from uipc.constitution import ElasticModuli2D, StrainLimitingBaraffWitkinShell

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = uipc.geometry.trimesh(vertices, faces)
    StrainLimitingBaraffWitkinShell().apply_to(
        mesh,
        moduli=ElasticModuli2D.youngs_poisson(1.0e5, 0.25),
        mass_density=200.0,
        thickness=0.002,
    )

    missing = []
    attrs = {}
    for domain_name, attr_name, expected_len, write_value in SURFACE_HETEROGENEOUS_REQUIRED_ATTRS:
        domain = getattr(mesh, domain_name)()
        attr = domain.find(attr_name)
        key = f"{domain_name}.{attr_name}"
        if not attr:
            missing.append(key)
            continue
        view = uipc.view(attr)
        if view.reshape(-1).shape[0] != expected_len:
            missing.append(key)
            attrs[key] = {"length": int(view.reshape(-1).shape[0]), "expected_length": int(expected_len)}
            continue
        view[:] = write_value.reshape(view.shape)
        attrs[key] = {"length": int(expected_len), "writable": True}

    if missing:
        return {
            "available": False,
            "reason": "missing_surface_heterogeneous_ipc_attrs",
            "missing": missing,
            "attrs": attrs,
        }
    return {"available": True, "attrs": attrs}


def surface_heterogeneous_backend_status(surface_status: dict[str, Any] | None = None) -> dict[str, Any]:
    if surface_status is None:
        surface_status = surface_backend_status()
    if not surface_status["available"]:
        return {
            "available": False,
            "code": "unsupported_surface_backend",
            "reason": "surface_backend_unavailable",
            "message": "surface heterogeneous materials require surface shell backend support",
            "missing": list(surface_status.get("missing", ["surface_shell"])),
            "surface_shell": surface_status,
        }
    try:
        probe = _surface_heterogeneous_attr_probe()
    except Exception as exc:
        return _missing_status(
            "broken_surface_heterogeneous_attr_probe",
            "surface heterogeneous materials require writable pyuipc shell material attrs",
            requirement="surface_heterogeneous_ipc_attrs",
            exc=exc,
        )
    if not probe["available"]:
        return {
            "available": False,
            "code": "unsupported_surface_backend",
            "reason": probe["reason"],
            "message": "surface heterogeneous materials require writable pyuipc shell material attrs",
            "missing": probe["missing"],
            "attrs": probe.get("attrs", {}),
        }
    return {
        "available": True,
        "requirements": {
            "surface_shell": True,
            "writable_ipc_attrs": True,
        },
        "attrs": probe["attrs"],
    }


def _surface_position_constraint_attr_probe() -> dict[str, Any]:
    import uipc
    from uipc.constitution import SoftPositionConstraint

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = uipc.geometry.trimesh(vertices, faces)
    SoftPositionConstraint().apply_to(mesh, strength_rate=123.0)

    missing = []
    attrs = {}
    for domain_name, attr_name, expected_len, write_value in SURFACE_POSITION_CONSTRAINT_REQUIRED_ATTRS:
        domain = getattr(mesh, domain_name)()
        attr = domain.find(attr_name)
        key = f"{domain_name}.{attr_name}"
        if not attr:
            missing.append(key)
            continue
        view = uipc.view(attr)
        if view.reshape(-1).shape[0] != expected_len:
            missing.append(key)
            attrs[key] = {"length": int(view.reshape(-1).shape[0]), "expected_length": int(expected_len)}
            continue
        view[:] = write_value.reshape(view.shape)
        attrs[key] = {"length": int(expected_len), "writable": True}

    if missing:
        return {
            "available": False,
            "reason": "missing_surface_position_constraint_ipc_attrs",
            "missing": missing,
            "attrs": attrs,
        }
    return {"available": True, "attrs": attrs}


def surface_position_constraint_status(surface_status: dict[str, Any] | None = None) -> dict[str, Any]:
    if surface_status is None:
        surface_status = surface_backend_status()
    if not surface_status["available"]:
        return {
            "available": False,
            "code": "unsupported_surface_backend",
            "reason": "surface_backend_unavailable",
            "message": "surface position constraints require surface shell backend support",
            "missing": list(surface_status.get("missing", ["surface_shell"])),
            "surface_shell": surface_status,
        }
    try:
        probe = _surface_position_constraint_attr_probe()
    except Exception as exc:
        return _missing_status(
            "broken_surface_position_constraint_attr_probe",
            "surface position constraints require writable pyuipc SoftPositionConstraint attrs",
            requirement="surface_position_constraint_ipc_attrs",
            exc=exc,
        )
    if not probe["available"]:
        return {
            "available": False,
            "code": "unsupported_surface_backend",
            "reason": probe["reason"],
            "message": "surface position constraints require writable pyuipc SoftPositionConstraint attrs",
            "missing": probe["missing"],
            "attrs": probe.get("attrs", {}),
        }
    return {
        "available": True,
        "requirements": {
            "surface_shell": True,
            "writable_soft_position_constraint_attrs": True,
        },
        "attrs": probe["attrs"],
    }


def capability_report(
    required_capabilities: Iterable[str] = (),
    *,
    diagnostic_scene: bool = False,
) -> dict[str, Any]:
    surface_status = surface_backend_status()
    surface_heterogeneous_status = surface_heterogeneous_backend_status(surface_status)
    surface_position_constraint = surface_position_constraint_status(surface_status)
    capabilities = capabilities_for_report(
        bool(surface_status["available"]),
        bool(surface_heterogeneous_status["available"]),
        bool(surface_position_constraint["available"]),
        diagnostic_scene=diagnostic_scene,
    )
    required = tuple(str(capability) for capability in required_capabilities)
    missing_required = [capability for capability in required if capability not in capabilities]
    return {
        "protocol": PROTOCOL,
        "capabilities": list(capabilities),
        "backend_requirements": {
            "surface_shell": surface_status,
            "surface_position_constraints": surface_position_constraint,
            "surface_heterogeneous_material_arrays": surface_heterogeneous_status,
        },
        "missing_required_capabilities": missing_required,
    }
