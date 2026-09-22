from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array


SUPPORTED_AABB_FRAMES = ("object_local", "env_local", "world")


def _float_dtype():
    return gs.np_float if gs._initialized else np.float32


def _int_dtype():
    return gs.np_int if gs._initialized else np.int32


def _as_float_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=_float_dtype())
    if not np.all(np.isfinite(array)):
        gs.raise_exception(f"{name} must contain only finite values.")
    return array


def normalize_aabb_box(aabb_box) -> np.ndarray:
    if isinstance(aabb_box, Mapping):
        if "box" in aabb_box:
            aabb_box = aabb_box["box"]
        elif "min" in aabb_box and "max" in aabb_box:
            aabb_box = [*aabb_box["min"], *aabb_box["max"]]
        else:
            gs.raise_exception("AABB mapping must contain either 'box' or both 'min' and 'max'.")

    box = _as_float_array(aabb_box, "AABB box").reshape((-1,))
    if box.shape != (6,):
        gs.raise_exception(f"AABB box must have shape (6,), got {box.shape}.")

    mins = box[:3]
    maxs = box[3:]
    if np.any(mins > maxs):
        gs.raise_exception("AABB min coordinates must be less than or equal to max coordinates.")

    return box.astype(_float_dtype(), copy=False)


def aabb_bounds(aabb_box) -> tuple[np.ndarray, np.ndarray]:
    box = normalize_aabb_box(aabb_box)
    return box[:3], box[3:]


def _aabb_corners(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=_float_dtype(),
    )


def _apply_transform_to_points(points: np.ndarray, transform, name: str) -> np.ndarray:
    if transform is None:
        return points

    transform = _as_float_array(transform, name)
    if transform.shape == (3,):
        return points + transform.reshape((1, 3))

    if transform.shape != (4, 4):
        gs.raise_exception(f"{name} must be a translation vector with shape (3,) or an affine matrix with shape (4, 4).")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=_float_dtype())):
        gs.raise_exception(f"{name} must be an affine transform with final row [0, 0, 0, 1].")

    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=_float_dtype())], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def transform_aabb(aabb_box, transform, name: str = "AABB transform") -> np.ndarray:
    mins, maxs = aabb_bounds(aabb_box)
    transformed = _apply_transform_to_points(_aabb_corners(mins, maxs), transform, name)
    return np.concatenate([transformed.min(axis=0), transformed.max(axis=0)]).astype(_float_dtype(), copy=False)


def aabb_to_env_local(aabb_box, frame: str = "env_local", *, object_to_env=None, world_to_env=None) -> np.ndarray:
    if frame not in SUPPORTED_AABB_FRAMES:
        gs.raise_exception(f"Unsupported AABB frame '{frame}'. Expected one of {SUPPORTED_AABB_FRAMES}.")

    if frame == "env_local":
        return normalize_aabb_box(aabb_box)
    if frame == "object_local":
        return transform_aabb(aabb_box, object_to_env, "object_to_env")
    return transform_aabb(aabb_box, world_to_env, "world_to_env")


def select_vertices_in_aabb(
    positions,
    aabb_box,
    *,
    frame: str = "env_local",
    object_to_env=None,
    world_to_env=None,
    atol: float = 0.0,
) -> np.ndarray:
    positions = _as_float_array(positions, "positions")
    if positions.ndim != 2 or positions.shape[1] != 3:
        gs.raise_exception(f"positions must have shape (n_vertices, 3), got {positions.shape}.")
    if atol < 0.0:
        gs.raise_exception("AABB selection tolerance must be non-negative.")

    mins, maxs = aabb_bounds(aabb_to_env_local(aabb_box, frame, object_to_env=object_to_env, world_to_env=world_to_env))
    mask = np.all((positions >= mins - atol) & (positions <= maxs + atol), axis=1)
    return np.nonzero(mask)[0].astype(_int_dtype(), copy=False)


def fem_entity_positions(entity, *, env_idx: int = 0, use_current: bool = True) -> np.ndarray:
    if use_current and getattr(entity, "active", False):
        positions = tensor_to_array(entity.get_state().pos)
    else:
        positions = tensor_to_array(entity.init_positions)

    positions = np.asarray(positions, dtype=_float_dtype())
    if positions.ndim == 3:
        if env_idx < 0 or env_idx >= positions.shape[0]:
            gs.raise_exception(f"env_idx {env_idx} is out of range for {positions.shape[0]} environments.")
        positions = positions[env_idx]
    if positions.ndim != 2 or positions.shape[1] != 3:
        gs.raise_exception(f"FEM entity positions must resolve to shape (n_vertices, 3), got {positions.shape}.")
    return positions


def select_fem_vertices_in_aabb(
    entity,
    aabb_box,
    *,
    frame: str = "env_local",
    env_idx: int = 0,
    use_current: bool = True,
    object_to_env=None,
    world_to_env=None,
    atol: float = 0.0,
) -> np.ndarray:
    positions = fem_entity_positions(entity, env_idx=env_idx, use_current=use_current)
    return select_vertices_in_aabb(
        positions,
        aabb_box,
        frame=frame,
        object_to_env=object_to_env,
        world_to_env=world_to_env,
        atol=atol,
    )
