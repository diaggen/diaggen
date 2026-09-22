"""Immutable public records for SAP rigid--FEM contact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class RigidFEMContactError(RuntimeError):
    """Base class for public rigid--FEM contact query errors."""


class RigidFEMContactUnavailableError(RigidFEMContactError):
    """Raised when a scene has no usable SAP rigid--FEM subsystem."""


class RigidFEMContactNotReadyError(RigidFEMContactError):
    """Raised when no SAP substep has completed since build or reset."""


class RigidFEMContactMode(str, Enum):
    """Final SAP contact mode at the solved substep velocity."""

    STICK = "STICK"
    SLIDE = "SLIDE"
    NO_CONTACT = "NO_CONTACT"


@dataclass(frozen=True, slots=True)
class RigidFEMWhitelistEntry:
    """Resolved coupler-collision policy for one rigid entity."""

    rigid_entity_idx: int
    rigid_entity_name: str
    collision_enabled: bool
    requested_link_names: tuple[str, ...] | None
    resolved_link_indices: tuple[int, ...]
    resolved_link_names: tuple[str, ...]
    resolved_geom_indices: tuple[int, ...]
    enabled_face_count: int
    total_face_count: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.rigid_entity_idx, "rigid_entity_idx")
        _require_name(self.rigid_entity_name, "rigid_entity_name")
        if type(self.collision_enabled) is not bool:
            raise TypeError("collision_enabled must be bool")
        if self.requested_link_names is not None:
            _require_names(self.requested_link_names, None, "requested_link_names")
        _require_indices(self.resolved_link_indices, "resolved_link_indices")
        _require_names(self.resolved_link_names, len(self.resolved_link_indices), "resolved_link_names")
        _require_indices(self.resolved_geom_indices, "resolved_geom_indices")
        _require_nonnegative_int(self.enabled_face_count, "enabled_face_count")
        _require_nonnegative_int(self.total_face_count, "total_face_count")
        if self.enabled_face_count > self.total_face_count:
            raise ValueError("enabled_face_count exceeds total_face_count")
        if not self.collision_enabled and self.enabled_face_count != 0:
            raise ValueError("a collision-disabled entity cannot have enabled faces")


@dataclass(frozen=True, slots=True)
class RigidFEMWhitelistReceipt:
    """Immutable build-time receipt for the SAP rigid-face allowlist."""

    entries: tuple[RigidFEMWhitelistEntry, ...]
    enabled_face_count: int
    total_face_count: int
    face_enabled_by_index: Mapping[int, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, RigidFEMWhitelistEntry) for item in self.entries
        ):
            raise TypeError("entries must be a tuple of RigidFEMWhitelistEntry")
        if len({item.rigid_entity_idx for item in self.entries}) != len(self.entries):
            raise ValueError("receipt rigid entity indices must be unique")
        if len({item.rigid_entity_name for item in self.entries}) != len(self.entries):
            raise ValueError("receipt rigid entity names must be unique")
        _require_nonnegative_int(self.enabled_face_count, "enabled_face_count")
        _require_nonnegative_int(self.total_face_count, "total_face_count")
        if self.enabled_face_count > self.total_face_count:
            raise ValueError("enabled_face_count exceeds total_face_count")
        if sum(item.enabled_face_count for item in self.entries) != self.enabled_face_count:
            raise ValueError("entry enabled-face counts do not match receipt")
        if sum(item.total_face_count for item in self.entries) != self.total_face_count:
            raise ValueError("entry total-face counts do not match receipt")
        if not isinstance(self.face_enabled_by_index, Mapping):
            raise TypeError("face_enabled_by_index must be a mapping")
        canonical = dict(self.face_enabled_by_index)
        if set(canonical) != set(range(self.total_face_count)):
            raise ValueError("face_enabled_by_index must cover every rigid face exactly once")
        if any(type(index) is not int or type(enabled) is not bool for index, enabled in canonical.items()):
            raise TypeError("face_enabled_by_index must map int indices to bool values")
        if sum(canonical.values()) != self.enabled_face_count:
            raise ValueError("face mask enabled count does not match receipt")
        object.__setattr__(self, "face_enabled_by_index", MappingProxyType(canonical))


@dataclass(frozen=True, slots=True)
class RigidFEMContactOwnershipReceipt:
    """Immutable build-time ownership facts exposed before the first step."""

    whitelist_receipt: RigidFEMWhitelistReceipt
    rigid_fem_contact_enabled: bool
    floor_tet_contact_enabled: bool
    floor_height_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.whitelist_receipt, RigidFEMWhitelistReceipt):
            raise TypeError("whitelist_receipt must be a RigidFEMWhitelistReceipt")
        if type(self.rigid_fem_contact_enabled) is not bool or type(self.floor_tet_contact_enabled) is not bool:
            raise TypeError("contact ownership flags must be bool")
        if not np.isfinite(self.floor_height_m):
            raise ValueError("floor_height_m must be finite")
        if not self.rigid_fem_contact_enabled and self.whitelist_receipt.enabled_face_count:
            raise ValueError("disabled rigid--FEM contact cannot expose enabled faces")


@dataclass(frozen=True, slots=True)
class RigidFEMContactBatch:
    """A fresh immutable copy of the last completed SAP substep contacts."""

    env_idx: np.ndarray
    rigid_entity_idx: np.ndarray
    rigid_link_idx: np.ndarray
    rigid_geom_idx: np.ndarray
    fem_entity_idx: np.ndarray
    fem_element_idx_local: np.ndarray
    rigid_entity_names: tuple[str, ...]
    rigid_link_names: tuple[str, ...]
    fem_entity_names: tuple[str, ...]
    point_m: np.ndarray
    normal_world: np.ndarray
    signed_gap_m: np.ndarray
    penetration_m: np.ndarray
    normal_impulse_ns: np.ndarray
    tangential_impulse_world_ns: np.ndarray
    relative_tangential_velocity_world_mps: np.ndarray
    modes: tuple[RigidFEMContactMode, ...]
    completed_scene_step: int
    completed_substep: int
    dt_s: float
    whitelist_receipt: RigidFEMWhitelistReceipt

    def __post_init__(self) -> None:
        index_fields = (
            "env_idx",
            "rigid_entity_idx",
            "rigid_link_idx",
            "rigid_geom_idx",
            "fem_entity_idx",
            "fem_element_idx_local",
        )
        count: int | None = None
        for field_name in index_fields:
            value = _immutable_array(getattr(self, field_name), np.dtype("int64"), 1, field_name)
            if np.any(value < 0):
                raise ValueError(f"{field_name} must be nonnegative")
            count = len(value) if count is None else count
            if len(value) != count:
                raise ValueError("all per-contact fields must have equal length")
            object.__setattr__(self, field_name, value)
        assert count is not None

        for field_name in ("rigid_entity_names", "rigid_link_names", "fem_entity_names"):
            _require_names(getattr(self, field_name), count, field_name)
        if not isinstance(self.modes, tuple) or len(self.modes) != count:
            raise TypeError("modes must be a tuple matching the contact count")
        if not all(isinstance(mode, RigidFEMContactMode) for mode in self.modes):
            raise TypeError("modes must contain only RigidFEMContactMode values")

        vector_fields = (
            "point_m",
            "normal_world",
            "tangential_impulse_world_ns",
            "relative_tangential_velocity_world_mps",
        )
        for field_name in vector_fields:
            value = _immutable_array(getattr(self, field_name), np.dtype("float64"), 2, field_name)
            if value.shape != (count, 3):
                raise ValueError(f"{field_name} must have shape ({count}, 3)")
            if not np.isfinite(value).all():
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)

        scalar_fields = ("signed_gap_m", "penetration_m", "normal_impulse_ns")
        for field_name in scalar_fields:
            value = _immutable_array(getattr(self, field_name), np.dtype("float64"), 1, field_name)
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"{field_name} must be a finite array with shape ({count},)")
            object.__setattr__(self, field_name, value)

        if count:
            if not np.allclose(np.linalg.norm(self.normal_world, axis=1), 1.0, rtol=0.0, atol=1e-10):
                raise ValueError("normal_world rows must be unit vectors")
            if np.any(self.penetration_m < 0.0) or not np.allclose(
                self.penetration_m, np.maximum(-self.signed_gap_m, 0.0), rtol=0.0, atol=1e-12
            ):
                raise ValueError("penetration_m must equal max(-signed_gap_m, 0)")
            if np.any(self.normal_impulse_ns < 0.0):
                raise ValueError("normal_impulse_ns must be nonnegative")
            for field_name in ("tangential_impulse_world_ns", "relative_tangential_velocity_world_mps"):
                tangent = getattr(self, field_name)
                if not np.allclose(np.sum(tangent * self.normal_world, axis=1), 0.0, rtol=0.0, atol=1e-9):
                    raise ValueError(f"{field_name} must be tangent to normal_world")
            no_contact = np.array([mode is RigidFEMContactMode.NO_CONTACT for mode in self.modes], dtype=bool)
            if np.any(self.normal_impulse_ns[no_contact] > 1e-10):
                raise ValueError("NO_CONTACT rows cannot carry normal impulse")
            if np.any(np.linalg.norm(self.tangential_impulse_world_ns[no_contact], axis=1) > 1e-10):
                raise ValueError("NO_CONTACT rows cannot carry tangential impulse")

        _require_nonnegative_int(self.completed_scene_step, "completed_scene_step")
        _require_nonnegative_int(self.completed_substep, "completed_substep")
        if type(self.dt_s) is not float or not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be a finite positive float")
        if not isinstance(self.whitelist_receipt, RigidFEMWhitelistReceipt):
            raise TypeError("whitelist_receipt must be RigidFEMWhitelistReceipt")
        receipt_by_entity = {entry.rigid_entity_idx: entry for entry in self.whitelist_receipt.entries}
        for row in range(count):
            entry = receipt_by_entity.get(int(self.rigid_entity_idx[row]))
            if entry is None or entry.rigid_entity_name != self.rigid_entity_names[row]:
                raise ValueError("rigid contact identity is absent from the whitelist receipt")
            link_identity = (int(self.rigid_link_idx[row]), self.rigid_link_names[row])
            if link_identity not in zip(entry.resolved_link_indices, entry.resolved_link_names, strict=True):
                raise ValueError("rigid contact link is outside the whitelist receipt")
            if int(self.rigid_geom_idx[row]) not in entry.resolved_geom_indices:
                raise ValueError("rigid contact geom is outside the whitelist receipt")

    def __len__(self) -> int:
        return len(self.env_idx)


def _immutable_array(value: object, dtype: np.dtype, ndim: int, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != ndim:
        raise TypeError(f"{label} must be an exact {dtype} ndarray with rank {ndim}")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _require_nonnegative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative int")


def _require_name(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty stripped string")


def _require_names(value: object, count: int | None, label: str) -> None:
    if not isinstance(value, tuple) or (count is not None and len(value) != count):
        raise TypeError(f"{label} must be a tuple with the required length")
    for item in value:
        _require_name(item, label)


def _require_indices(value: object, label: str) -> None:
    if not isinstance(value, tuple) or any(type(item) is not int or item < 0 for item in value):
        raise TypeError(f"{label} must be a tuple of nonnegative ints")
