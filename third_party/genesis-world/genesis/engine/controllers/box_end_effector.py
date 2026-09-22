from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

import genesis as gs
from genesis.utils import spatial_selection as su

DEFAULT_BOX_EE_SPEED = 0.6
DEFAULT_MAX_DISTANCE_SCALE = 1.0
MOTION_AXIS_VECTORS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}


def motion_axis_vector(motion_axis: str) -> np.ndarray:
    if not isinstance(motion_axis, str) or motion_axis not in MOTION_AXIS_VECTORS:
        gs.raise_exception(f"motion_axis must be one of {tuple(MOTION_AXIS_VECTORS)}, got {motion_axis!r}.")
    return np.asarray(MOTION_AXIS_VECTORS[motion_axis], dtype=_float_dtype())


def _float_dtype():
    return gs.np_float if gs._initialized else np.float32


def _envs_idx_arg(entity, env_idx: int):
    if entity.scene.n_envs == 0:
        return None
    return env_idx


def _copy_int_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=gs.np_int if gs._initialized else np.int32).copy()


def _copy_float_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=_float_dtype()).copy()


def _box_spec(spec, default_frame: str):
    frame = default_frame
    box = spec
    if isinstance(spec, Mapping):
        if "aabb_box" in spec:
            nested = spec["aabb_box"]
            if isinstance(nested, Mapping):
                frame = nested.get("frame", frame)
                box = nested
            else:
                box = nested
        else:
            frame = spec.get("frame", frame)
            box = spec
    return su.normalize_aabb_box(box), frame


def _box_tolerance_spec(spec):
    if not isinstance(spec, Mapping):
        return None
    candidates = []
    nested = spec.get("aabb_box")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    candidates.append(spec)
    for candidate in candidates:
        for key in ("selection_tolerance", "tolerance", "atol"):
            if key in candidate:
                return candidate[key]
    return None


def _selection_tolerance(value) -> float:
    if value is None:
        value = 0.0
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        gs.raise_exception(f"AABB selection tolerance must be finite and non-negative, got {value}.")
    return tolerance


def _entity_representation(entity) -> tuple[str, str]:
    from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

    if isinstance(entity.material, ClothMaterial):
        return "surface", "triangle"
    return "volumetric", "tetrahedron"


def _zero_selection_message(
    *,
    kind: str,
    identifier: str,
    entity,
    source_box: np.ndarray,
    env_local_box: np.ndarray,
    frame: str,
    env_idx: int,
    tolerance: float,
    selected_count: int,
) -> str:
    positions = su.fem_entity_positions(entity, env_idx=env_idx, use_current=True)
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    representation, primitive_kind = _entity_representation(entity)
    return (
        f"{kind} '{identifier}' selected no FEM vertices; "
        f"frame={frame}; "
        f"source_box={source_box.tolist()}; "
        f"env_local_box={env_local_box.tolist()}; "
        f"entity_bbox_min={bbox_min.tolist()}; "
        f"entity_bbox_max={bbox_max.tolist()}; "
        f"vertex_count={int(getattr(entity, 'n_vertices', positions.shape[0]))}; "
        f"selected_count={int(selected_count)}; "
        f"tolerance={float(tolerance)}; "
        f"env_idx={int(env_idx)}; "
        f"representation={representation}; "
        f"primitive_kind={primitive_kind}. "
        "Adjust the box bounds/frame or increase selection_tolerance/atol."
    )


@dataclass
class BoxAnchorRecord:
    anchor_id: str
    frame: str
    source_box: np.ndarray
    env_local_box: np.ndarray
    selected_vertices: np.ndarray
    selection_tolerance: float = 0.0
    optional: bool = False

    @property
    def selected_vertex_count(self) -> int:
        return int(self.selected_vertices.shape[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "frame": self.frame,
            "source_box": self.source_box.tolist(),
            "env_local_box": self.env_local_box.tolist(),
            "selected_vertices": self.selected_vertices.tolist(),
            "selected_vertex_count": self.selected_vertex_count,
            "selection_tolerance": float(self.selection_tolerance),
            "optional": self.optional,
        }


@dataclass
class BoxEndEffectorState:
    controller_id: str
    frame: str
    source_box: np.ndarray
    env_local_box: np.ndarray
    selected_vertices: np.ndarray
    target_positions: np.ndarray
    motion_axis: str = "+Y"
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=_float_dtype()))
    selection_tolerance: float = 0.0
    duration_steps: int = 0
    speed: float = DEFAULT_BOX_EE_SPEED
    distance_scale: float = 0.0
    distance: float = 0.0
    moved_distance: float = 0.0
    estimated_motion_steps: int = 0
    active: bool = False
    motion_active: bool = False
    optional: bool = False
    controller_policy: dict[str, object] | None = None
    controller_telemetry: dict[str, object] | None = None

    @property
    def selected_vertex_count(self) -> int:
        return int(self.selected_vertices.shape[0])

    def to_dict(self) -> dict[str, object]:
        payload = {
            "controller_id": self.controller_id,
            "frame": self.frame,
            "source_box": self.source_box.tolist(),
            "env_local_box": self.env_local_box.tolist(),
            "selected_vertices": self.selected_vertices.tolist(),
            "selected_vertex_count": self.selected_vertex_count,
            "target_positions": self.target_positions.tolist(),
            "motion_axis": self.motion_axis,
            "displacement": self.displacement.tolist(),
            "selection_tolerance": float(self.selection_tolerance),
            "duration_steps": int(self.duration_steps),
            "speed": float(self.speed),
            "distance_scale": float(self.distance_scale),
            "distance": float(self.distance),
            "moved_distance": float(self.moved_distance),
            "estimated_motion_steps": int(self.estimated_motion_steps),
            "active": bool(self.active),
            "motion_active": bool(self.motion_active),
            "optional": bool(self.optional),
        }
        if self.controller_policy is not None:
            payload["controller_policy"] = copy.deepcopy(self.controller_policy)
        if self.controller_telemetry is not None:
            payload["controller_telemetry"] = copy.deepcopy(self.controller_telemetry)
        return payload


def _selected_target_positions(entity, selected_vertices: np.ndarray, env_idx: int) -> np.ndarray:
    positions = su.fem_entity_positions(entity, env_idx=env_idx, use_current=True)
    return positions[selected_vertices]


def _empty_targets() -> np.ndarray:
    return np.empty((0, 3), dtype=_float_dtype())


def _constraint_env_idx(entity, env_idx: int) -> int:
    if entity.scene.n_envs == 0:
        return 0
    return env_idx


def _estimate_motion_steps(distance: float, speed: float, dt: float) -> int:
    per_step = speed * dt
    if not np.isfinite(per_step) or per_step <= 0.0:
        gs.raise_exception("BoxEE motion requires positive finite speed * dt.")
    return max(1, int(np.ceil(distance / per_step)))


def apply_static_box_anchors(
    entity,
    anchors,
    *,
    frame: str = "object_local",
    env_idx: int = 0,
    object_to_env=None,
    world_to_env=None,
    is_soft_constraint: bool = False,
    stiffness: float = 0.0,
    use_current_positions: bool = False,
    selection_tolerance: float = 0.0,
    atol: float | None = None,
) -> list[BoxAnchorRecord]:
    records: list[BoxAnchorRecord] = []
    default_tolerance = _selection_tolerance(selection_tolerance if atol is None else atol)
    for i, anchor in enumerate(anchors):
        source_box, anchor_frame = _box_spec(anchor, frame)
        optional = bool(anchor.get("optional", False)) if isinstance(anchor, Mapping) else False
        if isinstance(anchor, Mapping):
            anchor_id = str(anchor.get("anchor_id", anchor.get("id", f"anchor_{i}")))
            anchor_tolerance = _selection_tolerance(
                anchor.get(
                    "selection_tolerance",
                    anchor.get("tolerance", anchor.get("atol", default_tolerance)),
                )
            )
        else:
            anchor_id = f"anchor_{i}"
            anchor_tolerance = default_tolerance
        env_local_box = su.aabb_to_env_local(
            source_box,
            anchor_frame,
            object_to_env=object_to_env,
            world_to_env=world_to_env,
        )
        selected = su.select_fem_vertices_in_aabb(
            entity,
            source_box,
            frame=anchor_frame,
            env_idx=env_idx,
            use_current=use_current_positions,
            object_to_env=object_to_env,
            world_to_env=world_to_env,
            atol=anchor_tolerance,
        )
        if selected.size == 0 and not optional:
            gs.raise_exception(
                _zero_selection_message(
                    kind="Static box anchor",
                    identifier=anchor_id,
                    entity=entity,
                    source_box=source_box,
                    env_local_box=env_local_box,
                    frame=anchor_frame,
                    env_idx=env_idx,
                    tolerance=anchor_tolerance,
                    selected_count=selected.size,
                )
            )

        if selected.size > 0:
            targets = _selected_target_positions(entity, selected, env_idx)
            entity.set_vertex_constraints(
                selected,
                target_poss=targets,
                is_soft_constraint=is_soft_constraint,
                stiffness=stiffness,
                envs_idx=_envs_idx_arg(entity, env_idx),
            )

        records.append(
            BoxAnchorRecord(
                anchor_id=anchor_id,
                frame=anchor_frame,
                source_box=_copy_float_array(source_box),
                env_local_box=_copy_float_array(env_local_box),
                selected_vertices=_copy_int_array(selected),
                selection_tolerance=anchor_tolerance,
                optional=optional,
            )
        )
    return records


class BoxEndEffectorController:
    def __init__(
        self,
        entity,
        *,
        controller_id: str = "box_ee_0",
        env_idx: int = 0,
        object_to_env=None,
        world_to_env=None,
        force_limited_policy: Mapping[str, object] | None = None,
    ):
        self.entity = entity
        self.controller_id = controller_id
        self.env_idx = env_idx
        self.object_to_env = object_to_env
        self.world_to_env = world_to_env
        self._force_limited_policy = (
            copy.deepcopy(dict(force_limited_policy)) if force_limited_policy is not None else None
        )
        self._force_limited_telemetry: dict[str, object] | None = None
        self._state = BoxEndEffectorState(
            controller_id=controller_id,
            frame="env_local",
            source_box=np.zeros(6, dtype=_float_dtype()),
            env_local_box=np.zeros(6, dtype=_float_dtype()),
            selected_vertices=np.empty((0,), dtype=gs.np_int if gs._initialized else np.int32),
            target_positions=_empty_targets(),
            controller_policy=self._force_limited_policy,
        )
        self._preexisting_constraint_mask = np.empty((0,), dtype=bool)
        self._preexisting_constraint_targets = _empty_targets()
        self._preexisting_constraint_soft = np.empty((0,), dtype=bool)
        self._preexisting_constraint_stiffness = np.empty((0,), dtype=_float_dtype())
        self._motion_start_targets = _empty_targets()
        self._motion_direction = np.array([0.0, 1.0, 0.0], dtype=_float_dtype())
        self._motion_distance = 0.0
        self._motion_moved_distance = 0.0
        self._motion_speed = DEFAULT_BOX_EE_SPEED
        self._motion_estimated_steps = 0

    def _state_policy_fields(self) -> dict[str, object]:
        return {
            "controller_policy": self._force_limited_policy,
            "controller_telemetry": self._force_limited_telemetry,
        }

    @property
    def state(self) -> BoxEndEffectorState:
        return self._state

    def snapshot(self) -> dict[str, object]:
        return self._state.to_dict()

    def _uses_ipc_surface_constraints(self) -> bool:
        from genesis.engine.couplers import IPCCoupler
        from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

        return isinstance(self.entity.sim.coupler, IPCCoupler) and isinstance(self.entity.material, ClothMaterial)

    def _capture_preexisting_constraints(self, selected_vertices: np.ndarray) -> None:
        if self._uses_ipc_surface_constraints():
            query = self.entity.sim.coupler.query_surface_vertex_constraints(
                self.entity,
                selected_vertices,
                envs_idx=_envs_idx_arg(self.entity, self.env_idx),
            )
            mask = np.asarray(query["is_constrained"], dtype=bool)
            self._preexisting_constraint_mask = mask
            self._preexisting_constraint_targets = np.asarray(query["target_poss"], dtype=_float_dtype())[mask].copy()
            self._preexisting_constraint_soft = np.ones(np.count_nonzero(mask), dtype=bool)
            self._preexisting_constraint_stiffness = np.asarray(query["strength"], dtype=_float_dtype())[mask].copy()
            return

        if not self.entity.solver._constraints_initialized:
            self._preexisting_constraint_mask = np.zeros(selected_vertices.shape, dtype=bool)
            self._preexisting_constraint_targets = _empty_targets()
            self._preexisting_constraint_soft = np.empty((0,), dtype=bool)
            self._preexisting_constraint_stiffness = np.empty((0,), dtype=_float_dtype())
            return

        global_vertices = selected_vertices + self.entity.v_start
        env_idx = _constraint_env_idx(self.entity, self.env_idx)
        constraints = self.entity.solver.vertex_constraints
        mask = constraints.is_constrained.to_numpy()[global_vertices, env_idx].astype(bool)
        self._preexisting_constraint_mask = mask
        self._preexisting_constraint_targets = constraints.target_pos.to_numpy()[global_vertices[mask], env_idx].copy()
        self._preexisting_constraint_soft = constraints.is_soft_constraint.to_numpy()[
            global_vertices[mask], env_idx
        ].astype(bool)
        self._preexisting_constraint_stiffness = constraints.stiffness.to_numpy()[
            global_vertices[mask], env_idx
        ].astype(
            _float_dtype(),
            copy=True,
        )

    def _clear_preexisting_constraints(self) -> None:
        self._preexisting_constraint_mask = np.empty((0,), dtype=bool)
        self._preexisting_constraint_targets = _empty_targets()
        self._preexisting_constraint_soft = np.empty((0,), dtype=bool)
        self._preexisting_constraint_stiffness = np.empty((0,), dtype=_float_dtype())

    def _clear_motion(self) -> None:
        self._motion_start_targets = _empty_targets()
        self._motion_distance = 0.0
        self._motion_moved_distance = 0.0
        self._motion_speed = DEFAULT_BOX_EE_SPEED
        self._motion_estimated_steps = 0

    def _restore_preexisting_constraints(self, selected_vertices: np.ndarray) -> None:
        if self._preexisting_constraint_mask.size == 0 or not np.any(self._preexisting_constraint_mask):
            return

        restore_vertices = selected_vertices[self._preexisting_constraint_mask]
        for is_soft in np.unique(self._preexisting_constraint_soft):
            soft_mask = self._preexisting_constraint_soft == is_soft
            for stiffness in np.unique(self._preexisting_constraint_stiffness[soft_mask]):
                group_mask = soft_mask & (self._preexisting_constraint_stiffness == stiffness)
                self.entity.set_vertex_constraints(
                    restore_vertices[group_mask],
                    target_poss=self._preexisting_constraint_targets[group_mask],
                    is_soft_constraint=bool(is_soft),
                    stiffness=float(stiffness),
                    envs_idx=_envs_idx_arg(self.entity, self.env_idx),
                )

    def grasp(
        self,
        aabb_box,
        *,
        frame: str = "env_local",
        optional: bool = False,
        is_soft_constraint: bool = False,
        stiffness: float = 0.0,
        selection_tolerance: float | None = None,
        atol: float | None = None,
    ) -> BoxEndEffectorState:
        if self._state.active:
            gs.raise_exception(f"BoxEE controller '{self.controller_id}' already has an active grasp.")

        source_box, frame = _box_spec(aabb_box, frame)
        tolerance_value = atol if atol is not None else selection_tolerance
        if tolerance_value is None:
            tolerance_value = _box_tolerance_spec(aabb_box)
        tolerance = _selection_tolerance(tolerance_value)
        env_local_box = su.aabb_to_env_local(
            source_box,
            frame,
            object_to_env=self.object_to_env,
            world_to_env=self.world_to_env,
        )
        selected = su.select_fem_vertices_in_aabb(
            self.entity,
            source_box,
            frame=frame,
            env_idx=self.env_idx,
            use_current=True,
            object_to_env=self.object_to_env,
            world_to_env=self.world_to_env,
            atol=tolerance,
        )

        if selected.size == 0:
            if not optional:
                gs.raise_exception(
                    _zero_selection_message(
                        kind="BoxEE controller",
                        identifier=self.controller_id,
                        entity=self.entity,
                        source_box=source_box,
                        env_local_box=env_local_box,
                        frame=frame,
                        env_idx=self.env_idx,
                        tolerance=tolerance,
                        selected_count=selected.size,
                    )
                )
            self._state = BoxEndEffectorState(
                controller_id=self.controller_id,
                frame=frame,
                source_box=_copy_float_array(source_box),
                env_local_box=_copy_float_array(env_local_box),
                selected_vertices=_copy_int_array(selected),
                target_positions=_empty_targets(),
                selection_tolerance=tolerance,
                active=False,
                optional=True,
                **self._state_policy_fields(),
            )
            self._clear_preexisting_constraints()
            return self._state

        targets = _selected_target_positions(self.entity, selected, self.env_idx)
        if self._force_limited_policy is not None:
            if not is_soft_constraint:
                gs.raise_exception("Force-limited BoxEE requires native FEM soft constraints.")
            stiffness = float(self._force_limited_policy["total_stiffness_n_per_m"]) / float(selected.size)
        self._capture_preexisting_constraints(selected)
        self.entity.set_vertex_constraints(
            selected,
            target_poss=targets,
            is_soft_constraint=is_soft_constraint,
            stiffness=stiffness,
            envs_idx=_envs_idx_arg(self.entity, self.env_idx),
        )
        self._state = BoxEndEffectorState(
            controller_id=self.controller_id,
            frame=frame,
            source_box=_copy_float_array(source_box),
            env_local_box=_copy_float_array(env_local_box),
            selected_vertices=_copy_int_array(selected),
            target_positions=_copy_float_array(targets),
            selection_tolerance=tolerance,
            active=True,
            motion_active=False,
            optional=optional,
            **self._state_policy_fields(),
        )
        self._clear_motion()
        return self._state

    def _scene_dt(self, dt: float | None) -> float:
        if dt is None:
            dt = float(self.entity.scene.dt)
        if not np.isfinite(dt) or dt <= 0.0:
            gs.raise_exception(f"BoxEE motion dt must be positive, got {dt}.")
        return float(dt)

    def move_cardinal_axis(
        self,
        *,
        motion_axis: str,
        distance_scale: float,
        duration_steps: int,
        speed: float = DEFAULT_BOX_EE_SPEED,
        max_distance_scale: float = DEFAULT_MAX_DISTANCE_SCALE,
        dt: float | None = None,
    ) -> BoxEndEffectorState:
        if not self._state.active or self._state.selected_vertex_count == 0:
            gs.raise_exception(f"BoxEE controller '{self.controller_id}' has no active grasp to move.")
        if self._state.motion_active:
            gs.raise_exception(f"BoxEE controller '{self.controller_id}' already has an active motion.")
        if not np.isfinite(distance_scale) or distance_scale <= 0.0 or distance_scale > max_distance_scale:
            gs.raise_exception(f"distance_scale must be finite and in (0, {max_distance_scale}], got {distance_scale}.")
        if not np.isfinite(speed) or speed <= 0.0:
            gs.raise_exception(f"speed must be positive, got {speed}.")
        if int(duration_steps) <= 0:
            gs.raise_exception(f"duration_steps must be positive, got {duration_steps}.")

        dt = self._scene_dt(dt)
        direction = motion_axis_vector(motion_axis)
        mins, maxs = su.aabb_bounds(self._state.env_local_box)
        reference_extent = float(np.max(maxs - mins))
        distance = float(distance_scale * reference_extent)
        if not np.isfinite(distance) or distance <= 0.0:
            gs.raise_exception("distance_scale and AABB max extent must produce a positive move distance.")

        self._motion_start_targets = _copy_float_array(self._state.target_positions)
        self._motion_direction = direction
        self._motion_distance = distance
        self._motion_moved_distance = 0.0
        self._motion_speed = float(speed)
        self._motion_estimated_steps = _estimate_motion_steps(distance, speed, dt)
        self._state = BoxEndEffectorState(
            controller_id=self.controller_id,
            frame=self._state.frame,
            source_box=_copy_float_array(self._state.source_box),
            env_local_box=_copy_float_array(self._state.env_local_box),
            selected_vertices=_copy_int_array(self._state.selected_vertices),
            target_positions=_copy_float_array(self._state.target_positions),
            motion_axis=motion_axis,
            displacement=np.zeros(3, dtype=_float_dtype()),
            selection_tolerance=float(self._state.selection_tolerance),
            duration_steps=int(duration_steps),
            speed=float(speed),
            distance_scale=float(distance_scale),
            distance=distance,
            moved_distance=0.0,
            estimated_motion_steps=self._motion_estimated_steps,
            active=True,
            motion_active=True,
            optional=self._state.optional,
            **self._state_policy_fields(),
        )
        return self._state

    def prepare_step(self, *, dt: float | None = None) -> BoxEndEffectorState:
        """Advance commanded motion and cap the next native-FEM penalty-spring load."""
        self.advance_motion(steps=1, dt=dt)
        if self._force_limited_policy is None or not self._state.active or self._state.selected_vertex_count == 0:
            return self._state

        selected = self._state.selected_vertices
        positions = su.fem_entity_positions(self.entity, env_idx=self.env_idx, use_current=True)[selected]
        errors = self._state.target_positions - positions
        nominal_stiffness = float(self._force_limited_policy["total_stiffness_n_per_m"]) / float(selected.size)
        raw_forces = nominal_stiffness * errors
        raw_net_force = np.sum(raw_forces, axis=0)
        raw_magnitude = float(np.linalg.norm(raw_net_force))
        max_force = float(self._force_limited_policy["max_net_spring_force_n"])
        scale = min(1.0, max_force / raw_magnitude) if raw_magnitude > 0.0 else 1.0
        applied_net_force = raw_net_force * scale
        applied_magnitude = float(np.linalg.norm(applied_net_force))
        if not np.all(np.isfinite(errors)) or not np.isfinite(raw_magnitude) or not np.isfinite(applied_magnitude):
            gs.raise_exception("Force-limited BoxEE produced non-finite spring-force telemetry.")

        self.entity.set_vertex_constraints(
            selected,
            target_poss=self._state.target_positions,
            is_soft_constraint=True,
            stiffness=nominal_stiffness * scale,
            envs_idx=_envs_idx_arg(self.entity, self.env_idx),
        )
        previous = self._force_limited_telemetry or {}
        previous_summary = previous.get("summary", {})
        active_count = int(previous_summary.get("active_pre_step_count", 0)) + 1
        cap_count = int(previous_summary.get("cap_active_pre_step_count", 0)) + int(scale < 1.0)
        force_scope = {
            "quantity": "net_spring_force",
            "formula": "||sum_i f_i||",
            "excludes": ["sum_of_magnitudes", "stress", "torque", "contact_force", "internal_force"],
        }
        self._force_limited_telemetry = {
            "schema": "box_ee_controller_telemetry/v1",
            "policy_hash": self._force_limited_policy["policy_hash"],
            "policy_id": self._force_limited_policy["policy_id"],
            "total_stiffness_n_per_m": float(self._force_limited_policy["total_stiffness_n_per_m"]),
            "max_net_spring_force_n": max_force,
            "selected_vertex_count": int(selected.size),
            "nominal_per_vertex_stiffness_n_per_m": nominal_stiffness,
            "requested_displacement_m": self._state.displacement.astype(float).tolist(),
            "force_scope": force_scope,
            "current": {
                "raw_net_spring_force_vector_n": raw_net_force.astype(float).tolist(),
                "raw_net_spring_force_magnitude_n": raw_magnitude,
                "applied_net_spring_force_vector_n": applied_net_force.astype(float).tolist(),
                "applied_net_spring_force_magnitude_n": applied_magnitude,
                "stiffness_scale": scale,
                "cap_active": bool(scale < 1.0),
            },
            "summary": {
                "peak_raw_net_spring_force_magnitude_n": max(
                    float(previous_summary.get("peak_raw_net_spring_force_magnitude_n", 0.0)), raw_magnitude
                ),
                "peak_applied_net_spring_force_magnitude_n": max(
                    float(previous_summary.get("peak_applied_net_spring_force_magnitude_n", 0.0)), applied_magnitude
                ),
                "cap_active_pre_step_count": cap_count,
                "active_pre_step_count": active_count,
                "numerical_valid": True,
            },
        }
        self._state.controller_telemetry = self._force_limited_telemetry
        return self._state

    def move_positive_y(
        self,
        *,
        distance_scale: float,
        duration_steps: int,
        speed: float = DEFAULT_BOX_EE_SPEED,
        max_distance_scale: float = DEFAULT_MAX_DISTANCE_SCALE,
        dt: float | None = None,
    ) -> BoxEndEffectorState:
        """Compatibility wrapper for callers that predate signed-axis motion."""
        return self.move_cardinal_axis(
            motion_axis="+Y",
            distance_scale=distance_scale,
            duration_steps=duration_steps,
            speed=speed,
            max_distance_scale=max_distance_scale,
            dt=dt,
        )

    def move_positive_y_immediate(
        self,
        *,
        distance_scale: float,
        duration_steps: int,
        speed: float = DEFAULT_BOX_EE_SPEED,
        max_distance_scale: float = DEFAULT_MAX_DISTANCE_SCALE,
        dt: float | None = None,
    ) -> BoxEndEffectorState:
        state = self.move_positive_y(
            distance_scale=distance_scale,
            duration_steps=duration_steps,
            speed=speed,
            max_distance_scale=max_distance_scale,
            dt=dt,
        )
        if state.selected_vertex_count == 0:
            return state
        return self.advance_motion(steps=int(duration_steps), dt=dt)

    def advance_motion(self, *, steps: int = 1, dt: float | None = None) -> BoxEndEffectorState:
        if not self._state.active or not self._state.motion_active:
            return self._state
        if int(steps) < 0:
            gs.raise_exception(f"steps must be non-negative, got {steps}.")
        if int(steps) == 0:
            return self._state

        dt = self._scene_dt(dt)
        self._motion_moved_distance = min(
            self._motion_distance,
            self._motion_moved_distance + self._motion_speed * dt * int(steps),
        )
        displacement = self._motion_direction * self._motion_moved_distance
        targets = self._motion_start_targets + displacement.reshape((1, 3))

        self.entity.update_constraint_targets(
            self._state.selected_vertices,
            targets,
            envs_idx=_envs_idx_arg(self.entity, self.env_idx),
        )
        self._state = BoxEndEffectorState(
            controller_id=self.controller_id,
            frame=self._state.frame,
            source_box=_copy_float_array(self._state.source_box),
            env_local_box=_copy_float_array(self._state.env_local_box),
            selected_vertices=_copy_int_array(self._state.selected_vertices),
            target_positions=_copy_float_array(targets),
            motion_axis=self._state.motion_axis,
            displacement=displacement,
            selection_tolerance=float(self._state.selection_tolerance),
            duration_steps=int(self._state.duration_steps),
            speed=float(self._motion_speed),
            distance_scale=float(self._state.distance_scale),
            distance=float(self._motion_distance),
            moved_distance=float(self._motion_moved_distance),
            estimated_motion_steps=int(self._motion_estimated_steps),
            active=True,
            motion_active=self._motion_moved_distance < self._motion_distance,
            optional=self._state.optional,
            **self._state_policy_fields(),
        )
        if not self._state.motion_active:
            self._clear_motion()
        return self._state

    def grasp_and_move_cardinal_axis(
        self,
        aabb_box,
        *,
        frame: str = "env_local",
        motion_axis: str,
        distance_scale: float,
        duration_steps: int,
        speed: float = DEFAULT_BOX_EE_SPEED,
        max_distance_scale: float = DEFAULT_MAX_DISTANCE_SCALE,
        optional: bool = False,
        is_soft_constraint: bool = False,
        stiffness: float = 0.0,
        selection_tolerance: float | None = None,
        atol: float | None = None,
    ) -> BoxEndEffectorState:
        state = self.grasp(
            aabb_box,
            frame=frame,
            optional=optional,
            is_soft_constraint=is_soft_constraint,
            stiffness=stiffness,
            selection_tolerance=selection_tolerance,
            atol=atol,
        )
        if state.selected_vertex_count == 0:
            return state
        return self.move_cardinal_axis(
            motion_axis=motion_axis,
            distance_scale=distance_scale,
            duration_steps=duration_steps,
            speed=speed,
            max_distance_scale=max_distance_scale,
        )

    def grasp_and_move_positive_y(
        self,
        aabb_box,
        *,
        frame: str = "env_local",
        distance_scale: float,
        duration_steps: int,
        speed: float = DEFAULT_BOX_EE_SPEED,
        max_distance_scale: float = DEFAULT_MAX_DISTANCE_SCALE,
        optional: bool = False,
        is_soft_constraint: bool = False,
        stiffness: float = 0.0,
        selection_tolerance: float | None = None,
        atol: float | None = None,
    ) -> BoxEndEffectorState:
        return self.grasp_and_move_cardinal_axis(
            aabb_box,
            frame=frame,
            motion_axis="+Y",
            distance_scale=distance_scale,
            duration_steps=duration_steps,
            speed=speed,
            max_distance_scale=max_distance_scale,
            optional=optional,
            is_soft_constraint=is_soft_constraint,
            stiffness=stiffness,
            selection_tolerance=selection_tolerance,
            atol=atol,
        )

    def release(self) -> BoxEndEffectorState:
        if self._state.selected_vertex_count > 0:
            selected_vertices = self._state.selected_vertices
            if self._preexisting_constraint_mask.size == selected_vertices.size:
                new_live_vertices = selected_vertices[~self._preexisting_constraint_mask]
            else:
                new_live_vertices = selected_vertices
            if new_live_vertices.size > 0:
                self.entity.remove_vertex_constraints(
                    new_live_vertices,
                    envs_idx=_envs_idx_arg(self.entity, self.env_idx),
                )
            self._restore_preexisting_constraints(selected_vertices)
        self._clear_preexisting_constraints()
        self._clear_motion()
        self._state = BoxEndEffectorState(
            controller_id=self.controller_id,
            frame=self._state.frame,
            source_box=_copy_float_array(self._state.source_box),
            env_local_box=_copy_float_array(self._state.env_local_box),
            selected_vertices=np.empty((0,), dtype=gs.np_int if gs._initialized else np.int32),
            target_positions=_empty_targets(),
            motion_axis=self._state.motion_axis,
            displacement=_copy_float_array(self._state.displacement),
            selection_tolerance=float(self._state.selection_tolerance),
            duration_steps=self._state.duration_steps,
            speed=self._state.speed,
            distance_scale=self._state.distance_scale,
            active=False,
            optional=self._state.optional,
            **self._state_policy_fields(),
        )
        return self._state
