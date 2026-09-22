from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxEndEffectorController

from .protocol import GenesisLiveError

BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS = 0.1
BOX_EE_IPC_SURFACE_COMPLIANT_STRENGTH_RATE = 100.0
BOX_EE_COMPLIANCE_OVERRIDE_KEYS = frozenset(
    {
        "stiffness",
        "controller_stiffness",
        "strength",
        "strength_rate",
        "constraint_strength",
        "soft_constraint",
        "is_soft_constraint",
    }
)
BOX_EE_DIRECTION_ALIAS_KEYS = frozenset({"direction", "direction_vector", "motion_vector", "axis_vector", "vector"})
BOX_EE_FORCE_LIMITED_MODE = "native_fem_force_limited_box_ee/v1"
BOX_EE_FORCE_LIMITED_CAPABILITY = "native_fem_force_limited_box_ee"


class ActionRegistry:
    def __init__(self):
        self._actions: dict[str, dict[str, Any]] = {}

    def register(self, params: dict[str, Any]) -> dict[str, Any]:
        action_id = str(params.get("action_id") or params.get("probe_id") or f"action_{len(self._actions):04d}")
        spec = dict(params)
        spec["action_id"] = action_id
        self._actions[action_id] = spec
        return {"action_id": action_id, "registered": True, "action": spec.get("action")}

    def get(self, action_id: str) -> dict[str, Any] | None:
        return self._actions.get(action_id)

    def clear(self) -> None:
        self._actions.clear()


def _controller_spec(params: dict[str, Any]) -> dict[str, Any]:
    controllers = params.get("controllers")
    if not isinstance(controllers, list) or len(controllers) != 1 or not isinstance(controllers[0], dict):
        raise GenesisLiveError("invalid_controller_request", "box action requires exactly one controller object")
    return controllers[0]


def _entity_for_action(session, params: dict[str, Any]):
    entity_name = params.get("entity") or params.get("entity_name")
    if entity_name is None:
        return session.default_entity()
    return session.entity_by_name(str(entity_name))


def _reject_compliance_override_keys(params: dict[str, Any], controller_spec: dict[str, Any]) -> None:
    offenders = sorted(
        {f"params.{key}" for key in BOX_EE_COMPLIANCE_OVERRIDE_KEYS if key in params}
        | {f"controllers[0].{key}" for key in BOX_EE_COMPLIANCE_OVERRIDE_KEYS if key in controller_spec}
    )
    if offenders:
        raise GenesisLiveError(
            "invalid_controller_request",
            "BoxEE compliance controls are backend-owned; remove override field(s): " + ", ".join(offenders),
        )


def _validated_motion_axis(params: dict[str, Any], controller_spec: dict[str, Any]) -> str:
    offenders = sorted(
        [f"controllers[0].{key}" for key in BOX_EE_DIRECTION_ALIAS_KEYS if key in controller_spec]
        + [f"params.{key}" for key in BOX_EE_DIRECTION_ALIAS_KEYS if key in params]
    )
    if offenders:
        raise GenesisLiveError(
            "invalid_controller_request",
            "box controller accepts only motion_axis; remove direction alias field(s): " + ", ".join(offenders),
        )
    axis = controller_spec.get("motion_axis")
    if not isinstance(axis, str) or axis not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise GenesisLiveError("invalid_controller_request", "box controller requires one signed cardinal motion_axis")
    return axis


def _box_ee_compliant_policy(entity) -> tuple[bool, float]:
    from genesis.engine.couplers import IPCCoupler
    from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial

    if isinstance(entity.sim.coupler, IPCCoupler) and isinstance(entity.material, ClothMaterial):
        # Existing FEMEntity API transports IPC surface strength-rate through the stiffness keyword.
        return True, BOX_EE_IPC_SURFACE_COMPLIANT_STRENGTH_RATE
    return True, BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS


def _force_limited_policy(session, entity) -> dict[str, object] | None:
    policy = getattr(session, "_scene_config", {}).get("box_ee_controller_policy")
    if policy is None:
        return None
    required = {"mode", "policy_id", "total_stiffness_n_per_m", "max_net_spring_force_n"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise GenesisLiveError("invalid_controller_policy", "box_ee_controller_policy has unexpected fields")
    if (
        policy["mode"] != BOX_EE_FORCE_LIMITED_MODE
        or not isinstance(policy["policy_id"], str)
        or not policy["policy_id"]
    ):
        raise GenesisLiveError("invalid_controller_policy", "box_ee_controller_policy identity is invalid")
    for key in ("total_stiffness_n_per_m", "max_net_spring_force_n"):
        value = policy[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise GenesisLiveError(
                "invalid_controller_policy", f"box_ee_controller_policy.{key} must be finite and > 0"
            )
    from genesis.engine.couplers import IPCCoupler

    if isinstance(entity.sim.coupler, IPCCoupler):
        raise GenesisLiveError(
            "unsupported_controller_policy", "force-limited BoxEE currently supports native FEM only"
        )
    normalized = {
        "mode": policy["mode"],
        "policy_id": policy["policy_id"],
        "total_stiffness_n_per_m": float(policy["total_stiffness_n_per_m"]),
        "max_net_spring_force_n": float(policy["max_net_spring_force_n"]),
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **normalized,
        "policy_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "force_scope": {
            "quantity": "net_spring_force",
            "formula": "||sum_i f_i||",
            "excludes": ["sum_of_magnitudes", "stress", "torque", "contact_force", "internal_force"],
        },
    }


def apply_probe_action(session, params: dict[str, Any]) -> dict[str, Any]:
    action = params.get("action")
    if action is None and params.get("action_id"):
        registered = session.actions.get(str(params["action_id"]))
        if registered is None:
            raise GenesisLiveError("unknown_action", f"unknown registered action: {params['action_id']}")
        merged = dict(registered)
        merged.update(params)
        params = merged
        action = params.get("action")

    if action == "box_ee_grasp_and_move":
        entity_name, entity = _entity_for_action(session, params)
        controller_spec = _controller_spec(params)
        _reject_compliance_override_keys(params, controller_spec)
        controller_id = str(controller_spec.get("controller_id") or params.get("controller_id") or "box_ee_0")
        aabb_box = controller_spec.get("aabb_box")
        if aabb_box is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires aabb_box")
        distance_scale = controller_spec.get("distance_scale")
        if distance_scale is None:
            raise GenesisLiveError("invalid_controller_request", "box controller requires distance_scale")
        motion_axis = _validated_motion_axis(params, controller_spec)
        duration_steps = int(params.get("duration_steps", controller_spec.get("duration_steps", 1)))
        box_tolerance = None
        if isinstance(aabb_box, dict):
            box_tolerance = aabb_box.get("selection_tolerance", aabb_box.get("tolerance", aabb_box.get("atol")))
        selection_tolerance = controller_spec.get(
            "selection_tolerance",
            controller_spec.get(
                "tolerance",
                controller_spec.get("atol", params.get("selection_tolerance", params.get("atol", box_tolerance))),
            ),
        )

        force_limited_policy = _force_limited_policy(session, entity)
        controller = BoxEndEffectorController(
            entity,
            controller_id=controller_id,
            force_limited_policy=force_limited_policy,
        )
        frame = aabb_box.get("frame", "env_local") if isinstance(aabb_box, dict) else "env_local"
        is_soft_constraint, compliance_value = _box_ee_compliant_policy(entity)
        if force_limited_policy is not None:
            compliance_value = float(force_limited_policy["total_stiffness_n_per_m"])
        try:
            state = controller.grasp_and_move_cardinal_axis(
                aabb_box,
                frame=frame,
                motion_axis=motion_axis,
                distance_scale=float(distance_scale),
                duration_steps=duration_steps,
                speed=float(controller_spec.get("speed", params.get("speed", 0.6))),
                max_distance_scale=float(
                    controller_spec.get("max_distance_scale", params.get("max_distance_scale", 1.0))
                ),
                is_soft_constraint=is_soft_constraint,
                stiffness=compliance_value,
                selection_tolerance=None if selection_tolerance is None else float(selection_tolerance),
            )
        except gs.GenesisException as exc:
            message = str(exc)
            code = "probe_selection_failed" if "selected no FEM vertices" in message else "probe_action_failed"
            raise GenesisLiveError(
                code,
                message,
                details={"action": action, "entity": entity_name, "controller_id": controller_id},
            ) from exc
        measurement = params.get("measurement")
        if measurement is not None:
            if not isinstance(measurement, dict):
                raise GenesisLiveError("invalid_probe_measurement", "measurement must be an object")
            active_measurement = session.prepare_probe_measurement(
                measurement=measurement,
                entity_name=entity_name,
                controller_id=controller_id,
                target_vertices=state.selected_vertices,
            )
            session.publish_probe_measurement(controller_id, controller, active_measurement)
        else:
            session.controllers[controller_id] = controller
        return {
            "action": action,
            "entity": entity_name,
            "controller_id": controller_id,
            "controller_state": state.to_dict(),
            "selected_vertex_count": state.selected_vertex_count,
            "selected_vertices": state.selected_vertices.tolist(),
            "motion_axis": state.motion_axis,
        }

    if action == "probe_release":
        controller_id = str(params.get("controller_id", "")).strip()
        if not controller_id:
            raise GenesisLiveError("invalid_probe_measurement", "probe_release requires an explicit controller_id")
        controller = session.controllers.get(controller_id)
        if controller is None:
            raise GenesisLiveError("unknown_controller", f"unknown controller: {controller_id}")
        state = controller.release()
        if controller_id in session.active_measurement_by_controller:
            session.release_probe_measurement(controller_id, controller_state=state.to_dict())
        session.controllers.pop(controller_id, None)
        return {"action": action, "controller_id": controller_id, "controller_state": state.to_dict()}

    raise GenesisLiveError("unsupported_action", f"unsupported probe action: {action}")
