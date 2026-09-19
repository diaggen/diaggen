"""Diagnostics-owned binding for the calibrated force-limited BoxEE probe.

Genesis owns the generic penalty-spring mechanics; diagnostics owns the concrete calibrated
values, command schedule, evidence contract, and failure semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


DIAGNOSTIC_FORCE_LIMITED_POLICY_MODE = "native_fem_force_limited_box_ee/v1"
DIAGNOSTIC_FORCE_LIMITED_POLICY_ID = "hag4r-diagnostics-force-limited-v1"
DIAGNOSTIC_FORCE_LIMITED_TOTAL_STIFFNESS_N_PER_M = 200.0
DIAGNOSTIC_FORCE_LIMITED_MAX_NET_SPRING_FORCE_N = 5.0
DIAGNOSTIC_FORCE_LIMITED_CAPABILITY = "native_fem_force_limited_box_ee"
DIAGNOSTIC_FORCE_LIMITED_SCHEDULE_SCHEMA_VERSION = "hag4r-diagnostic-force-limited-schedule-v1"
DIAGNOSTIC_FORCE_LIMITED_TELEMETRY_SCHEMA_VERSION = "hag4r-diagnostic-force-limited-telemetry-v1"
DIAGNOSTIC_FORCE_LIMITED_HOLD_STEPS = 20
DIAGNOSTIC_FORCE_LIMITED_RECOVERY_STEPS = 100
DIAGNOSTIC_FORCE_LIMITED_MAX_LOAD_STEPS = 500
DIAGNOSTIC_FORCE_LIMITED_CALIBRATION_PROVENANCE = {
    "schema_version": "hag4r-diagnostic-force-limited-calibration-provenance-v1",
    "confirmed_report_path": "outputs/controller_force_limited_calibration/f2-calibration-v4-confirmation-20260822/calibration_report.json",
    "confirmed_report_sha256": "f3c45772c554ba9b3269d719ebdbe3c8e2492d6d7da690ecd6d166bc8c1abc07",
    "selection_report_path": "outputs/controller_force_limited_calibration/f2-calibration-v4-selection-20260822/calibration_report.json",
    "selection_report_sha256": "56a62f84c166b5c67c76da7492e1c2236ee07ecc8789b4aab35173ccffd26f0f",
    "protocol_hash": "083d0e90b2888fab8b0ba23264ad4c090e5cba1b3d04f21b49ac5d885512c672",
    "gate_contract_hash": "2bcfc53b0450aecbe96cd828a6fe36537ad1f4cc87c9d40bcbf8671019e75e4d",
    "selected_policy": {
        "candidate_id": "k200_f5",
        "total_stiffness_n_per_m": 200.0,
        "max_net_spring_force_n": 5.0,
        "policy_hash": "be9ae85d8c3577f907b8e4cea3a59f1419a72f8733db7da82d15046e95db70ab",
    },
}


def policy_payload() -> dict[str, Any]:
    return {
        "mode": DIAGNOSTIC_FORCE_LIMITED_POLICY_MODE,
        "policy_id": DIAGNOSTIC_FORCE_LIMITED_POLICY_ID,
        "total_stiffness_n_per_m": DIAGNOSTIC_FORCE_LIMITED_TOTAL_STIFFNESS_N_PER_M,
        "max_net_spring_force_n": DIAGNOSTIC_FORCE_LIMITED_MAX_NET_SPRING_FORCE_N,
    }


def policy_hash() -> str:
    canonical = json.dumps(policy_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calibration_provenance() -> dict[str, Any]:
    """Return the immutable F2 selection and confirmation evidence copied into diagnostics."""

    return json.loads(json.dumps(DIAGNOSTIC_FORCE_LIMITED_CALIBRATION_PROVENANCE))


def force_limited_probe_schedule(
    *,
    aabb_box: list[float] | tuple[float, ...],
    distance_scale: float,
    speed_m_s: float,
    timestep_s: float,
) -> dict[str, Any]:
    """Derive the fixed load window from immutable command geometry only."""

    if len(aabb_box) != 6 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in aabb_box):
        raise ValueError("force-limited diagnostic AABB must be six finite numbers")
    if any(float(aabb_box[axis]) >= float(aabb_box[axis + 3]) for axis in range(3)):
        raise ValueError("force-limited diagnostic AABB must have positive extents")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
        for value in (distance_scale, speed_m_s, timestep_s)
    ):
        raise ValueError("force-limited diagnostic distance scale, speed, and timestep must be finite and > 0")
    reference_extent_m = max(float(aabb_box[axis + 3]) - float(aabb_box[axis]) for axis in range(3))
    command_distance_m = float(distance_scale) * reference_extent_m
    motion_steps = max(1, math.ceil(command_distance_m / (float(speed_m_s) * float(timestep_s))))
    load_steps = motion_steps + DIAGNOSTIC_FORCE_LIMITED_HOLD_STEPS
    if load_steps > DIAGNOSTIC_FORCE_LIMITED_MAX_LOAD_STEPS:
        raise ValueError(
            "force-limited diagnostic command exceeds the frozen load-window cap: "
            f"motion_steps={motion_steps}, hold_steps={DIAGNOSTIC_FORCE_LIMITED_HOLD_STEPS}, "
            f"max_load_steps={DIAGNOSTIC_FORCE_LIMITED_MAX_LOAD_STEPS}"
        )
    return {
        "schema_version": DIAGNOSTIC_FORCE_LIMITED_SCHEDULE_SCHEMA_VERSION,
        "motion_steps": motion_steps,
        "hold_steps": DIAGNOSTIC_FORCE_LIMITED_HOLD_STEPS,
        "load_steps": load_steps,
        "recovery_steps": DIAGNOSTIC_FORCE_LIMITED_RECOVERY_STEPS,
        "command_distance_m": command_distance_m,
        "reference_extent_m": reference_extent_m,
        "speed_m_s": float(speed_m_s),
        "timestep_s": float(timestep_s),
        "response_adaptation": False,
    }


def _controller_action_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Genesis probe action payload is missing")
    nested = value.get("probe")
    return nested if isinstance(nested, Mapping) else value


def controller_state_from_action(value: Any) -> dict[str, Any]:
    action = _controller_action_payload(value)
    state = action.get("controller_state")
    if not isinstance(state, Mapping):
        raise ValueError("Genesis probe action is missing controller_state")
    return dict(state)


def validate_controller_policy(controller_state: Mapping[str, Any]) -> dict[str, Any]:
    policy = controller_state.get("controller_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Genesis controller_state is missing the force-limited policy")
    expected = policy_payload()
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Genesis controller policy does not match the diagnostics-owned calibrated policy")
    if policy.get("policy_hash") != policy_hash():
        raise ValueError("Genesis controller policy hash does not match the diagnostics-owned policy")
    return dict(policy)


def completed_controller_telemetry(
    *,
    controller_state: Mapping[str, Any],
    schedule: Mapping[str, Any],
    completed_measurement: Any,
) -> dict[str, Any]:
    """Validate and compact the final telemetry produced at runtime release."""

    policy = validate_controller_policy(controller_state)
    telemetry = controller_state.get("controller_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("Genesis completed force-limited probe is missing controller telemetry")
    if telemetry.get("schema") != "box_ee_controller_telemetry/v1":
        raise ValueError("Genesis completed controller telemetry has an unsupported schema")
    if telemetry.get("policy_hash") != policy_hash() or telemetry.get("policy_id") != DIAGNOSTIC_FORCE_LIMITED_POLICY_ID:
        raise ValueError("Genesis completed controller telemetry policy identity is inconsistent")
    if telemetry.get("selected_vertex_count") is None or int(telemetry["selected_vertex_count"]) <= 0:
        raise ValueError("Genesis completed controller telemetry has no selected vertices")
    if telemetry.get("total_stiffness_n_per_m") != DIAGNOSTIC_FORCE_LIMITED_TOTAL_STIFFNESS_N_PER_M:
        raise ValueError("Genesis completed controller telemetry stiffness is inconsistent")
    if telemetry.get("max_net_spring_force_n") != DIAGNOSTIC_FORCE_LIMITED_MAX_NET_SPRING_FORCE_N:
        raise ValueError("Genesis completed controller telemetry force cap is inconsistent")
    current = telemetry.get("current")
    summary = telemetry.get("summary")
    if not isinstance(current, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("Genesis completed controller telemetry is incomplete")
    required_numeric = (
        current.get("raw_net_spring_force_magnitude_n"),
        current.get("applied_net_spring_force_magnitude_n"),
        summary.get("peak_raw_net_spring_force_magnitude_n"),
        summary.get("peak_applied_net_spring_force_magnitude_n"),
        telemetry.get("nominal_per_vertex_stiffness_n_per_m"),
    )
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in required_numeric):
        raise ValueError("Genesis completed controller telemetry has non-finite force or stiffness values")
    if summary.get("numerical_valid") is not True or int(summary.get("active_pre_step_count", 0)) != int(schedule["load_steps"]):
        raise ValueError("Genesis completed controller telemetry does not prove the full frozen load window")
    applied_peak = float(summary["peak_applied_net_spring_force_magnitude_n"])
    tolerance_n = max(1.0e-6, 0.01 * DIAGNOSTIC_FORCE_LIMITED_MAX_NET_SPRING_FORCE_N)
    if applied_peak > DIAGNOSTIC_FORCE_LIMITED_MAX_NET_SPRING_FORCE_N + tolerance_n:
        raise ValueError("Genesis completed controller telemetry exceeded the diagnostics force cap")
    completed_schedule = getattr(completed_measurement, "schedule", None)
    completed_policy = getattr(completed_measurement, "controller_policy", None)
    completed_telemetry = getattr(completed_measurement, "controller_telemetry", None)
    completed_summary = getattr(completed_measurement, "force_summary", None)
    under_load = getattr(completed_measurement, "under_load", None)
    post_release = getattr(completed_measurement, "post_release", None)
    if not all(isinstance(value, Mapping) for value in (completed_schedule, completed_policy, completed_telemetry, completed_summary)):
        raise ValueError("Genesis completed force-limited probe omitted its scheduled controller evidence")
    if completed_schedule.get("recovery_steps") != schedule.get("recovery_steps"):
        raise ValueError("Genesis completed probe recovery schedule differs from the frozen diagnostics schedule")
    if completed_policy != policy or completed_telemetry != telemetry or completed_summary != summary:
        raise ValueError("Genesis completed probe controller evidence differs from release telemetry")
    if (
        under_load is None
        or post_release is None
        or int(completed_schedule.get("load_end_step", -1)) != int(getattr(under_load, "simulation_step", -2))
        or int(completed_schedule.get("release_step", -1)) != int(getattr(under_load, "simulation_step", -2))
        or int(getattr(post_release, "simulation_step", -1))
        != int(completed_schedule.get("release_step", -2)) + int(completed_schedule.get("recovery_steps", -3))
    ):
        raise ValueError("Genesis completed probe endpoints differ from the frozen schedule")
    return {
        "schema_version": DIAGNOSTIC_FORCE_LIMITED_TELEMETRY_SCHEMA_VERSION,
        "policy": policy,
        "selected_vertex_count": int(telemetry["selected_vertex_count"]),
        "nominal_per_vertex_stiffness_n_per_m": float(telemetry["nominal_per_vertex_stiffness_n_per_m"]),
        "peak_raw_net_spring_force_magnitude_n": float(summary["peak_raw_net_spring_force_magnitude_n"]),
        "peak_applied_net_spring_force_magnitude_n": applied_peak,
        "cap_active_pre_step_count": int(summary.get("cap_active_pre_step_count", 0)),
        "active_pre_step_count": int(summary["active_pre_step_count"]),
        "requested_displacement_m": list(telemetry.get("requested_displacement_m", [])),
        "numerical_valid": True,
        "force_scope": dict(telemetry.get("force_scope", {})),
        "schedule": dict(schedule),
        "completed_schedule": dict(completed_schedule),
    }


__all__ = [
    "DIAGNOSTIC_FORCE_LIMITED_CAPABILITY",
    "DIAGNOSTIC_FORCE_LIMITED_CALIBRATION_PROVENANCE",
    "DIAGNOSTIC_FORCE_LIMITED_MAX_LOAD_STEPS",
    "DIAGNOSTIC_FORCE_LIMITED_POLICY_ID",
    "DIAGNOSTIC_FORCE_LIMITED_RECOVERY_STEPS",
    "calibration_provenance",
    "completed_controller_telemetry",
    "controller_state_from_action",
    "force_limited_probe_schedule",
    "policy_hash",
    "policy_payload",
    "validate_controller_policy",
]
