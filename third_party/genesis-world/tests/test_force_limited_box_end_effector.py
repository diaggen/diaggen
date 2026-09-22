from types import SimpleNamespace

import numpy as np
import pytest

from genesis.engine.controllers import box_end_effector as box_ee
from genesis.engine.controllers.box_end_effector import BoxEndEffectorController, BoxEndEffectorState
from genesis.live import actions
from genesis.live.protocol import VOLUMETRIC_CAPABILITIES


POLICY = {
    "mode": "native_fem_force_limited_box_ee/v1",
    "policy_id": "hag4r-paip-force-limited-v1",
    "total_stiffness_n_per_m": 200.0,
    "max_net_spring_force_n": 5.0,
}


class _NativeEntity:
    def __init__(self):
        self.scene = SimpleNamespace(n_envs=0, dt=0.001)
        self.sim = SimpleNamespace(coupler=object())
        self.calls = []

    def set_vertex_constraints(self, vertices, **kwargs):
        self.calls.append((np.asarray(vertices).copy(), kwargs))


def test_force_limited_policy_is_canonical_and_advertised():
    session = SimpleNamespace(_scene_config={"box_ee_controller_policy": POLICY})
    policy = actions._force_limited_policy(session, _NativeEntity())

    assert "native_fem_force_limited_box_ee" in VOLUMETRIC_CAPABILITIES
    assert policy == {
        **POLICY,
        "policy_hash": "531cc3012bf34ded8907a262af9de8ea90ad446d6cf245c102e4bb6669e2715b",
        "force_scope": {
            "quantity": "net_spring_force",
            "formula": "||sum_i f_i||",
            "excludes": ["sum_of_magnitudes", "stress", "torque", "contact_force", "internal_force"],
        },
    }


def test_force_limiter_normalizes_total_stiffness_and_caps_net_vector(monkeypatch):
    entity = _NativeEntity()
    policy = actions._force_limited_policy(
        SimpleNamespace(_scene_config={"box_ee_controller_policy": POLICY}), entity
    )
    controller = BoxEndEffectorController(entity, force_limited_policy=policy)
    controller._state = BoxEndEffectorState(
        controller_id="box_ee_0",
        frame="env_local",
        source_box=np.zeros(6),
        env_local_box=np.zeros(6),
        selected_vertices=np.array([0, 1]),
        target_positions=np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        active=True,
        controller_policy=policy,
    )
    monkeypatch.setattr(
        box_ee.su,
        "fem_entity_positions",
        lambda *_args, **_kwargs: np.zeros((2, 3), dtype=np.float64),
    )

    state = controller.prepare_step()

    assert entity.calls[-1][1]["stiffness"] == pytest.approx(25.0)
    telemetry = state.to_dict()["controller_telemetry"]
    assert telemetry["nominal_per_vertex_stiffness_n_per_m"] == pytest.approx(100.0)
    assert telemetry["current"]["raw_net_spring_force_magnitude_n"] == pytest.approx(20.0)
    assert telemetry["current"]["applied_net_spring_force_magnitude_n"] == pytest.approx(5.0)
    assert telemetry["current"]["stiffness_scale"] == pytest.approx(0.25)
    assert telemetry["summary"] == {
        "peak_raw_net_spring_force_magnitude_n": pytest.approx(20.0),
        "peak_applied_net_spring_force_magnitude_n": pytest.approx(5.0),
        "cap_active_pre_step_count": 1,
        "active_pre_step_count": 1,
        "numerical_valid": True,
    }


def test_force_limiter_uses_net_vector_not_sum_of_magnitudes(monkeypatch):
    entity = _NativeEntity()
    policy = actions._force_limited_policy(
        SimpleNamespace(_scene_config={"box_ee_controller_policy": POLICY}), entity
    )
    controller = BoxEndEffectorController(entity, force_limited_policy=policy)
    controller._state = BoxEndEffectorState(
        controller_id="box_ee_0",
        frame="env_local",
        source_box=np.zeros(6),
        env_local_box=np.zeros(6),
        selected_vertices=np.array([0, 1]),
        target_positions=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        active=True,
        controller_policy=policy,
    )
    monkeypatch.setattr(
        box_ee.su,
        "fem_entity_positions",
        lambda *_args, **_kwargs: np.zeros((2, 3), dtype=np.float64),
    )

    telemetry = controller.prepare_step().to_dict()["controller_telemetry"]

    assert entity.calls[-1][1]["stiffness"] == pytest.approx(100.0)
    assert telemetry["current"]["raw_net_spring_force_magnitude_n"] == pytest.approx(0.0)
    assert telemetry["current"]["cap_active"] is False
