from types import SimpleNamespace

import numpy as np
import pytest

from genesis.engine.couplers import IPCCoupler
from genesis.engine.materials.FEM.cloth import Cloth as ClothMaterial
from genesis.live import actions
from genesis.live.protocol import GenesisLiveError


class _FakeState:
    selected_vertex_count = 2
    selected_vertices = np.array([1, 3], dtype=np.int64)

    def to_dict(self):
        return {
            "controller_id": "diag_box",
            "selected_vertices": self.selected_vertices.tolist(),
            "selected_vertex_count": self.selected_vertex_count,
        }


class _CapturedController:
    calls = []

    def __init__(self, entity, *, controller_id="box_ee_0"):
        self.entity = entity
        self.controller_id = controller_id

    def grasp_and_move_positive_y(self, aabb_box, **kwargs):
        self.__class__.calls.append(
            {
                "entity": self.entity,
                "controller_id": self.controller_id,
                "aabb_box": aabb_box,
                "kwargs": dict(kwargs),
            }
        )
        return _FakeState()


class _FakeSession:
    def __init__(self, entity):
        self.actions = actions.ActionRegistry()
        self.controllers = {}
        self.entities = {"body": entity}

    def default_entity(self):
        return "body", self.entities["body"]

    def entity_by_name(self, name: str):
        return name, self.entities[name]


def _valid_box_action():
    return {
        "action": "box_ee_grasp_and_move",
        "entity": "body",
        "duration_steps": 12,
        "controllers": [
            {
                "controller_id": "diag_box",
                "aabb_box": {"frame": "env_local", "box": [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]},
                "distance_scale": 0.5,
            }
        ],
    }


def _native_entity():
    return SimpleNamespace(sim=SimpleNamespace(coupler=object()), material=object())


def _ipc_surface_entity():
    return SimpleNamespace(
        sim=SimpleNamespace(coupler=IPCCoupler.__new__(IPCCoupler)),
        material=ClothMaterial(),
    )


@pytest.fixture(autouse=True)
def _capture_controller(monkeypatch):
    _CapturedController.calls.clear()
    monkeypatch.setattr(actions, "BoxEndEffectorController", _CapturedController)


def test_live_box_ee_native_uses_backend_owned_compliant_stiffness():
    session = _FakeSession(_native_entity())

    result = actions.apply_probe_action(session, _valid_box_action())

    assert actions.BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS == pytest.approx(0.1)
    assert result["selected_vertex_count"] == 2
    assert session.controllers["diag_box"].controller_id == "diag_box"
    call = _CapturedController.calls[-1]
    assert call["kwargs"]["is_soft_constraint"] is True
    assert call["kwargs"]["stiffness"] == pytest.approx(actions.BOX_EE_NATIVE_FEM_COMPLIANT_STIFFNESS)
    assert "stiffness" not in result["controller_state"]
    assert "is_soft_constraint" not in result["controller_state"]


def test_live_box_ee_ipc_surface_uses_backend_owned_strength_rate_through_stiffness_api():
    session = _FakeSession(_ipc_surface_entity())

    actions.apply_probe_action(session, _valid_box_action())

    call = _CapturedController.calls[-1]
    assert call["kwargs"]["is_soft_constraint"] is True
    assert call["kwargs"]["stiffness"] == pytest.approx(actions.BOX_EE_IPC_SURFACE_COMPLIANT_STRENGTH_RATE)


@pytest.mark.parametrize("key", sorted(actions.BOX_EE_COMPLIANCE_OVERRIDE_KEYS))
def test_live_box_ee_rejects_top_level_compliance_override_keys(key):
    session = _FakeSession(_native_entity())
    params = _valid_box_action()
    params[key] = 1.0

    with pytest.raises(GenesisLiveError) as exc_info:
        actions.apply_probe_action(session, params)

    assert exc_info.value.code == "invalid_controller_request"
    assert f"params.{key}" in exc_info.value.message
    assert not _CapturedController.calls


@pytest.mark.parametrize("key", sorted(actions.BOX_EE_COMPLIANCE_OVERRIDE_KEYS))
def test_live_box_ee_rejects_controller_compliance_override_keys(key):
    session = _FakeSession(_native_entity())
    params = _valid_box_action()
    params["controllers"][0][key] = 1.0

    with pytest.raises(GenesisLiveError) as exc_info:
        actions.apply_probe_action(session, params)

    assert exc_info.value.code == "invalid_controller_request"
    assert f"controllers[0].{key}" in exc_info.value.message
    assert not _CapturedController.calls


def test_live_box_ee_registered_apply_rejects_apply_time_compliance_override():
    session = _FakeSession(_native_entity())
    registered = session.actions.register(_valid_box_action())

    with pytest.raises(GenesisLiveError) as exc_info:
        actions.apply_probe_action(session, {"action_id": registered["action_id"], "strength": 1.0})

    assert exc_info.value.code == "invalid_controller_request"
    assert "params.strength" in exc_info.value.message
    assert not _CapturedController.calls


def test_live_box_ee_registered_apply_rejects_stored_controller_compliance_override():
    session = _FakeSession(_native_entity())
    params = _valid_box_action()
    params["controllers"][0]["stiffness"] = 1.0
    registered = session.actions.register(params)

    with pytest.raises(GenesisLiveError) as exc_info:
        actions.apply_probe_action(session, {"action_id": registered["action_id"]})

    assert exc_info.value.code == "invalid_controller_request"
    assert "controllers[0].stiffness" in exc_info.value.message
    assert not _CapturedController.calls
