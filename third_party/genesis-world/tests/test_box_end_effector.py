import igl
import numpy as np
import pytest

import genesis as gs
from genesis.engine.controllers.box_end_effector import (
    BoxEndEffectorController,
    apply_static_box_anchors,
    motion_axis_vector,
)
from genesis.utils.spatial_selection import aabb_to_env_local, select_vertices_in_aabb
from genesis.utils.misc import tensor_to_array


_TWO_TET_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_TWO_TETS = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)


def _write_tet_mesh(path):
    igl.writeMESH(str(path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))


def _build_two_tet_scene(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    _write_tet_mesh(mesh_path)

    scene = gs.Scene(
        show_viewer=False,
        fem_options=gs.options.FEMOptions(enable_vertex_constraints=True),
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(),
    )
    scene.build()
    return scene, entity


def _build_native_implicit_two_tet_scene(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    _write_tet_mesh(mesh_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.05,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=True,
            n_newton_iterations=2,
            n_pcg_iterations=100,
            damping_alpha=0.0,
            damping_beta=0.0,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(E=100.0, rho=100.0, model="linear_corotated"),
    )
    scene.build()
    return scene, entity


def _constraint_flags(scene, entity):
    flags = scene.fem_solver.vertex_constraints.is_constrained.to_numpy()
    return flags[entity.v_start : entity.v_start + entity.n_vertices, 0]


def _constraint_soft_flags(scene, entity):
    flags = scene.fem_solver.vertex_constraints.is_soft_constraint.to_numpy()
    return flags[entity.v_start : entity.v_start + entity.n_vertices, 0]


def _constraint_stiffness(scene, entity):
    stiffness = scene.fem_solver.vertex_constraints.stiffness.to_numpy()
    return stiffness[entity.v_start : entity.v_start + entity.n_vertices, 0]


def _constraint_target(scene, entity, vertex_idx):
    targets = scene.fem_solver.vertex_constraints.target_pos.to_numpy()
    return targets[entity.v_start + vertex_idx, 0]


def test_aabb_selection_supports_explicit_frames():
    positions = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(select_vertices_in_aabb(positions, [1, 2, 3, 1, 2, 3]), np.array([0]))
    np.testing.assert_array_equal(
        select_vertices_in_aabb(positions, [0, 0, 0, 0, 0, 0], frame="object_local", object_to_env=[1, 2, 3]),
        np.array([0]),
    )
    np.testing.assert_array_equal(
        select_vertices_in_aabb(positions, [2, 3, 4, 2, 3, 4], frame="world", world_to_env=[-1, -1, -1]),
        np.array([0]),
    )

    object_to_env = np.eye(4, dtype=np.float32)
    object_to_env[:3, 3] = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        aabb_to_env_local([0, 0, 0, 0, 0, 0], frame="object_local", object_to_env=object_to_env),
        np.array([1, 2, 3, 1, 2, 3], dtype=np.float32),
    )


def test_aabb_selection_rejects_bad_box_and_frame():
    positions = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(gs.GenesisException, match="shape"):
        select_vertices_in_aabb(positions, [0, 0, 0])
    with pytest.raises(gs.GenesisException, match="Unsupported"):
        select_vertices_in_aabb(positions, [0, 0, 0, 1, 1, 1], frame="camera")


def test_static_box_anchor_constrains_expected_vertices(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)

    records = apply_static_box_anchors(
        entity,
        [{"anchor_id": "origin_pin", "frame": "env_local", "box": [-0.05, -0.05, -0.05, 0.05, 0.05, 0.05]}],
    )

    assert records[0].selected_vertex_count == 1
    np.testing.assert_array_equal(records[0].selected_vertices, np.array([0], dtype=gs.np_int))
    flags = _constraint_flags(scene, entity)
    assert flags[0]
    assert not flags[1]


def test_static_box_anchor_defaults_to_hard_zero_stiffness(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)

    apply_static_box_anchors(
        entity,
        [{"anchor_id": "origin_pin", "frame": "env_local", "box": [-0.05, -0.05, -0.05, 0.05, 0.05, 0.05]}],
    )

    assert _constraint_flags(scene, entity)[0]
    assert not _constraint_soft_flags(scene, entity)[0]
    assert _constraint_stiffness(scene, entity)[0] == pytest.approx(0.0)


def test_static_box_anchor_forwards_soft_constraint_settings(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)

    records = apply_static_box_anchors(
        entity,
        [{"anchor_id": "origin_soft", "frame": "env_local", "box": [-0.05, -0.05, -0.05, 0.05, 0.05, 0.05]}],
        is_soft_constraint=True,
        stiffness=250.0,
    )

    assert records[0].selected_vertex_count == 1
    assert _constraint_flags(scene, entity)[0]
    assert _constraint_soft_flags(scene, entity)[0]
    assert _constraint_stiffness(scene, entity)[0] == pytest.approx(250.0)


def test_box_ee_grasp_selects_current_positions(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    current = tensor_to_array(entity.get_state().pos[0])
    moved = current + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    entity.set_position(moved)

    controller = BoxEndEffectorController(entity)
    state = controller.grasp([9.95, -0.05, -0.05, 10.05, 0.05, 0.05])

    np.testing.assert_array_equal(state.selected_vertices, np.array([0], dtype=gs.np_int))
    np.testing.assert_allclose(state.target_positions[0], moved[0], atol=1e-6)


def test_box_ee_move_updates_constraint_targets_deterministically(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    state = controller.move_positive_y(distance_scale=0.5, duration_steps=12, speed=0.6)

    np.testing.assert_array_equal(state.selected_vertices, np.array([3], dtype=gs.np_int))
    np.testing.assert_allclose(state.displacement, np.array([0.0, 0.0, 0.0], dtype=gs.np_float), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), np.array([0.0, 0.0, 1.0]), atol=1e-6)
    assert state.duration_steps == 12
    assert state.speed == pytest.approx(0.6)
    assert state.distance == pytest.approx(0.05)
    assert state.moved_distance == pytest.approx(0.0)
    assert state.estimated_motion_steps == 9
    assert state.motion_active

    state = controller.advance_motion(steps=12)

    np.testing.assert_allclose(state.displacement, np.array([0.0, 0.05, 0.0], dtype=gs.np_float), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), np.array([0.0, 0.05, 1.0]), atol=1e-6)
    assert state.moved_distance == pytest.approx(0.05)
    assert not state.motion_active


@pytest.mark.parametrize(
    ("axis", "expected"),
    (
        ("+X", [1.0, 0.0, 0.0]),
        ("-X", [-1.0, 0.0, 0.0]),
        ("+Y", [0.0, 1.0, 0.0]),
        ("-Y", [0.0, -1.0, 0.0]),
        ("+Z", [0.0, 0.0, 1.0]),
        ("-Z", [0.0, 0.0, -1.0]),
    ),
)
def test_box_ee_motion_axis_is_signed_cardinal(axis, expected):
    np.testing.assert_allclose(motion_axis_vector(axis), np.asarray(expected, dtype=gs.np_float))


def test_box_ee_motion_axis_rejects_raw_vector_or_unknown_axis():
    with pytest.raises(gs.GenesisException, match="motion_axis"):
        motion_axis_vector([1.0, 0.0, 0.0])
    with pytest.raises(gs.GenesisException, match="motion_axis"):
        motion_axis_vector("+Q")


@pytest.mark.parametrize(
    ("axis", "direction"),
    (
        ("+X", [1.0, 0.0, 0.0]),
        ("-X", [-1.0, 0.0, 0.0]),
        ("+Y", [0.0, 1.0, 0.0]),
        ("-Y", [0.0, -1.0, 0.0]),
        ("+Z", [0.0, 0.0, 1.0]),
        ("-Z", [0.0, 0.0, -1.0]),
    ),
)
def test_box_ee_cardinal_motion_moves_only_on_requested_signed_component(tmp_path, axis, direction):
    scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])
    state = controller.move_cardinal_axis(motion_axis=axis, distance_scale=0.5, duration_steps=12, speed=0.6)
    state = controller.advance_motion(steps=12)

    expected = np.asarray([0.0, 0.0, 1.0], dtype=gs.np_float) + np.asarray(direction, dtype=gs.np_float) * 0.05
    assert state.motion_axis == axis
    np.testing.assert_allclose(state.displacement, np.asarray(direction, dtype=gs.np_float) * 0.05, atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), expected, atol=1e-6)


def test_box_ee_motion_uses_speed_and_duration_steps(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, -0.05, 1.05, 0.05, 0.05])

    state = controller.move_positive_y(distance_scale=0.5, duration_steps=2, speed=0.5)

    expected_distance = 0.55
    expected_moved = 0.5 * scene.dt * 2
    assert state.distance == pytest.approx(expected_distance)
    assert state.moved_distance == pytest.approx(0.0)
    assert state.estimated_motion_steps == 110
    assert state.motion_active
    np.testing.assert_allclose(_constraint_target(scene, entity, 0), np.array([0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 1), np.array([1.0, 0.0, 0.0]), atol=1e-6)

    state = controller.advance_motion(steps=2)

    assert state.moved_distance == pytest.approx(expected_moved)
    assert state.motion_active
    np.testing.assert_allclose(_constraint_target(scene, entity, 0), np.array([0.0, expected_moved, 0.0]), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 1), np.array([1.0, expected_moved, 0.0]), atol=1e-6)

    state = controller.advance_motion(steps=200)

    assert state.moved_distance == pytest.approx(expected_distance)
    assert not state.motion_active
    np.testing.assert_allclose(_constraint_target(scene, entity, 1), np.array([1.0, expected_distance, 0.0]), atol=1e-6)


def test_box_ee_release_preserves_unrelated_static_anchor(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    apply_static_box_anchors(
        entity,
        [{"anchor_id": "bottom_pin", "frame": "env_local", "box": [-0.05, -0.05, -1.05, 0.05, 0.05, -0.95]}],
    )
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    controller.release()

    flags = _constraint_flags(scene, entity)
    assert not flags[3]
    assert flags[4]


def test_box_ee_release_restores_overlapping_static_anchor(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    apply_static_box_anchors(
        entity,
        [{"anchor_id": "top_pin", "frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]}],
    )
    original_target = _constraint_target(scene, entity, 3).copy()
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])
    controller.move_positive_y(distance_scale=0.5, duration_steps=12)

    controller.release()

    flags = _constraint_flags(scene, entity)
    assert flags[3]
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), original_target, atol=1e-6)


def test_box_ee_soft_grasp_requires_positive_stiffness(tmp_path):
    _scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    with pytest.raises(gs.GenesisException, match="finite stiffness > 0.0"):
        controller.grasp(
            [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05],
            is_soft_constraint=True,
        )

    assert not controller.state.active


def test_box_ee_soft_grasp_advance_motion_preserves_soft_stiffness(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    state = controller.grasp(
        [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05],
        is_soft_constraint=True,
        stiffness=500.0,
    )

    np.testing.assert_array_equal(state.selected_vertices, np.array([3], dtype=gs.np_int))
    assert _constraint_flags(scene, entity)[3]
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(500.0)

    controller.move_positive_y(distance_scale=0.5, duration_steps=12, speed=0.6)
    state = controller.advance_motion(steps=12)

    np.testing.assert_allclose(state.displacement, np.array([0.0, 0.05, 0.0], dtype=gs.np_float), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), np.array([0.0, 0.05, 1.0]), atol=1e-6)
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(500.0)

    position_at_update = tensor_to_array(entity.get_state().pos[0, 3]).copy()
    target = _constraint_target(scene, entity, 3).copy()
    for _ in range(4):
        scene.step(update_visualizer=False)

    final = tensor_to_array(entity.get_state().pos[0, 3]).copy()
    assert np.linalg.norm(final - target) < np.linalg.norm(position_at_update - target)
    assert np.dot(final - position_at_update, target - position_at_update) > 0.0


def test_box_ee_grasp_and_move_forwards_soft_constraint_settings(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    state = controller.grasp_and_move_positive_y(
        [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05],
        distance_scale=0.5,
        duration_steps=12,
        speed=0.6,
        is_soft_constraint=True,
        stiffness=750.0,
    )

    np.testing.assert_array_equal(state.selected_vertices, np.array([3], dtype=gs.np_int))
    assert _constraint_flags(scene, entity)[3]
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(750.0)

    state = controller.advance_motion(steps=12)

    np.testing.assert_allclose(state.displacement, np.array([0.0, 0.05, 0.0], dtype=gs.np_float), atol=1e-6)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), np.array([0.0, 0.05, 1.0]), atol=1e-6)
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(750.0)


def test_box_ee_grasp_and_move_soft_constraint_requires_positive_stiffness(tmp_path):
    _scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    with pytest.raises(gs.GenesisException, match="finite stiffness > 0.0"):
        controller.grasp_and_move_positive_y(
            [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05],
            distance_scale=0.5,
            duration_steps=12,
            is_soft_constraint=True,
        )

    assert not controller.state.active


def test_box_ee_release_restores_overlapping_soft_static_anchor(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    apply_static_box_anchors(
        entity,
        [{"anchor_id": "top_soft", "frame": "env_local", "box": [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05]}],
        is_soft_constraint=True,
        stiffness=123.0,
    )
    original_target = _constraint_target(scene, entity, 3).copy()
    controller = BoxEndEffectorController(entity)

    controller.grasp(
        [-0.05, -0.05, 0.95, 0.05, 0.05, 1.05],
        is_soft_constraint=True,
        stiffness=456.0,
    )
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(456.0)
    controller.move_positive_y(distance_scale=0.5, duration_steps=12)
    controller.advance_motion(steps=12)
    assert not np.allclose(_constraint_target(scene, entity, 3), original_target)

    controller.release()

    assert _constraint_flags(scene, entity)[3]
    assert _constraint_soft_flags(scene, entity)[3]
    assert _constraint_stiffness(scene, entity)[3] == pytest.approx(123.0)
    np.testing.assert_allclose(_constraint_target(scene, entity, 3), original_target, atol=1e-6)


def test_box_ee_empty_selection_fails_unless_optional(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)

    with pytest.raises(gs.GenesisException, match="selected no FEM vertices"):
        controller.grasp([10, 10, 10, 11, 11, 11])

    state = controller.grasp([10, 10, 10, 11, 11, 11], optional=True)
    assert state.selected_vertex_count == 0
    assert not state.active


def test_box_ee_rejects_out_of_range_distance_scale(tmp_path):
    _scene, entity = _build_two_tet_scene(tmp_path)
    controller = BoxEndEffectorController(entity)
    controller.grasp([-0.05, -0.05, 0.95, 0.05, 0.05, 1.05])

    with pytest.raises(gs.GenesisException, match="distance_scale"):
        controller.move_positive_y(distance_scale=1.5, duration_steps=1)
    with pytest.raises(gs.GenesisException, match="distance_scale"):
        controller.move_positive_y(distance_scale=0.0, duration_steps=1)
