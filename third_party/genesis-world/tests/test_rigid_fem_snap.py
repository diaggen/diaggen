"""Small coupled snap mechanics check, runnable with --noconftest against installed Genesis."""
from __future__ import annotations

import igl
import json
from dataclasses import asdict
import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.utils.geom import quat_to_R


@pytest.fixture
def snap_runtime():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    yield
    gs.destroy()


def _host(value):
    return value.detach().cpu().numpy()


@qd.kernel
def _snap_energy(handler: qd.template(), velocity: qd.types.vector(3, qd.f64), full: qd.i32) -> qd.f64:
    if full:
        handler.compute_contact_energy_gamma_G(handler.contact_pairs.sap_info, 0, velocity)
    else:
        handler.compute_contact_energy(handler.contact_pairs.sap_info, 0, velocity)
    return handler.contact_pairs[0].sap_info.energy


def test_native_support_activates_coarse_before_any_snap(snap_runtime, tmp_path):
    outer = np.asarray([[-.025, -.020, -.002], [.025, -.020, -.002],
                        [0., .025, -.002], [0., 0., .038]])
    vertices = np.vstack((outer, outer.mean(axis=0)))
    tets = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]])
    path = tmp_path / "native_support.mesh"
    igl.writeMESH(str(path), vertices, tets, np.empty((0, 3), dtype=np.int64))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, substeps=1, gravity=(0., 0., 0.)),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_floor=False),
        coupler_options=gs.options.SAPCouplerOptions(
            fem_floor_contact_type="tet", rigid_floor_contact_type="none", rigid_rigid_contact_type="none",
            enable_fem_self_tet_contact=False, enable_rigid_fem_contact=True,
            max_rigid_fem_snap_constraints=4, enable_rigid_fem_snap_coarse_preconditioner=True),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Box(pos=(.15, 0., .10), size=(.02, .02, .02)),
                     material=gs.materials.Rigid(enable_coup_collision=False), name="free_body")
    scene.add_entity(morph=gs.morphs.Box(pos=(0., 0., -.05), size=(.2, .2, .1), fixed=True),
                     material=gs.materials.Rigid(enable_coup_collision=True), name="support_table")
    scene.add_entity(morph=gs.morphs.TetMesh(file=str(path)),
                     material=gs.materials.FEM.Elastic(E=2e5, nu=.35, rho=950., model="linear_corotated"))
    scene.build(n_envs=1)
    scene.step(update_visualizer=False, refresh_visualizer=False)
    coupler = scene.sim.coupler
    assert coupler.rigid_fem_contact.n_contact_pairs[None] > 0
    assert coupler.fem_floor_tet_contact.n_contact_pairs[None] > 0
    reading = scene.get_rigid_fem_snap_constraints()
    assert reading["rows"] == []
    assert reading["coarse_active_group_count_by_batch"] == [1]
    assert reading["coarse_active_groups_by_batch"][0][0]["youngs_modulus_pa"] == pytest.approx(2e5)
    assert coupler.rigid_fem_snap_coarse.active.to_numpy()[0, 0]


def test_fem_fem_support_keeps_both_coarse_groups_after_handoff(snap_runtime, tmp_path):
    outer = np.asarray([[-.025, -.020, .10], [.025, -.020, .10], [0., .025, .10], [0., 0., .14]])
    vertices = np.vstack((outer, outer.mean(axis=0)))
    tets = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]])
    path = tmp_path / "fem_fem_handoff.mesh"
    igl.writeMESH(str(path), vertices, tets, np.empty((0, 3), dtype=np.int64))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, substeps=1, gravity=(0., 0., 0.)),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_floor=False),
        coupler_options=gs.options.SAPCouplerOptions(
            fem_floor_contact_type="none", rigid_floor_contact_type="none", rigid_rigid_contact_type="none",
            enable_fem_self_tet_contact=True, enable_rigid_fem_contact=True,
            max_rigid_fem_snap_constraints=4, enable_rigid_fem_snap_coarse_preconditioner=True),
        show_viewer=False,
    )
    rigids, fems = [], []
    for name, x, young in (("A", 0., 5e6), ("B", .2, 3e9)):
        rigids.append(scene.add_entity(
            morph=gs.morphs.Box(pos=(x, 0., .09), size=(.04, .04, .04)),
            material=gs.materials.Rigid(enable_coup_collision=True), name=f"rigid_{name}"))
        fems.append(scene.add_entity(
            morph=gs.morphs.TetMesh(file=str(path), pos=(x, 0., 0.)),
            material=gs.materials.FEM.Elastic(E=young, nu=.35, rho=950., model="linear_corotated"),
            name=f"fem_{name}"))
    scene.build(n_envs=2)
    coupler = scene.sim.coupler
    coarse = coupler.rigid_fem_snap_coarse
    for fem, rigid in zip(fems, rigids):
        scene.add_rigid_fem_snap_constraints(fem, [0], [rigid.links[0]], [10000.], env_idx=0)
    # Inject accepted rows to isolate coarse activation from contact detection and the force law.
    native = coupler.rigid_fem_contact
    native.contact_pairs.batch_idx[0] = 0
    native.contact_pairs.geom_idx0[0] = fems[1].el_start
    native.n_contact_pairs[None] = 1
    self_tet = coupler.fem_self_tet_contact
    self_tet.contact_pairs.batch_idx[0] = 0
    self_tet.contact_pairs.geom_idx0[0] = fems[0].el_start
    self_tet.contact_pairs.geom_idx1[0] = fems[1].el_start
    self_tet.n_contact_pairs[None] = 1
    coarse.refresh(scene.sim.cur_substep_local)
    np.testing.assert_array_equal(coarse.active.to_numpy(), [[True, True], [False, False]])

    scene.clear_rigid_fem_snap_constraints(fems[1])
    native.n_contact_pairs[None] = 0
    coarse.refresh(scene.sim.cur_substep_local)
    assert [row["fem_entity_idx"] for row in scene.get_rigid_fem_snap_constraints()["rows"]] == [fems[0].idx]
    np.testing.assert_array_equal(coarse.active.to_numpy(), [[True, True], [False, False]])
    norms = np.linalg.norm(coarse.basis.to_numpy()[0], axis=(1, 3))
    np.testing.assert_allclose(norms, 1., atol=1e-12)

    # With all snaps gone, both sides must still activate regardless of pair ordering.
    scene.clear_rigid_fem_snap_constraints()
    self_tet.contact_pairs.geom_idx0[0] = fems[1].el_start
    self_tet.contact_pairs.geom_idx1[0] = fems[0].el_start
    coarse.refresh(scene.sim.cur_substep_local)
    np.testing.assert_array_equal(coarse.active.to_numpy(), [[True, True], [False, False]])
    self_tet.n_contact_pairs[None] = 0
    coarse.refresh(scene.sim.cur_substep_local)
    assert not coarse.active.to_numpy().any()
    assert not coarse.basis.to_numpy().any()


def test_two_way_snap_momentum_torque_quadratic_and_clear(snap_runtime, tmp_path):
    outer = np.asarray([[-.025, -.020, .10], [.025, -.020, .10], [0., .025, .10], [0., 0., .14]])
    vertices = np.vstack((outer, outer.mean(axis=0)))
    tets = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]])
    path = tmp_path / "snap_body.mesh"
    igl.writeMESH(str(path), vertices, tets, np.empty((0, 3), dtype=np.int64))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, substeps=1, gravity=(0., 0., 0.)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_floor=False,
                                         damping_alpha=0., damping_beta=0., n_pcg_iterations=80),
        coupler_options=gs.options.SAPCouplerOptions(
            fem_floor_contact_type="none", rigid_floor_contact_type="none", rigid_rigid_contact_type="none",
            enable_fem_self_tet_contact=False, enable_rigid_fem_contact=False,
            max_rigid_fem_snap_constraints=4, n_sap_iterations=5, n_pcg_iterations=100,
            enable_rigid_fem_snap_coarse_preconditioner=True,
            # Test accuracy: default rTr <= 1e-6 permits ~1e-3 Ns residual, larger than this fixture impulse.
            pcg_threshold=1e-16, sap_convergence_atol=1e-14, sap_convergence_rtol=1e-10,
            enable_qualification_post_final_sap_health=True,
        ),
        show_viewer=False,
    )
    rigid = scene.add_entity(morph=gs.morphs.Box(pos=(0., 0., .09), size=(.04, .04, .04),
                                                       euler=(15., 0., 30.), offset_pos=(.005, 0., 0.)),
                             material=gs.materials.Rigid(rho=1000., enable_coup_collision=False), name="moving_link")
    fem = scene.add_entity(morph=gs.morphs.TetMesh(file=str(path)),
                           material=gs.materials.FEM.Elastic(E=2e5, nu=.35, rho=950., model="linear_corotated"),
                           name="soft_body")
    scene.build(n_envs=1)
    link = rigid.links[0]
    assert gs.options.SAPCouplerOptions().max_rigid_fem_snap_constraints == 0
    assert not gs.options.SAPCouplerOptions().enable_rigid_fem_snap_coarse_preconditioner
    before = _host(fem.get_state(track_grad=False).pos).copy()
    row_ids = scene.add_rigid_fem_snap_constraints(fem, [0], [link], [10000.], damping_time_s=.01)
    assert np.array_equal(row_ids, [0])
    assert np.array_equal(before, _host(fem.get_state(track_grad=False).pos))
    binding = scene.get_rigid_fem_snap_constraints()["rows"][0]
    np.testing.assert_allclose(quat_to_R(_host(link.get_quat(relative=False)).reshape(4)) @ np.asarray(binding["local_anchor_m"]) + _host(link.get_pos(relative=False)).reshape(3), before[0, 0], atol=1e-15)
    rigid.set_dofs_velocity(np.asarray([[.10, 0., 0., 0., 0., 0.]]))
    mass = float(rigid.get_mass())
    # Base-link origin and COM differ under morph offsets. Momentum uses COM velocity.
    initial_linear = _host(rigid.get_links_vel(ref="link_com")).reshape(3).copy()
    initial_com = _host(scene.sim.rigid_solver.get_links_pos([link.idx], ref="link_com")).reshape(3).copy()
    initial_position = _host(rigid.get_pos()).copy()
    initial_quat = _host(rigid.get_quat()).copy()
    scene.step(update_visualizer=False, refresh_visualizer=False)
    assert scene.sim.coupler.has_contact  # Snap alone activates SAP.
    reading = scene.get_rigid_fem_snap_constraints()
    row = reading["rows"][0]
    gamma = np.asarray(row["impulse_world_ns"])
    assert np.linalg.norm(gamma) > 1e-8
    assert not np.array_equal(initial_position, _host(rigid.get_pos()))
    assert not np.array_equal(initial_quat, _host(rigid.get_quat()))
    velocity = _host(fem.get_state(track_grad=False).vel)[0]
    volume = np.abs(np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])) / 6
    vertex_mass = np.zeros(len(vertices))
    np.add.at(vertex_mass, tets.ravel(), np.repeat(950. * volume / 4., 4))
    fem_momentum = (vertex_mass[:, None] * velocity).sum(axis=0)
    rigid_delta = mass * (_host(rigid.get_links_vel(ref="link_com")).reshape(3) - initial_linear)
    coupler = scene.sim.coupler
    handler = coupler.rigid_fem_snap
    actual_relative = velocity[0] - handler.Jt.to_numpy()[0].T @ _host(rigid.get_dofs_velocity()).reshape(-1)
    actual_gamma = handler.contact_pairs.weight.to_numpy()[0] * (
        handler.contact_pairs.vhat.to_numpy()[0] - actual_relative)
    np.testing.assert_allclose(gamma, actual_gamma, rtol=1e-12, atol=1e-14)
    diagnostics = {"mass_kg": mass, "initial_com_m": initial_com.tolist(),
                   "accepted_state_gamma_ns": actual_gamma.tolist(),
                   "final_fem_gradient": coupler.fem_state_v.gradient.to_numpy().tolist(),
                   "final_rigid_gradient": coupler.rigid_state_dof.gradient.to_numpy().tolist(),
                   "solver_health": asdict(scene.get_last_completed_solver_health().substeps[0]),
                   "link_origin_m": row["substep_link_origin_m"], "anchor_m": row["substep_anchor_world_m"],
                   "gamma_ns": gamma.tolist(), "fem_momentum_ns": fem_momentum.tolist(),
                   "rigid_delta_ns": rigid_delta.tolist(),
                   "initial_com_velocity_mps": initial_linear.tolist(),
                   "final_com_velocity_mps": _host(rigid.get_links_vel(ref="link_com")).reshape(3).tolist(),
                   "final_link_velocity_mps": _host(rigid.get_vel()).reshape(3).tolist(),
                   "final_angular_velocity_radps": _host(rigid.get_ang()).reshape(3).tolist()}
    (tmp_path / "mechanics_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    np.testing.assert_allclose(fem_momentum + rigid_delta, 0., atol=2e-7)
    np.testing.assert_allclose(fem_momentum, gamma, rtol=2e-4, atol=2e-7)
    np.testing.assert_allclose(rigid_delta, -gamma, rtol=2e-4, atol=2e-7)
    arm = np.asarray(row["substep_anchor_world_m"]) - initial_com
    expected_angular_impulse = np.cross(arm, -gamma)
    actual_angular_momentum = mass * .04**2 / 6. * _host(rigid.get_ang()).reshape(3)
    assert np.linalg.norm(actual_angular_momentum) > 1e-9
    np.testing.assert_allclose(actual_angular_momentum, expected_angular_impulse, rtol=2e-4, atol=2e-8)
    assert reading["completed_physical_substep_index"] == 0
    assert reading["substep_dt_s"] == .001

    # Exercise actual row energy/gradient/Hessian code at a finite displaced spring state.
    handler = scene.sim.coupler.rigid_fem_snap
    gap = np.array([.002, -.001, .003])
    u = np.array([.04, -.02, .01])
    handler.contact_pairs.gap[0] = gap
    scene.sim.coupler.compute_regularization(
        dofs_state=scene.sim.rigid_solver.dofs_state, entities_info=scene.sim.rigid_solver.entities_info,
        rigid_global_info=scene.sim.rigid_solver._rigid_global_info,
    )
    energy = _snap_energy(handler, u, 1)
    assert energy == pytest.approx(_snap_energy(handler, u, 0))
    actual_gamma = handler.contact_pairs.sap_info.gamma.to_numpy()[0]
    G = handler.contact_pairs.sap_info.G.to_numpy()[0]
    expected = -.001 * 10000. * (gap + (.001 + .01) * u)
    np.testing.assert_allclose(actual_gamma, expected, rtol=1e-12, atol=1e-12)
    epsilon = 1e-6
    gradient = np.array([(_snap_energy(handler, u + epsilon * axis, 0) -
                          _snap_energy(handler, u - epsilon * axis, 0)) / (2 * epsilon)
                         for axis in np.eye(3)])
    np.testing.assert_allclose(gradient, -actual_gamma, rtol=1e-8, atol=1e-10)
    jacobian = np.column_stack((np.eye(3), -handler.Jt.to_numpy()[0].T))
    direction = np.linspace(-.2, .3, jacobian.shape[1])
    du = jacobian @ direction
    _snap_energy(handler, u + epsilon * du, 1)
    plus = -jacobian.T @ handler.contact_pairs.sap_info.gamma.to_numpy()[0]
    _snap_energy(handler, u - epsilon * du, 1)
    minus = -jacobian.T @ handler.contact_pairs.sap_info.gamma.to_numpy()[0]
    np.testing.assert_allclose((plus - minus) / (2 * epsilon), jacobian.T @ G @ jacobian @ direction,
                               rtol=1e-8, atol=1e-10)

    # Actual coarse operator: full-H SPD and the independently known snap contribution.
    coarse = coupler.rigid_fem_snap_coarse
    saved_active = coupler.batch_active.to_numpy().copy()
    coupler.batch_active.fill(True)
    coarse.prepare()
    matrix_with_snap = coarse.matrix.to_numpy()[0, 0].copy()
    np.testing.assert_allclose(matrix_with_snap, matrix_with_snap.T, rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(matrix_with_snap).min() > 0.
    basis_at_vertex = coarse.basis.to_numpy()[0, 0, 0]
    handler.contact_pairs.sap_info.G[0] = np.zeros((3, 3))
    coarse.prepare()
    without_snap = coarse.matrix.to_numpy()[0, 0]
    np.testing.assert_allclose(matrix_with_snap - without_snap,
                               basis_at_vertex @ G @ basis_at_vertex.T, rtol=1e-9, atol=1e-12)
    handler.contact_pairs.sap_info.G[0] = G
    coarse.prepare()
    coupler.batch_active.from_numpy(saved_active)

    scene.clear_rigid_fem_snap_constraints()
    assert scene.get_rigid_fem_snap_constraints()["rows"] == []
    assert not coarse.active.to_numpy().any()
    free_velocity = _host(rigid.get_links_vel(ref="link_com")).copy()
    scene.step(update_visualizer=False, refresh_visualizer=False)
    assert not scene.sim.coupler.has_contact
    np.testing.assert_allclose(_host(rigid.get_links_vel(ref="link_com")), free_velocity, atol=1e-12)
    scene.add_rigid_fem_snap_constraints(fem, [0], [link], [10000.])
    scene.reset()
    assert scene.get_rigid_fem_snap_constraints()["rows"] == []


def test_selective_clear_preserves_other_fem_capture_and_response(snap_runtime, tmp_path):
    outer = np.asarray([[-.025, -.020, .10], [.025, -.020, .10], [0., .025, .10], [0., 0., .14]])
    vertices = np.vstack((outer, outer.mean(axis=0)))
    tets = np.asarray([[0, 1, 2, 4], [0, 1, 4, 3], [0, 4, 2, 3], [4, 1, 2, 3]])
    path = tmp_path / "two_fem.mesh"
    igl.writeMESH(str(path), vertices, tets, np.empty((0, 3), dtype=np.int64))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.001, substeps=1, gravity=(0., 0., 0.)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_floor=False),
        coupler_options=gs.options.SAPCouplerOptions(
            fem_floor_contact_type="none", rigid_floor_contact_type="none", rigid_rigid_contact_type="none",
            enable_fem_self_tet_contact=False, enable_rigid_fem_contact=False,
            max_rigid_fem_snap_constraints=4, enable_rigid_fem_snap_coarse_preconditioner=True,
            pcg_threshold=1e-16, sap_convergence_atol=1e-14, sap_convergence_rtol=1e-10),
        show_viewer=False,
    )
    rigids, fems = [], []
    for name, x in (("A", 0.), ("B", .2)):
        rigids.append(scene.add_entity(
            morph=gs.morphs.Box(pos=(x, 0., .09), size=(.04, .04, .04)),
            material=gs.materials.Rigid(enable_coup_collision=False), name=f"rigid_{name}"))
        fems.append(scene.add_entity(
            morph=gs.morphs.TetMesh(file=str(path), pos=(x, 0., 0.)),
            material=gs.materials.FEM.Elastic(E=2e5, nu=.35, rho=950., model="linear_corotated"),
            name=f"fem_{name}"))
    scene.build(n_envs=1)
    # B first makes A move to a different active row when B is removed.
    for index in (1, 0):
        scene.add_rigid_fem_snap_constraints(fems[index], [0], [rigids[index].links[0]], [10000.])
    fields = ("fem_entity_idx", "vertex_idx_local", "vertex_idx_global", "rigid_link_idx",
              "local_anchor_m", "capture_vertex_position_m", "stiffness_npm", "damping_time_s")
    before = scene.get_rigid_fem_snap_constraints()["rows"][1]
    scene.clear_rigid_fem_snap_constraints(fems[1])
    reading = scene.get_rigid_fem_snap_constraints()
    assert len(reading["rows"]) == 1
    assert reading["completed_physical_substep_index"] is None
    after = reading["rows"][0]
    assert after["row_id"] == 0
    assert {key: before[key] for key in fields} == {key: after[key] for key in fields}
    rigids[0].set_pos(np.asarray([[.002, 0., .09]]))
    scene.step(update_visualizer=False, refresh_visualizer=False)
    impulse = scene.get_rigid_fem_snap_constraints()["rows"][0]["impulse_world_ns"]
    assert np.linalg.norm(impulse) > 1e-8
    assert np.linalg.norm(_host(fems[0].get_state(track_grad=False).vel)) > 1e-8
    assert np.linalg.norm(_host(rigids[0].get_vel())) > 1e-8
    scene.clear_rigid_fem_snap_constraints()
    assert scene.get_rigid_fem_snap_constraints()["rows"] == []
