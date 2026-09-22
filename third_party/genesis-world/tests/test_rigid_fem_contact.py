from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import genesis as gs
from genesis.engine.couplers.sap_coupler import (
    _build_rigid_fem_face_whitelist,
    _rigid_fem_contact_sort_order,
)


def _receipt(enabled=True):
    entry = gs.RigidFEMWhitelistEntry(
        rigid_entity_idx=1,
        rigid_entity_name="robot",
        collision_enabled=enabled,
        requested_link_names=None,
        resolved_link_indices=(2,) if enabled else (),
        resolved_link_names=("finger",) if enabled else (),
        resolved_geom_indices=(3,) if enabled else (),
        enabled_face_count=1 if enabled else 0,
        total_face_count=1,
    )
    return gs.RigidFEMWhitelistReceipt(
        entries=(entry,),
        enabled_face_count=1 if enabled else 0,
        total_face_count=1,
        face_enabled_by_index={0: enabled},
    )


def _batch(**overrides):
    payload = dict(
        env_idx=np.array([0], dtype=np.int64),
        rigid_entity_idx=np.array([1], dtype=np.int64),
        rigid_link_idx=np.array([2], dtype=np.int64),
        rigid_geom_idx=np.array([3], dtype=np.int64),
        fem_entity_idx=np.array([4], dtype=np.int64),
        fem_element_idx_local=np.array([5], dtype=np.int64),
        rigid_entity_names=("robot",),
        rigid_link_names=("finger",),
        fem_entity_names=("soft",),
        point_m=np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        normal_world=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        signed_gap_m=np.array([-0.01], dtype=np.float64),
        penetration_m=np.array([0.01], dtype=np.float64),
        normal_impulse_ns=np.array([0.5], dtype=np.float64),
        tangential_impulse_world_ns=np.array([[0.1, 0.2, 0.0]], dtype=np.float64),
        relative_tangential_velocity_world_mps=np.array([[0.01, 0.02, 0.0]], dtype=np.float64),
        modes=(gs.RigidFEMContactMode.STICK,),
        completed_scene_step=7,
        completed_substep=15,
        dt_s=0.001,
        whitelist_receipt=_receipt(),
    )
    payload.update(overrides)
    return gs.RigidFEMContactBatch(**payload)


def test_public_batch_is_strict_fresh_and_immutable():
    source = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
    batch = _batch(point_m=source)
    assert len(batch) == 1
    assert batch.point_m.dtype == np.float64
    assert batch.env_idx.dtype == np.int64
    assert batch.modes == (gs.RigidFEMContactMode.STICK,)
    assert batch.whitelist_receipt.face_enabled_by_index == {0: True}
    assert not batch.point_m.flags.writeable
    assert not batch.env_idx.flags.writeable
    source[:] = 99.0
    assert np.array_equal(batch.point_m, [[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        batch.point_m[0, 0] = 1.0
    with pytest.raises(TypeError):
        batch.whitelist_receipt.face_enabled_by_index[0] = False


def test_public_build_ownership_receipt_is_available_before_step():
    ownership = gs.RigidFEMContactOwnershipReceipt(
        whitelist_receipt=_receipt(),
        rigid_fem_contact_enabled=True,
        floor_tet_contact_enabled=True,
        floor_height_m=0.0,
    )
    scene = object.__new__(gs.Scene)
    scene._is_built = True
    scene._sim = SimpleNamespace(
        _coupler=SimpleNamespace(get_rigid_fem_contact_ownership=lambda: ownership),
        destroy=lambda: None,
    )
    assert gs.Scene.get_rigid_fem_contact_ownership(scene) is ownership
    assert ownership.floor_tet_contact_enabled
    assert ownership.whitelist_receipt.enabled_face_count == 1


@pytest.mark.parametrize(
    "mode,normal_impulse,tangential_impulse",
    [
        (gs.RigidFEMContactMode.STICK, 0.5, (0.1, 0.2, 0.0)),
        (gs.RigidFEMContactMode.SLIDE, 0.5, (0.1, 0.2, 0.0)),
        (gs.RigidFEMContactMode.NO_CONTACT, 0.0, (0.0, 0.0, 0.0)),
    ],
)
def test_public_contact_modes_have_exact_impulse_semantics(mode, normal_impulse, tangential_impulse):
    batch = _batch(
        modes=(mode,),
        normal_impulse_ns=np.array([normal_impulse], dtype=np.float64),
        tangential_impulse_world_ns=np.array([tangential_impulse], dtype=np.float64),
    )
    assert batch.modes == (mode,)


def _fake_solver(requested=("left",), enabled=True):
    def geom(index, face_start, n_faces):
        return SimpleNamespace(idx=index, face_start=face_start, face_end=face_start + n_faces, n_faces=n_faces)

    left = SimpleNamespace(name="left", idx=2, geoms=(geom(4, 0, 2),))
    right = SimpleNamespace(name="right", idx=3, geoms=(geom(5, 2, 3),))
    material = SimpleNamespace(coup_collision_links=requested, enable_coup_collision=enabled)
    entity = SimpleNamespace(
        idx=1,
        name="robot",
        links=(left, right),
        material=material,
        n_faces=5,
    )
    return SimpleNamespace(n_faces=5, entities=(entity,))


def _write_two_finger_urdf(tmp_path):
    path = tmp_path / "two_finger.urdf"
    path.write_text(
        """<robot name="two_finger">
  <link name="base_link"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <link name="g2_left_link"><inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual></link>
  <link name="g2_right_link"><inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual></link>
  <joint name="g2_left_joint" type="prismatic"><parent link="base_link"/><child link="g2_left_link"/><origin xyz="0.02 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.02" effort="20" velocity="1"/></joint>
  <joint name="g2_right_joint" type="prismatic"><parent link="base_link"/><child link="g2_right_link"/><origin xyz="-0.02 0 0" rpy="0 0 0"/><axis xyz="-1 0 0"/><limit lower="0" upper="0.02" effort="20" velocity="1"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_build_whitelist_resolves_positive_links_without_mutating_geoms():
    solver = _fake_solver()
    mask, receipt = _build_rigid_fem_face_whitelist(solver)
    assert np.array_equal(mask, [True, True, False, False, False])
    assert receipt.entries[0].requested_link_names == ("left",)
    assert receipt.entries[0].resolved_link_indices == (2,)
    assert receipt.entries[0].resolved_geom_indices == (4,)
    assert receipt.enabled_face_count == 2
    assert solver.entities[0].links[1].geoms[0].n_faces == 3


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_disabled_ordinary_collider_preserves_named_sap_finger_whitelist(tmp_path, show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=2, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True, enable_qualification_safety_extrema=True),
        coupler_options=gs.options.SAPCouplerOptions(
            enable_rigid_fem_contact=True,
            fem_floor_contact_type="tet",
            rigid_floor_contact_type="none",
            rigid_rigid_contact_type="none",
        ),
        show_viewer=show_viewer,
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(file=str(_write_two_finger_urdf(tmp_path)), fixed=True, merge_fixed_links=False),
        material=gs.materials.Rigid(
            enable_coup_collision=True,
            coup_collision_links=("g2_left_link", "g2_right_link"),
        ),
        name="two_finger_robot",
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=0.01),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft",
    )
    scene.build()

    rigid_solver = scene.sim.rigid_solver
    assert rigid_solver.collider._ordinary_collision_enabled is False
    assert rigid_solver.collider._sdf is None
    assert rigid_solver.collider._gjk is None
    assert rigid_solver.collider._support_field is None
    assert robot.get_link(name="g2_left_link").n_geoms > 0
    assert robot.get_link(name="g2_right_link").n_geoms > 0
    with pytest.raises(gs.GenesisException, match="enable_collision=False"):
        robot.detect_collision()

    scene.step(update_visualizer=False)
    health = scene.get_last_completed_solver_health()
    assert len(health.substeps) == 2
    assert all(substep.fem_safety_extrema is not None for substep in health.substeps)
    batch = scene.get_rigid_fem_contacts()
    entry = next(entry for entry in batch.whitelist_receipt.entries if entry.rigid_entity_name == "two_finger_robot")
    assert entry.resolved_link_names == ("g2_left_link", "g2_right_link")
    assert entry.enabled_face_count > 0


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_robot_disabled_scene_query_is_ready_and_empty_after_completed_substep(show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(enable_rigid_fem_contact=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=False),
        material=gs.materials.Rigid(enable_coup_collision=False),
        name="disabled_rigid",
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(radius=0.02),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft",
    )
    scene.build()
    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()
    scene.step(update_visualizer=False)
    batch = scene.get_rigid_fem_contacts()
    assert isinstance(batch, gs.RigidFEMContactBatch)
    assert len(batch) == 0
    assert batch.completed_scene_step == 0
    assert batch.completed_substep == 0
    assert batch.whitelist_receipt.entries[0].collision_enabled is False


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_enabled_scene_exports_nonempty_deterministic_last_substep_batch(show_viewer, precision):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, -9.81)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(enable_rigid_fem_contact=True),
        show_viewer=show_viewer,
    )
    support = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.05), size=(0.12, 0.1, 0.1), fixed=True),
        material=gs.materials.Rigid(coup_collision_links=None),
        name="support",
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(1.0, 0.0, 0.0), size=(0.02, 0.02, 0.02), fixed=False),
        material=gs.materials.Rigid(enable_coup_collision=False),
        name="free_dummy",
    )
    soft_positive = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.025, 0.0, 0.016), size=(0.03, 0.03, 0.03)),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft_positive",
    )
    soft_negative = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.025, 0.0, 0.016), size=(0.03, 0.03, 0.03)),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft_negative",
    )
    ordinary_bits = tuple((geom.contype, geom.conaffinity) for geom in support.geoms)
    scene.build()
    assert tuple((geom.contype, geom.conaffinity) for geom in support.geoms) == ordinary_bits

    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()


    expected_fem_names = {soft_positive.name, soft_negative.name}
    for completed_step in range(40):
        scene.step(update_visualizer=False)
        batch = scene.get_rigid_fem_contacts()
        if expected_fem_names.issubset(batch.fem_entity_names):
            break
    else:
        raise AssertionError("both FEM entities must produce public support contacts within 40 steps")

    assert np.all(batch.env_idx == 0)
    assert set(batch.rigid_entity_names) == {support.name}
    assert set(batch.rigid_link_names) == {support.links[0].name}
    assert expected_fem_names.issubset(batch.fem_entity_names)
    fem_by_name = {entity.name: entity for entity in (soft_positive, soft_negative)}
    for name, element_local in zip(batch.fem_entity_names, batch.fem_element_idx_local, strict=True):
        assert 0 <= element_local < fem_by_name[name].n_elements
    assert set(batch.rigid_geom_idx).issubset({geom.idx for geom in support.geoms})
    assert np.all(batch.normal_impulse_ns >= 0.0)
    assert all(isinstance(mode, gs.RigidFEMContactMode) for mode in batch.modes)
    assert batch.point_m.shape == (len(batch), 3)
    assert batch.signed_gap_m.shape == (len(batch),)
    assert batch.completed_scene_step == completed_step
    assert batch.completed_substep == completed_step

    fresh = scene.get_rigid_fem_contacts()
    assert fresh is not batch
    assert np.array_equal(fresh.point_m, batch.point_m)
    scene.step(update_visualizer=False)
    next_batch = scene.get_rigid_fem_contacts()
    assert next_batch.completed_scene_step == completed_step + 1
    assert next_batch.completed_substep == completed_step + 1
    scene.reset()
    with pytest.raises(gs.RigidFEMContactNotReadyError):
        scene.get_rigid_fem_contacts()


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_heterogeneous_fixed_sphere_routing(show_viewer, precision):
    radii = (0.040, 0.020, 0.020, 0.040)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(
            enable_rigid_fem_contact=True,
            fem_floor_contact_type="none",
            rigid_floor_contact_type="none",
            rigid_rigid_contact_type="none",
        ),
        show_viewer=show_viewer,
    )
    spheres = scene.add_entity(
        morph=tuple(gs.morphs.Sphere(pos=(0.0, 0.0, 0.0), radius=radius, fixed=True) for radius in radii),
        material=gs.materials.Rigid(enable_coup_collision=True, coup_collision_links=None),
        name="heterogeneous_spheres",
    )
    # SAP currently requires at least one rigid DOF; keep the dummy far away and out of the rigid--FEM whitelist.
    scene.add_entity(
        morph=gs.morphs.Box(pos=(1.0, 0.0, 0.0), size=(0.01, 0.01, 0.01), fixed=False),
        material=gs.materials.Rigid(enable_coup_collision=False),
        name="free_dummy",
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.040, 0.0, 0.0), size=(0.030, 0.030, 0.030)),
        material=gs.materials.FEM.Elastic(model="linear_corotated"),
        name="soft_probe",
    )
    scene.build(n_envs=4)

    assert len(spheres.geoms) == 4
    geom_indices = tuple(geom.idx for geom in spheres.geoms)
    for env_idx, geom in enumerate(spheres.geoms):
        assert geom.active_envs_mask is not None
        assert geom.active_envs_mask.detach().cpu().numpy().tolist() == [i == env_idx for i in range(4)]

    aabbs = spheres.get_AABB(envs_idx=np.arange(4, dtype=np.int64)).detach().cpu().numpy()
    extents = aabbs[:, 1] - aabbs[:, 0]
    expected_extents = np.repeat(np.asarray(radii)[:, None] * 2.0, 3, axis=1)
    np.testing.assert_allclose(extents, expected_extents, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(extents[0], extents[3], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(extents[1], extents[2], rtol=0.0, atol=1e-12)

    scene.step(update_visualizer=False)
    rigid_tri_aabbs = scene.sim.coupler.rigid_tri_aabb.aabbs.to_numpy()
    for env_idx, active_geom_idx in enumerate(geom_indices):
        for geom in spheres.geoms:
            face_slice = slice(geom.face_start, geom.face_end)
            finite_faces = np.isfinite(rigid_tri_aabbs["min"][env_idx, face_slice]).all(axis=1)
            finite_faces &= np.isfinite(rigid_tri_aabbs["max"][env_idx, face_slice]).all(axis=1)
            assert int(finite_faces.sum()) == (geom.n_faces if geom.idx == active_geom_idx else 0)

    batch = scene.get_rigid_fem_contacts()
    active_rows = np.asarray(batch.normal_impulse_ns) > 0.0
    for row_idx in np.flatnonzero(active_rows):
        env_idx = int(batch.env_idx[row_idx])
        assert int(batch.rigid_geom_idx[row_idx]) == geom_indices[env_idx]
    assert set(np.asarray(batch.env_idx)[active_rows].tolist()) == {0, 3}
    assert set(np.asarray(batch.rigid_geom_idx)[active_rows].tolist()) == {geom_indices[0], geom_indices[3]}


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_rigid_fem_sap_mu_uses_per_environment_link_friction_ratio(show_viewer, precision):
    fem_mu = 0.4
    coup_friction = 0.6
    ratio_by_env = np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        coupler_options=gs.options.SAPCouplerOptions(
            enable_rigid_fem_contact=True,
            fem_floor_contact_type="none",
            rigid_floor_contact_type="none",
            rigid_rigid_contact_type="none",
        ),
        show_viewer=show_viewer,
    )
    rigid = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, -0.05), size=(0.12, 0.10, 0.10), fixed=False),
        material=gs.materials.Rigid(
            enable_coup_collision=True,
            coup_collision_links=None,
            coup_friction=coup_friction,
        ),
        name="coupled_rigid",
    )
    fem = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.01), size=(0.03, 0.03, 0.03)),
        material=gs.materials.FEM.Elastic(model="linear_corotated", friction_mu=fem_mu),
        name="soft",
    )
    scene.build(n_envs=4)

    rigid_link = rigid.links[0]
    link_idx_local = rigid_link.idx - rigid.link_start
    geom_indices = tuple(geom.idx for geom in rigid_link.geoms)
    assert geom_indices
    envs_idx = np.arange(4, dtype=np.int64)
    initial_ratios = rigid._solver.get_geoms_friction_ratio(geom_indices, envs_idx=envs_idx)
    np.testing.assert_allclose(
        initial_ratios.detach().cpu().numpy(),
        np.ones((4, len(geom_indices)), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    rigid.set_friction_ratio(
        torch.as_tensor(ratio_by_env[:, None], dtype=gs.tc_float, device=gs.device),
        links_idx_local=(link_idx_local,),
        envs_idx=envs_idx,
    )
    updated_ratios = rigid._solver.get_geoms_friction_ratio(geom_indices, envs_idx=envs_idx)
    np.testing.assert_allclose(
        updated_ratios.detach().cpu().numpy(),
        np.repeat(ratio_by_env[:, None], len(geom_indices), axis=1),
        rtol=0.0,
        atol=0.0,
    )

    for completed_step in range(40):
        scene.step(update_visualizer=False)
        handler = scene.sim.coupler.rigid_fem_contact
        n_contacts = int(handler.n_contact_pairs[None])
        batch_idx = np.asarray(handler.contact_pairs.batch_idx.to_numpy(), dtype=np.int64)[:n_contacts]
        if set(batch_idx.tolist()) == set(range(4)):
            break
    else:
        raise AssertionError("the B=4 fixture must produce rigid-FEM contact pairs in every environment")

    rigid_geom_idx = np.asarray(handler.contact_pairs.rigid_geom_idx.to_numpy(), dtype=np.int64)[:n_contacts]
    sap_mu = np.asarray(handler.contact_pairs.sap_info.mu.to_numpy(), dtype=np.float64)[:n_contacts]
    assert set(rigid_geom_idx.tolist()).issubset(set(geom_indices))
    expected_mu = np.sqrt(fem_mu * coup_friction * ratio_by_env[batch_idx])
    np.testing.assert_allclose(sap_mu, expected_mu, rtol=1e-10, atol=1e-12)
    for env_idx, ratio in enumerate(ratio_by_env):
        env_mu = sap_mu[batch_idx == env_idx]
        assert len(env_mu) > 0
        np.testing.assert_allclose(
            env_mu,
            np.full_like(env_mu, np.sqrt(fem_mu * coup_friction * ratio)),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.required
@pytest.mark.precision("64")
def test_rigid_fem_contact_patch_preconditioner_matches_direct_oracle(tmp_path, show_viewer):
    import igl
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    from genesis.engine.couplers.sap_coupler import CONTACT_PATCH_MGS_RELATIVE_NORM_SQUARED

    mesh_path = tmp_path / "tiny_tet_body.mesh"
    outer_vertices = np.asarray(
        [
            [-0.025, -0.020, 0.000],
            [0.025, -0.020, 0.000],
            [0.000, 0.025, 0.000],
            [0.000, 0.000, 0.040],
        ],
        dtype=np.float64,
    )
    # Hydroelastic pressure needs one interior vertex; splitting the outer tetra
    # at its centroid is the smallest volumetric fixture with a nonzero gradient.
    vertices = np.vstack((outer_vertices, outer_vertices.mean(axis=0)))
    igl.writeMESH(
        str(mesh_path),
        vertices,
        np.asarray(
            [
                [0, 1, 2, 4],
                [0, 1, 4, 3],
                [0, 4, 2, 3],
                [4, 1, 2, 3],
            ],
            dtype=np.int64,
        ),
        np.empty((0, 3), dtype=np.int64),
    )

    low_budgets = (1, 8, 20)
    high_budget = 80

    def build_and_step(patch_option, schwarz_option, budget, *, rigid_fixed=False):
        coupler_kwargs = {
            "n_sap_iterations": 1,
            "n_pcg_iterations": budget,
            "n_linesearch_iterations": 8,
            "pcg_threshold": 1.0e-14,
            "fem_floor_contact_type": "none",
            "enable_fem_self_tet_contact": False,
            "rigid_floor_contact_type": "none",
            "enable_rigid_fem_contact": True,
            "rigid_rigid_contact_type": "none",
            "enable_completed_solver_health": True,
        }
        if patch_option is not None:
            coupler_kwargs["enable_rigid_fem_contact_patch_preconditioner"] = patch_option
        if schwarz_option is not None:
            coupler_kwargs["enable_rigid_fem_contact_tet_schwarz_preconditioner"] = schwarz_option
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0e-3, substeps=1, gravity=(0.0, 0.0, 0.0)),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            fem_options=gs.options.FEMOptions(
                enable_floor=False,
                use_implicit_solver=True,
                n_newton_iterations=1,
                n_pcg_iterations=80,
                n_linesearch_iterations=0,
                damping_alpha=0.0,
                damping_beta=0.0,
            ),
            coupler_options=gs.options.SAPCouplerOptions(**coupler_kwargs),
            show_viewer=show_viewer,
        )
        rigid = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, -0.010),
                size=(0.060, 0.060, 0.030),
                fixed=rigid_fixed,
            ),
            material=gs.materials.Rigid(
                enable_coup_collision=True,
                coup_collision_links=None,
                coup_friction=0.4,
            ),
            name="direct_oracle_rigid",
        )
        if rigid_fixed:
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(1.0, 0.0, 0.0),
                    size=(0.01, 0.01, 0.01),
                    fixed=False,
                ),
                material=gs.materials.Rigid(enable_coup_collision=False),
                name="direct_oracle_free_dummy",
            )
        fem = scene.add_entity(
            morph=gs.morphs.TetMesh(file=str(mesh_path)),
            material=gs.materials.FEM.Elastic(
                E=2.0e5,
                nu=0.35,
                rho=950.0,
                friction_mu=0.35,
                model="linear_corotated",
            ),
            name="direct_oracle_fem",
        )
        scene.build()
        if not rigid_fixed:
            rigid.set_dofs_velocity(
                np.asarray([0.10, -0.07, 0.20, 0.0, 0.0, 0.0], dtype=np.float64)
            )
        scene.step(update_visualizer=False)
        batch = scene.get_rigid_fem_contacts()
        assert len(batch) > 0
        health = scene.get_last_completed_solver_health().substeps[0]
        snapshot = {
            "fem_velocity": np.asarray(fem.get_state(track_grad=False).vel.detach().cpu(), dtype=np.float64)[0],
            "rigid_velocity": np.asarray(rigid.get_dofs_velocity().detach().cpu(), dtype=np.float64).reshape(-1),
            "normal_impulse": np.asarray(batch.normal_impulse_ns, dtype=np.float64),
            "tangential_impulse": np.asarray(batch.tangential_impulse_world_ns, dtype=np.float64),
            "modes": batch.modes,
            "health": health,
        }
        return scene, rigid, fem, snapshot

    default_scene, _, _, default = build_and_step(None, None, low_budgets[0])
    false_scene, _, _, explicit_false = build_and_step(False, False, low_budgets[0])
    balanced_repeats_by_budget = {}
    for budget in low_budgets:
        balanced_repeats_by_budget[budget] = tuple(
            build_and_step(True, False, budget)[3] for _ in range(2)
        )
    balanced_scene, _, _, balanced_high = build_and_step(True, False, high_budget)
    treatment_by_budget = {}
    for budget in low_budgets:
        _, _, _, treatment_by_budget[budget] = build_and_step(True, True, budget)
    treatment_scene, treatment_rigid, treatment_fem, treatment_high = build_and_step(
        True, True, high_budget
    )
    fixed_scene, fixed_rigid, fixed_fem, fixed_snapshot = build_and_step(
        True, True, low_budgets[0], rigid_fixed=True
    )

    assert not hasattr(default_scene.sim.coupler, "rigid_fem_contact_patch_coarse_matrix")
    assert not hasattr(false_scene.sim.coupler, "rigid_fem_contact_patch_coarse_matrix")
    assert not hasattr(default_scene.sim.coupler, "rigid_fem_contact_tet_schwarz_packed_factor")
    assert not hasattr(false_scene.sim.coupler, "rigid_fem_contact_tet_schwarz_packed_factor")
    assert not hasattr(balanced_scene.sim.coupler, "rigid_fem_contact_tet_schwarz_packed_factor")
    assert default["health"].rigid_fem_contact_patch_preconditioner_enabled is False
    assert default["health"].rigid_fem_contact_patch_min_active_count_by_batch == ()
    assert explicit_false["health"].rigid_fem_contact_patch_preconditioner_enabled is False
    assert default["health"].rigid_fem_contact_tet_schwarz_preconditioner_enabled is False
    assert default["health"].rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch == ()
    assert explicit_false["health"].rigid_fem_contact_tet_schwarz_preconditioner_enabled is False
    assert balanced_high["health"].rigid_fem_contact_tet_schwarz_preconditioner_enabled is False
    for name in ("fem_velocity", "rigid_velocity", "normal_impulse", "tangential_impulse"):
        assert np.array_equal(default[name], explicit_false[name])
    assert default["modes"] == explicit_false["modes"]

    coupler = treatment_scene.sim.coupler
    health = treatment_high["health"]
    assert coupler._rigid_fem_contact_patch_n_templates == 1
    assert health.rigid_fem_contact_patch_preconditioner_enabled is True
    assert health.rigid_fem_contact_patch_min_active_count_by_batch == (1,)
    assert health.rigid_fem_contact_patch_all_coarse_finite_by_batch == (True,)
    assert health.rigid_fem_contact_tet_schwarz_preconditioner_enabled is True
    assert health.rigid_fem_contact_tet_schwarz_min_active_block_count_by_batch[0] > 0
    assert health.rigid_fem_contact_tet_schwarz_max_vertex_overlap_by_batch[0] >= 1
    assert health.rigid_fem_contact_tet_schwarz_max_link_overlap_by_batch[0] >= 1
    assert 0 < health.rigid_fem_contact_tet_schwarz_max_link_rank_by_batch[0] <= 6
    assert health.rigid_fem_contact_tet_schwarz_min_factor_pivot_by_batch[0] > 0.0
    assert health.rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch == (True,)
    rank = int(coupler.rigid_fem_contact_patch_rank.to_numpy()[0, 0])
    retained_mask = int(coupler.rigid_fem_contact_patch_retained_mask.to_numpy()[0, 0])
    assert rank == retained_mask.bit_count()
    assert health.rigid_fem_contact_patch_min_rank_by_batch[0] <= rank <= 6
    assert rank > 0

    def prepare_schwarz_stages(active_coupler, *, factor):
        active_coupler._begin_rigid_fem_contact_tet_schwarz_outer_setup()
        active_coupler._mark_rigid_fem_contact_tet_schwarz_pairs()
        active_coupler._compact_rigid_fem_contact_tet_schwarz_pairs()
        active_coupler._count_rigid_fem_contact_tet_schwarz_pair_multiplicity()
        active_coupler._assemble_rigid_fem_contact_tet_schwarz_blocks()
        for link_slot in range(active_coupler._rigid_fem_contact_patch_n_enabled_links):
            for mode in range(6):
                active_coupler._build_rigid_fem_contact_tet_schwarz_link_mass_response(
                    link_slot,
                    mode,
                    links_info=active_coupler.rigid_solver.links_info,
                    links_state=active_coupler.rigid_solver.links_state,
                    dofs_state=active_coupler.rigid_solver.dofs_state,
                    entities_info=active_coupler.rigid_solver.entities_info,
                    rigid_global_info=active_coupler.rigid_solver._rigid_global_info,
                )
                active_coupler._reduce_rigid_fem_contact_tet_schwarz_link_delassus_column(
                    link_slot,
                    mode,
                    links_info=active_coupler.rigid_solver.links_info,
                    links_state=active_coupler.rigid_solver.links_state,
                    dofs_state=active_coupler.rigid_solver.dofs_state,
                )
            active_coupler._initialize_rigid_fem_contact_tet_schwarz_link_twist_basis(
                link_slot
            )
            for mode in range(6):
                for previous in range(mode):
                    active_coupler._reduce_rigid_fem_contact_tet_schwarz_link_twist_dot(
                        link_slot, previous, mode
                    )
                    active_coupler._project_rigid_fem_contact_tet_schwarz_link_twist_mode(
                        link_slot, mode, previous
                    )
                active_coupler._reduce_rigid_fem_contact_tet_schwarz_link_twist_dot(
                    link_slot, mode, mode
                )
                active_coupler._finish_rigid_fem_contact_tet_schwarz_link_twist_mode(
                    link_slot, mode
                )
            active_coupler._factor_rigid_fem_contact_tet_schwarz_link_reduced_metric(
                link_slot
            )
            active_coupler._solve_rigid_fem_contact_tet_schwarz_dynamic_link_basis(
                link_slot
            )
            active_coupler._validate_rigid_fem_contact_tet_schwarz_dynamic_link_basis(
                link_slot,
                links_info=active_coupler.rigid_solver.links_info,
                links_state=active_coupler.rigid_solver.links_state,
                dofs_state=active_coupler.rigid_solver.dofs_state,
            )
            for mode in range(6):
                active_coupler._reset_rigid_fem_contact_tet_schwarz_link_basis_hessian_invalid(
                    link_slot, mode
                )
                active_coupler._copy_rigid_fem_contact_tet_schwarz_link_basis_to_p(
                    link_slot, mode
                )
                active_coupler._apply_rigid_fem_contact_tet_schwarz_hessian(
                    rigid_global_info=active_coupler.rigid_solver._rigid_global_info,
                )
                active_coupler._cache_rigid_fem_contact_tet_schwarz_link_basis_hessian(
                    link_slot, mode
                )
                active_coupler._finalize_rigid_fem_contact_tet_schwarz_link_basis_hessian_validity(
                    link_slot, mode
                )
        active_coupler._assemble_rigid_fem_contact_tet_schwarz_coupled_blocks()
        if factor:
            active_coupler._factor_rigid_fem_contact_tet_schwarz_blocks()

    # Public-contact finalization evaluates G once more at the accepted velocity. Rebuild
    # the exact coupled blocks so every device field and the direct oracle share that state.
    coupler.batch_active.from_numpy(np.asarray([True], dtype=np.bool_))
    coupler._begin_rigid_fem_contact_patch_outer_setup()
    prepare_schwarz_stages(coupler, factor=False)
    n_active_pairs = min(
        int(coupler.rigid_fem_contact_tet_schwarz_active_pair_count[None]),
        coupler._rigid_fem_contact_tet_schwarz_max_active_pairs,
    )
    assert n_active_pairs > 0
    assembled_packed = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_packed_factor.to_numpy(), dtype=np.float64
    )[:n_active_pairs].copy()
    assembled_hff = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_tet_packed_hff.to_numpy(), dtype=np.float64
    )[0].copy()
    pair_batch = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_pair_batch.to_numpy(), dtype=np.int64
    )[:n_active_pairs]
    pair_surface_slot = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_pair_surface_slot.to_numpy(), dtype=np.int64
    )[:n_active_pairs]
    pair_link_slot = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_pair_link_slot.to_numpy(), dtype=np.int64
    )[:n_active_pairs]
    assert np.array_equal(pair_batch, np.zeros(n_active_pairs, dtype=np.int64))
    block_vertices = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_block_vertices.to_numpy(), dtype=np.int64
    )
    vertex_multiplicity = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_vertex_multiplicity.to_numpy(), dtype=np.int64
    )[0]
    link_multiplicity = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_multiplicity.to_numpy(), dtype=np.int64
    )[0]
    link_rank = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_rank.to_numpy(), dtype=np.int64
    )[0]
    link_mask = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_retained_mask.to_numpy(), dtype=np.int64
    )[0]
    link_delassus = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_delassus.to_numpy(), dtype=np.float64
    )[0]
    link_twist_basis = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_twist_basis.to_numpy(), dtype=np.float64
    )[0]
    link_basis_rigid = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_basis_rigid.to_numpy(), dtype=np.float64
    )[0]
    link_reduced_factor = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_link_reduced_factor.to_numpy(), dtype=np.float64
    )[0]
    enabled_links = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_enabled_link_by_slot.to_numpy(), dtype=np.int64
    )
    coupler._factor_rigid_fem_contact_tet_schwarz_blocks()
    factor_packed = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_packed_factor.to_numpy(), dtype=np.float64
    )[:n_active_pairs].copy()
    factor_valid = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_factor_valid.to_numpy(), dtype=np.bool_
    )[:n_active_pairs]
    active_dimension = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_pair_active_dimension.to_numpy(), dtype=np.int64
    )[:n_active_pairs]
    factor_min_pivot = np.asarray(
        coupler.rigid_fem_contact_tet_schwarz_pair_min_pivot.to_numpy(), dtype=np.float64
    )[:n_active_pairs]
    assert factor_valid.all()
    coupler._prepare_rigid_fem_contact_patch_preconditioner()
    rank = int(coupler.rigid_fem_contact_patch_rank.to_numpy()[0, 0])
    retained_mask = int(coupler.rigid_fem_contact_patch_retained_mask.to_numpy()[0, 0])

    q_fem = np.asarray(coupler.rigid_fem_contact_patch_q_fem.to_numpy(), dtype=np.float64)[0, 0]
    q_rigid = np.asarray(coupler.rigid_fem_contact_patch_q_rigid.to_numpy(), dtype=np.float64)[0, 0]
    hq_fem = np.asarray(coupler.rigid_fem_contact_patch_hq_fem.to_numpy(), dtype=np.float64)[0, 0]
    hq_rigid = np.asarray(coupler.rigid_fem_contact_patch_hq_rigid.to_numpy(), dtype=np.float64)[0, 0]
    coarse_basis = np.column_stack(
        [np.concatenate((q_fem[:, mode].reshape(-1), q_rigid[:, mode])) for mode in range(6)]
    )
    hessian_coarse_basis = np.column_stack(
        [np.concatenate((hq_fem[:, mode].reshape(-1), hq_rigid[:, mode])) for mode in range(6)]
    )
    coarse_inverse = np.asarray(
        coupler.rigid_fem_contact_patch_coarse_inverse.to_numpy(), dtype=np.float64
    )[0, 0]

    n_vertices = treatment_fem.n_vertices
    n_dofs = treatment_rigid.n_dofs
    dimension = 3 * n_vertices + n_dofs

    def matrix_free(vector):
        vector = np.asarray(vector, dtype=np.float64)
        coupler.batch_pcg_active.from_numpy(np.asarray([True], dtype=np.bool_))
        coupler.pcg_fem_state_v.p.from_numpy(vector[: 3 * n_vertices].reshape(1, n_vertices, 3))
        coupler.pcg_rigid_state_dof.p.from_numpy(vector[3 * n_vertices :].reshape(1, n_dofs))
        coupler._apply_rigid_fem_contact_patch_hessian(
            0,
            0,
            rigid_global_info=treatment_scene.sim.rigid_solver._rigid_global_info
        )
        return np.concatenate(
            (
                np.asarray(coupler.pcg_fem_state_v.Ap.to_numpy(), dtype=np.float64).reshape(-1),
                np.asarray(coupler.pcg_rigid_state_dof.Ap.to_numpy(), dtype=np.float64).reshape(-1),
            )
        )

    dense_hessian = np.column_stack(
        [matrix_free(np.eye(dimension, dtype=np.float64)[:, column]) for column in range(dimension)]
    )
    matrix_scale = max(1.0, float(np.linalg.norm(dense_hessian, ord=np.inf)))
    np.testing.assert_allclose(
        dense_hessian,
        dense_hessian.T,
        rtol=1.0e-9,
        atol=1.0e-11 * matrix_scale,
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (dense_hessian + dense_hessian.T))
    assert float(eigenvalues[0]) > 1.0e-12 * matrix_scale
    deterministic_q = np.sin(np.arange(1, dimension + 1, dtype=np.float64) * 0.37)
    deterministic_hq = matrix_free(deterministic_q)
    np.testing.assert_allclose(
        deterministic_hq,
        dense_hessian @ deterministic_q,
        rtol=1.0e-9,
        atol=1.0e-11 * matrix_scale * np.linalg.norm(deterministic_q),
    )
    np.testing.assert_allclose(
        hessian_coarse_basis,
        dense_hessian @ coarse_basis,
        rtol=1.0e-9,
        atol=1.0e-11 * matrix_scale,
    )

    def unpack_lower(packed, dimension):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for row in range(dimension):
            for column in range(row + 1):
                matrix[row, column] = packed[row * (row + 1) // 2 + column]
        return matrix

    def link_info_vector(field):
        values = np.asarray(field.to_numpy())
        if coupler.rigid_solver._options.batch_links_info:
            return values[:, 0]
        return values

    rigid_mass = np.asarray(
        coupler.rigid_solver._rigid_global_info.mass_mat.to_numpy(), dtype=np.float64
    )[:, :, 0]
    rigid_mass_inverse = np.linalg.inv(rigid_mass)
    link_n_dofs = link_info_vector(coupler.rigid_solver.links_info.n_dofs).astype(np.int64)
    link_dof_end = link_info_vector(coupler.rigid_solver.links_info.dof_end).astype(np.int64)
    link_parent = link_info_vector(coupler.rigid_solver.links_info.parent_idx).astype(np.int64)
    link_origin = np.asarray(
        coupler.rigid_solver.links_state.i_pos.to_numpy(), dtype=np.float64
    )[:, 0]
    cdof_velocity = np.asarray(
        coupler.rigid_solver.dofs_state.cdof_vel.to_numpy(), dtype=np.float64
    )[:, 0]
    cdof_angular = np.asarray(
        coupler.rigid_solver.dofs_state.cdof_ang.to_numpy(), dtype=np.float64
    )[:, 0]

    def host_link_jacobian(global_link):
        jacobian = np.zeros((6, n_dofs), dtype=np.float64)
        selected_origin = link_origin[global_link]
        ancestor = int(global_link)
        while ancestor >= 0:
            dof_end = int(link_dof_end[ancestor])
            dof_start = dof_end - int(link_n_dofs[ancestor])
            for dof in range(dof_start, dof_end):
                angular = cdof_angular[dof]
                jacobian[:3, dof] = cdof_velocity[dof] + np.cross(
                    angular, selected_origin
                )
                jacobian[3:, dof] = angular
            ancestor = int(link_parent[ancestor])
        return jacobian

    def replay_link_mgs(delassus):
        basis = np.array(delassus, dtype=np.float64, copy=True)
        max_norm_squared = max(
            float(np.dot(basis[:, mode], basis[:, mode])) for mode in range(6)
        )
        threshold = CONTACT_PATCH_MGS_RELATIVE_NORM_SQUARED * max_norm_squared
        mask = 0
        for mode in range(6):
            for previous in range(mode):
                basis[:, mode] -= (
                    float(np.dot(basis[:, previous], basis[:, mode]))
                    * basis[:, previous]
                )
            norm_squared = float(np.dot(basis[:, mode], basis[:, mode]))
            if np.isfinite(norm_squared) and norm_squared > threshold:
                basis[:, mode] /= np.sqrt(norm_squared)
                mask |= 1 << mode
            else:
                basis[:, mode] = 0.0
        return basis, mask

    link_host = {}
    for link_slot in np.unique(pair_link_slot):
        global_link = int(enabled_links[link_slot])
        jacobian = host_link_jacobian(global_link)
        mass_response = rigid_mass_inverse @ jacobian.T
        delassus = jacobian @ mass_response
        host_basis, host_mask = replay_link_mgs(delassus)
        host_modes = np.asarray(
            [mode for mode in range(6) if host_mask & (1 << mode)], dtype=np.int64
        )
        assert int(link_mask[link_slot]) == host_mask
        assert int(link_rank[link_slot]) == len(host_modes)
        np.testing.assert_allclose(
            link_delassus[link_slot],
            delassus,
            rtol=1.0e-9,
            atol=1.0e-11 * max(1.0, float(np.linalg.norm(delassus, ord=np.inf))),
        )
        np.testing.assert_allclose(
            link_twist_basis[link_slot],
            host_basis,
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        retained_basis = host_basis[:, host_modes]
        if len(host_modes):
            reduced_metric = retained_basis.T @ delassus @ retained_basis
            expected_g_active = (
                mass_response @ retained_basis @ np.linalg.inv(reduced_metric)
            )
            production_g_active = link_basis_rigid[link_slot][:, host_modes]
            np.testing.assert_allclose(
                production_g_active,
                expected_g_active,
                rtol=1.0e-9,
                atol=1.0e-11 * max(1.0, float(np.linalg.norm(expected_g_active, ord=np.inf))),
            )
            np.testing.assert_allclose(
                retained_basis.T @ retained_basis,
                np.eye(len(host_modes)),
                rtol=1.0e-9,
                atol=1.0e-11,
            )
            np.testing.assert_allclose(
                jacobian @ production_g_active,
                retained_basis,
                rtol=1.0e-9,
                atol=1.0e-11,
            )
            reduced_factor = unpack_lower(link_reduced_factor[link_slot], 6)
            reduced_factor_active = reduced_factor[np.ix_(host_modes, host_modes)]
            np.testing.assert_allclose(
                reduced_factor_active @ reduced_factor_active.T,
                reduced_metric,
                rtol=1.0e-9,
                atol=1.0e-11 * max(1.0, float(np.linalg.norm(reduced_metric, ord=np.inf))),
            )
        else:
            production_g_active = np.empty((n_dofs, 0), dtype=np.float64)
        pinv_delassus = np.linalg.pinv(delassus)
        projector = delassus @ pinv_delassus
        np.testing.assert_allclose(
            host_basis @ host_basis.T,
            projector,
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        literal_g = mass_response @ pinv_delassus
        np.testing.assert_allclose(
            link_basis_rigid[link_slot] @ host_basis.T,
            literal_g,
            rtol=1.0e-9,
            atol=1.0e-11 * max(1.0, float(np.linalg.norm(literal_g, ord=np.inf))),
        )
        rejected_modes = [mode for mode in range(6) if not host_mask & (1 << mode)]
        if rejected_modes:
            assert np.array_equal(
                link_basis_rigid[link_slot][:, rejected_modes],
                np.zeros((n_dofs, len(rejected_modes)), dtype=np.float64),
            )
        link_host[int(link_slot)] = {
            "modes": host_modes,
            "basis": host_basis,
            "g": link_basis_rigid[link_slot],
            "literal_g": literal_g,
        }

    local_blocks = {}
    eps = np.finfo(np.float64).eps
    for pair_slot in range(n_active_pairs):
        surface_slot = int(pair_surface_slot[pair_slot])
        link_slot = int(pair_link_slot[pair_slot])
        vertices_for_block = block_vertices[surface_slot]
        fem_dofs = np.asarray(
            [3 * vertex + axis for vertex in vertices_for_block for axis in range(3)],
            dtype=np.int64,
        )
        hff = unpack_lower(assembled_hff[surface_slot], 12)
        hff = hff + np.tril(hff, -1).T
        expected_hff = dense_hessian[np.ix_(fem_dofs, fem_dofs)]
        np.testing.assert_allclose(
            hff,
            expected_hff,
            rtol=1.0e-9,
            atol=1.0e-11 * max(1.0, float(np.linalg.norm(expected_hff, ord=np.inf))),
        )

        modes = link_host[link_slot]["modes"]
        local_dimension = 12 + len(modes)
        active_coordinates = np.concatenate((np.arange(12), 12 + modes)).astype(np.int64)
        injection = np.zeros((dimension, local_dimension), dtype=np.float64)
        injection[np.ix_(fem_dofs, np.arange(12))] = np.eye(12)
        if len(modes):
            injection[3 * n_vertices :, 12:] = link_host[link_slot]["g"][:, modes]
        expected = injection.T @ dense_hessian @ injection
        assembled_full = unpack_lower(assembled_packed[pair_slot], 18)
        assembled_full = assembled_full + np.tril(assembled_full, -1).T
        assembled = assembled_full[np.ix_(active_coordinates, active_coordinates)]
        block_scale = max(1.0, float(np.linalg.norm(expected, ord=np.inf)))
        np.testing.assert_allclose(assembled, expected, rtol=1.0e-9, atol=1.0e-11 * block_scale)

        factor_full = unpack_lower(factor_packed[pair_slot], 18)
        factor = factor_full[np.ix_(active_coordinates, active_coordinates)]
        reconstruction_error = float(np.linalg.norm(factor @ factor.T - assembled, ord=np.inf))
        reconstruction_scale = local_dimension * eps * max(
            1.0, float(np.linalg.norm(assembled, ord=np.inf))
        )
        normalized_reconstruction = reconstruction_error / reconstruction_scale
        assert normalized_reconstruction <= 30.0, (
            pair_slot,
            reconstruction_error,
            reconstruction_scale,
            normalized_reconstruction,
        )
        host_factor = np.zeros_like(assembled)
        host_pivots = []
        for column in range(local_dimension):
            pivot = assembled[column, column] - float(
                np.dot(host_factor[column, :column], host_factor[column, :column])
            )
            host_pivots.append(pivot)
            host_factor[column, column] = np.sqrt(pivot)
            for row in range(column + 1, local_dimension):
                host_factor[row, column] = (
                    assembled[row, column]
                    - float(
                        np.dot(
                            host_factor[row, :column],
                            host_factor[column, :column],
                        )
                    )
                ) / host_factor[column, column]
        assert active_dimension[pair_slot] == local_dimension
        assert min(host_pivots) > 0.0
        np.testing.assert_allclose(
            factor_min_pivot[pair_slot], min(host_pivots), rtol=1.0e-9, atol=1.0e-11
        )
        local_blocks[pair_slot] = {
            "surface_slot": surface_slot,
            "link_slot": link_slot,
            "vertices": vertices_for_block,
            "fem_dofs": fem_dofs,
            "modes": modes,
            "injection": injection,
            "block": assembled,
        }
        print(
            "CONTACT_TET_SCHWARZ_BLOCK_METRIC",
            {
                "slot": pair_slot,
                "dimension": local_dimension,
                "assembly_max_abs": float(np.max(np.abs(assembled - expected))),
                "cholesky_normalized_residual": normalized_reconstruction,
            },
        )

    fixed_coupler = fixed_scene.sim.coupler
    fixed_health = fixed_snapshot["health"]
    assert fixed_rigid.n_dofs == 0
    assert fixed_health.rigid_fem_contact_tet_schwarz_preconditioner_enabled is True
    assert fixed_health.rigid_fem_contact_tet_schwarz_max_link_rank_by_batch == (0,)
    assert fixed_health.rigid_fem_contact_tet_schwarz_all_factors_valid_by_batch == (True,)
    fixed_coupler.batch_active.from_numpy(np.asarray([True], dtype=np.bool_))
    fixed_coupler._begin_rigid_fem_contact_patch_outer_setup()
    prepare_schwarz_stages(fixed_coupler, factor=False)
    fixed_pair_count = min(
        int(fixed_coupler.rigid_fem_contact_tet_schwarz_active_pair_count[None]),
        fixed_coupler._rigid_fem_contact_tet_schwarz_max_active_pairs,
    )
    assert fixed_pair_count > 0
    fixed_pair_surface = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_pair_surface_slot.to_numpy(),
        dtype=np.int64,
    )[:fixed_pair_count]
    fixed_pair_link = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_pair_link_slot.to_numpy(),
        dtype=np.int64,
    )[:fixed_pair_count]
    fixed_link_rank = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_link_rank.to_numpy(), dtype=np.int64
    )[0]
    fixed_link_mask = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_link_retained_mask.to_numpy(),
        dtype=np.int64,
    )[0]
    assert np.array_equal(fixed_link_rank[np.unique(fixed_pair_link)], 0)
    assert np.array_equal(fixed_link_mask[np.unique(fixed_pair_link)], 0)
    fixed_hff_packed = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_tet_packed_hff.to_numpy(),
        dtype=np.float64,
    )[0]
    fixed_coupler._factor_rigid_fem_contact_tet_schwarz_blocks()
    fixed_active_dimension = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_pair_active_dimension.to_numpy(),
        dtype=np.int64,
    )[:fixed_pair_count]
    fixed_factor_valid = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_factor_valid.to_numpy(), dtype=np.bool_
    )[:fixed_pair_count]
    assert np.array_equal(fixed_active_dimension, np.full(fixed_pair_count, 12))
    assert fixed_factor_valid.all()

    fixed_vertex_count = fixed_fem.n_vertices
    fixed_dimension = 3 * fixed_vertex_count

    def fixed_matrix_free(vector):
        fixed_coupler.batch_pcg_active.from_numpy(np.asarray([True], dtype=np.bool_))
        fixed_coupler.pcg_fem_state_v.p.from_numpy(
            np.asarray(vector, dtype=np.float64).reshape(1, fixed_vertex_count, 3)
        )
        fixed_coupler.pcg_rigid_state_dof.p.from_numpy(
            np.zeros((1, fixed_coupler.rigid_solver.n_dofs), dtype=np.float64)
        )
        fixed_coupler._apply_rigid_fem_contact_patch_hessian(
            0,
            0,
            rigid_global_info=fixed_coupler.rigid_solver._rigid_global_info,
        )
        return np.asarray(
            fixed_coupler.pcg_fem_state_v.Ap.to_numpy(), dtype=np.float64
        ).reshape(-1)

    fixed_hessian = np.column_stack(
        [
            fixed_matrix_free(np.eye(fixed_dimension, dtype=np.float64)[:, column])
            for column in range(fixed_dimension)
        ]
    )
    fixed_block_vertices = np.asarray(
        fixed_coupler.rigid_fem_contact_tet_schwarz_block_vertices.to_numpy(),
        dtype=np.int64,
    )
    for surface_slot in np.unique(fixed_pair_surface):
        fem_dofs = np.asarray(
            [
                3 * vertex + axis
                for vertex in fixed_block_vertices[surface_slot]
                for axis in range(3)
            ],
            dtype=np.int64,
        )
        fixed_hff = unpack_lower(fixed_hff_packed[surface_slot], 12)
        fixed_hff = fixed_hff + np.tril(fixed_hff, -1).T
        expected_fixed_hff = fixed_hessian[np.ix_(fem_dofs, fem_dofs)]
        np.testing.assert_allclose(
            fixed_hff,
            expected_fixed_hff,
            rtol=1.0e-9,
            atol=1.0e-11
            * max(1.0, float(np.linalg.norm(expected_fixed_hff, ord=np.inf))),
        )
    fixed_load = np.cos(np.arange(1, fixed_dimension + 1, dtype=np.float64) * 0.19)
    fixed_coupler.rigid_fem_contact_patch_correction_load_fem.from_numpy(
        fixed_load.reshape(1, fixed_vertex_count, 3)
    )
    fixed_coupler._apply_rigid_fem_contact_tet_schwarz_local_preconditioner()
    fixed_rigid_result = np.asarray(
        fixed_coupler.pcg_rigid_state_dof.z.to_numpy(), dtype=np.float64
    )
    assert fixed_rigid_result.shape == (1, fixed_coupler.rigid_solver.n_dofs)
    np.testing.assert_array_equal(fixed_rigid_result, np.zeros_like(fixed_rigid_result))

    coarse = np.asarray(coupler.rigid_fem_contact_patch_coarse_matrix.to_numpy(), dtype=np.float64)[0, 0]
    expected_coarse = coarse_basis.T @ dense_hessian @ coarse_basis
    for mode in range(6):
        if not retained_mask & (1 << mode):
            expected_coarse[mode, :] = 0.0
            expected_coarse[:, mode] = 0.0
            expected_coarse[mode, mode] = 1.0
    coarse_scale = max(1.0, float(np.linalg.norm(coarse, ord=np.inf)))
    np.testing.assert_allclose(coarse, expected_coarse, rtol=1.0e-9, atol=1.0e-11 * coarse_scale)
    assert np.all(np.isfinite(coarse))
    assert np.all(np.isfinite(coarse_inverse))
    np.testing.assert_allclose(coarse, coarse.T, rtol=1.0e-9, atol=1.0e-11 * coarse_scale)
    active_modes = np.asarray([mode for mode in range(6) if retained_mask & (1 << mode)], dtype=np.int64)
    inactive_modes = np.asarray([mode for mode in range(6) if not retained_mask & (1 << mode)], dtype=np.int64)
    assert len(active_modes) == rank
    for mode in inactive_modes:
        canonical = np.zeros(6, dtype=np.float64)
        canonical[mode] = 1.0
        assert np.array_equal(coarse[mode, :], canonical)
        assert np.array_equal(coarse[:, mode], canonical)
        assert np.array_equal(coarse_inverse[mode, :], canonical)
        assert np.array_equal(coarse_inverse[:, mode], canonical)
    active_coarse = coarse[np.ix_(active_modes, active_modes)]
    active_inverse = coarse_inverse[np.ix_(active_modes, active_modes)]
    active_eigenvalues = np.linalg.eigvalsh(0.5 * (active_coarse + active_coarse.T))
    assert float(active_eigenvalues[0]) > 0.0
    active_identity = np.eye(rank, dtype=np.float64)
    inverse_left_residual = float(np.linalg.norm(active_coarse @ active_inverse - active_identity, ord=np.inf))
    inverse_right_residual = float(np.linalg.norm(active_inverse @ active_coarse - active_identity, ord=np.inf))
    coarse_kappa_inf = float(
        np.linalg.norm(active_coarse, ord=np.inf) * np.linalg.norm(active_inverse, ord=np.inf)
    )
    inverse_residual_scale = rank * np.finfo(np.float64).eps * coarse_kappa_inf
    normalized_inverse_residual = max(inverse_left_residual, inverse_right_residual) / inverse_residual_scale
    assert normalized_inverse_residual <= 30.0, (
        inverse_left_residual,
        inverse_right_residual,
        inverse_residual_scale,
        normalized_inverse_residual,
        coarse_kappa_inf,
    )
    print(
        "BALANCED_PATCH_OPERATOR_METRICS",
        {
            "rank": rank,
            "h_symmetry_inf": float(np.linalg.norm(dense_hessian - dense_hessian.T, ord=np.inf)),
            "h_min_eigenvalue": float(eigenvalues[0]),
            "matrix_free_hq_max_abs": float(np.max(np.abs(deterministic_hq - dense_hessian @ deterministic_q))),
            "stored_hq_max_abs": float(np.max(np.abs(hessian_coarse_basis - dense_hessian @ coarse_basis))),
            "e_min_eigenvalue": float(active_eigenvalues[0]),
            "e_kappa_inf": coarse_kappa_inf,
            "inverse_left_inf": inverse_left_residual,
            "inverse_right_inf": inverse_right_residual,
            "inverse_normalized_residual": normalized_inverse_residual,
        },
    )

    def baseline_precondition(vector):
        vector = np.asarray(vector, dtype=np.float64)
        coupler.batch_pcg_active.from_numpy(np.asarray([True], dtype=np.bool_))
        coupler.rigid_fem_contact_patch_correction_load_fem.from_numpy(
            vector[: 3 * n_vertices].reshape(1, n_vertices, 3)
        )
        coupler.rigid_fem_contact_patch_correction_load_rigid.from_numpy(
            vector[3 * n_vertices :].reshape(1, n_dofs)
        )
        coupler._apply_rigid_fem_contact_patch_correction_p0(
            entities_info=coupler.rigid_solver.entities_info,
            rigid_global_info=coupler.rigid_solver._rigid_global_info,
        )
        return np.concatenate(
            (
                np.asarray(coupler.pcg_fem_state_v.z.to_numpy(), dtype=np.float64).reshape(-1),
                np.asarray(coupler.pcg_rigid_state_dof.z.to_numpy(), dtype=np.float64).reshape(-1),
            )
        )

    dense_baseline = np.column_stack(
        [baseline_precondition(np.eye(dimension, dtype=np.float64)[:, column]) for column in range(dimension)]
    )
    dense_schwarz = np.zeros_like(dense_baseline)
    dense_schwarz_literal_pinv = np.zeros_like(dense_baseline)
    for local in local_blocks.values():
        link_slot = local["link_slot"]
        modes = local["modes"]
        link_weight = 1.0 / np.sqrt(link_multiplicity[link_slot])
        weights = np.concatenate(
            (
                np.repeat(
                    1.0 / np.sqrt(vertex_multiplicity[local["vertices"]]), 3
                ),
                np.full(len(modes), link_weight, dtype=np.float64),
            )
        )
        weighted_inverse = (
            weights[:, None] * np.linalg.inv(local["block"]) * weights[None, :]
        )
        dense_schwarz += (
            local["injection"] @ weighted_inverse @ local["injection"].T
        )

        literal_injection = np.zeros((dimension, 18), dtype=np.float64)
        literal_injection[np.ix_(local["fem_dofs"], np.arange(12))] = np.eye(12)
        literal_injection[3 * n_vertices :, 12:] = link_host[link_slot]["literal_g"]
        literal_block = literal_injection.T @ dense_hessian @ literal_injection
        literal_weights = np.concatenate(
            (
                weights[:12],
                np.full(6, link_weight, dtype=np.float64),
            )
        )
        dense_schwarz_literal_pinv += (
            literal_injection
            @ (
                literal_weights[:, None]
                * np.linalg.pinv(literal_block)
                * literal_weights[None, :]
            )
            @ literal_injection.T
        )
    np.testing.assert_allclose(
        dense_schwarz,
        dense_schwarz_literal_pinv,
        rtol=1.0e-9,
        atol=1.0e-11 * max(1.0, float(np.linalg.norm(dense_schwarz, ord=np.inf))),
    )

    coarse_correction = coarse_basis @ coarse_inverse @ coarse_basis.T
    identity = np.eye(dimension, dtype=np.float64)
    inner_nested = (
        dense_baseline
        + (identity - dense_baseline @ dense_hessian)
        @ dense_schwarz
        @ (identity - dense_hessian @ dense_baseline)
    )
    balanced_six = (
        coarse_correction
        + (identity - coarse_correction @ dense_hessian)
        @ dense_baseline
        @ (identity - dense_hessian @ coarse_correction)
    )
    treatment = (
        coarse_correction
        + (identity - coarse_correction @ dense_hessian)
        @ inner_nested
        @ (identity - dense_hessian @ coarse_correction)
    )
    inner_scale = max(1.0, float(np.linalg.norm(inner_nested, ord=np.inf)))
    np.testing.assert_allclose(
        inner_nested,
        inner_nested.T,
        rtol=1.0e-9,
        atol=1.0e-11 * inner_scale,
    )
    inner_eigenvalues = np.linalg.eigvalsh(0.5 * (inner_nested + inner_nested.T))
    assert float(inner_eigenvalues[0]) > 1.0e-12 * inner_scale
    balanced_scale = max(1.0, float(np.linalg.norm(balanced_six, ord=np.inf)))
    np.testing.assert_allclose(
        balanced_six,
        balanced_six.T,
        rtol=1.0e-9,
        atol=1.0e-11 * balanced_scale,
    )
    balanced_eigenvalues = np.linalg.eigvalsh(0.5 * (balanced_six + balanced_six.T))
    assert float(balanced_eigenvalues[0]) > 1.0e-12 * balanced_scale
    treatment_scale = max(1.0, float(np.linalg.norm(treatment, ord=np.inf)))
    np.testing.assert_allclose(
        treatment,
        treatment.T,
        rtol=1.0e-9,
        atol=1.0e-11 * treatment_scale,
    )
    treatment_eigenvalues = np.linalg.eigvalsh(0.5 * (treatment + treatment.T))
    assert float(treatment_eigenvalues[0]) > 1.0e-12 * treatment_scale
    baseline_condition = float(np.linalg.cond(dense_baseline @ dense_hessian))
    balanced_condition = float(np.linalg.cond(balanced_six @ dense_hessian))
    treatment_condition = float(np.linalg.cond(treatment @ dense_hessian))
    condition_noise = 256.0 * np.finfo(np.float64).eps * max(balanced_condition, treatment_condition)
    assert treatment_condition + condition_noise < balanced_condition, (
        balanced_condition,
        treatment_condition,
        condition_noise,
    )
    print(
        "CONTACT_TET_SCHWARZ_PRECONDITIONER_METRICS",
        {
            "inner_symmetry_inf": float(np.linalg.norm(inner_nested - inner_nested.T, ord=np.inf)),
            "inner_min_eigenvalue": float(inner_eigenvalues[0]),
            "pnested_symmetry_inf": float(np.linalg.norm(treatment - treatment.T, ord=np.inf)),
            "pnested_min_eigenvalue": float(treatment_eigenvalues[0]),
            "baseline_condition": baseline_condition,
            "balanced_condition": balanced_condition,
            "treatment_condition": treatment_condition,
            "condition_noise": condition_noise,
        },
    )

    # The dense-H probe above overwrites one persistent HQ column. Restore the frozen
    # coarse data, then compare the source option-on action with the exact composition.
    coupler._prepare_rigid_fem_contact_patch_preconditioner()
    deterministic_residual = np.cos(np.arange(1, dimension + 1, dtype=np.float64) * 0.23)
    deterministic_p = np.sin(np.arange(1, dimension + 1, dtype=np.float64) * 0.41)
    coupler.batch_pcg_active.from_numpy(np.asarray([True], dtype=np.bool_))
    coupler.pcg_fem_state_v.r.from_numpy(
        deterministic_residual[: 3 * n_vertices].reshape(1, n_vertices, 3)
    )
    coupler.pcg_rigid_state_dof.r.from_numpy(deterministic_residual[3 * n_vertices :].reshape(1, n_dofs))
    coupler.pcg_fem_state_v.p.from_numpy(deterministic_p[: 3 * n_vertices].reshape(1, n_vertices, 3))
    coupler.pcg_rigid_state_dof.p.from_numpy(deterministic_p[3 * n_vertices :].reshape(1, n_dofs))
    coupler.rigid_fem_contact_patch_correction_load_fem.from_numpy(
        deterministic_residual[: 3 * n_vertices].reshape(1, n_vertices, 3)
    )
    coupler.rigid_fem_contact_patch_correction_load_rigid.from_numpy(
        deterministic_residual[3 * n_vertices :].reshape(1, n_dofs)
    )
    coupler._apply_rigid_fem_contact_tet_schwarz_local_preconditioner()
    local_z = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.z.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.z.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    expected_local_z = dense_schwarz @ deterministic_residual
    local_action_scale = max(1.0, float(np.linalg.norm(expected_local_z)))
    np.testing.assert_allclose(
        local_z,
        expected_local_z,
        rtol=1.0e-9,
        atol=1.0e-11 * local_action_scale,
    )
    coupler.rigid_fem_contact_patch_correction_load_fem.from_numpy(
        deterministic_residual[: 3 * n_vertices].reshape(1, n_vertices, 3)
    )
    coupler.rigid_fem_contact_patch_correction_load_rigid.from_numpy(
        deterministic_residual[3 * n_vertices :].reshape(1, n_dofs)
    )
    coupler._apply_rigid_fem_contact_tet_schwarz_nested_preconditioner()
    inner_z = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.z.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.z.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    expected_inner_z = inner_nested @ deterministic_residual
    inner_action_scale = max(1.0, float(np.linalg.norm(expected_inner_z)))
    np.testing.assert_allclose(
        inner_z,
        expected_inner_z,
        rtol=1.0e-9,
        atol=1.0e-11 * inner_action_scale,
    )
    restored_p = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.p.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.p.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    assert np.array_equal(restored_p, deterministic_p)
    coupler._apply_rigid_fem_contact_patch_correction()
    balanced_z = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.z.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.z.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    expected_z = treatment @ deterministic_residual
    action_scale = max(1.0, float(np.linalg.norm(expected_z)))
    np.testing.assert_allclose(balanced_z, expected_z, rtol=1.0e-9, atol=1.0e-11 * action_scale)
    restored_p = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.p.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.p.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    assert np.array_equal(restored_p, deterministic_p)
    print(
        "BALANCED_PATCH_ACTION_METRICS",
        {
            "pure_s_max_abs": float(np.max(np.abs(local_z - expected_local_z))),
            "inner_b_max_abs": float(np.max(np.abs(inner_z - expected_inner_z))),
            "outer_max_abs": float(np.max(np.abs(balanced_z - expected_z))),
            "scale": action_scale,
        },
    )

    coupler._rigid_fem_contact_patch_pcg_solve()

    right_hand_side = -np.concatenate(
        (
            np.asarray(coupler.fem_state_v.gradient.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.rigid_state_dof.gradient.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    direct = spsolve(csc_matrix(dense_hessian), right_hand_side)
    treatment_correction = np.concatenate(
        (
            np.asarray(coupler.pcg_fem_state_v.x.to_numpy(), dtype=np.float64).reshape(-1),
            np.asarray(coupler.pcg_rigid_state_dof.x.to_numpy(), dtype=np.float64).reshape(-1),
        )
    )
    np.testing.assert_allclose(treatment_correction, direct, rtol=2.0e-5, atol=2.0e-8)
    print(
        "BALANCED_PATCH_DIRECT_METRICS",
        {"error_2": float(np.linalg.norm(treatment_correction - direct))},
    )

    for budget in low_budgets:
        for name in ("fem_velocity", "rigid_velocity", "normal_impulse", "tangential_impulse"):
            high_value = np.asarray(balanced_high[name], dtype=np.float64)
            balanced_value = np.asarray(balanced_repeats_by_budget[budget][0][name], dtype=np.float64)
            balanced_repeat = np.asarray(balanced_repeats_by_budget[budget][1][name], dtype=np.float64)
            balanced_error = float(
                np.linalg.norm(balanced_value - high_value)
            )
            treatment_error = float(
                np.linalg.norm(np.asarray(treatment_by_budget[budget][name], dtype=np.float64) - high_value)
            )
            scaled_floor = 256.0 * np.finfo(np.float64).eps * max(
                1.0, float(np.linalg.norm(high_value))
            )
            repeat_delta = float(np.linalg.norm(balanced_value - balanced_repeat))
            noise = max(repeat_delta, scaled_floor)
            print(
                "BALANCED_PATCH_LOW_BUDGET_METRIC",
                {
                    "budget": budget,
                    "field": name,
                    "balanced_error": balanced_error,
                    "treatment_error": treatment_error,
                    "repeat_delta": repeat_delta,
                    "noise": noise,
                },
            )
            if name in ("fem_velocity", "rigid_velocity"):
                assert treatment_error + noise < balanced_error, (
                    budget,
                    name,
                    balanced_error,
                    treatment_error,
                    repeat_delta,
                    scaled_floor,
                    noise,
                )
            else:
                assert treatment_error <= balanced_error + noise, (
                    budget,
                    name,
                    balanced_error,
                    treatment_error,
                    repeat_delta,
                    scaled_floor,
                    noise,
                )

    for name in ("fem_velocity", "rigid_velocity", "normal_impulse", "tangential_impulse"):
        print(
            "BALANCED_PATCH_HIGH_SOLUTION_METRIC",
            {
                "field": name,
                "error_2": float(
                    np.linalg.norm(
                        np.asarray(treatment_high[name], dtype=np.float64)
                        - np.asarray(balanced_high[name], dtype=np.float64)
                    )
                ),
            },
        )
        np.testing.assert_allclose(treatment_high[name], balanced_high[name], rtol=2.0e-5, atol=2.0e-8)
    assert treatment_high["modes"] == balanced_high["modes"]
