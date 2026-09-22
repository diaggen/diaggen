import math

import igl
import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .utils import assert_allclose, get_hf_dataset


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

_HETEROGENEOUS_CUBE_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)
_HETEROGENEOUS_CUBE_TETS = np.array(
    [
        [3, 0, 2, 8],
        [2, 0, 1, 8],
        [5, 4, 6, 8],
        [6, 4, 7, 8],
        [1, 0, 5, 8],
        [5, 0, 4, 8],
        [7, 3, 6, 8],
        [6, 3, 2, 8],
        [4, 0, 7, 8],
        [7, 0, 3, 8],
        [2, 1, 6, 8],
        [6, 1, 5, 8],
    ],
    dtype=np.int64,
)


def _write_two_tet_mesh(path):
    igl.writeMESH(str(path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))


def _build_native_implicit_two_tet_scene(
    tmp_path,
    *,
    n_linesearch_iterations=0,
    n_envs=1,
    enable_rigid_mode_deflation=False,
    n_pcg_iterations=100,
    pcg_rtol=0.0,
    pcg_threshold=1.0e-6,
    gravity=(0.0, 0.0, 0.0),
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh_path = tmp_path / "two_tets.mesh"
    _write_two_tet_mesh(mesh_path)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.05,
            substeps=1,
            gravity=gravity,
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=True,
            n_newton_iterations=2,
            n_pcg_iterations=n_pcg_iterations,
            n_linesearch_iterations=n_linesearch_iterations,
            damping_alpha=0.0,
            damping_beta=0.0,
            pcg_rtol=pcg_rtol,
            pcg_threshold=pcg_threshold,
            enable_rigid_mode_deflation=enable_rigid_mode_deflation,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path), pos=(0.0, 0.0, 2.0)),
        material=gs.materials.FEM.Elastic(E=100.0, rho=100.0, model="linear_corotated"),
    )
    scene.build(n_envs=n_envs)
    return scene, entity


def _run_native_implicit_soft_pull(tmp_path, *, stiffness, n_linesearch_iterations=0, n_steps=4):
    scene, entity = _build_native_implicit_two_tet_scene(
        tmp_path,
        n_linesearch_iterations=n_linesearch_iterations,
    )
    vertex_idx = 1
    initial = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    target = initial + np.array([0.3, 0.0, 0.0], dtype=np.float64)
    target_tensor = torch.tensor(target[None], dtype=gs.tc_float, device=gs.device)
    entity.set_vertex_constraints([vertex_idx], target_tensor, is_soft_constraint=True, stiffness=stiffness)

    for _ in range(n_steps):
        scene.step(update_visualizer=False)

    final = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    return initial, target, final


def _native_constraint_slice(scene, entity):
    constraints = scene.fem_solver.vertex_constraints
    vertex_slice = slice(entity.v_start, entity.v_start + entity.n_vertices)
    return {
        "is_constrained": constraints.is_constrained.to_numpy()[vertex_slice, 0],
        "target_pos": constraints.target_pos.to_numpy()[vertex_slice, 0],
        "is_soft_constraint": constraints.is_soft_constraint.to_numpy()[vertex_slice, 0],
        "stiffness": constraints.stiffness.to_numpy()[vertex_slice, 0],
    }


def _native_implicit_pcg_metrics(scene):
    pcg_state = scene.fem_solver.pcg_state
    residual_squared = np.asarray(pcg_state.rTr.to_numpy(), dtype=np.float64)
    initial_residual_squared = np.asarray(pcg_state.rTr_initial.to_numpy(), dtype=np.float64)
    threshold = np.asarray(pcg_state.termination_threshold.to_numpy(), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative_residual_norm = np.sqrt(residual_squared / initial_residual_squared)
    return {
        "residual_squared": residual_squared,
        "initial_residual_squared": initial_residual_squared,
        "threshold": threshold,
        "relative_residual_norm": relative_residual_norm,
    }


def _write_heterogeneous_cube_mesh(path):
    igl.writeMESH(
        str(path),
        _HETEROGENEOUS_CUBE_VERTS,
        _HETEROGENEOUS_CUBE_TETS,
        np.empty((0, 3), dtype=np.int64),
    )


def _build_native_implicit_heterogeneous_cube_scene(tmp_path, *, enable_rigid_mode_deflation, n_pcg_iterations=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh_path = tmp_path / "heterogeneous_cube.mesh"
    _write_heterogeneous_cube_mesh(mesh_path)
    material_path = tmp_path / "heterogeneous_material.npz"
    np.savez(
        material_path,
        tet_E_nu=np.asarray([[1.0e2, 0.2]] * 6 + [[1.0e8, 0.2]] * 6, dtype=np.float64),
        tet_density=np.full(12, 100.0, dtype=np.float64),
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.05,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            enable_floor=False,
            use_implicit_solver=True,
            n_newton_iterations=1,
            n_pcg_iterations=n_pcg_iterations,
            n_linesearch_iterations=0,
            damping_alpha=0.0,
            damping_beta=0.0,
            enable_rigid_mode_deflation=enable_rigid_mode_deflation,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            fem_floor_contact_type="none",
            rigid_floor_contact_type="none",
            enable_rigid_fem_contact=False,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path), pos=(0.0, 0.0, 2.0)),
        material=gs.materials.FEM.Elastic(
            model="linear",
            heterogeneous=gs.materials.FEM.HeterogeneousMaterial(file=str(material_path)),
        ),
    )
    scene.build()
    return scene, entity


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_rigid_mode_deflation_defaults_off(tmp_path):
    options = gs.options.FEMOptions()
    assert options.enable_rigid_mode_deflation is False

    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    assert not hasattr(scene.fem_solver, "rigid_mode_coarse_matrix")
    scene.step(update_visualizer=False)
    assert np.isfinite(tensor_to_array(entity.get_state().pos)).all()


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_rigid_mode_deflation_converges_to_same_implicit_solution(tmp_path):
    scenes = []
    for enabled, directory in ((False, tmp_path / "baseline"), (True, tmp_path / "treatment")):
        scene, entity = _build_native_implicit_two_tet_scene(
            directory,
            enable_rigid_mode_deflation=enabled,
            n_pcg_iterations=120,
            pcg_rtol=1.0e-10,
            pcg_threshold=1.0e-20,
        )
        initial = tensor_to_array(entity.get_state().pos[0, 1]).copy()
        target = initial + np.asarray((0.3, 0.0, 0.0), dtype=np.float64)
        entity.set_vertex_constraints(
            [1],
            torch.as_tensor(target[None], dtype=gs.tc_float, device=gs.device),
            is_soft_constraint=True,
            stiffness=500.0,
        )
        scene.step(update_visualizer=False)
        scenes.append((scene, entity))

    baseline_state = scenes[0][1].get_state()
    treatment_state = scenes[1][1].get_state()
    np.testing.assert_allclose(
        tensor_to_array(treatment_state.pos), tensor_to_array(baseline_state.pos), rtol=1.0e-8, atol=2.0e-8
    )
    np.testing.assert_allclose(
        tensor_to_array(treatment_state.vel), tensor_to_array(baseline_state.vel), rtol=1.0e-8, atol=2.0e-8
    )
    for scene, _entity in scenes:
        metrics = _native_implicit_pcg_metrics(scene)
        assert all(
            residual <= threshold * 1.01
            for residual, threshold in zip(
                metrics["residual_squared"],
                metrics["threshold"],
            )
        )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_rigid_mode_deflation_improves_rigid_translation_at_fixed_budget(tmp_path):
    results = []
    for enabled, directory in ((False, tmp_path / "baseline"), (True, tmp_path / "treatment")):
        scene, entity = _build_native_implicit_heterogeneous_cube_scene(
            directory,
            enable_rigid_mode_deflation=enabled,
            n_pcg_iterations=4,
        )
        scene.step(update_visualizer=False)
        metrics = _native_implicit_pcg_metrics(scene)
        vertex_slice = slice(entity.v_start, entity.v_start + entity.n_vertices)
        velocities = tensor_to_array(entity.get_state().vel[0])
        masses = np.asarray(scene.fem_solver.elements_v_info.mass.to_numpy()[vertex_slice], dtype=np.float64)
        velocity = np.sum(velocities * masses[:, None], axis=0) / masses.sum()
        results.append((metrics, velocity))

    baseline_metrics, baseline_velocity = results[0]
    treatment_metrics, treatment_velocity = results[1]
    baseline_residual = baseline_metrics["relative_residual_norm"][0]
    treatment_residual = treatment_metrics["relative_residual_norm"][0]
    assert treatment_residual < 0.9 * baseline_residual
    expected_velocity = -9.81 * 0.05
    assert abs(treatment_velocity[2] - expected_velocity) < abs(baseline_velocity[2] - expected_velocity)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_rigid_mode_deflation_keeps_batch_coarse_solves_isolated(tmp_path):
    batched_scene, batched_entity = _build_native_implicit_two_tet_scene(
        tmp_path / "batched",
        n_envs=2,
        enable_rigid_mode_deflation=True,
        n_pcg_iterations=120,
        pcg_rtol=1.0e-10,
    )
    batched_initial = tensor_to_array(batched_entity.get_state().pos)
    batched_targets = np.stack(
        (batched_initial[0, 1] + np.asarray((0.25, 0.0, 0.0)),
         batched_initial[1, 1] + np.asarray((0.0, 0.25, 0.0)))
    )
    batched_entity.set_vertex_constraints(
        [1],
        torch.as_tensor(batched_targets, dtype=gs.tc_float, device=gs.device),
        is_soft_constraint=True,
        stiffness=500.0,
    )
    batched_scene.step(update_visualizer=False)
    batched_state = batched_entity.get_state()

    singles = []
    for env_index in range(2):
        scene, entity = _build_native_implicit_two_tet_scene(
            tmp_path / f"single_{env_index}",
            enable_rigid_mode_deflation=True,
            n_pcg_iterations=120,
            pcg_rtol=1.0e-10,
        )
        initial = tensor_to_array(entity.get_state().pos[0, 1]).copy()
        delta = np.asarray((0.25, 0.0, 0.0)) if env_index == 0 else np.asarray((0.0, 0.25, 0.0))
        entity.set_vertex_constraints(
            [1],
            torch.as_tensor((initial + delta)[None], dtype=gs.tc_float, device=gs.device),
            is_soft_constraint=True,
            stiffness=500.0,
        )
        scene.step(update_visualizer=False)
        singles.append(entity)

    for env_index, entity in enumerate(singles):
        np.testing.assert_allclose(
            tensor_to_array(batched_state.pos[env_index]),
            tensor_to_array(entity.get_state().pos[0]),
            rtol=1.0e-8,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            tensor_to_array(batched_state.vel[env_index]),
            tensor_to_array(entity.get_state().vel[0]),
            rtol=1.0e-8,
            atol=2.0e-8,
        )


@pytest.mark.required
def test_fem_partial_state_restore_only_mutates_selected_environment(tmp_path):
    """M3 replay must not restore one candidate's FEM state into every env."""
    scene, _ = _build_native_implicit_two_tet_scene(tmp_path, n_envs=2)
    snapshot = scene.capture_whole_batch_snapshot()
    state = scene.fem_solver.get_state(0)
    before = tensor_to_array(state.pos).copy()
    state.pos[0, 0, 0] += 0.125
    state.pos[1, 0, 0] += 0.375

    scene.fem_solver.set_state(0, state, envs_idx=[0])
    after = tensor_to_array(scene.fem_solver.get_state(0).pos)

    assert_allclose(after[0, 0, 0], before[0, 0, 0] + 0.125, tol=1e-12)
    assert_allclose(after[1], before[1], tol=1e-12)

    scene.restore_whole_batch_snapshot(snapshot)
    restored = tensor_to_array(scene.fem_solver.get_state(0).pos)
    assert_allclose(restored, before, tol=1e-12)


@pytest.mark.required
@pytest.mark.parametrize(("enable_floor", "expected_to_move_down"), ((True, False), (False, True)))
def test_legacy_fem_floor_can_be_disabled(enable_floor, expected_to_move_down, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1, gravity=(0.0, 0.0, 0.0)),
        fem_options=gs.options.FEMOptions(enable_floor=enable_floor),
        coupler_options=gs.options.LegacyCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    body = scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, -0.5)),
        material=gs.materials.FEM.Elastic(E=1.0e4, rho=1000.0),
    )
    scene.build()
    assert scene.fem_solver.enable_floor is enable_floor

    body.set_velocity((0.0, 0.0, -0.5))
    initial_z = float(body.get_state().pos[..., 2].mean())
    scene.step(update_visualizer=False)
    final_z = float(body.get_state().pos[..., 2].mean())

    if expected_to_move_down:
        assert final_z < initial_z - 1.0e-4
    else:
        assert final_z > initial_z - 1.0e-3


@pytest.mark.required
def test_interior_tetrahedralized_vertex(cube_verts_and_faces, box_obj_path, show_viewer):
    """
    Test tetrahedralization of a FEM entity with a small maxvolume value that introduces
    internal vertices during tetrahedralization:
      1. Verify all surface vertices lie exactly on the original quad faces of the mesh.
      2. Ensure the visualizer's mesh triangles match the FEM entity's surface triangles.
    """
    verts, faces = cube_verts_and_faces

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            minratio=1.5,
            verbose=1,
            maxvolume=0.01,
        ),
        material=gs.materials.FEM.Muscle(),
    )
    scene.build()

    state = fem.get_state()
    vertices = tensor_to_array(state.pos[0])
    surface_indices = np.unique(fem.surface_triangles)

    # Ensure there are interior vertices; this is a prerequisite for this test
    assert surface_indices.size < vertices.shape[0]

    # Verify each surface vertex lies on the original surface mesh
    def _point_on_surface(p, verts, faces, tol=1e-6):
        """Check if point p lies on any of the quad faces (as two triangles)."""
        for face in faces:
            # Convert 1-based face indices to 0-based
            idx = [i - 1 for i in face]
            # Extract vertices
            v0, v1, v2, v3 = [np.array(verts[i]) for i in idx]
            # Decompose quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
            for tri in ((v0, v1, v2), (v0, v2, v3)):
                a, b, c = tri
                # Compute normal for plane
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < tol:
                    continue
                # Check distance to plane
                distance = abs(np.dot(n / norm_n, p - a))
                if distance > tol:
                    continue
                # Barycentric coordinates
                v0v1 = b - a
                v0v2 = c - a
                v0p = p - a
                dot00 = np.dot(v0v2, v0v2)
                dot01 = np.dot(v0v2, v0v1)
                dot02 = np.dot(v0v2, v0p)
                dot11 = np.dot(v0v1, v0v1)
                dot12 = np.dot(v0v1, v0p)
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < tol:
                    continue
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                if u >= -tol and v >= -tol and (u + v) <= 1 + tol:
                    return True
        return False

    for idx in surface_indices:
        p = vertices[idx]
        assert _point_on_surface(p, verts, faces), (
            f"Surface vertex index {idx} with coordinate {p} does not lie on any original face"
        )

    # Verify whether surface faces in the visualizer mesh matches the surface faces of the FEM entity
    static_nodes = scene.visualizer.context.static_nodes
    fem_node_mesh = static_nodes[(0, fem.uid)].mesh

    (fem_node_primitive,) = fem_node_mesh.primitives
    fem_node_vertices = fem_node_primitive.positions
    fem_node_faces = fem_node_primitive.indices
    if fem_node_faces is None:
        fem_node_faces = np.arange(fem_node_vertices.shape[0]).reshape(-1, 3)

    def _make_triangle_set(verts, faces, tol=4):
        """
        Return a hashable, order-independent representation of a given set of triangle faces.

        Rounds each vertex coordinate to the given tolerance, sorts vertices within each triangle,
        and returns all triangles as a sorted tuple, eliminating any dependence on vertex or face order.
        """
        tri_set = set()
        for tri in faces:
            coords = [tuple(round(float(coord), tol) for coord in verts[i]) for i in tri]
            tri_set.add(tuple(sorted(coords)))
        return tuple(sorted(tri_set))

    # Triangles of FEM entity
    entity_tris = _make_triangle_set(vertices, fem.surface_triangles)

    # Triangles of visualizer
    viz_tris = _make_triangle_set(np.asarray(fem_node_vertices), np.asarray(fem_node_faces))

    assert entity_tris == viz_tris, (
        "FEM entity surface triangles and visualizer mesh triangles do not match.\n"
        f"Differences: {set(entity_tris) ^ set(viz_tris)}"
    )


@pytest.mark.required
@pytest.mark.parametrize("smooth", [True, False])
def test_fem_rasterizer_update_uses_surface_vertex_map(box_obj_path, show_viewer, monkeypatch, smooth):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            minratio=1.5,
            verbose=1,
            maxvolume=0.01,
        ),
        material=gs.materials.FEM.Muscle(),
        surface=gs.surfaces.Default(smooth=smooth),
    )
    scene.build()

    surf_idx = np.unique(fem.surface_triangles)
    assert surf_idx.size == fem.n_surface_vertices
    assert surf_idx.size < fem.n_vertices
    assert not np.array_equal(surf_idx, np.arange(surf_idx.size))

    context = scene.visualizer.context
    node_key = (0, fem.uid)
    np.testing.assert_array_equal(context._fem_surface_vertex_indices[node_key], surf_idx)

    base_pos = tensor_to_array(fem.get_state().pos[0])
    offsets = np.zeros_like(base_pos)
    offsets[:, 0] = np.arange(fem.n_vertices) * 1e-4
    offsets[:, 1] = (np.arange(fem.n_vertices) % 11) * 2e-4
    fem.set_position(base_pos + offsets)

    captured = []
    monkeypatch.setattr(context.jit, "update_normal", lambda node, update_data: None)
    monkeypatch.setattr(
        context.jit,
        "update_buffer",
        lambda buffer_id, data: captured.append(np.array(data, copy=True)),
    )
    context.update_fem()

    vertices_qd, _, _ = scene.sim.fem_solver.get_state_render(scene.sim.cur_substep_local)
    vertices_all = vertices_qd.to_numpy(dtype=gs.np_float)
    expected_compact = vertices_all[fem.v_start : fem.v_start + fem.n_vertices, 0][surf_idx]
    node = context.static_nodes[node_key]
    expected_uploaded = context._scene.reorder_vertices(node, expected_compact)

    assert captured
    assert captured[0].shape == node.mesh.primitives[0].positions.shape
    assert_allclose(captured[0], expected_uploaded, tol=1e-6)

    context.destroy()
    assert context._fem_surface_vertex_indices == {}


@pytest.mark.required
def test_maxvolume(box_obj_path, show_viewer):
    """Test that imposing a maximum element volume constraint produces a finer mesh (i.e., more elements)."""
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    fem1 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            verbose=1,
        ),
        material=gs.materials.FEM.Muscle(),
    )

    # Mesh with maximum element volume limited to 0.01
    fem2 = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            maxvolume=0.01,
            verbose=1,
        ),
        material=gs.materials.FEM.Muscle(),
    )

    assert len(fem1.elems) < len(fem2.elems), (
        f"Mesh with maxvolume=0.01 generated {len(fem2.elems)} elements; "
        f"expected more than {len(fem1.elems)} elements without a volume limit."
    )


@pytest.mark.required
def test_offset_pos(box_obj_path, show_viewer):
    POS = (0.2, -0.1, 0.3)
    OFFSET_POS = (0.05, 0.0, 0.1)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            pos=POS,
            offset_pos=OFFSET_POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )
    box_no_offset = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            pos=POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )
    scene.build()

    offset = box.get_state().pos - box_no_offset.get_state().pos
    assert_allclose(offset, OFFSET_POS, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize(
    "coupler_type, material_model",
    [
        (gs.options.SAPCouplerOptions, "linear"),
        (gs.options.SAPCouplerOptions, "linear_corotated"),
        (gs.options.LegacyCouplerOptions, "linear"),
    ],
)
def test_implicit_falling_sphere_box(coupler_type, material_model, show_viewer):
    SPHERE_POS = (0.4, -0.1, 0.1)
    SPHERE_RADIUS = 0.1
    SPHERE_RHO = 500.0
    BOX_POS = (0.0, 0.1, 0.3)
    BOX_SIZE = 0.05

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=3 if coupler_type == gs.options.SAPCouplerOptions else 2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=coupler_type(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, 0.0, 0.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.4, -0.1, 0.10),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            rho=SPHERE_RHO,
            model=material_model,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE * 2, BOX_SIZE * 2, BOX_SIZE * 2),
            pos=BOX_POS,
        ),
        material=gs.materials.FEM.Elastic(),
    )

    # Build the scene
    scene.build()

    sphere_mass_ref = (4 / 3 * np.pi * SPHERE_RADIUS**3) * SPHERE_RHO
    elems_mass_scaled = scene.fem_solver.elements_i.mass_scaled.to_numpy()
    sphere_elems_mass_scaled = elems_mass_scaled[sphere.el_start : (sphere.el_start + sphere.n_elements)]
    sphere_mass_1 = sphere_elems_mass_scaled.sum() / scene.fem_solver.vol_scale
    verts_mass = scene.fem_solver.elements_v_info.mass.to_numpy()
    sphere_verts_mass = verts_mass[sphere.v_start : (sphere.v_start + sphere.n_vertices)]
    sphere_mass_2 = sphere_verts_mass.sum()
    assert_allclose(sphere_mass_1, sphere_mass_2, tol=gs.EPS)
    assert_allclose(sphere_mass_ref, sphere_mass_1, rtol=0.01)  # Large tolerance due to tessellation

    # Run simulation
    n_steps = 40 if coupler_type == gs.options.SAPCouplerOptions else 150
    for _ in range(n_steps):
        scene.step()

    for entity, init_pos, entity_halfsize_ref in zip(scene.entities, (SPHERE_POS, BOX_POS), (SPHERE_RADIUS, BOX_SIZE)):
        # Not moving anymore
        state = entity.get_state()
        assert_allclose(state.vel, 0.0, tol=0.025)

        # Landed vertically
        pos = tensor_to_array(state.pos[0])
        BV, *_ = igl.bounding_box(pos)
        entity_center = 0.5 * (BV[0] + BV[-1])
        if coupler_type == gs.options.SAPCouplerOptions:
            tol = 1e-2 if material_model == "linear_corotated" else 1e-3
        else:
            tol = 5e-3
        assert_allclose(entity_center[:2], init_pos[:2], tol=tol)

        # Reasonable deformation if possible
        # FIXME: Compute theoretical deformation to be able to reduce the tolerance
        if coupler_type == gs.options.SAPCouplerOptions:
            entity_halfsize = 0.5 * (BV[0] - BV[-1])
            assert_allclose(entity_halfsize, entity_halfsize_ref, tol=5e-3)

        # Reasonable penetration depth
        # FIXME: Compute theoretical penetration depth to be able to reduce the tolerance
        penetration_depth_ref = 0.0
        tol = 1e-3 if coupler_type == gs.options.SAPCouplerOptions else 0.05
        assert_allclose(-state.pos[..., 2].min(), penetration_depth_ref, tol=tol)


# This test cannot be flagged as required because it takes 250s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_sap_coupler_collide_sphere_box(show_viewer):
    SPHERE_RADIUS = 0.1
    BOX_SIZE = 0.015

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.45),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, SPHERE_RADIUS),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            friction_mu=1.0,
            model="linear_corotated",
        ),
    )
    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            pos=(0.0, 0.0, 2 * SPHERE_RADIUS + BOX_SIZE),
            scale=BOX_SIZE,
        ),
        material=gs.materials.FEM.Elastic(
            E=1e4,
            rho=50.0,
            friction_mu=1.0,
            model="linear_corotated",
        ),
    )
    scene.build()

    # Run simulation
    for _ in range(40):
        scene.step()

    for entity, init_height in zip(scene.entities, (SPHERE_RADIUS, 2 * SPHERE_RADIUS + BOX_SIZE)):
        # Barely moving
        state = entity.get_state()
        assert_allclose(state.vel, 0.0, tol=0.05)

        # More or less at the initial position
        pos = tensor_to_array(state.pos[0])
        BV, *_ = igl.bounding_box(pos)
        entity_center = 0.5 * (BV[0] + BV[-1])
        assert_allclose(entity_center[:2], 0.0, tol=0.02)
        assert_allclose(entity_center[2], init_height, tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
@pytest.mark.parametrize("invalid_stiffness", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_native_implicit_soft_constraint_rejects_invalid_stiffness(tmp_path, invalid_stiffness):
    _scene, entity = _build_native_implicit_two_tet_scene(tmp_path)

    with pytest.raises(gs.GenesisException, match="finite stiffness > 0.0"):
        entity.set_vertex_constraints([0], is_soft_constraint=True, stiffness=invalid_stiffness)

    entity.set_vertex_constraints([0])


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_vertex_constraint_public_api_hard_default_stores_state(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    initial = tensor_to_array(entity.get_state().pos[0, 0]).copy()

    entity.set_vertex_constraints([0])

    constraints = _native_constraint_slice(scene, entity)
    assert constraints["is_constrained"][0]
    assert not constraints["is_soft_constraint"][0]
    assert constraints["stiffness"][0] == pytest.approx(0.0)
    np.testing.assert_allclose(constraints["target_pos"][0], initial, atol=1e-6)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_public_api_stores_state(tmp_path):
    from genesis.engine.couplers import IPCCoupler

    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    initial = tensor_to_array(entity.get_state().pos[0, 1]).copy()
    target = initial + np.array([0.2, 0.1, 0.0], dtype=np.float64)
    target_tensor = torch.tensor(target[None], dtype=gs.tc_float, device=gs.device)

    entity.set_vertex_constraints([1], target_tensor, is_soft_constraint=True, stiffness=500.0)

    constraints = _native_constraint_slice(scene, entity)
    assert constraints["is_constrained"][1]
    assert constraints["is_soft_constraint"][1]
    assert constraints["stiffness"][1] == pytest.approx(500.0)
    np.testing.assert_allclose(constraints["target_pos"][1], target, atol=1e-6)
    assert not isinstance(scene.sim.coupler, IPCCoupler)


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_update_target_retargets_energy(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    vertex_idx = 1
    initial = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    first_target = initial + np.array([0.2, 0.0, 0.0], dtype=np.float64)
    entity.set_vertex_constraints(
        [vertex_idx],
        torch.tensor(first_target[None], dtype=gs.tc_float, device=gs.device),
        is_soft_constraint=True,
        stiffness=500.0,
    )

    for _ in range(2):
        scene.step(update_visualizer=False)

    position_at_update = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    new_target = position_at_update + np.array([0.0, 0.3, 0.0], dtype=np.float64)
    entity.update_constraint_targets(
        [vertex_idx],
        torch.tensor(new_target[None], dtype=gs.tc_float, device=gs.device),
    )
    np.testing.assert_allclose(_native_constraint_slice(scene, entity)["target_pos"][vertex_idx], new_target, atol=1e-6)

    for _ in range(4):
        scene.step(update_visualizer=False)

    final = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    assert np.linalg.norm(final - new_target) < np.linalg.norm(position_at_update - new_target)
    assert np.dot(final - position_at_update, new_target - position_at_update) > 0.0


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_remove_disables_energy(tmp_path):
    scene, entity = _build_native_implicit_two_tet_scene(tmp_path)
    vertex_idx = 1
    initial = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    stale_target = initial + np.array([0.5, 0.0, 0.0], dtype=np.float64)
    entity.set_vertex_constraints(
        [vertex_idx],
        torch.tensor(stale_target[None], dtype=gs.tc_float, device=gs.device),
        is_soft_constraint=True,
        stiffness=1000.0,
    )

    entity.remove_vertex_constraints([vertex_idx])
    constraints = _native_constraint_slice(scene, entity)
    assert not constraints["is_constrained"][vertex_idx]

    for _ in range(4):
        scene.step(update_visualizer=False)

    final = tensor_to_array(entity.get_state().pos[0, vertex_idx]).copy()
    assert np.linalg.norm(final - initial) < 1e-6
    assert np.linalg.norm(final - stale_target) >= np.linalg.norm(initial - stale_target) - 1e-6


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_moves_toward_target(tmp_path):
    initial, target, final = _run_native_implicit_soft_pull(tmp_path, stiffness=500.0)

    initial_error = np.linalg.norm(initial - target)
    final_error = np.linalg.norm(final - target)
    assert final_error < initial_error
    assert np.dot(final - initial, target - initial) > 0.0


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_stiffness_monotonicity(tmp_path):
    _initial_low, target_low, final_low = _run_native_implicit_soft_pull(tmp_path / "low", stiffness=50.0)
    _initial_high, target_high, final_high = _run_native_implicit_soft_pull(tmp_path / "high", stiffness=1000.0)

    low_error = np.linalg.norm(final_low - target_low)
    high_error = np.linalg.norm(final_high - target_high)
    assert high_error < low_error


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_native_implicit_soft_constraint_with_linesearch(tmp_path):
    initial, target, final = _run_native_implicit_soft_pull(
        tmp_path,
        stiffness=500.0,
        n_linesearch_iterations=5,
    )

    assert np.linalg.norm(final - target) < np.linalg.norm(initial - target)
    assert np.dot(final - initial, target - initial) > 0.0


@pytest.mark.required
@pytest.mark.xfail(raises=AssertionError, reason="Constraint dynamics inconsistent with analytical formula")
@pytest.mark.parametrize("precision", ["64"])
def test_explicit_legacy_coupler_soft_constraint_box(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    DT = 0.01
    BOX_SIZE = 0.1
    CONSTRAINT_STIFFNESS = 1e1
    BOX_VELOCITY = torch.tensor([0.2, 0.0, 0.0])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.Elastic(
            rho=1.0 / BOX_SIZE**3,  # Unit mass
        ),
        surface=gs.surfaces.Default(
            color=(1, 1, 1, 0.5),
        ),
    )
    scene.build()

    verts_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    target_poss = box.init_positions[verts_idx]
    box.set_vertex_constraints(verts_idx, target_poss, is_soft_constraint=True, stiffness=CONSTRAINT_STIFFNESS)
    if show_viewer:
        scene.draw_debug_spheres(poss=target_poss, radius=0.01, color=(1, 0, 1, 1))

    # Initialize box velocity to non-zero value
    box.set_velocity(BOX_VELOCITY)

    # Check that the box has a spring dynamics
    omega = math.sqrt(8 * CONSTRAINT_STIFFNESS)
    for i in range(2000):
        pos = box.get_state().pos[..., :8, :].sum(dim=-2)
        pos_ref = BOX_VELOCITY * (i * DT) * math.exp(-omega * (i * DT))
        assert_allclose(pos, pos_ref, tol=1e-3)
        scene.step()


@pytest.mark.required
@pytest.mark.parametrize("use_implicit_solver", [False, True])
@pytest.mark.parametrize("precision", ["64"])
def test_hard_constraint(use_implicit_solver, show_viewer):
    DT = 0.01
    HEIGHT = 2.0  # It must be height enough to avoid hitting the ground when dropping the box at the end
    BOX_SIZE = 0.1
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [5, 7]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2 if use_implicit_solver else 100,
            gravity=(0.0, 0.0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=use_implicit_solver,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, HEIGHT + 0.5),
            camera_lookat=(0.0, 0.0, HEIGHT),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-BOX_SIZE / 2, BOX_SIZE / 2 + MOTION_RADIUS, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated" if use_implicit_solver else "stable_neohookean",
        ),
    )
    scene.build(n_envs=2)

    # Simulate
    n_steps = int(0.5 * math.pi / MOTION_SPEED)
    for it in range(n_steps):
        # Update the position of the fixed vertices
        target_poss = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
        target_poss[0, 0] = MOTION_RADIUS * math.sin(MOTION_SPEED * it)
        target_poss[0, 1] = MOTION_RADIUS * math.cos(MOTION_SPEED * it)
        target_poss[0, 2] = HEIGHT + BOX_SIZE
        target_poss[1, 0] = (MOTION_RADIUS + BOX_SIZE) * math.sin(MOTION_SPEED * it)
        target_poss[1, 1] = (MOTION_RADIUS + BOX_SIZE) * math.cos(MOTION_SPEED * it)
        target_poss[1, 2] = HEIGHT + BOX_SIZE
        box.set_vertex_constraints(VERTICES_IDX, target_poss)

        # Do one simulation step
        scene.step(update_visualizer=False)

        # Check that the constrained vertices are at their respective target positions
        corners = box.get_state().pos[..., :8, :]
        assert_allclose(corners[..., VERTICES_IDX, :], target_poss, tol=1e-8)

        # Check that the box is not more or less a box
        e_z = corners[..., 5, :] - corners[..., 4, :]
        e_z /= torch.linalg.norm(e_z, dim=-1, keepdim=True)
        e_y = corners[..., 6, :] - corners[..., 4, :]
        e_y /= torch.linalg.norm(e_y, dim=-1, keepdim=True)
        e_x = torch.cross(e_y, e_z, dim=-1)
        e_x /= torch.linalg.norm(e_x, dim=-1, keepdim=True)
        R = torch.stack((e_x, e_y, e_z), dim=-1)
        corners_local = (corners - corners[..., [0], :]) @ R + box.init_positions[0]
        assert_allclose(corners_local, box.init_positions[:8], tol=0.01)

        # Update the viewer if requested
        if show_viewer:
            if it % max(int(1e-3 / (MOTION_SPEED * DT)), 1) == 0:
                scene.visualizer.context.draw_debug_spheres(
                    poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8), persistent=True
                )
            scene.visualizer.update(force=False, auto=True)

    # Disable constraints
    box.remove_vertex_constraints()

    # Check that the box has been free-falling
    n_steps = 50
    com_pos_z_0 = box.get_state().pos[..., 8, 2]
    for _ in range(n_steps):
        scene.step()
    com_pos_z_f = box.get_state().pos[..., 8, 2]
    com_pos_delta = -0.5 * 9.81 * (n_steps * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_z_0, com_pos_delta, tol=0.05)


# This test cannot be flagged as required because it takes 400s to run on CPU.
# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_sap_coupler_hard_constraint_and_collision(show_viewer):
    DT = 0.01
    HEIGHT = 2.0  # It must be height enough to avoid hitting the ground when dropping the box at the end
    BOX_SIZE = 0.1
    SPHERE_RADIUS = 0.03
    MOTION_RADIUS = 0.1
    MOTION_SPEED = 0.005
    VERTICES_IDX = [4, 6]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=2,
            gravity=(0.0, 0.0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, 0.6, HEIGHT + 0.5),
            camera_lookat=(0.0, 0.0, HEIGHT),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-BOX_SIZE / 2, BOX_SIZE / 2 + MOTION_RADIUS, HEIGHT + BOX_SIZE / 2),
            size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
        ),
        material=gs.materials.FEM.Elastic(
            E=1e5,
            nu=0.45,
            rho=1000.0,
            model="linear_corotated",
        ),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(MOTION_RADIUS + BOX_SIZE, -SPHERE_RADIUS, HEIGHT - BOX_SIZE / 2),
            radius=SPHERE_RADIUS,
        ),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
        ),
    )
    scene.build()

    # Attach the sphere to its center
    sphere_poss = sphere.get_state().pos[0]
    sphere_poss -= torch.tensor(sphere.morph.pos)
    sphere_center_idx = int(torch.argmin(torch.linalg.norm(sphere_poss, dim=-1)))
    sphere_target_poss = sphere.init_positions[sphere_center_idx]
    sphere.set_vertex_constraints(sphere_center_idx, sphere_target_poss)

    # Simulate
    n_steps = int(0.5 * math.pi / MOTION_SPEED)
    box_pos_y_min = torch.tensor(float("inf"), dtype=gs.tc_float, device=gs.device)
    for it in range(n_steps):
        # Update the position of the fixed vertices
        target_poss = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
        target_poss[0, 0] = MOTION_RADIUS * math.sin(MOTION_SPEED * it)
        target_poss[0, 1] = MOTION_RADIUS * math.cos(MOTION_SPEED * it)
        target_poss[0, 2] = HEIGHT
        target_poss[1, 0] = (MOTION_RADIUS + BOX_SIZE) * math.sin(MOTION_SPEED * it)
        target_poss[1, 1] = (MOTION_RADIUS + BOX_SIZE) * math.cos(MOTION_SPEED * it)
        target_poss[1, 2] = HEIGHT
        box.set_vertex_constraints(VERTICES_IDX, target_poss)

        # Do one simulation step
        scene.step(update_visualizer=False)

        # Check that the constrained vertices are at their respective target positions
        corners = box.get_state().pos[..., :8, :]
        assert_allclose(corners[..., VERTICES_IDX, :], target_poss, tol=1e-8)
        sphere_center = sphere.get_state().pos[..., sphere_center_idx, :]
        assert_allclose(sphere_center, sphere_target_poss, tol=1e-8)

        # Check that the box is more or less a box
        e_z = corners[..., 5, :] - corners[..., 4, :]
        e_z /= torch.linalg.norm(e_z, dim=-1, keepdim=True)
        e_y = corners[..., 6, :] - corners[..., 4, :]
        e_y /= torch.linalg.norm(e_y, dim=-1, keepdim=True)
        e_x = torch.cross(e_y, e_z, dim=-1)
        e_x /= torch.linalg.norm(e_x, dim=-1, keepdim=True)
        R = torch.stack((e_x, e_y, e_z), dim=-1)
        corners_local = (corners - corners[..., [0], :]) @ R + box.init_positions[0]
        assert_allclose(corners_local, box.init_positions[:8], tol=0.02)

        # Check that the sphere is more or less a sphere
        sphere_poss = sphere.get_state().pos
        sphere_poss -= torch.tensor(sphere.morph.pos)
        sphere_dist_max = torch.linalg.norm(sphere_poss, dim=-1).max(dim=-1).values
        assert_allclose(sphere_dist_max, SPHERE_RADIUS, tol=0.01)

        # Check that the box is not going to far along y-axis due to collision with the sphere
        box_pos_y_min = torch.minimum(corners[..., [2, 3, 7], 1].min(dim=-1).values, box_pos_y_min)
        assert (box_pos_y_min > -0.05).all()

        # Update the viewer if requested
        if show_viewer:
            if it % max(int(1e-3 / (MOTION_SPEED * DT)), 1) == 0:
                scene.visualizer.context.draw_debug_spheres(
                    poss=target_poss, radius=0.005, color=(1, 0, 1, 0.8), persistent=True
                )
            scene.visualizer.update(force=False, auto=True)

    # Wait for a few extra steps
    for _ in range(10):
        scene.step()

    # Disable box constraints only
    box.remove_vertex_constraints(VERTICES_IDX)

    # Simulate for a while
    n_steps = 40
    com_pos_z_0 = box.get_state().pos[..., 8, 2]
    for _ in range(n_steps):
        # Do one simulation step
        scene.step()

        # Check that the box is not moving much further along y-axis despite removing constraints
        box_pos = box.get_state().pos
        assert (box_pos[..., 1].min(dim=-1).values > box_pos_y_min - 0.01).all()

    # Check that the box has been free-falling, but not the sphere
    com_pos_z_f = box_pos[..., 8, 2]
    com_pos_delta = -0.5 * 9.81 * (n_steps * DT) ** 2
    assert_allclose(com_pos_z_f - com_pos_z_0, com_pos_delta, tol=0.05)
    sphere_center = sphere.get_state().pos[..., sphere_center_idx, :]
    assert_allclose(sphere_center, sphere_target_poss, tol=1e-8)
