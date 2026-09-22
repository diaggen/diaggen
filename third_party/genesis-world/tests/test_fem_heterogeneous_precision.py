import igl
import numpy as np
import pytest
import genesis as gs

pytestmark = pytest.mark.cache(False)
_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)
_TETS = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)
_E_NU = np.array(
    [[np.nextafter(100000.0, np.inf), 0.25000000000000006], [200000.00000000003, 0.4000000000000001]],
    dtype=np.float64,
)
_DENSITY = np.array([1000.0000000000001, 2000.0000000000002], dtype=np.float64)


def _build_two_tet_scene(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    material_path = tmp_path / "material.npz"
    igl.writeMESH(str(mesh_path), _VERTS, _TETS, np.empty((0, 3), dtype=np.int64))
    np.savez(material_path, tet_E_nu=_E_NU, tet_density=_DENSITY)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(substeps=2),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_qualification_safety_extrema=True,
        ),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(
            model="linear_corotated",
            heterogeneous=gs.materials.FEM.HeterogeneousMaterial(file=str(material_path)),
        ),
    )
    scene.build()
    return scene, entity


@pytest.mark.precision("64")
def test_heterogeneous_f64_material_fields_preserve_loaded_values(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    expected = entity._heterogeneous_material_np
    elem_slice = slice(entity.el_start, entity.el_start + entity.n_elements)

    stored_mu = scene.fem_solver.elements_i.mu.to_numpy()[elem_slice]
    stored_lam = scene.fem_solver.elements_i.lam.to_numpy()[elem_slice]

    assert stored_mu.dtype == np.float64
    assert stored_lam.dtype == np.float64
    np.testing.assert_array_equal(stored_mu, expected.mu)
    np.testing.assert_array_equal(stored_lam, expected.lam)


@pytest.mark.precision("64")
def test_completed_implicit_frames_preserve_active_tetrahedra_for_safety_extrema(tmp_path):
    scene, _entity = _build_two_tet_scene(tmp_path)

    scene.step()

    first = scene.fem_solver.get_completed_substep_safety_extrema(completed_frame=1)
    assert first is not None
    assert first.no_inversion
