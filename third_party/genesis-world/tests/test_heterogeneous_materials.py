import igl
import numpy as np
import pytest

import genesis as gs
from genesis.utils.heterogeneous_materials import (
    HeterogeneousMaterial,
    load_heterogeneous_material,
)


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
_TWO_TET_E_NU = np.array([[1.0e5, 0.25], [2.0e5, 0.40]], dtype=np.float64)
_TWO_TET_DENSITY = np.array([1000.0, 2000.0], dtype=np.float64)


def _write_tet_mesh(path):
    igl.writeMESH(str(path), _TWO_TET_VERTS, _TWO_TETS, np.empty((0, 3), dtype=np.int64))


def _write_material_npz(path, *, e_nu=_TWO_TET_E_NU, density=_TWO_TET_DENSITY, labels=None):
    payload = {
        "tet_E_nu": np.asarray(e_nu),
        "tet_density": np.asarray(density),
    }
    if labels is not None:
        payload["tet_part_labels"] = np.asarray(labels)
    np.savez(path, **payload)


def _expected_mu_lam(e_nu):
    young = e_nu[:, 0]
    nu = e_nu[:, 1]
    mu = young / (2.0 * (1.0 + nu))
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def test_load_heterogeneous_material_accepts_hag4r_npz(tmp_path):
    material_path = tmp_path / "material.npz"
    labels = np.array([3, 7], dtype=np.int32)
    _write_material_npz(material_path, labels=labels)

    data = load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=2)
    expected_mu, expected_lam = _expected_mu_lam(_TWO_TET_E_NU)

    np.testing.assert_allclose(data.youngs_modulus, _TWO_TET_E_NU[:, 0])
    np.testing.assert_allclose(data.poisson_ratio, _TWO_TET_E_NU[:, 1])
    np.testing.assert_allclose(data.density, _TWO_TET_DENSITY)
    np.testing.assert_allclose(data.mu, expected_mu)
    np.testing.assert_allclose(data.lam, expected_lam)
    np.testing.assert_array_equal(data.labels, labels)
    assert data.metadata["row_count"] == 2
    assert data.metadata["has_labels"] is True
    assert data.metadata["youngs_modulus"]["max"] == pytest.approx(2.0e5)


def test_load_heterogeneous_material_accepts_column_density(tmp_path):
    material_path = tmp_path / "material_column_density.npz"
    _write_material_npz(material_path, density=_TWO_TET_DENSITY[:, None])

    data = load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=2)

    np.testing.assert_allclose(data.density, _TWO_TET_DENSITY)


def test_load_heterogeneous_material_rejects_row_count_mismatch(tmp_path):
    material_path = tmp_path / "material_bad_rows.npz"
    _write_material_npz(material_path)

    with pytest.raises(gs.GenesisException, match="shape"):
        load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=3)


def test_load_heterogeneous_material_rejects_missing_required_key(tmp_path):
    material_path = tmp_path / "material_missing_key.npz"
    np.savez(material_path, tet_E_nu=_TWO_TET_E_NU)

    with pytest.raises(gs.GenesisException, match="missing required key 'tet_density'"):
        load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=2)


def test_load_heterogeneous_material_rejects_bad_label_shape(tmp_path):
    material_path = tmp_path / "material_bad_labels.npz"
    _write_material_npz(material_path, labels=np.array([0, 1, 2], dtype=np.int32))

    with pytest.raises(gs.GenesisException, match="tet_part_labels"):
        load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=2)


@pytest.mark.parametrize(
    ("e_nu", "density", "message"),
    [
        (
            np.array([[np.nan, 0.25], [2.0e5, 0.40]], dtype=np.float64),
            _TWO_TET_DENSITY,
            "non-finite",
        ),
        (np.array([[0.0, 0.25], [2.0e5, 0.40]], dtype=np.float64), _TWO_TET_DENSITY, "Young's modulus"),
        (np.array([[1.0e5, 0.5], [2.0e5, 0.40]], dtype=np.float64), _TWO_TET_DENSITY, "Poisson"),
        (_TWO_TET_E_NU, np.array([1000.0, -1.0], dtype=np.float64), "positive"),
    ],
)
def test_load_heterogeneous_material_rejects_invalid_values(tmp_path, e_nu, density, message):
    material_path = tmp_path / "material_invalid_values.npz"
    _write_material_npz(material_path, e_nu=e_nu, density=density)

    with pytest.raises(gs.GenesisException, match=message):
        load_heterogeneous_material(HeterogeneousMaterial(file=material_path), tet_count=2)


def _build_two_tet_scene(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    material_path = tmp_path / "material.npz"
    _write_tet_mesh(mesh_path)
    _write_material_npz(material_path, labels=np.array([0, 1], dtype=np.int32))

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(
            heterogeneous=gs.materials.FEM.HeterogeneousMaterial(file=str(material_path)),
        ),
    )
    scene.build()
    return scene, entity


def test_heterogeneous_material_sets_solver_lame_fields(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    expected_mu, expected_lam = _expected_mu_lam(_TWO_TET_E_NU)

    elem_slice = slice(entity.el_start, entity.el_start + entity.n_elements)
    mu = scene.fem_solver.elements_i.mu.to_numpy()[elem_slice]
    lam = scene.fem_solver.elements_i.lam.to_numpy()[elem_slice]

    np.testing.assert_allclose(mu, expected_mu, rtol=1e-6)
    np.testing.assert_allclose(lam, expected_lam, rtol=1e-6)
    assert entity.heterogeneous_material_metadata["row_count"] == 2
    assert entity.heterogeneous_material_metadata["density"]["max"] == pytest.approx(2000.0)


def test_heterogeneous_density_sets_element_and_vertex_mass(tmp_path):
    scene, entity = _build_two_tet_scene(tmp_path)
    elem_slice = slice(entity.el_start, entity.el_start + entity.n_elements)
    vert_slice = slice(entity.v_start, entity.v_start + entity.n_vertices)

    volumes = scene.fem_solver.elements_i.V.to_numpy()[elem_slice]
    mass_scaled = scene.fem_solver.elements_i.mass_scaled.to_numpy()[elem_slice]
    vertex_mass = scene.fem_solver.elements_v_info.mass.to_numpy()[vert_slice]

    np.testing.assert_allclose(mass_scaled / scene.fem_solver.vol_scale, _TWO_TET_DENSITY * volumes, rtol=1e-6)

    expected_vertex_mass = np.zeros(entity.n_vertices)
    for tet, density, volume in zip(entity.elems, _TWO_TET_DENSITY, volumes, strict=True):
        expected_vertex_mass[tet] += 0.25 * density * volume
    np.testing.assert_allclose(vertex_mass, expected_vertex_mass, rtol=1e-6)


def test_scalar_fem_material_still_uses_scalar_solver_fields(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    _write_tet_mesh(mesh_path)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.25, rho=500.0),
    )
    scene.build()

    expected_mu, expected_lam = _expected_mu_lam(np.array([[1.0e5, 0.25], [1.0e5, 0.25]], dtype=np.float64))
    elem_slice = slice(entity.el_start, entity.el_start + entity.n_elements)
    mu = scene.fem_solver.elements_i.mu.to_numpy()[elem_slice]
    lam = scene.fem_solver.elements_i.lam.to_numpy()[elem_slice]
    mass_scaled = scene.fem_solver.elements_i.mass_scaled.to_numpy()[elem_slice]
    volumes = scene.fem_solver.elements_i.V.to_numpy()[elem_slice]

    np.testing.assert_allclose(mu, expected_mu, rtol=1e-6)
    np.testing.assert_allclose(lam, expected_lam, rtol=1e-6)
    np.testing.assert_allclose(mass_scaled / scene.fem_solver.vol_scale, 500.0 * volumes, rtol=1e-6)
    assert entity.heterogeneous_material_metadata is None
