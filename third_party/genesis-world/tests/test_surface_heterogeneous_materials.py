import numpy as np
import pytest

import genesis as gs
from genesis.utils.heterogeneous_materials import (
    SurfaceHeterogeneousMaterial,
    _surface_lame2d_from_e_nu,
    load_obj_triangle_faces,
    load_surface_heterogeneous_material,
    validate_surface_face_remap,
    validate_surface_mesh_material_contract,
)


_TWO_TRI_E_NU = np.array([[1.0e5, 0.25], [2.0e5, 0.40]], dtype=np.float64)
_TWO_TRI_DENSITY = np.array([1000.0, 2000.0], dtype=np.float64)
_TWO_TRI_THICKNESS = np.array([0.002, 0.003], dtype=np.float64)
_TWO_TRI_AREA_DENSITY = _TWO_TRI_DENSITY * _TWO_TRI_THICKNESS
_TWO_TRI_LABELS = np.array([3, 7], dtype=np.int32)


def _write_surface_material_npz(
    path,
    *,
    e_nu=_TWO_TRI_E_NU,
    density=_TWO_TRI_DENSITY,
    thickness=_TWO_TRI_THICKNESS,
    area_density=_TWO_TRI_AREA_DENSITY,
    labels=_TWO_TRI_LABELS,
    extra_payload=None,
):
    payload = {
        "tri_E_nu": np.asarray(e_nu),
        "tri_density": np.asarray(density),
        "tri_thickness": np.asarray(thickness),
        "tri_area_density": np.asarray(area_density),
    }
    if labels is not None:
        payload["tri_part_labels"] = np.asarray(labels)
    if extra_payload is not None:
        payload.update(extra_payload)
    np.savez(path, **payload)


def _expected_mu_lam(e_nu):
    young = e_nu[:, 0]
    nu = e_nu[:, 1]
    return _surface_lame2d_from_e_nu(young, nu)


def _write_two_triangle_obj(path):
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "f 1/10/20 3/30/40 2/50/60",
                "f -1 -2 -3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_load_surface_heterogeneous_material_accepts_hag4r_npz(tmp_path):
    material_path = tmp_path / "surface_material.npz"
    _write_surface_material_npz(material_path)

    data = load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)
    expected_mu, expected_lam = _expected_mu_lam(_TWO_TRI_E_NU)

    np.testing.assert_allclose(data.youngs_modulus, _TWO_TRI_E_NU[:, 0])
    np.testing.assert_allclose(data.poisson_ratio, _TWO_TRI_E_NU[:, 1])
    np.testing.assert_allclose(data.density, _TWO_TRI_DENSITY)
    np.testing.assert_allclose(data.thickness, _TWO_TRI_THICKNESS)
    np.testing.assert_allclose(data.area_density, _TWO_TRI_AREA_DENSITY)
    np.testing.assert_allclose(data.mu, expected_mu)
    np.testing.assert_allclose(data.lam, expected_lam)
    np.testing.assert_array_equal(data.labels, _TWO_TRI_LABELS)
    assert data.metadata["material_path"] == str(material_path)
    assert data.metadata["representation"] == "surface"
    assert data.metadata["primitive_kind"] == "triangle"
    assert data.metadata["row_count"] == 2
    assert data.metadata["e_nu_key"] == "tri_E_nu"
    assert data.metadata["density_key"] == "tri_density"
    assert data.metadata["thickness_key"] == "tri_thickness"
    assert data.metadata["area_density_key"] == "tri_area_density"
    assert data.metadata["labels_key"] == "tri_part_labels"
    assert data.metadata["has_labels"] is True
    assert data.metadata["youngs_modulus"]["max"] == pytest.approx(2.0e5)
    assert data.metadata["poisson_ratio"]["min"] == pytest.approx(0.25)
    assert data.metadata["density"]["mean"] == pytest.approx(1500.0)
    assert data.metadata["thickness"]["max"] == pytest.approx(0.003)
    assert data.metadata["area_density"]["mean"] == pytest.approx(4.0)
    assert data.metadata["mu"]["min"] == pytest.approx(expected_mu.min())
    assert data.metadata["lambda"]["max"] == pytest.approx(expected_lam.max())
    assert data.metadata["labels"] == {"min": 3, "max": 7, "count": 2}
    assert data.metadata["area_density_rtol"] == pytest.approx(1e-6)
    assert data.metadata["area_density_atol"] == pytest.approx(1e-9)


def test_surface_lame2d_matches_pyuipc_youngs_poisson():
    pytest.importorskip("uipc")
    from uipc.constitution import ElasticModuli2D

    e_nu = np.array([[1.0e5, 0.25], [2.0e5, 0.40]], dtype=np.float64)
    expected_mu, expected_lam = _expected_mu_lam(e_nu)

    for index, (youngs_modulus, poisson_ratio) in enumerate(e_nu):
        moduli = ElasticModuli2D.youngs_poisson(float(youngs_modulus), float(poisson_ratio))
        assert expected_mu[index] == pytest.approx(moduli.mu())
        assert expected_lam[index] == pytest.approx(moduli.lambda_())


def test_load_surface_heterogeneous_material_accepts_column_vectors(tmp_path):
    material_path = tmp_path / "surface_material_column_vectors.npz"
    _write_surface_material_npz(
        material_path,
        density=_TWO_TRI_DENSITY[:, None],
        thickness=_TWO_TRI_THICKNESS[:, None],
        area_density=_TWO_TRI_AREA_DENSITY[:, None],
    )

    data = load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)

    np.testing.assert_allclose(data.density, _TWO_TRI_DENSITY)
    np.testing.assert_allclose(data.thickness, _TWO_TRI_THICKNESS)
    np.testing.assert_allclose(data.area_density, _TWO_TRI_AREA_DENSITY)


def test_load_surface_heterogeneous_material_rejects_row_count_mismatch(tmp_path):
    material_path = tmp_path / "surface_material_bad_rows.npz"
    _write_surface_material_npz(material_path)

    with pytest.raises(gs.GenesisException, match="shape"):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=3)


def test_load_surface_heterogeneous_material_rejects_missing_surface_key_without_tet_fallback(tmp_path):
    material_path = tmp_path / "surface_material_missing_tri_density.npz"
    np.savez(
        material_path,
        tri_E_nu=_TWO_TRI_E_NU,
        tet_density=_TWO_TRI_DENSITY,
        tri_thickness=_TWO_TRI_THICKNESS,
        tri_area_density=_TWO_TRI_AREA_DENSITY,
    )

    with pytest.raises(gs.GenesisException, match="missing required key 'tri_density'"):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)


@pytest.mark.parametrize("labels", [_TWO_TRI_LABELS[:, None], np.array([3, 7, 9], dtype=np.int32)])
def test_load_surface_heterogeneous_material_rejects_bad_label_shape(tmp_path, labels):
    material_path = tmp_path / "surface_material_bad_labels.npz"
    _write_surface_material_npz(material_path, labels=labels)

    with pytest.raises(gs.GenesisException, match="tri_part_labels"):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)


@pytest.mark.parametrize("labels", [np.array([3.5, 7.0]), np.array([3.0, np.nan])])
def test_load_surface_heterogeneous_material_rejects_bad_label_values(tmp_path, labels):
    material_path = tmp_path / "surface_material_bad_label_values.npz"
    _write_surface_material_npz(material_path, labels=labels)

    with pytest.raises(gs.GenesisException, match="tri_part_labels"):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"e_nu": np.array([[np.nan, 0.25], [2.0e5, 0.40]], dtype=np.float64)}, "non-finite"),
        ({"density": np.array([1000.0, np.inf], dtype=np.float64)}, "non-finite"),
        ({"e_nu": np.array([[0.0, 0.25], [2.0e5, 0.40]], dtype=np.float64)}, "Young's modulus"),
        ({"e_nu": np.array([[1.0e5, 0.5], [2.0e5, 0.40]], dtype=np.float64)}, "Poisson"),
        ({"density": np.array([1000.0, -1.0], dtype=np.float64)}, "tri_density"),
        ({"thickness": np.array([0.002, 0.0], dtype=np.float64)}, "tri_thickness"),
        ({"area_density": np.array([2.0, -1.0], dtype=np.float64)}, "tri_area_density"),
    ],
)
def test_load_surface_heterogeneous_material_rejects_invalid_values(tmp_path, overrides, message):
    material_path = tmp_path / "surface_material_invalid_values.npz"
    _write_surface_material_npz(material_path, **overrides)

    with pytest.raises(gs.GenesisException, match=message):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)


def test_load_surface_heterogeneous_material_rejects_area_density_mismatch(tmp_path):
    material_path = tmp_path / "surface_material_bad_area_density.npz"
    bad_area_density = _TWO_TRI_AREA_DENSITY.copy()
    bad_area_density[1] += 1.0e-3
    _write_surface_material_npz(material_path, area_density=bad_area_density)

    with pytest.raises(gs.GenesisException, match="tri_area_density"):
        load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)


def test_load_obj_triangle_faces_preserves_source_face_order(tmp_path):
    mesh_path = tmp_path / "two_triangles.obj"
    _write_two_triangle_obj(mesh_path)

    faces = load_obj_triangle_faces(mesh_path)

    np.testing.assert_array_equal(faces, np.array([[0, 2, 1], [3, 2, 1]], dtype=faces.dtype))


def test_validate_surface_mesh_material_contract_accepts_matching_obj_and_npz(tmp_path):
    mesh_path = tmp_path / "two_triangles.obj"
    material_path = tmp_path / "surface_material.npz"
    _write_two_triangle_obj(mesh_path)
    _write_surface_material_npz(material_path)
    material_data = load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=2)

    metadata = validate_surface_mesh_material_contract(mesh_path, material_data)

    assert metadata == {
        "mesh_path": str(mesh_path),
        "triangle_count": 2,
        "face_order": "obj_source_order",
    }


def test_validate_surface_mesh_material_contract_rejects_row_count_mismatch(tmp_path):
    mesh_path = tmp_path / "two_triangles.obj"
    material_path = tmp_path / "surface_material_one_row.npz"
    _write_two_triangle_obj(mesh_path)
    _write_surface_material_npz(
        material_path,
        e_nu=_TWO_TRI_E_NU[:1],
        density=_TWO_TRI_DENSITY[:1],
        thickness=_TWO_TRI_THICKNESS[:1],
        area_density=_TWO_TRI_AREA_DENSITY[:1],
        labels=_TWO_TRI_LABELS[:1],
    )
    material_data = load_surface_heterogeneous_material(SurfaceHeterogeneousMaterial(file=material_path), triangle_count=1)

    with pytest.raises(gs.GenesisException, match="triangle count"):
        validate_surface_mesh_material_contract(mesh_path, material_data)


def test_load_obj_triangle_faces_rejects_non_triangle_face(tmp_path):
    mesh_path = tmp_path / "quad.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "f 1 2 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(gs.GenesisException, match="expected 3"):
        load_obj_triangle_faces(mesh_path)


def test_validate_surface_face_remap_accepts_permutation():
    remap = validate_surface_face_remap(np.array([1, 0]), source_triangle_count=2, backend_triangle_count=2)

    np.testing.assert_array_equal(remap, np.array([1, 0], dtype=remap.dtype))


@pytest.mark.parametrize("face_remap", [np.array([0, 0]), np.array([0, 2])])
def test_validate_surface_face_remap_rejects_duplicate_or_out_of_range_indices(face_remap):
    with pytest.raises(gs.GenesisException):
        validate_surface_face_remap(face_remap, source_triangle_count=2, backend_triangle_count=2)
