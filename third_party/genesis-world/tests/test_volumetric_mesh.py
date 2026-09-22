import igl
import numpy as np
import pytest

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.mesh as mu
from genesis.utils.volumetric_mesh import load_tet_mesh


_ONE_TET_VERTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_ONE_TET = np.array([[0, 1, 2, 3]], dtype=np.int64)
_ONE_TET_FACES = np.array([[0, 2, 1], [1, 2, 3], [0, 1, 3], [0, 3, 2]], dtype=np.int64)


def _write_mesh(path, verts, tets, faces=None):
    igl.writeMESH(
        str(path),
        np.asarray(verts, dtype=np.float64),
        np.asarray(tets, dtype=np.int64),
        np.asarray(faces if faces is not None else np.empty((0, 3), dtype=np.int64), dtype=np.int64),
    )


def _write_one_tet_mesh(path):
    _write_mesh(path, _ONE_TET_VERTS, _ONE_TET, _ONE_TET_FACES)


def test_load_tet_mesh_minimal_fixture(tmp_path):
    mesh_path = tmp_path / "one_tet.mesh"
    _write_one_tet_mesh(mesh_path)

    data = load_tet_mesh(mesh_path)

    np.testing.assert_allclose(data.verts, _ONE_TET_VERTS.astype(gs.np_float))
    np.testing.assert_array_equal(data.tets, _ONE_TET.astype(gs.np_int))
    np.testing.assert_array_equal(data.surface_triangles, _ONE_TET_FACES.astype(gs.np_int))
    np.testing.assert_array_equal(data.surface_triangle_tet_indices, np.zeros(4, dtype=gs.np_int))
    np.testing.assert_array_equal(data.boundary_vertex_indices, np.array([0, 1, 2, 3], dtype=gs.np_int))
    assert data.metadata["vertex_count"] == 4
    assert data.metadata["tet_count"] == 1
    assert data.metadata["surface_triangle_count"] == 4
    assert data.metadata["surface_source"] == "file"
    assert data.metadata["index_base"] == 0
    assert data.metadata["ignored_sections"] == []


def test_load_tet_mesh_derives_boundary_surface_when_faces_missing(tmp_path):
    mesh_path = tmp_path / "two_tets.mesh"
    verts = np.concatenate([_ONE_TET_VERTS, np.array([[0.0, 0.0, -1.0]])], axis=0)
    tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)
    _write_mesh(mesh_path, verts, tets)

    data = load_tet_mesh(mesh_path)
    sorted_surface = {tuple(row.tolist()) for row in np.sort(data.surface_triangles, axis=1)}

    assert data.surface_triangles.shape == (6, 3)
    assert (0, 1, 2) not in sorted_surface
    assert data.surface_triangle_tet_indices.shape == (6,)
    assert set(data.surface_triangle_tet_indices.tolist()) == {0, 1}
    assert data.metadata["surface_source"] == "derived"


@pytest.mark.parametrize(
    ("verts", "tets", "message"),
    [
        (_ONE_TET_VERTS, np.array([[0, 2, 1, 3]], dtype=np.int64), "inverted"),
        (
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            _ONE_TET,
            "degenerate",
        ),
        (_ONE_TET_VERTS, np.array([[0, 1, 1, 3]], dtype=np.int64), "repeats a vertex"),
        (_ONE_TET_VERTS, np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64), "duplicate"),
        (_ONE_TET_VERTS, np.array([[0, 1, 2, 99]], dtype=np.int64), "out of range"),
        (_ONE_TET_VERTS, np.array([[0, 1, 2, 2**32 + 3]], dtype=np.int64), "out of range"),
    ],
)
def test_load_tet_mesh_rejects_invalid_tets(monkeypatch, tmp_path, verts, tets, message):
    mesh_path = tmp_path / "invalid.mesh"
    mesh_path.write_text("", encoding="utf-8")

    def fake_read_mesh(_path):
        return verts, tets, np.empty((0, 3), dtype=np.int64)

    monkeypatch.setattr("genesis.utils.volumetric_mesh.igl.readMESH", fake_read_mesh)
    with pytest.raises(gs.GenesisException, match=message):
        load_tet_mesh(mesh_path)


def test_load_tet_mesh_rejects_large_out_of_range_surface_index(monkeypatch, tmp_path):
    mesh_path = tmp_path / "invalid_surface_index.mesh"
    mesh_path.write_text("", encoding="utf-8")

    def fake_read_mesh(_path):
        return _ONE_TET_VERTS, _ONE_TET, np.array([[0, 1, 2**32 + 2]], dtype=np.int64)

    monkeypatch.setattr("genesis.utils.volumetric_mesh.igl.readMESH", fake_read_mesh)
    with pytest.raises(gs.GenesisException, match="surface triangle indices are out of range"):
        load_tet_mesh(mesh_path)


def test_load_tet_mesh_rejects_invalid_explicit_surface(tmp_path):
    mesh_path = tmp_path / "invalid_surface.mesh"
    verts = np.concatenate([_ONE_TET_VERTS, np.array([[0.0, 0.0, -1.0]])], axis=0)
    tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int64)
    _write_mesh(mesh_path, verts, tets, faces=np.array([[0, 1, 2]], dtype=np.int64))

    with pytest.raises(gs.GenesisException, match="not a boundary face"):
        load_tet_mesh(mesh_path)


def test_tet_mesh_fem_sample_bypasses_tetgen(monkeypatch, tmp_path):
    mesh_path = tmp_path / "one_tet.mesh"
    _write_one_tet_mesh(mesh_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("TetGen path should not be called for TetMesh")

    monkeypatch.setattr(eu, "mesh_to_elements", fail_if_called)
    monkeypatch.setattr(mu, "tetrahedralize_mesh", fail_if_called)
    monkeypatch.setattr(eu, "split_all_surface_tets", fail_if_called)
    monkeypatch.setattr(mu.tetgen, "TetGen", fail_if_called)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(),
    )

    np.testing.assert_array_equal(entity.elems, _ONE_TET.astype(gs.np_int))
    np.testing.assert_array_equal(entity.surface_triangles, _ONE_TET_FACES.astype(gs.np_int))
    np.testing.assert_array_equal(entity.surface_triangle_tet_indices, np.zeros(4, dtype=gs.np_int))


def test_tet_mesh_entity_exposes_surface_metadata(tmp_path):
    mesh_path = tmp_path / "one_tet.mesh"
    _write_one_tet_mesh(mesh_path)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path)),
        material=gs.materials.FEM.Elastic(),
    )

    np.testing.assert_array_equal(entity.surface_triangles, _ONE_TET_FACES.astype(gs.np_int))
    np.testing.assert_array_equal(entity.surface_triangle_tet_indices, np.zeros(4, dtype=gs.np_int))
    np.testing.assert_array_equal(entity.boundary_vertex_indices, np.array([0, 1, 2, 3], dtype=gs.np_int))
    assert entity.volumetric_mesh_metadata["mesh_path"] == str(mesh_path)
    assert entity.volumetric_mesh_metadata["surface_source"] == "file"
    np.testing.assert_allclose(entity.volumetric_mesh_metadata["bbox_min"], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(entity.volumetric_mesh_metadata["bbox_max"], np.array([1.0, 1.0, 1.0]))


def test_tet_mesh_entity_records_applied_transform_metadata(tmp_path):
    mesh_path = tmp_path / "one_tet.mesh"
    _write_one_tet_mesh(mesh_path)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        morph=gs.morphs.TetMesh(file=str(mesh_path), scale=(2.0, 3.0, 4.0), pos=(1.0, 2.0, 3.0)),
        material=gs.materials.FEM.Elastic(),
    )

    metadata = entity.volumetric_mesh_metadata
    np.testing.assert_allclose(metadata["bbox_scaled_min"], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(metadata["bbox_scaled_max"], np.array([3.0, 5.0, 7.0]))
    assert metadata["applied_transform"]["scale"] == (2.0, 3.0, 4.0)
    assert metadata["applied_transform"]["pos"] == (1.0, 2.0, 3.0)
