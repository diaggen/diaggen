import json
from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.live.capabilities import capability_report, surface_backend_status
from genesis.live.protocol import GenesisLiveError
from genesis.live.session import GenesisLiveSession
from genesis.utils.heterogeneous_materials import _surface_lame2d_from_e_nu


_SURFACE_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
f 2 4 3
"""

_SOURCE_FACES = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
_TRI_E_NU = np.array([[1.0e5, 0.25], [2.0e5, 0.40]], dtype=np.float64)
_TRI_DENSITY = np.array([1000.0, 2000.0], dtype=np.float64)
_TRI_THICKNESS = np.array([0.002, 0.003], dtype=np.float64)
_TRI_AREA_DENSITY = _TRI_DENSITY * _TRI_THICKNESS
_TRI_LABELS = np.array([3, 7], dtype=np.int32)
_EXPECTED_VERTEX_VOLUME = np.array([1.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 1.0], dtype=np.float64)
_EXPECTED_VERTEX_THICKNESS = np.array([0.002, 0.0025, 0.0025, 0.003], dtype=np.float64)


def _require_surface_backend():
    status = surface_backend_status()
    if not status["available"]:
        pytest.skip(f"surface backend unavailable: {status}")


def _write_surface_obj(tmp_path: Path) -> Path:
    mesh_path = tmp_path / "two_triangles.obj"
    mesh_path.write_text(_SURFACE_OBJ, encoding="utf-8")
    return mesh_path


def _write_surface_material_npz(
    path: Path,
    *,
    e_nu=_TRI_E_NU,
    density=_TRI_DENSITY,
    thickness=_TRI_THICKNESS,
    area_density=_TRI_AREA_DENSITY,
    labels=_TRI_LABELS,
):
    payload = {
        "tri_E_nu": np.asarray(e_nu),
        "tri_density": np.asarray(density),
        "tri_thickness": np.asarray(thickness),
        "tri_area_density": np.asarray(area_density),
    }
    if labels is not None:
        payload["tri_part_labels"] = np.asarray(labels)
    np.savez(path, **payload)


def _write_scene_config(tmp_path: Path, *, material_path: Path | None = None) -> Path:
    mesh_path = _write_surface_obj(tmp_path)
    if material_path is None:
        material_path = tmp_path / "surface_materials.npz"
        _write_surface_material_npz(material_path)
    config = {
        "backend": "cuda",
        "sim_options": {"dt": 0.001, "gravity": [0.0, 0.0, 0.0]},
        "coupler_options": {
            "contact_enable": False,
            "enable_rigid_ground_contact": False,
            "enable_rigid_rigid_contact": False,
        },
        "entities": [
            {
                "name": "surface",
                "morph": {"type": "surface_mesh", "file": str(mesh_path)},
                "material": {
                    "type": "surface_shell",
                    "E": 10000.0,
                    "nu": 0.45,
                    "rho": 200.0,
                    "thickness": 0.001,
                    "bending_stiffness": 0.0,
                    "friction_mu": 0.1,
                    "heterogeneous": {"kind": "surface_triangles", "material_file": str(material_path)},
                },
            }
        ],
    }
    config_path = tmp_path / "surface_scene.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _cloth_geometry(session: GenesisLiveSession):
    pytest.importorskip("uipc")
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot

    geometries = []
    visitor = SceneVisitor(session.scene.sim.coupler._ipc_scene)
    for geom_slot in visitor.geometries():
        if not isinstance(geom_slot, SimplicialComplexSlot):
            continue
        geom = geom_slot.geometry()
        solver_type_attr = geom.meta().find("solver_type")
        if not solver_type_attr:
            continue
        (solver_type,) = solver_type_attr.view()
        if solver_type == "cloth":
            geometries.append(geom)
    assert len(geometries) == 1
    return geometries[0]


def _expected_mu_lam():
    young = _TRI_E_NU[:, 0]
    nu = _TRI_E_NU[:, 1]
    return _surface_lame2d_from_e_nu(young, nu)


@pytest.mark.parametrize("backend", [None])
def test_live_surface_heterogeneous_scene_binds_backend_attrs(tmp_path, backend):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))

        geom = _cloth_geometry(session)
        expected_mu, expected_lam = _expected_mu_lam()
        lambda_attr = geom.triangles().find("lambda")
        mu_attr = geom.triangles().find("mu")
        thickness_attr = geom.vertices().find("thickness")
        volume_attr = geom.vertices().find("volume")
        mass_density_attr = geom.meta().find("mass_density")

        np.testing.assert_allclose(lambda_attr.view(), expected_lam)
        np.testing.assert_allclose(mu_attr.view(), expected_mu)
        assert lambda_attr.view()[0] != pytest.approx(lambda_attr.view()[1])
        assert mu_attr.view()[0] != pytest.approx(mu_attr.view()[1])
        np.testing.assert_allclose(thickness_attr.view(), _EXPECTED_VERTEX_THICKNESS)
        np.testing.assert_allclose(volume_attr.view(), _EXPECTED_VERTEX_VOLUME)
        assert mass_density_attr.view().item() == pytest.approx(1.0)
        assert float(np.sum(volume_attr.view()) * mass_density_attr.view().item()) == pytest.approx(4.0)
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
def test_live_surface_heterogeneous_face_order_matches_source_rows(tmp_path, backend):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))

        entity = session.entities["surface"]
        np.testing.assert_array_equal(entity.surface_triangles, _SOURCE_FACES)
        geom = _cloth_geometry(session)
        _expected_mu, expected_lam = _expected_mu_lam()
        np.testing.assert_allclose(geom.triangles().find("lambda").view(), expected_lam)
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
def test_live_surface_heterogeneous_stats_are_in_snapshots(tmp_path, backend):
    _require_surface_backend()
    try:
        session = GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path)))

        geometry = session.dispatch("geometry.context.get", {})
        fused = session.dispatch("observation.fused", {})
        material = fused["material"]
        metadata = material["heterogeneous_metadata"]

        assert material["heterogeneous"] is True
        assert material["heterogeneous_kind"] == "surface_triangles"
        assert metadata["row_count"] == 2
        assert metadata["total_area"] == pytest.approx(1.0)
        assert metadata["total_mass"] == pytest.approx(4.0)
        assert metadata["youngs_modulus"]["max"] == pytest.approx(2.0e5)
        assert metadata["poisson_ratio"]["max"] == pytest.approx(0.40)
        assert metadata["lambda"]["max"] == pytest.approx(_expected_mu_lam()[1].max())
        assert metadata["mu"]["min"] == pytest.approx(_expected_mu_lam()[0].min())
        assert metadata["density"]["mean"] == pytest.approx(1500.0)
        assert metadata["thickness"]["mean"] == pytest.approx(0.0025)
        assert metadata["area_density"]["mean"] == pytest.approx(4.0)
        assert metadata["labels"] == {"min": 3, "max": 7, "count": 2}
        assert metadata["vertex_mass"]["max"] == pytest.approx(4.0 / 3.0)
        assert metadata["vertex_thickness"]["mean"] == pytest.approx(0.0025)
        assert geometry["surface_total_area"] == pytest.approx(1.0)
        assert geometry["surface_total_mass"] == pytest.approx(4.0)
        assert fused["geometry"]["surface_total_area"] == pytest.approx(1.0)
        assert fused["geometry"]["surface_total_mass"] == pytest.approx(4.0)
    finally:
        gs.destroy()


@pytest.mark.parametrize("backend", [None])
def test_live_surface_heterogeneous_invalid_npz_fails_before_scene_build(monkeypatch, tmp_path, backend):
    bad_material_path = tmp_path / "bad_surface_materials.npz"
    _write_surface_material_npz(
        bad_material_path,
        e_nu=_TRI_E_NU[:1],
        density=_TRI_DENSITY[:1],
        thickness=_TRI_THICKNESS[:1],
        area_density=_TRI_AREA_DENSITY[:1],
        labels=_TRI_LABELS[:1],
    )

    def fail_scene(*_args, **_kwargs):
        raise AssertionError("invalid surface heterogeneous NPZ must fail before gs.Scene is constructed")

    def fail_surface_backend_status():
        raise AssertionError("invalid surface heterogeneous NPZ must fail before backend probing")

    monkeypatch.setattr(gs, "Scene", fail_scene)
    monkeypatch.setattr("genesis.live.session.surface_backend_status", fail_surface_backend_status)

    with pytest.raises(GenesisLiveError) as exc_info:
        GenesisLiveSession(scene_config_path=str(_write_scene_config(tmp_path, material_path=bad_material_path)))

    assert exc_info.value.code == "invalid_surface_heterogeneous_material"
    assert exc_info.value.details["material_file"] == str(bad_material_path)


@pytest.mark.parametrize("backend", [None])
def test_print_capabilities_includes_surface_heterogeneity_after_backend_probe(capsys):
    from genesis.live import server

    if not surface_backend_status()["available"]:
        pytest.skip("surface backend unavailable")

    assert server.main(
        [
            "--print-capabilities",
            "--require-capability",
            "surface_mesh_import",
            "--require-capability",
            "surface_shell_diagnostics",
            "--require-capability",
            "heterogeneous_surface_material_arrays",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "heterogeneous_surface_material_arrays" in payload["capabilities"]
    assert not payload["missing_required_capabilities"]


@pytest.mark.parametrize("backend", [None])
def test_surface_heterogeneous_backend_probe_rejects_missing_attrs(monkeypatch):
    import genesis.live.capabilities as capabilities

    monkeypatch.setattr(
        capabilities,
        "surface_backend_status",
        lambda: {"available": True, "requirements": {"uipc": True, "cuda": True, "ipc_coupler": True}},
    )
    monkeypatch.setattr(
        capabilities,
        "_surface_heterogeneous_attr_probe",
        lambda: {
            "available": False,
            "reason": "missing_surface_heterogeneous_ipc_attrs",
            "missing": ["triangles.lambda"],
            "attrs": {},
        },
    )

    report = capability_report(required_capabilities=["heterogeneous_surface_material_arrays"])

    assert "heterogeneous_surface_material_arrays" not in report["capabilities"]
    assert report["missing_required_capabilities"] == ["heterogeneous_surface_material_arrays"]
    status = report["backend_requirements"]["surface_heterogeneous_material_arrays"]
    assert status["available"] is False
    assert status["reason"] == "missing_surface_heterogeneous_ipc_attrs"
    assert status["missing"] == ["triangles.lambda"]
