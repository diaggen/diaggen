"""Pure, no-GPU checks for the Clawhauser SAP drop validation example."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


EXAMPLE = Path(__file__).parents[1] / "examples" / "validation" / "clawhauser_tetmesh_drop.py"


def _module():
    spec = importlib.util.spec_from_file_location("clawhauser_tetmesh_drop", EXAMPLE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tet_j_and_health_definitions():
    module = _module()
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]])
    np.testing.assert_allclose(module.tet_j_ratios(rest, rest, tets), [1.0])

    compressed = rest.copy()
    compressed[3, 2] = 0.5
    healthy = module.classify_health(compressed, np.zeros_like(compressed), module.tet_j_ratios(rest, compressed, tets), 0.75)
    assert healthy == {"finite": True, "inversion": False, "near_inversion": True, "healthy": True, "min_j": 0.5, "min_j_tet": 0}

    inverted = rest.copy()
    inverted[3, 2] = -0.1
    result = module.classify_health(inverted, np.zeros_like(inverted), module.tet_j_ratios(rest, inverted, tets), 0.05)
    assert result["inversion"] and not result["healthy"]

    nonfinite = module.classify_health(rest, np.full_like(rest, np.nan), np.array([1.0]), 0.05)
    assert not nonfinite["finite"] and not nonfinite["healthy"] and not nonfinite["inversion"]
    with pytest.raises(ValueError, match="rest tetrahedra"):
        module.tet_j_ratios(rest, rest, np.array([[0, 1, 2, 2]]))
    invalid_rest = rest.copy()
    invalid_rest[3, 2] = np.nan
    with pytest.raises(ValueError, match="rest tetrahedra"):
        module.tet_j_ratios(invalid_rest, rest, tets)


def test_source_enforces_sap_tet_only_contract():
    source = EXAMPLE.read_text()
    tree = ast.parse(source)
    assert "SAPCouplerOptions(fem_floor_contact_type=\"tet\")" in source
    for token in ("precision=\"64\"", "use_implicit_solver=True", "EXPECTED_VERTEX_COUNT = 1840", "EXPECTED_TET_COUNT = 6850", "HeterogeneousMaterial", "tet_E_nu", "tet_density", "tet_part_labels", "global_min_z", "CAMERA_RESOLUTION", "CAMERA_OFFSET", "CAMERA_UP", "centroid_tracking", "set_tracking_camera", "CAMERA_FOV", "drop_pose", "video_size_bytes", "artifact_missing", "rendered_frame_count", "video_duration_expected_seconds"):
        assert token in source
    assert " or contact or " not in source
    assert "LegacyCouplerOptions" not in source
    assert "morphs.Plane" not in source
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [getattr(node.func, "attr", "") for node in calls]
    assert "control_dofs_position" not in names
    assert "set_vertex_constraints" not in names


def test_create_output_dir_rejects_nonempty_directory(tmp_path):
    module = _module()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "artifact").write_text("already here")
    with pytest.raises(FileExistsError):
        module.create_output_dir(str(occupied))
