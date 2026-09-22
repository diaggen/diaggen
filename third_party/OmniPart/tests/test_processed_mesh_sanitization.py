from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modules.part_synthesis import process_utils
from modules.part_synthesis.utils import postprocessing_utils


def _mesh(vertices: np.ndarray, faces: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        vertices=torch.from_numpy(vertices),
        faces=torch.from_numpy(faces),
    )


def _prepared_fixture() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0e-10, 0.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 1],
            [0, 1, 4],
            [0, 1, 2],
            [0, 1, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_prepare_processed_mesh_filters_exact_zero_area_faces_and_compacts(
    monkeypatch: pytest.MonkeyPatch,
):
    vertices, faces = _prepared_fixture()
    monkeypatch.setattr(
        postprocessing_utils,
        "postprocess_mesh",
        lambda *_args, **_kwargs: (vertices, faces),
    )

    prepared_vertices, prepared_faces = postprocessing_utils.prepare_processed_mesh(
        _mesh(vertices, faces),
        simplify=0.0,
        fill_holes=False,
        verbose=False,
    )

    assert prepared_vertices.dtype == np.float32
    assert prepared_faces.dtype == np.int32
    assert prepared_vertices.flags.c_contiguous
    assert prepared_faces.flags.c_contiguous
    np.testing.assert_array_equal(
        prepared_vertices,
        vertices[[0, 1, 2, 5]],
    )
    np.testing.assert_array_equal(prepared_faces, [[0, 1, 2], [0, 1, 3]])
    assert not postprocessing_utils._exact_zero_area_face_mask(
        prepared_vertices,
        prepared_faces,
    ).any()


def test_prepare_processed_mesh_rejects_all_degenerate_output(
    monkeypatch: pytest.MonkeyPatch,
):
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 1], [0, 1, 2]], dtype=np.int32)
    monkeypatch.setattr(
        postprocessing_utils,
        "postprocess_mesh",
        lambda *_args, **_kwargs: (vertices, faces),
    )

    with pytest.raises(ValueError, match="no nonzero-area faces"):
        postprocessing_utils.prepare_processed_mesh(
            _mesh(vertices, faces),
            simplify=0.0,
            fill_holes=False,
            verbose=False,
        )


def test_exact_zero_area_predicate_keeps_tiny_nonzero_float32_cross_product():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0e-15, 0.0, 0.0], [0.0, 1.0e-15, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    sanitized_vertices, sanitized_faces = (
        postprocessing_utils._sanitize_exact_zero_area_processed_faces(vertices, faces)
    )

    np.testing.assert_array_equal(sanitized_vertices, vertices)
    np.testing.assert_array_equal(sanitized_faces, faces)
    assert not postprocessing_utils._exact_zero_area_face_mask(vertices, faces).item()


@pytest.mark.parametrize(
    "vertices,faces,error",
    [
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            "vertices must be finite",
        ),
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, -1]], dtype=np.int32),
            "indices are out of range",
        ),
        (
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 3]], dtype=np.int32),
            "indices are out of range",
        ),
    ],
)
def test_prepare_processed_mesh_rejects_invalid_producer_topology(
    monkeypatch: pytest.MonkeyPatch,
    vertices: np.ndarray,
    faces: np.ndarray,
    error: str,
):
    monkeypatch.setattr(
        postprocessing_utils,
        "postprocess_mesh",
        lambda *_args, **_kwargs: (vertices, faces),
    )

    with pytest.raises(ValueError, match=error):
        postprocessing_utils.prepare_processed_mesh(
            _mesh(vertices, faces),
            simplify=0.0,
            fill_holes=False,
            verbose=False,
        )


def test_external_processed_mesh_with_zero_area_fails_at_validator_and_to_glb():
    vertices, faces = _prepared_fixture()
    external_processed_mesh = (
        np.ascontiguousarray(vertices),
        np.ascontiguousarray(faces),
    )

    with pytest.raises(ValueError, match="nonzero area"):
        postprocessing_utils._validate_processed_mesh(external_processed_mesh)
    with pytest.raises(ValueError, match="nonzero area"):
        postprocessing_utils.to_glb(
            None,
            None,
            textured=False,
            processed_mesh=external_processed_mesh,
        )

    valid_mesh = (
        np.ascontiguousarray(vertices[[0, 1, 2]]),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    glb = postprocessing_utils.to_glb(
        None,
        None,
        textured=False,
        processed_mesh=valid_mesh,
    )
    assert glb.faces.shape == (1, 3)


def test_save_parts_outputs_fans_one_sanitized_mesh_to_both_consumers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    vertices, faces = _prepared_fixture()
    seen = {}

    class _ExportableGlb:
        def export(self, path):
            path.write_bytes(b"glb")

    class _Savable:
        def save_ply(self, _path):
            pass

    monkeypatch.setattr(
        postprocessing_utils,
        "postprocess_mesh",
        lambda *_args, **_kwargs: (vertices, faces),
    )
    monkeypatch.setattr(
        postprocessing_utils,
        "to_glb",
        lambda *_args, processed_mesh, **_kwargs: seen.setdefault(
            "to_glb_processed_mesh", processed_mesh
        ) and _ExportableGlb(),
    )
    monkeypatch.setattr(process_utils, "merge_gaussians", lambda _items: _Savable())
    monkeypatch.setattr(
        process_utils,
        "exploded_gaussians",
        lambda _items, **_kwargs: _Savable(),
    )
    monkeypatch.setattr(
        process_utils,
        "write_appearance_bundle",
        lambda **kwargs: seen.setdefault("source_parts", kwargs["source_parts"]),
    )

    process_utils.save_parts_outputs(
        {
            "gaussian": [object(), object()],
            "mesh": [_mesh(vertices, faces), _mesh(vertices, faces)],
        },
        str(tmp_path),
        simplify_ratio=0.0,
        save_video=False,
        save_glb=True,
        textured=False,
        seed=3,
    )

    glb_vertices, glb_faces = seen["to_glb_processed_mesh"]
    source_part = seen["source_parts"][0]
    assert source_part.vertices is glb_vertices
    assert source_part.faces is glb_faces
    np.testing.assert_array_equal(source_part.vertices, vertices[[0, 1, 2, 5]])
    np.testing.assert_array_equal(source_part.faces, [[0, 1, 2], [0, 1, 3]])
    assert not postprocessing_utils._exact_zero_area_face_mask(
        source_part.vertices,
        source_part.faces,
    ).any()
