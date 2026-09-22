from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from modules.part_synthesis import appearance_evidence
from modules.part_synthesis.appearance_evidence import (
    ALPHA_THRESHOLD,
    APPEARANCE_SCHEMA_VERSION,
    AppearanceSourcePart,
    BUNDLE_VALIDATION_SCHEMA_VERSION,
    CAMERA_RADIUS,
    CAPTURE_DEPTH_MARGIN_ABSOLUTE,
    CAPTURE_DEPTH_MARGIN_FRACTION,
    FOV_DEGREES,
    INTERNAL_Z_UP_TO_GLTF_Y_UP,
    RESOLUTION,
    VIEW_COUNT,
    _camera_depth_extrema,
    _combine_source_surface,
    _derive_capture_planes,
    _sha256,
    _unpremultiply_to_straight_srgb,
)


def _part(
    part_index: int,
    source_output_index: int,
    offset: float = 0.0,
) -> AppearanceSourcePart:
    vertices = np.asarray(
        [
            [offset + 0.0, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return AppearanceSourcePart(
        part_index=part_index,
        source_output_index=source_output_index,
        source_glb_name=f"part{source_output_index}.glb",
        vertices=vertices,
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def test_internal_to_gltf_transform_is_column_vector_contract():
    point = np.asarray([2.0, 3.0, 5.0, 1.0], dtype=np.float32)
    transformed = INTERNAL_Z_UP_TO_GLTF_Y_UP @ point
    np.testing.assert_array_equal(
        transformed,
        np.asarray([2.0, 5.0, -3.0, 1.0], dtype=np.float32),
    )


def test_combine_source_surface_preserves_order_offsets_and_part_ids():
    vertices, faces, face_part_id = _combine_source_surface(
        [_part(0, 1), _part(1, 3, offset=2.0)]
    )

    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert face_part_id.dtype == np.int32
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(face_part_id, [0, 1])
    np.testing.assert_array_equal(vertices.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(vertices.max(axis=0), [3.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "parts",
    [
        [_part(1, 1)],
        [_part(0, 1), _part(0, 2)],
        [],
    ],
)
def test_combine_source_surface_rejects_noncontinuous_or_empty_parts(parts):
    with pytest.raises(ValueError):
        _combine_source_surface(parts)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda part: replace(
            part,
            vertices=np.empty((0, 3), dtype=np.float32),
        ),
        lambda part: replace(
            part,
            faces=np.asarray([[0, 1, 8]], dtype=np.int32),
        ),
        lambda part: replace(
            part,
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        ),
        lambda part: replace(
            part,
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_combine_source_surface_rejects_invalid_meshes(mutation):
    with pytest.raises(ValueError):
        _combine_source_surface([mutation(_part(0, 1))])


def test_black_foreground_stays_black_with_independent_nonzero_alpha():
    premultiplied_rgb = np.zeros((1, 1, 3), dtype=np.float32)
    alpha = np.asarray([[0.75]], dtype=np.float32)

    straight_rgb = _unpremultiply_to_straight_srgb(premultiplied_rgb, alpha)
    rgb_uint8 = np.rint(straight_rgb * 255.0).astype(np.uint8)
    alpha_uint8 = np.rint(alpha * 255.0).astype(np.uint8)

    np.testing.assert_array_equal(rgb_uint8, np.zeros((1, 1, 3), dtype=np.uint8))
    assert alpha_uint8.item() > 0


def test_unpremultiply_applies_threshold_only_to_rgb():
    premultiplied_rgb = np.asarray(
        [[[0.1, 0.2, 0.3], [0.01, 0.01, 0.01]]],
        dtype=np.float32,
    )
    alpha = np.asarray([[0.5, ALPHA_THRESHOLD / 2]], dtype=np.float32)

    straight_rgb = _unpremultiply_to_straight_srgb(premultiplied_rgb, alpha)

    np.testing.assert_allclose(straight_rgb[0, 0], [0.2, 0.4, 0.6])
    np.testing.assert_array_equal(straight_rgb[0, 1], [0.0, 0.0, 0.0])
    assert np.rint(alpha[0, 1] * 255).astype(np.uint8) > 0


def test_sha256_matches_standard_library(tmp_path):
    path = tmp_path / "payload.bin"
    payload = b"omnipart appearance evidence"
    path.write_bytes(payload)

    assert _sha256(path) == hashlib.sha256(payload).hexdigest()


def test_dynamic_capture_planes_strictly_contain_elongated_source():
    vertices = np.asarray(
        [
            [-0.5, 0.0, 1.49],
            [0.5, 0.0, 2.51],
            [0.0, 0.1, 2.0],
        ],
        dtype=np.float32,
    )
    extrinsics = np.repeat(
        np.eye(4, dtype=np.float32)[None],
        VIEW_COUNT,
        axis=0,
    )

    near, far = _derive_capture_planes(
        vertices,
        torch.from_numpy(extrinsics),
    )
    minimum, maximum = _camera_depth_extrema(vertices, extrinsics)

    assert 0.0 < near < minimum
    assert maximum < far
    assert far > 1.6


def test_static_schema_camera_and_array_contracts_are_frozen():
    assert APPEARANCE_SCHEMA_VERSION == "hag4r-omnipart-appearance-v1"
    assert (
        BUNDLE_VALIDATION_SCHEMA_VERSION
        == "hag4r-omnipart-appearance-validation-v1"
    )
    assert (VIEW_COUNT, RESOLUTION) == (64, 512)
    assert (
        FOV_DEGREES,
        CAMERA_RADIUS,
        CAPTURE_DEPTH_MARGIN_ABSOLUTE,
        CAPTURE_DEPTH_MARGIN_FRACTION,
        ALPHA_THRESHOLD,
    ) == (
        40.0,
        2.0,
        0.1,
        0.1,
        0.05,
    )
    specs = appearance_evidence._APPEARANCE_VIEW_SPECS
    assert set(specs) == {
        "rgb",
        "alpha",
        "source_depth",
        "source_normal",
        "source_part_id",
        "extrinsics",
        "intrinsics",
    }
    assert specs["rgb"] == (
        np.dtype("uint8"),
        (64, 512, 512, 3),
    )
    assert specs["alpha"] == (
        np.dtype("uint8"),
        (64, 512, 512),
    )
    assert specs["source_depth"] == (
        np.dtype("float32"),
        (64, 512, 512),
    )
    assert specs["source_normal"] == (
        np.dtype("float16"),
        (64, 512, 512, 3),
    )
    assert specs["source_part_id"] == (
        np.dtype("int16"),
        (64, 512, 512),
    )

    writer_source = inspect.getsource(appearance_evidence.write_appearance_bundle)
    for field in (
        '"schema_version"',
        '"producer"',
        '"capture"',
        '"coordinate_frames"',
        '"encodings"',
        '"source_bounds"',
        '"parts"',
        '"qa_statistics"',
        '"files"',
    ):
        assert field in writer_source


def test_writer_uses_hammersley_white_override_and_no_ply_or_color_mask_fallback():
    module_source = inspect.getsource(appearance_evidence)

    assert "sphere_hammersley_sequence(i, VIEW_COUNT)" in module_source
    assert "colors_overwrite=white_override" in module_source
    assert "merged_gs.ply" not in module_source
    assert "np.any(observation > 0)" not in module_source
