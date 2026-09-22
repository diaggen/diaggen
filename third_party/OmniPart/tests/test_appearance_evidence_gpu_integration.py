from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from modules.part_synthesis.appearance_evidence import (
    APPEARANCE_SCHEMA_VERSION,
    RESOLUTION,
    VIEW_COUNT,
    AppearanceSourcePart,
    _sha256,
    _validate_manifest_and_bundle,
    _write_contact_sheet,
    write_appearance_bundle,
)
from modules.part_synthesis.representations.gaussian.gaussian_model import Gaussian


def _output_dir() -> Path:
    configured = os.environ.get("HAG4R_F6_OMNIPART_GPU_OUTPUT_DIR")
    if not configured:
        raise RuntimeError(
            "HAG4R_F6_OMNIPART_GPU_OUTPUT_DIR must explicitly select the "
            "HAG4R Feature 6 evidence directory"
        )
    output_dir = Path(configured).resolve()
    if (
        "outputs" not in output_dir.parts
        or "feature_6_texture_hard_acceptance" not in output_dir.parts
    ):
        raise ValueError("GPU integration output must stay in the Feature 6 evidence root")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _repo_relative_output_path(path: Path) -> str:
    path = Path(path)
    try:
        outputs_index = path.parts.index("outputs")
    except ValueError as exc:
        raise ValueError("evidence path must be under outputs/") from exc
    relative = Path(*path.parts[outputs_index:])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence path must be a safe repo-relative output path")
    return relative.as_posix()


def _black_gaussian() -> Gaussian:
    axis = torch.linspace(-0.36, 0.36, 9, device="cuda")
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    xyz = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.full_like(xx.reshape(-1), 0.45)],
        dim=1,
    )
    count = xyz.shape[0]
    gaussian = Gaussian(
        aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
        sh_degree=0,
        mininum_kernel_size=0.0,
        scaling_bias=0.01,
        opacity_bias=0.1,
        scaling_activation="exp",
        device="cuda",
    )
    gaussian.from_xyz(xyz)
    gaussian.from_scaling(torch.full((count, 3), 0.055, device="cuda"))
    gaussian.from_rotation(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").repeat(count, 1)
    )
    gaussian.from_opacity(torch.full((count, 1), 0.95, device="cuda"))
    # Degree-zero Gaussian SH is decoded as C0 * coefficient + 0.5.
    # This coefficient therefore produces true black foreground.
    gaussian.from_features(
        torch.full(
            (count, 1, 3),
            -0.5 / 0.28209479177387814,
            device="cuda",
        )
    )
    return gaussian


def _adjacent_source_parts() -> tuple[AppearanceSourcePart, AppearanceSourcePart]:
    left_vertices = np.asarray(
        [
            [-0.42, -0.42, 0.45],
            [0.0, -0.42, 0.45],
            [0.0, 0.42, 0.45],
            [-0.42, 0.42, 0.45],
        ],
        dtype=np.float32,
    )
    right_vertices = np.asarray(
        [
            [0.0, -0.42, 0.45],
            [0.42, -0.42, 0.45],
            [0.42, 0.42, 0.45],
            [0.0, 0.42, 0.45],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return (
        AppearanceSourcePart(
            part_index=0,
            source_output_index=1,
            source_glb_name="part1.glb",
            vertices=left_vertices,
            faces=faces,
        ),
        AppearanceSourcePart(
            part_index=1,
            source_output_index=2,
            source_glb_name="part2.glb",
            vertices=right_vertices,
            faces=faces,
        ),
    )


def test_production_writer_real_cuda_black_alpha_adjacent_parts_and_round_trip():
    assert torch.cuda.is_available()
    output_dir = _output_dir()
    manifest_path = write_appearance_bundle(
        output_dir=output_dir,
        merged_gaussian=_black_gaussian(),
        source_parts=_adjacent_source_parts(),
        seed=42,
    )
    torch.cuda.synchronize()

    appearance_dir = manifest_path.parent
    validation = _validate_manifest_and_bundle(appearance_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(appearance_dir / "appearance_views.npz", allow_pickle=False) as archive:
        views = {name: archive[name] for name in archive.files}
    with np.load(appearance_dir / "source_surface.npz", allow_pickle=False) as archive:
        surface = {name: archive[name] for name in archive.files}

    assert validation["valid"] is True
    assert manifest["schema_version"] == APPEARANCE_SCHEMA_VERSION
    assert manifest["capture"]["view_count"] == VIEW_COUNT
    assert manifest["capture"]["resolution"] == [RESOLUTION, RESOLUTION]
    assert views["rgb"].shape == (64, 512, 512, 3)
    assert views["alpha"].shape == (64, 512, 512)
    assert views["source_part_id"].shape == (64, 512, 512)
    assert views["rgb"].dtype == np.uint8
    assert views["alpha"].dtype == np.uint8
    assert views["source_part_id"].dtype == np.int16
    assert int(np.count_nonzero(views["alpha"])) > 0
    foreground = views["alpha"] >= round(0.05 * 255)
    assert int(np.count_nonzero(foreground)) > 0
    assert int(views["rgb"][foreground].max()) <= 1

    labels = views["source_part_id"]
    assert set(np.unique(labels).tolist()) == {-1, 0, 1}
    assert all(part["visible_pixel_count"] > 0 for part in manifest["parts"])
    assert np.all(np.any(labels >= 0, axis=(1, 2)))
    alpha_foreground = views["alpha"] >= round(0.05 * 255)
    geometry_foreground = labels >= 0
    overlap = int(np.count_nonzero(alpha_foreground & geometry_foreground))
    alpha_count = int(np.count_nonzero(alpha_foreground))
    geometry_count = int(np.count_nonzero(geometry_foreground))
    assert overlap / alpha_count > 0.5
    assert overlap / geometry_count > 0.95
    vertices_homogeneous = np.concatenate(
        [
            surface["vertices"],
            np.ones((len(surface["vertices"]), 1), dtype=np.float32),
        ],
        axis=1,
    )
    camera_vertices = np.einsum(
        "vij,nj->vni",
        views["extrinsics"],
        vertices_homogeneous,
    )
    vertex_depth = camera_vertices[..., 2]
    assert float(manifest["capture"]["near"]) < float(vertex_depth.min())
    assert float(vertex_depth.max()) < float(manifest["capture"]["far"])
    horizontal_neighbors = (labels[:, :, :-1] == 0) & (labels[:, :, 1:] == 1)
    reverse_neighbors = (labels[:, :, :-1] == 1) & (labels[:, :, 1:] == 0)
    vertical_neighbors = (labels[:, :-1, :] == 0) & (labels[:, 1:, :] == 1)
    reverse_vertical = (labels[:, :-1, :] == 1) & (labels[:, 1:, :] == 0)
    assert any(
        int(np.count_nonzero(neighbors)) > 0
        for neighbors in (
            horizontal_neighbors,
            reverse_neighbors,
            vertical_neighbors,
            reverse_vertical,
        )
    )
    np.testing.assert_array_equal(np.unique(surface["face_part_id"]), [0, 1])

    alpha_contact_sheet_path = output_dir / "alpha_contact_sheet.png"
    _write_contact_sheet(
        np.repeat(views["alpha"][..., None], 3, axis=-1),
        alpha_contact_sheet_path,
    )
    part_palette = np.zeros((*labels.shape, 3), dtype=np.uint8)
    part_palette[labels == 0] = np.asarray([255, 64, 64], dtype=np.uint8)
    part_palette[labels == 1] = np.asarray([64, 255, 64], dtype=np.uint8)
    part_id_contact_sheet_path = output_dir / "part_id_contact_sheet.png"
    _write_contact_sheet(part_palette, part_id_contact_sheet_path)

    summary = {
        "schema_version": "hag4r-feature-6-omnipart-gpu-contract-v1",
        "cuda_device": torch.cuda.get_device_name(0),
        "real_cuda_gaussian_renderer": True,
        "real_cuda_nvdiffrast_gbuffer": True,
        "view_count": VIEW_COUNT,
        "resolution": [RESOLUTION, RESOLUTION],
        "alpha_nonzero_pixel_count": int(np.count_nonzero(views["alpha"])),
        "alpha_geometry_overlap_pixel_count": overlap,
        "geometry_recall_against_alpha": overlap / alpha_count,
        "geometry_precision_against_alpha": overlap / geometry_count,
        "capture_near": manifest["capture"]["near"],
        "capture_far": manifest["capture"]["far"],
        "source_vertex_depth_min": float(vertex_depth.min()),
        "source_vertex_depth_max": float(vertex_depth.max()),
        "all_views_have_gbuffer_foreground": True,
        "black_foreground_max_rgb": int(views["rgb"][foreground].max()),
        "source_part_labels": np.unique(labels).tolist(),
        "adjacent_part_transition_count": int(
            np.count_nonzero(horizontal_neighbors)
            + np.count_nonzero(reverse_neighbors)
            + np.count_nonzero(vertical_neighbors)
            + np.count_nonzero(reverse_vertical)
        ),
        "manifest_path": _repo_relative_output_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "appearance_views_sha256": _sha256(
            appearance_dir / "appearance_views.npz"
        ),
        "source_surface_sha256": _sha256(appearance_dir / "source_surface.npz"),
        "contact_sheet_path": _repo_relative_output_path(
            appearance_dir / "qa" / "appearance_contact_sheet.png"
        ),
        "alpha_contact_sheet_path": _repo_relative_output_path(
            alpha_contact_sheet_path
        ),
        "part_id_contact_sheet_path": _repo_relative_output_path(
            part_id_contact_sheet_path
        ),
    }
    (output_dir / "gpu_contract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
