from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import genesis as gs
from genesis.engine.controllers.box_end_effector import BoxEndEffectorController, apply_static_box_anchors
from genesis.utils.misc import tensor_to_array


BOX_SIZE = (0.45, 0.12, 0.12)
BOX_MAXVOLUME = 2.0e-4
BOX_NOBISECT = False
BOX_MINRATIO = 1.2
CASE_NAMES = ("high_k_high_E", "low_k_high_E", "high_k_low_E")
VIDEO_NOTE = (
    "MP4s are software-projected validation visualizations from simulated tet FEM states; "
    "they are not Genesis camera renders. Red markers are hard anchors, blue markers are actual grasp vertices, "
    "and yellow x markers are soft target setpoints."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate native implicit volumetric FEM soft-actuator stiffness and material stiffness."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation/volumetric_soft_actuator"))
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("32", "64"), default="64")
    parser.add_argument("--k-high", type=float, default=2.0e5)
    parser.add_argument("--k-low", type=float, default=10.0)
    parser.add_argument("--e-high", type=float, default=5.0e4)
    parser.add_argument("--e-low", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-newton", type=int, default=3)
    parser.add_argument("--n-pcg", type=int, default=300)
    parser.add_argument("--n-linesearch", type=int, default=5)
    parser.add_argument("--pcg-threshold", type=float, default=1.0e-8)
    parser.add_argument("--damping-alpha", type=float, default=0.0)
    parser.add_argument("--damping-beta", type=float, default=0.0)
    parser.add_argument("--ramp-steps", type=int, default=36)
    parser.add_argument("--hold-steps", type=int, default=24)
    parser.add_argument("--total-steps", type=int, default=72)
    parser.add_argument("--command-scale", type=float, default=0.15)
    parser.add_argument("--slab-fraction", type=float, default=0.08)
    parser.add_argument("--selection-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=24)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    settle_steps = args.total_steps - args.ramp_steps - args.hold_steps
    if settle_steps < 0:
        raise ValueError(
            "total_steps must be at least ramp_steps + hold_steps; "
            f"got total_steps={args.total_steps}, ramp_steps={args.ramp_steps}, hold_steps={args.hold_steps}."
        )
    if args.hold_steps < 1 or args.ramp_steps < 1:
        raise ValueError("ramp_steps and hold_steps must be positive.")
    if args.video_width < 320 or args.video_height < 240:
        raise ValueError("Video resolution must be at least 320x240.")
    for name in ("k_high", "k_low", "e_high", "e_low", "rho", "dt", "command_scale", "slab_fraction"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if args.k_high / args.k_low < 100.0:
        raise ValueError(f"k_high / k_low must be >= 100, got {args.k_high / args.k_low}.")
    if args.e_high / args.e_low < 20.0:
        raise ValueError(f"e_high / e_low must be >= 20, got {args.e_high / args.e_low}.")


def _run_dir(args: argparse.Namespace) -> Path:
    run_id = args.run_id
    if run_id == "auto":
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / run_id


def _case_settings(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    return {
        "high_k_high_E": {"k": float(args.k_high), "E": float(args.e_high)},
        "low_k_high_E": {"k": float(args.k_low), "E": float(args.e_high)},
        "high_k_low_E": {"k": float(args.k_high), "E": float(args.e_low)},
    }


def _thresholds() -> dict[str, float]:
    return {
        "high_k_high_E_target_error_norm_max": 0.15,
        "low_k_high_E_target_error_norm_min": 0.30,
        "low_over_high_target_error_norm_ratio_min": 3.0,
        "high_k_high_E_strain_rms_max": 0.05,
        "high_k_low_E_strain_rms_min": 0.15,
        "low_E_over_high_E_strain_rms_ratio_min": 3.0,
        "high_k_low_E_target_error_norm_max": 0.25,
        "tet_edge_count_min": 10,
        "mp4_frame_count_min": 24,
        "mp4_width_min": 320,
        "mp4_height_min": 240,
        "mp4_luminance_std_min": 5.0,
        "mp4_first_last_mean_abs_diff_min": 1.0,
    }


def _init_genesis(args: argparse.Namespace) -> None:
    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend, precision=args.precision, logging_level="warning", seed=0)


def _make_scene(args: argparse.Namespace, youngs_modulus: float):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            enable_vertex_constraints=True,
            use_implicit_solver=True,
            n_newton_iterations=args.n_newton,
            n_pcg_iterations=args.n_pcg,
            n_linesearch_iterations=args.n_linesearch,
            pcg_threshold=args.pcg_threshold,
            damping_alpha=args.damping_alpha,
            damping_beta=args.damping_beta,
        ),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=BOX_SIZE,
            maxvolume=BOX_MAXVOLUME,
            nobisect=BOX_NOBISECT,
            minratio=BOX_MINRATIO,
        ),
        material=gs.materials.FEM.Elastic(
            E=youngs_modulus,
            rho=args.rho,
            model="linear_corotated",
        ),
    )
    scene.build()
    return scene, entity


def _positions(entity) -> np.ndarray:
    return np.asarray(tensor_to_array(entity.get_state().pos[0]), dtype=np.float64).copy()


def _tet_edges(tets: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=np.int64):
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            a, b = int(tet[i]), int(tet[j])
            edges.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(edges), dtype=np.int64)


def _metric_edges(tets: np.ndarray, anchor_vertices: np.ndarray, grasp_vertices: np.ndarray) -> np.ndarray:
    anchor_set = {int(v) for v in anchor_vertices}
    grasp_set = {int(v) for v in grasp_vertices}
    selected = []
    for i, j in _tet_edges(tets):
        both_anchor = int(i) in anchor_set and int(j) in anchor_set
        both_grasp = int(i) in grasp_set and int(j) in grasp_set
        if not both_anchor and not both_grasp:
            selected.append((int(i), int(j)))
    return np.asarray(selected, dtype=np.int64)


def _aabbs(rest_positions: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    bbox_min = rest_positions.min(axis=0)
    bbox_max = rest_positions.max(axis=0)
    extents = bbox_max - bbox_min
    l0 = float(np.linalg.norm(extents))
    d_cmd = float(args.command_scale * l0)
    slab_width = float(args.slab_fraction * extents[0])
    if slab_width <= 0.0:
        raise ValueError(f"Anchor/grasp slab width must be positive, got {slab_width}.")
    tol = float(args.selection_tolerance)
    anchor_box = np.array(
        [
            bbox_min[0] - tol,
            bbox_min[1] - tol,
            bbox_min[2] - tol,
            bbox_min[0] + slab_width,
            bbox_max[1] + tol,
            bbox_max[2] + tol,
        ],
        dtype=np.float64,
    )
    grasp_box = np.array(
        [
            bbox_max[0] - slab_width,
            bbox_min[1] - tol,
            bbox_min[2] - tol,
            bbox_max[0] + tol,
            bbox_max[1] + tol,
            bbox_max[2] + tol,
        ],
        dtype=np.float64,
    )
    return anchor_box, grasp_box, slab_width, l0, d_cmd


def _install_constraints(
    entity,
    case_name: str,
    args: argparse.Namespace,
    stiffness: float,
    rest_positions: np.ndarray,
):
    anchor_box, grasp_box, slab_width, l0, d_cmd = _aabbs(rest_positions, args)
    anchor_records = apply_static_box_anchors(
        entity,
        [{"anchor_id": "min_x_hard_anchor", "frame": "env_local", "box": anchor_box}],
        is_soft_constraint=False,
        stiffness=0.0,
        selection_tolerance=args.selection_tolerance,
    )
    anchor_vertices = np.asarray(anchor_records[0].selected_vertices, dtype=np.int64)

    controller = BoxEndEffectorController(entity, controller_id=f"{case_name}_max_x_box_ee")
    grasp_state = controller.grasp(
        grasp_box,
        frame="env_local",
        is_soft_constraint=True,
        stiffness=stiffness,
        selection_tolerance=args.selection_tolerance,
    )
    grasp_vertices = np.asarray(grasp_state.selected_vertices, dtype=np.int64)

    overlap = np.intersect1d(anchor_vertices, grasp_vertices)
    if anchor_vertices.size == 0 or grasp_vertices.size == 0:
        raise RuntimeError(
            f"Anchor and grasp selections must be nonempty; got anchor={anchor_vertices.size}, "
            f"grasp={grasp_vertices.size}."
        )
    if overlap.size > 0:
        raise RuntimeError(f"Anchor and grasp selections must be disjoint; overlap={overlap.tolist()}.")

    return {
        "controller": controller,
        "anchor_box": anchor_box,
        "grasp_box": grasp_box,
        "slab_width": slab_width,
        "L0": l0,
        "D_cmd": d_cmd,
        "anchor_vertices": anchor_vertices,
        "grasp_vertices": grasp_vertices,
    }


def _record_target(controller: BoxEndEffectorController) -> np.ndarray:
    return np.asarray(controller.state.target_positions, dtype=np.float64).copy()


def _start_motion(
    controller: BoxEndEffectorController,
    grasp_box: np.ndarray,
    d_cmd: float,
    args: argparse.Namespace,
) -> None:
    mins = grasp_box[:3]
    maxs = grasp_box[3:]
    reference_extent = float(np.max(maxs - mins))
    if reference_extent <= 0.0:
        raise ValueError(f"Grasp AABB reference extent must be positive, got {reference_extent}.")
    distance_scale = d_cmd / reference_extent
    speed = d_cmd / (float(args.ramp_steps) * float(args.dt))
    controller.move_positive_y(
        distance_scale=distance_scale,
        duration_steps=args.ramp_steps,
        speed=speed,
        max_distance_scale=max(1.0, distance_scale * 1.01 + 0.01),
        dt=args.dt,
    )


def _simulate_case(case_name: str, settings: dict[str, float], args: argparse.Namespace, output_dir: Path):
    scene, entity = _make_scene(args, settings["E"])
    rest_positions = _positions(entity)
    constraints = _install_constraints(entity, case_name, args, settings["k"], rest_positions)
    controller = constraints["controller"]

    settle_steps = args.total_steps - args.ramp_steps - args.hold_steps
    recorded_positions = [_positions(entity)]
    recorded_targets = [_record_target(controller)]

    for _ in range(settle_steps):
        scene.step(update_visualizer=False)
        recorded_positions.append(_positions(entity))
        recorded_targets.append(_record_target(controller))

    _start_motion(controller, constraints["grasp_box"], constraints["D_cmd"], args)
    for _ in range(args.ramp_steps):
        controller.advance_motion(steps=1, dt=args.dt)
        scene.step(update_visualizer=False)
        recorded_positions.append(_positions(entity))
        recorded_targets.append(_record_target(controller))

    for _ in range(args.hold_steps):
        scene.step(update_visualizer=False)
        recorded_positions.append(_positions(entity))
        recorded_targets.append(_record_target(controller))

    positions = np.stack(recorded_positions, axis=0)
    targets = np.stack(recorded_targets, axis=0)
    expected_frames = args.total_steps + 1
    if positions.shape[0] != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} recorded states, got {positions.shape[0]}.")

    tets = np.asarray(tensor_to_array(entity.get_el2v()), dtype=np.int64)
    surface_triangles = np.asarray(entity.surface_triangles, dtype=np.int64)
    metric_edges = _metric_edges(tets, constraints["anchor_vertices"], constraints["grasp_vertices"])
    metrics = _compute_metrics(
        positions=positions,
        targets=targets,
        rest_positions=rest_positions,
        grasp_vertices=constraints["grasp_vertices"],
        metric_edges=metric_edges,
        hold_steps=args.hold_steps,
        d_cmd=constraints["D_cmd"],
        l0=constraints["L0"],
    )

    mp4_path = output_dir / f"{case_name}.mp4"
    _write_software_projection_mp4(
        path=mp4_path,
        case_name=case_name,
        positions=positions,
        targets=targets,
        surface_triangles=surface_triangles,
        anchor_vertices=constraints["anchor_vertices"],
        grasp_vertices=constraints["grasp_vertices"],
        width=args.video_width,
        height=args.video_height,
        fps=args.video_fps,
    )
    mp4_check = _check_mp4(mp4_path, _thresholds())

    return {
        "settings": {"k": settings["k"], "E": settings["E"], "rho": float(args.rho)},
        "vertex_count": int(rest_positions.shape[0]),
        "tet_count": int(tets.shape[0]),
        "surface_triangle_count": int(surface_triangles.shape[0]),
        "recorded_state_count": int(positions.shape[0]),
        "anchor_vertex_count": int(constraints["anchor_vertices"].size),
        "grasp_vertex_count": int(constraints["grasp_vertices"].size),
        "anchor_vertices": constraints["anchor_vertices"].astype(int).tolist(),
        "grasp_vertices": constraints["grasp_vertices"].astype(int).tolist(),
        "metric_edge_count": int(metric_edges.shape[0]),
        "D_cmd": float(constraints["D_cmd"]),
        "L0": float(constraints["L0"]),
        "slab_width": float(constraints["slab_width"]),
        "anchor_box": constraints["anchor_box"].astype(float).tolist(),
        "grasp_box": constraints["grasp_box"].astype(float).tolist(),
        "metrics": metrics,
        "artifacts": {"mp4": str(mp4_path)},
        "mp4_check": mp4_check,
    }


def _compute_metrics(
    *,
    positions: np.ndarray,
    targets: np.ndarray,
    rest_positions: np.ndarray,
    grasp_vertices: np.ndarray,
    metric_edges: np.ndarray,
    hold_steps: int,
    d_cmd: float,
    l0: float,
) -> dict[str, float]:
    if metric_edges.shape[0] < _thresholds()["tet_edge_count_min"]:
        raise RuntimeError(f"Under-resolved deformation metric: only {metric_edges.shape[0]} eligible tet edges.")

    hold_positions = positions[-hold_steps:]
    hold_targets = targets[-hold_steps:]
    grasp_positions = hold_positions[:, grasp_vertices, :]
    target_errors = np.linalg.norm(grasp_positions - hold_targets, axis=-1)
    final_errors = target_errors[-1]

    edges_i = metric_edges[:, 0]
    edges_j = metric_edges[:, 1]
    rest_lengths = np.linalg.norm(rest_positions[edges_i] - rest_positions[edges_j], axis=-1)
    hold_lengths = np.linalg.norm(hold_positions[:, edges_i, :] - hold_positions[:, edges_j, :], axis=-1)
    denom = np.maximum(rest_lengths, 1.0e-6 * l0)
    strain = (hold_lengths - rest_lengths.reshape((1, -1))) / denom.reshape((1, -1))

    return {
        "target_error_norm": float(math.sqrt(float(np.mean(target_errors**2))) / max(d_cmd, 1.0e-9)),
        "target_error_rms": float(math.sqrt(float(np.mean(target_errors**2)))),
        "final_target_error_rms": float(math.sqrt(float(np.mean(final_errors**2)))),
        "final_target_error_max": float(np.max(final_errors)),
        "hold_target_error_max": float(np.max(target_errors)),
        "strain_rms": float(math.sqrt(float(np.mean(strain**2)))),
        "strain_max_abs": float(np.max(np.abs(strain))),
    }


def _projection_basis(positions: np.ndarray, width: int, height: int):
    view_dir = np.array([1.0, -1.6, 0.8], dtype=np.float64)
    view_dir /= np.linalg.norm(view_dir)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, view_dir)
    right /= np.linalg.norm(right)
    up = np.cross(view_dir, right)
    center = positions.reshape((-1, 3)).mean(axis=0)
    rel = positions.reshape((-1, 3)) - center.reshape((1, 3))
    xy = np.column_stack((rel @ right, rel @ up))
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1.0e-9)
    scale = 0.76 * min(width / span[0], height / span[1])
    return center, right, up, view_dir, scale


def _project(
    points: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    scale: float,
    width: int,
    height: int,
):
    rel = points - center.reshape((1, 3))
    x = width * 0.5 + (rel @ right) * scale
    y = height * 0.5 - (rel @ up) * scale
    return np.column_stack((x, y))


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for x in range(0, width, 32):
        color = (222, 226, 229) if x % 64 else (205, 211, 216)
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for y in range(0, height, 32):
        color = (222, 226, 229) if y % 64 else (205, 211, 216)
        draw.line([(0, y), (width, y)], fill=color, width=1)


def _draw_vertex_markers(
    draw: ImageDraw.ImageDraw,
    projected: np.ndarray,
    indices: np.ndarray,
    color: tuple[int, int, int],
):
    for idx in indices:
        x, y = projected[int(idx)]
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(20, 25, 30), width=1)


def _draw_target_markers(draw: ImageDraw.ImageDraw, projected_targets: np.ndarray) -> None:
    for x, y in projected_targets:
        r = 5
        draw.line((x - r, y - r, x + r, y + r), fill=(239, 198, 54), width=2)
        draw.line((x - r, y + r, x + r, y - r), fill=(239, 198, 54), width=2)
        draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(124, 92, 6), width=1)


def _write_software_projection_mp4(
    *,
    path: Path,
    case_name: str,
    positions: np.ndarray,
    targets: np.ndarray,
    surface_triangles: np.ndarray,
    anchor_vertices: np.ndarray,
    grasp_vertices: np.ndarray,
    width: int,
    height: int,
    fps: int,
) -> None:
    center, right, up, view_dir, scale = _projection_basis(positions, width, height)
    frames = []
    x_rest = positions[0, :, 0]
    x_min = float(x_rest.min())
    x_range = max(float(x_rest.max() - x_min), 1.0e-9)

    if targets.shape[0] != positions.shape[0]:
        raise ValueError(f"Expected one target set per recorded frame, got {targets.shape[0]} for {positions.shape[0]}.")
    if targets.shape[1] != grasp_vertices.shape[0]:
        raise ValueError(
            f"Expected one target per grasp vertex, got {targets.shape[1]} targets for {grasp_vertices.shape[0]} vertices."
        )

    for verts, frame_targets in zip(positions, targets):
        image = Image.new("RGB", (width, height), (236, 239, 242))
        draw = ImageDraw.Draw(image)
        _draw_grid(draw, width, height)

        projected = _project(verts, center, right, up, scale, width, height)
        projected_targets = _project(frame_targets, center, right, up, scale, width, height)
        face_depths = verts[surface_triangles].mean(axis=1) @ view_dir
        for tri_idx in np.argsort(face_depths):
            tri = surface_triangles[int(tri_idx)]
            polygon = [tuple(projected[int(v)]) for v in tri]
            tone = float((x_rest[tri].mean() - x_min) / x_range)
            fill = (
                int(62 + 155 * tone),
                int(120 + 78 * (1.0 - abs(tone - 0.5) * 2.0)),
                int(190 - 85 * tone),
            )
            draw.polygon(polygon, fill=fill, outline=(34, 43, 49))

        actual_grasp = projected[grasp_vertices]
        for actual, target in zip(actual_grasp, projected_targets):
            draw.line((actual[0], actual[1], target[0], target[1]), fill=(214, 176, 44), width=1)
        _draw_vertex_markers(draw, projected, anchor_vertices, (204, 53, 59))
        _draw_vertex_markers(draw, projected, grasp_vertices, (43, 105, 191))
        _draw_target_markers(draw, projected_targets)

        draw.rectangle((8, 8, 456, 50), fill=(248, 249, 250), outline=(93, 104, 112))
        draw.text((14, 14), case_name, fill=(22, 28, 33))
        draw.text((14, 32), "red=anchor  blue=actual grasp  yellow=setpoint", fill=(22, 28, 33))
        frames.append(np.asarray(image))

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def _luminance(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame[..., :3], dtype=np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _check_mp4(path: Path, thresholds: dict[str, float]) -> dict[str, object]:
    failures = []
    if not path.exists():
        failures.append(f"{path} does not exist.")
        return {"passed": False, "failures": failures}
    size_bytes = int(path.stat().st_size)
    if size_bytes <= 0:
        failures.append(f"{path} is empty.")
        return {"passed": False, "failures": failures, "size_bytes": size_bytes}

    frames = [np.asarray(frame) for frame in imageio.mimread(path)]
    frame_count = len(frames)
    if frame_count == 0:
        failures.append(f"{path} decoded to zero frames.")
        return {"passed": False, "failures": failures, "size_bytes": size_bytes, "frame_count": frame_count}
    if frame_count < thresholds["mp4_frame_count_min"]:
        failures.append(f"{path} decoded to {frame_count} frames.")
    height, width = frames[0].shape[:2]
    if width < thresholds["mp4_width_min"] or height < thresholds["mp4_height_min"]:
        failures.append(f"{path} resolution is {width}x{height}.")

    sample_indices = [0, frame_count // 2, frame_count - 1]
    luminance_std = [float(np.std(_luminance(frames[i]))) for i in sample_indices]
    for label, value in zip(("first", "middle", "last"), luminance_std):
        if value < thresholds["mp4_luminance_std_min"]:
            failures.append(f"{path} {label} frame luminance std is {value:.3f}.")

    first_last_diff = float(
        np.mean(np.abs(frames[0][..., :3].astype(np.float64) - frames[-1][..., :3].astype(np.float64)))
    )
    if first_last_diff < thresholds["mp4_first_last_mean_abs_diff_min"]:
        failures.append(f"{path} first-last mean abs pixel diff is {first_last_diff:.3f}.")

    return {
        "passed": not failures,
        "failures": failures,
        "size_bytes": size_bytes,
        "frame_count": frame_count,
        "width": int(width),
        "height": int(height),
        "luminance_std_first_middle_last": luminance_std,
        "first_last_mean_abs_pixel_diff": first_last_diff,
    }


def _build_artifacts(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "validation_report.md"
    return {
        "output_dir": str(output_dir),
        "metrics_json": str(metrics_path),
        "validation_report": str(report_path),
        "mp4s": {name: str(output_dir / f"{name}.mp4") for name in CASE_NAMES},
    }


def _common_anchor_grasp(case_results: dict[str, dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    first = case_results[CASE_NAMES[0]]
    anchors = {
        "kind": "hard min-x slab",
        "box": first["anchor_box"],
        "slab_width": first["slab_width"],
        "selected_vertices": first["anchor_vertices"],
        "selected_vertex_count": first["anchor_vertex_count"],
    }
    grasp = {
        "kind": "soft BoxEE max-x slab",
        "box": first["grasp_box"],
        "selected_vertices": first["grasp_vertices"],
        "selected_vertex_count": first["grasp_vertex_count"],
        "motion_axis": "positive_y",
        "D_cmd": first["D_cmd"],
        "ramp_steps": None,
        "hold_steps": None,
    }
    return anchors, grasp


def _metrics_payload(args: argparse.Namespace, output_dir: Path, case_results: dict[str, dict[str, object]]):
    artifacts = _build_artifacts(args, output_dir)
    anchors, grasp = _common_anchor_grasp(case_results)
    grasp["ramp_steps"] = int(args.ramp_steps)
    grasp["hold_steps"] = int(args.hold_steps)

    first = case_results[CASE_NAMES[0]]
    return {
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "script": str(Path(__file__).resolve()),
        },
        "mesh": {
            "type": "gs.morphs.Box",
            "size": list(BOX_SIZE),
            "maxvolume": BOX_MAXVOLUME,
            "nobisect": BOX_NOBISECT,
            "minratio": BOX_MINRATIO,
            "vertex_count": first["vertex_count"],
            "tet_count": first["tet_count"],
            "surface_triangle_count": first["surface_triangle_count"],
            "validation_basis": "native tetrahedral FEM body with tet-vertex soft target constraints",
        },
        "solver_options": {
            "backend": args.backend,
            "precision": args.precision,
            "dt": float(args.dt),
            "substeps": 1,
            "use_implicit_solver": True,
            "n_newton_iterations": int(args.n_newton),
            "n_pcg_iterations": int(args.n_pcg),
            "n_linesearch_iterations": int(args.n_linesearch),
            "pcg_threshold": float(args.pcg_threshold),
            "damping_alpha": float(args.damping_alpha),
            "damping_beta": float(args.damping_beta),
            "gravity": [0.0, 0.0, 0.0],
            "ramp_steps": int(args.ramp_steps),
            "hold_steps": int(args.hold_steps),
            "total_steps": int(args.total_steps),
            "settle_steps": int(args.total_steps - args.ramp_steps - args.hold_steps),
        },
        "cases": case_results,
        "anchors": anchors,
        "grasp": grasp,
        "artifacts": artifacts,
        "thresholds": _thresholds(),
        "video_note": VIDEO_NOTE,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Volumetric Soft Actuator Validation",
        "",
        VIDEO_NOTE,
        "",
        "The simulated entity is a tetrahedral native FEM Box. Elastic behavior remains volume-based, and soft target "
        "constraints are applied to tet vertices through BoxEE rather than IPC shell-only vertices.",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(payload["command"]["argv"]),
        "```",
        "",
        "## Settings",
        "",
        f"- Mesh: `gs.morphs.Box(size={tuple(payload['mesh']['size'])}, maxvolume={payload['mesh']['maxvolume']}, "
        f"nobisect={payload['mesh']['nobisect']}, minratio={payload['mesh']['minratio']})`",
        f"- Solver: backend `{payload['solver_options']['backend']}`, precision "
        f"`{payload['solver_options']['precision']}`, "
        f"dt `{payload['solver_options']['dt']}`, Newton `{payload['solver_options']['n_newton_iterations']}`, "
        f"PCG `{payload['solver_options']['n_pcg_iterations']}`, line search "
        f"`{payload['solver_options']['n_linesearch_iterations']}`",
        f"- Damping: alpha `{payload['solver_options']['damping_alpha']}`, beta "
        f"`{payload['solver_options']['damping_beta']}`",
        f"- Motion: settle `{payload['solver_options']['settle_steps']}`, ramp "
        f"`{payload['solver_options']['ramp_steps']}`, hold `{payload['solver_options']['hold_steps']}`, total "
        f"`{payload['solver_options']['total_steps']}`",
        f"- k_high `{payload['cases']['high_k_high_E']['settings']['k']}`, "
        f"k_low `{payload['cases']['low_k_high_E']['settings']['k']}`, "
        f"E_high `{payload['cases']['high_k_high_E']['settings']['E']}`, "
        f"E_low `{payload['cases']['high_k_low_E']['settings']['E']}`",
        "",
        "## Metrics",
        "",
        "| case | k | E | target_error_norm | strain_rms | hold_target_error_max | MP4 check |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case_name in CASE_NAMES:
        case = payload["cases"][case_name]
        metrics = case["metrics"]
        mp4_check = "pass" if case["mp4_check"]["passed"] else "fail"
        lines.append(
            f"| {case_name} | {case['settings']['k']:.6g} | {case['settings']['E']:.6g} | "
            f"{metrics['target_error_norm']:.6g} | {metrics['strain_rms']:.6g} | "
            f"{metrics['hold_target_error_max']:.6g} | {mp4_check} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `metrics.json`: `{payload['artifacts']['metrics_json']}`",
            f"- `validation_report.md`: `{payload['artifacts']['validation_report']}`",
        ]
    )
    for case_name in CASE_NAMES:
        lines.append(f"- `{case_name}.mp4`: `{payload['artifacts']['mp4s'][case_name]}`")

    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- high_k_high_E target_error_norm <= "
            f"{payload['thresholds']['high_k_high_E_target_error_norm_max']}",
            f"- low_k_high_E target_error_norm >= "
            f"{payload['thresholds']['low_k_high_E_target_error_norm_min']}",
            f"- low/high target error ratio >= "
            f"{payload['thresholds']['low_over_high_target_error_norm_ratio_min']}",
            f"- high_k_high_E strain_rms <= {payload['thresholds']['high_k_high_E_strain_rms_max']}",
            f"- high_k_low_E strain_rms >= {payload['thresholds']['high_k_low_E_strain_rms_min']}",
            f"- low-E/high-E strain ratio >= {payload['thresholds']['low_E_over_high_E_strain_rms_ratio_min']}",
            f"- high_k_low_E target_error_norm <= "
            f"{payload['thresholds']['high_k_low_E_target_error_norm_max']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_failures(payload: dict[str, object]) -> list[str]:
    thresholds = payload["thresholds"]
    cases = payload["cases"]
    failures: list[str] = []

    for case_name in CASE_NAMES:
        case = cases[case_name]
        if case["anchor_vertex_count"] <= 0 or case["grasp_vertex_count"] <= 0:
            failures.append(f"{case_name}: anchor and grasp selections must be nonempty.")
        if set(case["anchor_vertices"]) & set(case["grasp_vertices"]):
            failures.append(f"{case_name}: anchor and grasp selections overlap.")
        if case["metric_edge_count"] < thresholds["tet_edge_count_min"]:
            failures.append(f"{case_name}: metric edge count {case['metric_edge_count']} is under-resolved.")
        if not case["mp4_check"]["passed"]:
            failures.extend([f"{case_name}: {failure}" for failure in case["mp4_check"]["failures"]])

    first_anchor = cases[CASE_NAMES[0]]["anchor_vertices"]
    first_grasp = cases[CASE_NAMES[0]]["grasp_vertices"]
    for case_name in CASE_NAMES[1:]:
        if cases[case_name]["anchor_vertices"] != first_anchor:
            failures.append(f"{case_name}: anchor vertices differ from high_k_high_E.")
        if cases[case_name]["grasp_vertices"] != first_grasp:
            failures.append(f"{case_name}: grasp vertices differ from high_k_high_E.")

    high_target = cases["high_k_high_E"]["metrics"]["target_error_norm"]
    low_target = cases["low_k_high_E"]["metrics"]["target_error_norm"]
    low_E_target = cases["high_k_low_E"]["metrics"]["target_error_norm"]
    high_strain = cases["high_k_high_E"]["metrics"]["strain_rms"]
    low_E_strain = cases["high_k_low_E"]["metrics"]["strain_rms"]

    if high_target > thresholds["high_k_high_E_target_error_norm_max"]:
        failures.append(f"high_k_high_E target_error_norm {high_target:.6g} is too high.")
    if low_target < thresholds["low_k_high_E_target_error_norm_min"]:
        failures.append(f"low_k_high_E target_error_norm {low_target:.6g} is too low.")
    target_ratio = low_target / max(high_target, 1.0e-6)
    if target_ratio < thresholds["low_over_high_target_error_norm_ratio_min"]:
        failures.append(f"low/high target_error_norm ratio {target_ratio:.6g} is too low.")
    if high_strain > thresholds["high_k_high_E_strain_rms_max"]:
        failures.append(f"high_k_high_E strain_rms {high_strain:.6g} is too high.")
    if low_E_strain < thresholds["high_k_low_E_strain_rms_min"]:
        failures.append(f"high_k_low_E strain_rms {low_E_strain:.6g} is too low.")
    strain_ratio = low_E_strain / max(high_strain, 1.0e-6)
    if strain_ratio < thresholds["low_E_over_high_E_strain_rms_ratio_min"]:
        failures.append(f"low-E/high-E strain_rms ratio {strain_ratio:.6g} is too low.")
    if low_E_target > thresholds["high_k_low_E_target_error_norm_max"]:
        failures.append(f"high_k_low_E target_error_norm {low_E_target:.6g} is too high.")

    return failures


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_dir = _run_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    _init_genesis(args)

    case_results = {}
    for case_name, settings in _case_settings(args).items():
        case_results[case_name] = _simulate_case(case_name, settings, args, output_dir)

    payload = _metrics_payload(args, output_dir, case_results)
    artifacts = payload["artifacts"]
    _write_json(Path(artifacts["metrics_json"]), payload)
    _write_report(Path(artifacts["validation_report"]), payload)

    failures = _collect_failures(payload)
    if failures:
        raise RuntimeError("Validation failed:\n" + "\n".join(f"- {failure}" for failure in failures))


if __name__ == "__main__":
    main()
