#!/usr/bin/env python3
"""SAP-only gravity-drop diagnostic for the exact heterogeneous Clawhauser tet mesh.

This deliberately contains no robot, constraint, force, or action API.  Its purpose is
to make a failing FEM/SAP floor-contact run as inspectable as a successful one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ASSET_ROOT = Path("/home/eric/research/HAG4R/outputs/run_pipeline/mesh_fidelity_clawhauser_20260802")
MESH_NAME = "final_mesh.mesh"
MATERIAL_NAME = "heterogeneous_params.npz"
EXPECTED_VERTEX_COUNT = 1840
EXPECTED_TET_COUNT = 6850
DEFAULT_OUTPUT_ROOT = Path("outputs/clawhauser_tetmesh_drop")
DEFAULT_DT = 1.0 / 60.0
DEFAULT_SUBSTEPS = 3
DEFAULT_DROP_HEIGHT = 0.25
FLOOR_Z = 0.0
CONTACT_MARGIN = 1.0e-3
CAMERA_RESOLUTION = (640, 480)
CAMERA_OFFSET = (0.42, -0.52, 0.30)
CAMERA_UP = (0.0, 0.0, 1.0)
CAMERA_FOV = 38
CAMERA_GUI = False


def normalize_positions(positions: np.ndarray, *, name: str, expected_vertices: int | None = None) -> np.ndarray:
    """Normalize a FEM position tensor exported as (N, 3) or (1, N, 3)."""
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} positions must have shape (N, 3) or (1, N, 3), got {array.shape}")
    if expected_vertices is not None and array.shape != (expected_vertices, 3):
        raise ValueError(f"{name} positions must have shape ({expected_vertices}, 3), got {array.shape}")
    return array


def normalize_tets(tets: np.ndarray, *, expected_tets: int | None = None) -> np.ndarray:
    """Normalize and validate tetrahedral connectivity before determinant evaluation."""
    array = np.asarray(tets, dtype=np.int64)
    if array.ndim != 2 or array.shape[1:] != (4,):
        raise ValueError(f"tet connectivity must have shape (T, 4), got {array.shape}")
    if expected_tets is not None and array.shape != (expected_tets, 4):
        raise ValueError(f"tet connectivity must have shape ({expected_tets}, 4), got {array.shape}")
    return array


def tet_j_ratios(rest_positions: np.ndarray, current_positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return signed det(D_current) / det(D_rest) for every tetrahedron."""
    rest_positions = normalize_positions(rest_positions, name="rest")
    current_positions = normalize_positions(current_positions, name="current")
    tets = normalize_tets(tets)
    if rest_positions.shape != current_positions.shape or np.any(tets < 0) or np.any(tets >= len(rest_positions)):
        raise ValueError("rest/current positions or tet indices are inconsistent")
    rest = rest_positions[tets]
    current = current_positions[tets]
    d_rest = np.stack((rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0], rest[:, 3] - rest[:, 0]), axis=-1)
    rest_det = np.linalg.det(d_rest)
    if not np.isfinite(rest_det).all() or np.any(rest_det == 0.0):
        raise ValueError("rest tetrahedra must have finite, non-zero signed determinants")
    d_current = np.stack(
        (current[:, 1] - current[:, 0], current[:, 2] - current[:, 0], current[:, 3] - current[:, 0]), axis=-1
    )
    return np.linalg.det(d_current) / rest_det


def classify_health(
    positions: np.ndarray, velocities: np.ndarray, j_ratios: np.ndarray, near_j_threshold: float
) -> dict[str, Any]:
    """Apply the run's finite/inversion/near-inversion definitions to one step."""
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(j_ratios).all())
    min_j = float(np.min(j_ratios)) if j_ratios.size and np.isfinite(j_ratios).all() else None
    inversion = bool(finite and min_j is not None and min_j <= 0.0)
    near_inversion = bool(finite and not inversion and min_j is not None and min_j < near_j_threshold)
    return {
        "finite": finite,
        "inversion": inversion,
        "near_inversion": near_inversion,
        "healthy": bool(finite and not inversion),
        "min_j": min_j,
        "min_j_tet": int(np.argmin(j_ratios)) if finite and j_ratios.size else None,
    }


def json_safe(value: Any) -> Any:
    """Convert NumPy values recursively and encode non-finite/missing values as null."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def create_output_dir(requested: str | None) -> Path:
    """Create an empty explicit output dir, or a non-overwriting timestamped default."""
    if requested is None:
        output_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    output_dir = Path(requested)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to mix artifacts into non-empty output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Empty directory for artifacts (default: timestamped outputs/ run).")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--drop-height", type=float, default=DEFAULT_DROP_HEIGHT)
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate and target render cadence.")
    parser.add_argument("--frame-stride", type=int, default=None, help="Simulation steps between rendered video frames.")
    parser.add_argument("--near-j-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Recorded for reproducibility; this fixed scene is deterministic.")
    parser.add_argument("--headless", action="store_true", help="Disable the interactive Genesis viewer.")
    args = parser.parse_args()
    if args.duration <= 0.0 or args.dt <= 0.0 or args.substeps <= 0 or args.drop_height <= 0.0:
        parser.error("duration, dt, substeps, and drop-height must be positive")
    if args.fps <= 0 or (args.frame_stride is not None and args.frame_stride <= 0):
        parser.error("fps and frame-stride must be positive")
    if args.near_j_threshold <= 0.0:
        parser.error("near-j-threshold must be positive")
    return args


def _save_frame(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def set_tracking_camera(camera: Any, positions: np.ndarray) -> np.ndarray:
    """Center the fixed-offset camera on the current FEM mesh for every rendered frame."""
    centroid = np.mean(positions, axis=0)
    camera.set_pose(pos=centroid + np.asarray(CAMERA_OFFSET), lookat=centroid, up=CAMERA_UP)
    return centroid


def _event(step: int, sim_time: float, health: dict[str, Any], min_z: float, max_speed: float) -> dict[str, Any]:
    return {
        "step": int(step),
        "time": float(sim_time),
        "min_j": health["min_j"],
        "min_j_tet": health["min_j_tet"],
        "min_z": float(min_z),
        "max_speed": float(max_speed),
    }


def main() -> int:
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    metrics_path = output_dir / "metrics.json"
    telemetry_path = output_dir / "telemetry.npz"
    video_path = output_dir / "drop.mp4"
    metrics: dict[str, Any] = {
        "status": "setup_failure",
        "termination_reason": "not_started",
        "asset": {"root": ASSET_ROOT, "mesh": ASSET_ROOT / MESH_NAME, "materials": ASSET_ROOT / MATERIAL_NAME},
        "expected_vertex_count": EXPECTED_VERTEX_COUNT,
        "expected_tet_count": EXPECTED_TET_COUNT,
        "precision": "64",
        "duration": args.duration,
        "dt": args.dt,
        "substeps": args.substeps,
        "gravity": [0.0, 0.0, -9.81],
        "floor_height": FLOOR_Z,
        "drop_height": args.drop_height,
        "drop_pose": [0.0, 0.0, args.drop_height],
        "seed": args.seed,
        "near_j_threshold": args.near_j_threshold,
        "sap_contact_only": True,
        "sap_floor_type": "tet",
        "legacy_fallback_used": False,
        "collision_plane_added": False,
        "floor_z": FLOOR_Z,
        "contact_geometric_margin": CONTACT_MARGIN,
        "render": {
            "video": video_path,
            "frames_dir": frames_dir,
            "fps": args.fps,
            "frame_stride": args.frame_stride,
            "camera": {
                "resolution": CAMERA_RESOLUTION,
                "mode": "centroid_tracking",
                "offset": CAMERA_OFFSET,
                "up": CAMERA_UP,
                "fov": CAMERA_FOV,
                "GUI": CAMERA_GUI,
            },
        },
        "frame_manifest": {},
    }
    telemetry: dict[str, list[Any]] = {
        key: []
        for key in (
            "step", "time", "min_j", "min_j_tet", "finite", "inversion", "near_inversion", "min_z", "max_speed",
            "contact_observed", "centroid_x", "centroid_y", "centroid_z", "bbox_min_x", "bbox_min_y", "bbox_min_z",
            "bbox_max_x", "bbox_max_y", "bbox_max_z",
        )
    }
    cam = None
    recording = False
    exit_code = 1

    try:
        import genesis as gs
        from genesis.utils.misc import tensor_to_array
        from genesis.utils.volumetric_mesh import load_tet_mesh

        np.random.seed(args.seed)
        mesh_path = ASSET_ROOT / MESH_NAME
        material_path = ASSET_ROOT / MATERIAL_NAME
        if not mesh_path.is_file() or not material_path.is_file():
            raise FileNotFoundError(f"locked Clawhauser asset is unavailable under {ASSET_ROOT}")
        raw_mesh = load_tet_mesh(mesh_path)
        if raw_mesh.verts.shape != (EXPECTED_VERTEX_COUNT, 3) or raw_mesh.tets.shape != (EXPECTED_TET_COUNT, 4):
            raise ValueError(f"unexpected Clawhauser mesh shape: {raw_mesh.verts.shape}, {raw_mesh.tets.shape}")
        with np.load(material_path) as material_npz:
            if (
                material_npz["tet_E_nu"].shape != (EXPECTED_TET_COUNT, 2)
                or material_npz["tet_density"].shape != (EXPECTED_TET_COUNT,)
                or material_npz["tet_part_labels"].shape != (EXPECTED_TET_COUNT,)
            ):
                raise ValueError("heterogeneous NPZ does not match the exact 6,850-tet material contract")

        gs.init(backend=gs.gpu, precision="64")
        frame_stride = args.frame_stride or max(1, round((1.0 / args.dt) / args.fps))
        metrics["frame_stride"] = frame_stride
        metrics["render"]["frame_stride"] = frame_stride
        metrics["render"]["cadence_seconds"] = frame_stride * args.dt
        sap_options = gs.options.SAPCouplerOptions(fem_floor_contact_type="tet")
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=args.dt, substeps=args.substeps, gravity=(0.0, 0.0, -9.81), floor_height=FLOOR_Z
            ),
            fem_options=gs.options.FEMOptions(use_implicit_solver=True),
            coupler_options=sap_options,
            viewer_options=gs.options.ViewerOptions(camera_pos=CAMERA_OFFSET, camera_lookat=(0.0, 0.0, 0.0)),
            show_viewer=not args.headless,
            show_FPS=False,
        )
        entity = scene.add_entity(
            morph=gs.morphs.TetMesh(file=str(mesh_path), pos=tuple(metrics["drop_pose"])),
            material=gs.materials.FEM.Elastic(
                model="linear_corotated",
                heterogeneous=gs.materials.FEM.HeterogeneousMaterial(file=str(material_path)),
            ),
        )
        cam = scene.add_camera(
            res=CAMERA_RESOLUTION, pos=CAMERA_OFFSET, lookat=(0.0, 0.0, 0.0), fov=CAMERA_FOV, GUI=CAMERA_GUI
        )
        scene.build()
        if not isinstance(scene.sim.coupler.options, gs.options.SAPCouplerOptions) or scene.sim.coupler.options.fem_floor_contact_type != "tet":
            raise RuntimeError("SAP tetrahedral floor contact was not configured")
        if entity.n_vertices != EXPECTED_VERTEX_COUNT or entity.n_elements != EXPECTED_TET_COUNT:
            raise RuntimeError("Genesis did not retain the exact Clawhauser volumetric topology")

        rest = normalize_positions(tensor_to_array(entity.init_positions), name="rest", expected_vertices=EXPECTED_VERTEX_COUNT)
        tets = normalize_tets(entity.elems, expected_tets=EXPECTED_TET_COUNT)
        initial_state = entity.get_state(track_grad=False)
        initial_pos = normalize_positions(tensor_to_array(initial_state.pos), name="initial", expected_vertices=EXPECTED_VERTEX_COUNT)
        initial_vel = normalize_positions(tensor_to_array(initial_state.vel), name="initial velocity", expected_vertices=EXPECTED_VERTEX_COUNT)
        initial_j = tet_j_ratios(rest, initial_pos, tets)
        initial_min_z = float(np.min(initial_pos[:, 2]))
        if initial_min_z <= FLOOR_Z:
            raise RuntimeError(f"drop begins at or below SAP floor: initial_min_z={initial_min_z}")
        metrics.update(
            {
                "observed_vertex_count": entity.n_vertices,
                "observed_tet_count": entity.n_elements,
                "initial_min_z": initial_min_z,
                "fem": {"use_implicit_solver": True, "model": "linear_corotated", "heterogeneous_material": str(material_path)},
                "sap_options": sap_options.model_dump(),
            }
        )

        cam.start_recording()
        recording = True
        set_tracking_camera(cam, initial_pos)
        rgb, _, _, _ = cam.render()
        rendered_frame_count = 1
        initial_frame = frames_dir / "frame_000000.png"
        _save_frame(rgb, initial_frame)
        metrics["frame_manifest"]["initial"] = {"path": initial_frame, "step": 0, "time": 0.0}
        first_contact: dict[str, Any] | None = None
        first_near: dict[str, Any] | None = None
        first_inversion: dict[str, Any] | None = None
        last_pre_contact_frame: dict[str, Any] | None = {"path": initial_frame, "step": 0, "time": 0.0}
        post_contact_step: int | None = None
        steps_requested = int(np.ceil(args.duration / args.dt))
        for step in range(1, steps_requested + 1):
            scene.step()
            state = entity.get_state(track_grad=False)
            positions = normalize_positions(tensor_to_array(state.pos), name="current", expected_vertices=EXPECTED_VERTEX_COUNT)
            velocities = normalize_positions(tensor_to_array(state.vel), name="current velocity", expected_vertices=EXPECTED_VERTEX_COUNT)
            j_ratios = tet_j_ratios(rest, positions, tets)
            health = classify_health(positions, velocities, j_ratios, args.near_j_threshold)
            min_z = float(np.min(positions[:, 2])) if np.isfinite(positions).all() else float("nan")
            max_speed = float(np.max(np.linalg.norm(velocities, axis=1))) if np.isfinite(velocities).all() else float("nan")
            centroid = np.mean(positions, axis=0) if np.isfinite(positions).all() else np.full(3, np.nan)
            bbox_min = np.min(positions, axis=0) if np.isfinite(positions).all() else np.full(3, np.nan)
            bbox_max = np.max(positions, axis=0) if np.isfinite(positions).all() else np.full(3, np.nan)
            sim_time = step * args.dt
            contact = bool(health["healthy"] and min_z <= FLOOR_Z + CONTACT_MARGIN)
            for key, value in {
                "step": step,
                "time": sim_time,
                "min_j": np.nan if health["min_j"] is None else health["min_j"],
                "min_j_tet": -1 if health["min_j_tet"] is None else health["min_j_tet"],
                "finite": health["finite"],
                "inversion": health["inversion"],
                "near_inversion": health["near_inversion"],
                "min_z": min_z,
                "max_speed": max_speed,
                "contact_observed": contact,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
                "bbox_min_x": bbox_min[0],
                "bbox_min_y": bbox_min[1],
                "bbox_min_z": bbox_min[2],
                "bbox_max_x": bbox_max[0],
                "bbox_max_y": bbox_max[1],
                "bbox_max_z": bbox_max[2],
            }.items():
                telemetry[key].append(value)
            event = _event(step, sim_time, health, min_z, max_speed)
            if health["near_inversion"] and first_near is None:
                first_near = event
            newly_observed_contact = bool(contact and first_contact is None)
            if newly_observed_contact:
                first_contact = event
                post_contact_step = step + max(1, round(0.25 / args.dt))
                if last_pre_contact_frame is not None:
                    metrics["frame_manifest"]["pre_contact"] = last_pre_contact_frame

            if not health["healthy"]:
                first_inversion = event if health["inversion"] else None
                first_nonfinite = event if not health["finite"] else None
                evidence_path = output_dir / f"failure_step_{step:06d}.npz"
                np.savez_compressed(
                    evidence_path,
                    positions=positions,
                    velocities=velocities,
                    j_ratios=j_ratios,
                    failing_tet=-1 if health["min_j_tet"] is None else health["min_j_tet"],
                    step=step,
                    time=sim_time,
                )
                metrics["terminal_evidence_path"] = evidence_path
                metrics["status"] = "inversion" if health["inversion"] else "nonfinite"
                metrics["termination_reason"] = "first tetrahedral inversion" if health["inversion"] else "non-finite FEM state"
                metrics["first_inversion"] = first_inversion
                metrics["first_nonfinite"] = first_nonfinite
                # Failure evidence is durable before a best-effort terminal render.
                try:
                    set_tracking_camera(cam, positions)
                    rgb, _, _, _ = cam.render()
                    rendered_frame_count += 1
                    frame_path = frames_dir / f"frame_{step:06d}.png"
                    _save_frame(rgb, frame_path)
                    metrics["frame_manifest"]["terminal_failure"] = {"path": frame_path, "step": step, "time": sim_time}
                except Exception as render_exc:
                    metrics["terminal_render_error"] = str(render_exc)
                break

            must_render = step % frame_stride == 0 or newly_observed_contact or step == post_contact_step or step == steps_requested
            if must_render:
                set_tracking_camera(cam, positions)
                rgb, _, _, _ = cam.render()
                rendered_frame_count += 1
                frame_path = frames_dir / f"frame_{step:06d}.png"
                _save_frame(rgb, frame_path)
                frame_info = {"path": frame_path, "step": step, "time": sim_time}
                if first_contact is None:
                    last_pre_contact_frame = frame_info
                if contact and "first_contact" not in metrics["frame_manifest"]:
                    metrics["frame_manifest"]["first_contact"] = frame_info
                if post_contact_step == step:
                    metrics["frame_manifest"]["post_contact"] = frame_info
        else:
            metrics["status"] = "completed"
            metrics["termination_reason"] = "completed requested duration"
            exit_code = 0

        completed_steps = len(telemetry["step"])
        final_event = (
            _event(
                telemetry["step"][-1], telemetry["time"][-1],
                {"min_j": telemetry["min_j"][-1], "min_j_tet": telemetry["min_j_tet"][-1]},
                telemetry["min_z"][-1], telemetry["max_speed"][-1],
            )
            if completed_steps
            else None
        )
        metrics.update(
            {
                "completed_steps": completed_steps,
                "simulated_time": 0.0 if not completed_steps else telemetry["time"][-1],
                "initial_min_j": float(np.min(initial_j)),
                "final": final_event,
                "global_min_j": None if not completed_steps else float(np.nanmin(telemetry["min_j"])),
                "global_min_z": None if not completed_steps else float(np.nanmin(telemetry["min_z"])),
                "global_max_speed": None if not completed_steps else float(np.nanmax(telemetry["max_speed"])),
                "rendered_frame_count": rendered_frame_count,
                "video_duration_expected_seconds": rendered_frame_count / args.fps,
                "first_near_inversion": first_near,
                "first_inversion": first_inversion,
                "first_nonfinite": locals().get("first_nonfinite"),
                "first_contact": first_contact,
                "final_min_z": None if not completed_steps else telemetry["min_z"][-1],
            }
        )
        if "final" in metrics["frame_manifest"]:
            del metrics["frame_manifest"]["final"]
        if completed_steps and metrics["status"] == "completed":
            final_frame = frames_dir / f"frame_{telemetry['step'][-1]:06d}.png"
            metrics["frame_manifest"]["final"] = {"path": final_frame, "step": telemetry["step"][-1], "time": telemetry["time"][-1]}
    except Exception as exc:
        if metrics["status"] not in {"inversion", "nonfinite"}:
            metrics["status"] = "setup_failure" if not telemetry["step"] else "render_failure"
            metrics["termination_reason"] = type(exc).__name__
            metrics["error"] = str(exc)
        else:
            metrics["terminal_render_error"] = str(exc)
    finally:
        if cam is not None and recording:
            try:
                cam.stop_recording(save_to_filename=str(video_path), fps=args.fps)
            except Exception as exc:
                if metrics["status"] in {"inversion", "nonfinite"}:
                    metrics["video_finalization_error"] = str(exc)
                else:
                    metrics["status"] = "render_failure"
                    metrics["termination_reason"] = "video finalization failed"
                    metrics["error"] = str(exc)
                exit_code = 1
        if not video_path.is_file() or video_path.stat().st_size == 0:
            if metrics["status"] in {"inversion", "nonfinite"}:
                metrics["video_finalization_error"] = "MP4 artifact is missing or empty"
                metrics["artifact_missing"] = str(video_path)
            else:
                metrics["status"] = "render_failure"
                metrics["termination_reason"] = "MP4 artifact is missing or empty"
                exit_code = 1
        np.savez_compressed(**{key: np.asarray(value) for key, value in telemetry.items()}, file=telemetry_path)
        metrics["telemetry_path"] = telemetry_path
        metrics["video_exists"] = video_path.is_file()
        metrics["video_size_bytes"] = video_path.stat().st_size if video_path.is_file() else 0
        metrics_path.write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")

    print(f"Clawhauser SAP drop: {metrics['status']} ({metrics['termination_reason']})")
    print(f"Artifacts: {output_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
