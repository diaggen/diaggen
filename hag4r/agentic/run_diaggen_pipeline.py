from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hag4r.agentic.run_config import DEFAULT_RUN_CONFIG_PATH, load_run_config
from hag4r.tools.genesis.diagnostic_timing import (
    DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
    DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
    DIAGNOSTIC_SCENE_TIMESTEP_S,
)
from hag4r.tools.genesis.live_protocol import DEFAULT_READY_TIMEOUT_S
from hag4r.agentic.runtime_state import (
    build_worker_gpu_binding,
    seed_full_image_state,
    seed_post_mesh_processing_diagnostics_state,
    state_path,
    worker_id_from_environment,
)


RUNTIME_KIND = "codex_skill"
RUNTIME_ENTRYPOINT = "run-diaggen-pipeline"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_outputs_directory(*, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    outputs_path = root / "outputs"
    if outputs_path.is_symlink() and not outputs_path.is_dir():
        raise NotADirectoryError(f"repo outputs symlink does not resolve to a directory: {outputs_path}")
    if outputs_path.exists() and not outputs_path.is_dir():
        raise NotADirectoryError(f"repo outputs path is not a directory: {outputs_path}")
    outputs_path.mkdir(parents=True, exist_ok=True)
    return outputs_path


def _build_mesh_extra_args(args: argparse.Namespace) -> tuple[str, ...]:
    extra_args: list[str] = []
    forwarded_flags = (
        ("part_target_faces", "--part_target_faces"),
        ("part_reduction", "--part_reduction"),
        ("part_floater_face_ratio", "--part_floater_face_ratio"),
        ("part_min_component_faces", "--part_min_component_faces"),
        ("global_target_faces", "--global_target_faces"),
        ("decimate_target_reduction", "--decimate_target_reduction"),
        ("sdf_grid_res", "--sdf_grid_res"),
        ("sdf_bbox_padding", "--sdf_bbox_padding"),
        ("sdf_epsilon_voxels", "--sdf_epsilon_voxels"),
        ("sdf_sign_method", "--sdf_sign_method"),
    )
    for attr_name, flag in forwarded_flags:
        value = getattr(args, attr_name)
        if value is not None:
            extra_args.extend([flag, str(value)])
    extra_args.extend(args.mesh_extra_arg or [])
    return tuple(extra_args)


def _validate_outputs_run_root(run_root: Path, *, repo_root: Path) -> Path:
    lexical_root = repo_root.expanduser().resolve()
    lexical_path = run_root.expanduser()
    if not lexical_path.is_absolute():
        lexical_path = lexical_root / lexical_path
    lexical_path = lexical_path.absolute()
    outputs_root = (lexical_root / "outputs").absolute()
    try:
        lexical_path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(
            "run-diaggen-pipeline requires --run_root to be under repo-local outputs/; "
            f"got {run_root}"
        ) from exc
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=RUNTIME_ENTRYPOINT,
        description=(
            "Initialize one HAG4R run-diaggen-pipeline Codex skill run. "
            "This script prepares state and paths only; Codex, not Python, runs the "
            "orchestrator and stage agents from .agents/skills/run-diaggen-pipeline."
        ),
    )
    parser.add_argument("--source_image", type=Path, help="Raw RGB image for the full image-to-asset path.")
    parser.add_argument(
        "--diagnostics_revision_snapshot_dir",
        type=Path,
        help="Completed revision snapshot directory for diagnostics-only post-mesh-processing analysis.",
    )
    parser.add_argument(
        "--diagnostics-stage-only",
        "--diagnostics_stage_only",
        dest="diagnostics_stage_only",
        action="store_true",
        help="Plan only Genesis diagnostics for a revision snapshot; do not plan final export.",
    )
    parser.add_argument(
        "--diagnostics-reroute-and-export",
        "--diagnostics_reroute_and_export",
        dest="diagnostics_reroute_and_export",
        action="store_true",
        help=(
            "For a revision snapshot, permit diagnostic repair reroutes and then "
            "post-mesh texture plus final export."
        ),
    )
    parser.add_argument(
        "--require-new-run-root",
        action="store_true",
        help=(
            "For diagnostics-only initialization, fail if the resolved run root already "
            "exists instead of reusing it."
        ),
    )
    parser.add_argument("--fidelity", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--output_tag")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--run_root", type=Path, default=Path("outputs/agentic_asset_refinement"))
    parser.add_argument(
        "--terminate_after_stage",
        choices=("image_cleanup", "omnipart", "material_inference"),
        help="Terminate a validation run after the named stage instead of completing the full pipeline.",
    )
    parser.add_argument(
        "--config",
        "--run_config",
        dest="config",
        type=Path,
        default=DEFAULT_RUN_CONFIG_PATH,
        help="YAML run config under configs/ controlling Genesis diagnostics enablement.",
    )
    parser.add_argument("--object_description", default="")
    parser.add_argument("--user_hints", default="")
    parser.add_argument(
        "--sim-diagnostics-max-runs",
        "--sim_diagnostics_max_runs",
        dest="sim_diagnostics_max_runs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--sim-diagnostics-max-episodes",
        "--sim_diagnostics_max_episodes",
        dest="sim_diagnostics_max_episodes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sim-diagnostics-max-actions-per-episode",
        "--sim_diagnostics_max_actions_per_episode",
        dest="sim_diagnostics_max_actions_per_episode",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sim-diagnostics-report-path",
        "--sim_diagnostics_report_path",
        dest="sim_diagnostics_report_path",
        type=Path,
    )
    parser.add_argument(
        "--sim-diagnostics-episode-timeout-s",
        "--sim_diagnostics_episode_timeout_s",
        dest="sim_diagnostics_episode_timeout_s",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--sim-diagnostics-live-host",
        "--sim_diagnostics_live_host",
        dest="sim_diagnostics_live_host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--sim-diagnostics-live-port",
        "--sim_diagnostics_live_port",
        dest="sim_diagnostics_live_port",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sim-diagnostics-live-ready-timeout-s",
        "--sim_diagnostics_live_ready_timeout_s",
        dest="sim_diagnostics_live_ready_timeout_s",
        type=int,
        default=int(DEFAULT_READY_TIMEOUT_S),
    )
    parser.add_argument(
        "--sim-diagnostics-live-heartbeat-ms",
        "--sim_diagnostics_live_heartbeat_ms",
        dest="sim_diagnostics_live_heartbeat_ms",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--sim-diagnostics-live-client-lease-timeout-ms",
        "--sim_diagnostics_live_client_lease_timeout_ms",
        dest="sim_diagnostics_live_client_lease_timeout_ms",
        type=int,
        default=30000,
    )
    parser.add_argument("--genesis-root", "--genesis_root", dest="genesis_root", type=Path)
    parser.add_argument("--genesis-env-path", "--genesis_env_path", dest="genesis_env_path", type=Path)
    parser.add_argument("--genesis-live-command", "--genesis_live_command", dest="genesis_live_command")
    parser.add_argument(
        "--sim-diagnostics-probe-max-vertices",
        "--sim_diagnostics_probe_max_vertices",
        dest="sim_diagnostics_probe_max_vertices",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sim-diagnostics-probe-max-distance-m",
        "--sim_diagnostics_probe_max_distance_m",
        dest="sim_diagnostics_probe_max_distance_m",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--sim-diagnostics-probe-max-speed-m-s",
        "--sim_diagnostics_probe_max_speed_m_s",
        dest="sim_diagnostics_probe_max_speed_m_s",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sim-diagnostics-probe-max-duration-steps",
        "--sim_diagnostics_probe_max_duration_steps",
        dest="sim_diagnostics_probe_max_duration_steps",
        type=int,
        default=DEFAULT_DIAGNOSTIC_SIMULATE_STEPS,
        help=(
            f"Maximum resume/probe simulation-step count; must cover one default "
            f"{DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S:g}s diagnostic simulate window "
            f"({DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S:g}s / {DIAGNOSTIC_SCENE_TIMESTEP_S:g}s = "
            f"{DEFAULT_DIAGNOSTIC_SIMULATE_STEPS} steps)."
        ),
    )
    parser.add_argument("--material_params_json", type=Path)
    parser.add_argument("--object_description_path", type=Path)
    parser.add_argument("--part_target_faces", type=int)
    parser.add_argument("--part_reduction", type=float)
    parser.add_argument("--part_floater_face_ratio", type=float)
    parser.add_argument("--part_min_component_faces", type=int)
    parser.add_argument("--global_target_faces", type=int)
    parser.add_argument("--decimate_target_reduction", type=float)
    parser.add_argument("--sdf_grid_res", type=int)
    parser.add_argument("--sdf_bbox_padding", type=float)
    parser.add_argument("--sdf_epsilon_voxels", type=float)
    parser.add_argument("--sdf_sign_method")
    parser.add_argument("--mesh_extra_arg", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    initializer_profile_clock = None
    try:
        from hag4r.agentic.runtime_profiler import capture_clock

        initializer_profile_clock = capture_clock()
    except Exception:
        # Profiling is optional observability and must not affect initialization.
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    primary_inputs = [
        args.source_image is not None,
        args.diagnostics_revision_snapshot_dir is not None,
    ]
    if sum(primary_inputs) != 1:
        parser.error("provide exactly one primary input: --source_image or --diagnostics_revision_snapshot_dir")
    if args.diagnostics_stage_only and args.diagnostics_revision_snapshot_dir is None:
        parser.error("--diagnostics-stage-only requires --diagnostics_revision_snapshot_dir")
    if (
        args.diagnostics_reroute_and_export
        and args.diagnostics_revision_snapshot_dir is None
    ):
        parser.error(
            "--diagnostics-reroute-and-export requires --diagnostics_revision_snapshot_dir"
        )
    if args.diagnostics_stage_only and args.diagnostics_reroute_and_export:
        parser.error(
            "--diagnostics-stage-only and --diagnostics-reroute-and-export are mutually exclusive"
        )
    if args.require_new_run_root and args.diagnostics_revision_snapshot_dir is None:
        parser.error("--require-new-run-root requires --diagnostics_revision_snapshot_dir")
    if args.terminate_after_stage is not None and args.source_image is None:
        parser.error("--terminate_after_stage currently supports only --source_image full-image runs")

    root = _repo_root()
    ensure_outputs_directory(repo_root=root)
    args.run_root = _validate_outputs_run_root(args.run_root, repo_root=root)
    run_config = load_run_config(args.config, repo_root=root)
    diagnostics_only = args.diagnostics_revision_snapshot_dir is not None
    if diagnostics_only and not run_config.genesis_diagnostics.enable:
        parser.error("--diagnostics_revision_snapshot_dir requires genesis_diagnostics.enable=true in the run config")
    genesis_live_command = (
        run_config.genesis_diagnostics.genesis_live_command
        if args.genesis_live_command is None
        else args.genesis_live_command.strip()
    )
    if not genesis_live_command:
        parser.error("--genesis-live-command must be non-empty")
    worker_id = worker_id_from_environment(os.environ, run_id=args.run_id)
    gpu_binding = build_worker_gpu_binding(os.environ)
    seed_kwargs = {
        "run_config": run_config,
        "runs_root": args.run_root,
        "fidelity": args.fidelity,
        "output_tag": args.output_tag,
        "object_description": args.object_description,
        "object_description_path": args.object_description_path,
        "material_params_json": args.material_params_json,
        "user_hints": args.user_hints,
        "mesh_extra_args": _build_mesh_extra_args(args),
        "sim_diagnostics_max_runs": args.sim_diagnostics_max_runs,
        "sim_diagnostics_max_episodes": args.sim_diagnostics_max_episodes,
        "sim_diagnostics_max_actions_per_episode": args.sim_diagnostics_max_actions_per_episode,
        "sim_diagnostics_report_path": args.sim_diagnostics_report_path,
        "sim_diagnostics_episode_timeout_s": args.sim_diagnostics_episode_timeout_s,
        "sim_diagnostics_live_host": args.sim_diagnostics_live_host,
        "sim_diagnostics_live_port": args.sim_diagnostics_live_port,
        "sim_diagnostics_live_ready_timeout_s": args.sim_diagnostics_live_ready_timeout_s,
        "sim_diagnostics_live_heartbeat_ms": args.sim_diagnostics_live_heartbeat_ms,
        "sim_diagnostics_live_client_lease_timeout_ms": args.sim_diagnostics_live_client_lease_timeout_ms,
        "sim_diagnostics_probe_max_vertices": args.sim_diagnostics_probe_max_vertices,
        "sim_diagnostics_probe_max_distance_m": args.sim_diagnostics_probe_max_distance_m,
        "sim_diagnostics_probe_max_speed_m_s": args.sim_diagnostics_probe_max_speed_m_s,
        "sim_diagnostics_probe_max_duration_steps": args.sim_diagnostics_probe_max_duration_steps,
        "genesis_root": args.genesis_root or run_config.genesis_diagnostics.genesis_root,
        "genesis_env_path": args.genesis_env_path or run_config.genesis_diagnostics.genesis_env_path,
        "genesis_live_command": genesis_live_command,
        "worker_id": worker_id,
        "gpu_binding": gpu_binding,
        "runtime_kind": RUNTIME_KIND,
        "runtime_entrypoint": RUNTIME_ENTRYPOINT,
    }
    if diagnostics_only:
        diagnostics_seed_kwargs = {
            key: seed_kwargs[key]
            for key in (
                "run_config",
                "runs_root",
                "output_tag",
                "sim_diagnostics_max_runs",
                "sim_diagnostics_max_episodes",
                "sim_diagnostics_max_actions_per_episode",
                "sim_diagnostics_report_path",
                "sim_diagnostics_episode_timeout_s",
                "sim_diagnostics_live_host",
                "sim_diagnostics_live_port",
                "sim_diagnostics_live_ready_timeout_s",
                "sim_diagnostics_live_heartbeat_ms",
                "sim_diagnostics_live_client_lease_timeout_ms",
                "sim_diagnostics_probe_max_vertices",
                "sim_diagnostics_probe_max_distance_m",
                "sim_diagnostics_probe_max_speed_m_s",
                "sim_diagnostics_probe_max_duration_steps",
                "genesis_root",
                "genesis_env_path",
                "genesis_live_command",
                "worker_id",
                "gpu_binding",
                "runtime_kind",
                "runtime_entrypoint",
            )
        }
        state = seed_post_mesh_processing_diagnostics_state(
            diagnostics_revision_snapshot_dir=args.diagnostics_revision_snapshot_dir,
            diagnostics_stage_only=args.diagnostics_stage_only,
            diagnostics_reroute_and_export=args.diagnostics_reroute_and_export,
            require_new_run_root=args.require_new_run_root,
            run_id=args.run_id,
            **diagnostics_seed_kwargs,
        )
    else:
        state = seed_full_image_state(
            source_image=args.source_image,
            run_id=args.run_id,
            terminate_after_stage=args.terminate_after_stage,
            **seed_kwargs,
        )
    if initializer_profile_clock is not None:
        try:
            from hag4r.agentic.runtime_profiler import start_automatic_profiling

            start_automatic_profiling(
                state["run_root"],
                initializer_started=initializer_profile_clock,
            )
        except Exception:
            # A broken or unavailable profiler must leave the initialized run usable.
            pass
    result = {
        "status": "initialized",
        "runtime_kind": RUNTIME_KIND,
        "runtime_entrypoint": RUNTIME_ENTRYPOINT,
        "run_root": str(state["run_root"]),
        "state_path": str(state_path(state["run_root"])),
        "codex_skill_path": ".agents/skills/run-diaggen-pipeline/SKILL.md",
        "next_action": (
            "Run the Codex orchestrator skill at .agents/skills/run-diaggen-pipeline/SKILL.md "
            "for this one-asset state. Python must only be used for deterministic tools/scripts."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
