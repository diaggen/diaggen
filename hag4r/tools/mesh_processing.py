from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hag4r.agentic.state import ArtifactRole, Stage, StageRunResult
from hag4r.tools.common import (
    EnvName,
    _artifact,
    _conda_module_argv,
    _dict_stage_result,
    _path_for_repo,
    _repo_root,
    _run_subprocess_stage,
)
from hag4r.tools.volumetric_meshing import (
    PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION,
    component_resolution_report_for_request,
    mesh_processing_render_paths,
    tet_component_resolution_paths,
)
from hag4r.tools.metric_mesh_scaling import (
    build_metric_scaling_metadata,
    mesh_volume_metadata,
    write_json,
)


def run_assign_params(
    *,
    part_labels_path: Path,
    inferred_params_path: Path,
    output_tag: str,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
    timeout_s: int = 1800,
) -> StageRunResult:
    root = repo_root or _repo_root()
    resolved_output_dir = _path_for_repo(output_dir, root) if output_dir is not None else root / "outputs/assign_params_to_prims" / output_tag
    expected_path = _path_for_repo(output_path, root) if output_path is not None else resolved_output_dir / "volumetric_params.npz"
    output_args: tuple[object, ...] = ("--output_path", expected_path)
    if output_dir is not None:
        output_args = (*output_args, "--output_dir", resolved_output_dir)
    return _run_subprocess_stage(
        name="hag4r_assign_params_to_prims",
        stage=Stage.PRIMITIVE_ASSIGNMENT,
        argv=_conda_module_argv(
            EnvName.HAG4R_TOOLS,
            "scripts.assign_params_to_prims",
            (
                "--part_labels",
                _path_for_repo(part_labels_path, root),
                "--inferred_params_path",
                _path_for_repo(inferred_params_path, root),
                *output_args,
            ),
        ),
        cwd=root,
        conda_env=EnvName.HAG4R_TOOLS,
        timeout_s=timeout_s,
        expected_artifacts=(_artifact(ArtifactRole.PARTWISE_PARAMS, expected_path, Stage.PRIMITIVE_ASSIGNMENT),),
        read_paths=(_path_for_repo(part_labels_path, root), _path_for_repo(inferred_params_path, root)),
        write_paths=(expected_path.parent,),
        log_dir=log_dir,
        repo_root=root,
    )


def _repo_relative(path: Path, repo_root: Path) -> str:
    candidate = path if path.is_absolute() else repo_root / path
    repo_outputs = repo_root / "outputs"
    try:
        output_relative = candidate.resolve().relative_to(repo_outputs.resolve())
        return str(Path("outputs") / output_relative)
    except ValueError:
        pass
    try:
        return str(candidate.absolute().relative_to(repo_root.absolute()))
    except ValueError:
        pass
    resolved = candidate.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _require_outputs_path(path: Path, repo_root: Path, *, label: str) -> None:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    repo_outputs = (repo_root / "outputs").resolve()
    try:
        resolved.relative_to(repo_outputs)
        return
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be under the repository outputs/ tree: {resolved}") from exc
    if not rel.parts or rel.parts[0] != "outputs":
        raise ValueError(f"{label} must be under outputs/: {rel}")


def write_per_part_volumetric_meshing_request(
    *,
    input_mesh_dir: Path,
    part_labels_path: Path,
    inferred_params_path: Path,
    partwise_params_path: Path,
    mesh_processing_plan: dict[str, Any],
    output_mesh: Path,
    heterogeneous_params_path: Path,
    metric_mesh_scaling_path: Path,
    volume_topology_path: Path,
    request_path: Path,
    surface_combination_strategy: str = "manifold_union",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or _repo_root()
    resolved_request_path = _path_for_repo(request_path, root)
    generated_paths = {
        "request_path": resolved_request_path,
        "output_mesh_path": _path_for_repo(output_mesh, root),
        "heterogeneous_params_path": _path_for_repo(heterogeneous_params_path, root),
        "metric_mesh_scaling_path": _path_for_repo(metric_mesh_scaling_path, root),
        "volume_topology_path": _path_for_repo(volume_topology_path, root),
    }
    for label, path in generated_paths.items():
        _require_outputs_path(path, root, label=label)
    payload = {
        "schema_version": PER_PART_VOLUME_MESHING_REQUEST_SCHEMA_VERSION,
        "input_mesh_dir": _repo_relative(_path_for_repo(input_mesh_dir, root), root),
        "part_labels_path": _repo_relative(_path_for_repo(part_labels_path, root), root),
        "inferred_params_path": _repo_relative(_path_for_repo(inferred_params_path, root), root),
        "partwise_params_path": _repo_relative(_path_for_repo(partwise_params_path, root), root),
        "mesh_processing_plan": mesh_processing_plan,
        "surface_combination_strategy": surface_combination_strategy,
        "output_mesh_path": _repo_relative(generated_paths["output_mesh_path"], root),
        "heterogeneous_params_path": _repo_relative(generated_paths["heterogeneous_params_path"], root),
        "metric_mesh_scaling_path": _repo_relative(generated_paths["metric_mesh_scaling_path"], root),
        "volume_topology_path": _repo_relative(generated_paths["volume_topology_path"], root),
    }
    resolved_request_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_monolithic_mesh(
    *,
    input_mesh_dir: Path,
    output_mesh: Path,
    fidelity: str = "medium",
    mesh_extra_args: tuple[str, ...] = (),
    part_labels_path: Path | None = None,
    inferred_params_path: Path | None = None,
    partwise_params_path: Path | None = None,
    mesh_processing_plan: dict[str, Any] | None = None,
    heterogeneous_params_path: Path | None = None,
    metric_mesh_scaling_path: Path | None = None,
    volume_topology_path: Path | None = None,
    request_path: Path | None = None,
    surface_combination_strategy: str = "manifold_union",
    repo_root: Path | None = None,
    log_dir: Path | None = None,
    timeout_s: int = 7200,
) -> StageRunResult:
    root = repo_root or _repo_root()
    resolved_output_mesh = _path_for_repo(output_mesh, root)
    resolved_input_mesh_dir = _path_for_repo(input_mesh_dir, root)
    if mesh_processing_plan is not None:
        missing = [
            name
            for name, value in {
                "part_labels_path": part_labels_path,
                "inferred_params_path": inferred_params_path,
                "partwise_params_path": partwise_params_path,
                "heterogeneous_params_path": heterogeneous_params_path,
                "metric_mesh_scaling_path": metric_mesh_scaling_path,
                "volume_topology_path": volume_topology_path,
                "request_path": request_path,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"per-part volumetric meshing is missing required paths: {', '.join(missing)}")
        assert part_labels_path is not None
        assert inferred_params_path is not None
        assert partwise_params_path is not None
        assert heterogeneous_params_path is not None
        assert metric_mesh_scaling_path is not None
        assert volume_topology_path is not None
        assert request_path is not None
        resolved_request_path = _path_for_repo(request_path, root)
        write_per_part_volumetric_meshing_request(
            input_mesh_dir=resolved_input_mesh_dir,
            part_labels_path=_path_for_repo(part_labels_path, root),
            inferred_params_path=_path_for_repo(inferred_params_path, root),
            partwise_params_path=_path_for_repo(partwise_params_path, root),
            mesh_processing_plan=mesh_processing_plan,
            output_mesh=resolved_output_mesh,
            heterogeneous_params_path=_path_for_repo(heterogeneous_params_path, root),
            metric_mesh_scaling_path=_path_for_repo(metric_mesh_scaling_path, root),
            volume_topology_path=_path_for_repo(volume_topology_path, root),
            request_path=resolved_request_path,
            surface_combination_strategy=surface_combination_strategy,
            repo_root=root,
        )
        render_paths = mesh_processing_render_paths(
            output_mesh_path=resolved_output_mesh,
            heterogeneous_params_path=_path_for_repo(heterogeneous_params_path, root),
        )
        expected = (
            _artifact(ArtifactRole.MONOLITHIC_MESH, resolved_output_mesh, Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.MONOLITHIC_PARAMS, _path_for_repo(heterogeneous_params_path, root), Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.REPORT, _path_for_repo(metric_mesh_scaling_path, root), Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.REPORT, _path_for_repo(volume_topology_path, root), Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.YOUNGS_MODULUS_RENDER, render_paths["target_E"], Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.DENSITY_RENDER, render_paths["target_density"], Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.PART_ID_RENDER, render_paths["target_part"], Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.PART_ID_RENDER, render_paths["source_part"], Stage.MESH_PROCESSING),
        )
        result = _run_subprocess_stage(
            name="hag4r_per_part_volumetric_monolithic_mesh",
            stage=Stage.MESH_PROCESSING,
            argv=_conda_module_argv(
                EnvName.HAG4R_TOOLS,
                "hag4r.mesh",
                ("--per_part_volume_meshing_request", resolved_request_path),
                repo_root=root,
            ),
            cwd=root,
            conda_env=EnvName.HAG4R_TOOLS,
            timeout_s=timeout_s,
            expected_artifacts=expected,
            read_paths=(
                resolved_input_mesh_dir,
                _path_for_repo(part_labels_path, root),
                _path_for_repo(inferred_params_path, root),
                _path_for_repo(partwise_params_path, root),
                resolved_request_path,
            ),
            write_paths=(
                resolved_output_mesh.parent,
                _path_for_repo(heterogeneous_params_path, root).parent,
                _path_for_repo(metric_mesh_scaling_path, root).parent,
                _path_for_repo(volume_topology_path, root).parent,
            ),
            env={"PYVISTA_OFF_SCREEN": "true", "MESA_GL_VERSION_OVERRIDE": "3.2"},
            log_dir=log_dir,
            repo_root=root,
        )
        pending_report = component_resolution_report_for_request(resolved_request_path, repo_root=root)
        if pending_report is not None and pending_report.get("status") == "needs_component_resolution":
            resolution_paths = tet_component_resolution_paths(resolved_output_mesh)
            return StageRunResult(
                name=result.name,
                stage=result.stage,
                success=False,
                returncode=result.returncode,
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                artifacts=(
                    _artifact(ArtifactRole.REPORT, resolution_paths.report_path, Stage.MESH_PROCESSING),
                    _artifact(ArtifactRole.REPORT, resolution_paths.candidate_npz_path, Stage.MESH_PROCESSING),
                ),
                metrics={
                    "retryable": True,
                    "status_override": "needs_component_resolution",
                    "component_resolution_report_path": _repo_relative(
                        resolution_paths.report_path,
                        root,
                    ),
                    "candidate_npz_path": _repo_relative(
                        resolution_paths.candidate_npz_path,
                        root,
                    ),
                    "component_count": pending_report.get("component_count"),
                    "selected_component_id": pending_report.get("selected_component_id"),
                    "dropped_component_ids": pending_report.get("dropped_component_ids", []),
                    "dropped_part_indices": pending_report.get("dropped_part_indices", []),
                },
                error="tetrahedral material domain needs component resolution",
            )
        return result
    expected = (
        _artifact(
            ArtifactRole.MONOLITHIC_MESH,
            resolved_output_mesh,
            Stage.MESH_PROCESSING,
        ),
    )
    return _run_subprocess_stage(
        name="hag4r_combined_to_monolithic",
        stage=Stage.MESH_PROCESSING,
        argv=_conda_module_argv(
            EnvName.HAG4R_TOOLS,
            "hag4r.mesh",
            (
                "--input_mesh_dir",
                resolved_input_mesh_dir,
                "--output_mesh",
                resolved_output_mesh,
                "--fidelity",
                fidelity,
                *mesh_extra_args,
            ),
        ),
        cwd=root,
        conda_env=EnvName.HAG4R_TOOLS,
        timeout_s=timeout_s,
        expected_artifacts=expected,
        read_paths=(resolved_input_mesh_dir,),
        write_paths=(resolved_output_mesh.parent,),
        env={"PYVISTA_OFF_SCREEN": "true", "MESA_GL_VERSION_OVERRIDE": "3.2"},
        log_dir=log_dir,
        repo_root=root,
    )


def run_compute_unscaled_monolithic_volume(
    *,
    input_mesh: Path,
    output_metadata_path: Path,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
) -> StageRunResult:
    del log_dir
    root = repo_root or _repo_root()
    resolved_input_mesh = _path_for_repo(input_mesh, root)
    resolved_output_metadata = _path_for_repo(output_metadata_path, root)
    payload = mesh_volume_metadata(resolved_input_mesh)
    write_json(resolved_output_metadata, payload)
    return _dict_stage_result(
        name="hag4r_compute_unscaled_monolithic_mesh_volume",
        stage=Stage.MESH_PROCESSING,
        payload=payload,
        artifacts=(_artifact(ArtifactRole.REPORT, resolved_output_metadata, Stage.MESH_PROCESSING),),
    )


def run_scale_monolithic_mesh_to_metric(
    *,
    input_mesh: Path,
    output_mesh: Path,
    volume_metadata_path: Path,
    scaling_metadata_path: Path,
    target_real_world_volume_m3: float,
    estimate_rationale: str,
    object_context: str = "",
    repo_root: Path | None = None,
    log_dir: Path | None = None,
) -> StageRunResult:
    del log_dir
    root = repo_root or _repo_root()
    resolved_input_mesh = _path_for_repo(input_mesh, root)
    resolved_output_mesh = _path_for_repo(output_mesh, root)
    resolved_volume_metadata = _path_for_repo(volume_metadata_path, root)
    resolved_scaling_metadata = _path_for_repo(scaling_metadata_path, root)
    if not resolved_volume_metadata.exists():
        raise FileNotFoundError(
            "metric mesh scaling requires prior unscaled-volume metadata: "
            f"{resolved_volume_metadata}"
        )
    volume_payload = mesh_volume_metadata(resolved_input_mesh)
    import json

    stored_volume_payload = json.loads(resolved_volume_metadata.read_text(encoding="utf-8"))
    stored_volume = float(stored_volume_payload.get("unscaled_volume_m3", "nan"))
    current_volume = float(volume_payload["unscaled_volume_m3"])
    if abs(stored_volume - current_volume) > max(1e-12, abs(current_volume) * 1e-9):
        raise ValueError(
            "stored unscaled volume does not match current mesh volume: "
            f"stored={stored_volume}, current={current_volume}"
        )

    payload = build_metric_scaling_metadata(
        input_mesh_path=resolved_input_mesh,
        output_mesh_path=resolved_output_mesh,
        volume_metadata_path=resolved_volume_metadata,
        volume_metadata=stored_volume_payload,
        target_real_world_volume_m3=target_real_world_volume_m3,
        estimate_rationale=estimate_rationale,
        object_context=object_context,
    )
    write_json(resolved_scaling_metadata, payload)
    return _dict_stage_result(
        name="hag4r_scale_monolithic_mesh_to_metric",
        stage=Stage.MESH_PROCESSING,
        payload=payload,
        artifacts=(
            _artifact(ArtifactRole.MONOLITHIC_MESH, resolved_output_mesh, Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.REPORT, resolved_volume_metadata, Stage.MESH_PROCESSING),
            _artifact(ArtifactRole.REPORT, resolved_scaling_metadata, Stage.MESH_PROCESSING),
        ),
    )




__all__ = [
    "run_compute_unscaled_monolithic_volume",
    "run_scale_monolithic_mesh_to_metric",
    "run_assign_params",
    "run_monolithic_mesh",
]
