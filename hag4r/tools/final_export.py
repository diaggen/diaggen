from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hag4r.agentic.state import ArtifactRole, Stage, StageRunResult, is_subpath, to_json_dict, write_json
from hag4r.tools.common import _artifact, _dict_stage_result


CLEANED_IMAGE_EXPORT_NAME = "cleaned_image.png"
CLEANUP_REPORT_EXPORT_NAME = "image_cleanup.json"
ASSET_REFINEMENT_EXPORT_NAME = "asset_refinement"
SIM_DIAGNOSTICS_EXPORT_NAME = "sim_diagnostics"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path, repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root()
    return path if path.is_absolute() else root / path


def _replace_tree_preserving_relpaths(
    source_dir: Path,
    target_dir: Path,
    preserve_relpaths: tuple[Path, ...],
) -> None:
    def merge_preserved(source: Path, target: Path) -> None:
        if source.is_dir():
            for child in source.rglob("*"):
                relative = child.relative_to(source)
                child_target = target / relative
                if child.is_dir():
                    child_target.mkdir(parents=True, exist_ok=True)
                elif not child_target.exists():
                    child_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, child_target)
            return
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}.preserve.", dir=target_dir.parent))
    try:
        if target_dir.exists():
            for relpath in preserve_relpaths:
                source_preserved = target_dir / relpath
                if not source_preserved.exists():
                    continue
                target_preserved = temp_dir / relpath
                target_preserved.parent.mkdir(parents=True, exist_ok=True)
                if source_preserved.is_dir():
                    shutil.copytree(source_preserved, target_preserved, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_preserved, target_preserved)
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        for relpath in preserve_relpaths:
            source_preserved = temp_dir / relpath
            if not source_preserved.exists():
                continue
            target_preserved = target_dir / relpath
            merge_preserved(source_preserved, target_preserved)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _replace_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


@dataclass(frozen=True)
class FinalExportRequest:
    run_id: str
    output_dir: Path
    mesh_path: Path
    heterogeneous_params_path: Path
    inferred_params_path: Path
    volume_topology_path: Path
    metric_mesh_scaling_path: Path
    appearance_bundle_dir: Path
    youngs_modulus_render_path: Path | None
    density_render_path: Path | None
    target_part_render_path: Path | None
    source_part_render_path: Path | None
    cleaned_image_path: Path | None
    cleanup_report_path: Path | None
    agentic_run_dir: Path | None
    sim_diagnostics_dir: Path | None
    export_paths: dict[str, Path]


def build_final_export_request(
    *,
    run_id: str,
    mesh_path: Path,
    heterogeneous_params_path: Path,
    inferred_params_path: Path,
    volume_topology_path: Path,
    output_dir: Path,
    metric_mesh_scaling_path: Path,
    appearance_bundle_dir: Path,
    youngs_modulus_render_path: Path | None = None,
    density_render_path: Path | None = None,
    target_part_render_path: Path | None = None,
    source_part_render_path: Path | None = None,
    cleaned_image_path: Path | None = None,
    cleanup_report_path: Path | None = None,
    agentic_run_dir: Path | None = None,
    sim_diagnostics_dir: Path | None = None,
) -> FinalExportRequest:
    if (cleaned_image_path is None) != (cleanup_report_path is None):
        raise ValueError("cleaned_image_path and cleanup_report_path must be provided together")
    resolved_output_dir = _path_for_repo(output_dir)
    resolved_mesh = _path_for_repo(mesh_path)
    resolved_heterogeneous_params = _path_for_repo(heterogeneous_params_path)
    resolved_inferred_params = _path_for_repo(inferred_params_path)
    resolved_volume_topology = _path_for_repo(volume_topology_path)
    resolved_metric_mesh_scaling_path = _path_for_repo(metric_mesh_scaling_path)
    resolved_appearance_bundle_dir = _path_for_repo(appearance_bundle_dir)
    export_paths = {
        "final_mesh.mesh": resolved_output_dir / "final_mesh.mesh",
        "heterogeneous_params.npz": resolved_output_dir / "heterogeneous_params.npz",
        "inferred_params.json": resolved_output_dir / "inferred_params.json",
        "volume_topology.json": resolved_output_dir / "volume_topology.json",
        "metric_mesh_scaling.json": resolved_output_dir / "metric_mesh_scaling.json",
        "final_export_manifest.json": resolved_output_dir / "final_export_manifest.json",
    }
    resolved_youngs_modulus_render_path = (
        _path_for_repo(youngs_modulus_render_path) if youngs_modulus_render_path is not None else None
    )
    resolved_density_render_path = _path_for_repo(density_render_path) if density_render_path is not None else None
    resolved_target_part_render_path = (
        _path_for_repo(target_part_render_path) if target_part_render_path is not None else None
    )
    resolved_source_part_render_path = (
        _path_for_repo(source_part_render_path) if source_part_render_path is not None else None
    )
    resolved_cleaned_image_path = _path_for_repo(cleaned_image_path) if cleaned_image_path is not None else None
    resolved_cleanup_report_path = _path_for_repo(cleanup_report_path) if cleanup_report_path is not None else None
    resolved_agentic_run_dir = _path_for_repo(agentic_run_dir) if agentic_run_dir is not None else None
    resolved_sim_diagnostics_dir = _path_for_repo(sim_diagnostics_dir) if sim_diagnostics_dir is not None else None
    return FinalExportRequest(
        run_id=run_id,
        output_dir=resolved_output_dir,
        mesh_path=resolved_mesh,
        heterogeneous_params_path=resolved_heterogeneous_params,
        inferred_params_path=resolved_inferred_params,
        volume_topology_path=resolved_volume_topology,
        metric_mesh_scaling_path=resolved_metric_mesh_scaling_path,
        appearance_bundle_dir=resolved_appearance_bundle_dir,
        youngs_modulus_render_path=resolved_youngs_modulus_render_path,
        density_render_path=resolved_density_render_path,
        target_part_render_path=resolved_target_part_render_path,
        source_part_render_path=resolved_source_part_render_path,
        cleaned_image_path=resolved_cleaned_image_path,
        cleanup_report_path=resolved_cleanup_report_path,
        agentic_run_dir=resolved_agentic_run_dir,
        sim_diagnostics_dir=resolved_sim_diagnostics_dir,
        export_paths=export_paths,
    )


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _material_payload(request: FinalExportRequest) -> dict[str, Any]:
    return _read_json_object(request.inferred_params_path, name="inferred_params.json")


def _validate_tet_params(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        forbidden = sorted(key for key in keys if key.startswith("tri" + "_") or "thickness" in key)
        if forbidden:
            raise ValueError("canonical heterogeneous_params.npz contains legacy key(s): " + ", ".join(forbidden))
        missing = sorted(key for key in ("tet_E_nu", "tet_density", "tet_part_labels") if key not in keys)
        if missing:
            raise ValueError("canonical heterogeneous_params.npz missing required key(s): " + ", ".join(missing))
        tet_count = int(np.asarray(data["tet_part_labels"]).shape[0])
    return {"keys": sorted(keys), "tet_count": tet_count}


def _find_forbidden_canonical_fields(value: Any, prefix: str = "$") -> list[str]:
    forbidden = {
        "representation",
        "expected_" + "representation",
        "representation_" + "decision",
        "representation_" + "status",
        "representation_" + "audit",
        "shell_" + "thickness_m",
        "wall_" + "thickness_m",
        "surface_processing_audit_path",
        "final_homogeneous_params_path",
        "uniform_params_path",
    }
    if isinstance(value, dict):
        hits = [f"{prefix}.{key}" for key in value if key in forbidden or str(key).startswith("tri" + "_")]
        for key, item in value.items():
            hits.extend(_find_forbidden_canonical_fields(item, f"{prefix}.{key}"))
        return hits
    if isinstance(value, list):
        hits: list[str] = []
        for index, item in enumerate(value):
            hits.extend(_find_forbidden_canonical_fields(item, f"{prefix}[{index}]"))
        return hits
    return []


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_canonical_final_export_contract(request: FinalExportRequest) -> dict[str, Any]:
    if request.mesh_path.suffix.lower() != ".mesh":
        raise ValueError(f"canonical final mesh must be a .mesh tetrahedral file: {request.mesh_path}")
    params_summary = _validate_tet_params(request.heterogeneous_params_path)
    material_payload = _material_payload(request)
    legacy_fields = _find_forbidden_canonical_fields(material_payload)
    if legacy_fields:
        raise ValueError("inferred_params.json contains legacy field(s): " + ", ".join(legacy_fields))
    volume_topology = _read_json_object(request.volume_topology_path, name="volume_topology.json")
    legacy_fields = _find_forbidden_canonical_fields(volume_topology)
    if legacy_fields:
        raise ValueError("volume_topology.json contains legacy field(s): " + ", ".join(legacy_fields))
    metric_mesh_scaling = _read_json_object(request.metric_mesh_scaling_path, name="metric_mesh_scaling.json")
    legacy_fields = _find_forbidden_canonical_fields(metric_mesh_scaling)
    if legacy_fields:
        raise ValueError("metric_mesh_scaling.json contains legacy field(s): " + ", ".join(legacy_fields))
    return {
        "params": params_summary,
        "volume_topology": {
            key: volume_topology.get(key)
            for key in ("schema_version", "volume_topology", "tet_count", "tet_budget_status", "warnings")
            if key in volume_topology
        },
        "metric_mesh_scaling": {
            key: metric_mesh_scaling.get(key)
            for key in ("schema_version", "target_max_dimension_m", "scale_factor")
            if key in metric_mesh_scaling
        },
    }


def write_final_export_bundle(
    request: FinalExportRequest,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    from hag4r.tools.post_mesh_texture import validate_post_mesh_texture_bundle

    root = repo_root or _repo_root()
    if not is_subpath(request.output_dir, root) and not is_subpath(request.output_dir, root / "outputs"):
        raise ValueError(f"Refusing to write final export outside HAG4R repo root: {request.output_dir}")
    for target_path in request.export_paths.values():
        if not is_subpath(target_path, root) and not is_subpath(target_path, root / "outputs"):
            raise ValueError(f"Refusing to write final export artifact outside HAG4R repo root: {target_path}")
    canonical_sources = {
        "final_mesh.mesh": request.mesh_path,
        "heterogeneous_params.npz": request.heterogeneous_params_path,
        "inferred_params.json": request.inferred_params_path,
        "volume_topology.json": request.volume_topology_path,
        "metric_mesh_scaling.json": request.metric_mesh_scaling_path,
    }
    auxiliary_sources: dict[str, Path] = {}
    if request.youngs_modulus_render_path is not None:
        auxiliary_sources["heterogeneous_youngs_modulus.png"] = request.youngs_modulus_render_path
    if request.density_render_path is not None:
        auxiliary_sources["heterogeneous_density.png"] = request.density_render_path
    if request.target_part_render_path is not None:
        auxiliary_sources["heterogeneous_part_ids.png"] = request.target_part_render_path
    if request.source_part_render_path is not None:
        auxiliary_sources["source_part_ids.png"] = request.source_part_render_path
    if request.cleaned_image_path is not None and request.cleanup_report_path is not None:
        auxiliary_sources[CLEANED_IMAGE_EXPORT_NAME] = request.cleaned_image_path
        auxiliary_sources[CLEANUP_REPORT_EXPORT_NAME] = request.cleanup_report_path
    missing = [str(path) for path in (*canonical_sources.values(), *auxiliary_sources.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError("Final export is missing source artifact(s): " + ", ".join(missing))
    appearance_source = request.appearance_bundle_dir.resolve()
    output_resolved = request.output_dir.resolve()
    if request.appearance_bundle_dir.is_symlink() or not request.appearance_bundle_dir.is_dir():
        raise NotADirectoryError(
            f"required post-mesh appearance bundle is not a regular directory: "
            f"{request.appearance_bundle_dir}"
        )
    if (
        appearance_source == output_resolved
        or is_subpath(appearance_source, output_resolved)
        or is_subpath(output_resolved, appearance_source)
    ):
        raise ValueError("appearance source and final export output must not overlap")
    validate_post_mesh_texture_bundle(
        appearance_source,
        validate_glb=True,
        repo_root=root,
    )
    contract_summary = _validate_canonical_final_export_contract(request)
    if request.agentic_run_dir is not None:
        if not request.agentic_run_dir.exists():
            raise FileNotFoundError(f"Agentic run artifact snapshot source is missing: {request.agentic_run_dir}")
        if not request.agentic_run_dir.is_dir():
            raise NotADirectoryError(f"Agentic run artifact snapshot source is not a directory: {request.agentic_run_dir}")
    if request.sim_diagnostics_dir is not None:
        if not request.sim_diagnostics_dir.exists():
            raise FileNotFoundError(f"Simulation diagnostics export source is missing: {request.sim_diagnostics_dir}")
        if not request.sim_diagnostics_dir.is_dir():
            raise NotADirectoryError(f"Simulation diagnostics export source is not a directory: {request.sim_diagnostics_dir}")
        ownership_source = request.sim_diagnostics_dir / "runtime_ownership.json"
        if not ownership_source.is_file():
            raise FileNotFoundError(
                f"Simulation diagnostics runtime ownership artifact is missing: {ownership_source}"
            )

    request.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_names = {*request.export_paths, "appearance"}
    for child in request.output_dir.iterdir():
        if child.name in canonical_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    copied: dict[str, str] = {}
    for name, source_path in canonical_sources.items():
        target_path = request.export_paths[name]
        shutil.copy2(source_path, target_path)
        copied[name] = str(target_path)
    appearance_target = request.output_dir / "appearance"
    if appearance_target.exists():
        shutil.rmtree(appearance_target)
    shutil.copytree(appearance_source, appearance_target)
    appearance_relative_files = sorted(
        (
            path.relative_to(request.output_dir).as_posix()
            for path in appearance_target.rglob("*")
            if path.is_file()
        )
    )
    for relative in appearance_relative_files:
        copied[relative] = str(request.output_dir / relative)

    source_directories: dict[str, str] = {}
    if request.agentic_run_dir is not None:
        source_directories[ASSET_REFINEMENT_EXPORT_NAME] = str(request.agentic_run_dir)
    if request.sim_diagnostics_dir is not None:
        source_directories[SIM_DIAGNOSTICS_EXPORT_NAME] = str(request.sim_diagnostics_dir)
    manifest = {
        "run_id": request.run_id,
        "canonical_files": [
            "final_mesh.mesh",
            "heterogeneous_params.npz",
            "inferred_params.json",
            "volume_topology.json",
            "metric_mesh_scaling.json",
            "final_export_manifest.json",
            *appearance_relative_files,
        ],
        "source_artifacts": {name: _file_record(path) for name, path in canonical_sources.items()},
        "auxiliary_source_artifacts": {
            name: _file_record(path) for name, path in auxiliary_sources.items()
        },
        "source_directories": source_directories,
        "exported_artifacts": {
            name: _file_record(Path(path)) if Path(path).is_file() else path
            for name, path in copied.items()
        },
        "validation": contract_summary,
    }
    if request.sim_diagnostics_dir is not None:
        manifest["source_artifacts"]["diagnostic_runtime_ownership"] = str(
            request.sim_diagnostics_dir / "runtime_ownership.json"
        )
    manifest["metric_mesh_scaling"] = json.loads(request.metric_mesh_scaling_path.read_text(encoding="utf-8"))
    manifest["volume_topology"] = json.loads(request.volume_topology_path.read_text(encoding="utf-8"))
    write_json(request.export_paths["final_export_manifest.json"], manifest)
    return {
        "status": "written",
        "manifest_path": str(request.export_paths["final_export_manifest.json"]),
        "exported_artifacts": copied,
    }


def final_export_request_payload(
    run_id: str,
    mesh_path: str,
    heterogeneous_params_path: str,
    inferred_params_path: str,
    volume_topology_path: str,
    output_dir: str,
    metric_mesh_scaling_path: str,
    appearance_bundle_dir: str,
    youngs_modulus_render_path: str | None = None,
    density_render_path: str | None = None,
    target_part_render_path: str | None = None,
    source_part_render_path: str | None = None,
    cleaned_image_path: str | None = None,
    cleanup_report_path: str | None = None,
    agentic_run_dir: str | None = None,
    sim_diagnostics_dir: str | None = None,
) -> dict[str, Any]:
    plan = build_final_export_request(
        run_id=run_id,
        mesh_path=Path(mesh_path),
        heterogeneous_params_path=Path(heterogeneous_params_path),
        inferred_params_path=Path(inferred_params_path),
        volume_topology_path=Path(volume_topology_path),
        output_dir=Path(output_dir),
        metric_mesh_scaling_path=Path(metric_mesh_scaling_path),
        appearance_bundle_dir=Path(appearance_bundle_dir),
        youngs_modulus_render_path=Path(youngs_modulus_render_path) if youngs_modulus_render_path else None,
        density_render_path=Path(density_render_path) if density_render_path else None,
        target_part_render_path=Path(target_part_render_path) if target_part_render_path else None,
        source_part_render_path=Path(source_part_render_path) if source_part_render_path else None,
        cleaned_image_path=Path(cleaned_image_path) if cleaned_image_path else None,
        cleanup_report_path=Path(cleanup_report_path) if cleanup_report_path else None,
        agentic_run_dir=Path(agentic_run_dir) if agentic_run_dir else None,
        sim_diagnostics_dir=Path(sim_diagnostics_dir) if sim_diagnostics_dir else None,
    )
    payload = to_json_dict(plan)
    payload["status"] = "ready"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a stable HAG4R final export bundle.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--heterogeneous_params_path", required=True)
    parser.add_argument("--inferred_params_path", required=True)
    parser.add_argument("--volume_topology_path", required=True)
    parser.add_argument("--metric_mesh_scaling_path", required=True)
    parser.add_argument("--appearance_bundle_dir", required=True)
    parser.add_argument("--youngs_modulus_render_path")
    parser.add_argument("--density_render_path")
    parser.add_argument("--target_part_render_path")
    parser.add_argument("--source_part_render_path")
    parser.add_argument("--cleaned_image_path")
    parser.add_argument("--cleanup_report_path")
    parser.add_argument("--agentic_run_dir")
    parser.add_argument("--sim_diagnostics_dir")
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_final_export_request(
        run_id=args.run_id,
        mesh_path=Path(args.mesh_path),
        heterogeneous_params_path=Path(args.heterogeneous_params_path),
        inferred_params_path=Path(args.inferred_params_path),
        volume_topology_path=Path(args.volume_topology_path),
        output_dir=Path(args.output_dir),
        metric_mesh_scaling_path=Path(args.metric_mesh_scaling_path),
        appearance_bundle_dir=Path(args.appearance_bundle_dir),
        youngs_modulus_render_path=Path(args.youngs_modulus_render_path) if args.youngs_modulus_render_path else None,
        density_render_path=Path(args.density_render_path) if args.density_render_path else None,
        target_part_render_path=Path(args.target_part_render_path) if args.target_part_render_path else None,
        source_part_render_path=Path(args.source_part_render_path) if args.source_part_render_path else None,
        cleaned_image_path=Path(args.cleaned_image_path) if args.cleaned_image_path else None,
        cleanup_report_path=Path(args.cleanup_report_path) if args.cleanup_report_path else None,
        agentic_run_dir=Path(args.agentic_run_dir) if args.agentic_run_dir else None,
        sim_diagnostics_dir=Path(args.sim_diagnostics_dir) if args.sim_diagnostics_dir else None,
    )
    payload: dict[str, Any] = write_final_export_bundle(plan)
    print(json.dumps(payload, indent=2))
    return 0


def run_final_export(
    *,
    run_id: str,
    mesh_path: Path,
    heterogeneous_params_path: Path,
    inferred_params_path: Path,
    volume_topology_path: Path,
    output_dir: Path,
    metric_mesh_scaling_path: Path,
    appearance_bundle_dir: Path,
    youngs_modulus_render_path: Path | None = None,
    density_render_path: Path | None = None,
    target_part_render_path: Path | None = None,
    source_part_render_path: Path | None = None,
    cleaned_image_path: Path | None = None,
    cleanup_report_path: Path | None = None,
    agentic_run_dir: Path | None = None,
    sim_diagnostics_dir: Path | None = None,
    repo_root: Path | None = None,
) -> StageRunResult:
    root = repo_root or _repo_root()

    def existing_optional(path: Path | None) -> Path | None:
        if path is None:
            return None
        resolved = _path_for_repo(path, root)
        return resolved if resolved.exists() else None

    request = build_final_export_request(
        run_id=run_id,
        mesh_path=mesh_path,
        heterogeneous_params_path=heterogeneous_params_path,
        inferred_params_path=inferred_params_path,
        volume_topology_path=volume_topology_path,
        output_dir=output_dir,
        metric_mesh_scaling_path=metric_mesh_scaling_path,
        appearance_bundle_dir=appearance_bundle_dir,
        youngs_modulus_render_path=existing_optional(youngs_modulus_render_path),
        density_render_path=existing_optional(density_render_path),
        target_part_render_path=existing_optional(target_part_render_path),
        source_part_render_path=existing_optional(source_part_render_path),
        cleaned_image_path=cleaned_image_path,
        cleanup_report_path=cleanup_report_path,
        agentic_run_dir=agentic_run_dir,
        sim_diagnostics_dir=sim_diagnostics_dir,
    )
    payload = write_final_export_bundle(request, repo_root=root)
    artifacts = [
        _artifact(ArtifactRole.FINAL_MESH, request.export_paths["final_mesh.mesh"], Stage.FINAL_EXPORT),
        _artifact(ArtifactRole.FINAL_PARAMS, request.export_paths["heterogeneous_params.npz"], Stage.FINAL_EXPORT),
        _artifact(ArtifactRole.INFERRED_PARAMS, request.export_paths["inferred_params.json"], Stage.FINAL_EXPORT),
        _artifact(ArtifactRole.REPORT, request.export_paths["volume_topology.json"], Stage.FINAL_EXPORT),
        _artifact(ArtifactRole.REPORT, request.export_paths["metric_mesh_scaling.json"], Stage.FINAL_EXPORT),
        _artifact(ArtifactRole.REPORT, request.export_paths["final_export_manifest.json"], Stage.FINAL_EXPORT),
        _artifact(
            ArtifactRole.TEXTURED_VISUAL_MESH,
            request.output_dir / "appearance" / "visual_mesh.glb",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.VISUAL_TO_PHYSICS_BINDING,
            request.output_dir / "appearance" / "visual_to_physics.npz",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.VISUAL_MANIFEST,
            request.output_dir / "appearance" / "visual_manifest.json",
            Stage.POST_MESH_TEXTURE,
        ),
        _artifact(
            ArtifactRole.TEXTURE_QA_REPORT,
            request.output_dir / "appearance" / "qa" / "report.json",
            Stage.POST_MESH_TEXTURE,
        ),
    ]
    return _dict_stage_result(
        name="hag4r_final_export_bundle",
        stage=Stage.FINAL_EXPORT,
        payload=payload,
        artifacts=tuple(artifacts),
    )


__all__ = [
    "CLEANED_IMAGE_EXPORT_NAME",
    "CLEANUP_REPORT_EXPORT_NAME",
    "ASSET_REFINEMENT_EXPORT_NAME",
    "SIM_DIAGNOSTICS_EXPORT_NAME",
    "FinalExportRequest",
    "build_final_export_request",
    "main",
    "final_export_request_payload",
    "run_final_export",
    "write_final_export_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
