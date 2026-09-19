from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from hag4r.agentic.state import ArtifactRole, Stage, StageRunResult
from hag4r.tools.common import (
    EnvName,
    VIEW_NAMES,
    _artifact,
    _conda_module_argv,
    _path_for_cwd,
    _path_for_repo,
    _repo_root,
    _run_subprocess_stage,
    resolve_conda_env,
)


def run_omnipart_generation(
    segmentation_manifest_path: Path,
    output_root: Path,
    *,
    repo_root: Path | None = None,
    conda_env: str = EnvName.OMNIPART,
    timeout_s: int = 7200,
    log_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> StageRunResult:
    root = repo_root or _repo_root()
    cwd = root / "third_party/OmniPart"
    resolved_segmentation_manifest_path = _path_for_repo(segmentation_manifest_path, root)
    manifest = json.loads(resolved_segmentation_manifest_path.read_text(encoding="utf-8"))
    object_name = str(manifest["object_name"])
    step16_inputs = manifest["omnipart_step16_inputs"]
    step16_read_paths = tuple(_path_for_repo(Path(path), root) for path in step16_inputs.values())
    resolved_output_root = _path_for_repo(output_root, root)
    output_dir = resolved_output_root / object_name
    shutil.rmtree(output_dir, ignore_errors=True)
    local_ckpts = (
        cwd / "ckpt/partfield_encoder.ckpt",
        cwd / "ckpt/bbox_gen.ckpt",
    )
    expected = (
        _artifact(ArtifactRole.PART_LABELS, output_dir / "part_labels.npz", Stage.OMNIPART_GENERATION),
        _artifact(ArtifactRole.COMBINED_SURFACE, output_dir / "mesh_combined_colored_by_parts.glb", Stage.OMNIPART_GENERATION),
        _artifact(ArtifactRole.COMBINED_TET_MESH, output_dir / "mesh_combined.mesh", Stage.OMNIPART_GENERATION),
        _artifact(
            ArtifactRole.OMNIPART_APPEARANCE_BUNDLE,
            output_dir / "appearance" / "appearance_manifest.json",
            Stage.OMNIPART_GENERATION,
            description="OmniPart appearance bundle manifest",
        ),
        *(
            _artifact(
                ArtifactRole.SEGMENTED_VIEW,
                output_dir / f"mesh_combined_part_labels_{view}.png",
                Stage.OMNIPART_GENERATION,
            )
            for view in VIEW_NAMES
        ),
    )
    argv = _conda_module_argv(
        conda_env,
        "scripts.inference_omnipart",
        (
            "--segmentation_manifest",
            _path_for_cwd(segmentation_manifest_path, cwd=cwd, repo_root=root),
            "--output_root",
            _path_for_cwd(output_root, cwd=cwd, repo_root=root),
        ),
    )
    return _run_subprocess_stage(
        name="omnipart_generate_parts",
        stage=Stage.OMNIPART_GENERATION,
        argv=argv,
        cwd=cwd,
        conda_env=resolve_conda_env(conda_env),
        timeout_s=timeout_s,
        expected_artifacts=expected,
        read_paths=(resolved_segmentation_manifest_path, *step16_read_paths, *local_ckpts),
        write_paths=(resolved_output_root,),
        env={
            "PYVISTA_OFF_SCREEN": "true",
            "MESA_GL_VERSION_OVERRIDE": "3.2",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **dict(env or {}),
        },
        log_dir=log_dir,
        repo_root=root,
    )


__all__ = ["run_omnipart_generation"]
