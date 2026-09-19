from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-07-skill-suite-agentic-state-v1"
DEFAULT_RUN_ROOT = Path("outputs/agentic_asset_refinement")


class Stage(str, Enum):
    ORCHESTRATION = "orchestration"
    IMAGE_CLEANUP = "image_cleanup"
    PROMPT_PLANNING = "prompt_planning"
    SEGMENTATION = "segmentation"
    OMNIPART_GENERATION = "omnipart_generation"
    MATERIAL_INFERENCE = "material_inference"
    PRIMITIVE_ASSIGNMENT = "primitive_assignment"
    MESH_PROCESSING = "mesh_processing"
    MONOLITHIC_TRANSFER = "monolithic_transfer"
    POST_MESH_TEXTURE = "post_mesh_texture"
    FINAL_EXPORT = "final_export"
    GENESIS_DIAGNOSTICS = "genesis_diagnostics"
    REFINEMENT_ROUTING = "refinement_routing"


class ArtifactRole(str, Enum):
    SOURCE_IMAGE = "source_image"
    OBJECT_DESCRIPTION = "object_description"
    CLEANED_SOURCE_IMAGE = "cleaned_source_image"
    PROCESSED_IMAGE = "processed_image"
    SEGMENTATION_MASK = "segmentation_mask"
    PART_MESH_DIR = "part_mesh_dir"
    COMBINED_SURFACE = "combined_surface"
    COMBINED_TET_MESH = "combined_tet_mesh"
    OMNIPART_APPEARANCE_BUNDLE = "omnipart_appearance_bundle"
    PART_LABELS = "part_labels"
    SEGMENTED_VIEW = "segmented_view"
    INFERRED_PARAMS = "inferred_params"
    PARTWISE_PARAMS = "partwise_params"
    MONOLITHIC_MESH = "monolithic_mesh"
    MONOLITHIC_PARAMS = "monolithic_params"
    FINAL_MESH = "final_mesh"
    FINAL_PARAMS = "final_params"
    TEXTURED_VISUAL_MESH = "textured_visual_mesh"
    VISUAL_TO_PHYSICS_BINDING = "visual_to_physics_binding"
    VISUAL_MANIFEST = "visual_manifest"
    TEXTURE_QA_REPORT = "texture_qa_report"
    DENSITY_RENDER = "density_render"
    YOUNGS_MODULUS_RENDER = "youngs_modulus_render"
    PART_ID_RENDER = "part_id_render"
    AGENTIC_RUN_ARTIFACT_SNAPSHOT = "agentic_run_artifact_snapshot"
    GENESIS_CONFIG = "genesis_config"
    GENESIS_FRAME = "genesis_frame"
    GENESIS_LOG = "genesis_log"
    STIFFNESS_MAP = "stiffness_map"
    STRESS_MAP = "stress_map"
    TOOL_LOG = "tool_log"
    REPORT = "report"


class ObservationSignal(str, Enum):
    LOADABILITY = "loadability"
    PART_IDENTITY = "part_identity"
    STIFFNESS_CONTRAST = "stiffness_contrast"
    DEFORMATION = "deformation"
    CONTACT_STABILITY = "contact_stability"
    COLLISION_OR_PENETRATION = "collision_or_penetration"
    STRESS_OR_VON_MISES = "stress_or_von_mises"
    VISUAL_PLAUSIBILITY = "visual_plausibility"
    SIMULATOR_CAPABILITY = "simulator_capability"


class RefinementTarget(str, Enum):
    IMAGE_CLEANUP = "image_cleanup"
    OBJECT_DESCRIPTION = "object_description"
    SEGMENTATION = "segmentation"
    OMNIPART_GENERATION = "omnipart_generation"
    MESH_PROCESSING = "mesh_processing"
    MATERIAL_INFERENCE = "material_inference"
    MATERIAL_ASSIGNMENT = "material_assignment"
    GENESIS_TEST = "genesis_test"
    HUMAN_REVIEW = "human_review"
    ACCEPT = "accept"


class SimDiagnosticRoute(str, Enum):
    ACCEPT = "accept"
    SEGMENTATION = "segmentation"
    MATERIAL_INFERENCE = "material_inference"
    MESH_PROCESSING = "mesh_processing"


class GenesisActionType(str, Enum):
    NONE = "none"
    KEY_TAP = "key_tap"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    KEY_SEQUENCE = "key_sequence"
    TERMINATE = "terminate"


class GenesisKeyboardKey(str, Enum):
    W = "W"
    A = "A"
    S = "S"
    D = "D"
    Q = "Q"
    E = "E"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    U = "U"
    O = "O"
    Z = "Z"
    M = "M"
    R = "R"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class ArtifactRef:
    role: ArtifactRole
    path: Path
    stage: Stage | None = None
    description: str = ""
    required: bool = True


def artifact_exists(artifact: ArtifactRef) -> bool:
    if artifact.role == ArtifactRole.GENESIS_FRAME and artifact.path.is_dir():
        return any(artifact.path.glob("frame_*.png"))
    return artifact.path.exists()


@dataclass(frozen=True)
class PartIdentity:
    part_index: int
    part_color_rgb: tuple[int, int, int]
    part_name: str = ""
    source: str = "part_labels.npz"
    notes: str = ""


@dataclass(frozen=True)
class PartMap:
    part_labels_path: Path
    parts: tuple[PartIdentity, ...] = ()
    label_fields: tuple[str, ...] = (
        "part_colors",
        "tet_part_labels",
        "tet_vertex_part_labels",
        "surface_vertex_part_labels",
    )
    provenance: str = "OmniPart part_labels.npz"


@dataclass(frozen=True)
class AssetBundle:
    source_image: Path
    run_id: str
    omnipart_output_dir: Path | None = None
    part_map: PartMap | None = None
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PreflightIssue:
    severity: Severity
    message: str
    path: Path | None = None
    stage: Stage | None = None


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    issues: tuple[PreflightIssue, ...] = ()
    checked_paths: tuple[Path, ...] = ()
    checked_envs: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageRunResult:
    name: str
    stage: Stage
    success: bool
    returncode: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class DiagnosticObservation:
    signal: ObservationSignal
    severity: Severity
    summary: str
    part_indices: tuple[int, ...] = ()
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    supported: bool = True


@dataclass(frozen=True)
class GenesisAction:
    action_type: GenesisActionType
    payload: dict[str, Any] = field(default_factory=dict)
    scheduled_time_s: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class GenesisEpisodeSpec:
    diagnostic_run_index: int
    episode_index: int
    control_mode: str
    scene_config_path: Path
    output_dir: Path
    session_mode: str = "live"
    live_protocol: str = ""
    ready_file_path: Path | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    baseline_frame: int | None = None
    final_frame: int | None = None
    planned_actions: tuple[GenesisAction, ...] = ()
    question: str = ""


@dataclass(frozen=True)
class GenesisEpisodeResult:
    spec: GenesisEpisodeSpec
    success: bool
    returncode: int | None
    started_at: str
    finished_at: str
    live_mode: bool = False
    paused_guaranteed: bool = False
    accepted_frame: int | None = None
    applied_frame: int | None = None
    observed_effect: bool | None = None
    unsupported_capabilities: tuple[str, ...] = ()
    stdout_log_path: Path | None = None
    stderr_log_path: Path | None = None
    observations: tuple[DiagnosticObservation, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    unsupported_state_fields: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class SimDiagnosticCue:
    route: SimDiagnosticRoute
    reason: str
    issue_signals: tuple[ObservationSignal, ...] = ()
    segmentation_hints: tuple[str, ...] = ()
    material_inference_hints: tuple[str, ...] = ()
    mesh_processing_hints: tuple[str, ...] = ()
    part_indices: tuple[int, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementAction:
    target: RefinementTarget
    reason: str
    source_signal: ObservationSignal | None = None
    part_indices: tuple[int, ...] = ()
    priority: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactManifest:
    schema_version: str
    run_id: str
    artifacts: list[ArtifactRef] = field(default_factory=list)
    part_map: PartMap | None = None
    observations: list[DiagnosticObservation] = field(default_factory=list)
    actions: list[RefinementAction] = field(default_factory=list)
    orchestration_plan: dict[str, Any] = field(default_factory=dict)
    refinement_plan: dict[str, Any] = field(default_factory=dict)
    stage_status: dict[str, str] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)
    model_calls: dict[str, int] = field(default_factory=dict)
    stage_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AssetRunState:
    run_id: str
    source_image: Path
    run_root: Path = DEFAULT_RUN_ROOT
    schema_version: str = SCHEMA_VERSION
    user_hints: str = ""
    artifact_manifest: ArtifactManifest | None = None
    bundle: AssetBundle | None = None
    stage_records: list[StageRunResult] = field(default_factory=list)
    observations: list[DiagnosticObservation] = field(default_factory=list)
    actions: list[RefinementAction] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    def manifest(self) -> ArtifactManifest:
        if self.artifact_manifest is None:
            artifacts = list(self.bundle.artifacts) if self.bundle else []
            part_map = self.bundle.part_map if self.bundle else None
            self.artifact_manifest = ArtifactManifest(
                schema_version=self.schema_version,
                run_id=self.run_id,
                artifacts=artifacts,
                part_map=part_map,
                observations=list(self.observations),
                actions=list(self.actions),
                retry_counts=dict(self.retry_counts),
                stage_records=[to_json_dict(record) for record in self.stage_records],
            )
        return self.artifact_manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def to_json_dict(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(value, default=_json_default))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json_default, indent=2) + "\n", encoding="utf-8")


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        pass
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_git_branch(repo_path: Path) -> str | None:
    git_path = repo_path / ".git"
    if git_path.is_file():
        text = git_path.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_dir = (repo_path / text.split(":", 1)[1].strip()).resolve()
        else:
            return None
    else:
        git_dir = git_path

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    prefix = "ref: refs/heads/"
    if head.startswith(prefix):
        return head[len(prefix):]
    return None


def preflight_paths(
    *,
    name: str,
    stage: Stage,
    cwd: Path,
    repo_root: Path,
    read_paths: tuple[Path, ...] = (),
    write_paths: tuple[Path, ...] = (),
    conda_env: str | None = None,
    allow_external_writes: bool = False,
) -> PreflightReport:
    issues: list[PreflightIssue] = []
    checked_paths: list[Path] = [cwd]

    if not cwd.exists():
        issues.append(
            PreflightIssue(
                severity=Severity.ERROR,
                stage=stage,
                path=cwd,
                message=f"cwd does not exist for {name}: {cwd}",
            )
        )

    for path in write_paths:
        checked_paths.append(path)
        if (
            not allow_external_writes
            and not is_subpath(path, repo_root)
            and not is_subpath(path, repo_root / "outputs")
        ):
            issues.append(
                PreflightIssue(
                    severity=Severity.BLOCKER,
                    stage=stage,
                    path=path,
                    message="write path is outside the HAG4R repository",
                )
            )

    for path in read_paths:
        checked_paths.append(path)
        if not path.exists():
            issues.append(
                PreflightIssue(
                    severity=Severity.ERROR,
                    stage=stage,
                    path=path,
                    message=f"required input or local asset is missing: {path}",
                )
            )

    ok = not any(issue.severity in {Severity.ERROR, Severity.BLOCKER} for issue in issues)
    return PreflightReport(
        ok=ok,
        issues=tuple(issues),
        checked_paths=tuple(checked_paths),
        checked_envs=(conda_env,) if conda_env else (),
    )
