from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from hag4r.agentic.state import ArtifactRole, Stage, StageRunResult, is_subpath, to_json_dict
from hag4r.tools.common import _artifact, _dict_stage_result
from hag4r.tools.object_description import normalize_object_description_response, object_description_text


IMAGE_CLEANUP_REPORT_SCHEMA_VERSION = "hag4r-image-cleanup-report-v2"
IMAGE_CLEANUP_MAX_ATTEMPTS = 2
VALIDATION_CHECKS: tuple[str, ...] = (
    "object_identity",
    "single_complete_instance",
    "silhouette_proportions_and_openings",
    "part_boundaries_joints_and_compliance",
    "material_cues",
    "pose_and_visibility",
    "background_and_artifacts",
    "description_alignment",
)
IMAGEGEN_EVIDENCE_REF_KEYS = frozenset({"tool_call_ref", "generated_image_path", "artifact_id", "transcript_ref"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_for_repo(path: Path, repo_root: Path | None = None) -> Path:
    root = repo_root or _repo_root()
    return path if path.is_absolute() else root / path


def _resolve_allowed_output_roots(*, repo_root: Path, allowed_output_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    roots = allowed_output_roots or (repo_root / "outputs", repo_root)
    return tuple(_path_for_repo(Path(root), repo_root).resolve() for root in roots)


def _require_path_under_roots(path: Path, roots: tuple[Path, ...], *, label: str) -> None:
    if not any(is_subpath(path, root) for root in roots):
        raise ValueError(f"{label} must be under an allowed output root ({', '.join(str(root) for root in roots)}): {path}")


def _require_imagegen_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("imagegen_invocation_evidence must be a non-empty mapping")
    payload = to_json_dict(dict(evidence))
    if str(payload.get("tool", "") or "").strip() != "imagegen" and str(payload.get("kind", "") or "").strip() != "codex_builtin_imagegen":
        raise ValueError("imagegen_invocation_evidence must identify built-in imagegen via `tool: imagegen` or `kind: codex_builtin_imagegen`")
    if not any(str(payload.get(key, "") or "").strip() for key in IMAGEGEN_EVIDENCE_REF_KEYS):
        raise ValueError("imagegen_invocation_evidence must include at least one imagegen reference: " + ", ".join(sorted(IMAGEGEN_EVIDENCE_REF_KEYS)))
    return payload


def _verify_existing_image(path: Path) -> tuple[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"cleaned image does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"cleaned image is not a file: {path}")
    image_bytes = path.stat().st_size
    if image_bytes <= 0:
        raise ValueError(f"cleaned image is empty: {path}")
    with Image.open(path) as image:
        image.verify()
    return image_bytes, hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_description(*, source_image: Path, candidate_image: Path, attempt: Mapping[str, Any], object_name_hint: str, user_hints: str) -> dict[str, Any]:
    name = str(attempt.get("inferred_object_name", "") or "").strip()
    description = str(attempt.get("object_description", "") or "").strip()
    if not name:
        raise ValueError("generated cleanup attempt requires non-empty inferred_object_name")
    if not description:
        raise ValueError("generated cleanup attempt requires non-empty object_description")
    return normalize_object_description_response(
        {"inferred_object_name": name, "object_description": description},
        source_image=source_image,
        image_paths=(candidate_image, source_image),
        generator="codex_image_cleanup_stage",
        object_name_hint=object_name_hint,
        user_hints=user_hints,
        repair_provenance={"author": "codex_image_cleanup_stage", "tool": "register_image_cleanup_stage", "imagegen": "builtin_codex_imagegen"},
    )


def _normalize_validation(value: Any, *, attempt_index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("generated cleanup attempt requires a validation mapping")
    payload = to_json_dict(dict(value))
    if int(payload.get("attempt_index", 0) or 0) != attempt_index:
        raise ValueError("validation.attempt_index must match attempt_index")
    verdict = str(payload.get("verdict", "") or "").strip()
    if verdict not in {"pass", "fail"}:
        raise ValueError("validation.verdict must be 'pass' or 'fail'")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(VALIDATION_CHECKS):
        raise ValueError("validation.checks must contain exactly the cleanup-validation check keys")
    normalized_checks: dict[str, dict[str, Any]] = {}
    for key in VALIDATION_CHECKS:
        check = checks[key]
        if not isinstance(check, Mapping) or not isinstance(check.get("passed"), bool):
            raise ValueError(f"validation.checks.{key} requires boolean passed")
        evidence = str(check.get("evidence", "") or "").strip()
        if not evidence:
            raise ValueError(f"validation.checks.{key} requires non-empty evidence")
        normalized_checks[key] = {"passed": check["passed"], "evidence": evidence}
    all_passed = all(check["passed"] for check in normalized_checks.values())
    if (verdict == "pass") != all_passed:
        raise ValueError("validation.verdict must equal the conjunction of all check results")
    summary = str(payload.get("summary", "") or "").strip()
    if not summary:
        raise ValueError("validation.summary must be non-empty")
    return {"attempt_index": attempt_index, "verdict": verdict, "checks": normalized_checks, "summary": summary}


def _normalize_attempts(*, attempts: list[dict[str, Any]], candidate_paths: Mapping[int, Path], source_image: Path, object_name_hint: str, user_hints: str, allowed_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    if not isinstance(attempts, list) or not attempts or len(attempts) > IMAGE_CLEANUP_MAX_ATTEMPTS:
        raise ValueError(f"attempts must contain one or two records (max {IMAGE_CLEANUP_MAX_ATTEMPTS})")
    normalized: list[dict[str, Any]] = []
    for expected_index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or int(attempt.get("attempt_index", 0) or 0) != expected_index:
            raise ValueError("attempts must be consecutive and start at attempt_index=1")
        status = str(attempt.get("status", "") or "").strip()
        if status == "generation_failed":
            error = str(attempt.get("error", "") or "").strip()
            if not error:
                raise ValueError("generation_failed attempt requires non-empty error")
            record: dict[str, Any] = {"attempt_index": expected_index, "status": status, "error": error}
            if attempt.get("imagegen_prompt_summary"):
                record["imagegen_prompt_summary"] = str(attempt["imagegen_prompt_summary"]).strip()
            if attempt.get("imagegen_invocation_evidence"):
                record["imagegen_invocation_evidence"] = _require_imagegen_evidence(attempt["imagegen_invocation_evidence"])
            normalized.append(record)
            continue
        if status != "generated":
            raise ValueError("attempt.status must be 'generated' or 'generation_failed'")
        candidate_path = candidate_paths.get(expected_index)
        if candidate_path is None:
            raise ValueError(f"missing state-owned candidate path for attempt {expected_index}")
        _require_path_under_roots(candidate_path, allowed_roots, label=f"attempt_{expected_index}_image_path")
        image_bytes, image_sha256 = _verify_existing_image(candidate_path)
        prompt_summary = str(attempt.get("imagegen_prompt_summary", "") or "").strip()
        if not prompt_summary:
            raise ValueError("generated cleanup attempt requires non-empty imagegen_prompt_summary")
        evidence = _require_imagegen_evidence(attempt.get("imagegen_invocation_evidence", {}))
        description = _normalized_description(source_image=source_image, candidate_image=candidate_path, attempt=attempt, object_name_hint=object_name_hint, user_hints=user_hints)
        validation = _normalize_validation(attempt.get("validation"), attempt_index=expected_index)
        normalized.append({
            "attempt_index": expected_index,
            "status": status,
            "candidate_image_path": str(candidate_path),
            "candidate_image_bytes": image_bytes,
            "candidate_image_sha256": image_sha256,
            "imagegen_prompt_summary": prompt_summary,
            "imagegen_invocation_evidence": evidence,
            "generated_object_description": description,
            "validation": validation,
        })
    return normalized


def _validate_selection(attempts: list[dict[str, Any]], *, selected_attempt_index: int, selection_reason: str) -> tuple[dict[str, Any], str]:
    if not selection_reason.strip():
        raise ValueError("selection_reason must be non-empty")
    by_index = {attempt["attempt_index"]: attempt for attempt in attempts}
    selected = by_index.get(selected_attempt_index)
    if not selected or selected["status"] != "generated":
        raise ValueError("selected_attempt_index must identify a generated attempt")
    generated = [attempt for attempt in attempts if attempt["status"] == "generated"]
    if not generated:
        raise ValueError("at least one generated cleanup attempt is required")
    validation_passes = [attempt for attempt in generated if attempt["validation"]["verdict"] == "pass"]
    if len(attempts) == 2 and attempts[0]["status"] == "generated" and attempts[0]["validation"]["verdict"] == "pass":
        raise ValueError("attempt 2 is forbidden after attempt 1 passes validation")
    if validation_passes:
        if selected["validation"]["verdict"] != "pass":
            raise ValueError("a validation-passing attempt must be selected")
        if selected_attempt_index == 2 and len(attempts) != 2:
            raise ValueError("attempt 2 cannot be selected without two attempts")
        return selected, "validation_pass"
    if len(attempts) != IMAGE_CLEANUP_MAX_ATTEMPTS:
        raise ValueError("a validation-failed candidate may be selected only after two exhausted attempts")
    return selected, "best_available_after_exhaustion"


def register_image_cleanup_result(
    source_image: Path,
    candidate_paths: Mapping[int, Path],
    cleaned_image_path: Path,
    object_description_output_path: Path,
    *,
    attempts: list[dict[str, Any]],
    selected_attempt_index: int,
    selection_reason: str,
    output_report_path: Path,
    object_name_hint: str = "",
    user_hints: str = "",
    repo_root: Path | None = None,
    allowed_output_roots: tuple[Path, ...] = (),
) -> StageRunResult:
    root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_source = _path_for_repo(source_image, root).resolve()
    resolved_cleaned = _path_for_repo(cleaned_image_path, root).resolve()
    resolved_description = _path_for_repo(object_description_output_path, root).resolve()
    resolved_report = _path_for_repo(output_report_path, root).resolve()
    allowed_roots = _resolve_allowed_output_roots(repo_root=root, allowed_output_roots=allowed_output_roots)
    if not resolved_source.is_file():
        raise FileNotFoundError(f"source image is missing or not a file: {resolved_source}")
    for path, label in ((resolved_cleaned, "cleaned_image_path"), (resolved_description, "object_description_output_path"), (resolved_report, "output_report_path")):
        _require_path_under_roots(path, allowed_roots, label=label)
    resolved_candidates = {int(index): _path_for_repo(path, root).resolve() for index, path in candidate_paths.items()}
    normalized_attempts = _normalize_attempts(attempts=attempts, candidate_paths=resolved_candidates, source_image=resolved_source, object_name_hint=object_name_hint, user_hints=user_hints, allowed_roots=allowed_roots)
    selected, acceptance_basis = _validate_selection(normalized_attempts, selected_attempt_index=int(selected_attempt_index), selection_reason=str(selection_reason))
    selected_candidate = Path(selected["candidate_image_path"])
    resolved_cleaned.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_candidate, resolved_cleaned)
    image_bytes, image_sha256 = _verify_existing_image(resolved_cleaned)
    description = dict(selected["generated_object_description"])
    resolved_description.parent.mkdir(parents=True, exist_ok=True)
    resolved_description.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not object_description_text(description).strip():
        raise RuntimeError("image cleanup registration produced an empty object description")
    report = {
        "schema_version": IMAGE_CLEANUP_REPORT_SCHEMA_VERSION,
        "max_attempts": IMAGE_CLEANUP_MAX_ATTEMPTS,
        "source_image": str(resolved_source),
        "output_image_path": str(resolved_cleaned),
        "object_description_path": str(resolved_description),
        "report_path": str(resolved_report),
        "attempts": normalized_attempts,
        "selection": {"selected_attempt_index": selected_attempt_index, "selection_reason": str(selection_reason).strip(), "acceptance_basis": acceptance_basis},
        "provenance": {"generator": "codex_builtin_imagegen", "imagegen_invocation_evidence": selected["imagegen_invocation_evidence"]},
        "imagegen_prompt_summary": selected["imagegen_prompt_summary"],
        "generated_object_description": description,
        "image_bytes": image_bytes,
        "image_sha256": image_sha256,
    }
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    resolved_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {"status": "registered", "source_image": str(resolved_source), "output_image_path": str(resolved_cleaned), "object_description_path": str(resolved_description), "cleanup_report_path": str(resolved_report), "description": description, "attempts": normalized_attempts, "selection": report["selection"], "report": report}
    return _dict_stage_result(name="image_cleanup", stage=Stage.IMAGE_CLEANUP, payload=payload, artifacts=(
        _artifact(ArtifactRole.OBJECT_DESCRIPTION, resolved_description, Stage.IMAGE_CLEANUP),
        _artifact(ArtifactRole.CLEANED_SOURCE_IMAGE, resolved_cleaned, Stage.IMAGE_CLEANUP),
        _artifact(ArtifactRole.REPORT, resolved_report, Stage.IMAGE_CLEANUP),
    ))


__all__ = ["IMAGE_CLEANUP_MAX_ATTEMPTS", "IMAGE_CLEANUP_REPORT_SCHEMA_VERSION", "VALIDATION_CHECKS", "register_image_cleanup_result"]
