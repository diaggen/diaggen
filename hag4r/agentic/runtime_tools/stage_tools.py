from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Callable
from hag4r.agentic.diagnostic_cues import diagnostic_hint_text_for_route
from hag4r.agentic.genesis_vlm_schemas import normalize_persisted_vlm_final_recommendation
from hag4r.agentic.runtime_state import (
    FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE,
    POST_MESH_TEXTURE_STAGE_SEQUENCE,
    DIAGNOSTIC_EXPORT_POLICY,
    _is_final_revise_export_policy,
    append_history,
    active_revision_root,
    build_full_image_revision_paths,
    load_state,
    mark_active_revision_ready_for_export,
    promote_exported_revision,
    record_stage,
    record_tool_call,
    resolve_export_revision,
    save_state,
    state_path,
    validate_revision_path_ownership,
    worker_cuda_visible_devices,
)
from hag4r.agentic.state import (
    ArtifactRef,
    ArtifactRole,
    SimDiagnosticRoute,
    Stage,
    StageRunResult,
    to_json_dict,
)
from hag4r.tools.final_export import run_final_export
from hag4r.tools.image_cleanup import IMAGE_CLEANUP_MAX_ATTEMPTS, register_image_cleanup_result
from hag4r.tools.material_inference import (
    DEFAULT_MATERIAL_MAX_ATTEMPTS,
    MaterialInferenceContext,
    build_gpt_staged_material_inference_request,
    build_material_index_context,
    build_material_output_payload,
    load_material_part_labels,
    material_validation_attempt,
    material_scratch_path_for_output,
    validate_agent_material_fill_mode_payload,
    validate_agent_material_geometry_payload,
    validate_agent_material_repair_payload,
    validate_agent_material_semantics_payload,
    write_material_output_payload,
)
from hag4r.tools.mesh_processing import (
    run_assign_params,
    run_monolithic_mesh,
)
from hag4r.tools.volumetric_meshing import (
    MESH_PROCESSING_PLAN_SCHEMA_VERSION,
    build_mesh_processing_plan,
    load_color_locked_material_predictions,
    resolve_disconnected_tet_components_from_request_json,
)
from hag4r.tools.object_description import object_description_text
from hag4r.tools.omnipart import run_omnipart_generation
from hag4r.tools.post_mesh_texture import (
    run_post_mesh_texture,
    validate_post_mesh_texture_bundle,
)
from hag4r.tools.segmentation import run_sam3_omnipart_2d_segmentation
SEGMENTATION_MAX_FINAL_PARTS = 6
DEFAULT_SEGMENTATION_MAX_ATTEMPTS = 3
STAGE_LOCAL_TOOL_NAMES: tuple[str, ...] = (
    "register_image_cleanup_stage",
    "run_sam3_omnipart_2d_segmentation_stage",
    "run_omnipart_generate_parts_stage",
    "load_material_part_labels_stage",
    "build_material_indexed_parts_stage",
    "run_material_geometry_analysis_stage",
    "decide_material_fill_mode_stage",
    "run_material_semantics_inference_stage",
    "validate_repair_material_predictions_stage",
    "write_inferred_material_params_stage",
    "run_assign_params_to_prims_stage",
    "select_mesh_processing_fidelities_stage",
    "run_combined_to_monolithic_stage",
    "resolve_disconnected_tet_components_stage",
    "run_post_mesh_texture_stage",
    "run_final_export_bundle_stage",
)
def _worker_gpu_env(state: dict[str, Any]) -> dict[str, str] | None:
    cuda_visible_devices = worker_cuda_visible_devices(state)
    if cuda_visible_devices is None:
        return None
    return {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}
def _require_state(run_root: str | Path) -> dict[str, Any]:
    state = load_state(run_root)
    if not state:
        raise FileNotFoundError(f"runtime state does not exist: {state_path(run_root)}")
    return state
def _require_stage(state: dict[str, Any], stage_name: str) -> None:
    if stage_name not in state.get("stages", {}):
        raise KeyError(f"stage is not in runtime state: {stage_name}")
    planned = list(state.get("planned_global_stage_sequence", ()))
    if stage_name not in planned:
        return
    for prior_stage in planned[: planned.index(stage_name)]:
        prior = state.get("stages", {}).get(prior_stage, {})
        if not isinstance(prior, dict) or prior.get("ok") is not True:
            raise RuntimeError(f"stage prerequisite is incomplete: {prior_stage}")
def _repo_root(state: dict[str, Any]) -> Path:
    return Path(str(state["repo_root"])).expanduser().resolve()
def _log_dir(state: dict[str, Any]) -> Path:
    return Path(str(state["paths"]["logs_dir"])).expanduser().resolve()
def _path(state: dict[str, Any], key: str) -> Path:
    value = state.get("paths", {}).get(key)
    if not value:
        raise KeyError(f"runtime path is missing: paths.{key}")
    return Path(str(value)).expanduser().resolve()
def _optional_path(state: dict[str, Any], key: str) -> Path | None:
    value = state.get("paths", {}).get(key)
    return Path(str(value)).expanduser().resolve() if value else None
def _diagnostic_cues(state: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    route_state = state.get("route_state", {})
    cues = route_state.get("sim_diagnostic_cues", ()) if isinstance(route_state, dict) else ()
    if not cues:
        cues = state.get("sim_diagnostic_cues", ())
    return tuple(cue for cue in cues if isinstance(cue, dict))
def _diagnostic_hint(state: dict[str, Any], route: SimDiagnosticRoute) -> str:
    return diagnostic_hint_text_for_route(_diagnostic_cues(state), route)
def _stage_runtime_diagnostic_repair_plan(state: dict[str, Any], stage_key: str) -> str:
    stage_runtime = state.get("stage_runtime", {})
    stage_record = stage_runtime.get(stage_key, {}) if isinstance(stage_runtime, dict) else {}
    if not isinstance(stage_record, dict):
        return ""
    return str(stage_record.get("diagnostic_repair_plan", "") or "").strip()
def _record_stage_runtime_diagnostic_repair_plan(
    state: dict[str, Any],
    *,
    stage_key: str,
    tool_name: str,
    diagnostic_repair_plan: str,
) -> dict[str, Any]:
    plan = diagnostic_repair_plan.strip()
    if not plan:
        return state
    stage_record = state.setdefault("stage_runtime", {}).setdefault(stage_key, {})
    stage_record["diagnostic_repair_plan"] = plan
    stage_record.setdefault("diagnostic_repair_plan_updates", []).append(
        {
            "tool_name": tool_name,
            "diagnostic_repair_plan": plan,
        }
    )
    return state
def _diagnostic_hint_with_stage_plan(state: dict[str, Any], route: SimDiagnosticRoute, stage_key: str) -> str:
    stage_plan = _stage_runtime_diagnostic_repair_plan(state, stage_key)
    return "\n\n".join(
        text
        for text in (
            _diagnostic_hint(state, route),
            f"Codex stage digested diagnostic repair plan:\n{stage_plan}" if stage_plan else "",
        )
        if text.strip()
    )
def _object_description(state: dict[str, Any]) -> str:
    if (
        state.get("run_mode") == "full_image"
        or state.get("diagnostics_reroute_and_export") is True
    ):
        description_path = _optional_path(state, "object_description_path")
        if description_path is None or not description_path.exists():
            raise RuntimeError(
                "full-image downstream stages require the object-description artifact "
                "generated by the merged image cleanup stage"
            )
        return object_description_text(description_path)
    inputs = state.get("inputs", {})
    manual = str(inputs.get("object_description", "") or "")
    if manual:
        return manual
    description_path = _optional_path(state, "object_description_path")
    if description_path is not None and description_path.exists():
        return object_description_text(description_path)
    return str(state.get("object_name", ""))
def _summarize_diagnostic_cues(cues: tuple[dict[str, Any], ...]) -> str:
    summaries: list[str] = []
    for cue in cues:
        route = str(cue.get("route", "") or "").strip()
        diagnostic_cues = cue.get("diagnostic_cues", ())
        cue_text = " | ".join(
            str(item).strip()
            for item in diagnostic_cues
            if isinstance(item, str) and item.strip()
        ) if isinstance(diagnostic_cues, list | tuple) else ""
        if route and cue_text:
            summaries.append(f"{route}: {cue_text}")
    return " | ".join(summaries)


def _diagnostics_terminal_accepts_export(state: dict[str, Any]) -> bool:
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics.get("enabled"):
        return False
    diagnostic_stage = state.get("stages", {}).get("genesis_live_diagnostic_loop", {})
    if not isinstance(diagnostic_stage, dict) or diagnostic_stage.get("ok") is not True:
        return False
    terminal = diagnostics.get("terminal", {})
    if not isinstance(terminal, dict) or terminal.get("status") != "success":
        return False
    route = str(terminal.get("route", "") or "")
    return bool(terminal.get("ready")) or route == SimDiagnosticRoute.ACCEPT.value


def _artifact(role: ArtifactRole, path: Path, stage: Stage, *, required: bool = True) -> ArtifactRef:
    return ArtifactRef(role=role, path=path, stage=stage, required=required)
def _stage_summary(result: StageRunResult) -> dict[str, Any]:
    status = str(result.metrics.get("status_override", "success" if result.success else "failed"))
    return {
        "name": result.name,
        "status": status,
        "ok": bool(result.success),
        "stage": result.stage.value,
        "returncode": result.returncode,
        "stdout_path": str(result.stdout_path) if result.stdout_path else None,
        "stderr_path": str(result.stderr_path) if result.stderr_path else None,
        "error": result.error,
        "artifacts": [to_json_dict(artifact) for artifact in result.artifacts],
        "metrics": to_json_dict(result.metrics),
    }


def _final_revise_export_manifest_provenance(state: dict[str, Any]) -> dict[str, Any]:
    decision = state.get("route_state", {}).get("orchestrator_diagnostic_decision", {})
    if not isinstance(decision, dict) or decision.get("diagnostic_export_policy") != DIAGNOSTIC_EXPORT_POLICY:
        return {}
    if not _is_final_revise_export_policy(state, decision):
        raise RuntimeError("final-revise export manifest provenance failed policy validation")
    terminal = state.get("diagnostics", {}).get("terminal", {})
    recommendation = terminal.get("recommendation") if isinstance(terminal, dict) else None
    if not isinstance(recommendation, dict):
        raise RuntimeError("final-revise export manifest is missing the durable child recommendation")
    recommendation = normalize_persisted_vlm_final_recommendation(recommendation)
    cues = list(recommendation.get("diagnostic_cues", []))
    active_revision = str(state.get("active_revision", "") or "")
    return {
        "export_revision": active_revision,
        "accepted_revision": None,
        "diagnostics_accepted": False,
        "diagnostic_export_policy": DIAGNOSTIC_EXPORT_POLICY,
        "diagnostic_verdict": "revise",
        "child_recommendation": "revise",
        "child_route": str(recommendation["route"]),
        "diagnostic_cues": cues,
    }


def _rewrite_final_export_manifest_last(state: dict[str, Any]) -> None:
    manifest_path = _path(state, "final_export_manifest_path")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted_revision"] = state.get("accepted_revision")
    manifest["export_revision"] = state.get("exported_revision") or state.get("accepted_revision")
    manifest["asset_refinement_source_dir"] = str(active_revision_root(state))
    provenance = _final_revise_export_manifest_provenance(state)
    if provenance:
        manifest.update(provenance)
    else:
        for key in (
            "diagnostics_accepted",
            "diagnostic_export_policy",
            "diagnostic_verdict",
            "child_recommendation",
            "child_route",
            "diagnostic_cues",
        ):
            manifest.pop(key, None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def _run_state_stage(
    run_root: str,
    *,
    stage_name: str,
    tool_name: str,
    inputs_summary: dict[str, Any],
    invoke: Callable[[dict[str, Any]], StageRunResult],
    allow_retryable_failure: bool = False,
    after_success: Callable[[dict[str, Any]], None] | None = None,
    after_failure: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    state = _require_state(run_root)
    _require_stage(state, stage_name)
    append_history(
        run_root,
        tool_name,
        f"started {stage_name}",
        event="stage_tool_started",
        detail={"stage_name": stage_name, "tool_name": tool_name},
    )
    record_tool_call(
        run_root,
        stage_name,
        tool_name,
        status="started",
        inputs_summary=inputs_summary,
    )
    try:
        result = invoke(state)
    except Exception as exc:
        message = str(exc)
        record_tool_call(run_root, stage_name, tool_name, status="failed", error=message)
        record_stage(run_root, stage_name, False, error=message, status="failed")
        if after_failure is not None:
            after_failure(message)
        append_history(
            run_root,
            tool_name,
            message,
            event="stage_tool_failed",
            detail={"stage_name": stage_name, "tool_name": tool_name, "error": message},
        )
        raise RuntimeError(f"{stage_name} failed: {message}") from exc
    summary = _stage_summary(result)
    record_tool_call(
        run_root,
        stage_name,
        tool_name,
        status=str(summary["status"]),
        outputs_summary={
            "ok": summary["ok"],
            "returncode": summary["returncode"],
            "artifact_count": len(summary["artifacts"]),
        },
        error=str(summary["error"] or ""),
    )
    updated_state = record_stage(
        run_root,
        stage_name,
        bool(summary["ok"]),
        outputs={"result": summary},
        stats={"returncode": summary["returncode"], "metrics": summary["metrics"]},
        artifacts={"expected": summary["artifacts"]},
        error=str(summary["error"] or ""),
        status=str(summary["status"]),
    )
    if summary["ok"] and after_success is not None:
        try:
            after_success(updated_state)
        except Exception as exc:
            message = str(exc)
            record_tool_call(
                run_root,
                stage_name,
                tool_name,
                status="failed",
                error=message,
            )
            record_stage(
                run_root,
                stage_name,
                False,
                outputs={"result": summary},
                stats={"returncode": summary["returncode"], "metrics": summary["metrics"]},
                artifacts={"expected": summary["artifacts"]},
                error=message,
                status="failed",
            )
            failed_state = load_state(run_root)
            failed_state["completed_global_stage_sequence"] = [
                name
                for name in failed_state.get("completed_global_stage_sequence", [])
                if name != stage_name
            ]
            failed_state["status"] = "failed"
            save_state(failed_state, run_root)
            if after_failure is not None:
                after_failure(message)
            append_history(
                run_root,
                tool_name,
                message,
                event="stage_tool_failed",
                detail={
                    "stage_name": stage_name,
                    "tool_name": tool_name,
                    "error": message,
                    "failure_phase": "after_success",
                },
            )
            raise RuntimeError(f"{stage_name} failed after runner success: {message}") from exc
    if (
        summary["ok"]
        and stage_name == "hag4r_final_export_bundle"
        and (
            updated_state.get("run_mode") == "full_image"
            or updated_state.get("diagnostics_reroute_and_export") is True
        )
    ):
        _rewrite_final_export_manifest_last(load_state(run_root))
    if not summary["ok"]:
        if after_failure is not None:
            after_failure(str(summary["error"] or f"{stage_name} failed"))
        append_history(
            run_root,
            tool_name,
            str(summary["error"] or f"{stage_name} failed"),
            event="stage_tool_failed",
            detail={"stage_name": stage_name, "tool_name": tool_name, "error": summary["error"]},
        )
        if allow_retryable_failure and bool(result.metrics.get("retryable")):
            return {"stage_name": stage_name, "tool_name": tool_name, "result": summary, "state_path": str(state_path(run_root))}
        raise RuntimeError(f"{stage_name} failed: {summary['error']}")
    append_history(
        run_root,
        tool_name,
        f"{summary['status']} {stage_name}",
        event="stage_tool_succeeded",
        detail={"stage_name": stage_name, "tool_name": tool_name, "status": summary["status"]},
    )
    return {"stage_name": stage_name, "tool_name": tool_name, "result": summary, "state_path": str(state_path(run_root))}
def _stage_tool_attempt_index(state: dict[str, Any], *, stage_name: str, tool_name: str) -> int:
    stage = state.get("stages", {}).get(stage_name, {})
    tool_calls = stage.get("tool_calls", ()) if isinstance(stage, dict) else ()
    previous_starts = sum(
        1
        for call in tool_calls
        if isinstance(call, dict)
        and call.get("tool_name") == tool_name
        and call.get("status") == "started"
    )
    return previous_starts + 1
def _record_segmentation_attempt(
    run_root: str,
    *,
    attempt: dict[str, Any],
) -> None:
    state = load_state(run_root)
    retry_state = state.setdefault("segmentation_retry", {})
    retry_state.setdefault("max_final_parts", SEGMENTATION_MAX_FINAL_PARTS)
    retry_state.setdefault("attempts", []).append(to_json_dict(attempt))
    retry_state["last_attempt"] = to_json_dict(attempt)
    if attempt.get("status") == "accepted":
        retry_state["accepted_attempt"] = to_json_dict(attempt)
    save_state(state, run_root)
def _summarize_material_stage_attempts(stage_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for attempt in stage_attempts:
        if not isinstance(attempt, dict):
            continue
        summary = {
            "status": str(attempt.get("status", "")),
            "stage": str(attempt.get("stage", "")),
            "error": str(attempt.get("error", "")),
        }
        if attempt.get("attempt_index") is not None:
            summary["attempt_index"] = int(attempt["attempt_index"])
        summaries.append(summary)
    return summaries
def _record_material_retry_phase(
    run_root: str,
    *,
    phase: str,
    status: str,
    scratch_path: Path,
    stage_attempts: list[dict[str, Any]],
) -> None:
    attempts = _summarize_material_stage_attempts(stage_attempts)
    attempt_count = sum(1 for attempt in attempts if "attempt_index" in attempt)
    failed_attempt_count = sum(
        1
        for attempt in attempts
        if "attempt_index" in attempt and attempt.get("status") == "failed"
    )
    phase_record = {
        "phase": phase,
        "status": status,
        "max_attempts": DEFAULT_MATERIAL_MAX_ATTEMPTS,
        "attempt_count": attempt_count,
        "failed_attempt_count": failed_attempt_count,
        "scratch_path": str(scratch_path),
        "attempts": attempts,
    }
    state = load_state(run_root)
    retry_state = state.setdefault("material_retry", {})
    retry_state["max_attempts"] = DEFAULT_MATERIAL_MAX_ATTEMPTS
    retry_state.setdefault("attempts", []).append(to_json_dict(phase_record))
    retry_state["last_attempt"] = to_json_dict(phase_record)
    if status == "accepted":
        retry_state["last_successful_phase"] = to_json_dict(phase_record)
        if phase == "validation_repair":
            retry_state["accepted_attempt"] = to_json_dict(phase_record)
    elif status == "max_attempts_exhausted":
        retry_state.pop("accepted_attempt", None)
    save_state(state, run_root)
    append_history(
        run_root,
        "material_retry",
        f"{phase} {status} with {attempt_count}/{DEFAULT_MATERIAL_MAX_ATTEMPTS} stage attempts",
        event="material_retry_recorded",
        detail=to_json_dict(phase_record),
    )
MATERIAL_STAGE_NAME = "hag4r_gpt_staged_material_inference"
MATERIAL_SCRATCH_SCHEMA_VERSION = "hag4r-material-inference-scratch-v1"
def _material_request_from_state(state: dict[str, Any]):
    return build_gpt_staged_material_inference_request(
        image_dir=_path(state, "omnipart_output_dir"),
        part_labels_path=_path(state, "part_labels_path"),
        output_path=_path(state, "inferred_params_path"),
        object_description=_object_description(state),
        object_description_path=_optional_path(state, "object_description_path"),
        processed_image_path=_optional_path(state, "processed_white_bg_png_path")
        or (_path(state, "omnipart_output_dir") / f"{state['object_name']}_processed.png"),
        diagnostic_cues=_diagnostic_cues(state),
        diagnostic_hint_text=_diagnostic_hint_with_stage_plan(
            state,
            SimDiagnosticRoute.MATERIAL_INFERENCE,
            "material_inference",
        ),
    )
def _record_material_diagnostic_repair_plan_if_present(
    state: dict[str, Any],
    *,
    run_root: str,
    tool_name: str,
    diagnostic_repair_plan: str,
) -> dict[str, Any]:
    if not diagnostic_repair_plan.strip():
        return state
    state = load_state(run_root) or state
    state = _record_stage_runtime_diagnostic_repair_plan(
        state,
        stage_key="material_inference",
        tool_name=tool_name,
        diagnostic_repair_plan=diagnostic_repair_plan,
    )
    save_state(state, run_root)
    return state
def _material_scratch_path(state: dict[str, Any]) -> Path:
    return material_scratch_path_for_output(_path(state, "inferred_params_path"))
def _read_material_scratch(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"material inference scratch does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
def _write_material_scratch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
def _scratch_context(request, scratch: dict[str, Any]) -> MaterialInferenceContext:
    context = scratch.get("context")
    if not isinstance(context, dict):
        raise RuntimeError("material inference scratch is missing `context`; run build_material_indexed_parts_stage first")
    return MaterialInferenceContext(
        request=request,
        part_colors=list(context["part_colors"]),
        indexed_parts=list(context["indexed_parts"]),
        view_image_paths=[(str(item[0]), str(item[1])) for item in context["view_image_paths"]],
        processed_image_path=Path(str(context["processed_image_path"])),
        part_visibility_summary=list(context["part_visibility_summary"]),
        part_geometry_summary=list(context["part_geometry_summary"]),
        pairwise_part_relations=list(context["pairwise_part_relations"]),
    )
def _scratch_context_payload(context: MaterialInferenceContext) -> dict[str, Any]:
    return {
        "part_colors": context.part_colors,
        "indexed_parts": context.indexed_parts,
        "view_image_paths": [[name, path] for name, path in context.view_image_paths],
        "processed_image_path": str(context.processed_image_path),
        "part_visibility_summary": context.part_visibility_summary,
        "part_geometry_summary": context.part_geometry_summary,
        "pairwise_part_relations": context.pairwise_part_relations,
    }
def _run_material_phase_tool(
    run_root: str,
    *,
    tool_name: str,
    inputs_summary: dict[str, Any],
    invoke: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    state = _require_state(run_root)
    _require_stage(state, MATERIAL_STAGE_NAME)
    append_history(
        run_root,
        tool_name,
        f"started {MATERIAL_STAGE_NAME}",
        event="stage_tool_started",
        detail={"stage_name": MATERIAL_STAGE_NAME, "tool_name": tool_name},
    )
    record_tool_call(
        run_root,
        MATERIAL_STAGE_NAME,
        tool_name,
        status="started",
        inputs_summary=inputs_summary,
    )
    try:
        payload = invoke(state)
    except Exception as exc:
        message = str(exc)
        record_tool_call(run_root, MATERIAL_STAGE_NAME, tool_name, status="failed", error=message)
        record_stage(run_root, MATERIAL_STAGE_NAME, False, error=message, status="failed")
        append_history(
            run_root,
            tool_name,
            message,
            event="stage_tool_failed",
            detail={"stage_name": MATERIAL_STAGE_NAME, "tool_name": tool_name, "error": message},
        )
        raise RuntimeError(f"{MATERIAL_STAGE_NAME} failed: {message}") from exc
    record_tool_call(
        run_root,
        MATERIAL_STAGE_NAME,
        tool_name,
        status="success",
        outputs_summary=payload,
    )
    append_history(
        run_root,
        tool_name,
        f"success {MATERIAL_STAGE_NAME}",
        event="stage_tool_succeeded",
        detail={"stage_name": MATERIAL_STAGE_NAME, "tool_name": tool_name, "status": "success"},
    )
    return {
        "stage_name": MATERIAL_STAGE_NAME,
        "tool_name": tool_name,
        "result": {"status": "success", "ok": True, "metrics": payload},
        "state_path": str(state_path(run_root)),
    }
def register_image_cleanup_stage(
    run_root: str,
    attempts: list[dict[str, Any]],
    selected_attempt_index: int,
    selection_reason: str,
) -> dict[str, Any]:
    """Register one selected candidate from the bounded image-cleanup attempt loop."""
    stage_name = "image_cleanup"
    tool_name = "register_image_cleanup_stage"
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("attempts must be a non-empty list")
    if len(attempts) > IMAGE_CLEANUP_MAX_ATTEMPTS:
        raise ValueError(f"image cleanup permits at most {IMAGE_CLEANUP_MAX_ATTEMPTS} attempts")
    def invoke(state: dict[str, Any]) -> StageRunResult:
        hint_inputs = state.get("inputs", {})
        user_hints = "\n".join(
            text
            for text in (
                f"Manual object-description hint: {hint_inputs.get('object_description')}"
                if str(hint_inputs.get("object_description", "") or "").strip()
                else "",
                str(hint_inputs.get("user_hints", "") or ""),
            )
            if text.strip()
        )
        return register_image_cleanup_result(
            _path(state, "source_image"),
            {
                1: _path(state, "image_cleanup_attempt_1_path"),
                2: _path(state, "image_cleanup_attempt_2_path"),
            },
            _path(state, "cleaned_image_path"),
            _path(state, "object_description_path"),
            output_report_path=_path(state, "cleanup_report_path"),
            attempts=attempts,
            selected_attempt_index=selected_attempt_index,
            selection_reason=selection_reason,
            object_name_hint=str(state.get("object_name", "")),
            user_hints=user_hints,
            repo_root=_repo_root(state),
            allowed_output_roots=(
                Path(str(state["run_root"])).expanduser().resolve(),
                _path(state, "image_cleanup_dir"),
                _repo_root(state) / "outputs",
            ),
        )
    result = _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "attempt_count": len(attempts),
            "selected_attempt_index": selected_attempt_index,
            "selection_reason": str(selection_reason or "").strip(),
        },
        invoke=invoke,
    )
    state = load_state(run_root)
    state.setdefault("runtime_events", []).append(
        {
            "event": "codex_imagegen_cleanup_registered",
            "stage_name": stage_name,
            "tool_name": tool_name,
            "attempt_count": len(attempts),
            "selected_attempt_index": selected_attempt_index,
            "cleaned_image_path": str(_path(state, "cleaned_image_path")),
            "cleanup_report_path": str(_path(state, "cleanup_report_path")),
        }
    )
    save_state(state, run_root)
    return result
def run_sam3_omnipart_2d_segmentation_stage(
    run_root: str,
    sam3_prompt: str,
    max_attempts: int = DEFAULT_SEGMENTATION_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Run SAM3-owned OmniPart 2D segmentation using an agent-authored SAM3 prompt."""
    stage_name = "sam3_omnipart_2d_segmentation"
    tool_name = "run_sam3_omnipart_2d_segmentation_stage"
    def invoke(state: dict[str, Any]) -> StageRunResult:
        if not str(sam3_prompt or "").strip():
            raise ValueError("sam3_prompt must be non-empty")
        if int(max_attempts) <= 0:
            raise ValueError("max_attempts must be positive")
        attempt_index = _stage_tool_attempt_index(state, stage_name=stage_name, tool_name=tool_name)
        _object_description(state)
        description_path = _optional_path(state, "object_description_path")
        result = run_sam3_omnipart_2d_segmentation(
            _path(state, "cleaned_image_path"),
            sam3_prompt,
            _path(state, "segmentation_dir"),
            object_description_path=description_path if description_path and description_path.exists() else None,
            repo_root=_repo_root(state),
            log_dir=_log_dir(state),
            env=_worker_gpu_env(state),
        )
        if not result.success:
            return result
        manifest_path = _path(state, "segmentation_manifest_path")
        if not manifest_path.exists():
            return result
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_part_count = int(manifest.get("final_part_count", manifest.get("part_count", -1)))
        attempt = {
            "attempt_index": attempt_index,
            "max_attempts": int(max_attempts),
            "max_final_parts": SEGMENTATION_MAX_FINAL_PARTS,
            "final_part_count": final_part_count,
            "segmentation_manifest_path": str(manifest_path),
            "sam3_prompt": sam3_prompt,
        }
        metrics = dict(result.metrics)
        metrics.update(attempt)
        if final_part_count <= SEGMENTATION_MAX_FINAL_PARTS:
            _record_segmentation_attempt(
                run_root,
                attempt={**attempt, "status": "accepted"},
            )
            return StageRunResult(
                name=result.name,
                stage=result.stage,
                success=True,
                returncode=result.returncode,
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                artifacts=result.artifacts,
                metrics=metrics,
                error=result.error,
            )
        retryable = attempt_index < int(max_attempts)
        status = "retry_required" if retryable else "max_attempts_exhausted"
        message = (
            f"segmentation final_part_count={final_part_count} exceeds cap "
            f"{SEGMENTATION_MAX_FINAL_PARTS} on attempt {attempt_index}/{int(max_attempts)}"
        )
        _record_segmentation_attempt(
            run_root,
            attempt={**attempt, "status": status},
        )
        return StageRunResult(
            name=result.name,
            stage=result.stage,
            success=False,
            returncode=result.returncode,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            artifacts=result.artifacts,
            metrics={**metrics, "retryable": retryable},
            error=message,
        )
    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root, "sam3_prompt": sam3_prompt, "max_attempts": max_attempts},
        invoke=invoke,
        allow_retryable_failure=True,
    )
def run_omnipart_generate_parts_stage(run_root: str) -> dict[str, Any]:
    """Run OmniPart generation using runtime state."""
    stage_name = "omnipart_generate_parts"
    tool_name = "run_omnipart_generate_parts_stage"
    def invoke(state: dict[str, Any]) -> StageRunResult:
        return run_omnipart_generation(
            _path(state, "segmentation_manifest_path"),
            _path(state, "omnipart_root"),
            repo_root=_repo_root(state),
            log_dir=_log_dir(state),
            env=_worker_gpu_env(state),
        )
    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root},
        invoke=invoke,
    )
def load_material_part_labels_stage(run_root: str, diagnostic_repair_plan: str = "") -> dict[str, Any]:
    """Load material-inference part labels and initialize the phase scratch file."""
    tool_name = "load_material_part_labels_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        part_label_payload = load_material_part_labels(request)
        scratch = {
            "schema_version": MATERIAL_SCRATCH_SCHEMA_VERSION,
            "request_payload": request.request_payload,
            "scratch_path": str(scratch_path),
            "part_label_payload": part_label_payload,
            "phases": ["part_labels_loaded"],
        }
        _write_material_scratch(scratch_path, scratch)
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(part_label_payload["part_colors"]),
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def build_material_indexed_parts_stage(run_root: str, diagnostic_repair_plan: str = "") -> dict[str, Any]:
    """Build exact indexed parts and image-derived material-inference summaries."""
    tool_name = "build_material_indexed_parts_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = build_material_index_context(request, scratch["part_label_payload"])
        scratch["context"] = _scratch_context_payload(context)
        scratch.setdefault("phases", []).append("indexed_context_built")
        _write_material_scratch(scratch_path, scratch)
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(context.indexed_parts),
            "indexed_parts": context.indexed_parts,
            "view_count": len(context.view_image_paths),
            "view_image_paths": [[name, path] for name, path in context.view_image_paths],
            "processed_image_path": str(context.processed_image_path),
            "part_visibility_summary": context.part_visibility_summary,
            "part_geometry_summary": context.part_geometry_summary,
            "pairwise_part_relations": context.pairwise_part_relations,
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def run_material_geometry_analysis_stage(
    run_root: str,
    geometry_payload: dict[str, Any],
    stage_notes: list[Any] | None = None,
    diagnostic_repair_plan: str = "",
) -> dict[str, Any]:
    """Validate and store the agent-authored index-locked material geometry analysis."""
    tool_name = "run_material_geometry_analysis_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = _scratch_context(request, scratch)
        geometry_result, canonical_geometry = validate_agent_material_geometry_payload(
            geometry_payload,
            context,
        )
        attempts = [
            material_validation_attempt(status="success", stage="part_geometry", attempt_index=1)
        ]
        scratch["geometry"] = {
            "result": geometry_result,
            "canonical_geometry": canonical_geometry,
            "stage_notes": list(stage_notes or []),
            "stage_attempts": attempts,
        }
        scratch.setdefault("phases", []).append("geometry_analyzed")
        _write_material_scratch(scratch_path, scratch)
        _record_material_retry_phase(
            run_root,
            phase="geometry",
            status="accepted",
            scratch_path=scratch_path,
            stage_attempts=attempts,
        )
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(canonical_geometry),
            "attempt_count": len(attempts),
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "part_count": len(geometry_payload.get("parts", [])) if isinstance(geometry_payload, dict) else None,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def decide_material_fill_mode_stage(
    run_root: str,
    fill_mode_payload: dict[str, Any],
    stage_notes: list[Any] | None = None,
    diagnostic_repair_plan: str = "",
) -> dict[str, Any]:
    """Validate and store the agent-authored per-part material fill-mode decision."""
    tool_name = "decide_material_fill_mode_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = _scratch_context(request, scratch)
        if not isinstance(scratch.get("geometry"), dict):
            raise RuntimeError("material inference scratch is missing geometry results")
        fill_mode_result, canonical_fill_modes = validate_agent_material_fill_mode_payload(
            fill_mode_payload,
            context,
        )
        attempts = [
            material_validation_attempt(status="success", stage="fill_mode", attempt_index=1)
        ]
        mode_counts = {mode: 0 for mode in ("solid_fill", "hollow_wall")}
        for entry in canonical_fill_modes:
            mode_counts[str(entry["volume_fill_mode"])] += 1
        scratch["fill_mode"] = {
            "result": fill_mode_result,
            "canonical_fill_modes": canonical_fill_modes,
            "stage_notes": list(stage_notes or []),
            "stage_attempts": attempts,
        }
        scratch.setdefault("phases", []).append("fill_mode_decided")
        _write_material_scratch(scratch_path, scratch)
        _record_material_retry_phase(
            run_root,
            phase="fill_mode",
            status="accepted",
            scratch_path=scratch_path,
            stage_attempts=attempts,
        )
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(canonical_fill_modes),
            "mode_counts": mode_counts,
            "attempt_count": len(attempts),
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "part_count": len(fill_mode_payload.get("parts", [])) if isinstance(fill_mode_payload, dict) else None,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def run_material_semantics_inference_stage(
    run_root: str,
    material_payload: dict[str, Any],
    stage_notes: list[Any] | None = None,
    diagnostic_repair_plan: str = "",
) -> dict[str, Any]:
    """Validate and store the agent-authored material semantics and numeric candidate payload."""
    tool_name = "run_material_semantics_inference_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = _scratch_context(request, scratch)
        geometry = scratch.get("geometry")
        fill_mode = scratch.get("fill_mode")
        if not isinstance(geometry, dict):
            raise RuntimeError("material inference scratch is missing geometry results")
        if not isinstance(fill_mode, dict):
            raise RuntimeError("material inference scratch is missing fill-mode decision")
        material_result, canonical_candidate_parts = validate_agent_material_semantics_payload(
            material_payload,
            context,
        )
        attempts = [
            material_validation_attempt(status="success", stage="constitutive_parameters", attempt_index=1)
        ]
        scratch["semantics"] = {
            "result": material_result,
            "canonical_candidate_parts": canonical_candidate_parts,
            "stage_notes": list(stage_notes or []),
            "stage_attempts": attempts,
        }
        scratch.setdefault("phases", []).append("semantics_inferred")
        _write_material_scratch(scratch_path, scratch)
        _record_material_retry_phase(
            run_root,
            phase="semantics",
            status="accepted",
            scratch_path=scratch_path,
            stage_attempts=attempts,
        )
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(material_result.get("parts", [])),
            "attempt_count": len(attempts),
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "part_count": len(material_payload.get("parts", [])) if isinstance(material_payload, dict) else None,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def validate_repair_material_predictions_stage(
    run_root: str,
    repair_payload: dict[str, Any] | None = None,
    diagnostic_repair_plan: str = "",
) -> dict[str, Any]:
    """Validate material predictions and, when needed, validate an agent-authored repair payload."""
    tool_name = "validate_repair_material_predictions_stage"
    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = _scratch_context(request, scratch)
        geometry = scratch.get("geometry")
        fill_mode = scratch.get("fill_mode")
        semantics = scratch.get("semantics")
        if not isinstance(geometry, dict):
            raise RuntimeError("material inference scratch is missing geometry results")
        if not isinstance(fill_mode, dict):
            raise RuntimeError("material inference scratch is missing fill-mode decision")
        if not isinstance(semantics, dict):
            raise RuntimeError("material inference scratch is missing semantic material results")
        try:
            material_result, canonical_parts = validate_agent_material_repair_payload(
                semantics["result"],
                context,
            )
        except Exception as validation_err:
            validation_attempt = material_validation_attempt(
                status="failed",
                stage="material_validation",
                error=str(validation_err),
            )
            if repair_payload is None:
                attempts = [validation_attempt]
                scratch["validated_material"] = {
                    "result": None,
                    "canonical_parts": [],
                    "stage_attempts": attempts,
                    "status": "failed",
                    "validation_error": str(validation_err),
                    "repair_error": "repair_payload was not provided",
                }
                scratch.setdefault("phases", []).append("material_validation_failed")
                _write_material_scratch(scratch_path, scratch)
                _record_material_retry_phase(
                    run_root,
                    phase="validation_repair",
                    status="max_attempts_exhausted",
                    scratch_path=scratch_path,
                    stage_attempts=attempts,
                )
                raise ValueError(
                    "Material output failed validation and no agent-authored repair_payload was provided. "
                    f"Validation error: {validation_err}"
                ) from validation_err
            try:
                material_result, canonical_parts = validate_agent_material_repair_payload(
                    repair_payload,
                    context,
                )
                validation_attempts = [
                    validation_attempt,
                    material_validation_attempt(
                        status="success",
                        stage="constitutive_parameters_repair",
                        attempt_index=1,
                    ),
                ]
            except Exception as repair_err:
                validation_attempts = [
                    validation_attempt,
                    material_validation_attempt(
                        status="failed",
                        stage="constitutive_parameters_repair",
                        error=str(repair_err),
                        attempt_index=1,
                    ),
                ]
                scratch["validated_material"] = {
                    "result": None,
                    "canonical_parts": [],
                    "stage_attempts": validation_attempts,
                    "status": "failed",
                    "validation_error": str(validation_err),
                    "repair_error": str(repair_err),
                }
                scratch.setdefault("phases", []).append("material_validation_failed")
                _write_material_scratch(scratch_path, scratch)
                _record_material_retry_phase(
                    run_root,
                    phase="validation_repair",
                    status="max_attempts_exhausted",
                    scratch_path=scratch_path,
                    stage_attempts=validation_attempts,
                )
                raise ValueError(
                    "Agent-authored material repair_payload failed validation. "
                    f"Original validation error: {validation_err}. Repair error: {repair_err}"
                ) from repair_err
        else:
            if repair_payload is not None:
                raise ValueError(
                    "repair_payload was provided, but the material semantics payload already passes validation"
                )
            validation_attempts = [
                material_validation_attempt(status="success", stage="material_validation")
            ]
        scratch["validated_material"] = {
            "result": material_result,
            "canonical_parts": canonical_parts,
            "stage_attempts": validation_attempts,
        }
        scratch.setdefault("phases", []).append("material_validated")
        _write_material_scratch(scratch_path, scratch)
        _record_material_retry_phase(
            run_root,
            phase="validation_repair",
            status="accepted",
            scratch_path=scratch_path,
            stage_attempts=validation_attempts,
        )
        return {
            "scratch_path": str(scratch_path),
            "part_count": len(canonical_parts),
            "repair_attempt_count": sum(
                1 for attempt in validation_attempts if attempt.get("stage") == "constitutive_parameters_repair"
            ),
        }
    return _run_material_phase_tool(
        run_root,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "has_repair_payload": repair_payload is not None,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def write_inferred_material_params_stage(run_root: str, diagnostic_repair_plan: str = "") -> dict[str, Any]:
    """Write final inferred material parameters after successful validation."""
    stage_name = MATERIAL_STAGE_NAME
    tool_name = "write_inferred_material_params_stage"
    def invoke(state: dict[str, Any]) -> StageRunResult:
        state = _record_material_diagnostic_repair_plan_if_present(
            state,
            run_root=run_root,
            tool_name=tool_name,
            diagnostic_repair_plan=diagnostic_repair_plan,
        )
        request = _material_request_from_state(state)
        scratch_path = _material_scratch_path(state)
        scratch = _read_material_scratch(scratch_path)
        context = _scratch_context(request, scratch)
        geometry = scratch.get("geometry")
        fill_mode = scratch.get("fill_mode")
        semantics = scratch.get("semantics")
        validated = scratch.get("validated_material")
        if not isinstance(geometry, dict):
            raise RuntimeError("material inference scratch is missing geometry results")
        if not isinstance(fill_mode, dict):
            raise RuntimeError("material inference scratch is missing fill-mode decision")
        if not isinstance(semantics, dict):
            raise RuntimeError("material inference scratch is missing semantic material results")
        if not isinstance(validated, dict):
            raise RuntimeError("material inference scratch is missing validated material predictions")
        payload = build_material_output_payload(
            context,
            geometry_result=geometry["result"],
            canonical_geometry=geometry["canonical_geometry"],
            geometry_stage_notes=geometry["stage_notes"],
            geometry_stage_attempts=geometry["stage_attempts"],
            fill_mode_result=fill_mode["result"],
            canonical_fill_modes=fill_mode["canonical_fill_modes"],
            fill_mode_stage_notes=fill_mode["stage_notes"],
            fill_mode_stage_attempts=fill_mode["stage_attempts"],
            material_result=validated["result"],
            canonical_parts=validated["canonical_parts"],
            semantics_stage_notes=semantics["stage_notes"],
            semantics_stage_attempts=semantics["stage_attempts"],
            repair_stage_attempts=validated["stage_attempts"],
        )
        write_metadata = write_material_output_payload(payload, request.output_path)
        scratch["final_payload_path"] = str(request.output_path)
        scratch["final_payload_summary"] = write_metadata
        scratch.setdefault("phases", []).append("final_material_payload_written")
        _write_material_scratch(scratch_path, scratch)
        return StageRunResult(
            name=stage_name,
            stage=Stage.MATERIAL_INFERENCE,
            success=True,
            artifacts=(
                _artifact(ArtifactRole.INFERRED_PARAMS, request.output_path, Stage.MATERIAL_INFERENCE),
            ),
            metrics={
                "scratch_path": str(scratch_path),
                "payload": write_metadata,
            },
        )
    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={
            "run_root": run_root,
            "has_diagnostic_repair_plan": bool(diagnostic_repair_plan.strip()),
        },
        invoke=invoke,
    )
def run_assign_params_to_prims_stage(run_root: str) -> dict[str, Any]:
    """Assign inferred material parameters to OmniPart primitives."""
    stage_name = "hag4r_assign_params_to_prims"
    tool_name = "run_assign_params_to_prims_stage"
    def invoke(state: dict[str, Any]) -> StageRunResult:
        return run_assign_params(
            part_labels_path=_path(state, "part_labels_path"),
            inferred_params_path=_path(state, "inferred_params_path"),
            output_tag=str(state["output_tag"]),
            output_path=_path(state, "partwise_params_path"),
            output_dir=_path(state, "partwise_params_path").parent,
            repo_root=_repo_root(state),
            log_dir=_log_dir(state),
        )
    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root},
        invoke=invoke,
    )


def select_mesh_processing_fidelities_stage(
    run_root: str,
    part_fidelities: list[dict[str, Any]],
    target_max_dimension_m: float,
    estimate_rationale: str,
    object_semantics: str = "",
    diagnostic_cue_summary: str = "",
) -> dict[str, Any]:
    """Select per-part mesh fidelities and target metric size."""
    tool_name = "select_mesh_processing_fidelities_stage"
    state = _require_state(run_root)
    cues = _diagnostic_cues(state)
    resolved_object_semantics = object_semantics.strip() or _object_description(state)
    resolved_cue_summary = diagnostic_cue_summary.strip() or _summarize_diagnostic_cues(cues)
    indexed_parts = load_color_locked_material_predictions(
        _path(state, "inferred_params_path"),
        _path(state, "part_labels_path"),
    )
    plan = build_mesh_processing_plan(
        part_fidelities=part_fidelities,
        indexed_parts=indexed_parts,
        target_max_dimension_m=target_max_dimension_m,
        estimate_rationale=estimate_rationale,
        object_semantics=resolved_object_semantics,
        diagnostic_cue_summary=resolved_cue_summary,
        diagnostic_cues=to_json_dict(cues),
        tool_name=tool_name,
    )
    state["mesh_processing_plan"] = plan
    state.setdefault("mesh_config", {})["mesh_processing_plan"] = plan
    save_state(state, run_root)
    append_history(
        run_root,
        tool_name,
        "selected per-part volumetric mesh fidelities",
        event="mesh_processing_fidelities_selected",
        detail=plan,
    )
    return {
        "tool_name": tool_name,
        "run_root": run_root,
        "state_path": str(state_path(run_root)),
        "mesh_processing_plan": plan,
    }


def _persist_mesh_processing_success_artifacts(
    state: dict[str, Any],
    *,
    output_mesh_path: Path,
    heterogeneous_params_path: Path,
    metric_mesh_scaling_path: Path,
    volume_topology_path: Path,
    request_path: Path,
) -> None:
    topology_payload = json.loads(volume_topology_path.read_text(encoding="utf-8"))
    mesh_processing_plan = state.get("mesh_processing_plan")
    if not isinstance(mesh_processing_plan, dict):
        raise RuntimeError("mesh_processing_plan is missing while persisting mesh artifacts")
    mesh_processing_plan.setdefault("artifacts", {})
    mesh_processing_plan["artifacts"].update(
        {
            "mesh_path": str(output_mesh_path),
            "heterogeneous_params_path": str(heterogeneous_params_path),
            "metric_mesh_scaling_path": str(metric_mesh_scaling_path),
            "volume_topology_path": str(volume_topology_path),
            "request_path": str(request_path),
        }
    )
    state["mesh_processing_plan"] = mesh_processing_plan
    state.setdefault("mesh_config", {})["mesh_processing_plan"] = mesh_processing_plan
    state["volume_topology"] = {
        "schema_version": topology_payload.get("schema_version"),
        "volume_topology": topology_payload.get("volume_topology"),
        "tet_count": topology_payload.get("tet_count"),
        "tet_budget_status": topology_payload.get("tet_budget_status"),
        "warnings": topology_payload.get("warnings", []),
    }


def run_combined_to_monolithic_stage(run_root: str) -> dict[str, Any]:
    """Generate the monolithic mesh from OmniPart outputs."""
    stage_name = "hag4r_combined_to_monolithic"
    tool_name = "run_combined_to_monolithic_stage"
    def invoke(state: dict[str, Any]) -> StageRunResult:
        mesh_processing_plan = state.get("mesh_processing_plan")
        if not isinstance(mesh_processing_plan, dict):
            raise RuntimeError(
                "select_mesh_processing_fidelities_stage must run before run_combined_to_monolithic_stage"
            )
        if mesh_processing_plan.get("schema_version") != MESH_PROCESSING_PLAN_SCHEMA_VERSION:
            raise RuntimeError(
                f"mesh_processing_plan.schema_version must be {MESH_PROCESSING_PLAN_SCHEMA_VERSION}"
            )
        output_mesh_path = _path(state, "monolithic_mesh_path")
        metric_mesh_scaling_path = _optional_path(state, "metric_mesh_scaling_path")
        if metric_mesh_scaling_path is None:
            metric_mesh_scaling_path = _path(state, "monolithic_params_dir") / "metric_mesh_scaling.json"
        volume_topology_path = _path(state, "volume_topology_path")
        request_path = _path(state, "mesh_processing_request_path")
        result = run_monolithic_mesh(
            input_mesh_dir=_path(state, "omnipart_output_dir"),
            output_mesh=output_mesh_path,
            part_labels_path=_path(state, "part_labels_path"),
            inferred_params_path=_path(state, "inferred_params_path"),
            partwise_params_path=_path(state, "partwise_params_path"),
            mesh_processing_plan=mesh_processing_plan,
            heterogeneous_params_path=_path(state, "monolithic_params_path"),
            metric_mesh_scaling_path=metric_mesh_scaling_path,
            volume_topology_path=volume_topology_path,
            request_path=request_path,
            repo_root=_repo_root(state),
            log_dir=_log_dir(state),
        )
        if result.success and volume_topology_path.exists():
            _persist_mesh_processing_success_artifacts(
                state,
                output_mesh_path=output_mesh_path,
                heterogeneous_params_path=_path(state, "monolithic_params_path"),
                metric_mesh_scaling_path=metric_mesh_scaling_path,
                volume_topology_path=volume_topology_path,
                request_path=request_path,
            )
            save_state(state, run_root)
        return result
    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root},
        invoke=invoke,
        allow_retryable_failure=True,
    )


def resolve_disconnected_tet_components_stage(run_root: str, rationale: str) -> dict[str, Any]:
    """Resolve a pending disconnected tet component candidate by keeping the max-volume component."""
    stage_name = "hag4r_combined_to_monolithic"
    tool_name = "resolve_disconnected_tet_components_stage"

    def invoke(state: dict[str, Any]) -> StageRunResult:
        stage = state.get("stages", {}).get(stage_name, {})
        if not isinstance(stage, dict) or stage.get("status") != "needs_component_resolution":
            raise RuntimeError(
                "resolve_disconnected_tet_components_stage requires "
                "hag4r_combined_to_monolithic status=needs_component_resolution"
            )
        request_path = _path(state, "mesh_processing_request_path")
        output_mesh_path = _path(state, "monolithic_mesh_path")
        heterogeneous_params_path = _path(state, "monolithic_params_path")
        metric_mesh_scaling_path = _optional_path(state, "metric_mesh_scaling_path")
        if metric_mesh_scaling_path is None:
            metric_mesh_scaling_path = _path(state, "monolithic_params_dir") / "metric_mesh_scaling.json"
        volume_topology_path = _path(state, "volume_topology_path")
        result = resolve_disconnected_tet_components_from_request_json(
            request_path,
            rationale=rationale,
            repo_root=_repo_root(state),
        )
        _persist_mesh_processing_success_artifacts(
            state,
            output_mesh_path=output_mesh_path,
            heterogeneous_params_path=heterogeneous_params_path,
            metric_mesh_scaling_path=metric_mesh_scaling_path,
            volume_topology_path=volume_topology_path,
            request_path=request_path,
        )
        state.setdefault("mesh_component_resolution", {})["last_result"] = {
            "status": "resolved",
            "rationale": rationale,
            "volume_topology_path": str(volume_topology_path),
        }
        save_state(state, run_root)
        return StageRunResult(
            name="hag4r_resolve_disconnected_tet_components",
            stage=Stage.MESH_PROCESSING,
            success=True,
            returncode=0,
            artifacts=(
                _artifact(ArtifactRole.MONOLITHIC_MESH, result.mesh_path, Stage.MESH_PROCESSING),
                _artifact(ArtifactRole.MONOLITHIC_PARAMS, result.heterogeneous_params_path, Stage.MESH_PROCESSING),
                _artifact(ArtifactRole.REPORT, result.metric_mesh_scaling_path, Stage.MESH_PROCESSING),
                _artifact(ArtifactRole.REPORT, result.volume_topology_path, Stage.MESH_PROCESSING),
            ),
            metrics={
                "tet_count": result.tet_count,
                "tet_budget_status": result.tet_budget_status,
                "warnings": list(result.warnings),
            },
        )

    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root, "rationale": rationale},
        invoke=invoke,
    )


def run_post_mesh_texture_stage(run_root: str) -> dict[str, Any]:
    """Build the exact textured visual bundle for the active revision."""
    stage_name = POST_MESH_TEXTURE_STAGE_SEQUENCE[0]
    tool_name = "run_post_mesh_texture_stage"

    def invoke(state: dict[str, Any]) -> StageRunResult:
        if state.get("diagnostics_stage_only") is True:
            raise RuntimeError("diagnostics-stage-only runs cannot execute post-mesh texture")
        validate_revision_path_ownership(state)
        revision_id = str(state.get("active_revision", "") or "")
        if not revision_id:
            raise RuntimeError("post-mesh texture requires an active revision")
        revision = state.get("revisions", {}).get(revision_id, {})
        decision = state.get("route_state", {}).get(
            "orchestrator_diagnostic_decision", {}
        )
        persisted_policy = (
            decision.get("diagnostic_export_policy")
            if isinstance(decision, dict)
            else None
        )
        if persisted_policy not in (None, ""):
            if persisted_policy != DIAGNOSTIC_EXPORT_POLICY:
                raise RuntimeError(
                    "post-mesh texture contains an unknown diagnostic export policy"
                )
            if not _is_final_revise_export_policy(state):
                raise RuntimeError(
                    "post-mesh texture contains an invalid final-revise export policy"
                )
        diagnostics_enabled = bool(state.get("diagnostics", {}).get("enabled"))
        if diagnostics_enabled:
            final_revise_policy = _is_final_revise_export_policy(state)
            if final_revise_policy:
                if revision.get("status") != "active":
                    raise RuntimeError(
                        "final-revise diagnostics texture generation requires "
                        "revision status=active"
                    )
            elif revision.get("status") != "diagnostics_accepted":
                raise RuntimeError(
                    "diagnostics-enabled texture generation requires "
                    "revision status=diagnostics_accepted or the validated final-revise policy"
                )
            diagnostic = state.get("stages", {}).get(
                "genesis_live_diagnostic_loop", {}
            )
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("ok") is not True
                or diagnostic.get("status") != "success"
            ):
                raise RuntimeError(
                    "diagnostics-enabled texture generation requires successful diagnostics"
                )
        else:
            if revision.get("status") != "active":
                raise RuntimeError(
                    "diagnostics-disabled texture generation requires revision status=active"
                )
            incomplete = [
                name
                for name in FULL_IMAGE_PRE_DIAGNOSTIC_STAGE_SEQUENCE
                if (
                    state.get("stages", {}).get(name, {}).get("ok") is not True
                    or state.get("stages", {}).get(name, {}).get("status")
                    != "success"
                )
            ]
            if incomplete:
                raise RuntimeError(
                    "post-mesh texture pre-diagnostic stages are incomplete: "
                    + ", ".join(incomplete)
                )

        revision_root = active_revision_root(state)
        material_source_raw = (
            state.get("inputs", {}).get("material_params_json")
            or state.get("paths", {}).get("material_params_source_path")
        )
        diagnostics_report_raw = state.get("paths", {}).get(
            "sim_diagnostics_report_path"
        )
        canonical_paths = build_full_image_revision_paths(
            repo_root=_repo_root(state),
            run_root=Path(str(state["run_root"])).expanduser().resolve(),
            revision_root_path=revision_root,
            output_tag=str(state["output_tag"]),
            object_name=str(state["object_name"]),
            source_image=_path(state, "source_image"),
            material_params_source_path=(
                Path(str(material_source_raw)).expanduser().resolve()
                if material_source_raw
                else None
            ),
            sim_diagnostics_report_path=(
                Path(str(diagnostics_report_raw)).expanduser().resolve()
                if diagnostics_report_raw
                else None
            ),
        )
        expected_output_dir = Path(
            str(canonical_paths["post_mesh_texture_dir"])
        ).resolve()
        expected_appearance_manifest = Path(
            str(canonical_paths["omnipart_appearance_manifest_path"])
        ).resolve()
        if _path(state, "post_mesh_texture_dir") != expected_output_dir:
            raise ValueError(
                "paths.post_mesh_texture_dir does not match the canonical "
                f"active-revision path: expected={expected_output_dir}"
            )
        if (
            _path(state, "omnipart_appearance_manifest_path")
            != expected_appearance_manifest
        ):
            raise ValueError(
                "paths.omnipart_appearance_manifest_path does not match the canonical "
                f"active-revision path: expected={expected_appearance_manifest}"
            )
        input_keys = (
            "omnipart_appearance_manifest_path",
            "monolithic_mesh_path",
            "monolithic_params_path",
            "metric_mesh_scaling_path",
            "inferred_params_path",
        )
        for key in input_keys:
            value = _path(state, key)
            try:
                value.relative_to(revision_root)
            except ValueError as exc:
                raise ValueError(
                    f"post-mesh texture input is not active-revision scoped: {key}={value}"
                ) from exc
        output_dir = _path(state, "post_mesh_texture_dir")
        try:
            output_dir.relative_to(revision_root)
        except ValueError as exc:
            raise ValueError(
                "post-mesh texture output is not active-revision scoped: "
                f"{output_dir}"
            ) from exc
        return run_post_mesh_texture(
            appearance_manifest_path=_path(
                state, "omnipart_appearance_manifest_path"
            ),
            monolithic_mesh_path=_path(state, "monolithic_mesh_path"),
            heterogeneous_params_path=_path(state, "monolithic_params_path"),
            metric_mesh_scaling_path=_path(state, "metric_mesh_scaling_path"),
            inferred_material_path=_path(state, "inferred_params_path"),
            output_dir=output_dir,
            env=_worker_gpu_env(state),
            repo_root=_repo_root(state),
            log_dir=_log_dir(state),
        )

    def after_success(_: dict[str, Any]) -> None:
        persisted = load_state(run_root)
        validate_post_mesh_texture_bundle(
            _path(persisted, "post_mesh_texture_dir"),
            validate_glb=True,
        )
        mark_active_revision_ready_for_export(
            run_root,
            reason="post-mesh texture stage completed with a validated visual bundle",
        )

    def after_failure(_: str) -> None:
        failed = load_state(run_root)
        failed["status"] = "failed"
        save_state(failed, run_root)

    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root},
        invoke=invoke,
        after_success=after_success,
        after_failure=after_failure,
    )


def run_final_export_bundle_stage(run_root: str) -> dict[str, Any]:
    """Write the final HAG4R export bundle."""
    stage_name = "hag4r_final_export_bundle"
    tool_name = "run_final_export_bundle_stage"

    def invoke(state: dict[str, Any]) -> StageRunResult:
        if state.get("active_revision"):
            export_revision_id, export_revision_root = resolve_export_revision(state)
            sim_diagnostics_dir = (
                export_revision_root / "sim_diagnostics"
                if state.get("diagnostics", {}).get("enabled")
                else None
            )
        else:
            export_revision_id = ""
            export_revision_root = Path(str(state["run_root"])).expanduser().resolve()
            sim_diagnostics_dir = (
                _path(state, "sim_diagnostics_dir")
                if state.get("diagnostics", {}).get("enabled")
                else None
            )
        result = run_final_export(
            run_id=str(state["output_tag"]),
            mesh_path=_path(state, "monolithic_mesh_path"),
            heterogeneous_params_path=_path(state, "monolithic_params_path"),
            inferred_params_path=_path(state, "inferred_params_path"),
            volume_topology_path=_path(state, "volume_topology_path"),
            output_dir=_path(state, "final_export_dir"),
            metric_mesh_scaling_path=_path(state, "metric_mesh_scaling_path"),
            appearance_bundle_dir=_path(state, "post_mesh_texture_dir"),
            youngs_modulus_render_path=_path(state, "monolithic_params_dir") / f"{state['output_tag']}_target_E.png",
            density_render_path=_path(state, "monolithic_params_dir") / f"{state['output_tag']}_target_density.png",
            target_part_render_path=_path(state, "monolithic_params_dir") / f"{state['output_tag']}_target_part.png",
            source_part_render_path=_path(state, "monolithic_params_dir") / f"{state['output_tag']}_source_part.png",
            cleaned_image_path=_optional_path(state, "cleaned_image_path"),
            cleanup_report_path=_optional_path(state, "cleanup_report_path"),
            agentic_run_dir=export_revision_root,
            sim_diagnostics_dir=sim_diagnostics_dir,
            repo_root=_repo_root(state),
        )
        return result

    def after_success(_: dict[str, Any]) -> None:
        persisted = load_state(run_root)
        if not persisted.get("active_revision"):
            return
        export_revision_id, _ = resolve_export_revision(persisted)
        manifest_path = _path(persisted, "final_export_manifest_path")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"final export manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"final export manifest must be an object: {manifest_path}")
        promote_exported_revision(
            run_root,
            export_revision_id,
            manifest_payload=manifest,
        )
        promoted = load_state(run_root)
        _rewrite_final_export_manifest_last(promoted)
        if _is_final_revise_export_policy(promoted):
            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            promoted["revisions"][export_revision_id]["manifest_payload"] = final_manifest
            promoted["status"] = "success"
            save_state(promoted, run_root)

    return _run_state_stage(
        run_root,
        stage_name=stage_name,
        tool_name=tool_name,
        inputs_summary={"run_root": run_root},
        invoke=invoke,
        after_success=after_success,
    )
STAGE_LOCAL_TOOLS = (
    register_image_cleanup_stage,
    run_sam3_omnipart_2d_segmentation_stage,
    run_omnipart_generate_parts_stage,
    load_material_part_labels_stage,
    build_material_indexed_parts_stage,
    run_material_geometry_analysis_stage,
    decide_material_fill_mode_stage,
    run_material_semantics_inference_stage,
    validate_repair_material_predictions_stage,
    write_inferred_material_params_stage,
    run_assign_params_to_prims_stage,
    select_mesh_processing_fidelities_stage,
    run_combined_to_monolithic_stage,
    resolve_disconnected_tet_components_stage,
    run_post_mesh_texture_stage,
    run_final_export_bundle_stage,
)
__all__ = [
    "STAGE_LOCAL_TOOL_NAMES",
    "STAGE_LOCAL_TOOLS",
    "build_material_indexed_parts_stage",
    "decide_material_fill_mode_stage",
    "load_material_part_labels_stage",
    "register_image_cleanup_stage",
    "run_material_geometry_analysis_stage",
    "run_material_semantics_inference_stage",
    "validate_repair_material_predictions_stage",
    "write_inferred_material_params_stage",
    "run_assign_params_to_prims_stage",
    "run_combined_to_monolithic_stage",
    "resolve_disconnected_tet_components_stage",
    "run_post_mesh_texture_stage",
    "run_final_export_bundle_stage",
    "run_sam3_omnipart_2d_segmentation_stage",
    "run_omnipart_generate_parts_stage",
    "select_mesh_processing_fidelities_stage",
]
