from __future__ import annotations
from typing import Any
from hag4r.agentic.runtime_state import _utc_now, append_history, load_state, save_state, state_path
STAGE_TERMINAL_TOOL_NAMES: tuple[str, ...] = (
    "submit_image_cleanup_stage",
    "halt_image_cleanup_stage",
    "submit_segmentation_stage",
    "halt_segmentation_stage",
    "submit_omnipart_stage",
    "halt_omnipart_stage",
    "submit_material_inference_stage",
    "halt_material_inference_stage",
    "submit_mesh_processing_stage",
    "halt_mesh_processing_stage",
    "submit_post_mesh_texture_stage",
    "halt_post_mesh_texture_stage",
    "submit_final_export_stage",
    "halt_final_export_stage",
)
TERMINAL_STAGE_BY_TOOL: dict[str, str] = {
    "submit_image_cleanup_stage": "image_cleanup",
    "halt_image_cleanup_stage": "image_cleanup",
    "submit_segmentation_stage": "segmentation",
    "halt_segmentation_stage": "segmentation",
    "submit_omnipart_stage": "omnipart",
    "halt_omnipart_stage": "omnipart",
    "submit_material_inference_stage": "material_inference",
    "halt_material_inference_stage": "material_inference",
    "submit_mesh_processing_stage": "mesh_processing",
    "halt_mesh_processing_stage": "mesh_processing",
    "submit_post_mesh_texture_stage": "post_mesh_texture",
    "halt_post_mesh_texture_stage": "post_mesh_texture",
    "submit_final_export_stage": "final_export",
    "halt_final_export_stage": "final_export",
}


def _assert_terminal_admission(state: dict[str, Any], *, stage_key: str) -> None:
    """Reject a second terminal for a logical stage before any mutation."""
    stage_runtime = state.get("stage_runtime", {})
    target_runtime = stage_runtime.get(stage_key, {})
    if "terminal" in target_runtime:
        raise RuntimeError(f"stage {stage_key} already terminalized")
    if stage_key != "final_export" and "terminal" in stage_runtime.get("final_export", {}):
        raise RuntimeError(
            f"cannot terminalize stage {stage_key} after final_export terminal"
        )


def _record_terminal(
    run_root: str,
    *,
    tool_name: str,
    status: str,
    summary: str = "",
    error: str = "",
    diagnostic_response_summary: str = "",
) -> dict[str, str]:
    state = load_state(run_root)
    if not state:
        raise FileNotFoundError(f"runtime state does not exist: {state_path(run_root)}")
    stage_key = TERMINAL_STAGE_BY_TOOL[tool_name]
    _assert_terminal_admission(state, stage_key=stage_key)
    timestamp = _utc_now()
    terminal = {
        "tool_name": tool_name,
        "stage_key": stage_key,
        "status": status,
        "summary": summary,
        "error": error,
        "timestamp": timestamp,
    }
    if diagnostic_response_summary.strip():
        terminal["diagnostic_response_summary"] = diagnostic_response_summary.strip()
    state.setdefault("stage_runtime", {}).setdefault(stage_key, {})["terminal"] = terminal
    if diagnostic_response_summary.strip():
        state["stage_runtime"][stage_key]["diagnostic_response_summary"] = diagnostic_response_summary.strip()
    state.setdefault("runtime_events", []).append({"event": "stage_terminal", **terminal})
    save_state(state, run_root)
    try:
        from hag4r.agentic.runtime_profiler import record_stage_terminal

        record_stage_terminal(
            run_root,
            stage=stage_key,
            operation=tool_name,
            status=status,
        )
    except Exception:
        # Profiling is observability-only and must never affect stage submission.
        pass
    if status == "success":
        event = "stage_submitted"
    elif status == "halted":
        event = "stage_halted"
    else:
        event = "stage_terminal_recorded"
    append_history(
        run_root,
        tool_name,
        summary or error or f"{stage_key} {status}",
        event=event,
        detail=terminal,
    )
    return {
        "stage_key": stage_key,
        "status": status,
        "summary": summary,
        "error": error,
        "diagnostic_response_summary": diagnostic_response_summary.strip(),
        "state_path": str(state_path(run_root)),
    }
def submit_image_cleanup_stage(run_root: str, summary: str = "") -> dict[str, str]:
    """Submit successful completion of the image cleanup stage."""
    return _record_terminal(
        run_root,
        tool_name="submit_image_cleanup_stage",
        status="pending_validation",
        summary=summary,
    )
def halt_image_cleanup_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the image cleanup stage."""
    return _record_terminal(run_root, tool_name="halt_image_cleanup_stage", status="halted", error=error)
def submit_segmentation_stage(run_root: str, summary: str = "", diagnostic_response_summary: str = "") -> dict[str, str]:
    """Submit successful completion of the segmentation stage."""
    return _record_terminal(
        run_root,
        tool_name="submit_segmentation_stage",
        status="pending_validation",
        summary=summary,
        diagnostic_response_summary=diagnostic_response_summary,
    )
def halt_segmentation_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the segmentation stage."""
    return _record_terminal(run_root, tool_name="halt_segmentation_stage", status="halted", error=error)
def submit_omnipart_stage(run_root: str, summary: str = "") -> dict[str, str]:
    """Submit successful completion of the OmniPart stage."""
    return _record_terminal(run_root, tool_name="submit_omnipart_stage", status="pending_validation", summary=summary)
def halt_omnipart_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the OmniPart stage."""
    return _record_terminal(run_root, tool_name="halt_omnipart_stage", status="halted", error=error)
def submit_material_inference_stage(run_root: str, summary: str = "", diagnostic_response_summary: str = "") -> dict[str, str]:
    """Submit successful completion of the material-inference stage."""
    return _record_terminal(
        run_root,
        tool_name="submit_material_inference_stage",
        status="pending_validation",
        summary=summary,
        diagnostic_response_summary=diagnostic_response_summary,
    )
def halt_material_inference_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the material-inference stage."""
    return _record_terminal(run_root, tool_name="halt_material_inference_stage", status="halted", error=error)
def submit_mesh_processing_stage(run_root: str, summary: str = "", diagnostic_response_summary: str = "") -> dict[str, str]:
    """Submit successful completion of the mesh-processing stage."""
    return _record_terminal(
        run_root,
        tool_name="submit_mesh_processing_stage",
        status="pending_validation",
        summary=summary,
        diagnostic_response_summary=diagnostic_response_summary,
    )
def halt_mesh_processing_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the mesh-processing stage."""
    return _record_terminal(run_root, tool_name="halt_mesh_processing_stage", status="halted", error=error)
def submit_post_mesh_texture_stage(run_root: str, summary: str = "") -> dict[str, str]:
    """Submit successful completion of the post-mesh-texture stage."""
    return _record_terminal(
        run_root,
        tool_name="submit_post_mesh_texture_stage",
        status="pending_validation",
        summary=summary,
    )
def halt_post_mesh_texture_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the post-mesh-texture stage."""
    return _record_terminal(
        run_root,
        tool_name="halt_post_mesh_texture_stage",
        status="halted",
        error=error,
    )
def submit_final_export_stage(run_root: str, summary: str = "") -> dict[str, str]:
    """Submit successful completion of the final-export stage."""
    return _record_terminal(run_root, tool_name="submit_final_export_stage", status="pending_validation", summary=summary)
def halt_final_export_stage(run_root: str, error: str = "") -> dict[str, str]:
    """Halt the final-export stage."""
    return _record_terminal(run_root, tool_name="halt_final_export_stage", status="halted", error=error)
STAGE_TERMINAL_TOOLS = (
    submit_image_cleanup_stage,
    halt_image_cleanup_stage,
    submit_segmentation_stage,
    halt_segmentation_stage,
    submit_omnipart_stage,
    halt_omnipart_stage,
    submit_material_inference_stage,
    halt_material_inference_stage,
    submit_mesh_processing_stage,
    halt_mesh_processing_stage,
    submit_post_mesh_texture_stage,
    halt_post_mesh_texture_stage,
    submit_final_export_stage,
    halt_final_export_stage,
)
__all__ = [
    "STAGE_TERMINAL_TOOL_NAMES",
    "STAGE_TERMINAL_TOOLS",
    "halt_final_export_stage",
    "halt_image_cleanup_stage",
    "halt_material_inference_stage",
    "halt_mesh_processing_stage",
    "halt_post_mesh_texture_stage",
    "halt_omnipart_stage",
    "halt_segmentation_stage",
    "submit_final_export_stage",
    "submit_image_cleanup_stage",
    "submit_material_inference_stage",
    "submit_mesh_processing_stage",
    "submit_post_mesh_texture_stage",
    "submit_omnipart_stage",
    "submit_segmentation_stage",
]
