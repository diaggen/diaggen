from __future__ import annotations

from importlib import import_module
from typing import Any


_STAGE_TOOL_EXPORTS = {
    "STAGE_LOCAL_TOOL_NAMES",
    "STAGE_LOCAL_TOOLS",
    "build_material_indexed_parts_stage",
    "decide_material_fill_mode_stage",
    "load_material_part_labels_stage",
    "run_assign_params_to_prims_stage",
    "run_combined_to_monolithic_stage",
    "resolve_disconnected_tet_components_stage",
    "run_post_mesh_texture_stage",
    "run_final_export_bundle_stage",
    "run_material_geometry_analysis_stage",
    "run_material_semantics_inference_stage",
    "run_omnipart_generate_parts_stage",
    "register_image_cleanup_stage",
    "run_sam3_omnipart_2d_segmentation_stage",
    "select_mesh_processing_fidelities_stage",
    "validate_repair_material_predictions_stage",
    "write_inferred_material_params_stage",
}


def __getattr__(name: str) -> Any:
    if name == "ALL_RUNTIME_TOOLS":
        stage_tools = import_module("hag4r.agentic.runtime_tools.stage_tools")
        transition_tools = import_module("hag4r.agentic.runtime_tools.pipeline_transition")
        return (*stage_tools.STAGE_LOCAL_TOOLS, *transition_tools.PIPELINE_TRANSITION_TOOLS)
    if name == "PIPELINE_TRANSITION_TOOLS":
        return import_module("hag4r.agentic.runtime_tools.pipeline_transition").PIPELINE_TRANSITION_TOOLS
    if name == "apply_diagnostic_recommendation":
        return import_module("hag4r.agentic.runtime_tools.pipeline_transition").apply_diagnostic_recommendation
    if name == "STAGE_TERMINAL_TOOLS":
        return import_module("hag4r.agentic.runtime_tools.stage_terminal").STAGE_TERMINAL_TOOLS
    if name == "DIAGNOSTIC_TOOLS":
        return import_module("hag4r.agentic.runtime_tools.diagnostics").DIAGNOSTIC_TOOLS
    if name in _STAGE_TOOL_EXPORTS:
        return getattr(import_module("hag4r.agentic.runtime_tools.stage_tools"), name)
    raise AttributeError(name)


__all__ = [
    "ALL_RUNTIME_TOOLS",
    "DIAGNOSTIC_TOOLS",
    "PIPELINE_TRANSITION_TOOLS",
    "STAGE_LOCAL_TOOL_NAMES",
    "STAGE_LOCAL_TOOLS",
    "STAGE_TERMINAL_TOOLS",
    "apply_diagnostic_recommendation",
    *_STAGE_TOOL_EXPORTS,
]
