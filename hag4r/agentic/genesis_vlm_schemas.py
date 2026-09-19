from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from hag4r.agentic.state import ObservationSignal, SimDiagnosticRoute
from hag4r.tools.probe_targeting import (
    ALLOWED_REGION_HINTS,
    ANCHOR_TARGET_EDIT_SCHEMA_VERSION,
    FACE_ADJUST_KEYS,
    OBSERVED_MISMATCH_VALUES,
    PROBE_TARGET_EDIT_SCHEMA_VERSION,
    PROBE_TARGET_INTENT_SCHEMA_VERSION,
    PROBE_TARGET_INTENT_V2_SCHEMA_VERSION,
    TRANSLATION_KEYS,
    ALLOWED_MECHANICS_PROBE_MODES,
    ALLOWED_MOTION_AXES,
)
from hag4r.tools.triple_view_evidence import TRIPLE_VIEW_PANEL_ORDER
from hag4r.tools.genesis.diagnostic_timing import DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
from hag4r.tools.genesis.live_protocol import PUBLIC_LIVE_TOOL_NAMES


VLM_PRE_EPISODE_SCHEMA_VERSION = "hag4r-genesis-vlm-pre-episode-v1"
VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION = "hag4r-genesis-diagnostic-session-plan-v2"
VLM_ACTION_SCHEMA_VERSION = "hag4r-genesis-vlm-action-v5"
VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION = "hag4r-genesis-vlm-recommendation-v3"
_VLM_FINAL_RECOMMENDATION_V2_SCHEMA_VERSION = "hag4r-genesis-vlm-recommendation-v2"
VLM_REFLECTION_SCHEMA_VERSION = "hag4r-genesis-vlm-reflection-v2"
VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION = "hag4r-source-semantic-material-audit-v1"
ANCHOR_BBOX_PROPOSAL_FRAME = "local_mesh"
ANCHOR_TARGET_INTENT_SCHEMA_VERSION = "hag4r-diagnostic-anchor-target-intent-v1"

MAX_PRE_EPISODE_PIN_BOXES = 4
MAX_DIAGNOSTIC_SESSION_ANCHORS = 4
DEFAULT_MAX_RESUME_STEPS = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
DEFAULT_MAX_PROBE_VERTICES = 64
DEFAULT_MAX_PROBE_DISTANCE_M = 0.25
DEFAULT_MAX_PROBE_SPEED_M_S = 1.0
DEFAULT_MAX_PROBE_DURATION_STEPS = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
DEFAULT_MAX_CONTACTS = 1024
DEFAULT_MAX_DEFORMATION_SAMPLE_VERTICES = 4096
DEFAULT_MAX_FRAME_EDGE_PX = 2048

COMPILED_PROBE_TARGET_TOOL_NAMES = frozenset()
MODEL_FACING_LIVE_TOOL_NAMES = frozenset(PUBLIC_LIVE_TOOL_NAMES)
ALLOWED_VLM_TOOLS = MODEL_FACING_LIVE_TOOL_NAMES | COMPILED_PROBE_TARGET_TOOL_NAMES
ALLOWED_REFLECTION_PHASES = frozenset(
    {
        "pre_episode",
        "post_suite",
        "post_observation",
        "post_probe",
        "runtime_failure",
        "pre_terminal",
    }
)
ALLOWED_REFLECTION_ARTIFACT_KINDS = frozenset(
    {
        "workspace_artifact",
        "tool_result",
        "visual_evidence",
        "video",
        "report",
        "runtime_file",
    }
)
ALLOWED_REFLECTION_UNCERTAINTY = frozenset({"low", "medium", "high"})
ALLOWED_SOURCE_SEMANTIC_MATERIAL_STRENGTH = frozenset({"weak", "moderate", "strong"})
ALLOWED_SOURCE_SEMANTIC_MATERIAL_CALIBRATION_CONFIDENCE = frozenset({"low", "medium", "high"})
ALLOWED_SOURCE_SEMANTIC_MATERIAL_TRIGGER_REF_KINDS = frozenset(
    {"workspace_artifact", "tool_result", "visual_evidence"}
)
ALLOWED_DIAGNOSTIC_ANCHOR_TYPES = frozenset(
    {
        "support_contact",
        "grip_root",
        "joint_hinge",
    }
)
ALLOWED_DIAGNOSTIC_ANCHOR_UNCERTAINTY = frozenset({"low", "medium", "high"})
ALLOWED_SETUP_SEMANTIC_STATUS = frozenset({"matched", "mismatched", "unresolved"})
ALLOWED_SETUP_DOF_STATUS = frozenset({"preserved", "not_preserved", "unresolved"})
ALLOWED_SETUP_CONCERN_DISPOSITION = frozenset({"resolved", "unresolved"})
ALLOWED_PROBE_SEMANTIC_STATUS = frozenset({"matched", "mismatched", "unresolved"})
ALLOWED_OVERLAP_CONCERN_DISPOSITION = frozenset({"resolved", "unresolved"})
ALLOWED_DIAGNOSTIC_ANCHOR_PROBE_RELATIONSHIPS = frozenset({"distinct", "overlap_exception"})
ALLOWED_ANCHOR_TARGET_REGION_HINTS = (
    "full_part",
    "support_contact",
    "stable_base",
    "grip_root",
    "hinge_side",
    "root",
    "tip",
    "edge_band",
)
ALLOWED_REFLECTION_NEXT_ACTIONS = frozenset(
    {
        "define_episode",
        "preview_diagnostic_episode_setup",
        "simulation_reset",
        "simulate",
        "submit_diagnostic_anchor_target_intent",
        "compile_diagnostic_anchor_target",
        "preview_diagnostic_anchor_target",
        "revise_diagnostic_anchor_target",
        "submit_diagnostic_probe_target_intent",
        "compile_diagnostic_probe_target",
        "preview_diagnostic_probe_target",
        "revise_diagnostic_probe_target",
        "simulate_or_revise_diagnostic_probe_target",
        "record_diagnostic_evidence",
        "close_genesis_live_session",
        "submit_diagnostic_recommendation",
        "halt_diagnostics",
        "none",
    }
)

ALLOWED_PROBE_ACTIONS = frozenset(
    {
        "box_ee_grasp_and_move",
        "probe_release",
    }
)
DISALLOWED_PROBE_ACTIONS = frozenset(
    {
        "uniform_force",
        "clear_probe",
        "vertex_pin",
        "vertex_drag",
        "vertex_controller_tighten",
        "box_ee_grasp_and_hold",
        "box_ee_pin_vertices",
        "box_ee_move",
        "box_ee_push_pull",
        "box_ee_stretch",
    }
)
BOX_EE_ACTIONS = frozenset(
    {
        "box_ee_grasp_and_move",
    }
)
BOX_EE_MOVEMENT_ACTIONS = frozenset(
    {
        "box_ee_grasp_and_move",
    }
)


class GenesisVlmSchemaError(ValueError):
    """Raised when a Genesis diagnostic VLM payload fails schema validation."""


@dataclass(frozen=True)
class GenesisVlmActionLimits:
    max_resume_steps: int = DEFAULT_MAX_RESUME_STEPS
    max_probe_vertices: int = DEFAULT_MAX_PROBE_VERTICES
    max_probe_distance_m: float = DEFAULT_MAX_PROBE_DISTANCE_M
    max_probe_speed_m_s: float = DEFAULT_MAX_PROBE_SPEED_M_S
    max_probe_duration_steps: int = DEFAULT_MAX_PROBE_DURATION_STEPS
    max_contacts: int = DEFAULT_MAX_CONTACTS
    max_deformation_sample_vertices: int = DEFAULT_MAX_DEFORMATION_SAMPLE_VERTICES
    max_frame_edge_px: int = DEFAULT_MAX_FRAME_EDGE_PX


def genesis_vlm_action_limits_from_config(config: Any) -> GenesisVlmActionLimits:
    max_action_steps = _diagnostic_step_cap_from_config(config)
    return GenesisVlmActionLimits(
        max_resume_steps=max_action_steps,
        max_probe_vertices=_positive_config_int(config, "sim_diagnostics_probe_max_vertices", DEFAULT_MAX_PROBE_VERTICES),
        max_probe_distance_m=_positive_config_float(
            config,
            "sim_diagnostics_probe_max_distance_m",
            DEFAULT_MAX_PROBE_DISTANCE_M,
        ),
        max_probe_speed_m_s=_positive_config_float(
            config,
            "sim_diagnostics_probe_max_speed_m_s",
            DEFAULT_MAX_PROBE_SPEED_M_S,
        ),
        max_probe_duration_steps=max_action_steps,
    )


def validate_vlm_pre_episode_plan(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    data = _coerce_payload_object(payload, "pre-episode plan")
    _require_exact_keys(data, {"schema_version", "episode_intent", "pinning"}, "pre-episode plan")
    _require_version(data, "schema_version", VLM_PRE_EPISODE_SCHEMA_VERSION)

    pinning = _require_object(data, "pinning", "pre-episode plan")
    _require_exact_keys(pinning, {"enabled", "boxes"}, "pinning")
    enabled = _require_bool(pinning, "enabled", "pinning")
    boxes_value = _require_list(pinning, "boxes", "pinning")
    if len(boxes_value) > MAX_PRE_EPISODE_PIN_BOXES:
        raise GenesisVlmSchemaError(f"pinning.boxes must contain at most {MAX_PRE_EPISODE_PIN_BOXES} boxes")
    if not enabled and boxes_value:
        raise GenesisVlmSchemaError("pinning.boxes must be empty when pinning.enabled is false")

    return {
        "schema_version": data["schema_version"],
        "episode_intent": _require_non_empty_str(data, "episode_intent", "pre-episode plan"),
        "pinning": {
            "enabled": enabled,
            "boxes": [_validate_pin_box(box, index) for index, box in enumerate(boxes_value)],
        },
    }


def validate_diagnostic_session_plan(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    data = _coerce_payload_object(payload, "diagnostic session plan")
    _require_exact_keys(
        data,
        {"schema_version", "session_intent", "candidate_region_count", "regions", "omitted_region_summary", "paired_comparisons"},
        "diagnostic session plan",
    )
    _require_version(data, "schema_version", VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION)
    candidate_count = _require_bounded_int(
        data, "candidate_region_count", "diagnostic session plan", minimum=2
    )
    regions_value = _require_list(data, "regions", "diagnostic session plan")
    if not 2 <= len(regions_value) <= MAX_DIAGNOSTIC_SESSION_ANCHORS:
        raise GenesisVlmSchemaError("diagnostic session plan.regions must contain 2 through 4 regions")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_ranks: set[int] = set()
    seen_concerns: set[str] = set()
    regions = []
    for index, region_value in enumerate(regions_value):
        region = _validate_diagnostic_session_region(region_value, index=index)
        if region["region_id"] in seen_ids:
            raise GenesisVlmSchemaError("diagnostic session plan region_id values must be unique")
        if region["name"] in seen_names:
            raise GenesisVlmSchemaError("diagnostic session plan region name values must be unique")
        if region["risk_rank"] in seen_ranks:
            raise GenesisVlmSchemaError("diagnostic session plan selected risk_rank values must be unique")
        for concern in region["setup_anchor"]["concerns"]:
            if concern["concern_id"] in seen_concerns:
                raise GenesisVlmSchemaError("diagnostic session plan concern_id values must be unique across regions")
            seen_concerns.add(concern["concern_id"])
        seen_ids.add(region["region_id"])
        seen_names.add(region["name"])
        seen_ranks.add(region["risk_rank"])
        regions.append(region)
    if [region["risk_rank"] for region in regions] != sorted(seen_ranks):
        raise GenesisVlmSchemaError("diagnostic session plan selected regions must be in ascending risk_rank order")
    omitted_value = _require_list(data, "omitted_region_summary", "diagnostic session plan")
    omitted: list[dict[str, Any]] = []
    omitted_ranks: set[int] = set()
    for index, item in enumerate(omitted_value):
        context = f"diagnostic session plan omitted_region_summary[{index}]"
        if not isinstance(item, Mapping):
            raise GenesisVlmSchemaError(f"{context} must be an object")
        item = dict(item)
        _require_exact_keys(item, {"name", "semantic_region", "risk_rank", "omission_reason"}, context)
        rank = _require_bounded_int(item, "risk_rank", context, minimum=1)
        if rank in seen_ranks or rank in omitted_ranks:
            raise GenesisVlmSchemaError("diagnostic session plan selected and omitted risk ranks must be unique")
        omitted_ranks.add(rank)
        omitted.append({
            "name": _require_non_empty_str(item, "name", context),
            "semantic_region": _require_non_empty_str(item, "semantic_region", context),
            "risk_rank": rank,
            "omission_reason": _require_non_empty_str(item, "omission_reason", context),
        })
    if candidate_count <= MAX_DIAGNOSTIC_SESSION_ANCHORS:
        if len(regions) != candidate_count or omitted:
            raise GenesisVlmSchemaError("candidate_region_count <= 4 requires every candidate selected and no omissions")
        expected = set(range(1, candidate_count + 1))
    else:
        if len(regions) != MAX_DIAGNOSTIC_SESSION_ANCHORS or len(omitted) != candidate_count - MAX_DIAGNOSTIC_SESSION_ANCHORS:
            raise GenesisVlmSchemaError("candidate_region_count > 4 requires exactly the top four selected and every remainder omitted")
        expected = set(range(1, candidate_count + 1))
        if [region["risk_rank"] for region in regions] != [1, 2, 3, 4]:
            raise GenesisVlmSchemaError("candidate_region_count > 4 requires selected ranks exactly 1,2,3,4")
        if [item["risk_rank"] for item in omitted] != list(range(5, candidate_count + 1)):
            raise GenesisVlmSchemaError("omitted_region_summary must be in ascending continuous rank order after the top four")
    if seen_ranks | omitted_ranks != expected:
        raise GenesisVlmSchemaError("diagnostic session plan selected plus omitted risk ranks must be continuous 1..candidate_region_count")
    pairs_value = _require_list(data, "paired_comparisons", "diagnostic session plan")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    paired_regions: set[str] = set()
    unordered_pairs: set[frozenset[str]] = set()
    for index, value in enumerate(pairs_value):
        context = f"diagnostic session plan paired_comparisons[{index}]"
        if not isinstance(value, Mapping):
            raise GenesisVlmSchemaError(f"{context} must be an object")
        item = dict(value)
        _require_exact_keys(item, {"pair_id", "region_ids"}, context)
        pair_id = _require_non_empty_str(item, "pair_id", context)
        region_ids = _require_list(item, "region_ids", context)
        if len(region_ids) != 2 or any(not isinstance(region_id, str) or not region_id.strip() for region_id in region_ids):
            raise GenesisVlmSchemaError(f"{context}.region_ids must contain exactly two non-empty region IDs")
        normalized_ids = [str(region_id).strip() for region_id in region_ids]
        if normalized_ids[0] == normalized_ids[1] or any(region_id not in seen_ids for region_id in normalized_ids):
            raise GenesisVlmSchemaError(f"{context}.region_ids must name two distinct selected regions")
        pair_key = frozenset(normalized_ids)
        if pair_id in pair_ids or pair_key in unordered_pairs:
            raise GenesisVlmSchemaError("diagnostic session plan paired_comparisons has duplicate pair identity or membership")
        if any(region_id in paired_regions for region_id in normalized_ids):
            raise GenesisVlmSchemaError("diagnostic session plan region may belong to at most one paired comparison")
        pair_ids.add(pair_id)
        unordered_pairs.add(pair_key)
        paired_regions.update(normalized_ids)
        pairs.append({"pair_id": pair_id, "region_ids": normalized_ids})

    return {
        "schema_version": data["schema_version"],
        "session_intent": _require_non_empty_str(data, "session_intent", "diagnostic session plan"),
        "candidate_region_count": candidate_count,
        "regions": regions,
        "omitted_region_summary": omitted,
        "paired_comparisons": pairs,
    }


def _validate_diagnostic_session_region(value: Any, *, index: int) -> dict[str, Any]:
    context = f"diagnostic session plan regions[{index}]"
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError(f"{context} must be an object")
    data = dict(value)
    _require_exact_keys(data, {
        "region_id", "name", "semantic_region", "physical_hypothesis", "desired_interaction",
        "episode_intent", "termination_condition", "risk_rank", "selection_rationale", "setup_anchor",
    }, context)
    setup = data.get("setup_anchor")
    if not isinstance(setup, Mapping):
        raise GenesisVlmSchemaError(f"{context}.setup_anchor must be an object")
    setup = dict(setup)
    _require_exact_keys(setup, {
        "anchor_type", "anchor_region", "uncertainty", "relationship_to_probe", "relationship_rationale", "concerns",
    }, f"{context}.setup_anchor")
    concerns_value = _require_list(setup, "concerns", f"{context}.setup_anchor")
    concerns: list[dict[str, str]] = []
    for concern_index, concern in enumerate(concerns_value):
        concern_context = f"{context}.setup_anchor.concerns[{concern_index}]"
        if not isinstance(concern, Mapping):
            raise GenesisVlmSchemaError(f"{concern_context} must be an object")
        concern = dict(concern)
        _require_exact_keys(concern, {"concern_id", "concern_type", "summary"}, concern_context)
        concerns.append({key: _require_non_empty_str(concern, key, concern_context) for key in ("concern_id", "concern_type", "summary")})
    return {
        "region_id": _require_non_empty_str(data, "region_id", context),
        "name": _require_non_empty_str(data, "name", context),
        "semantic_region": _require_non_empty_str(data, "semantic_region", context),
        "physical_hypothesis": _require_non_empty_str(data, "physical_hypothesis", context),
        "desired_interaction": _require_non_empty_str(data, "desired_interaction", context),
        "episode_intent": _require_non_empty_str(data, "episode_intent", context),
        "termination_condition": _require_non_empty_str(data, "termination_condition", context),
        "risk_rank": _require_bounded_int(data, "risk_rank", context, minimum=1),
        "selection_rationale": _require_non_empty_str(data, "selection_rationale", context),
        "setup_anchor": {
            "anchor_type": _require_allowed_string(setup, "anchor_type", f"{context}.setup_anchor", allowed=ALLOWED_DIAGNOSTIC_ANCHOR_TYPES),
            "anchor_region": _require_non_empty_str(setup, "anchor_region", f"{context}.setup_anchor"),
            "uncertainty": _require_allowed_string(setup, "uncertainty", f"{context}.setup_anchor", allowed=ALLOWED_DIAGNOSTIC_ANCHOR_UNCERTAINTY),
            "relationship_to_probe": _require_allowed_string(setup, "relationship_to_probe", f"{context}.setup_anchor", allowed=ALLOWED_DIAGNOSTIC_ANCHOR_PROBE_RELATIONSHIPS),
            "relationship_rationale": _require_non_empty_str(setup, "relationship_rationale", f"{context}.setup_anchor"),
            "concerns": concerns,
        },
    }


def _validate_source_semantic_material_ref(
    value: Any,
    *,
    context: str,
    allowed_kinds: frozenset[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError(f"{context} must be an object")
    data = dict(value)
    _require_exact_keys(data, {"kind", "ref", "note"}, context)
    kind = _require_non_empty_str(data, "kind", context)
    if kind not in allowed_kinds:
        raise GenesisVlmSchemaError(f"{context}.kind must be one of {sorted(allowed_kinds)}")
    return {
        "kind": kind,
        "ref": _require_non_empty_str(data, "ref", context),
        "note": _require_non_empty_str(data, "note", context),
    }


def _validate_source_semantic_material_group(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError(f"{context} must be an object")
    data = dict(value)
    _require_exact_keys(data, {"role", "part_indices"}, context)
    indices = _require_list(data, "part_indices", context)
    if not indices:
        raise GenesisVlmSchemaError(f"{context}.part_indices must not be empty")
    normalized_indices: list[int] = []
    for index, value in enumerate(indices):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GenesisVlmSchemaError(f"{context}.part_indices[{index}] must be a non-negative integer")
        if value in normalized_indices:
            raise GenesisVlmSchemaError(f"{context}.part_indices must be unique")
        normalized_indices.append(value)
    return {"role": _require_non_empty_str(data, "role", context), "part_indices": normalized_indices}


def validate_source_semantic_material_audit(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Validate the agent-authored semantic contract for a dynamic material audit.

    Deliberately excludes any measured modulus, ratio, hash, or runtime lineage:
    those facts are derived by the diagnostics runtime after it resolves current
    evidence and authoritative active-revision artifacts.
    """
    data = _coerce_payload_object(payload, "source-semantic material audit")
    context = "source-semantic material audit"
    _require_exact_keys(
        data,
        {"schema_version", "concern_summary", "trigger_reflection_index", "trigger_evidence_refs", "invariants"},
        context,
    )
    _require_version(data, "schema_version", VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION)
    trigger_reflection_index = _require_bounded_int(
        data, "trigger_reflection_index", context, minimum=0
    )
    trigger_refs_value = _require_list(data, "trigger_evidence_refs", context)
    if not trigger_refs_value:
        raise GenesisVlmSchemaError(f"{context}.trigger_evidence_refs must not be empty")
    trigger_refs = [
        _validate_source_semantic_material_ref(
            item,
            context=f"{context}.trigger_evidence_refs[{index}]",
            allowed_kinds=ALLOWED_SOURCE_SEMANTIC_MATERIAL_TRIGGER_REF_KINDS,
        )
        for index, item in enumerate(trigger_refs_value)
    ]
    invariant_values = _require_list(data, "invariants", context)
    if not invariant_values:
        raise GenesisVlmSchemaError(f"{context}.invariants must not be empty")
    invariants: list[dict[str, Any]] = []
    invariant_ids: set[str] = set()
    for index, value in enumerate(invariant_values):
        invariant_context = f"{context}.invariants[{index}]"
        if not isinstance(value, Mapping):
            raise GenesisVlmSchemaError(f"{invariant_context} must be an object")
        invariant = dict(value)
        _require_exact_keys(
            invariant,
            {
                "invariant_id", "stiffer_group", "softer_group", "source_refs", "semantic_basis",
                "semantic_strength", "calibration_confidence", "hard_min_ratio", "target_min_ratio", "target_max_ratio", "rationale",
            },
            invariant_context,
        )
        invariant_id = _require_non_empty_str(invariant, "invariant_id", invariant_context)
        if invariant_id in invariant_ids:
            raise GenesisVlmSchemaError(f"{context}.invariants invariant_id values must be unique")
        invariant_ids.add(invariant_id)
        stiffer_group = _validate_source_semantic_material_group(
            invariant["stiffer_group"], context=f"{invariant_context}.stiffer_group"
        )
        softer_group = _validate_source_semantic_material_group(
            invariant["softer_group"], context=f"{invariant_context}.softer_group"
        )
        if set(stiffer_group["part_indices"]) & set(softer_group["part_indices"]):
            raise GenesisVlmSchemaError(f"{invariant_context} stiffer_group and softer_group must not overlap")
        source_refs_value = _require_list(invariant, "source_refs", invariant_context)
        if not source_refs_value:
            raise GenesisVlmSchemaError(f"{invariant_context}.source_refs must not be empty")
        source_refs = [
            _validate_source_semantic_material_ref(
                item,
                context=f"{invariant_context}.source_refs[{source_index}]",
                allowed_kinds=frozenset({"workspace_artifact"}),
            )
            for source_index, item in enumerate(source_refs_value)
        ]
        semantic_strength = _require_allowed_string(
            invariant,
            "semantic_strength",
            invariant_context,
            allowed=ALLOWED_SOURCE_SEMANTIC_MATERIAL_STRENGTH,
        )
        calibration_confidence = _require_allowed_string(
            invariant,
            "calibration_confidence",
            invariant_context,
            allowed=ALLOWED_SOURCE_SEMANTIC_MATERIAL_CALIBRATION_CONFIDENCE,
        )
        hard_min_ratio = _require_finite_number(invariant, "hard_min_ratio", invariant_context)
        target_min_ratio = _require_finite_number(invariant, "target_min_ratio", invariant_context)
        target_max_ratio = _require_finite_number(invariant, "target_max_ratio", invariant_context)
        if not 1.0 < hard_min_ratio <= target_min_ratio <= target_max_ratio:
            raise GenesisVlmSchemaError(
                f"{invariant_context} thresholds must satisfy 1 < hard_min_ratio <= target_min_ratio <= target_max_ratio"
            )
        invariants.append(
            {
                "invariant_id": invariant_id,
                "stiffer_group": stiffer_group,
                "softer_group": softer_group,
                "source_refs": source_refs,
                "semantic_basis": _require_non_empty_str(invariant, "semantic_basis", invariant_context),
                "semantic_strength": semantic_strength,
                "calibration_confidence": calibration_confidence,
                "hard_min_ratio": hard_min_ratio,
                "target_min_ratio": target_min_ratio,
                "target_max_ratio": target_max_ratio,
                "rationale": _require_non_empty_str(invariant, "rationale", invariant_context),
            }
        )
    return {
        "schema_version": VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION,
        "concern_summary": _require_non_empty_str(data, "concern_summary", context),
        "trigger_reflection_index": trigger_reflection_index,
        "trigger_evidence_refs": trigger_refs,
        "invariants": invariants,
    }


def validate_diagnostic_anchor_region(payload: Mapping[str, Any] | str, *, index: int = 0) -> dict[str, Any]:
    data = _coerce_payload_object(payload, f"diagnostic session anchor[{index}]")
    context = f"diagnostic session anchors[{index}]"
    _require_allowed_keys(
        data,
        {
            "anchor_id",
            "name",
            "anchor_type",
            "semantic_region",
            "physical_hypothesis",
            "episode_intent",
            "termination_condition",
            "uncertainty",
            "concerns",
        },
        context,
    )
    return {
        "anchor_id": _require_non_empty_str(data, "anchor_id", context),
        "name": _require_non_empty_str(data, "name", context),
        "anchor_type": _require_allowed_string(
            data,
            "anchor_type",
            context,
            allowed=ALLOWED_DIAGNOSTIC_ANCHOR_TYPES,
        ),
        "semantic_region": _require_non_empty_str(data, "semantic_region", context),
        "physical_hypothesis": _require_non_empty_str(data, "physical_hypothesis", context),
        "episode_intent": _require_non_empty_str(data, "episode_intent", context),
        "termination_condition": _require_non_empty_str(data, "termination_condition", context),
        "uncertainty": _require_allowed_string(
            data,
            "uncertainty",
            context,
            allowed=ALLOWED_DIAGNOSTIC_ANCHOR_UNCERTAINTY,
        ),
        "concerns": _validate_diagnostic_anchor_concerns(data["concerns"], context),
    }


def validate_anchor_bbox_proposal(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Legacy/internal validator retained for old fixtures and migration tests."""
    data = _coerce_payload_object(payload, "anchor_bbox_proposal")
    _require_exact_keys(data, {"frame", "box", "reason"}, "anchor_bbox_proposal")
    frame = _require_allowed_string(
        data,
        "frame",
        "anchor_bbox_proposal",
        allowed=frozenset({ANCHOR_BBOX_PROPOSAL_FRAME}),
    )
    box = _require_vector(data, "box", "anchor_bbox_proposal", length=6)
    if box[0] >= box[3] or box[1] >= box[4] or box[2] >= box[5]:
        raise GenesisVlmSchemaError("anchor_bbox_proposal.box min values must be strictly less than max values")
    return {
        "frame": frame,
        "box": box,
        "reason": _require_non_empty_str(data, "reason", "anchor_bbox_proposal"),
    }


def validate_vlm_action_decision(
    payload: Mapping[str, Any] | str,
    *,
    limits: GenesisVlmActionLimits | None = None,
) -> dict[str, Any]:
    action_limits = limits or GenesisVlmActionLimits()
    data = _coerce_payload_object(payload, "action decision")
    _require_exact_keys(
        data,
        {"schema_version", "decision_id", "tool", "arguments", "observe_after", "expected_observation", "rationale"},
        "action decision",
    )
    _require_version(data, "schema_version", VLM_ACTION_SCHEMA_VERSION)

    decision_id = _require_bounded_int(data, "decision_id", "action decision", minimum=1)
    tool = _require_non_empty_str(data, "tool", "action decision")
    if tool not in ALLOWED_VLM_TOOLS:
        raise GenesisVlmSchemaError(f"tool must be one of {sorted(ALLOWED_VLM_TOOLS)}")

    arguments = _require_object(data, "arguments", "action decision")
    observe_after = _validate_observe_after(data["observe_after"])
    expected_observation_value = data["expected_observation"]
    if not isinstance(expected_observation_value, str):
        raise GenesisVlmSchemaError("action decision.expected_observation must be a string")
    if tool == "simulation_reset":
        normalized_arguments = _validate_simulation_reset_arguments(arguments)
        expected_observation = expected_observation_value.strip()
    elif tool == "inspect_genesis_runtime_logs":
        normalized_arguments = _validate_inspect_genesis_runtime_logs_arguments(arguments)
        expected_observation = expected_observation_value.strip()
    elif tool == "simulate":
        normalized_arguments = _validate_simulate_arguments(arguments, action_limits)
        expected_observation = expected_observation_value.strip()
    elif tool == "resume_simulation":
        normalized_arguments = _validate_resume_arguments(arguments, action_limits)
        expected_observation = expected_observation_value.strip()
    elif tool == "pause_and_observe":
        normalized_arguments = _validate_pause_and_observe_arguments(arguments)
        expected_observation = expected_observation_value.strip()
    elif tool == "query_live_geometry_context":
        normalized_arguments = _validate_query_live_geometry_context_arguments(arguments)
        expected_observation = expected_observation_value.strip()
    else:
        raise GenesisVlmSchemaError(f"tool has no VLM validator: {tool}")
    if not expected_observation:
        raise GenesisVlmSchemaError("action decision.expected_observation must be non-empty for public tools")

    return {
        "schema_version": data["schema_version"],
        "decision_id": decision_id,
        "tool": tool,
        "arguments": normalized_arguments,
        "observe_after": observe_after,
        "expected_observation": expected_observation,
        "rationale": _require_non_empty_str(data, "rationale", "action decision"),
    }


def _validate_diagnostic_cues(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise GenesisVlmSchemaError("final recommendation.diagnostic_cues must be a non-empty array")
    cues: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise GenesisVlmSchemaError(
                f"final recommendation.diagnostic_cues[{index}] must be a non-empty string"
            )
        cues.append(item.strip())
    return cues


def _validate_v2_final_recommendation(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the retired exact v2 shape for persisted-state migration only."""
    _require_exact_keys(
        data,
        {"schema_version", "recommendation", "route", "ready", "issue_signals", "reason", "part_indices", "stage_hints"},
        "persisted v2 final recommendation",
    )
    _require_version(data, "schema_version", _VLM_FINAL_RECOMMENDATION_V2_SCHEMA_VERSION)
    recommendation = _require_non_empty_str(data, "recommendation", "persisted v2 final recommendation")
    if recommendation not in {"accept", "revise"}:
        raise GenesisVlmSchemaError("recommendation must be accept or revise")
    route = _require_enum_value(data, "route", "persisted v2 final recommendation", SimDiagnosticRoute)
    ready = _require_bool(data, "ready", "persisted v2 final recommendation")
    issue_signals = _validate_issue_signals(data["issue_signals"])
    part_indices = _validate_unique_int_list(data["part_indices"], "part_indices", minimum=0)
    stage_hints = _validate_stage_hints(data["stage_hints"])
    reason = _require_non_empty_str(data, "reason", "persisted v2 final recommendation")
    if recommendation == "accept":
        if not ready or route != SimDiagnosticRoute.ACCEPT.value:
            raise GenesisVlmSchemaError("accept recommendation requires ready=true and route=accept")
        if issue_signals or part_indices or any(stage_hints.values()):
            raise GenesisVlmSchemaError("accept recommendation requires empty revision signals, stage_hints, and part_indices")
    else:
        if ready or route == SimDiagnosticRoute.ACCEPT.value:
            raise GenesisVlmSchemaError("revise recommendation requires ready=false and a repair route")
        if not stage_hints[route]:
            raise GenesisVlmSchemaError(f"revise recommendation requires stage_hints.{route}")
    return {
        "schema_version": data["schema_version"],
        "recommendation": recommendation,
        "route": route,
        "ready": ready,
        "issue_signals": issue_signals,
        "reason": reason,
        "part_indices": part_indices,
        "stage_hints": stage_hints,
    }


def validate_vlm_final_recommendation(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    data = _coerce_payload_object(payload, "final recommendation")
    _require_version(data, "schema_version", VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION)
    recommendation = _require_non_empty_str(data, "recommendation", "final recommendation")
    if recommendation not in {"accept", "revise"}:
        raise GenesisVlmSchemaError("recommendation must be accept or revise")
    if recommendation == "revise":
        _require_exact_keys(
            data,
            {"schema_version", "recommendation", "route", "diagnostic_cues"},
            "revision recommendation",
        )
        route = _require_enum_value(data, "route", "revision recommendation", SimDiagnosticRoute)
        if route == SimDiagnosticRoute.ACCEPT.value:
            raise GenesisVlmSchemaError("revise recommendation cannot use accept route")
        return {
            "schema_version": data["schema_version"],
            "recommendation": recommendation,
            "route": route,
            "diagnostic_cues": _validate_diagnostic_cues(data["diagnostic_cues"]),
        }
    _require_exact_keys(
        data,
        {"schema_version", "recommendation", "route", "ready", "issue_signals", "reason", "part_indices", "stage_hints"},
        "accept recommendation",
    )
    route = _require_enum_value(data, "route", "accept recommendation", SimDiagnosticRoute)
    ready = _require_bool(data, "ready", "accept recommendation")
    issue_signals = _validate_issue_signals(data["issue_signals"])
    part_indices = _validate_unique_int_list(data["part_indices"], "part_indices", minimum=0)
    stage_hints = _validate_stage_hints(data["stage_hints"])
    reason = _require_non_empty_str(data, "reason", "accept recommendation")
    if not ready or route != SimDiagnosticRoute.ACCEPT.value:
        raise GenesisVlmSchemaError("accept recommendation requires ready=true and route=accept")
    if issue_signals or part_indices or any(stage_hints.values()):
        raise GenesisVlmSchemaError("accept recommendation requires empty revision signals, stage_hints, and part_indices")
    return {
        "schema_version": data["schema_version"],
        "recommendation": recommendation,
        "route": route,
        "ready": ready,
        "issue_signals": issue_signals,
        "reason": reason,
        "part_indices": part_indices,
        "stage_hints": stage_hints,
    }


def normalize_persisted_vlm_final_recommendation(
    payload: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Normalize a persisted v2/v3 terminal recommendation to the public v3 shape."""
    data = _coerce_payload_object(payload, "persisted final recommendation")
    if data.get("schema_version") == VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION:
        return validate_vlm_final_recommendation(data)
    legacy = _validate_v2_final_recommendation(data)
    if legacy["recommendation"] == "revise":
        route = legacy["route"]
        return {
            "schema_version": VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION,
            "recommendation": "revise",
            "route": route,
            "diagnostic_cues": list(legacy["stage_hints"][route]),
        }
    return {
        **legacy,
        "schema_version": VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION,
    }


def validate_vlm_reflection_evidence(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    data = _coerce_payload_object(payload, "reflection evidence")
    # The nested judgment is required by the runtime only once a valid M2
    # endpoint pair exists; preserve validation of non-measured legacy phases.
    required_keys = {
            "schema_version",
            "phase",
            "episode_id",
            "artifact_refs",
            "observation",
            "reflection",
            "uncertainty",
            "knowledge_base_entry_ids",
            "route_relevance",
            "next_action",
        }
    _require_allowed_keys(data, {*required_keys, "material_response_judgment"}, "reflection evidence")
    _require_exact_keys(
        {key: value for key, value in data.items() if key != "material_response_judgment"},
        required_keys,
        "reflection evidence",
    )
    _require_version(data, "schema_version", VLM_REFLECTION_SCHEMA_VERSION)
    phase = _require_non_empty_str(data, "phase", "reflection evidence")
    if phase not in ALLOWED_REFLECTION_PHASES:
        raise GenesisVlmSchemaError(f"reflection evidence.phase must be one of {sorted(ALLOWED_REFLECTION_PHASES)}")
    episode_id = _require_non_empty_str(data, "episode_id", "reflection evidence")
    if not _is_valid_reflection_episode_id(episode_id):
        raise GenesisVlmSchemaError("reflection evidence.episode_id must be pre_episode or match episode_####")
    artifact_refs = _validate_reflection_artifact_refs(data["artifact_refs"])
    uncertainty = _require_non_empty_str(data, "uncertainty", "reflection evidence")
    if uncertainty not in ALLOWED_REFLECTION_UNCERTAINTY:
        raise GenesisVlmSchemaError(
            f"reflection evidence.uncertainty must be one of {sorted(ALLOWED_REFLECTION_UNCERTAINTY)}"
        )
    knowledge_base_entry_ids = _validate_knowledge_base_entry_ids(data["knowledge_base_entry_ids"])
    route_relevance = _validate_reflection_route_relevance(
        data["route_relevance"],
        allow_empty=phase == "runtime_failure",
    )
    next_action = _require_non_empty_str(data, "next_action", "reflection evidence")
    if next_action not in ALLOWED_REFLECTION_NEXT_ACTIONS:
        raise GenesisVlmSchemaError(
            f"reflection evidence.next_action must be one of {sorted(ALLOWED_REFLECTION_NEXT_ACTIONS)}"
        )
    if phase == "runtime_failure":
        if episode_id == "pre_episode":
            raise GenesisVlmSchemaError("runtime_failure reflection requires an episode_#### id")
        if len(artifact_refs) != 1 or artifact_refs[0]["kind"] != "tool_result":
            raise GenesisVlmSchemaError("runtime_failure reflection requires exactly one tool_result artifact ref")
        if not _is_tool_result_ref(artifact_refs[0]["ref"]):
            raise GenesisVlmSchemaError("runtime_failure reflection tool_result ref must match tool_result:<positive-int>")
        if knowledge_base_entry_ids:
            raise GenesisVlmSchemaError("runtime_failure reflection must not include knowledge_base_entry_ids")
        if route_relevance:
            raise GenesisVlmSchemaError("runtime_failure reflection must not include route_relevance")
        if next_action != "close_genesis_live_session":
            raise GenesisVlmSchemaError(
                "runtime_failure reflection next_action must equal close_genesis_live_session"
            )
        if data.get("material_response_judgment") is not None:
            raise GenesisVlmSchemaError("runtime_failure reflection must not include material_response_judgment")
        judgment = None
    else:
        raw_judgment = data.get("material_response_judgment")
        if phase == "post_probe" and raw_judgment is not None:
            if not isinstance(raw_judgment, Mapping):
                raise GenesisVlmSchemaError("post_probe material_response_judgment must be an object")
            judgment_data = dict(raw_judgment)
            _require_exact_keys(judgment_data, {"decision", "suggested_route"}, "material_response_judgment")
            decision = _require_non_empty_str(judgment_data, "decision", "material_response_judgment")
            suggested_route = _require_non_empty_str(judgment_data, "suggested_route", "material_response_judgment")
            allowed_routes = {SimDiagnosticRoute.ACCEPT.value, *(route.value for route in SimDiagnosticRoute)}
            if decision not in {"accept", "revise"} or suggested_route not in allowed_routes:
                raise GenesisVlmSchemaError("material_response_judgment must be binary accept|revise with a legal suggested_route")
            if (decision == "accept") != (suggested_route == SimDiagnosticRoute.ACCEPT.value):
                raise GenesisVlmSchemaError("material_response_judgment decision and suggested_route conflict")
            judgment = {"decision": decision, "suggested_route": suggested_route}
        elif raw_judgment is not None:
            raise GenesisVlmSchemaError("material_response_judgment is only allowed for post_probe reflections")
        else:
            judgment = None
    return {
        "schema_version": data["schema_version"],
        "phase": phase,
        "episode_id": episode_id,
        "artifact_refs": artifact_refs,
        "observation": _require_non_empty_str(data, "observation", "reflection evidence"),
        "reflection": _require_non_empty_str(data, "reflection", "reflection evidence"),
        "uncertainty": uncertainty,
        "knowledge_base_entry_ids": knowledge_base_entry_ids,
        "route_relevance": route_relevance,
        "next_action": next_action,
        "material_response_judgment": judgment,
    }


def _coerce_payload_object(payload: Mapping[str, Any] | str, schema_name: str) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GenesisVlmSchemaError(f"{schema_name} must be a JSON object") from exc
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError(f"{schema_name} must be a JSON object")
    return dict(payload)


def _is_valid_reflection_episode_id(value: str) -> bool:
    if value == "pre_episode":
        return True
    return len(value) == len("episode_0001") and value.startswith("episode_") and value.removeprefix("episode_").isdigit()


def _validate_reflection_artifact_refs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError("reflection evidence.artifact_refs must be a list")
    if not value:
        raise GenesisVlmSchemaError("reflection evidence.artifact_refs must not be empty")
    if len(value) > 8:
        raise GenesisVlmSchemaError("reflection evidence.artifact_refs must contain at most 8 entries")
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise GenesisVlmSchemaError(f"reflection evidence.artifact_refs[{index}] must be an object")
        artifact_ref = dict(item)
        _require_exact_keys(artifact_ref, {"kind", "ref", "note"}, f"reflection evidence.artifact_refs[{index}]")
        kind = _require_non_empty_str(artifact_ref, "kind", f"reflection evidence.artifact_refs[{index}]")
        if kind not in ALLOWED_REFLECTION_ARTIFACT_KINDS:
            raise GenesisVlmSchemaError(
                f"reflection evidence.artifact_refs[{index}].kind must be one of {sorted(ALLOWED_REFLECTION_ARTIFACT_KINDS)}"
            )
        normalized.append(
            {
                "kind": kind,
                "ref": _require_non_empty_str(artifact_ref, "ref", f"reflection evidence.artifact_refs[{index}]"),
                "note": _require_non_empty_str(artifact_ref, "note", f"reflection evidence.artifact_refs[{index}]"),
            }
        )
    return normalized


def _validate_knowledge_base_entry_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError("reflection evidence.knowledge_base_entry_ids must be a list")
    if len(value) > 8:
        raise GenesisVlmSchemaError("reflection evidence.knowledge_base_entry_ids must contain at most 8 entries")
    seen: set[str] = set()
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not _is_stable_knowledge_base_id(item):
            raise GenesisVlmSchemaError(
                f"reflection evidence.knowledge_base_entry_ids[{index}] must be a stable kebab id ending in -v<positive-int>"
            )
        if item in seen:
            raise GenesisVlmSchemaError("reflection evidence.knowledge_base_entry_ids must contain unique values")
        seen.add(item)
        normalized.append(item)
    return normalized


def _is_stable_knowledge_base_id(value: str) -> bool:
    head, separator, version = value.rpartition("-v")
    return bool(
        separator
        and head
        and version.isdigit()
        and int(version) > 0
        and all(part and part.isalnum() and part == part.lower() for part in head.split("-"))
    )


def _is_tool_result_ref(value: str) -> bool:
    prefix, separator, index = value.partition(":")
    return prefix == "tool_result" and separator == ":" and index.isdigit() and int(index) > 0


def _validate_reflection_route_relevance(value: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError("reflection evidence.route_relevance must be a list")
    if not value and not allow_empty:
        raise GenesisVlmSchemaError("reflection evidence.route_relevance must not be empty")
    allowed = {SimDiagnosticRoute.ACCEPT.value, *(route.value for route in SimDiagnosticRoute)}
    seen: set[str] = set()
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item not in allowed:
            raise GenesisVlmSchemaError(f"reflection evidence.route_relevance[{index}] must be one of {sorted(allowed)}")
        if item in seen:
            raise GenesisVlmSchemaError("reflection evidence.route_relevance must contain unique values")
        seen.add(item)
        normalized.append(item)
    return normalized


def _positive_config_int(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _diagnostic_step_cap_from_config(config: Any) -> int:
    value = _positive_config_int(config, "sim_diagnostics_probe_max_duration_steps", DEFAULT_MAX_PROBE_DURATION_STEPS)
    if value < DEFAULT_DIAGNOSTIC_SIMULATE_STEPS:
        raise GenesisVlmSchemaError(
            f"sim_diagnostics_probe_max_duration_steps must be >= {DEFAULT_DIAGNOSTIC_SIMULATE_STEPS}"
        )
    return value


def _positive_config_float(config: Any, name: str, default: float) -> float:
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        return default
    return float(value)


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    keys = set(data)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {', '.join(missing)}")
    if unknown:
        raise GenesisVlmSchemaError(f"{context} has unknown field(s): {', '.join(unknown)}")
    for key in expected:
        if data[key] is None:
            raise GenesisVlmSchemaError(f"{context}.{key} must not be null")


def _reject_keys(data: Mapping[str, Any], rejected: set[str], context: str) -> None:
    present = sorted(set(data) & rejected)
    if present:
        raise GenesisVlmSchemaError(f"{context} must not include VLM-owned field(s): {', '.join(present)}")


def _require_version(data: Mapping[str, Any], key: str, expected: str) -> None:
    if data[key] != expected:
        raise GenesisVlmSchemaError(f"{key} must equal {expected}")


def _require_object(data: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError(f"{context}.{key} must be an object")
    return dict(value)


def _require_list(data: Mapping[str, Any], key: str, context: str) -> list[Any]:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if not isinstance(value, list):
        raise GenesisVlmSchemaError(f"{context}.{key} must be a list")
    return value


def _require_bool(data: Mapping[str, Any], key: str, context: str) -> bool:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if not isinstance(value, bool):
        raise GenesisVlmSchemaError(f"{context}.{key} must be a boolean")
    return value


def _require_non_empty_str(data: Mapping[str, Any], key: str, context: str) -> str:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise GenesisVlmSchemaError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _require_allowed_string(data: Mapping[str, Any], key: str, context: str, *, allowed: frozenset[str]) -> str:
    value = _require_non_empty_str(data, key, context)
    if value not in allowed:
        raise GenesisVlmSchemaError(f"{context}.{key} must be one of {sorted(allowed)}")
    return value


def _validate_diagnostic_anchor_concerns(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError(f"{context}.concerns must be a list")
    concerns = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise GenesisVlmSchemaError(f"{context}.concerns[{index}] must be a string")
        text = item.strip()
        if not text:
            raise GenesisVlmSchemaError(f"{context}.concerns[{index}] must be a non-empty string")
        concerns.append(text)
    return concerns


def _payload_text_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        fragments: list[str] = []
        for key, item in value.items():
            fragments.extend(_payload_text_fragments(key))
            fragments.extend(_payload_text_fragments(item))
        return fragments
    if isinstance(value, list | tuple | set):
        fragments = []
        for item in value:
            fragments.extend(_payload_text_fragments(item))
        return fragments
    return [str(value)]


def _reject_raw_runtime_terms(
    value: Any,
    context: str,
    *,
    forbidden_terms: tuple[str, ...],
    label: str,
) -> None:
    normalized_terms = tuple(term.lower() for term in forbidden_terms)
    for fragment in _payload_text_fragments(value):
        lowered = fragment.lower()
        if any(term in lowered for term in normalized_terms):
            raise GenesisVlmSchemaError(f"{context} must not contain raw runtime {label} term: {fragment}")
        if "[" in fragment and "]" in fragment and any(ch.isdigit() for ch in fragment):
            raise GenesisVlmSchemaError(f"{context} must not contain raw coordinate-list payloads")


STRUCTURED_COMPLIANCE_CONTROL_KEYS = frozenset(
    {
        "stiffness",
        "controller_stiffness",
        "strength",
        "strength_rate",
        "constraint_strength",
        "soft_constraint",
        "is_soft_constraint",
    }
)
_CONTROLLER_COMPLIANCE_VERBS = (
    "set",
    "tune",
    "adjust",
    "increase",
    "decrease",
    "lower",
    "raise",
    "enable",
    "disable",
    "request",
    "specify",
    "override",
    "encode",
    "choose",
    "use",
    "make",
)
_CONTROLLER_COMPLIANCE_PHRASES = (
    "controller stiffness",
    "control stiffness",
    "controller strength",
    "control strength",
    "controller strength rate",
    "controller gain",
    "controller force",
    "force gain",
    "raw controller",
)
_CONTROLLER_COMPLIANCE_FIELD_TEXT = (
    "strength_rate",
    "constraint_strength",
    "soft_constraint",
    "is_soft_constraint",
)


def _reject_structured_compliance_control_keys(value: Any, context: str, *, path: str | None = None) -> None:
    path = context if path is None else path
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if normalized_key in STRUCTURED_COMPLIANCE_CONTROL_KEYS:
                raise GenesisVlmSchemaError(
                    f"{context} must not include controller compliance control field: {child_path}"
                )
            _reject_structured_compliance_control_keys(item, context, path=child_path)
        return
    if isinstance(value, list | tuple | set):
        for index, item in enumerate(value):
            _reject_structured_compliance_control_keys(item, context, path=f"{path}[{index}]")


def _reject_controller_compliance_tuning_text(value: Any, context: str) -> None:
    for fragment in _payload_text_fragments(value):
        lowered = fragment.lower()
        normalized = lowered.replace("_", " ")
        if any(token in lowered for token in _CONTROLLER_COMPLIANCE_FIELD_TEXT):
            raise GenesisVlmSchemaError(
                f"{context} must not request low-level controller compliance controls: {fragment}"
            )
        if any(phrase in normalized for phrase in _CONTROLLER_COMPLIANCE_PHRASES):
            raise GenesisVlmSchemaError(
                f"{context} must not request low-level controller compliance controls: {fragment}"
            )
        if "soft constraint" in normalized and any(verb in normalized for verb in _CONTROLLER_COMPLIANCE_VERBS):
            raise GenesisVlmSchemaError(
                f"{context} must not request low-level controller compliance controls: {fragment}"
            )
        for phrase in ("set force", "tune force", "adjust force", "set gain", "tune gain", "adjust gain"):
            if phrase in normalized:
                raise GenesisVlmSchemaError(
                    f"{context} must not request low-level controller compliance controls: {fragment}"
                )


def _reject_probe_target_raw_runtime_terms(value: Any, context: str) -> None:
    forbidden_terms = (
        "aabb_box",
        "vertices",
        "vertex_ids",
        "direction",
        "speed",
        "controller",
        "controllers",
        "scene_json",
        "model_deformable_json",
        "generated_asset_json",
        "target_position",
        "distance_scale",
        "box_ee",
    )
    _reject_raw_runtime_terms(value, context, forbidden_terms=forbidden_terms, label="controller")
    _reject_controller_compliance_tuning_text(value, context)


def _reject_anchor_target_raw_runtime_terms(value: Any, context: str) -> None:
    forbidden_terms = (
        "anchor_bbox_proposal",
        "aabb_box",
        "pin_box",
        "vertex_ids",
        "mesh_path",
        "scene_json",
        "model_deformable_json",
        "generated_asset_json",
        "genesis json",
        "genesis scene json",
        "controller",
        "controllers",
        "boxee",
        "box_ee",
        "target_position",
        "distance_scale",
    )
    _reject_raw_runtime_terms(value, context, forbidden_terms=forbidden_terms, label="anchor")
    _reject_controller_compliance_tuning_text(value, context)


def _part_grounding_has_part(part_grounding_context: Mapping[str, Any], selected_part_id: int) -> bool:
    parts_by_id = part_grounding_context.get("parts_by_id")
    if isinstance(parts_by_id, Mapping):
        return selected_part_id in parts_by_id or str(selected_part_id) in parts_by_id
    parts = part_grounding_context.get("parts")
    if isinstance(parts, list):
        return any(isinstance(part, Mapping) and part.get("part_id") == selected_part_id for part in parts)
    return False


def validate_diagnostic_anchor_target_intent(
    payload: Mapping[str, Any],
    *,
    part_grounding_context: Mapping[str, Any] | None = None,
    session_anchor_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError("diagnostic anchor target intent must be an object")
    data = dict(payload)
    context = "diagnostic_anchor_target_intent"
    _require_exact_keys(
        data,
        {
            "schema_version",
            "anchor_id",
            "anchor_intent",
            "selected_part_id",
            "region_hint",
            "candidate_part_reasoning",
            "physical_boundary_condition",
            "uncertainty",
            "concerns",
        },
        context,
    )
    _require_version(data, "schema_version", ANCHOR_TARGET_INTENT_SCHEMA_VERSION)
    normalized = {
        "schema_version": ANCHOR_TARGET_INTENT_SCHEMA_VERSION,
        "anchor_id": _require_non_empty_str(data, "anchor_id", context),
        "anchor_intent": _require_non_empty_str(data, "anchor_intent", context),
        "selected_part_id": _require_bounded_int(data, "selected_part_id", context, minimum=0),
        "region_hint": _require_allowed_string(
            data,
            "region_hint",
            context,
            allowed=frozenset(ALLOWED_ANCHOR_TARGET_REGION_HINTS),
        ),
        "candidate_part_reasoning": _require_non_empty_str(data, "candidate_part_reasoning", context),
        "physical_boundary_condition": _require_non_empty_str(data, "physical_boundary_condition", context),
        "uncertainty": _require_allowed_string(data, "uncertainty", context, allowed=ALLOWED_REFLECTION_UNCERTAINTY),
        "concerns": _validate_diagnostic_anchor_concerns(data["concerns"], context),
    }
    # Concerns are evidence prose, not an executable target/control surface.
    # Keep structural validation above, but allow an agent to describe an
    # observed runtime symptom honestly (for example, "vertices" or BoxEE).
    _reject_anchor_target_raw_runtime_terms(
        {key: value for key, value in normalized.items() if key != "concerns"}, context
    )
    if session_anchor_ids is not None and normalized["anchor_id"] not in session_anchor_ids:
        raise GenesisVlmSchemaError(f"{context}.anchor_id is not present in the active diagnostic session plan")
    if part_grounding_context is not None and not _part_grounding_has_part(
        part_grounding_context,
        normalized["selected_part_id"],
    ):
        raise GenesisVlmSchemaError(f"{context}.selected_part_id is not present in the runtime part_grounding table")
    return normalized


def diagnostic_anchor_target_intent_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "anchor_id",
            "anchor_intent",
            "selected_part_id",
            "region_hint",
            "candidate_part_reasoning",
            "physical_boundary_condition",
            "uncertainty",
            "concerns",
        ],
        "properties": {
            "schema_version": {"const": ANCHOR_TARGET_INTENT_SCHEMA_VERSION},
            "anchor_id": {"type": "string"},
            "anchor_intent": {"type": "string"},
            "selected_part_id": {"type": "integer", "minimum": 0},
            "region_hint": {"enum": list(ALLOWED_ANCHOR_TARGET_REGION_HINTS)},
            "candidate_part_reasoning": {"type": "string"},
            "physical_boundary_condition": {"type": "string"},
            "uncertainty": {"enum": sorted(ALLOWED_REFLECTION_UNCERTAINTY)},
            "concerns": {"type": "array", "items": {"type": "string"}},
        },
    }


def validate_diagnostic_probe_target_intent(
    payload: Mapping[str, Any],
    *,
    part_grounding_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError("diagnostic probe target intent must be an object")
    data = dict(payload)
    context = "diagnostic_probe_target_intent"
    version = data.get("schema_version")
    is_v2 = version == PROBE_TARGET_INTENT_V2_SCHEMA_VERSION
    if version not in {PROBE_TARGET_INTENT_SCHEMA_VERSION, PROBE_TARGET_INTENT_V2_SCHEMA_VERSION}:
        raise GenesisVlmSchemaError(
            f"{context}.schema_version must be {PROBE_TARGET_INTENT_SCHEMA_VERSION!r} or "
            f"{PROBE_TARGET_INTENT_V2_SCHEMA_VERSION!r}"
        )
    # The top-level intent envelope is deliberately forward-compatible.  The
    # canonical mapping below is explicit, so unknown fields are accepted and
    # then discarded rather than becoming executable runtime controls.
    selected_part_id = _require_bounded_int(data, "selected_part_id", context, minimum=0)
    group_value = data.get("semantic_group_part_ids", [selected_part_id])
    if not isinstance(group_value, list) or not group_value:
        raise GenesisVlmSchemaError(f"{context}.semantic_group_part_ids must be a non-empty list")
    semantic_group_part_ids: list[int] = []
    for index, part_id in enumerate(group_value):
        if isinstance(part_id, bool) or not isinstance(part_id, int) or part_id < 0:
            raise GenesisVlmSchemaError(f"{context}.semantic_group_part_ids[{index}] must be a non-negative integer")
        if part_id in semantic_group_part_ids:
            raise GenesisVlmSchemaError(f"{context}.semantic_group_part_ids must be unique")
        semantic_group_part_ids.append(part_id)
    if selected_part_id not in semantic_group_part_ids:
        raise GenesisVlmSchemaError(f"{context}.semantic_group_part_ids must include selected_part_id")
    grounding_value = data.get("semantic_group_part_grounding")
    semantic_group_part_grounding: list[dict[str, Any]] = []
    if len(semantic_group_part_ids) == 1:
        if grounding_value is not None:
            raise GenesisVlmSchemaError(
                f"{context}.semantic_group_part_grounding is only valid for a non-singleton semantic group"
            )
    else:
        if not isinstance(grounding_value, list) or len(grounding_value) != len(semantic_group_part_ids):
            raise GenesisVlmSchemaError(
                f"{context}.semantic_group_part_grounding must provide one entry for each semantic group part"
            )
        for index, entry in enumerate(grounding_value):
            entry_context = f"{context}.semantic_group_part_grounding[{index}]"
            if not isinstance(entry, Mapping):
                raise GenesisVlmSchemaError(f"{entry_context} must be an object")
            _require_exact_keys(
                entry,
                {"part_id", "part_name", "part_semantics", "member_to_planned_region_rationale"},
                entry_context,
            )
            part_id = _require_bounded_int(entry, "part_id", entry_context, minimum=0)
            if part_id != semantic_group_part_ids[index]:
                raise GenesisVlmSchemaError(
                    f"{entry_context}.part_id must match semantic_group_part_ids[{index}]"
                )
            semantic_group_part_grounding.append(
                {
                    "part_id": part_id,
                    "part_name": _require_non_empty_str(entry, "part_name", entry_context),
                    "part_semantics": _require_non_empty_str(entry, "part_semantics", entry_context),
                    "member_to_planned_region_rationale": _require_non_empty_str(
                        entry, "member_to_planned_region_rationale", entry_context
                    ),
                }
            )
    normalized = {
        "schema_version": PROBE_TARGET_INTENT_V2_SCHEMA_VERSION if is_v2 else PROBE_TARGET_INTENT_SCHEMA_VERSION,
        "target_intent": _require_non_empty_str(data, "target_intent", context),
        "physical_hypothesis": _require_non_empty_str(data, "physical_hypothesis", context),
        "desired_interaction": _require_non_empty_str(data, "desired_interaction", context),
        "candidate_part_reasoning": _require_non_empty_str(data, "candidate_part_reasoning", context),
        "selected_part_id": selected_part_id,
        "semantic_group_part_ids": semantic_group_part_ids,
        "semantic_group_part_grounding": semantic_group_part_grounding,
        "region_hint": _require_allowed_string(data, "region_hint", context, allowed=frozenset(ALLOWED_REGION_HINTS)),
        "uncertainty": _require_allowed_string(data, "uncertainty", context, allowed=ALLOWED_REFLECTION_UNCERTAINTY),
        "concerns": _validate_diagnostic_anchor_concerns(data["concerns"], context),
    }
    if is_v2:
        normalized.update(
            {
                "mechanics_probe_mode": _require_allowed_string(
                    data, "mechanics_probe_mode", context, allowed=frozenset(ALLOWED_MECHANICS_PROBE_MODES)
                ),
                "motion_axis": _require_allowed_string(
                    data, "motion_axis", context, allowed=frozenset(ALLOWED_MOTION_AXES)
                ),
                "motion_axis_rationale": _require_non_empty_str(data, "motion_axis_rationale", context),
            }
        )
    else:
        normalized.update(
            {
                "mechanics_probe_mode": "none",
                "motion_axis": "+Y",
                "motion_axis_rationale": "legacy v1 ordinary probe compatibility",
            }
        )
    _reject_probe_target_raw_runtime_terms(
        {
            key: value
            for key, value in normalized.items()
            if key not in {"concerns", "motion_axis", "motion_axis_rationale"}
        },
        context,
    )
    if part_grounding_context is not None:
        for part_id in normalized["semantic_group_part_ids"]:
            if not _part_grounding_has_part(part_grounding_context, part_id):
                raise GenesisVlmSchemaError(
                    f"{context}.semantic_group_part_ids contains a part absent from the runtime part_grounding table"
                )
    return normalized


def diagnostic_probe_target_intent_json_schema() -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": True,
        "required": [
            "schema_version",
            "target_intent",
            "physical_hypothesis",
            "desired_interaction",
            "candidate_part_reasoning",
            "selected_part_id",
            "region_hint",
            "uncertainty",
            "concerns",
        ],
        "properties": {
            "schema_version": {"enum": [PROBE_TARGET_INTENT_SCHEMA_VERSION, PROBE_TARGET_INTENT_V2_SCHEMA_VERSION]},
            "target_intent": {"type": "string"},
            "physical_hypothesis": {"type": "string"},
            "desired_interaction": {"type": "string"},
            "candidate_part_reasoning": {"type": "string"},
            "selected_part_id": {"type": "integer", "minimum": 0},
            "semantic_group_part_ids": {"type": "array", "minItems": 1, "items": {"type": "integer", "minimum": 0}},
            "semantic_group_part_grounding": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["part_id", "part_name", "part_semantics", "member_to_planned_region_rationale"],
                    "properties": {
                        "part_id": {"type": "integer", "minimum": 0},
                        "part_name": {"type": "string"},
                        "part_semantics": {"type": "string"},
                        "member_to_planned_region_rationale": {"type": "string"},
                    },
                },
            },
            "region_hint": {"enum": list(ALLOWED_REGION_HINTS)},
            "uncertainty": {"enum": sorted(ALLOWED_REFLECTION_UNCERTAINTY)},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "mechanics_probe_mode": {"enum": list(ALLOWED_MECHANICS_PROBE_MODES)},
            "motion_axis": {"enum": list(ALLOWED_MOTION_AXES)},
            "motion_axis_rationale": {"type": "string"},
        },
    }
    schema["allOf"] = [
        {
            "if": {"properties": {"schema_version": {"const": PROBE_TARGET_INTENT_V2_SCHEMA_VERSION}}},
            "then": {"required": ["mechanics_probe_mode", "motion_axis", "motion_axis_rationale"]},
        }
    ]
    return schema


def _validate_probe_target_edit_number(
    data: Mapping[str, Any],
    *,
    key: str,
    context: str,
    minimum: float,
    maximum: float,
) -> float:
    value = _require_finite_number(data, key, context)
    if value < minimum or value > maximum:
        raise GenesisVlmSchemaError(f"{context}.{key} must be in [{minimum:g}, {maximum:g}]")
    return value


def validate_diagnostic_probe_target_edit(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError("diagnostic probe target edit must be an object")
    data = dict(payload)
    context = "diagnostic_probe_target_edit"
    _require_exact_keys(
        data,
        {
            "schema_version",
            "face_adjust_percent",
            "translation_fraction",
            "evidence_refs",
            "observed_mismatch",
            "view_basis",
            "reason",
        },
        context,
    )
    _require_version(data, "schema_version", PROBE_TARGET_EDIT_SCHEMA_VERSION)
    evidence_refs = data["evidence_refs"]
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise GenesisVlmSchemaError(f"{context}.evidence_refs must be a non-empty list")
    normalized_evidence_refs = []
    for index, item in enumerate(evidence_refs):
        text = str(item).strip()
        if not text:
            raise GenesisVlmSchemaError(f"{context}.evidence_refs[{index}] must be non-empty")
        normalized_evidence_refs.append(text)
    observed_mismatch = _require_non_empty_str(data, "observed_mismatch", context)
    if observed_mismatch not in OBSERVED_MISMATCH_VALUES:
        raise GenesisVlmSchemaError(
            f"{context}.observed_mismatch must be one of: {', '.join(OBSERVED_MISMATCH_VALUES)}"
        )
    view_basis = data["view_basis"]
    if not isinstance(view_basis, list) or not view_basis:
        raise GenesisVlmSchemaError(f"{context}.view_basis must be a non-empty list")
    normalized_view_basis = []
    for index, item in enumerate(view_basis):
        text = str(item).strip()
        if text not in TRIPLE_VIEW_PANEL_ORDER:
            raise GenesisVlmSchemaError(
                f"{context}.view_basis[{index}] must be one of: {', '.join(TRIPLE_VIEW_PANEL_ORDER)}"
            )
        if text not in normalized_view_basis:
            normalized_view_basis.append(text)
    face = _require_object(data, "face_adjust_percent", context)
    translation = _require_object(data, "translation_fraction", context)
    _require_allowed_keys(face, set(FACE_ADJUST_KEYS), f"{context}.face_adjust_percent")
    _require_allowed_keys(translation, set(TRANSLATION_KEYS), f"{context}.translation_fraction")
    normalized_face = {
        key: _validate_probe_target_edit_number(
            face,
            key=key,
            context=f"{context}.face_adjust_percent",
            minimum=-50.0,
            maximum=50.0,
        )
        for key in face
    }
    normalized_translation = {
        key: _validate_probe_target_edit_number(
            translation,
            key=key,
            context=f"{context}.translation_fraction",
            minimum=-0.5,
            maximum=0.5,
        )
        for key in translation
    }
    reason = _require_non_empty_str(data, "reason", context)
    _reject_probe_target_raw_runtime_terms({"reason": reason}, context)
    return {
        "schema_version": PROBE_TARGET_EDIT_SCHEMA_VERSION,
        "face_adjust_percent": normalized_face,
        "translation_fraction": normalized_translation,
        "evidence_refs": normalized_evidence_refs,
        "observed_mismatch": observed_mismatch,
        "view_basis": normalized_view_basis,
        "reason": reason,
    }


def diagnostic_probe_target_edit_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "face_adjust_percent",
            "translation_fraction",
            "evidence_refs",
            "observed_mismatch",
            "view_basis",
            "reason",
        ],
        "properties": {
            "schema_version": {"const": PROBE_TARGET_EDIT_SCHEMA_VERSION},
            "face_adjust_percent": {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: {"type": "number", "minimum": -50, "maximum": 50} for key in FACE_ADJUST_KEYS},
            },
            "translation_fraction": {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: {"type": "number", "minimum": -0.5, "maximum": 0.5} for key in TRANSLATION_KEYS},
            },
            "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "observed_mismatch": {"enum": list(OBSERVED_MISMATCH_VALUES)},
            "view_basis": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": list(TRIPLE_VIEW_PANEL_ORDER)},
            },
            "reason": {"type": "string"},
        },
    }


def validate_diagnostic_anchor_target_edit(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GenesisVlmSchemaError("diagnostic anchor target edit must be an object")
    data = dict(payload)
    context = "diagnostic_anchor_target_edit"
    _require_exact_keys(
        data,
        {
            "schema_version",
            "face_adjust_percent",
            "translation_fraction",
            "evidence_refs",
            "observed_mismatch",
            "view_basis",
            "reason",
        },
        context,
    )
    _require_version(data, "schema_version", ANCHOR_TARGET_EDIT_SCHEMA_VERSION)
    evidence_refs = data["evidence_refs"]
    if not isinstance(evidence_refs, list) or not evidence_refs:
        raise GenesisVlmSchemaError(f"{context}.evidence_refs must be a non-empty list")
    normalized_evidence_refs = []
    for index, item in enumerate(evidence_refs):
        text = str(item).strip()
        if not text:
            raise GenesisVlmSchemaError(f"{context}.evidence_refs[{index}] must be non-empty")
        normalized_evidence_refs.append(text)
    observed_mismatch = _require_non_empty_str(data, "observed_mismatch", context)
    if observed_mismatch not in OBSERVED_MISMATCH_VALUES:
        raise GenesisVlmSchemaError(
            f"{context}.observed_mismatch must be one of: {', '.join(OBSERVED_MISMATCH_VALUES)}"
        )
    view_basis = data["view_basis"]
    if not isinstance(view_basis, list) or not view_basis:
        raise GenesisVlmSchemaError(f"{context}.view_basis must be a non-empty list")
    normalized_view_basis = []
    for index, item in enumerate(view_basis):
        text = str(item).strip()
        if text not in TRIPLE_VIEW_PANEL_ORDER:
            raise GenesisVlmSchemaError(
                f"{context}.view_basis[{index}] must be one of: {', '.join(TRIPLE_VIEW_PANEL_ORDER)}"
            )
        if text not in normalized_view_basis:
            normalized_view_basis.append(text)
    face = _require_object(data, "face_adjust_percent", context)
    translation = _require_object(data, "translation_fraction", context)
    _require_allowed_keys(face, set(FACE_ADJUST_KEYS), f"{context}.face_adjust_percent")
    _require_allowed_keys(translation, set(TRANSLATION_KEYS), f"{context}.translation_fraction")
    normalized_face = {
        key: _validate_probe_target_edit_number(
            face,
            key=key,
            context=f"{context}.face_adjust_percent",
            minimum=-50.0,
            maximum=50.0,
        )
        for key in face
    }
    normalized_translation = {
        key: _validate_probe_target_edit_number(
            translation,
            key=key,
            context=f"{context}.translation_fraction",
            minimum=-0.5,
            maximum=0.5,
        )
        for key in translation
    }
    reason = _require_non_empty_str(data, "reason", context)
    _reject_anchor_target_raw_runtime_terms({"reason": reason}, context)
    return {
        "schema_version": ANCHOR_TARGET_EDIT_SCHEMA_VERSION,
        "face_adjust_percent": normalized_face,
        "translation_fraction": normalized_translation,
        "evidence_refs": normalized_evidence_refs,
        "observed_mismatch": observed_mismatch,
        "view_basis": normalized_view_basis,
        "reason": reason,
    }


def diagnostic_anchor_target_edit_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "face_adjust_percent",
            "translation_fraction",
            "evidence_refs",
            "observed_mismatch",
            "view_basis",
            "reason",
        ],
        "properties": {
            "schema_version": {"const": ANCHOR_TARGET_EDIT_SCHEMA_VERSION},
            "face_adjust_percent": {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: {"type": "number", "minimum": -50, "maximum": 50} for key in FACE_ADJUST_KEYS},
            },
            "translation_fraction": {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: {"type": "number", "minimum": -0.5, "maximum": 0.5} for key in TRANSLATION_KEYS},
            },
            "evidence_refs": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            "observed_mismatch": {"enum": list(OBSERVED_MISMATCH_VALUES)},
            "view_basis": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": list(TRIPLE_VIEW_PANEL_ORDER)},
            },
            "reason": {"type": "string"},
        },
    }


def _require_finite_number(data: Mapping[str, Any], key: str, context: str) -> float:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GenesisVlmSchemaError(f"{context}.{key} must be a finite number")
    return float(value)


def _require_bounded_int(
    data: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenesisVlmSchemaError(f"{context}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise GenesisVlmSchemaError(f"{context}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise GenesisVlmSchemaError(f"{context}.{key} must be <= {maximum}")
    return value


def _optional_bool(data: Mapping[str, Any], key: str, context: str, normalized: dict[str, Any]) -> None:
    if key in data:
        normalized[key] = _require_bool(data, key, context)


def _optional_bounded_int(
    data: Mapping[str, Any],
    key: str,
    context: str,
    normalized: dict[str, Any],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if key in data:
        normalized[key] = _require_bounded_int(data, key, context, minimum=minimum, maximum=maximum)


def _optional_non_empty_str(data: Mapping[str, Any], key: str, context: str, normalized: dict[str, Any]) -> None:
    if key in data:
        normalized[key] = _require_non_empty_str(data, key, context)


def _optional_finite_number(
    data: Mapping[str, Any],
    key: str,
    context: str,
    normalized: dict[str, Any],
    *,
    minimum: float | None = None,
    maximum_exclusive: float | None = None,
) -> None:
    if key not in data:
        return
    value = _require_finite_number(data, key, context)
    if minimum is not None and value < minimum:
        raise GenesisVlmSchemaError(f"{context}.{key} must be >= {minimum:g}")
    if maximum_exclusive is not None and value >= maximum_exclusive:
        raise GenesisVlmSchemaError(f"{context}.{key} must be < {maximum_exclusive:g}")
    normalized[key] = value


def _require_enum_value(data: Mapping[str, Any], key: str, context: str, enum_type: type) -> str:
    value = _require_non_empty_str(data, key, context)
    allowed = {item.value for item in enum_type}
    if value not in allowed:
        raise GenesisVlmSchemaError(f"{context}.{key} must be one of {sorted(allowed)}")
    return value


def _validate_pin_box(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError(f"pinning.boxes[{index}] must be an object")
    data = dict(value)
    context = f"pinning.boxes[{index}]"
    _require_allowed_keys(data, {"anchor_id", "name", "obj_id", "box", "reason"}, context)
    obj_id = _require_bounded_int(data, "obj_id", context, minimum=0, maximum=0)
    box = _require_vector(data, "box", context, length=6)
    if box[0] > box[3] or box[1] > box[4] or box[2] > box[5]:
        raise GenesisVlmSchemaError(f"pinning.boxes[{index}].box min values must be <= max values")
    if box[0] == box[3] or box[1] == box[4] or box[2] == box[5]:
        raise GenesisVlmSchemaError(f"pinning.boxes[{index}].box volume must be positive")
    normalized = {
        "name": _require_non_empty_str(data, "name", context),
        "obj_id": obj_id,
        "box": box,
        "reason": _require_non_empty_str(data, "reason", context),
    }
    _optional_non_empty_str(data, "anchor_id", context, normalized)
    return normalized


def _require_vector(data: Mapping[str, Any], key: str, context: str, *, length: int = 3) -> list[float]:
    if key not in data or data[key] is None:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {key}")
    value = data[key]
    if not isinstance(value, list) or len(value) != length:
        raise GenesisVlmSchemaError(f"{context}.{key} must be a finite {length}-vector")
    vector = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise GenesisVlmSchemaError(f"{context}.{key}[{index}] must be a finite number")
        vector.append(float(item))
    return vector


def _is_zero_vector(value: list[float]) -> bool:
    return all(item == 0.0 for item in value)


def _validate_observe_after(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError("observe_after must be an object")
    data = dict(value)
    _require_exact_keys(data, {"contact", "deformation", "material", "frame"}, "observe_after")
    return {key: _require_bool(data, key, "observe_after") for key in ("contact", "deformation", "material", "frame")}


def _validate_pause_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms"}, "pause_simulation.arguments")
    _require_allowed_keys(data, {"reason", "wait"}, "pause_simulation.arguments")
    normalized: dict[str, Any] = {}
    _optional_non_empty_str(data, "reason", "pause_simulation.arguments", normalized)
    _optional_bool(data, "wait", "pause_simulation.arguments", normalized)
    return normalized


def _validate_pause_and_observe_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms"}, "pause_and_observe.arguments")
    _require_allowed_keys(data, set(), "pause_and_observe.arguments")
    return {}


def _validate_simulation_reset_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms", "action"}, "simulation_reset.arguments")
    _require_allowed_keys(data, set(), "simulation_reset.arguments")
    return {}


def _validate_inspect_genesis_runtime_logs_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    context = "inspect_genesis_runtime_logs.arguments"
    expected_keys = {"stream", "cursors", "max_lines", "contains", "context_lines"}
    missing = sorted(expected_keys - set(data))
    unknown = sorted(set(data) - expected_keys)
    if missing:
        raise GenesisVlmSchemaError(f"{context} missing required field(s): {', '.join(missing)}")
    if unknown:
        raise GenesisVlmSchemaError(f"{context} has unknown field(s): {', '.join(unknown)}")
    stream = _require_allowed_string(
        data,
        "stream",
        context,
        allowed=frozenset({"stdout", "stderr", "both"}),
    )
    selected_streams = {"stdout", "stderr"} if stream == "both" else {stream}
    raw_cursors = data["cursors"]
    if raw_cursors is None:
        cursor_values: dict[str, Any] = {}
    elif isinstance(raw_cursors, Mapping):
        cursor_values = dict(raw_cursors)
    else:
        raise GenesisVlmSchemaError(f"{context}.cursors must be null or an object")
    unknown_cursor_keys = set(cursor_values) - selected_streams
    if unknown_cursor_keys:
        raise GenesisVlmSchemaError(
            f"{context}.cursors contains unselected stream keys: {', '.join(sorted(unknown_cursor_keys))}"
        )
    cursors = {
        stream_name: _require_bounded_int(
            cursor_values,
            stream_name,
            f"{context}.cursors",
            minimum=1,
        )
        if stream_name in cursor_values
        else 1
        for stream_name in ("stdout", "stderr")
        if stream_name in selected_streams
    }
    contains = data["contains"]
    if not isinstance(contains, str):
        raise GenesisVlmSchemaError(f"{context}.contains must be a string")
    if len(contains) > 512:
        raise GenesisVlmSchemaError(f"{context}.contains must contain at most 512 characters")
    return {
        "stream": stream,
        "cursors": cursors,
        "max_lines": _require_bounded_int(data, "max_lines", context, minimum=1, maximum=500),
        "contains": contains,
        "context_lines": _require_bounded_int(data, "context_lines", context, minimum=0, maximum=5),
    }


def _validate_simulate_action(data: Any, limits: GenesisVlmActionLimits) -> dict[str, Any]:
    context = "simulate.arguments.action"
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise GenesisVlmSchemaError(f"{context} must be an object")
    raw = dict(data)
    _reject_structured_compliance_control_keys(raw, context)
    _reject_keys(
        raw,
        {
            "session_id",
            "timeout_ms",
            "action_id",
            "probe_id",
            "controller_id",
            "controller",
            "controllers",
            "aabb_box",
            "box",
            "target_position",
            "delta",
            "vertices",
            "force",
            "gain",
            "probe_apply",
            "probe_release",
        },
        context,
    )
    action_type = _require_non_empty_str(raw, "type", context)
    if action_type == "compiled_probe":
        _require_allowed_keys(raw, {"type", "compiled_probe_target_id", "env_id"}, context)
        normalized = {
            "type": "compiled_probe",
            "compiled_probe_target_id": _require_non_empty_str(raw, "compiled_probe_target_id", context),
            "env_id": 0,
        }
        if "env_id" in raw:
            normalized["env_id"] = _require_bounded_int(raw, "env_id", context, minimum=0)
        return normalized
    if action_type == "release_probe":
        _require_allowed_keys(raw, {"type", "compiled_probe_target_id"}, context)
        return {
            "type": "release_probe",
            "compiled_probe_target_id": _require_non_empty_str(raw, "compiled_probe_target_id", context),
        }
    raise GenesisVlmSchemaError(f"{context}.type must be one of ['compiled_probe', 'release_probe']")


def _validate_simulate_arguments(data: Mapping[str, Any], limits: GenesisVlmActionLimits) -> dict[str, Any]:
    context = "simulate.arguments"
    _reject_keys(
        data,
        {
            "session_id",
            "timeout_ms",
            "mode",
            "pause_after",
            "diagnostic_visual",
            "contact",
            "deformation",
            "material",
            "depth",
            "von_mises",
        },
        context,
    )
    unknown = sorted(set(data) - {"steps", "action"})
    if unknown:
        raise GenesisVlmSchemaError(f"{context} has unknown field(s): {', '.join(unknown)}")
    normalized = {
        "steps": _require_bounded_int(
            data,
            "steps",
            context,
            minimum=1,
            maximum=limits.max_resume_steps,
        ),
        "action": None,
    }
    if "action" in data and data["action"] is not None:
        normalized["action"] = _validate_simulate_action(data["action"], limits)
    return normalized


def _validate_query_live_geometry_context_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms"}, "query_live_geometry_context.arguments")
    _require_allowed_keys(data, {"env_id", "obj_id"}, "query_live_geometry_context.arguments")
    normalized = {"env_id": 0, "obj_id": 0}
    if "env_id" in data:
        normalized["env_id"] = _require_bounded_int(data, "env_id", "query_live_geometry_context.arguments", minimum=0)
    if "obj_id" in data:
        normalized["obj_id"] = _require_bounded_int(data, "obj_id", "query_live_geometry_context.arguments", minimum=0)
    return normalized


def _validate_resume_arguments(data: Mapping[str, Any], limits: GenesisVlmActionLimits) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms"}, "resume_simulation.arguments")
    _require_exact_keys(data, {"mode", "steps", "pause_after"}, "resume_simulation.arguments")
    mode = _require_non_empty_str(data, "mode", "resume_simulation.arguments")
    if mode != "bounded":
        raise GenesisVlmSchemaError("resume_simulation.arguments.mode must be bounded")
    pause_after = _require_bool(data, "pause_after", "resume_simulation.arguments")
    if pause_after is not True:
        raise GenesisVlmSchemaError("resume_simulation.arguments.pause_after must be true")
    return {
        "mode": mode,
        "steps": _require_bounded_int(
            data,
            "steps",
            "resume_simulation.arguments",
            minimum=1,
            maximum=limits.max_resume_steps,
        ),
        "pause_after": pause_after,
    }


def _validate_contact_arguments(data: Mapping[str, Any], limits: GenesisVlmActionLimits) -> dict[str, Any]:
    del limits
    _reject_keys(data, {"session_id", "timeout_ms"}, "get_contact_states.arguments")
    _require_allowed_keys(data, set(), "get_contact_states.arguments")
    return {}


def _validate_deformation_arguments(data: Mapping[str, Any], limits: GenesisVlmActionLimits) -> dict[str, Any]:
    del limits
    _reject_keys(data, {"session_id", "timeout_ms"}, "get_deformation_states.arguments")
    _require_allowed_keys(data, set(), "get_deformation_states.arguments")
    return {}


def _validate_material_arguments(data: Mapping[str, Any]) -> dict[str, Any]:
    _reject_keys(data, {"session_id", "timeout_ms"}, "set_material_params.arguments")
    _require_allowed_keys(data, {"scope", "young", "poisson", "friction_mu", "bending_weight"}, "set_material_params.arguments")
    material_keys = {"young", "poisson", "friction_mu", "bending_weight"}
    if not (set(data) & material_keys):
        raise GenesisVlmSchemaError("set_material_params.arguments must include at least one material value")
    if ("young" in data) != ("poisson" in data):
        raise GenesisVlmSchemaError("set_material_params.arguments young and poisson must be supplied together")

    normalized: dict[str, Any] = {}
    if "scope" in data:
        scope = data["scope"]
        if not isinstance(scope, Mapping):
            raise GenesisVlmSchemaError("set_material_params.arguments.scope must be an object")
        scope_dict = dict(scope)
        if scope_dict != {"type": "global"}:
            raise GenesisVlmSchemaError("set_material_params.arguments.scope must be exactly {'type': 'global'}")
        normalized["scope"] = {"type": "global"}
    _optional_finite_number(data, "young", "set_material_params.arguments", normalized, minimum=0.0)
    if "young" in normalized and normalized["young"] <= 0.0:
        raise GenesisVlmSchemaError("set_material_params.arguments.young must be > 0")
    _optional_finite_number(
        data,
        "poisson",
        "set_material_params.arguments",
        normalized,
        minimum=0.0,
        maximum_exclusive=0.5,
    )
    _optional_finite_number(data, "friction_mu", "set_material_params.arguments", normalized, minimum=0.0)
    _optional_finite_number(data, "bending_weight", "set_material_params.arguments", normalized, minimum=0.0)
    return normalized


def _require_allowed_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise GenesisVlmSchemaError(f"{context} has unknown field(s): {', '.join(unknown)}")
    for key, value in data.items():
        if value is None:
            raise GenesisVlmSchemaError(f"{context}.{key} must not be null")


def _validate_unique_int_list(value: Any, context: str, *, minimum: int) -> list[int]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError(f"{context} must be a list")
    seen: set[int] = set()
    normalized = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise GenesisVlmSchemaError(f"{context}[{index}] must be an integer")
        if item < minimum:
            raise GenesisVlmSchemaError(f"{context}[{index}] must be >= {minimum}")
        if item in seen:
            raise GenesisVlmSchemaError(f"{context} must contain unique integers")
        seen.add(item)
        normalized.append(item)
    return normalized


def _validate_issue_signals(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError("issue_signals must be a list")
    allowed = {item.value for item in ObservationSignal}
    seen: set[str] = set()
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item not in allowed:
            raise GenesisVlmSchemaError(f"issue_signals[{index}] must be one of {sorted(allowed)}")
        if item in seen:
            raise GenesisVlmSchemaError("issue_signals must contain unique values")
        seen.add(item)
        normalized.append(item)
    return normalized


def _validate_stage_hints(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise GenesisVlmSchemaError("stage_hints must be an object")
    expected = {
        SimDiagnosticRoute.SEGMENTATION.value,
        SimDiagnosticRoute.MATERIAL_INFERENCE.value,
        SimDiagnosticRoute.MESH_PROCESSING.value,
    }
    _require_exact_keys(value, expected, "stage_hints")
    return {
        key: _validate_non_empty_str_list(value[key], f"stage_hints.{key}")
        for key in sorted(expected)
    }


def _validate_non_empty_str_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise GenesisVlmSchemaError(f"{context} must be a list")
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise GenesisVlmSchemaError(f"{context}[{index}] must be a non-empty string")
        normalized.append(item.strip())
    return normalized


def vlm_pre_episode_plan_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "episode_intent", "pinning"],
        "properties": {
            "schema_version": {"type": "string", "const": VLM_PRE_EPISODE_SCHEMA_VERSION},
            "episode_intent": {"type": "string"},
            "pinning": {
                "type": "object",
                "additionalProperties": False,
                "required": ["enabled", "boxes"],
                "properties": {
                    "enabled": {"type": "boolean"},
                    "boxes": {
                        "type": "array",
                        "maxItems": MAX_PRE_EPISODE_PIN_BOXES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "obj_id", "box", "reason"],
                            "properties": {
                                "anchor_id": {"type": "string"},
                                "name": {"type": "string"},
                                "obj_id": {"type": "integer"},
                                "box": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 6,
                                    "maxItems": 6,
                                },
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }


def diagnostic_session_plan_json_schema() -> dict[str, Any]:
    concern = {
        "type": "object", "additionalProperties": False,
        "required": ["concern_id", "concern_type", "summary"],
        "properties": {"concern_id": {"type": "string"}, "concern_type": {"type": "string"}, "summary": {"type": "string"}},
    }
    setup_anchor = {
        "type": "object", "additionalProperties": False,
        "required": ["anchor_type", "anchor_region", "uncertainty", "relationship_to_probe", "relationship_rationale", "concerns"],
        "properties": {
            "anchor_type": {"type": "string", "enum": sorted(ALLOWED_DIAGNOSTIC_ANCHOR_TYPES)},
            "anchor_region": {"type": "string"},
            "uncertainty": {"type": "string", "enum": sorted(ALLOWED_DIAGNOSTIC_ANCHOR_UNCERTAINTY)},
            "relationship_to_probe": {"type": "string", "enum": sorted(ALLOWED_DIAGNOSTIC_ANCHOR_PROBE_RELATIONSHIPS)}, "relationship_rationale": {"type": "string"},
            "concerns": {"type": "array", "items": concern},
        },
    }
    region = {
        "type": "object", "additionalProperties": False,
        "required": ["region_id", "name", "semantic_region", "physical_hypothesis", "desired_interaction", "episode_intent", "termination_condition", "risk_rank", "selection_rationale", "setup_anchor"],
        "properties": {
            "region_id": {"type": "string"}, "name": {"type": "string"}, "semantic_region": {"type": "string"},
            "physical_hypothesis": {"type": "string"}, "desired_interaction": {"type": "string"}, "episode_intent": {"type": "string"},
            "termination_condition": {"type": "string"}, "risk_rank": {"type": "integer", "minimum": 1},
            "selection_rationale": {"type": "string"}, "setup_anchor": setup_anchor,
        },
    }
    pair = {
        "type": "object", "additionalProperties": False,
        "required": ["pair_id", "region_ids"],
        "properties": {
            "pair_id": {"type": "string", "minLength": 1},
            "region_ids": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string", "minLength": 1}},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "session_intent", "candidate_region_count", "regions", "omitted_region_summary", "paired_comparisons"],
        "properties": {
            "schema_version": {"type": "string", "const": VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION},
            "session_intent": {"type": "string"},
            "candidate_region_count": {"type": "integer", "minimum": 2},
            "regions": {"type": "array", "minItems": 2, "maxItems": MAX_DIAGNOSTIC_SESSION_ANCHORS, "items": region},
            "omitted_region_summary": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "semantic_region", "risk_rank", "omission_reason"], "properties": {"name": {"type": "string"}, "semantic_region": {"type": "string"}, "risk_rank": {"type": "integer", "minimum": 1}, "omission_reason": {"type": "string"}}}},
            "paired_comparisons": {"type": "array", "items": pair},
        },
    }


def source_semantic_material_audit_json_schema() -> dict[str, Any]:
    ref_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "ref", "note"],
        "properties": {
            "kind": {"type": "string"},
            "ref": {"type": "string", "minLength": 1},
            "note": {"type": "string", "minLength": 1},
        },
    }
    group_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["role", "part_indices"],
        "properties": {
            "role": {"type": "string", "minLength": 1},
            "part_indices": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer", "minimum": 0}},
        },
    }
    invariant_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "invariant_id", "stiffer_group", "softer_group", "source_refs", "semantic_basis",
            "semantic_strength", "calibration_confidence", "hard_min_ratio", "target_min_ratio", "target_max_ratio", "rationale",
        ],
        "properties": {
            "invariant_id": {"type": "string", "minLength": 1},
            "stiffer_group": group_schema,
            "softer_group": group_schema,
            "source_refs": {
                "type": "array", "minItems": 1,
                "items": {**ref_schema, "properties": {**ref_schema["properties"], "kind": {"const": "workspace_artifact"}}},
            },
            "semantic_basis": {"type": "string", "minLength": 1},
            "semantic_strength": {"enum": sorted(ALLOWED_SOURCE_SEMANTIC_MATERIAL_STRENGTH)},
            "calibration_confidence": {"enum": sorted(ALLOWED_SOURCE_SEMANTIC_MATERIAL_CALIBRATION_CONFIDENCE)},
            "hard_min_ratio": {"type": "number", "exclusiveMinimum": 1},
            "target_min_ratio": {"type": "number", "exclusiveMinimum": 1},
            "target_max_ratio": {"type": "number", "exclusiveMinimum": 1},
            "rationale": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "concern_summary", "trigger_reflection_index", "trigger_evidence_refs", "invariants"],
        "properties": {
            "schema_version": {"const": VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION},
            "concern_summary": {"type": "string", "minLength": 1},
            "trigger_reflection_index": {"type": "integer", "minimum": 0},
            "trigger_evidence_refs": {
                "type": "array", "minItems": 1,
                "items": {**ref_schema, "properties": {**ref_schema["properties"], "kind": {"enum": sorted(ALLOWED_SOURCE_SEMANTIC_MATERIAL_TRIGGER_REF_KINDS)}}},
            },
            "invariants": {"type": "array", "minItems": 1, "items": invariant_schema},
        },
    }


def anchor_bbox_proposal_json_schema() -> dict[str, Any]:
    """Legacy/internal schema retained for old fixtures and migration tests."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["frame", "box", "reason"],
        "properties": {
            "frame": {"type": "string", "const": ANCHOR_BBOX_PROPOSAL_FRAME},
            "box": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 6,
                "maxItems": 6,
            },
            "reason": {"type": "string"},
        },
    }


def vlm_action_decision_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "decision_id",
            "tool",
            "arguments",
            "observe_after",
            "expected_observation",
            "rationale",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": VLM_ACTION_SCHEMA_VERSION},
            "decision_id": {"type": "integer"},
            "tool": {"type": "string", "enum": sorted(ALLOWED_VLM_TOOLS)},
            "arguments": {
                "type": "object",
                "additionalProperties": True,
            },
            "observe_after": {
                "type": "object",
                "additionalProperties": False,
                "required": ["contact", "deformation", "material", "frame"],
                "properties": {
                    "contact": {"type": "boolean"},
                    "deformation": {"type": "boolean"},
                    "material": {"type": "boolean"},
                    "frame": {"type": "boolean"},
                },
            },
            "expected_observation": {"type": "string"},
            "rationale": {"type": "string"},
        },
    }


def vlm_final_recommendation_json_schema() -> dict[str, Any]:
    signal_names = [signal.value for signal in ObservationSignal]
    stage_hint_routes = [
        SimDiagnosticRoute.SEGMENTATION.value,
        SimDiagnosticRoute.MATERIAL_INFERENCE.value,
        SimDiagnosticRoute.MESH_PROCESSING.value,
    ]
    common_version = {"type": "string", "const": VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION}
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["schema_version", "recommendation", "route", "diagnostic_cues"],
                "properties": {
                    "schema_version": common_version,
                    "recommendation": {"const": "revise"},
                    "route": {"type": "string", "enum": stage_hint_routes},
                    "diagnostic_cues": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    },
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema_version", "recommendation", "route", "ready",
                    "issue_signals", "reason", "part_indices", "stage_hints",
                ],
                "properties": {
                    "schema_version": common_version,
                    "recommendation": {"const": "accept"},
                    "route": {"const": SimDiagnosticRoute.ACCEPT.value},
                    "ready": {"const": True},
                    "issue_signals": {"type": "array", "maxItems": 0, "items": {"type": "string", "enum": signal_names}},
                    "reason": {"type": "string", "minLength": 1, "pattern": r".*\S.*"},
                    "part_indices": {"type": "array", "maxItems": 0, "items": {"type": "integer"}},
                    "stage_hints": {
                        "type": "object", "additionalProperties": False,
                        "required": stage_hint_routes,
                        "properties": {
                            route: {"type": "array", "maxItems": 0, "items": {"type": "string"}}
                            for route in stage_hint_routes
                        },
                    },
                },
            },
        ]
    }


__all__ = [
    "ALLOWED_ANCHOR_TARGET_REGION_HINTS",
    "ALLOWED_DIAGNOSTIC_ANCHOR_TYPES",
    "ALLOWED_DIAGNOSTIC_ANCHOR_UNCERTAINTY",
    "ALLOWED_SOURCE_SEMANTIC_MATERIAL_STRENGTH",
    "ALLOWED_SOURCE_SEMANTIC_MATERIAL_CALIBRATION_CONFIDENCE",
    "ALLOWED_PROBE_ACTIONS",
    "ALLOWED_VLM_TOOLS",
    "ANCHOR_BBOX_PROPOSAL_FRAME",
    "ANCHOR_TARGET_INTENT_SCHEMA_VERSION",
    "BOX_EE_ACTIONS",
    "BOX_EE_MOVEMENT_ACTIONS",
    "COMPILED_PROBE_TARGET_TOOL_NAMES",
    "DEFAULT_MAX_CONTACTS",
    "DEFAULT_MAX_DEFORMATION_SAMPLE_VERTICES",
    "DEFAULT_MAX_FRAME_EDGE_PX",
    "DEFAULT_MAX_PROBE_DISTANCE_M",
    "DEFAULT_MAX_PROBE_DURATION_STEPS",
    "DEFAULT_MAX_PROBE_SPEED_M_S",
    "DEFAULT_MAX_PROBE_VERTICES",
    "DEFAULT_MAX_RESUME_STEPS",
    "DISALLOWED_PROBE_ACTIONS",
    "DEFAULT_DIAGNOSTIC_SIMULATE_STEPS",
    "MAX_DIAGNOSTIC_SESSION_ANCHORS",
    "MAX_PRE_EPISODE_PIN_BOXES",
    "MODEL_FACING_LIVE_TOOL_NAMES",
    "GenesisVlmActionLimits",
    "GenesisVlmSchemaError",
    "VLM_ACTION_SCHEMA_VERSION",
    "VLM_DIAGNOSTIC_SESSION_SCHEMA_VERSION",
    "VLM_FINAL_RECOMMENDATION_SCHEMA_VERSION",
    "VLM_PRE_EPISODE_SCHEMA_VERSION",
    "VLM_REFLECTION_SCHEMA_VERSION",
    "VLM_SOURCE_SEMANTIC_MATERIAL_AUDIT_SCHEMA_VERSION",
    "anchor_bbox_proposal_json_schema",
    "diagnostic_anchor_target_edit_json_schema",
    "diagnostic_anchor_target_intent_json_schema",
    "diagnostic_session_plan_json_schema",
    "source_semantic_material_audit_json_schema",
    "diagnostic_probe_target_edit_json_schema",
    "diagnostic_probe_target_intent_json_schema",
    "genesis_vlm_action_limits_from_config",
    "validate_diagnostic_anchor_target_edit",
    "validate_diagnostic_anchor_target_intent",
    "validate_diagnostic_anchor_region",
    "validate_diagnostic_probe_target_edit",
    "validate_diagnostic_probe_target_intent",
    "validate_anchor_bbox_proposal",
    "validate_diagnostic_session_plan",
    "validate_source_semantic_material_audit",
    "validate_vlm_action_decision",
    "validate_vlm_final_recommendation",
    "normalize_persisted_vlm_final_recommendation",
    "validate_vlm_pre_episode_plan",
    "validate_vlm_reflection_evidence",
    "vlm_action_decision_json_schema",
    "vlm_final_recommendation_json_schema",
    "vlm_pre_episode_plan_json_schema",
]
