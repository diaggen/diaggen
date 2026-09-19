from __future__ import annotations

import json
from dataclasses import is_dataclass
from enum import Enum
from typing import Any

from hag4r.agentic.state import SimDiagnosticRoute, to_json_dict


DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION = "hag4r-diagnostic-repair-brief-v3"
MAX_DIAGNOSTIC_BRIEF_TEXT_CHARS = 900

STAGE_HINT_FIELDS: dict[SimDiagnosticRoute, str] = {
    SimDiagnosticRoute.SEGMENTATION: "segmentation_hints",
    SimDiagnosticRoute.MATERIAL_INFERENCE: "material_inference_hints",
    SimDiagnosticRoute.MESH_PROCESSING: "mesh_processing_hints",
}


REPAIR_ROUTE_TO_DESTINATION_STAGE: dict[str, str] = {
    SimDiagnosticRoute.SEGMENTATION.value: "segmentation",
    SimDiagnosticRoute.MATERIAL_INFERENCE.value: "material_inference",
    SimDiagnosticRoute.MESH_PROCESSING.value: "mesh_processing",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_json_dict(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple):
        return [text for item in value if (text := str(item).strip())]
    text = str(value).strip()
    return [text] if text else []


def _bounded_text(value: Any, *, max_chars: int = MAX_DIAGNOSTIC_BRIEF_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_diagnostic_cue(cue: Any) -> dict[str, Any]:
    payload = _jsonable(cue)
    if not isinstance(payload, dict):
        raise ValueError(f"Diagnostic cue must be a dict-like object, got {type(cue).__name__}")
    normalized = dict(payload)
    retired_keys = {"image_cleanup_hints", "object_description_hints"}.intersection(normalized)
    if retired_keys:
        raise ValueError("retired diagnostic cue key(s): " + ", ".join(sorted(retired_keys)))
    route = normalized.get("route", SimDiagnosticRoute.ACCEPT.value)
    route_value = route.value if isinstance(route, SimDiagnosticRoute) else str(route)
    if route_value not in {member.value for member in SimDiagnosticRoute}:
        raise ValueError(f"unsupported diagnostic route: {route_value!r}")
    if route_value == SimDiagnosticRoute.ACCEPT.value:
        raise ValueError("diagnostic repair cue cannot use route=accept")
    cues = _string_list(normalized.get("diagnostic_cues", []))
    if not cues:
        # Read-only migration for persisted v2 cue records.
        hint_field = STAGE_HINT_FIELDS[SimDiagnosticRoute(route_value)]
        cues = _string_list(normalized.get(hint_field, []))
        if not cues:
            cues = _string_list(normalized.get("reason", ""))
    if not cues:
        raise ValueError("diagnostic repair cue requires at least one non-empty diagnostic cue")
    return {"route": route_value, "diagnostic_cues": cues}


def normalize_diagnostic_cues(cues: tuple[Any, ...] | list[Any]) -> list[dict[str, Any]]:
    return [normalize_diagnostic_cue(cue) for cue in cues]


def diagnostic_cues_for_route(cues: tuple[Any, ...] | list[Any], route: SimDiagnosticRoute | str) -> list[dict[str, Any]]:
    target_route = SimDiagnosticRoute(route)
    return [
        cue
        for cue in normalize_diagnostic_cues(cues)
        if cue.get("route") == target_route.value
    ]


def diagnostic_hint_text_for_route(cues: tuple[Any, ...] | list[Any], route: SimDiagnosticRoute | str) -> str:
    target_route = SimDiagnosticRoute(route)
    if target_route is SimDiagnosticRoute.ACCEPT:
        return ""
    selected = diagnostic_cues_for_route(cues, target_route)
    hints: list[str] = []
    for cue in selected:
        for hint in cue["diagnostic_cues"]:
            text = str(hint).strip()
            if text:
                hints.append(text)
    if not hints:
        return ""
    return json.dumps({"route": target_route.value, "hints": hints}, indent=2)


def repair_route_to_destination_stage(route: SimDiagnosticRoute | str) -> str:
    route_value = route.value if isinstance(route, SimDiagnosticRoute) else str(route)
    return REPAIR_ROUTE_TO_DESTINATION_STAGE.get(route_value, "")


def _probe_report_summary(probe_report: Any) -> dict[str, Any]:
    if not isinstance(probe_report, dict):
        return {}
    probes = probe_report.get("probes", [])
    probe_items = [probe for probe in probes if isinstance(probe, dict)] if isinstance(probes, list | tuple) else []
    return {
        "diagnostic_run_index": probe_report.get("diagnostic_run_index"),
        "episode_count": int(probe_report.get("episode_count", 0) or 0),
        "probe_count": len(probe_items),
        "probe_tools": _string_list([probe.get("tool", "") for probe in probe_items])[:8],
        "report_refs": {
            str(key): str(value)
            for key, value in (probe_report.get("report_refs", {}) or {}).items()
            if str(value).strip()
        } if isinstance(probe_report.get("report_refs", {}), dict) else {},
    }


def build_diagnostic_repair_brief(
    *,
    recommendation: dict[str, Any],
    probe_report: dict[str, Any],
    summary_path: str,
    first_rerouted_stage_skill_path: str,
    source_revision: str = "",
) -> dict[str, Any]:
    route = str(recommendation["route"])
    destination_stage = repair_route_to_destination_stage(route)
    if not destination_stage:
        return {}
    diagnostic_cues = _string_list(recommendation.get("diagnostic_cues", []))
    if not diagnostic_cues:
        return {}
    return {
        "schema_version": DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION,
        "route": route,
        "designated_destination_stage": destination_stage,
        "diagnostic_cues": diagnostic_cues,
        "probe_report_summary": _probe_report_summary(probe_report),
        "diagnostic_summary_path": summary_path,
        "source_revision": source_revision,
        "first_rerouted_stage_skill_path": first_rerouted_stage_skill_path,
    }


def diagnostic_repair_brief_for_stage(state: dict[str, Any], stage_key: str) -> dict[str, Any]:
    route_state = state.get("route_state", {})
    brief = route_state.get("diagnostic_repair_brief", {}) if isinstance(route_state, dict) else {}
    if not brief:
        brief = state.get("diagnostic_repair_brief", {})
    if not isinstance(brief, dict):
        return {}
    if str(brief.get("designated_destination_stage", "") or "") != stage_key:
        return {}
    return dict(brief)


def compact_diagnostic_repair_summary(brief: dict[str, Any]) -> str:
    if not brief:
        return ""
    parts: list[str] = []
    route = str(brief.get("route", "") or "").strip()
    if route:
        parts.append(f"route={route}")
    cues = _string_list(brief.get("diagnostic_cues", []))
    if cues:
        parts.append("cues=" + " | ".join(_bounded_text(cue, max_chars=180) for cue in cues[:4]))
    return "; ".join(parts)


def diagnostic_repair_brief_text(brief: dict[str, Any]) -> str:
    if not brief:
        return ""
    payload = {
        "route": brief.get("route", ""),
        "designated_destination_stage": brief.get("designated_destination_stage", ""),
        "diagnostic_cues": brief.get("diagnostic_cues", []),
        "probe_report_summary": brief.get("probe_report_summary", {}),
        "diagnostic_summary_path": brief.get("diagnostic_summary_path", ""),
        "source_revision": brief.get("source_revision", ""),
        "first_rerouted_stage_skill_path": brief.get("first_rerouted_stage_skill_path", ""),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "DIAGNOSTIC_REPAIR_BRIEF_SCHEMA_VERSION",
    "REPAIR_ROUTE_TO_DESTINATION_STAGE",
    "STAGE_HINT_FIELDS",
    "build_diagnostic_repair_brief",
    "compact_diagnostic_repair_summary",
    "diagnostic_cues_for_route",
    "diagnostic_hint_text_for_route",
    "diagnostic_repair_brief_for_stage",
    "diagnostic_repair_brief_text",
    "normalize_diagnostic_cue",
    "normalize_diagnostic_cues",
    "repair_route_to_destination_stage",
]
