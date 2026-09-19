from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROFILE_SCHEMA_VERSION = "hag4r-runtime-profile-v1"
DEFAULT_SAMPLE_INTERVAL_S = 5.0
DEFAULT_REPORT_INTERVAL_S = 60.0
DEFAULT_MAX_MONITOR_SECONDS = 24 * 60 * 60

LOGICAL_STAGE_ORDER: tuple[str, ...] = (
    "image_cleanup",
    "segmentation",
    "omnipart",
    "material_inference",
    "mesh_processing",
    "genesis_diagnostics",
    "post_mesh_texture",
    "final_export",
)

ARTIFACT_STAGE_BY_PATH_KEY: dict[str, str] = {
    "cleaned_image_path": "image_cleanup",
    "segmentation_manifest_path": "segmentation",
    "part_labels_path": "omnipart",
    "inferred_params_path": "material_inference",
    "monolithic_mesh_path": "mesh_processing",
    "uniform_params_path": "mesh_processing",
    "sim_diagnostics_report_path": "genesis_diagnostics",
    "post_mesh_texture_request_path": "post_mesh_texture",
    "textured_visual_mesh_path": "post_mesh_texture",
    "albedo_path": "post_mesh_texture",
    "visual_to_physics_binding_path": "post_mesh_texture",
    "visual_manifest_path": "post_mesh_texture",
    "texture_qa_report_path": "post_mesh_texture",
    "final_export_manifest_path": "final_export",
}


@dataclass(frozen=True)
class ProfileClock:
    utc_timestamp: str
    monotonic_ns: int


def capture_clock() -> ProfileClock:
    return ProfileClock(
        utc_timestamp=datetime.now(timezone.utc).isoformat(),
        monotonic_ns=time.monotonic_ns(),
    )


def logical_stage_name(stage_name: str) -> str:
    if stage_name in LOGICAL_STAGE_ORDER:
        return stage_name
    if stage_name == "sam3_omnipart_2d_segmentation":
        return "segmentation"
    if stage_name == "omnipart_generate_parts":
        return "omnipart"
    if stage_name == "hag4r_gpt_staged_material_inference":
        return "material_inference"
    if stage_name == "genesis_live_diagnostic_loop":
        return "genesis_diagnostics"
    if stage_name == "hag4r_post_mesh_texture":
        return "post_mesh_texture"
    if stage_name == "hag4r_final_export_bundle":
        return "final_export"
    if stage_name.startswith("hag4r_"):
        return "mesh_processing"
    return stage_name


def _profile_dir(run_root: str | Path) -> Path:
    root = Path(run_root).expanduser().resolve()
    if "outputs" not in root.parts:
        raise ValueError(f"profiling run root must be under outputs/: {root}")
    return root / "profiling"


def profile_events_path(run_root: str | Path) -> Path:
    return _profile_dir(run_root) / "runtime_events.jsonl"


def resource_samples_path(run_root: str | Path) -> Path:
    return _profile_dir(run_root) / "resource_samples.jsonl"


def _json_friendly(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_friendly(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(child) for child in value]
    return str(value)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(_json_friendly(payload), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, line)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def record_event(
    run_root: str | Path,
    event: str,
    *,
    stage: str = "",
    operation: str = "",
    status: str = "",
    span_id: str = "",
    parent_span_id: str = "",
    metadata: dict[str, Any] | None = None,
    clock: ProfileClock | None = None,
) -> bool:
    """Append one profile event without allowing profiler failures into the pipeline."""

    try:
        observed = clock or capture_clock()
        root = Path(run_root).expanduser().resolve()
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "event": event,
            "run_id": root.name,
            "stage": logical_stage_name(stage) if stage else "",
            "operation": operation,
            "status": status,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "utc_timestamp": observed.utc_timestamp,
            "monotonic_ns": observed.monotonic_ns,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "metadata": metadata or {},
        }
        _append_json_line(profile_events_path(root), payload)
        return True
    except Exception:
        return False


def start_span(
    run_root: str | Path,
    *,
    stage: str,
    operation: str,
    parent_span_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    ensure_resource_monitor_running(run_root)
    span_id = uuid.uuid4().hex
    record_event(
        run_root,
        "span_started",
        stage=stage,
        operation=operation,
        status="started",
        span_id=span_id,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )
    return span_id


def finish_span(
    run_root: str | Path,
    span_id: str,
    *,
    stage: str,
    operation: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    return record_event(
        run_root,
        "span_finished",
        stage=stage,
        operation=operation,
        status=status,
        span_id=span_id,
        metadata=metadata,
    )


def record_tool_status(
    run_root: str | Path,
    *,
    stage: str,
    tool_name: str,
    status: str,
) -> bool:
    if status == "started":
        ensure_resource_monitor_running(run_root)
    return record_event(
        run_root,
        "tool_status",
        stage=stage,
        operation=tool_name,
        status=status,
    )


def record_stage_terminal(
    run_root: str | Path,
    *,
    stage: str,
    operation: str,
    status: str,
) -> bool:
    return record_event(
        run_root,
        "stage_terminal",
        stage=stage,
        operation=operation,
        status=status,
    )


def start_automatic_profiling(
    run_root: str | Path,
    *,
    initializer_started: ProfileClock,
) -> bool:
    """Initialize profiling and launch its detached monitor; always fail open."""

    if os.environ.get("HAG4R_DISABLE_RUNTIME_PROFILING", "").strip().lower() in {"1", "true", "yes"}:
        return False
    try:
        root = Path(run_root).expanduser().resolve()
        record_event(root, "asset_started", status="running", clock=initializer_started)
        record_event(root, "initializer_completed", status="success")
        if not (root / "state.json").is_file():
            return True
        _spawn_resource_monitor(root)
        return True
    except Exception as error:
        record_event(
            run_root,
            "profiler_component_failed",
            operation="resource_monitor_start",
            status="failed",
            metadata={"error_type": type(error).__name__, "error": str(error)},
        )
        return False


def _spawn_resource_monitor(run_root: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "hag4r.agentic.runtime_profiler",
        "monitor",
        "--run-root",
        str(run_root),
    ]
    subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _resource_monitor_is_running(run_root: Path) -> bool:
    lock_path = _profile_dir(run_root) / "resource_monitor.lock"
    try:
        process_id = int(lock_path.read_text(encoding="utf-8").strip())
        command_line = Path(f"/proc/{process_id}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8",
            errors="replace",
        )
    except (OSError, ValueError):
        return False
    return "hag4r.agentic.runtime_profiler" in command_line and str(run_root) in command_line


def ensure_resource_monitor_running(run_root: str | Path) -> bool:
    """Restart a missing monitor from existing tool paths without involving the supervisor."""

    if os.environ.get("HAG4R_DISABLE_RUNTIME_PROFILING", "").strip().lower() in {"1", "true", "yes"}:
        return False
    try:
        root = Path(run_root).expanduser().resolve()
        if not (root / "state.json").is_file() or not profile_events_path(root).is_file():
            return False
        if _resource_monitor_is_running(root):
            return True
        _spawn_resource_monitor(root)
        return True
    except Exception:
        return False


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _seconds(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("utc_timestamp"), str):
            records.append(payload)
    records.sort(key=lambda item: _parse_timestamp(str(item["utc_timestamp"])))
    return records


def _paired_tool_spans(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    open_explicit: dict[str, dict[str, Any]] = {}
    open_implicit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        event_kind = str(event.get("event", ""))
        stage = logical_stage_name(str(event.get("stage", "")))
        operation = str(event.get("operation", ""))
        if event_kind == "span_started" and event.get("span_id"):
            open_explicit[str(event["span_id"])] = event
            continue
        if event_kind == "span_finished" and event.get("span_id"):
            start = open_explicit.pop(str(event["span_id"]), None)
            if start is not None:
                started_at = str(start["utc_timestamp"])
                ended_at = str(event["utc_timestamp"])
                spans.append(
                    {
                        "stage": stage or logical_stage_name(str(start.get("stage", ""))),
                        "operation": operation or str(start.get("operation", "")),
                        "status": str(event.get("status", "")),
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "duration_seconds": _seconds(
                            _parse_timestamp(started_at),
                            _parse_timestamp(ended_at),
                        ),
                    }
                )
            continue
        if event_kind != "tool_status":
            continue
        key = (stage, operation)
        status = str(event.get("status", ""))
        if status == "started":
            open_implicit.setdefault(key, []).append(event)
            continue
        candidates = open_implicit.get(key, [])
        if candidates:
            start = candidates.pop(0)
            started_at = str(start["utc_timestamp"])
            ended_at = str(event["utc_timestamp"])
            spans.append(
                {
                    "stage": stage,
                    "operation": operation,
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_seconds": _seconds(
                        _parse_timestamp(started_at),
                        _parse_timestamp(ended_at),
                    ),
                }
            )
    return spans


def _clip_and_union(
    spans: Iterable[dict[str, Any]],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for span in spans:
        start = max(window_start, _parse_timestamp(str(span["started_at"])))
        end = min(window_end, _parse_timestamp(str(span["ended_at"])))
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _stage_attempt_profile(
    *,
    stage: str,
    attempt: int,
    window_start: datetime,
    window_end: datetime,
    tool_spans: list[dict[str, Any]],
    terminal_status: str,
    terminal_operation: str,
) -> dict[str, Any]:
    matching = [
        span
        for span in tool_spans
        if logical_stage_name(str(span["stage"])) == stage
        and _parse_timestamp(str(span["ended_at"])) > window_start
        and _parse_timestamp(str(span["started_at"])) < window_end
    ]
    intervals = _clip_and_union(matching, window_start=window_start, window_end=window_end)
    wall_seconds = _seconds(window_start, window_end)
    tool_seconds = sum(_seconds(start, end) for start, end in intervals)
    if intervals:
        entry_gap_seconds = _seconds(window_start, intervals[0][0])
        terminal_gap_seconds = _seconds(intervals[-1][1], window_end)
        inter_tool_gap_seconds = sum(
            _seconds(previous[1], current[0])
            for previous, current in zip(intervals, intervals[1:])
        )
    else:
        entry_gap_seconds = wall_seconds
        inter_tool_gap_seconds = 0.0
        terminal_gap_seconds = 0.0
    agent_non_tool_seconds = max(0.0, wall_seconds - tool_seconds)
    return {
        "stage": stage,
        "attempt": attempt,
        "status": terminal_status,
        "terminal_operation": terminal_operation,
        "started_at": window_start.isoformat(),
        "ended_at": window_end.isoformat(),
        "wall_seconds": wall_seconds,
        "tool_seconds": tool_seconds,
        "agent_non_tool_seconds": agent_non_tool_seconds,
        "entry_gap_seconds": entry_gap_seconds,
        "inter_tool_gap_seconds": inter_tool_gap_seconds,
        "terminal_gap_seconds": terminal_gap_seconds,
        "tool_call_count": len(matching),
    }


def _resource_summary(run_root: str | Path) -> dict[str, Any]:
    samples = _load_json_lines(resource_samples_path(run_root))
    selected_index: int | None = None
    selected_uuid = ""
    try:
        state_path = Path(run_root).expanduser().resolve() / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        binding = state.get("worker_isolation", {}).get("gpu_binding", {})
        token = str(binding.get("effective_cuda_visible_devices") or "").strip()
        if token.isdigit():
            selected_index = int(token)
        elif token:
            selected_uuid = token
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    gpu_utils: list[float] = []
    for sample in samples:
        for gpu in sample.get("gpus", []) if isinstance(sample.get("gpus"), list) else []:
            if selected_index is not None and gpu.get("index") != selected_index:
                continue
            if selected_uuid and gpu.get("uuid") != selected_uuid:
                continue
            try:
                gpu_utils.append(float(gpu["utilization_gpu_percent"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "sample_count": len(samples),
        "gpu_sample_count": len(gpu_utils),
        "selected_gpu": selected_index if selected_index is not None else selected_uuid or None,
        "mean_gpu_utilization_percent": sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
        "gpu_busy_sample_percent": (
            100.0 * sum(value >= 10.0 for value in gpu_utils) / len(gpu_utils) if gpu_utils else None
        ),
    }


def build_profile_report(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    events = _load_json_lines(profile_events_path(root))
    tool_spans = _paired_tool_spans(events)
    if not events:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "run_id": root.name,
            "status": "no_profile_events",
            "stage_attempts": [],
            "resource_summary": _resource_summary(root),
        }

    asset_start_event = next((event for event in events if event.get("event") == "asset_started"), events[0])
    initializer_event = next(
        (event for event in events if event.get("event") == "initializer_completed"),
        asset_start_event,
    )
    asset_start = _parse_timestamp(str(asset_start_event["utc_timestamp"]))
    initializer_end = _parse_timestamp(str(initializer_event["utc_timestamp"]))
    terminal_events = [event for event in events if event.get("event") == "stage_terminal"]
    attempts: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    cursor = initializer_end
    for terminal in terminal_events:
        terminal_time = _parse_timestamp(str(terminal["utc_timestamp"]))
        if terminal_time < cursor:
            continue
        stage = logical_stage_name(str(terminal.get("stage", "unknown")))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        attempts.append(
            _stage_attempt_profile(
                stage=stage,
                attempt=stage_counts[stage],
                window_start=cursor,
                window_end=terminal_time,
                tool_spans=tool_spans,
                terminal_status=str(terminal.get("status", "")),
                terminal_operation=str(terminal.get("operation", "")),
            )
        )
        cursor = terminal_time

    last_event_time = _parse_timestamp(str(events[-1]["utc_timestamp"]))
    resource_samples = _load_json_lines(resource_samples_path(root))
    last_resource_time = (
        _parse_timestamp(str(resource_samples[-1]["utc_timestamp"]))
        if resource_samples
        else last_event_time
    )
    observed_end = max(cursor, last_event_time, last_resource_time)
    stage_wall = sum(float(item["wall_seconds"]) for item in attempts)
    tool_seconds = sum(float(item["tool_seconds"]) for item in attempts)
    agent_non_tool_seconds = sum(float(item["agent_non_tool_seconds"]) for item in attempts)
    initializer_seconds = _seconds(asset_start, initializer_end)
    in_progress_tail_seconds = _seconds(cursor, observed_end)
    observed_wall_seconds = _seconds(asset_start, observed_end)
    attributed_seconds = initializer_seconds + stage_wall + in_progress_tail_seconds
    artifact_milestones = [
        {
            "stage": logical_stage_name(str(event.get("stage", ""))),
            "path_key": str(event.get("operation", "")),
            "observed_at": str(event.get("utc_timestamp", "")),
            "seconds_from_asset_start": _seconds(
                asset_start,
                _parse_timestamp(str(event["utc_timestamp"])),
            ),
            "metadata": event.get("metadata", {}),
        }
        for event in events
        if event.get("event") == "artifact_observed"
    ]
    try:
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "run_id": root.name,
        "status": "terminal" if isinstance(state, dict) and _pipeline_terminal(state) else "in_progress",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_started_at": asset_start.isoformat(),
        "observed_ended_at": observed_end.isoformat(),
        "observed_wall_seconds": observed_wall_seconds,
        "initializer_seconds": initializer_seconds,
        "stage_wall_seconds": stage_wall,
        "tool_seconds": tool_seconds,
        "agent_non_tool_seconds": agent_non_tool_seconds,
        "in_progress_tail_seconds": in_progress_tail_seconds,
        "unattributed_seconds": max(0.0, observed_wall_seconds - attributed_seconds),
        "attributed_percent": (
            100.0 * min(observed_wall_seconds, attributed_seconds) / observed_wall_seconds
            if observed_wall_seconds > 0
            else 100.0
        ),
        "stage_attempts": attempts,
        "tool_spans": tool_spans,
        "artifact_milestones": artifact_milestones,
        "resource_summary": _resource_summary(root),
    }


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Automatic Runtime Profile — {report['run_id']}",
        "",
        f"Status: `{report['status']}`  ",
        f"Observed wall time: **{_format_duration(float(report.get('observed_wall_seconds', 0.0)))}**  ",
        f"Attributed: **{float(report.get('attributed_percent', 0.0)):.1f}%**  ",
        f"Initializer: **{_format_duration(float(report.get('initializer_seconds', 0.0)))}**  ",
        f"Measured tool execution: **{_format_duration(float(report.get('tool_seconds', 0.0)))}**  ",
        f"Agent non-tool wall time: **{_format_duration(float(report.get('agent_non_tool_seconds', 0.0)))}**",
        "",
        "`agent_non_tool` is the automatically measured time outside known tool spans. It can include",
        "model-service latency, instruction loading, artifact inspection, reasoning, and response synthesis.",
        "",
        "## Stage attempts",
        "",
        "| Stage | Attempt | Status | Wall | Tools | Agent non-tool | Entry gap | Inter-tool gap | Terminal gap |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for attempt in report.get("stage_attempts", []):
        lines.append(
            (
                "| {stage} | {attempt} | {status} | {wall} | {tools} | {non_tool} | "
                "{entry} | {inter} | {terminal} |"
            ).format(
                stage=attempt["stage"],
                attempt=attempt["attempt"],
                status=attempt["status"],
                wall=_format_duration(float(attempt["wall_seconds"])),
                tools=_format_duration(float(attempt["tool_seconds"])),
                non_tool=_format_duration(float(attempt["agent_non_tool_seconds"])),
                entry=_format_duration(float(attempt["entry_gap_seconds"])),
                inter=_format_duration(float(attempt["inter_tool_gap_seconds"])),
                terminal=_format_duration(float(attempt["terminal_gap_seconds"])),
            )
        )
    milestones = report.get("artifact_milestones", [])
    if milestones:
        lines.extend(
            [
                "",
                "## Artifact milestones",
                "",
                "| Stage | Artifact | First observed after asset start |",
                "|---|---|---:|",
            ]
        )
        for milestone in milestones:
            lines.append(
                f"| {milestone['stage']} | {milestone['path_key']} | "
                f"{_format_duration(float(milestone['seconds_from_asset_start']))} |"
            )
    resources = report.get("resource_summary", {})
    lines.extend(["", "## Resource sampling", ""])
    lines.append(f"Samples: **{int(resources.get('sample_count', 0))}**")
    mean_gpu = resources.get("mean_gpu_utilization_percent")
    busy_gpu = resources.get("gpu_busy_sample_percent")
    if mean_gpu is not None:
        lines.append(f"Mean GPU utilization: **{float(mean_gpu):.1f}%**")
    if busy_gpu is not None:
        lines.append(f"GPU busy samples (>=10% utilization): **{float(busy_gpu):.1f}%**")
    lines.append("")
    return "\n".join(lines)


def write_profile_report(run_root: str | Path) -> bool:
    try:
        root = Path(run_root).expanduser().resolve()
        report = build_profile_report(root)
        directory = _profile_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "runtime_profile.json"
        markdown_path = directory / "runtime_profile.md"
        json_tmp = json_path.with_suffix(f".json.{os.getpid()}.tmp")
        md_tmp = markdown_path.with_suffix(f".md.{os.getpid()}.tmp")
        json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_tmp.write_text(_markdown_report(report), encoding="utf-8")
        os.replace(json_tmp, json_path)
        os.replace(md_tmp, markdown_path)
        return True
    except Exception:
        return False


def _read_proc_cpu() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, ValueError, IndexError):
        return None


def _read_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                result[f"{key.lower()}_kib"] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return result


def _read_disk_sectors() -> dict[str, int]:
    read_sectors = 0
    written_sectors = 0
    try:
        for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if (
                len(fields) < 10
                or fields[2].startswith(("loop", "ram"))
                or not (Path("/sys/block") / fields[2]).exists()
            ):
                continue
            read_sectors += int(fields[5])
            written_sectors += int(fields[9])
    except (OSError, ValueError, IndexError):
        return {}
    return {
        "disk_read_sectors": read_sectors,
        "disk_written_sectors": written_sectors,
    }


def _sample_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "utilization_gpu_percent": float(fields[2]),
                    "memory_used_mib": float(fields[3]),
                    "memory_total_mib": float(fields[4]),
                    "power_draw_w": float(fields[5]) if fields[5] not in {"N/A", "[N/A]"} else None,
                }
            )
        except ValueError:
            continue
    return gpus


def _resource_sample(previous_cpu: tuple[int, int] | None) -> tuple[dict[str, Any], tuple[int, int] | None]:
    clock = capture_clock()
    current_cpu = _read_proc_cpu()
    cpu_percent: float | None = None
    if previous_cpu is not None and current_cpu is not None:
        total_delta = current_cpu[0] - previous_cpu[0]
        idle_delta = current_cpu[1] - previous_cpu[1]
        if total_delta > 0:
            cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = None
    payload: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "utc_timestamp": clock.utc_timestamp,
        "monotonic_ns": clock.monotonic_ns,
        "cpu_utilization_percent": cpu_percent,
        "load_average_1m": load_1m,
        "load_average_5m": load_5m,
        "load_average_15m": load_15m,
        "gpus": _sample_gpus(),
    }
    payload.update(_read_memory())
    payload.update(_read_disk_sectors())
    return payload, current_cpu


def _record_new_artifacts(
    run_root: Path,
    state: dict[str, Any],
    seen_artifacts: set[tuple[str, str]],
) -> None:
    paths = state.get("paths", {})
    if not isinstance(paths, dict):
        return
    active_revision = str(state.get("active_revision") or "")
    for path_key, stage in ARTIFACT_STAGE_BY_PATH_KEY.items():
        raw_path = paths.get(path_key)
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        identity = (path_key, str(path))
        if identity in seen_artifacts or not path.is_file():
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        seen_artifacts.add(identity)
        record_event(
            run_root,
            "artifact_observed",
            stage=stage,
            operation=path_key,
            status="present",
            metadata={
                "active_revision": active_revision,
                "size_bytes": size_bytes,
            },
        )


def _pipeline_terminal(state: dict[str, Any]) -> bool:
    stage_runtime = state.get("stage_runtime", {})
    if isinstance(stage_runtime, dict):
        for stage_key, stage_record in stage_runtime.items():
            terminal = stage_record.get("terminal", {}) if isinstance(stage_record, dict) else {}
            status = str(terminal.get("status", "")) if isinstance(terminal, dict) else ""
            if status in {"halted", "failed", "error"}:
                return True
            if stage_key == "final_export" and status:
                return True
        terminate_after_stage = str(state.get("terminate_after_stage") or "")
        if terminate_after_stage:
            terminal = stage_runtime.get(terminate_after_stage, {})
            if isinstance(terminal, dict) and terminal.get("terminal"):
                return True
    diagnostic_terminal = state.get("diagnostics", {}).get("terminal", {})
    if isinstance(diagnostic_terminal, dict) and diagnostic_terminal.get("status") in {"halted", "failed"}:
        return True
    return False


def monitor_run(
    run_root: str | Path,
    *,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    report_interval_s: float = DEFAULT_REPORT_INTERVAL_S,
    max_monitor_seconds: float = DEFAULT_MAX_MONITOR_SECONDS,
) -> int:
    root = Path(run_root).expanduser().resolve()
    directory = _profile_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "resource_monitor.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        record_event(root, "resource_monitor_started", status="running")
        started = time.monotonic()
        last_report = 0.0
        previous_cpu: tuple[int, int] | None = None
        seen_artifacts: set[tuple[str, str]] = set()
        while time.monotonic() - started < max_monitor_seconds:
            try:
                sample, previous_cpu = _resource_sample(previous_cpu)
                _append_json_line(resource_samples_path(root), sample)
            except Exception as error:
                record_event(
                    root,
                    "profiler_component_failed",
                    operation="resource_sample",
                    status="failed",
                    metadata={"error_type": type(error).__name__, "error": str(error)},
                )
            now = time.monotonic()
            if now - last_report >= report_interval_s:
                write_profile_report(root)
                last_report = now
            state_file = root / "state.json"
            try:
                state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
            except (OSError, json.JSONDecodeError):
                state = {}
            if isinstance(state, dict):
                _record_new_artifacts(root, state, seen_artifacts)
            if isinstance(state, dict) and _pipeline_terminal(state):
                record_event(root, "resource_monitor_stopped", status="pipeline_terminal")
                write_profile_report(root)
                return 0
            time.sleep(max(0.25, sample_interval_s))
        record_event(root, "resource_monitor_stopped", status="max_runtime_reached")
        write_profile_report(root)
        return 0
    except Exception as error:
        record_event(
            root,
            "profiler_component_failed",
            operation="resource_monitor",
            status="failed",
            metadata={"error_type": type(error).__name__, "error": str(error)},
        )
        return 0
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-open HAG4R runtime profiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--run-root", required=True)
    monitor.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    monitor.add_argument("--report-interval-s", type=float, default=DEFAULT_REPORT_INTERVAL_S)
    monitor.add_argument("--max-monitor-seconds", type=float, default=DEFAULT_MAX_MONITOR_SECONDS)
    report = subparsers.add_parser("report")
    report.add_argument("--run-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "monitor":
        return monitor_run(
            args.run_root,
            sample_interval_s=args.sample_interval_s,
            report_interval_s=args.report_interval_s,
            max_monitor_seconds=args.max_monitor_seconds,
        )
    return 0 if write_profile_report(args.run_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
