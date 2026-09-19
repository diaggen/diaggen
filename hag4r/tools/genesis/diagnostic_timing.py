from __future__ import annotations

import math

DIAGNOSTIC_SCENE_TIMESTEP_S = 0.001
DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS = 10
DIAGNOSTIC_RENDER_FPS = 30
DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S = 0.1


def min_diagnostic_sim_steps(
    *,
    duration_s: float = DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
    timestep_s: float = DIAGNOSTIC_SCENE_TIMESTEP_S,
) -> int:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("diagnostic duration_s must be finite and > 0")
    if not math.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("diagnostic timestep_s must be finite and > 0")
    return max(1, math.ceil(duration_s / timestep_s))


def min_diagnostic_captured_frames(
    *,
    duration_s: float = DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
    timestep_s: float = DIAGNOSTIC_SCENE_TIMESTEP_S,
    capture_every_n_steps: int = DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS,
) -> int:
    if (
        isinstance(capture_every_n_steps, bool)
        or not isinstance(capture_every_n_steps, int)
        or capture_every_n_steps <= 0
    ):
        raise ValueError("diagnostic capture_every_n_steps must be a positive integer")
    return min_diagnostic_sim_steps(duration_s=duration_s, timestep_s=timestep_s * capture_every_n_steps)


DEFAULT_DIAGNOSTIC_SIMULATE_STEPS = min_diagnostic_sim_steps(
    duration_s=DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S,
)
ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS = DEFAULT_DIAGNOSTIC_SIMULATE_STEPS
ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS = 500


def adaptive_compiled_probe_resume_steps(estimated_motion_steps: int | float) -> int:
    if isinstance(estimated_motion_steps, bool) or not isinstance(estimated_motion_steps, (int, float)):
        raise TypeError("estimated_motion_steps must be an integer-like finite positive number")
    if not math.isfinite(float(estimated_motion_steps)) or float(estimated_motion_steps) <= 0.0:
        raise ValueError("estimated_motion_steps must be finite and > 0")
    rounded_steps = int(estimated_motion_steps)
    if float(rounded_steps) != float(estimated_motion_steps):
        raise ValueError("estimated_motion_steps must be integer-like")
    return min(
        max(ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS, rounded_steps),
        ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS,
    )


__all__ = [
    "ADAPTIVE_COMPILED_PROBE_MAX_RESUME_STEPS",
    "ADAPTIVE_COMPILED_PROBE_MIN_RESUME_STEPS",
    "DEFAULT_DIAGNOSTIC_SIMULATE_DURATION_S",
    "DEFAULT_DIAGNOSTIC_SIMULATE_STEPS",
    "DIAGNOSTIC_LIVE_CAPTURE_EVERY_N_STEPS",
    "DIAGNOSTIC_RENDER_FPS",
    "DIAGNOSTIC_SCENE_TIMESTEP_S",
    "adaptive_compiled_probe_resume_steps",
    "min_diagnostic_captured_frames",
    "min_diagnostic_sim_steps",
]
