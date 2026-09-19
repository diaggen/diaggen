from __future__ import annotations

from hag4r.agentic.runtime_state import apply_diagnostic_recommendation


PIPELINE_TRANSITION_TOOLS = (apply_diagnostic_recommendation,)


__all__ = [
    "PIPELINE_TRANSITION_TOOLS",
    "apply_diagnostic_recommendation",
]
