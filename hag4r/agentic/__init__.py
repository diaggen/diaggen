"""Public skill-suite API for HAG4R asset generation."""

from importlib import import_module

from hag4r.agentic.state import (
    ArtifactManifest,
    ArtifactRef,
    ArtifactRole,
    AssetBundle,
    AssetRunState,
    DiagnosticObservation,
    GenesisAction,
    GenesisActionType,
    GenesisEpisodeResult,
    GenesisEpisodeSpec,
    GenesisKeyboardKey,
    ObservationSignal,
    PartIdentity,
    PartMap,
    RefinementAction,
    RefinementTarget,
    Severity,
    SimDiagnosticCue,
    SimDiagnosticRoute,
    Stage,
    StageRunResult,
)


_LAZY_EXPORTS = {
    "GenesisDiagnosticsConfig": ("hag4r.agentic.run_config", "GenesisDiagnosticsConfig"),
    "RunConfig": ("hag4r.agentic.run_config", "RunConfig"),
    "load_run_config": ("hag4r.agentic.run_config", "load_run_config"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'hag4r.agentic' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ArtifactManifest",
    "ArtifactRef",
    "ArtifactRole",
    "AssetBundle",
    "AssetRunState",
    "DiagnosticObservation",
    "GenesisAction",
    "GenesisActionType",
    "GenesisEpisodeResult",
    "GenesisEpisodeSpec",
    "GenesisKeyboardKey",
    "ObservationSignal",
    "PartIdentity",
    "PartMap",
    "RefinementAction",
    "RefinementTarget",
    "Severity",
    "SimDiagnosticCue",
    "SimDiagnosticRoute",
    "Stage",
    "StageRunResult",
    "GenesisDiagnosticsConfig",
    "RunConfig",
    "load_run_config",
]
