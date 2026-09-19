from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenesisDiagnosticsConfig:
    enable: bool
    genesis_root: Path | None = None
    genesis_env_path: Path | None = None
    genesis_live_command: str = "python -m genesis.live.server"


@dataclass(frozen=True)
class RunConfig:
    genesis_diagnostics: GenesisDiagnosticsConfig
    source_path: Path
    raw_config: dict[str, Any]


DEFAULT_RUN_CONFIG_PATH = Path("configs/config_genesis_diagnostics_enabled.yaml")
_GENESIS_DIAGNOSTICS_FIELDS = frozenset(
    {"enable", "genesis_root", "genesis_env_path", "genesis_live_command"}
)


def _unknown_keys(mapping: dict[str, Any], allowed: frozenset[str]) -> list[str]:
    return sorted(str(key) for key in mapping if str(key) not in allowed)


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _resolve_config_path(path: str | Path, repo_root: str | Path | None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        base = Path(repo_root).expanduser() if repo_root is not None else Path.cwd()
        resolved = base / resolved
    return resolved.resolve()


def _resolve_config_relative_path(path: str, *, source_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = source_dir / resolved
    return resolved.resolve()


def _parse_genesis_diagnostics(value: Any, *, source_dir: Path) -> GenesisDiagnosticsConfig:
    path = "genesis_diagnostics"
    mapping = _require_mapping(value, path)
    unknown = _unknown_keys(mapping, _GENESIS_DIAGNOSTICS_FIELDS)
    if unknown:
        raise ValueError(f"{path} has unknown key(s): {', '.join(unknown)}")
    if "enable" not in mapping:
        raise ValueError(f"{path}.enable is required")
    enable = mapping["enable"]
    if not isinstance(enable, bool):
        raise ValueError(f"{path}.enable must be a bool")
    genesis_root = mapping.get("genesis_root")
    if genesis_root is not None and (not isinstance(genesis_root, str) or not genesis_root.strip()):
        raise ValueError(f"{path}.genesis_root must be a non-empty string")
    genesis_env_path = mapping.get("genesis_env_path")
    if genesis_env_path is not None and (not isinstance(genesis_env_path, str) or not genesis_env_path.strip()):
        raise ValueError(f"{path}.genesis_env_path must be a non-empty string")
    genesis_live_command = mapping.get("genesis_live_command", "python -m genesis.live.server")
    if not isinstance(genesis_live_command, str) or not genesis_live_command.strip():
        raise ValueError(f"{path}.genesis_live_command must be a non-empty string")
    return GenesisDiagnosticsConfig(
        enable=enable,
        genesis_root=(
            _resolve_config_relative_path(genesis_root, source_dir=source_dir)
            if isinstance(genesis_root, str)
            else None
        ),
        genesis_env_path=(
            _resolve_config_relative_path(genesis_env_path, source_dir=source_dir)
            if isinstance(genesis_env_path, str)
            else None
        ),
        genesis_live_command=genesis_live_command.strip(),
    )


def load_run_config(path: str | Path, *, repo_root: str | Path | None = None) -> RunConfig:
    source_path = _resolve_config_path(path, repo_root)
    if not source_path.exists():
        raise FileNotFoundError(f"run config does not exist: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"run config is empty: {source_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"run config root must be a mapping: {source_path}")
    if "genesis_diagnostics" not in raw:
        raise ValueError("run config missing top-level key: genesis_diagnostics")
    genesis_diagnostics = _parse_genesis_diagnostics(
        raw["genesis_diagnostics"],
        source_dir=source_path.parent,
    )
    return RunConfig(
        genesis_diagnostics=genesis_diagnostics,
        source_path=source_path,
        raw_config=raw,
    )


def run_config_audit_state(config: RunConfig) -> dict[str, Any]:
    diagnostics = {
        "enable": config.genesis_diagnostics.enable,
        "genesis_live_command": config.genesis_diagnostics.genesis_live_command,
    }
    if config.genesis_diagnostics.genesis_root is not None:
        diagnostics["genesis_root"] = str(config.genesis_diagnostics.genesis_root)
    if config.genesis_diagnostics.genesis_env_path is not None:
        diagnostics["genesis_env_path"] = str(config.genesis_diagnostics.genesis_env_path)
    return {
        "source_path": str(config.source_path),
        "genesis_diagnostics": diagnostics,
    }


__all__ = [
    "DEFAULT_RUN_CONFIG_PATH",
    "GenesisDiagnosticsConfig",
    "RunConfig",
    "load_run_config",
    "run_config_audit_state",
]
