#!/usr/bin/env python3
from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / ".agents/skills").is_dir():
            return candidate
    raise RuntimeError("update_diagnostic_faults.py is not inside the HAG4R repository")


REPO_ROOT = _find_repo_root()
KB_PATH = REPO_ROOT / ".agents/diaggen_diag_knowledge/diagnostic_faults.md"
LOCK_PATH = KB_PATH.with_name("diagnostic_faults.md.lock")
ENTRY_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*$")
FIELDS = (
    "signature_tokens",
    "phase",
    "directly_proven_condition",
    "likely_diagnosed_cause",
    "discriminating_checks",
    "owner_route",
    "status",
    "provenance",
)
STATUSES = {"candidate", "verified", "deprecated"}
MAX_TEXT = 4000
MAX_ITEMS = 32
DEFAULT_PREAMBLE = "# Diagnostic Faults\n\n"


def _paths() -> tuple[Path, Path]:
    override = os.environ.get("_HAG4R_DIAGNOSTIC_KB_TEST_PATH")
    if override:
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            raise ValueError("diagnostic KB path override is test-only")
        kb = Path(override).expanduser().resolve()
        return kb, kb.with_name(kb.name + ".lock")
    return KB_PATH, LOCK_PATH


def _bounded_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = " ".join(value.split())
    if len(normalized) > MAX_TEXT:
        raise ValueError(f"{name} exceeds {MAX_TEXT} characters")
    return normalized


def _bounded_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_ITEMS:
        raise ValueError(f"{name} must contain 1..{MAX_ITEMS} strings")
    normalized = [_bounded_text(item, f"{name} item") for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique values")
    return normalized


def _signature_key(tokens: list[str]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(token.lower().split()) for token in tokens}))


def _parse(text: str) -> tuple[str, list[dict[str, Any]]]:
    matches = list(re.finditer(r"(?m)^## ([a-z0-9][a-z0-9-]*)\n", text))
    preamble = text[: matches[0].start()] if matches else text
    entries: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        entry: dict[str, Any] = {"id": match.group(1)}
        for line in body.splitlines():
            field = next((name for name in FIELDS if line.startswith(f"- {name}: ")), None)
            if field is None:
                continue
            raw = line.removeprefix(f"- {field}: ").strip()
            if raw.startswith("`") and raw.endswith("`"):
                raw = raw[1:-1]
            try:
                entry[field] = json.loads(raw)
            except json.JSONDecodeError:
                entry[field] = raw
        missing = [field for field in FIELDS if field not in entry]
        if missing:
            raise ValueError(f"knowledge entry {entry['id']} is missing fields: {missing}")
        entries.append(entry)
    return preamble.rstrip() + "\n\n", entries


def _validate_entry(data: dict[str, Any], *, candidate_only: bool) -> dict[str, Any]:
    expected = {"id", *FIELDS}
    if set(data) != expected:
        raise ValueError(f"entry keys must equal {sorted(expected)}")
    entry_id = _bounded_text(data["id"], "id")
    if not ENTRY_ID_RE.fullmatch(entry_id):
        raise ValueError("id must be lowercase kebab-case ending in -v<positive-int>")
    status = _bounded_text(data["status"], "status")
    if status not in STATUSES or (candidate_only and status != "candidate"):
        raise ValueError("append-candidate status must equal candidate")
    return {
        "id": entry_id,
        "signature_tokens": _bounded_list(data["signature_tokens"], "signature_tokens"),
        "phase": _bounded_text(data["phase"], "phase"),
        "directly_proven_condition": _bounded_text(data["directly_proven_condition"], "directly_proven_condition"),
        "likely_diagnosed_cause": _bounded_text(data["likely_diagnosed_cause"], "likely_diagnosed_cause"),
        "discriminating_checks": _bounded_list(data["discriminating_checks"], "discriminating_checks"),
        "owner_route": _bounded_text(data["owner_route"], "owner_route"),
        "status": status,
        "provenance": _bounded_list(data["provenance"], "provenance"),
    }


def _validate_existing_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = [_validate_entry(entry, candidate_only=False) for entry in entries]
    if validated != entries:
        raise ValueError("existing knowledge entries must already be normalized and valid")
    ids = [entry["id"] for entry in validated]
    signatures = [_signature_key(entry["signature_tokens"]) for entry in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("existing knowledge entries contain duplicate ids")
    if len(signatures) != len(set(signatures)):
        raise ValueError("existing knowledge entries contain duplicate normalized signatures")
    return validated


def _assert_allowed_change(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    command: str,
    target_id: str,
) -> None:
    before_by_id = {entry["id"]: entry for entry in before}
    after_by_id = {entry["id"]: entry for entry in after}
    if command == "append-candidate":
        if set(after_by_id) != {*before_by_id, target_id}:
            raise ValueError("append-candidate may only add one entry")
        if any(after_by_id[entry_id] != entry for entry_id, entry in before_by_id.items()):
            raise ValueError("append-candidate may not mutate existing entries")
        return
    if set(after_by_id) != set(before_by_id):
        raise ValueError("append-provenance may not add, delete, or rename entries")
    for entry_id, entry in before_by_id.items():
        updated = after_by_id[entry_id]
        if entry_id != target_id:
            if updated != entry:
                raise ValueError("append-provenance may not mutate unrelated entries")
            continue
        immutable_before = {key: value for key, value in entry.items() if key != "provenance"}
        immutable_after = {key: value for key, value in updated.items() if key != "provenance"}
        if immutable_after != immutable_before:
            raise ValueError(
                "append-provenance may not mutate, promote, deprecate, or replace an entry"
            )
        if updated["provenance"][: len(entry["provenance"])] != entry["provenance"]:
            raise ValueError("append-provenance may only append provenance")


def _render(preamble: str, entries: list[dict[str, Any]]) -> str:
    chunks = [preamble.rstrip()]
    for entry in entries:
        lines = [f"## {entry['id']}", ""]
        for field in FIELDS:
            lines.append(f"- {field}: `{json.dumps(entry[field], ensure_ascii=False, separators=(',', ':'))}`")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks).rstrip() + "\n"


def _write_atomic(path: Path, text: str) -> None:
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if "temporary" in locals() and temporary.exists():
            temporary.unlink()
        raise


def update(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    kb_path, lock_path = _paths()
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        text = kb_path.read_text(encoding="utf-8") if kb_path.is_file() else DEFAULT_PREAMBLE
        preamble, entries = _parse(text)
        entries = _validate_existing_entries(entries)
        before = copy.deepcopy(entries)
        if command == "append-candidate":
            candidate = _validate_entry(payload, candidate_only=True)
            if any(entry["id"] == candidate["id"] for entry in entries):
                raise ValueError(f"duplicate knowledge entry id: {candidate['id']}")
            signature = _signature_key(candidate["signature_tokens"])
            if any(_signature_key(entry["signature_tokens"]) == signature for entry in entries):
                raise ValueError("duplicate normalized knowledge signature")
            entries.append(candidate)
            result = {"status": "appended", "id": candidate["id"]}
        elif command == "append-provenance":
            if set(payload) != {"id", "provenance"}:
                raise ValueError("append-provenance keys must equal id and provenance")
            entry_id = _bounded_text(payload["id"], "id")
            provenance = _bounded_text(payload["provenance"], "provenance")
            entry = next((item for item in entries if item["id"] == entry_id), None)
            if entry is None:
                raise ValueError(f"unknown knowledge entry id: {entry_id}")
            if provenance in entry["provenance"]:
                return {"status": "unchanged", "id": entry_id}
            entry["provenance"].append(provenance)
            result = {"status": "appended", "id": entry_id}
        else:
            raise ValueError("command must be append-candidate or append-provenance")
        _assert_allowed_change(before, entries, command=command, target_id=result["id"])
        _validate_existing_entries(entries)
        _write_atomic(kb_path, _render(preamble, entries))
        return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise ValueError("usage: update_diagnostic_faults.py append-candidate|append-provenance")
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("stdin must contain one JSON object")
    print(json.dumps(update(argv[1], payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
