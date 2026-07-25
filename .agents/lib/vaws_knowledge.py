#!/usr/bin/env python3
"""Validate the JSON-compatible YAML knowledge files used by domain skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
KNOWLEDGE_FILES = {
    "version-compatibility.yaml": "version-compatibility",
    "model-capabilities.yaml": "model-capabilities",
    "parallelism-compatibility.yaml": "parallelism-compatibility",
    "backend-constraints.yaml": "backend-constraints",
    "validation-rules.yaml": "validation-rules",
    "known-failure-signatures.yaml": "known-failure-signatures",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAFE_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
ENTRY_STATUSES = frozenset({"active", "deprecated", "experimental"})


class KnowledgeError(ValueError):
    """Raised when a knowledge file violates the v1 contract."""


def load_knowledge_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(
            f"{path} must use JSON-compatible YAML so stdlib validation works: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise KnowledgeError(f"{path}: document root must be an object")
    return payload


def validate_knowledge_document(
    payload: Mapping[str, Any], *, expected_kind: str, path: str
) -> None:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != expected_kind:
        errors.append(f"kind must be {expected_kind!r}")
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not DATE_RE.fullmatch(updated_at):
        errors.append("updated_at must use YYYY-MM-DD")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("entries must be an array")
        entries = []

    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not SAFE_ENTRY_ID_RE.fullmatch(entry_id):
            errors.append(f"{prefix}.id must be a lowercase safe identifier")
        elif entry_id in seen_ids:
            errors.append(f"{prefix}.id is duplicated: {entry_id}")
        else:
            seen_ids.add(entry_id)
        for field in ("source", "applicable_versions", "updated_at"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        entry_date = entry.get("updated_at")
        if isinstance(entry_date, str) and not DATE_RE.fullmatch(entry_date):
            errors.append(f"{prefix}.updated_at must use YYYY-MM-DD")
        if entry.get("status") not in ENTRY_STATUSES:
            errors.append(
                f"{prefix}.status must be one of: {', '.join(sorted(ENTRY_STATUSES))}"
            )
        if "rule" not in entry or not isinstance(entry["rule"], Mapping):
            errors.append(f"{prefix}.rule must be an object")

    allowed = {"schema_version", "kind", "updated_at", "entries"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(unknown)}")
    if errors:
        raise KnowledgeError(f"{path}: " + "; ".join(errors))


def validate_knowledge_dir(root: Path) -> list[str]:
    validated: list[str] = []
    errors: list[str] = []
    for filename, kind in KNOWLEDGE_FILES.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing knowledge file: {path}")
            continue
        try:
            payload = load_knowledge_file(path)
            validate_knowledge_document(payload, expected_kind=kind, path=str(path))
        except KnowledgeError as exc:
            errors.append(str(exc))
        else:
            validated.append(filename)
    unknown = sorted(
        path.name for path in root.glob("*.yaml") if path.name not in KNOWLEDGE_FILES
    )
    if unknown:
        errors.append(f"unknown knowledge files: {', '.join(unknown)}")
    if errors:
        raise KnowledgeError("\n".join(errors))
    return validated
