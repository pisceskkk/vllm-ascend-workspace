#!/usr/bin/env python3
"""Deterministic local state helpers for Jiguang evaluation runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from vllm_version_pairing import check_workspace_vllm_pairing

NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".s", ".S"}
NATIVE_NAMES = {"CMakeLists.txt", "setup.py", "pyproject.toml"}
NATIVE_PREFIXES = ("csrc/", "cmake/", "kernels/", "ops/")


class JiguangRuntimeError(RuntimeError):
    """Raised for invalid local runtime state or workspace evidence."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise JiguangRuntimeError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return completed.stdout.strip()


def _repository_snapshot(repo: Path) -> dict[str, Any]:
    if not repo.is_dir():
        raise JiguangRuntimeError(f"repository does not exist: {repo}")
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False)
    pushed = False
    if upstream:
        pushed = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, upstream],
            capture_output=True,
            check=False,
        ).returncode == 0
    return {
        "path": str(repo),
        "commit_sha": commit,
        "clean": not bool(status),
        "status": status.splitlines(),
        "upstream": upstream or None,
        "pushed": pushed,
    }


def workspace_gate(
    repo_root: Path,
    *,
    explicit_vllm_commit: str | None = None,
) -> dict[str, Any]:
    repositories = {
        "workspace": _repository_snapshot(repo_root),
        "vllm": _repository_snapshot(repo_root / "vllm"),
        "vllm_ascend": _repository_snapshot(repo_root / "vllm-ascend"),
    }
    blockers: list[str] = []
    for name, snapshot in repositories.items():
        if not snapshot["clean"]:
            blockers.append(f"{name} has uncommitted or untracked changes")
        if not snapshot["upstream"]:
            blockers.append(f"{name} has no configured upstream branch")
        elif not snapshot["pushed"]:
            blockers.append(f"{name} HEAD is not contained in its upstream branch")
    pairing = check_workspace_vllm_pairing(
        repo_root,
        explicit_vllm_commit=explicit_vllm_commit,
    )
    if pairing.get("status") != "ready":
        blockers.append(pairing.get("reason", "vLLM/vllm-ascend pairing could not be proven"))
    return {
        "outcome": "ready" if not blockers else "blocked",
        "explicit_opt_in_required": True,
        "repositories": repositories,
        "vllm_version_pairing": pairing,
        "blockers": blockers,
    }


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native_index(repo: Path) -> list[str]:
    lines = _git(repo, "ls-files", "-s").splitlines()
    selected: list[str] = []
    for line in lines:
        _, _, path = line.partition("\t")
        candidate = Path(path)
        if (
            candidate.name in NATIVE_NAMES
            or candidate.suffix in NATIVE_SUFFIXES
            or path.startswith(NATIVE_PREFIXES)
        ):
            selected.append(line)
    return sorted(selected)


def native_code_hash(repo_root: Path) -> str:
    return canonical_hash(
        {
            "vllm": _native_index(repo_root / "vllm"),
            "vllm_ascend": _native_index(repo_root / "vllm-ascend"),
        }
    )


def runtime_hash(image_digest: str, components: dict[str, Any]) -> str:
    normalized = image_digest.strip()
    moving = {"auto", "rc", "main", "stable", "latest"}
    immutable = "@sha256:" in normalized or (":" in normalized and not normalized.endswith(":latest"))
    if not normalized or normalized.lower() in moving or not immutable:
        raise JiguangRuntimeError("a concrete image digest or immutable tag is required")
    return canonical_hash({"image_digest": normalized, "components": components})


def load_runtime_records(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "machines": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JiguangRuntimeError(f"cannot read runtime state {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("machines"), dict):
        raise JiguangRuntimeError("unsupported Jiguang runtime state")
    return payload


def write_runtime_records(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def plan_runtime(
    *,
    machine: str,
    image_digest: str,
    components: dict[str, Any],
    repo_root: Path,
    state_path: Path,
    force_clean: bool = False,
) -> dict[str, Any]:
    current_native = native_code_hash(repo_root)
    current_runtime = runtime_hash(image_digest, components)
    record = load_runtime_records(state_path)["machines"].get(machine)
    if record is None:
        decision, reason = "create", "first Jiguang evaluation on this machine"
    elif force_clean:
        decision, reason = "replace", "clean environment explicitly requested"
    elif record.get("runtime_hash") != current_runtime:
        decision, reason = "replace", "base image or runtime components changed"
    elif record.get("native_code_hash") != current_native:
        decision, reason = "replace", "native code or build metadata changed"
    elif record.get("health") != "ready":
        decision, reason = "replace", "recorded runtime is not healthy"
    else:
        decision, reason = "reuse", "runtime and native hashes match"
    return {
        "outcome": "planned",
        "machine": machine,
        "decision": decision,
        "reason": reason,
        "container_name": "vaws-jiguang" if decision == "reuse" else "vaws-jiguang-next",
        "runtime_hash": current_runtime,
        "native_code_hash": current_native,
        "current": record,
    }


def record_runtime(state_path: Path, machine: str, record: dict[str, Any]) -> dict[str, Any]:
    required = {"container_name", "generation", "image_digest", "runtime_hash", "native_code_hash", "health"}
    missing = sorted(required - set(record))
    if missing:
        raise JiguangRuntimeError(f"runtime record missing fields: {', '.join(missing)}")
    payload = load_runtime_records(state_path)
    payload["machines"][machine] = dict(record)
    write_runtime_records(state_path, payload)
    return payload["machines"][machine]


def command_preview(parts: Iterable[str]) -> str:
    """Return a human-readable command without executing an SSH-backed workflow."""
    import shlex

    return shlex.join(list(parts))
