#!/usr/bin/env python3
"""Strict vLLM/vllm-ascend source pairing checks.

The vllm-ascend commit is the owner of the default compatibility contract.
Unless a caller supplies an explicit vLLM commit, the only accepted source is
``.github/vllm-main-verified.commit`` from the checked-out vllm-ascend HEAD.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


PIN_PATH = ".github/vllm-main-verified.commit"
FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
COMMIT_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


class VllmVersionPairingError(RuntimeError):
    """Raised when the source-version pairing contract cannot be resolved."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise VllmVersionPairingError(
            f"git {' '.join(args)} failed in {repo}: {detail or 'unknown git error'}"
        )
    return completed.stdout.strip()


def _repo_head(repo: Path, label: str) -> str:
    if not repo.is_dir():
        raise VllmVersionPairingError(f"{label} repository does not exist: {repo}")
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    if not FULL_COMMIT_RE.fullmatch(head):
        raise VllmVersionPairingError(f"{label} HEAD is not a full commit SHA: {head!r}")
    return head.lower()


def _resolve_explicit_commit(vllm_repo: Path, value: str) -> str:
    candidate = value.strip()
    if not COMMIT_PREFIX_RE.fullmatch(candidate):
        raise VllmVersionPairingError(
            "an explicit vLLM override must be a 7-40 character hexadecimal commit SHA"
        )
    if FULL_COMMIT_RE.fullmatch(candidate):
        return candidate.lower()
    resolved = _git(
        vllm_repo,
        "rev-parse",
        "--verify",
        f"{candidate}^{{commit}}",
        check=False,
    )
    if not FULL_COMMIT_RE.fullmatch(resolved):
        raise VllmVersionPairingError(
            f"explicit vLLM commit {candidate!r} is not available in {vllm_repo}"
        )
    return resolved.lower()


def _resolve_verified_pin(vllm_ascend_repo: Path) -> str:
    raw = _git(
        vllm_ascend_repo,
        "show",
        f"HEAD:{PIN_PATH}",
        check=False,
    ).strip()
    if not raw:
        raise VllmVersionPairingError(
            f"vllm-ascend HEAD does not publish required compatibility pin {PIN_PATH}"
        )
    if not FULL_COMMIT_RE.fullmatch(raw):
        raise VllmVersionPairingError(
            f"{PIN_PATH} at vllm-ascend HEAD must contain exactly one full 40-character commit SHA"
        )
    return raw.lower()


def resolve_vllm_pairing(
    *,
    vllm_repo: Path,
    vllm_ascend_repo: Path,
    explicit_vllm_commit: str | None = None,
) -> dict[str, Any]:
    """Resolve the exact vLLM commit required by the current vllm-ascend HEAD."""

    vllm_repo = vllm_repo.resolve()
    vllm_ascend_repo = vllm_ascend_repo.resolve()
    vllm_ascend_commit = _repo_head(vllm_ascend_repo, "vllm-ascend")
    branch = _git(vllm_ascend_repo, "branch", "--show-current", check=False) or None

    if explicit_vllm_commit is not None:
        required = _resolve_explicit_commit(vllm_repo, explicit_vllm_commit)
        source = "explicit-vllm-commit"
        source_path = None
    else:
        required = _resolve_verified_pin(vllm_ascend_repo)
        source = "vllm-ascend-head-verified-pin"
        source_path = str(vllm_ascend_repo / PIN_PATH)

    return {
        "status": "ok",
        "vllm_ref": required,
        "required_vllm_commit": required,
        "ref_type": "commit",
        "source": (
            source
            if source_path is None
            else f"{vllm_ascend_commit}:{PIN_PATH}"
        ),
        "source_path": source_path,
        "precedence": source,
        "explicit_override": explicit_vllm_commit is not None,
        "vllm_ascend_commit": vllm_ascend_commit,
        "vllm_ascend_branch": branch,
    }


def check_workspace_vllm_pairing(
    workspace_root: Path,
    *,
    explicit_vllm_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the workspace vLLM HEAD matches the resolved contract."""

    workspace_root = workspace_root.resolve()
    vllm_repo = workspace_root / "vllm"
    vllm_ascend_repo = workspace_root / "vllm-ascend"
    try:
        resolved = resolve_vllm_pairing(
            vllm_repo=vllm_repo,
            vllm_ascend_repo=vllm_ascend_repo,
            explicit_vllm_commit=explicit_vllm_commit,
        )
        actual = _repo_head(vllm_repo, "vllm")
    except VllmVersionPairingError as exc:
        return {
            "status": "blocked",
            "reason": str(exc),
            "workspace_root": str(workspace_root),
            "explicit_override": explicit_vllm_commit is not None,
        }

    payload = {
        **resolved,
        "workspace_root": str(workspace_root),
        "actual_vllm_commit": actual,
    }
    if actual != resolved["required_vllm_commit"]:
        payload.update(
            {
                "status": "blocked",
                "reason": (
                    "vLLM/vllm-ascend version pairing mismatch: "
                    f"vllm HEAD is {actual}, required is {resolved['required_vllm_commit']}"
                ),
                "next_action": (
                    "check out the required vLLM commit, or rerun the owning workflow with "
                    "an explicit --vllm-commit override"
                ),
            }
        )
        return payload

    payload["status"] = "ready"
    return payload
