#!/usr/bin/env python3
"""Classify GitHub CLI auth without confusing network failures with bad credentials."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any, Callable


KNOWLEDGE_ID = "github-cli-network-enabled-auth-validation"
NETWORK_CONTEXTS = ("unknown", "restricted", "enabled")
DEFAULT_TIMEOUT_SECONDS = 15.0

TRANSPORT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"could not resolve host",
        r"failed to connect",
        r"connection (?:timed out|refused)",
        r"network is unreachable",
        r"tls handshake timeout",
        r"i/o timeout",
        r"temporary failure in name resolution",
    )
)
AUTH_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"token[^\r\n]*invalid",
        r"invalid token",
        r"bad credentials",
        r"http[^\r\n]*401",
        r"requires authentication",
        r"authentication failed",
    )
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(
    command: list[str],
    *,
    timeout: float,
    runner: Runner,
) -> dict[str, Any]:
    try:
        completed = runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }
    except FileNotFoundError:
        return {"returncode": None, "stdout": "", "stderr": "", "not_found": True}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "timed_out": False,
    }


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _public_check(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in ("returncode", "timed_out", "not_found")
        if result.get(key) is not None
    }


def classify_github_auth(
    *,
    network_context: str = "unknown",
    gh_path: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Return two-axis GitHub auth/network state without exposing token text."""

    if network_context not in NETWORK_CONTEXTS:
        raise ValueError(f"unsupported network context: {network_context}")

    resolved_gh = gh_path or shutil.which("gh")
    if not resolved_gh:
        return {
            "installed": False,
            "auth_state": "not_installed",
            "network_state": "unknown",
            "network_context": network_context,
            "logged_in": False,
            "diagnostic_code": "gh_not_installed",
            "retry_required": "none",
        }

    execute = runner or subprocess.run
    auth = _run(
        [resolved_gh, "auth", "status", "--hostname", "github.com"],
        timeout=timeout,
        runner=execute,
    )
    api = _run(
        [resolved_gh, "api", "user", "--jq", ".login"],
        timeout=timeout,
        runner=execute,
    )
    login = api["stdout"].strip() if api.get("returncode") == 0 else ""
    combined_errors = "\n".join((auth.get("stderr", ""), api.get("stderr", "")))
    timed_out = bool(auth.get("timed_out") or api.get("timed_out"))
    transport_failed = timed_out or _matches(TRANSPORT_PATTERNS, combined_errors)
    auth_rejected = _matches(AUTH_PATTERNS, combined_errors)

    result: dict[str, Any] = {
        "installed": True,
        "path": resolved_gh,
        "network_context": network_context,
        "checks": {
            "auth_status": _public_check(auth),
            "api_user": _public_check(api),
        },
        "knowledge_id": KNOWLEDGE_ID,
    }
    if api.get("returncode") == 0 and login:
        result.update(
            {
                "auth_state": "authenticated",
                "network_state": "reachable",
                "logged_in": True,
                "diagnostic_code": "api_user_ok",
                "retry_required": "none",
                "user_login": login,
            }
        )
        return result

    if transport_failed:
        result.update(
            {
                "auth_state": "unverified",
                "network_state": "unavailable",
                "logged_in": None,
                "diagnostic_code": "transport_error",
                "retry_required": "network_enabled",
            }
        )
        return result

    if network_context == "restricted":
        result.update(
            {
                "auth_state": "unverified",
                "network_state": "unknown",
                "logged_in": None,
                "diagnostic_code": "restricted_context_ambiguous",
                "retry_required": "network_enabled",
            }
        )
        return result

    if network_context == "enabled" and auth_rejected:
        result.update(
            {
                "auth_state": "auth_failed",
                "network_state": "reachable",
                "logged_in": False,
                "diagnostic_code": "auth_rejected",
                "retry_required": "none",
            }
        )
        return result

    result.update(
        {
            "auth_state": "unverified",
            "network_state": "unknown",
            "logged_in": None,
            "diagnostic_code": "ambiguous",
            "retry_required": "network_enabled",
        }
    )
    return result
