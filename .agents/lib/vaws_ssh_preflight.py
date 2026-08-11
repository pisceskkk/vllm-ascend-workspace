#!/usr/bin/env python3
"""Read-only local OpenSSH client configuration preflight."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable


KNOWLEDGE_ID = "machine-management-openssh-system-config-ownership"
DEFAULT_TIMEOUT_SECONDS = 10.0
BAD_OWNER_PATTERN = re.compile(
    r"bad owner or permissions on (?P<path>[^\r\n]+)",
    re.IGNORECASE,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _safe_tail(value: str | None, limit: int = 1200) -> str:
    return (value or "")[-limit:]


def _path_observation(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path)}
    try:
        link_stat = path.lstat()
    except OSError as exc:
        payload.update({"exists": False, "error": str(exc)})
        return payload

    payload.update(
        {
            "exists": True,
            "is_symlink": stat.S_ISLNK(link_stat.st_mode),
            "link_uid": link_stat.st_uid,
            "link_gid": link_stat.st_gid,
            "link_mode": f"{stat.S_IMODE(link_stat.st_mode):04o}",
        }
    )
    if payload["is_symlink"]:
        try:
            payload["link_target"] = os.readlink(path)
        except OSError as exc:
            payload["link_target_error"] = str(exc)

    try:
        effective = path.stat()
    except OSError as exc:
        payload["target_error"] = str(exc)
        return payload

    mode = stat.S_IMODE(effective.st_mode)
    payload.update(
        {
            "resolved_path": str(path.resolve(strict=False)),
            "uid": effective.st_uid,
            "gid": effective.st_gid,
            "mode": f"{mode:04o}",
            "group_or_other_writable": bool(mode & 0o022),
            "root_owned": effective.st_uid == 0,
        }
    )
    return payload


def inspect_reported_path(value: str) -> list[dict[str, Any]]:
    """Inspect a reported absolute path, its symlink target, and SSH ancestors."""

    cleaned = value.strip().strip("'\"")
    reported = Path(cleaned)
    if not reported.is_absolute():
        return []

    paths: list[Path] = []
    ssh_root = Path("/etc/ssh")
    try:
        relative = reported.relative_to(ssh_root)
    except ValueError:
        relative = None

    if relative is not None:
        current = ssh_root
        paths.append(current)
        for part in relative.parts[:-1]:
            current = current / part
            paths.append(current)
    paths.append(reported)

    try:
        resolved = reported.resolve(strict=False)
    except OSError:
        resolved = reported
    if resolved != reported:
        paths.append(resolved)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        token = str(path)
        if token not in seen:
            seen.add(token)
            unique.append(path)
    return [_path_observation(path) for path in unique]


def ssh_client_preflight(
    host: str,
    *,
    port: int = 22,
    user: str = "root",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Parse effective SSH configuration without opening a network connection."""

    destination = f"{user}@{host}"
    ssh_path = shutil.which("ssh")
    base: dict[str, Any] = {
        "target": {"host": host, "port": port, "user": user, "destination": destination},
        "check": "ssh-config",
        "command": ["ssh", "-G", "-p", str(port), destination],
        "mutated": False,
    }
    if not ssh_path:
        return {
            **base,
            "status": "blocked",
            "category": "ssh_client_missing",
            "message": "local OpenSSH client was not found",
        }

    execute = runner or subprocess.run
    try:
        completed = execute(
            base["command"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {
            **base,
            "status": "blocked",
            "category": "ssh_client_missing",
            "message": "local OpenSSH client was not found",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "blocked",
            "category": "ssh_config_timeout",
            "message": f"ssh configuration preflight timed out after {timeout:g}s",
            "stderr_tail": _safe_tail(exc.stderr if isinstance(exc.stderr, str) else None),
        }

    stderr = completed.stderr or ""
    if completed.returncode == 0:
        return {
            **base,
            "status": "ready",
            "category": "ssh_config_valid",
            "message": "local OpenSSH client configuration parsed successfully",
        }

    owner_match = BAD_OWNER_PATTERN.search(stderr)
    if owner_match:
        offending_path = owner_match.group("path").strip().strip("'\"")
        return {
            **base,
            "status": "blocked",
            "category": "local_ssh_config_permissions",
            "knowledge_id": KNOWLEDGE_ID,
            "message": "OpenSSH rejected a system client configuration path before connecting",
            "offending_path": offending_path,
            "observations": inspect_reported_path(offending_path),
            "repair_required": True,
            "auto_repaired": False,
            "verification_command": base["command"],
            "stderr_tail": _safe_tail(stderr),
        }

    return {
        **base,
        "status": "blocked",
        "category": "ssh_config_invalid",
        "message": "local OpenSSH client configuration could not be parsed",
        "stderr_tail": _safe_tail(stderr),
    }


def blocked_status_payload(result: dict[str, Any], *, action: str) -> dict[str, Any]:
    """Convert a failed preflight into the shared public workflow status shape."""

    payload = {
        "success": False,
        "status": "blocked",
        "action": action,
        "message": result.get("message", "local SSH client preflight failed"),
        "ssh_preflight": result,
    }
    if result.get("knowledge_id"):
        payload["knowledge_id"] = result["knowledge_id"]
    return payload
