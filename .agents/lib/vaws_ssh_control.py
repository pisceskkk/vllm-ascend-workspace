#!/usr/bin/env python3
"""Resolve the local execution plane used for OpenSSH commands.

Codex filesystem sandboxes may expose host-root-owned files as the overflow
UID (65534).  OpenSSH correctly rejects that view of ``/etc/ssh`` even though
the same files are ``root:root`` in the owning WSL distribution.  This module
allows the workspace to run OpenSSH in that host WSL namespace without
disabling or bypassing OpenSSH's configuration checks.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / ".remote-dev" / "state" / "ssh-control-plane.json"
VALID_MODES = {"auto", "native", "host-wsl"}
ENV_MODE = "VAWS_SSH_CONTROL_PLANE"
ENV_CONFIG = "VAWS_SSH_CONTROL_PLANE_CONFIG"
ENV_WSL_DISTRO = "VAWS_SSH_WSL_DISTRO"
ENV_WSL_USER = "VAWS_SSH_WSL_USER"


class SshControlPlaneError(RuntimeError):
    """Raised when an explicitly selected SSH control plane is invalid."""


@dataclass(frozen=True)
class SshControlPlane:
    mode: str
    command_prefix: tuple[str, ...]
    source: str
    distribution: str | None = None
    user: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "command_prefix": list(self.command_prefix),
            "source": self.source,
            "distribution": self.distribution,
            "user": self.user,
        }


def _config_path(env: Mapping[str, str]) -> Path:
    override = env.get(ENV_CONFIG)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CONFIG_PATH


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SshControlPlaneError(f"invalid SSH control-plane config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SshControlPlaneError(f"SSH control-plane config must be a JSON object: {path}")
    return payload


def _find_wsl_executable() -> str | None:
    discovered = shutil.which("wsl.exe")
    if discovered:
        return discovered
    for candidate in (
        Path("/mnt/c/Windows/System32/wsl.exe"),
        Path("/mnt/c/windows/system32/wsl.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _sandbox_root_is_overflow_uid() -> bool:
    try:
        return Path("/etc/ssh").stat().st_uid == 65534
    except OSError:
        return False


def ssh_host_execution_required(
    *, env: Mapping[str, str] | None = None
) -> bool:
    """Return whether OpenSSH must be invoked outside the Codex sandbox.

    WSL Codex sandboxes can map host UID/GID 0 to the overflow identity 65534.
    OpenSSH then rejects otherwise-valid system configuration ownership.  Do
    not interpret that sandbox-only view as host filesystem damage and do not
    try to repair privileged paths from the sandbox.
    """

    values = os.environ if env is None else env
    return bool(values.get("WSL_DISTRO_NAME")) and _sandbox_root_is_overflow_uid()


def resolve_ssh_control_plane(
    *,
    env: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> SshControlPlane:
    values = os.environ if env is None else env
    path = config_path or _config_path(values)
    config = _load_config(path)

    configured_mode = values.get(ENV_MODE) or config.get("mode") or "auto"
    mode = str(configured_mode).strip().lower()
    if mode not in VALID_MODES:
        raise SshControlPlaneError(
            f"invalid SSH control-plane mode {configured_mode!r}; expected one of {sorted(VALID_MODES)}"
        )

    source = ENV_MODE if values.get(ENV_MODE) else (str(path) if config else "auto-detect")
    if mode == "auto":
        should_delegate = ssh_host_execution_required(env=values)
        mode = "host-wsl" if should_delegate else "native"

    if mode == "native":
        ssh_path = shutil.which("ssh")
        if not ssh_path:
            raise SshControlPlaneError("local OpenSSH client was not found")
        return SshControlPlane(mode="native", command_prefix=("ssh",), source=source)

    wsl_executable = _find_wsl_executable()
    if not wsl_executable:
        raise SshControlPlaneError("host-wsl SSH control plane requires wsl.exe")
    distribution = str(
        values.get(ENV_WSL_DISTRO)
        or config.get("distribution")
        or values.get("WSL_DISTRO_NAME")
        or ""
    ).strip()
    user = str(values.get(ENV_WSL_USER) or config.get("user") or getpass.getuser()).strip()
    if not distribution:
        raise SshControlPlaneError("host-wsl SSH control plane requires a WSL distribution")
    if not user:
        raise SshControlPlaneError("host-wsl SSH control plane requires a WSL user")
    return SshControlPlane(
        mode="host-wsl",
        command_prefix=(wsl_executable, "-d", distribution, "-u", user, "--", "ssh"),
        source=source,
        distribution=distribution,
        user=user,
    )


def ssh_command_prefix() -> list[str]:
    """Return the argv prefix replacing a literal local ``ssh`` command."""

    return list(resolve_ssh_control_plane().command_prefix)
