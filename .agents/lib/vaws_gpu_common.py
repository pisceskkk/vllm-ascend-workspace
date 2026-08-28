#!/usr/bin/env python3
"""Shared transport and configuration helpers for gpu_* workspace tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vaws_ssh_control import ssh_command_prefix
from vaws_ssh_preflight import ssh_client_preflight

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GPU_STATE_ROOT = ROOT / ".vaws-local" / "gpu-workspaces"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class GpuToolError(RuntimeError):
    """Raised for a user-actionable GPU tool failure."""


@dataclass(frozen=True)
class GpuTarget:
    host: str
    user: str = "root"
    port: int = 22
    container: str = ""
    identity_file: Path | None = None
    ssh_config: Path | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "container": self.container,
            "identity_file": str(self.identity_file) if self.identity_file else None,
            "ssh_config": str(self.ssh_config) if self.ssh_config else None,
        }


def progress(tool: str, phase: str, **fields: object) -> None:
    payload = {"tool": tool, "phase": phase, **fields}
    print(
        f"__VAWS_GPU_PROGRESS__={json.dumps(payload, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def validate_safe_name(value: str, field: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise GpuToolError(f"{field} must match {SAFE_NAME.pattern}")
    return value


def add_target_arguments(
    parser: argparse.ArgumentParser, *, require_container: bool = True
) -> None:
    parser.add_argument("--workspace-config", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--port", type=int)
    parser.add_argument("--container")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument(
        "--ssh-config", type=Path, help="explicit OpenSSH config, e.g. /dev/null"
    )
    parser.set_defaults(_gpu_require_container=require_container)


def load_workspace_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuToolError(f"invalid GPU workspace config {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise GpuToolError(f"unsupported GPU workspace config: {resolved}")
    return payload


def target_from_args(args: argparse.Namespace) -> GpuTarget:
    config = load_workspace_config(getattr(args, "workspace_config", None))
    host = args.host or config.get("host")
    if not host or any(character.isspace() for character in str(host)):
        raise GpuToolError("--host or --workspace-config with a valid host is required")
    container = args.container or config.get("container") or ""
    if getattr(args, "_gpu_require_container", True) and not container:
        raise GpuToolError(
            "--container or --workspace-config with a container is required"
        )
    if container:
        validate_safe_name(str(container), "container")
    identity = args.identity_file or config.get("identity_file")
    ssh_config = args.ssh_config or config.get("ssh_config")
    port = int(args.port or config.get("port") or 22)
    if port < 1 or port > 65535:
        raise GpuToolError("SSH port must be between 1 and 65535")
    return GpuTarget(
        host=str(host),
        user=str(args.user or config.get("user") or "root"),
        port=port,
        container=str(container),
        identity_file=Path(identity).expanduser().resolve() if identity else None,
        ssh_config=Path(ssh_config).expanduser().resolve() if ssh_config else None,
    )


def ssh_base(target: GpuTarget) -> list[str]:
    command = [
        *ssh_command_prefix(),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "LogLevel=ERROR",
    ]
    if target.ssh_config:
        command.extend(["-F", str(target.ssh_config)])
    if target.identity_file:
        command.extend(["-i", str(target.identity_file), "-o", "IdentitiesOnly=yes"])
    command.extend(["-p", str(target.port), f"{target.user}@{target.host}"])
    return command


def ssh_config_check(target: GpuTarget) -> None:
    result = ssh_client_preflight(
        target.host,
        port=target.port,
        user=target.user,
        ssh_config=target.ssh_config,
        timeout=20,
    )
    if result.get("status") != "ready":
        category = result.get("category", "ssh_config_invalid")
        message = result.get("message", "SSH client configuration preflight failed")
        raise GpuToolError(f"SSH preflight failed ({category}): {message}")


def remote_bash(
    target: GpuTarget,
    script: str,
    *,
    timeout: int = 300,
) -> subprocess.CompletedProcess[bytes]:
    ssh_config_check(target)
    process = subprocess.run(
        ssh_base(target) + ["bash", "-s"],
        input=script.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return process


def remote_command(
    target: GpuTarget,
    command: list[str],
    *,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    ssh_config_check(target)
    remote = "bash -lc " + shlex.quote(shlex.join(command))
    return subprocess.run(
        ssh_base(target) + [remote],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def remote_upload_bytes(
    target: GpuTarget,
    remote_path: str,
    input_bytes: bytes,
    *,
    timeout: int = 600,
) -> str:
    """Upload bytes with a dedicated stream, then hash and atomically rename."""

    if not remote_path.startswith("/") or any(
        character.isspace() for character in remote_path
    ):
        raise GpuToolError("remote upload path must be absolute and contain no whitespace")
    expected = hashlib.sha256(input_bytes).hexdigest()
    temporary = remote_path + ".part"
    prepare = remote_bash(
        target,
        "\n".join(
            [
                "set -eu",
                f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}",
                f"rm -f -- {shlex.quote(temporary)}",
            ]
        ),
        timeout=30,
    )
    require_success(prepare, "GPU upload preparation")
    ssh_config_check(target)
    transfer = subprocess.run(
        ssh_base(target) + [f"dd of={shlex.quote(temporary)} bs=4M status=none"],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    require_success(transfer, "GPU binary upload")
    finalize = remote_bash(
        target,
        "\n".join(
            [
                "set -eu",
                f"vaws_observed=$(sha256sum {shlex.quote(temporary)} | cut -d ' ' -f1)",
                f"test \"$vaws_observed\" = {expected}",
                f"mv -f -- {shlex.quote(temporary)} {shlex.quote(remote_path)}",
                'printf \'%s\\n\' "$vaws_observed"',
            ]
        ),
        timeout=timeout,
    )
    require_success(finalize, "GPU upload finalize")
    observed = finalize.stdout.decode(errors="replace").strip().splitlines()[-1]
    if observed != expected:
        raise GpuToolError(
            f"remote upload checksum mismatch: expected {expected}, got {observed}"
        )
    return observed


def docker_exec_command(
    target: GpuTarget,
    command: list[str],
    *,
    workdir: str = "/workspace/vllm",
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return remote_command(
        target,
        ["docker", "exec", "-w", workdir, target.container, *command],
        timeout=timeout,
    )


def require_success(process: subprocess.CompletedProcess[Any], context: str) -> None:
    if process.returncode == 0:
        return
    stderr = (
        process.stderr.decode(errors="replace")
        if isinstance(process.stderr, bytes)
        else process.stderr
    )
    raise GpuToolError(
        f"{context} failed ({process.returncode}): {(stderr or '').strip()[-2000:]}"
    )


def shell_assignment(name: str, value: object) -> str:
    return f"{name}={shlex.quote(str(value))}"
