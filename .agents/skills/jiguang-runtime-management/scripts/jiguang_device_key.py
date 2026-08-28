#!/usr/bin/env python3
"""Bind one local SSH identity to a workspace container and Jiguang device."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[4]
MACHINE_SCRIPTS = ROOT / ".agents" / "skills" / "machine-management" / "scripts"
if str(MACHINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MACHINE_SCRIPTS))

import _workflow_common as workflow  # noqa: E402
import manage_machine as machine_ops  # noqa: E402


CREDENTIAL_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_.-]{2,127}$")
WINDOWS_POWERSHELL = Path("/mnt/c/windows/System32/WindowsPowerShell/v1.0/powershell.exe")
CREDENTIAL_SCRIPT = ROOT / ".jiguang" / "host" / "set_jiguang_credential.ps1"


class KeyMaterial(NamedTuple):
    private_key: Path
    public_key: Path
    fingerprint: str


def emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("outcome") in {"planned", "ready", "success"} else 2


def run_checked(command: list[str], *, label: str) -> str:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed")
    return completed.stdout.strip()


def resolve_key_material(private_key_file: str | None = None) -> KeyMaterial:
    if private_key_file:
        private_key = Path(private_key_file).expanduser().resolve()
    else:
        public_default = machine_ops.find_public_key(None)
        private_default = machine_ops.private_key_for_public_key(public_default)
        if private_default is None:
            raise RuntimeError("no local private key matched the workspace public key")
        private_key = private_default
    if not private_key.is_file():
        raise RuntimeError("local private key file does not exist")
    public_key = Path(str(private_key) + ".pub")
    if not public_key.is_file():
        raise RuntimeError("local public key file does not exist beside the private key")

    derived = run_checked(["ssh-keygen", "-y", "-f", str(private_key)], label="private-key derivation")
    recorded = machine_ops.load_public_key(public_key)
    if derived.split()[:2] != recorded.split()[:2]:
        raise RuntimeError("local public key does not match the selected private key")
    fingerprint_output = run_checked(
        ["ssh-keygen", "-lf", str(public_key), "-E", "sha256"],
        label="public-key fingerprint",
    )
    fields = fingerprint_output.split()
    if len(fields) < 2 or not fields[1].startswith("SHA256:"):
        raise RuntimeError("ssh-keygen returned an invalid fingerprint")
    return KeyMaterial(private_key=private_key, public_key=public_key, fingerprint=fields[1])


def windows_path(path: Path) -> str:
    return run_checked(["wslpath", "-w", str(path.resolve())], label="Windows path conversion")


def store_private_key_reference(target: str, material: KeyMaterial) -> None:
    if not CREDENTIAL_TARGET_RE.fullmatch(target):
        raise ValueError("invalid Windows credential target")
    command = [
        str(WINDOWS_POWERSHELL),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        windows_path(CREDENTIAL_SCRIPT),
        "-Target",
        target,
        "-SecretFile",
        windows_path(material.private_key),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Windows Credential Manager private-key reference write failed")


def container_key_check(record: dict[str, Any], material: KeyMaterial) -> dict[str, Any]:
    target = machine_ops.SshTarget(
        host=record["host"]["ip"],
        user="root",
        port=int(record["container"]["ssh_port"]),
    )
    return machine_ops.check_direct_ssh(target, identity_file=material.private_key)


def plan(
    record: dict[str, Any],
    material: KeyMaterial,
    credential_target: str,
) -> dict[str, Any]:
    if not CREDENTIAL_TARGET_RE.fullmatch(credential_target):
        raise ValueError("invalid Windows credential target")
    check = container_key_check(record, material)
    if not check.get("ok"):
        return {
            "outcome": "blocked",
            "error": "selected local key is not accepted by the managed container",
            "machine": record["alias"],
            "container_ssh_port": int(record["container"]["ssh_port"]),
        }
    return {
        "outcome": "planned",
        "action": "bind_jiguang_device_key",
        "machine": record["alias"],
        "container_name": record["container"]["name"],
        "container_ssh_port": int(record["container"]["ssh_port"]),
        "key_fingerprint": material.fingerprint,
        "credential_target": credential_target,
        "container_key_verified": True,
        "registration_auth_type": "SSH_KEY",
        "requires_confirm": True,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--machine", required=True)
    root.add_argument("--credential-target", required=True)
    root.add_argument("--private-key-file", required=True)
    root.add_argument("--confirm", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        record = workflow.find_record(args.machine)
        if record is None:
            return emit({"outcome": "blocked", "error": "machine is not managed by this workspace"})
        target = workflow.host_target(
            host=record["host"]["ip"],
            user=record["host"]["user"],
            port=record["host"]["port"],
        )
        preflight = workflow.ssh_client_preflight_blocker(target)
        if preflight is not None:
            return emit({"outcome": "blocked", "preflight": preflight})
        material = resolve_key_material(args.private_key_file)
        result = plan(record, material, args.credential_target)
        if result.get("outcome") != "planned" or not args.confirm:
            return emit(result)
        store_private_key_reference(args.credential_target, material)
        return emit({**result, "outcome": "success", "credential_stored": True})
    except (ValueError, RuntimeError, workflow.WorkflowError, machine_ops.MachineManagementError) as exc:
        return emit({"outcome": "blocked", "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
