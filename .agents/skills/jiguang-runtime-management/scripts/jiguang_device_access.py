#!/usr/bin/env python3
"""Configure password-plus-key SSH access for one Jiguang device container."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
MACHINE_SCRIPTS = ROOT / ".agents" / "skills" / "machine-management" / "scripts"
for path in (LIB, MACHINE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _workflow_common as workflow  # noqa: E402
import manage_machine as machine_ops  # noqa: E402


CONFIGURE_SCRIPT = r'''set -euo pipefail
container="$1"
port="$2"
docker inspect "$container" >/dev/null
docker exec -i "$container" bash -s -- "$port" <<'INNER'
set -euo pipefail
port="$1"
cfg=/etc/ssh/sshd_vaws_config
test -f "$cfg"
backup="${cfg}.bak.$(date +%F-%H%M%S)"
cp -p "$cfg" "$backup"
python3 - "$cfg" "$port" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
updates = {
    "port": f"Port {int(sys.argv[2])}",
    "permitrootlogin": "PermitRootLogin yes",
    "passwordauthentication": "PasswordAuthentication yes",
    "pubkeyauthentication": "PubkeyAuthentication yes",
    "usepam": "UsePAM no",
}
seen = set()
output = []
for line in path.read_text().splitlines():
    stripped = line.lstrip()
    key = stripped.split(None, 1)[0].lower() if stripped and not stripped.startswith("#") else ""
    if key in updates:
        if key not in seen:
            output.append(updates[key])
            seen.add(key)
        continue
    output.append(line)
for key, value in updates.items():
    if key not in seen:
        output.append(value)
path.write_text("\n".join(output) + "\n")
PY
ssh-keygen -A >/dev/null 2>&1 || true
mkdir -p /run/sshd
/usr/sbin/sshd -t -f "$cfg"
printf '%s\n' "$backup"
INNER
python3 - "$container" "$port" <<'PY'
import json, sys
print("__VAWS_JSON__=" + json.dumps({
    "success": True,
    "status": "configured",
    "container_name": sys.argv[1],
    "container_ssh_port": int(sys.argv[2]),
    "config_path": "/etc/ssh/sshd_vaws_config",
    "password_authentication": True,
    "pubkey_authentication": True,
}))
PY
'''


RESTART_SCRIPT = r'''set -euo pipefail
container="$1"
port="$2"
docker exec -i "$container" bash -s -- "$port" <<'INNER'
set -euo pipefail
port="$1"
cfg=/etc/ssh/sshd_vaws_config
/usr/sbin/sshd -t -f "$cfg"
if [ -f /run/sshd_vaws.pid ]; then
  pid=$(cat /run/sshd_vaws.pid 2>/dev/null || true)
  if [ -n "$pid" ] && [ "$(cat /proc/$pid/comm 2>/dev/null || true)" = "sshd" ]; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f /run/sshd_vaws.pid
fi
pkill -f '/etc/ssh/sshd_vaws_config' 2>/dev/null || true
/usr/sbin/sshd -f "$cfg"
for _ in 1 2 3 4 5; do
  if command -v ss >/dev/null 2>&1 && ss -ltnH | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
    exit 0
  fi
  sleep 1
done
exit 1
INNER
python3 - "$container" "$port" <<'PY'
import json, sys
print("__VAWS_JSON__=" + json.dumps({
    "success": True,
    "status": "restarted",
    "container_name": sys.argv[1],
    "container_ssh_port": int(sys.argv[2]),
}))
PY
'''


def emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("outcome") in {"planned", "success", "ready"} else 2


def target_for(record: dict[str, Any]) -> machine_ops.SshTarget:
    return machine_ops.SshTarget(
        host=record["host"]["ip"],
        user=record["host"].get("user", "root"),
        port=int(record["host"].get("port", 22)),
    )


def run_host_script(
    target: machine_ops.SshTarget,
    source: str,
    args: Sequence[str],
) -> dict[str, Any]:
    result = machine_ops.run_remote_script(
        target,
        source,
        args=args,
        batch_mode=True,
        timeout_seconds=90,
    )
    return machine_ops.assert_remote_success(result)


def read_password_from_stdin() -> str:
    value = sys.stdin.readline().rstrip("\r\n")
    if not 16 <= len(value) <= 256:
        raise ValueError("device password must contain 16-256 characters")
    if "\x00" in value or "\n" in value or "\r" in value or ":" in value:
        raise ValueError("device password contains an unsupported character")
    return value


def set_root_password(
    target: machine_ops.SshTarget,
    container_name: str,
    password: str,
) -> None:
    identity_file = machine_ops.private_key_for_public_key(machine_ops.find_public_key(None))
    remote_command = machine_ops.remote_shell_command(
        ["docker", "exec", "-i", container_name, "chpasswd"]
    )
    command = machine_ops.ssh_command(
        target,
        batch_mode=True,
        identity_file=identity_file,
    ) + [remote_command]
    completed = subprocess.run(
        command,
        input=f"root:{password}\n",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"container chpasswd failed with exit code {completed.returncode}")


def plan(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": "planned",
        "action": "configure_jiguang_device_access",
        "machine": record["alias"],
        "container_name": record["container"]["name"],
        "container_ssh_port": int(record["container"]["ssh_port"]),
        "config_path": "/etc/ssh/sshd_vaws_config",
        "preserve_container_port": True,
        "password_source": "stdin",
        "password_authentication": True,
        "pubkey_authentication": True,
        "requires_confirm": True,
    }


def apply(record: dict[str, Any], password: str) -> dict[str, Any]:
    target = target_for(record)
    blocker = workflow.ssh_client_preflight_blocker(target)
    if blocker is not None:
        return {"outcome": "blocked", "preflight": blocker}
    container_name = record["container"]["name"]
    port = int(record["container"]["ssh_port"])
    configured = run_host_script(target, CONFIGURE_SCRIPT, [container_name, str(port)])
    set_root_password(target, container_name, password)
    restarted = run_host_script(target, RESTART_SCRIPT, [container_name, str(port)])
    verified = workflow.verify_machine(record)
    if not verified.get("ready"):
        return {
            "outcome": "blocked",
            "error": "container access changed but key-based readiness verification failed",
            "configured": configured,
            "restarted": restarted,
            "verify": verified,
        }
    return {
        "outcome": "success",
        "changed": True,
        "machine": record["alias"],
        "container_name": container_name,
        "container_ssh_port": port,
        "password_authentication": True,
        "pubkey_authentication": True,
        "configured": configured,
        "restarted": restarted,
        "verify": {"ready": True, "status": verified.get("status")},
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--machine", required=True)
    root.add_argument("--password-stdin", action="store_true")
    root.add_argument("--confirm", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        record = workflow.find_record(args.machine)
        if record is None:
            return emit({"outcome": "blocked", "error": "machine is not managed by this workspace"})
        if not args.confirm:
            return emit(plan(record))
        if not args.password_stdin:
            return emit({"outcome": "blocked", "error": "confirmed access configuration requires --password-stdin"})
        password = read_password_from_stdin()
        return emit(apply(record, password))
    except (ValueError, RuntimeError, workflow.WorkflowError, machine_ops.MachineManagementError) as exc:
        return emit({"outcome": "blocked", "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
