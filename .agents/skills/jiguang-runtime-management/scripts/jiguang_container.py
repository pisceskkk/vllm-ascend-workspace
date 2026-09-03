#!/usr/bin/env python3
"""Ensure or roll back the persistent per-machine vaws-jiguang container."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
MACHINE_SCRIPTS = ROOT / ".agents" / "skills" / "machine-management" / "scripts"
for path in (LIB, MACHINE_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _workflow_common as workflow  # noqa: E402
import manage_machine as machine_ops  # noqa: E402
from vaws_jiguang import (  # noqa: E402
    JiguangRuntimeError,
    load_runtime_records,
    plan_runtime,
    record_runtime,
    workspace_gate,
)
from jiguang_device_key import resolve_key_material  # noqa: E402

DEFAULT_STATE = ROOT / ".vaws-local" / "jiguang" / "runtimes.json"


def object_json(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return payload


def emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("outcome") in {"success", "ready", "planned"} else 2


def host_target(record: dict[str, Any]) -> machine_ops.SshTarget:
    return machine_ops.SshTarget(
        host=record["host"]["ip"],
        user=record["host"].get("user", "root"),
        port=int(record["host"].get("port", 22)),
    )


PROMOTE_SCRIPT = r'''set -euo pipefail
candidate="$1"
current="$2"
previous="$3"
docker inspect "$candidate" >/dev/null
if docker inspect "$current" >/dev/null 2>&1; then
  docker stop --time 30 "$current" >/dev/null
  docker rename "$current" "$previous"
fi
docker rename "$candidate" "$current"
docker start "$current" >/dev/null
python3 - "$current" "$previous" <<'PY'
import json, sys
print("__VAWS_JSON__=" + json.dumps({"success": True, "status": "promoted", "container_name": sys.argv[1], "previous_container": sys.argv[2]}))
PY
'''


ROLLBACK_SCRIPT = r'''set -euo pipefail
current="$1"
previous="$2"
failed="$3"
docker inspect "$current" >/dev/null
docker inspect "$previous" >/dev/null
docker stop --time 30 "$current" >/dev/null
docker rename "$current" "$failed"
docker rename "$previous" "$current"
docker start "$current" >/dev/null
python3 - "$current" "$failed" <<'PY'
import json, sys
print("__VAWS_JSON__=" + json.dumps({"success": True, "status": "rolled_back", "container_name": sys.argv[1], "failed_container": sys.argv[2]}))
PY
'''


def run_host_script(target: machine_ops.SshTarget, source: str, args: list[str]) -> dict[str, Any]:
    result = machine_ops.run_remote_script(
        target,
        source,
        args=args,
        batch_mode=True,
        timeout_seconds=180,
    )
    return machine_ops.assert_remote_success(result)


def ensure(args: argparse.Namespace) -> dict[str, Any]:
    gate = workspace_gate(
        args.repo_root.resolve(),
        explicit_vllm_commit=args.vllm_commit,
    )
    if gate["outcome"] != "ready":
        return gate
    machine_record = workflow.find_record(args.machine)
    if machine_record is None:
        return {"outcome": "blocked", "error": "machine is not managed by this workspace"}
    key_material = resolve_key_material(args.private_key_file)
    plan = plan_runtime(
        machine=args.machine,
        image_digest=args.image,
        components=args.runtime_components_json,
        repo_root=args.repo_root.resolve(),
        state_path=args.state_file.resolve(),
        force_clean=args.force_clean,
    )
    if plan["decision"] == "reuse":
        return {
            **plan,
            "outcome": "ready",
            "changed": False,
            "ssh_key_fingerprint": key_material.fingerprint,
        }
    if not args.confirm:
        return {
            **plan,
            "outcome": "planned",
            "changed": False,
            "requires_confirm": True,
            "ssh_key_fingerprint": key_material.fingerprint,
        }

    target = host_target(machine_record)
    probe = workflow.probe_host(
        target,
        image=args.image,
        machine_type=machine_record["host"].get("machine_type"),
        managed_prefix="vaws-jiguang",
    )
    port = probe.get("free_port")
    if not isinstance(port, int):
        return {"outcome": "blocked", "error": "host probe did not return a free container SSH port", "probe": probe}
    candidate = "vaws-jiguang-next"
    bootstrap = workflow.bootstrap_container(
        target,
        host=machine_record["host"]["ip"],
        container_name=candidate,
        container_ssh_port=port,
        image=args.image,
        workdir="/vllm-workspace",
        namespace=machine_record.get("namespace"),
        machine_type=machine_record["host"].get("machine_type"),
        soc=machine_record["host"].get("soc"),
        replace_container_on_image_change=True,
        use_prepared_image_cache=True,
        visible_devices=None,
        public_key_file=str(key_material.public_key),
    )
    if bootstrap.get("status") in {"blocked", "needs_input", "needs_repair"} or bootstrap.get("success") is False:
        return {"outcome": "blocked", "error": "candidate runtime bootstrap failed", "bootstrap": bootstrap}
    smoke_args = argparse.Namespace(
        host=machine_record["host"]["ip"],
        user="root",
        container_ssh_port=port,
        python=None,
    )
    smoke = machine_ops.smoke_payload(smoke_args)
    if not smoke.get("success", smoke.get("ok", False)):
        return {"outcome": "blocked", "error": "candidate runtime validation failed", "smoke": smoke}

    current_record = plan.get("current") or {}
    generation = int(current_record.get("generation") or 0) + 1
    previous_name = f"vaws-jiguang-prev-{generation - 1}"
    promoted = run_host_script(target, PROMOTE_SCRIPT, [candidate, "vaws-jiguang", previous_name])
    selected_image = bootstrap.get("selected_image") or bootstrap.get("run_image") or args.image
    runtime_record = {
        "container_name": "vaws-jiguang",
        "container_ssh_port": port,
        "generation": generation,
        "image_digest": selected_image,
        "runtime_hash": plan["runtime_hash"],
        "native_code_hash": plan["native_code_hash"],
        "health": "ready",
        "ssh_key_fingerprint": key_material.fingerprint,
        "previous": {
            "container_name": previous_name,
            "container_ssh_port": current_record.get("container_ssh_port"),
            "generation": current_record.get("generation"),
            "image_digest": current_record.get("image_digest"),
            "runtime_hash": current_record.get("runtime_hash"),
            "native_code_hash": current_record.get("native_code_hash"),
        } if current_record else None,
    }
    record_runtime(args.state_file.resolve(), args.machine, runtime_record)
    return {
        "outcome": "success",
        "changed": True,
        "decision": plan["decision"],
        "reason": plan["reason"],
        "runtime": runtime_record,
        "promotion": promoted,
    }


def rollback(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm:
        return {"outcome": "planned", "action": "rollback", "requires_confirm": True}
    gate = workspace_gate(
        args.repo_root.resolve(),
        explicit_vllm_commit=args.vllm_commit,
    )
    if gate["outcome"] != "ready":
        return gate
    machine_record = workflow.find_record(args.machine)
    if machine_record is None:
        return {"outcome": "blocked", "error": "machine is not managed by this workspace"}
    state = load_runtime_records(args.state_file.resolve())
    current = state["machines"].get(args.machine)
    previous = current.get("previous") if isinstance(current, dict) else None
    if not isinstance(previous, dict) or not previous.get("container_name"):
        return {"outcome": "blocked", "error": "no recorded rollback generation"}
    failed_name = f"vaws-jiguang-failed-{current['generation']}"
    result = run_host_script(
        host_target(machine_record),
        ROLLBACK_SCRIPT,
        ["vaws-jiguang", previous["container_name"], failed_name],
    )
    restored = {
        "container_name": "vaws-jiguang",
        "container_ssh_port": previous.get("container_ssh_port"),
        "generation": previous.get("generation"),
        "image_digest": previous.get("image_digest"),
        "runtime_hash": previous.get("runtime_hash"),
        "native_code_hash": previous.get("native_code_hash"),
        "health": "ready",
        "previous": None,
        "failed_container": failed_name,
    }
    record_runtime(args.state_file.resolve(), args.machine, restored)
    return {"outcome": "success", "runtime": restored, "rollback": result}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--repo-root", type=Path, default=ROOT)
    root.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    actions = root.add_subparsers(dest="action", required=True)
    status = actions.add_parser("status")
    status.add_argument("--machine", required=True)
    ensure_parser = actions.add_parser("ensure")
    ensure_parser.add_argument("--machine", required=True)
    ensure_parser.add_argument("--image", required=True)
    ensure_parser.add_argument("--runtime-components-json", type=object_json, default={})
    ensure_parser.add_argument("--force-clean", action="store_true")
    ensure_parser.add_argument("--private-key-file", required=True)
    ensure_parser.add_argument("--confirm", action="store_true")
    ensure_parser.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")
    rollback_parser = actions.add_parser("rollback")
    rollback_parser.add_argument("--machine", required=True)
    rollback_parser.add_argument("--confirm", action="store_true")
    rollback_parser.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.action == "status":
            record = load_runtime_records(args.state_file.resolve())["machines"].get(args.machine)
            result = {"outcome": "ready" if record else "blocked", "runtime": record}
        elif args.action == "ensure":
            result = ensure(args)
        else:
            result = rollback(args)
    except (JiguangRuntimeError, workflow.WorkflowError, machine_ops.MachineManagementError) as exc:
        result = {"outcome": "blocked", "error": str(exc)}
    return emit(result)


if __name__ == "__main__":
    raise SystemExit(main())
