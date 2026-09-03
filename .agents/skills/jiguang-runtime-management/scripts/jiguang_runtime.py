#!/usr/bin/env python3
"""Gate a workspace and plan or record its dedicated Jiguang runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_jiguang import JiguangRuntimeError, plan_runtime, record_runtime, workspace_gate  # noqa: E402

DEFAULT_STATE = ROOT / ".vaws-local" / "jiguang" / "runtimes.json"


def object_json(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    root.add_argument("--repo-root", type=Path, default=ROOT)
    root.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    actions = root.add_subparsers(dest="action", required=True)
    gate = actions.add_parser("gate")
    gate.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")
    plan = actions.add_parser("plan")
    plan.add_argument("--machine", required=True)
    plan.add_argument("--image-digest", required=True)
    plan.add_argument("--runtime-components-json", type=object_json, default={})
    plan.add_argument("--force-clean", action="store_true")
    record = actions.add_parser("record")
    record.add_argument("--machine", required=True)
    record.add_argument("--record-json", required=True, type=object_json)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.action == "gate":
            result = workspace_gate(
                args.repo_root.resolve(),
                explicit_vllm_commit=args.vllm_commit,
            )
        elif args.action == "plan":
            result = plan_runtime(
                machine=args.machine,
                image_digest=args.image_digest,
                components=args.runtime_components_json,
                repo_root=args.repo_root.resolve(),
                state_path=args.state_file.resolve(),
                force_clean=args.force_clean,
            )
        else:
            result = {
                "outcome": "success",
                "record": record_runtime(args.state_file.resolve(), args.machine, args.record_json),
            }
    except (JiguangRuntimeError, json.JSONDecodeError) as exc:
        result = {"outcome": "blocked", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("outcome") in {"ready", "planned", "success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
