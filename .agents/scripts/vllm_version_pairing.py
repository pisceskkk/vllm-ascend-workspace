#!/usr/bin/env python3
"""Resolve or check the workspace's strict vLLM/vllm-ascend source pairing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vllm_version_pairing import (  # noqa: E402
    VllmVersionPairingError,
    check_workspace_vllm_pairing,
    resolve_vllm_pairing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = parser.add_subparsers(dest="action", required=True)

    resolve = actions.add_parser("resolve", help="resolve the required exact vLLM commit")
    resolve.add_argument("--workspace-root", type=Path, default=Path("."))
    resolve.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")

    check = actions.add_parser("check", help="check the local vLLM HEAD against the contract")
    check.add_argument("--workspace-root", type=Path, default=Path("."))
    check.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = args.workspace_root.resolve()
    if args.action == "check":
        payload = check_workspace_vllm_pairing(
            workspace_root,
            explicit_vllm_commit=args.vllm_commit,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ready" else 2

    try:
        payload = resolve_vllm_pairing(
            vllm_repo=workspace_root / "vllm",
            vllm_ascend_repo=workspace_root / "vllm-ascend",
            explicit_vllm_commit=args.vllm_commit,
        )
    except VllmVersionPairingError as exc:
        payload = {"status": "failed", "reason": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
