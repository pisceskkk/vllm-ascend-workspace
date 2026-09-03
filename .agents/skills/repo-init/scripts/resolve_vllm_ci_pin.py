#!/usr/bin/env python3
"""Resolve the exact vLLM commit paired with the local vllm-ascend HEAD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vllm_version_pairing import VllmVersionPairingError, resolve_vllm_pairing  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve the strict vLLM commit paired with a vllm-ascend checkout.",
        allow_abbrev=False,
    )
    parser.add_argument("--vllm-ascend-dir", type=Path, default=Path("vllm-ascend"))
    parser.add_argument("--vllm-dir", type=Path, default=Path("vllm"))
    parser.add_argument("--vllm-commit", help="explicit user-specified vLLM commit override")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = resolve_vllm_pairing(
            vllm_repo=args.vllm_dir,
            vllm_ascend_repo=args.vllm_ascend_dir,
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
