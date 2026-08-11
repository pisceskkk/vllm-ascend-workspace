#!/usr/bin/env python3
"""Classify GitHub CLI auth and network state without printing credentials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vaws_github_auth import NETWORK_CONTEXTS, classify_github_auth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--network-context", choices=NETWORK_CONTEXTS, default="unknown")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    result = classify_github_auth(
        network_context=args.network_context,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if result["auth_state"] == "authenticated":
        return 0
    if result["auth_state"] == "unverified":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
