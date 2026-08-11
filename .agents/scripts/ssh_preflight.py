#!/usr/bin/env python3
"""Run the read-only local OpenSSH client configuration preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vaws_ssh_preflight import ssh_client_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    result = ssh_client_preflight(
        args.host,
        port=args.port,
        user=args.user,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
