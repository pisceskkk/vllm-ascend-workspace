#!/usr/bin/env python3
"""Delegate a Jiguang Run lease to the existing host-local NPU coordinator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COORDINATOR = ROOT / ".agents" / "skills" / "session-management" / "scripts" / "npu_coordination.py"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: jiguang_lease.py <npu_coordination arguments>", file=sys.stderr)
        return 2
    completed = subprocess.run([sys.executable, str(COORDINATOR), *sys.argv[1:]], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
