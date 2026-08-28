#!/usr/bin/env python3
"""Attach a canonical Jiguang archive pointer to Run Manifest v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_run_manifest import add_artifact, load_manifest, write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary_json)
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = load_manifest(args.manifest)
    updated = add_artifact(
        manifest,
        name="jiguang-summary",
        kind="external-evaluation-summary",
        uri=args.archive_url,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
    write_manifest(args.manifest, updated)
    print(json.dumps({"outcome": "success", "manifest": str(args.manifest), "artifact": "jiguang-summary"}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
