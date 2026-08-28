#!/usr/bin/env python3
"""Build a canonical Jiguang performance-evaluation request."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def object_json(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--device-id", action="append", dest="device_ids", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--submodule-shas-json", type=object_json, required=True)
    parser.add_argument("--configuration-json", type=object_json, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()
    if not GIT_SHA_RE.fullmatch(args.commit_sha):
        parser.error("--commit-sha must be a full 40- or 64-character lowercase Git SHA")
    if args.repetitions < 1 or args.warmup_runs < 1:
        parser.error("repetitions and warmup-runs must be positive")
    for name, sha in args.submodule_shas_json.items():
        if not isinstance(sha, str) or not GIT_SHA_RE.fullmatch(sha):
            parser.error(f"submodule SHA for {name!r} is invalid")
    if not args.configuration_json:
        parser.error("--configuration-json must be a non-empty object")
    payload = {
        "name": args.name,
        "app_id": args.app_id,
        "deployment_id": args.deployment_id,
        "device_ids": args.device_ids,
        "evaluation_type": "performance",
        "model": args.model,
        "configuration": args.configuration_json,
        "repetitions": args.repetitions,
        "warmup_runs": args.warmup_runs,
        "commit_sha": args.commit_sha,
        "submodule_shas": args.submodule_shas_json,
    }
    print(json.dumps({"outcome": "planned", "payload": payload}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
