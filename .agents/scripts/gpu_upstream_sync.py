#!/usr/bin/env python3
"""Inspect or apply a guarded vLLM-only checkout to an upstream ref."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def git(repo: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return process.stdout.strip()


def inspect(repo: Path, target: str) -> dict[str, object]:
    current = git(repo, "rev-parse", "HEAD")
    resolved = git(repo, "rev-parse", f"{target}^{{commit}}")
    status = git(repo, "status", "--porcelain")
    files = git(repo, "diff", "--name-only", f"{current}...{resolved}").splitlines()
    return {
        "repository": str(repo),
        "current": current,
        "target": target,
        "target_commit": resolved,
        "clean": not bool(status),
        "changed_files": files,
        "vllm_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    try:
        payload = inspect(repo, args.target)
        if args.apply:
            if not payload["clean"]:
                raise RuntimeError(
                    "refusing checkout: vLLM repository has local changes"
                )
            git(repo, "checkout", "--detach", str(payload["target_commit"]))
            payload["action"] = "checked_out"
            payload["current"] = git(repo, "rev-parse", "HEAD")
        else:
            payload["action"] = "inspection_only"
        payload["status"] = "passed"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
