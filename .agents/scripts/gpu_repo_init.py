#!/usr/bin/env python3
"""Inspect or initialize only the vLLM repository for GPU workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cwd: Path, command: list[str], *, check: bool = True) -> str:
    process = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if check and process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return process.stdout.strip()


def remote_map(repo: Path) -> dict[str, str]:
    output = run(repo, ["git", "remote", "-v"], check=False)
    remotes: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "(fetch)":
            remotes[parts[0]] = parts[1]
    return remotes


def set_remote(repo: Path, name: str, url: str) -> None:
    existing = run(repo, ["git", "remote"], check=False).splitlines()
    command = ["git", "remote", "set-url" if name in existing else "add", name, url]
    run(repo, command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--vllm-repo", type=Path)
    parser.add_argument("--init-submodule", action="store_true")
    parser.add_argument("--origin-url")
    parser.add_argument(
        "--upstream-url", default="https://github.com/vllm-project/vllm.git"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    repo = (args.vllm_repo or workspace / "vllm").expanduser().resolve()
    try:
        actions: list[str] = []
        if args.apply and args.init_submodule:
            run(workspace, ["git", "submodule", "update", "--init", "--", "vllm"])
            actions.append("initialized_vllm_submodule")
        if not (repo / ".git").exists():
            raise RuntimeError(f"vLLM repository is not initialized: {repo}")
        if args.apply:
            set_remote(repo, "upstream", args.upstream_url)
            actions.append("configured_upstream")
            if args.origin_url:
                set_remote(repo, "origin", args.origin_url)
                actions.append("configured_origin")
        payload = {
            "status": "passed",
            "mode": "apply" if args.apply else "inspect",
            "workspace": str(workspace),
            "vllm_repo": str(repo),
            "head": run(repo, ["git", "rev-parse", "HEAD"]),
            "remotes": remote_map(repo),
            "actions": actions,
            "excluded_repository": "vllm-ascend",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
