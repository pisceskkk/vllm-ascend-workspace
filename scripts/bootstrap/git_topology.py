#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import BootstrapError, ensure_repo_root, git, print_json, validate_github_user


MANAGED = {
    "vllm": {
        "origin": "https://github.com/{user}/vllm.git",
        "upstream": "https://github.com/vllm-project/vllm.git",
    },
    "vllm-ascend": {
        "origin": "https://github.com/{user}/vllm-ascend.git",
        "upstream": "https://github.com/vllm-project/vllm-ascend.git",
    },
}


def ensure_submodule_repo(repo_root: Path, name: str) -> Path:
    path = (repo_root / name).resolve()
    proc = git(path, ["rev-parse", "--show-toplevel"], check=False)
    if proc.returncode != 0:
        raise BootstrapError(f"{name} is not initialized; run submodule init first")
    top = Path(proc.stdout.strip()).resolve()
    if top != path:
        raise BootstrapError(f"{name} git root is {top}, expected {path}")
    return path


def remote_url(repo: Path, name: str) -> str | None:
    proc = git(repo, ["remote", "get-url", name], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def set_remote(repo: Path, name: str, url: str) -> dict[str, Any]:
    before = remote_url(repo, name)
    if before is None:
        git(repo, ["remote", "add", name, url])
        action = "add"
    elif before != url:
        git(repo, ["remote", "set-url", name, url])
        action = "set-url"
    else:
        action = "unchanged"
    return {"remote": name, "before": before, "after": url, "action": action}


def configure(repo_root: Path, github_user: str) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    github_user = validate_github_user(github_user)
    result: dict[str, Any] = {"github_user": github_user, "repos": {}}
    for name, remotes in MANAGED.items():
        repo = ensure_submodule_repo(repo_root, name)
        entries = []
        for remote_name, template in remotes.items():
            entries.append(set_remote(repo, remote_name, template.format(user=github_user)))
        result["repos"][name] = {
            "path": str(repo),
            "changes": entries,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure managed submodule remotes.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--github-user", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print_json(configure(Path(args.repo_root), args.github_user))
        return 0
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
