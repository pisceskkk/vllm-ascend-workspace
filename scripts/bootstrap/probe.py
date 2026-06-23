#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import ensure_repo_root, full_commit, git, parse_github_repo, repo_root_from, status_rows


SUBMODULES = ("vllm", "vllm-ascend", "aisbench_auto_tools", "benchmark")


def command_version(argv: list[str]) -> str | None:
    if not shutil.which(argv[0]):
        return None
    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else None


def repo_info(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "initialized": False,
    }
    if not path.exists():
        return result
    proc = git(path, ["rev-parse", "--show-toplevel"], check=False)
    if proc.returncode != 0:
        result["error"] = proc.stderr.strip() or "not a git repository"
        return result
    root = Path(proc.stdout.strip()).resolve()
    if root != path.resolve():
        result["error"] = f"git root resolves to {root}"
        return result
    result["initialized"] = True
    result["head"] = full_commit(path, "HEAD")
    result["dirty_entries"] = status_rows(path)[:20]
    result["dirty"] = bool(result["dirty_entries"])
    remotes: dict[str, Any] = {}
    remote_proc = git(path, ["remote"], check=False)
    if remote_proc.returncode == 0:
        for name in [line.strip() for line in remote_proc.stdout.splitlines() if line.strip()]:
            url_proc = git(path, ["remote", "get-url", name], check=False)
            url = url_proc.stdout.strip() if url_proc.returncode == 0 else None
            remotes[name] = {"url": url, "github_repo": parse_github_repo(url)}
    result["remotes"] = remotes
    return result


def submodule_status(repo_root: Path) -> list[dict[str, str]]:
    proc = git(repo_root, ["submodule", "status", "--recursive"], check=False)
    rows: list[dict[str, str]] = []
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or "failed to inspect submodules"}]
    for line in proc.stdout.splitlines():
        if not line:
            continue
        state = line[0]
        parts = line[1:].strip().split()
        rows.append({
            "state": state,
            "commit": parts[0] if parts else "",
            "path": parts[1] if len(parts) > 1 else "",
            "detail": " ".join(parts[2:]) if len(parts) > 2 else "",
        })
    return rows


def running_vllm_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return []
    rows: list[str] = []
    self_pid = str(os.getpid())
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(self_pid):
            continue
        lowered = stripped.lower()
        if "vllm serve" in lowered or "vllm.entrypoints" in lowered:
            rows.append(stripped)
    return rows[:20]


def collect(repo_root: Path) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    pip_proc = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "repo_root": str(repo_root),
        "python": sys.executable,
        "pip": pip_proc.stdout.strip() if pip_proc.returncode == 0 else None,
        "git": command_version(["git", "--version"]),
        "workspace": repo_info(repo_root),
        "submodule_status": submodule_status(repo_root),
        "repos": {
            name: repo_info(repo_root / name)
            for name in SUBMODULES
        },
        "running_vllm_processes": running_vllm_processes(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe bootstrap preflight state.")
    parser.add_argument("--repo-root", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from(Path.cwd())
    print(json.dumps(collect(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
