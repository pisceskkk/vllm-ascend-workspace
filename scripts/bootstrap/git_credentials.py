#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import BootstrapError, ensure_repo_root, git, print_json, run, validate_github_user


REPOS = ("vllm", "vllm-ascend")


def credential_file(repo_root: Path) -> Path:
    return repo_root / ".vaws-local" / "git-credentials"


def helper_value(repo_root: Path) -> str:
    return f"store --file={credential_file(repo_root)}"


def is_repo_root(repo: Path) -> bool:
    proc = git(repo, ["rev-parse", "--show-toplevel"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    return Path(proc.stdout.strip()).resolve() == repo.resolve()


def configure_helper(repo_root: Path, repo: Path) -> None:
    git(repo, ["config", "--local", "credential.helper", helper_value(repo_root)])
    git(repo, ["config", "--local", "credential.useHttpPath", "true"])


def config_values(repo: Path, key: str) -> list[str]:
    proc = git(repo, ["config", "--local", "--get-all", key], check=False)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def replace_config_values(repo: Path, key: str, values: list[str]) -> None:
    git(repo, ["config", "--local", "--unset-all", key], check=False)
    for value in values:
        git(repo, ["config", "--local", "--add", key, value])


def credential_payload(github_user: str, repo_name: str, token: str | None = None) -> str:
    lines = [
        "protocol=https",
        "host=github.com",
        f"path={github_user}/{repo_name}.git",
        f"username={github_user}",
    ]
    if token is not None:
        lines.append(f"password={token}")
    return "\n".join(lines) + "\n\n"


def approve(repo: Path, payload: str) -> None:
    run(["git", "-C", str(repo), "credential", "reject"], input_text=payload, check=False)
    run(["git", "-C", str(repo), "credential", "approve"], input_text=payload, check=True)


def store_token(repo_root: Path, github_user: str, token: str) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    github_user = validate_github_user(github_user)
    cred_file = credential_file(repo_root)
    cred_file.parent.mkdir(parents=True, exist_ok=True)
    cred_file.parent.chmod(0o700)
    result: dict[str, Any] = {"credential_file": str(cred_file), "repos": {}}
    for repo_name in REPOS:
        repo = repo_root / repo_name
        if not is_repo_root(repo):
            raise BootstrapError(f"{repo_name} is not initialized")
        configure_helper(repo_root, repo)
        payload = credential_payload(github_user, repo_name, token)
        approve(repo, payload)
        result["repos"][repo_name] = "configured"
    cred_file.chmod(0o600)
    return result


def clear_token(repo_root: Path) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    cred_file = credential_file(repo_root)
    removed = False
    if cred_file.exists():
        cred_file.unlink()
        removed = True
    repos: dict[str, str] = {}
    managed_helper = helper_value(repo_root)
    for repo_name in REPOS:
        repo = repo_root / repo_name
        if not repo.exists() or not is_repo_root(repo):
            repos[repo_name] = "not-initialized"
            continue
        helpers = config_values(repo, "credential.helper")
        remaining = [value for value in helpers if value != managed_helper]
        if len(remaining) != len(helpers):
            replace_config_values(repo, "credential.helper", remaining)
            if not remaining:
                git(repo, ["config", "--local", "--unset-all", "credential.useHttpPath"], check=False)
            repos[repo_name] = "cleared"
        else:
            repos[repo_name] = "no-bootstrap-helper"
    return {"credential_file": str(cred_file), "removed": removed, "repos": repos}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage bootstrap Git credentials.")
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="cmd", required=True)
    set_cmd = sub.add_parser("set")
    set_cmd.add_argument("--github-user", required=True)
    set_cmd.add_argument("--token", required=True)
    sub.add_parser("clear")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = Path(args.repo_root)
        if args.cmd == "set":
            print_json(store_token(root, args.github_user, args.token))
        elif args.cmd == "clear":
            print_json(clear_token(root))
        return 0
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
