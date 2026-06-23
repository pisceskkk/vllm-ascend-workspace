#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    BootstrapError,
    ensure_clean_or_confirm,
    ensure_repo_root,
    full_commit,
    git,
    print_json,
    prompt_choice,
    prompt_text,
    short_commit,
)


def current_branch(repo: Path) -> str | None:
    proc = git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], check=False)
    if proc.returncode != 0:
        return None
    branch = proc.stdout.strip()
    return None if branch == "HEAD" else branch


def tracking_branch(repo: Path) -> str | None:
    proc = git(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def update_workspace(repo_root: Path) -> dict[str, Any]:
    ensure_repo_root(repo_root)
    before = full_commit(repo_root, "HEAD")
    git(repo_root, ["pull", "--ff-only"])
    after = full_commit(repo_root, "HEAD")
    return {"before": before, "after": after, "changed": before != after}


def init_submodules(repo_root: Path) -> dict[str, Any]:
    ensure_repo_root(repo_root)
    git(repo_root, ["submodule", "sync", "--recursive"])
    git(repo_root, ["submodule", "update", "--init", "--recursive"])
    status = git(repo_root, ["submodule", "status", "--recursive"], check=False)
    return {"status": status.stdout.strip().splitlines() if status.returncode == 0 else []}


def remote_names(repo: Path) -> list[str]:
    ensure_git_repo_root(repo)
    proc = git(repo, ["remote"], check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def ensure_git_repo_root(repo: Path) -> Path:
    repo = repo.resolve()
    proc = git(repo, ["rev-parse", "--show-toplevel"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise BootstrapError(f"not an initialized git repository: {repo}")
    top = Path(proc.stdout.strip()).resolve()
    if top != repo:
        raise BootstrapError(
            f"{repo} is not an initialized repository root; git root resolved to {top}. "
            "Initialize submodules before using this helper."
        )
    return repo


def fetch_all(repo: Path) -> dict[str, Any]:
    repo = ensure_git_repo_root(repo)
    result: dict[str, Any] = {"repo": str(repo), "remotes": {}}
    for name in remote_names(repo):
        proc = git(repo, ["fetch", "--tags", name], check=False)
        result["remotes"][name] = {
            "status": "ok" if proc.returncode == 0 else "failed",
            "stderr": proc.stderr.strip()[-1200:],
        }
    return result


def log_lines(repo: Path, ref: str, limit: int = 8) -> list[str]:
    repo = ensure_git_repo_root(repo)
    proc = git(
        repo,
        [
            "log",
            "--date=short",
            f"--max-count={limit}",
            "--format=%h %cd %d %s",
            ref,
        ],
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]


def recent_tags(repo: Path, limit: int = 20) -> list[str]:
    repo = ensure_git_repo_root(repo)
    proc = git(
        repo,
        [
            "for-each-ref",
            "refs/tags",
            "--sort=-creatordate",
            f"--count={limit}",
            "--format=%(refname:short) %(creatordate:short) %(subject)",
        ],
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.rstrip() for line in proc.stdout.splitlines() if line.strip()]


def preview(repo: Path) -> dict[str, Any]:
    repo = ensure_git_repo_root(repo)
    return {
        "current": log_lines(repo, "HEAD", limit=1),
        "upstream_main": log_lines(repo, "upstream/main"),
        "origin_main": log_lines(repo, "origin/main"),
        "recent_tags": recent_tags(repo),
    }


def resolve_ref(repo: Path, ref: str) -> str:
    repo = ensure_git_repo_root(repo)
    proc = git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise BootstrapError(f"cannot resolve ref in {repo.name}: {ref}")
    return proc.stdout.strip()


def checkout_ref(repo: Path, ref: str, *, label: str, allow_dirty_prompt: bool = True) -> dict[str, Any]:
    repo = ensure_git_repo_root(repo)
    if allow_dirty_prompt:
        ensure_clean_or_confirm(repo, label)
    before = full_commit(repo, "HEAD")
    target = resolve_ref(repo, ref)
    if before == target:
        return {"repo": str(repo), "ref": ref, "before": before, "after": target, "changed": False}
    git(repo, ["checkout", ref])
    after = full_commit(repo, "HEAD")
    return {"repo": str(repo), "ref": ref, "before": before, "after": after, "changed": before != after}


def resolve_ci_pin(repo_root: Path) -> dict[str, Any]:
    resolver = repo_root / ".agents" / "skills" / "repo-init" / "scripts" / "resolve_vllm_ci_pin.py"
    proc = subprocess.run(
        [sys.executable, str(resolver), "--vllm-ascend-dir", str(repo_root / "vllm-ascend")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"failed to parse CI pin resolver output: {proc.stdout}") from exc
    payload["returncode"] = proc.returncode
    return payload


def selectable_tags(repo: Path) -> list[str]:
    rows = recent_tags(repo)
    return [row.split(maxsplit=1)[0] for row in rows]


def choose_ref_interactive(
    repo: Path,
    *,
    label: str,
    default_kind: str,
    ci_ref: str | None = None,
) -> str:
    keyed_refs: dict[str, str] = {}
    kind_to_key: dict[str, str] = {}
    options: list[tuple[str, str]] = []

    def add(kind: str, text: str, ref: str) -> None:
        key = str(len(options) + 1)
        options.append((key, text))
        keyed_refs[key] = ref
        kind_to_key[kind] = key

    if ci_ref is not None:
        add("ci", f"CI-pinned ref: {ci_ref}", ci_ref)
    add("current", f"current checkout: {short_commit(repo) or 'unknown'}", "HEAD")
    if resolve_ref_exists(repo, "upstream/main"):
        add("upstream", f"latest upstream/main: {short_commit(repo, 'upstream/main') or 'unknown'}", "upstream/main")
    if resolve_ref_exists(repo, "origin/main"):
        add("origin", f"latest origin/main: {short_commit(repo, 'origin/main') or 'unknown'}", "origin/main")
    if selectable_tags(repo):
        add("tag", "recent tag", "__tag__")
    add("custom", "custom ref/tag/commit", "__custom__")

    default = kind_to_key.get(default_kind) or kind_to_key["current"]
    selection = prompt_choice(f"Choose {label} version:", options, default=default)
    ref = keyed_refs[selection]
    if ref == "__tag__":
        tags = selectable_tags(repo)
        tag_options = [(str(index + 1), tag) for index, tag in enumerate(tags[:20])]
        tag_choice = prompt_choice(f"Choose {label} tag:", tag_options, default="1")
        return tags[int(tag_choice) - 1]
    if ref == "__custom__":
        return prompt_text(f"Enter {label} ref/tag/commit")
    return ref


def resolve_ref_exists(repo: Path, ref: str) -> bool:
    repo = ensure_git_repo_root(repo)
    proc = git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    return proc.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap source update helpers.")
    parser.add_argument("--repo-root", default=".")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update-workspace")
    sub.add_parser("init-submodules")
    fetch_cmd = sub.add_parser("fetch")
    fetch_cmd.add_argument("repo")
    preview_cmd = sub.add_parser("preview")
    preview_cmd.add_argument("repo")
    ci_cmd = sub.add_parser("resolve-ci-pin")
    ci_cmd.set_defaults(cmd="resolve-ci-pin")
    checkout_cmd = sub.add_parser("checkout")
    checkout_cmd.add_argument("repo")
    checkout_cmd.add_argument("ref")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = Path(args.repo_root).resolve()
        if args.cmd == "update-workspace":
            print_json(update_workspace(root))
        elif args.cmd == "init-submodules":
            print_json(init_submodules(root))
        elif args.cmd == "fetch":
            print_json(fetch_all(root / args.repo))
        elif args.cmd == "preview":
            print_json(preview(root / args.repo))
        elif args.cmd == "resolve-ci-pin":
            print_json(resolve_ci_pin(root))
        elif args.cmd == "checkout":
            print_json(checkout_ref(root / args.repo, args.ref, label=args.repo, allow_dirty_prompt=False))
        return 0
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
