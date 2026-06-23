#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from common import (
    BootstrapError,
    bootstrap_state_dir,
    ensure_clean_or_confirm,
    ensure_repo_root,
    full_commit,
    infer_github_user,
    prompt_choice,
    prompt_secret,
    prompt_text,
    prompt_yes_no,
    section,
    short_commit,
    validate_github_user,
    write_json,
)
import git_credentials
import git_topology
import install_runtime
import probe
import source_update
import verify_runtime
import personal_container


def print_preview(title: str, payload: dict[str, Any], *, ci_pin: dict[str, Any] | None = None) -> None:
    section(title)
    for key, label in (
        ("current", "Current checkout"),
        ("upstream_main", "Latest upstream/main"),
        ("origin_main", "Latest origin/main"),
    ):
        rows = payload.get(key) or []
        if rows:
            print(f"{label}:")
            for row in rows[:8]:
                print(f"  {row}")
    tags = payload.get("recent_tags") or []
    if tags:
        print("Recent tags:")
        for row in tags[:12]:
            print(f"  {row}")
    if ci_pin is not None:
        print("CI-pinned vLLM:")
        if ci_pin.get("status") == "ok":
            print(f"  ref: {ci_pin.get('vllm_ref')}")
            print(f"  source: {ci_pin.get('source')}")
            print(f"  precedence: {ci_pin.get('precedence')}")
        else:
            print(f"  failed: {ci_pin.get('reason')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive vllm-ascend-workspace bootstrap.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--github-user")
    parser.add_argument("--no-token", action="store_true", help="do not ask for or store a GitHub token")
    parser.add_argument("--clear-token", action="store_true", help="clear bootstrap-managed Git credentials and exit")
    parser.add_argument("--skip-workspace-update", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument(
        "-p",
        "--persional",
        "--personal",
        "--personal-only",
        dest="personal_only",
        action="store_true",
        help="only run optional personal container setup",
    )
    parser.add_argument("--with-personal", action="store_true", help="run personal setup after normal bootstrap")
    parser.add_argument("--git-user-name")
    parser.add_argument("--git-user-email")
    parser.add_argument("--deepseek-base-url")
    parser.add_argument("--deepseek-model")
    parser.add_argument("--rtk-url")
    parser.add_argument("--vllm-ascend-ref")
    parser.add_argument("--vllm-ref")
    parser.add_argument(
        "--vllm-ref-mode",
        choices=("ci-pinned", "current", "upstream-main", "origin-main", "custom"),
        default="ci-pinned",
    )
    parser.add_argument("--yes", action="store_true", help="accept defaults for non-secret confirmations")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    return parser.parse_args()


def confirm(prompt: str, *, default: bool, assume_yes: bool) -> bool:
    return default if assume_yes else prompt_yes_no(prompt, default=default)


def choose_github_user(repo_root: Path, provided: str | None) -> str:
    if provided:
        return validate_github_user(provided)
    inferred = infer_github_user(repo_root)
    value = prompt_text("GitHub username for your vllm/vllm-ascend forks", default=inferred)
    return validate_github_user(value)


def maybe_update_workspace(repo_root: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_workspace_update:
        return {"status": "skipped", "reason": "--skip-workspace-update"}
    section("Workspace update")
    print(f"Current workspace commit: {short_commit(repo_root) or 'unknown'}")
    tracking = source_update.tracking_branch(repo_root)
    print(f"Tracking branch: {tracking or 'none'}")
    if not tracking:
        return {"status": "skipped", "reason": "workspace branch has no upstream"}
    dirty = probe.repo_info(repo_root).get("dirty")
    if dirty:
        print("Workspace has local changes; skipping automatic pull.")
        return {"status": "skipped", "reason": "workspace has local changes"}
    if not confirm("Fast-forward workspace from its tracking branch?", default=True, assume_yes=args.yes):
        return {"status": "skipped", "reason": "user declined"}
    return source_update.update_workspace(repo_root)


def select_vllm_ascend(repo_root: Path, ref_arg: str | None) -> dict[str, Any]:
    repo = repo_root / "vllm-ascend"
    section("Fetch vllm-ascend")
    fetch = source_update.fetch_all(repo)
    for remote, info in fetch.get("remotes", {}).items():
        print(f"{remote}: {info.get('status')}")
        if info.get("status") != "ok" and info.get("stderr"):
            print(f"  {info['stderr']}")
    preview = source_update.preview(repo)
    print_preview("vllm-ascend preview", preview)
    ref = ref_arg
    if not ref:
        ref = source_update.choose_ref_interactive(
            repo,
            label="vllm-ascend",
            default_kind="current",
        )
    checkout = source_update.checkout_ref(repo, ref, label="vllm-ascend")
    return {"fetch": fetch, "selected_ref": ref, "checkout": checkout}


def select_vllm(repo_root: Path, ref_arg: str | None, ref_mode: str) -> dict[str, Any]:
    repo = repo_root / "vllm"
    section("Fetch vllm")
    fetch = source_update.fetch_all(repo)
    for remote, info in fetch.get("remotes", {}).items():
        print(f"{remote}: {info.get('status')}")
        if info.get("status") != "ok" and info.get("stderr"):
            print(f"  {info['stderr']}")
    ci_pin = source_update.resolve_ci_pin(repo_root)
    preview = source_update.preview(repo)
    print_preview("vllm preview", preview, ci_pin=ci_pin)
    if ref_arg:
        ref = ref_arg
    elif ref_mode == "ci-pinned":
        if ci_pin.get("status") == "ok":
            ref = str(ci_pin["vllm_ref"])
        else:
            print("CI pin could not be resolved; choose vllm manually.")
            ref = source_update.choose_ref_interactive(repo, label="vllm", default_kind="current")
    elif ref_mode == "current":
        ref = "HEAD"
    elif ref_mode == "upstream-main":
        ref = "upstream/main"
    elif ref_mode == "origin-main":
        ref = "origin/main"
    else:
        ref = source_update.choose_ref_interactive(
            repo,
            label="vllm",
            default_kind="ci" if ci_pin.get("status") == "ok" else "current",
            ci_ref=ci_pin.get("vllm_ref") if ci_pin.get("status") == "ok" else None,
        )
    checkout = source_update.checkout_ref(repo, ref, label="vllm")
    return {"fetch": fetch, "ci_pin": ci_pin, "selected_ref": ref, "checkout": checkout}


def maybe_store_token(repo_root: Path, github_user: str, *, args: argparse.Namespace) -> dict[str, Any]:
    if args.no_token:
        return {"status": "skipped", "reason": "--no-token"}
    section("Git credentials")
    if not confirm(
        "Configure a GitHub token for passwordless push to your forks?",
        default=False,
        assume_yes=False,
    ):
        return {"status": "skipped", "reason": "user declined"}
    token = os.environ.get("GITHUB_TOKEN") or prompt_secret("GitHub token")
    result = git_credentials.store_token(repo_root, github_user, token)
    result["status"] = "configured"
    return result


def maybe_install(repo_root: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_install:
        return {"status": "skipped", "reason": "--skip-install"}
    section("Runtime reinstall")
    processes = probe.running_vllm_processes()
    if processes:
        print("Detected possible running vLLM processes:")
        for row in processes:
            print(f"  {row}")
        if not confirm("Continue reinstall while these processes may be running?", default=False, assume_yes=args.yes):
            raise BootstrapError("stopped because vLLM processes may be running")
    print(f"Python: {args.python}")
    print("This will uninstall vllm/vllm-ascend and reinstall from ./vllm and ./vllm-ascend.")
    if not confirm("Proceed with runtime reinstall?", default=True, assume_yes=args.yes):
        return {"status": "skipped", "reason": "user declined"}
    install_result = install_runtime.reinstall(repo_root, python=args.python, skip_deps=args.skip_deps)
    verify_result = verify_runtime.verify(repo_root, python=args.python)
    return {"install": install_result, "verify": verify_result, "status": "verified"}


def run_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = ensure_repo_root(Path(args.repo_root))
    state: dict[str, Any] = {"repo_root": str(repo_root), "status": "started"}
    if args.clear_token:
        state["clear_token"] = git_credentials.clear_token(repo_root)
        state["status"] = "cleared-token"
        return state
    if args.personal_only:
        state["personal"] = personal_container.run_personal_setup(args)
        state["status"] = "ok"
        return state

    section("Preflight")
    preflight = probe.collect(repo_root)
    state["preflight"] = preflight
    print(f"Repo root: {repo_root}")
    print(f"Python: {preflight.get('python')}")
    print(f"pip: {preflight.get('pip')}")
    print(f"git: {preflight.get('git')}")

    state["workspace_update"] = maybe_update_workspace(repo_root, args=args)

    section("Submodules")
    state["submodules"] = source_update.init_submodules(repo_root)
    print("Submodules initialized.")

    github_user = choose_github_user(repo_root, args.github_user)
    section("Managed remotes")
    state["topology"] = git_topology.configure(repo_root, github_user)
    print("Configured vllm/vllm-ascend origin and upstream remotes.")

    state["credentials"] = maybe_store_token(repo_root, github_user, args=args)
    state["vllm_ascend"] = select_vllm_ascend(repo_root, args.vllm_ascend_ref)
    state["vllm"] = select_vllm(repo_root, args.vllm_ref, args.vllm_ref_mode)

    section("Final source selection")
    print(f"vllm-ascend: {full_commit(repo_root / 'vllm-ascend')}")
    print(f"vllm:        {full_commit(repo_root / 'vllm')}")

    state["runtime"] = maybe_install(repo_root, args=args)
    if args.with_personal:
        state["personal"] = personal_container.run_personal_setup(args)
    state["final"] = {
        "workspace": full_commit(repo_root),
        "vllm": full_commit(repo_root / "vllm"),
        "vllm_ascend": full_commit(repo_root / "vllm-ascend"),
    }
    state["status"] = "ok"
    return state


def main() -> int:
    args = parse_args()
    try:
        state = run_bootstrap(args)
        state_path = bootstrap_state_dir(Path(args.repo_root)) / "state.json"
        write_json(state_path, state)
        section("Bootstrap complete")
        print(f"Status: {state.get('status')}")
        print(f"State: {state_path}")
        if state.get("runtime", {}).get("install", {}).get("log_dir"):
            print(f"Install logs: {state['runtime']['install']['log_dir']}")
        return 0
    except BootstrapError as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("bootstrap interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
