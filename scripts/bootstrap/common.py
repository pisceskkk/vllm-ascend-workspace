#!/usr/bin/env python3
from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


class BootstrapError(RuntimeError):
    """User-facing bootstrap failure."""


GITHUB_RE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"
)
GITHUB_USER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


@dataclass(frozen=True)
class Command:
    argv: list[str]
    cwd: Path | None = None

    def display(self) -> str:
        prefix = f"(cd {shlex.quote(str(self.cwd))} &&) " if self.cwd else ""
        return prefix + " ".join(shlex.quote(part) for part in self.argv)


def eprint(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


def section(title: str) -> None:
    eprint()
    eprint(f"== {title} ==")


def run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = list(argv)
    if cmd and cmd[0] == "git":
        cmd = ["git", "-c", "safe.directory=*", *cmd[1:]]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise BootstrapError(
            f"command failed ({proc.returncode}): "
            f"{' '.join(shlex.quote(part) for part in cmd)}\n{detail}"
        )
    return proc


def run_streamed(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path,
) -> None:
    cmd = list(argv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    eprint(f"[run] {' '.join(shlex.quote(part) for part in cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        if cwd:
            log_file.write(f"# cwd: {cwd}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            eprint(line.rstrip("\n"))
        rc = proc.wait()
    if rc != 0:
        raise BootstrapError(
            f"command failed ({rc}); see log: {log_path}"
        )


def git(repo: Path, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args], check=check)


def repo_root_from(path: Path) -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"], cwd=path, check=True)
    return Path(proc.stdout.strip()).resolve()


def ensure_repo_root(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise BootstrapError(f"not a git repository root: {repo_root}")
    actual = repo_root_from(repo_root)
    if actual != repo_root:
        raise BootstrapError(f"expected repo root {repo_root}, git root is {actual}")
    if not (repo_root / ".gitmodules").exists():
        raise BootstrapError(f"missing .gitmodules under {repo_root}")
    return repo_root


def parse_github_repo(url: str | None) -> str | None:
    if not url:
        return None
    match = GITHUB_RE.match(url.strip())
    if not match:
        return None
    return f"{match.group('owner')}/{match.group('repo')}"


def infer_github_user(repo_root: Path) -> str | None:
    proc = git(repo_root, ["remote", "get-url", "origin"], check=False)
    repo = parse_github_repo(proc.stdout.strip() if proc.returncode == 0 else None)
    if not repo:
        return None
    owner = repo.split("/", 1)[0]
    if owner in {"vllm-project", "maoxx241"}:
        return None
    return owner


def validate_github_user(value: str) -> str:
    value = value.strip()
    if not GITHUB_USER_RE.match(value):
        raise BootstrapError(
            "GitHub username must be 1-39 chars, using letters, digits, "
            "or hyphen, and cannot start/end with hyphen"
        )
    return value


def prompt_text(prompt: str, *, default: str | None = None, required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value or not required:
            return value
        eprint("Value is required.")


def prompt_secret(prompt: str) -> str:
    while True:
        value = getpass.getpass(f"{prompt}: ").strip()
        if value:
            return value
        eprint("Value is required.")


def prompt_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        value = input(f"{prompt}{suffix}: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        eprint("Please answer y or n.")


def prompt_choice(prompt: str, options: list[tuple[str, str]], *, default: str) -> str:
    eprint(prompt)
    valid = {key for key, _ in options}
    for key, label in options:
        marker = " (default)" if key == default else ""
        eprint(f"  {key}. {label}{marker}")
    while True:
        value = input(f"Selection [{default}]: ").strip()
        if not value:
            value = default
        if value in valid:
            return value
        eprint(f"Choose one of: {', '.join(sorted(valid))}")


def short_commit(repo: Path, ref: str = "HEAD") -> str | None:
    proc = git(repo, ["rev-parse", "--short", ref], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def full_commit(repo: Path, ref: str = "HEAD") -> str | None:
    proc = git(repo, ["rev-parse", ref], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def status_rows(repo: Path) -> list[str]:
    proc = git(repo, ["status", "--porcelain"], check=False)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def ensure_clean_or_confirm(repo: Path, label: str) -> None:
    rows = status_rows(repo)
    if not rows:
        return
    section(f"{label} has local changes")
    for line in rows[:20]:
        eprint(f"  {line}")
    if len(rows) > 20:
        eprint(f"  ... {len(rows) - 20} more")
    if not prompt_yes_no("Continue anyway?", default=False):
        raise BootstrapError(f"stopped because {label} has local changes")


def bootstrap_state_dir(repo_root: Path) -> Path:
    target = repo_root / ".vaws-local" / "bootstrap"
    target.mkdir(parents=True, exist_ok=True)
    return target


def new_log_dir(repo_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = bootstrap_state_dir(repo_root) / "logs" / stamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
