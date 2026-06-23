#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from common import BootstrapError, ensure_repo_root, new_log_dir, print_json, run, run_streamed


PIP_INDEX_URL = "https://repo.huaweicloud.com/repository/pypi/simple"
PIP_TRUSTED_HOST = "repo.huaweicloud.com"
MINIMAL_VLLM_BUILD_DEPS = [
    "setuptools",
    "wheel",
    "setuptools-rust",
    "packaging",
    "ninja",
    "cmake",
]


def pip_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env["PIP_CONFIG_FILE"] = "/dev/null"
    env["PIP_INDEX_URL"] = PIP_INDEX_URL
    env["PIP_TRUSTED_HOST"] = PIP_TRUSTED_HOST
    if extra:
        env.update(extra)
    return env


def build_jobs_env() -> dict[str, str]:
    jobs = str(min(os.cpu_count() or 1, 128))
    max_jobs = os.environ.get("MAX_JOBS") or os.environ.get("VAWS_BUILD_JOBS") or jobs
    return {
        "VAWS_BUILD_JOBS": os.environ.get("VAWS_BUILD_JOBS", max_jobs),
        "MAX_JOBS": max_jobs,
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", max_jobs),
    }


def pip_show(repo_root: Path, python: str) -> str:
    proc = run(
        [python, "-m", "pip", "show", "vllm", "vllm-ascend", "vllm_ascend"],
        cwd=repo_root,
        check=False,
    )
    return proc.stdout.strip()


def reinstall(repo_root: Path, *, python: str, skip_deps: bool = False) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    log_dir = new_log_dir(repo_root)
    vllm = repo_root / "vllm"
    vllm_ascend = repo_root / "vllm-ascend"
    if not (vllm / ".git").exists():
        raise BootstrapError("vllm submodule is not initialized")
    if not (vllm_ascend / ".git").exists():
        raise BootstrapError("vllm-ascend submodule is not initialized")

    run_streamed(
        [python, "-m", "pip", "uninstall", "-y", "vllm", "vllm-ascend", "vllm_ascend"],
        cwd=repo_root,
        env=pip_env(),
        log_path=log_dir / "00-uninstall.log",
    )
    if not skip_deps:
        run_streamed(
            [python, "-m", "pip", "install", *MINIMAL_VLLM_BUILD_DEPS],
            cwd=repo_root,
            env=pip_env(),
            log_path=log_dir / "01-vllm-minimal-build-deps.log",
        )
    run_streamed(
        [
            python,
            "-m",
            "pip",
            "install",
            "-v",
            "-e",
            ".",
            "--no-build-isolation",
            "--no-deps",
        ],
        cwd=vllm,
        env=pip_env({
            "VLLM_TARGET_DEVICE": "empty",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        }),
        log_path=log_dir / "02-vllm-editable.log",
    )
    if not skip_deps:
        requirements = vllm_ascend / "requirements.txt"
        if not requirements.exists():
            raise BootstrapError(f"missing requirements file: {requirements}")
        run_streamed(
            [python, "-m", "pip", "install", "-r", str(requirements)],
            cwd=vllm_ascend,
            env=pip_env(),
            log_path=log_dir / "03-vllm-ascend-requirements.log",
        )
    run_streamed(
        [
            python,
            "-m",
            "pip",
            "install",
            "-v",
            "-e",
            ".",
            "--no-build-isolation",
            "--no-deps",
        ],
        cwd=vllm_ascend,
        env=pip_env(build_jobs_env()),
        log_path=log_dir / "04-vllm-ascend-editable.log",
    )
    return {
        "status": "installed",
        "log_dir": str(log_dir),
        "pip_index_url": PIP_INDEX_URL,
        "python": python,
        "pip_show": pip_show(repo_root, python),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reinstall vLLM and vLLM Ascend from source.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--skip-deps", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print_json(reinstall(Path(args.repo_root), python=args.python, skip_deps=args.skip_deps))
        return 0
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
