#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from common import BootstrapError, ensure_repo_root, new_log_dir, print_json, run_streamed


SMOKE = r"""
import sys
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_ascend

print("python:", sys.executable)
print("torch:", getattr(torch, "__version__", "unknown"))
print("vllm:", getattr(vllm, "__version__", "unknown"), getattr(vllm, "__file__", "unknown"))
print("vllm_ascend:", getattr(vllm_ascend, "__file__", "unknown"))
"""


VERIFY_DEPS = r"""
import sys
from importlib.metadata import PackageNotFoundError, requires, version
from packaging.requirements import Requirement

missing = []
bad = []
for raw in requires("vllm-ascend") or []:
    req = Requirement(raw)
    if req.marker and not req.marker.evaluate():
        continue
    name = req.name
    try:
        installed = version(name)
    except PackageNotFoundError:
        missing.append(name)
        continue
    if req.specifier and installed not in req.specifier:
        bad.append(f"{name} {installed} not in {req.specifier}")

if missing or bad:
    print("dependency verification failed")
    for item in missing:
        print(f"missing: {item}")
    for item in bad:
        print(f"mismatch: {item}")
    sys.exit(1)
print("vllm-ascend dependency verification: ok")
"""


def verify(repo_root: Path, *, python: str) -> dict[str, Any]:
    repo_root = ensure_repo_root(repo_root)
    log_dir = new_log_dir(repo_root)
    run_streamed(
        [python, "-c", SMOKE],
        cwd=repo_root,
        env=os.environ.copy(),
        log_path=log_dir / "verify-imports.log",
    )
    run_streamed(
        [python, "-c", VERIFY_DEPS],
        cwd=repo_root,
        env=os.environ.copy(),
        log_path=log_dir / "verify-deps.log",
    )
    return {"status": "verified", "log_dir": str(log_dir), "python": python}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify vLLM Ascend runtime imports.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print_json(verify(Path(args.repo_root), python=args.python))
        return 0
    except BootstrapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
