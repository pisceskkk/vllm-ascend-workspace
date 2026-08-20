#!/usr/bin/env python3
"""Map one vLLM-only Git diff to a minimum NVIDIA GPU validation plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CATEGORY_RULES = {
    "native_build": ("csrc/", "cmake/", "setup.py", "pyproject.toml"),
    "attention_kv": (
        "vllm/attention/",
        "vllm/v1/attention/",
        "vllm/distributed/kv_transfer/",
    ),
    "distributed": ("vllm/distributed/", "vllm/executor/", "vllm/v1/executor/"),
    "serving_api": ("vllm/entrypoints/", "vllm/engine/", "vllm/v1/engine/"),
    "model_execution": ("vllm/model_executor/", "vllm/v1/worker/", "vllm/worker/"),
    "scheduler": ("vllm/v1/core/", "vllm/core/"),
    "tests": ("tests/",),
    "docs": ("docs/", "README"),
}


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    process = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", f"{base}...{head}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return [line for line in process.stdout.splitlines() if line]


def classify(paths: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for category, prefixes in CATEGORY_RULES.items():
        matches = [
            path
            for path in paths
            if any(path == prefix or path.startswith(prefix) for prefix in prefixes)
        ]
        if matches:
            categories[category] = matches
    classified = {path for values in categories.values() for path in values}
    unclassified = [path for path in paths if path not in classified]
    if unclassified:
        categories["other"] = unclassified
    return categories


def validation_plan(categories: dict[str, list[str]]) -> list[dict[str, str]]:
    plan = [{"check": "static_and_targeted_tests", "reason": "all vLLM code changes"}]
    keys = set(categories)
    if keys - {"docs", "tests"}:
        plan.extend(
            [
                {"check": "gpu_import_smoke", "reason": "runtime code changed"},
                {
                    "check": "gpu_online_correctness",
                    "reason": "validate OpenAI response behavior",
                },
            ]
        )
    if keys & {"native_build", "attention_kv", "model_execution"}:
        plan.append(
            {
                "check": "cuda_image_build_or_aligned_image",
                "reason": "native/runtime compatibility risk",
            }
        )
    if keys & {"distributed", "attention_kv", "scheduler"}:
        plan.append(
            {
                "check": "multi_gpu_smoke",
                "reason": "parallel execution or scheduling changed",
            }
        )
    if keys & {
        "attention_kv",
        "distributed",
        "model_execution",
        "scheduler",
        "serving_api",
    }:
        plan.append(
            {"check": "gpu_serving_benchmark", "reason": "hot serving path changed"}
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    try:
        paths = changed_files(repo, args.base, args.head)
        categories = classify(paths)
        payload = {
            "status": "passed",
            "repository": str(repo),
            "base": args.base,
            "head": args.head,
            "changed_files": paths,
            "categories": categories,
            "validation_plan": validation_plan(categories),
            "excluded_runtime_dependencies": [
                "vllm-ascend",
                "torch_npu",
                "HCCL",
                "NPU leases",
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
