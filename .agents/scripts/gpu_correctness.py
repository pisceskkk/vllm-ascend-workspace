#!/usr/bin/env python3
"""Run or compare deterministic OpenAI-chat smoke evidence for vLLM on GPU."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_gpu_common import (  # noqa: E402
    GpuToolError,
    add_target_arguments,
    print_json,
    progress,
    remote_bash,
    require_success,
    shell_assignment,
    target_from_args,
)

REMOTE_CLIENT = r"""
import json, pathlib, sys, urllib.request
base_url, model, output, prompts_json = sys.argv[1:]
prompts = json.loads(prompts_json)
results = []
for prompt in prompts:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 0}).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    results.append({"prompt": prompt, "content": payload["choices"][0]["message"]["content"], "finish_reason": payload["choices"][0].get("finish_reason"), "usage": payload.get("usage")})
evidence = {"schema_version": 1, "model": model, "base_url": base_url, "results": results}
path = pathlib.Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
print(json.dumps(evidence, sort_keys=True))
"""


def normalized(path: Path) -> list[tuple[str, str, str | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        (str(item["prompt"]), str(item["content"]).strip(), item.get("finish_reason"))
        for item in payload.get("results", [])
    ]


def cmd_run(args: argparse.Namespace) -> int:
    target = target_from_args(args)
    prompts = args.prompt or [
        "Reply with exactly: vllm-gpu-ok",
        "What is 17 + 25? Answer with only the number.",
    ]
    output = args.output or "/workspace/vllm/.vllm-gpu-state/correctness/latest.json"
    command = [
        "python3",
        "-c",
        REMOTE_CLIENT,
        args.base_url,
        args.model,
        output,
        json.dumps(prompts),
    ]
    script = "\n".join(
        [
            "set -euo pipefail",
            shell_assignment("vaws_container", target.container),
            f'docker exec -w /workspace/vllm "$vaws_container" {shlex.join(command)}',
        ]
    )
    progress("gpu_correctness", "run", host=target.host, container=target.container)
    process = remote_bash(target, script, timeout=args.timeout_seconds)
    require_success(process, "GPU correctness smoke")
    evidence = json.loads(process.stdout.decode(errors="replace").splitlines()[-1])
    print_json({"status": "passed", "artifact": output, "evidence": evidence})
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = normalized(args.baseline)
    candidate = normalized(args.candidate)
    passed = baseline == candidate
    print_json(
        {
            "status": "passed" if passed else "failed",
            "equal": passed,
            "baseline_cases": len(baseline),
            "candidate_cases": len(candidate),
            "mismatches": [
                {"index": index, "baseline": left, "candidate": right}
                for index, (left, right) in enumerate(
                    zip(baseline, candidate, strict=False)
                )
                if left != right
            ],
        }
    )
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    add_target_arguments(run)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", action="append")
    run.add_argument("--output")
    run.add_argument("--timeout-seconds", type=int, default=600)
    run.set_defaults(func=cmd_run)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except (GpuToolError, OSError, json.JSONDecodeError, KeyError) as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
