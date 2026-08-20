#!/usr/bin/env python3
"""Run one native vLLM serving benchmark in a configured GPU container."""

from __future__ import annotations

import argparse
import datetime as dt
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        target = target_from_args(args)
        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        artifact = f"/workspace/vllm/.vllm-gpu-state/benchmarks/{run_id}"
        command = [
            "vllm",
            "bench",
            "serve",
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--dataset-name",
            args.dataset_name,
            "--num-prompts",
            str(args.num_prompts),
            "--request-rate",
            str(args.request_rate),
            *args.extra_arg,
        ]
        inner = f"set -o pipefail; {shlex.join(command)} 2>&1 | tee {shlex.quote(artifact + '/benchmark.log')}"
        script = "\n".join(
            [
                "set -euo pipefail",
                shell_assignment("vaws_container", target.container),
                shell_assignment("vaws_artifact", artifact),
                'docker exec "$vaws_container" mkdir -p "$vaws_artifact"',
                f'docker exec -w /workspace/vllm "$vaws_container" bash -lc {shlex.quote(inner)}',
            ]
        )
        progress(
            "gpu_benchmark",
            "run",
            host=target.host,
            container=target.container,
            run_id=run_id,
        )
        process = remote_bash(target, script, timeout=args.timeout_seconds)
        require_success(process, "GPU benchmark")
        print_json(
            {
                "status": "passed",
                "run_id": run_id,
                "artifact_dir": artifact,
                "command": command,
                "output": process.stdout.decode(errors="replace"),
            }
        )
        return 0
    except GpuToolError as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
