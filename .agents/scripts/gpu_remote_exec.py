#!/usr/bin/env python3
"""Execute one command in a configured vLLM NVIDIA GPU container."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_gpu_common import (  # noqa: E402
    GpuToolError,
    add_target_arguments,
    docker_exec_command,
    print_json,
    progress,
    target_from_args,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser)
    parser.add_argument("--workdir", default="/workspace/vllm")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        target = target_from_args(args)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise GpuToolError("a command is required after --")
        progress("gpu_remote_exec", "run", host=target.host, container=target.container)
        process = docker_exec_command(
            target, command, workdir=args.workdir, timeout=args.timeout_seconds
        )
        payload = {
            "status": "passed" if process.returncode == 0 else "failed",
            "target": target.public_dict(),
            "workdir": args.workdir,
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        print_json(payload)
        return 0 if process.returncode == 0 else process.returncode
    except GpuToolError as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
