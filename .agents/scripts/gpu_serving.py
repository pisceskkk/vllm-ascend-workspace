#!/usr/bin/env python3
"""Start, inspect, log, or stop one named vLLM service in a GPU container."""

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
    validate_safe_name,
)

STATE_ROOT = "/workspace/vllm/.vllm-gpu-state/services"


def remote_script(args: argparse.Namespace, target) -> str:
    name = validate_safe_name(args.name, "service name")
    lines = [
        "set -euo pipefail",
        shell_assignment("vaws_container", target.container),
        shell_assignment("vaws_name", name),
        shell_assignment("vaws_state_root", STATE_ROOT),
        'vaws_state="$vaws_state_root/$vaws_name"',
        'docker container inspect "$vaws_container" >/dev/null',
    ]
    if args.action == "start":
        command = [
            *( ["env", f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}"] if args.cuda_visible_devices else []),
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--host",
            args.listen_host,
            "--port",
            str(args.service_port),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
        ]
        for value in args.extra_arg:
            command.append(value)
        metadata = json.dumps(
            {
                "name": name,
                "model": args.model,
                "host": args.listen_host,
                "port": args.service_port,
                "tensor_parallel_size": args.tensor_parallel_size,
                "cuda_visible_devices": args.cuda_visible_devices,
                "command": command,
            },
            sort_keys=True,
        )
        inner = "\n".join(
            [
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(STATE_ROOT + '/' + name)}",
                f'test ! -f {shlex.quote(STATE_ROOT + "/" + name + "/pid")} || ! kill -0 "$(cat {shlex.quote(STATE_ROOT + "/" + name + "/pid")})" 2>/dev/null',
                f"printf '%s\\n' {shlex.quote(metadata)} > {shlex.quote(STATE_ROOT + '/' + name + '/config.json')}",
                f"printf '%s\\n' \"$$\" > {shlex.quote(STATE_ROOT + '/' + name + '/pid')}",
                f"exec {shlex.join(command)} >> {shlex.quote(STATE_ROOT + '/' + name + '/service.log')} 2>&1",
            ]
        )
        lines.extend(
            [
                *(
                    [
                        f'docker exec "$vaws_container" test -e {shlex.quote(args.model)}'
                    ]
                    if args.model.startswith("/")
                    else []
                ),
                f'docker exec -d -w /workspace/vllm "$vaws_container" bash -lc {shlex.quote(inner)}',
                'for vaws_i in $(seq 1 50); do docker exec "$vaws_container" test -s "$vaws_state/pid" && break; sleep 0.1; done',
                'vaws_pid=$(docker exec "$vaws_container" cat "$vaws_state/pid")',
                'docker exec "$vaws_container" kill -0 "$vaws_pid"',
                'printf \'status=started\\npid=%s\\nstate=%s\\nlog=%s\\n\' "$vaws_pid" "$vaws_state" "$vaws_state/service.log"',
            ]
        )
    elif args.action == "status":
        health_code = (
            "import sys,urllib.request; "
            f"u='http://127.0.0.1:{args.service_port}/health'; "
            "r=urllib.request.urlopen(u,timeout=2); sys.exit(0 if r.status==200 else 1)"
        )
        lines.extend(
            [
                'vaws_pid=$(docker exec "$vaws_container" cat "$vaws_state/pid" 2>/dev/null || true)',
                "vaws_running=false",
                'if [ -n "$vaws_pid" ] && docker exec "$vaws_container" kill -0 "$vaws_pid" 2>/dev/null; then vaws_running=true; fi',
                f'vaws_healthy=false; if [ "$vaws_running" = true ] && docker exec "$vaws_container" python3 -c {shlex.quote(health_code)}; then vaws_healthy=true; fi',
                'printf \'status=ok\\nrunning=%s\\nhealthy=%s\\npid=%s\\nstate=%s\\nlog=%s\\n\' "$vaws_running" "$vaws_healthy" "$vaws_pid" "$vaws_state" "$vaws_state/service.log"',
            ]
        )
    elif args.action == "logs":
        lines.append(
            f'docker exec "$vaws_container" tail -n {args.lines} "$vaws_state/service.log"'
        )
    elif args.action == "stop":
        lines.extend(
            [
                'vaws_pid=$(docker exec "$vaws_container" cat "$vaws_state/pid" 2>/dev/null || true)',
                "if [ -z \"$vaws_pid\" ]; then printf 'status=already_stopped\\n'; exit 0; fi",
                'docker exec "$vaws_container" kill -TERM "$vaws_pid" 2>/dev/null || true',
                'for vaws_i in $(seq 1 50); do if ! docker exec "$vaws_container" kill -0 "$vaws_pid" 2>/dev/null; then break; fi; sleep 0.2; done',
                'if docker exec "$vaws_container" kill -0 "$vaws_pid" 2>/dev/null; then docker exec "$vaws_container" kill -KILL "$vaws_pid"; fi',
                'docker exec "$vaws_container" rm -f "$vaws_state/pid"',
                "printf 'status=stopped\\npid=%s\\n' \"$vaws_pid\"",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser)
    parser.add_argument("action", choices=("start", "status", "logs", "stop"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--model")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--service-port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--lines", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "start" and not args.model:
            raise GpuToolError("start requires --model")
        target = target_from_args(args)
        progress(
            "gpu_serving",
            args.action,
            host=target.host,
            container=target.container,
            name=args.name,
        )
        process = remote_bash(target, remote_script(args, target), timeout=120)
        require_success(process, f"GPU service {args.action}")
        stdout = process.stdout.decode(errors="replace")
        if args.action == "logs":
            print_json({"status": "passed", "name": args.name, "logs": stdout})
        else:
            fields = dict(
                line.split("=", 1) for line in stdout.splitlines() if "=" in line
            )
            print_json({"status": "passed", "name": args.name, **fields})
        return 0
    except GpuToolError as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
