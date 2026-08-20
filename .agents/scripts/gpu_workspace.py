#!/usr/bin/env python3
"""Create, show, or probe a vLLM-only NVIDIA GPU workspace target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_gpu_common import (  # noqa: E402
    DEFAULT_GPU_STATE_ROOT,
    GpuToolError,
    add_target_arguments,
    print_json,
    progress,
    remote_bash,
    require_success,
    shell_assignment,
    target_from_args,
)

GPU_TOOL_NAMES = (
    "gpu_workspace.py",
    "gpu_repo_init.py",
    "gpu_remote_exec.py",
    "gpu_code_parity.py",
    "gpu_serving.py",
    "gpu_benchmark.py",
    "gpu_correctness.py",
    "gpu_change_validation.py",
    "gpu_performance_regression.py",
    "gpu_upstream_sync.py",
)


def probe(target) -> dict[str, object]:
    script = "\n".join(
        [
            "set -euo pipefail",
            shell_assignment("vaws_container", target.container),
            "command -v docker >/dev/null",
            "command -v nvidia-smi >/dev/null",
            "vaws_gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)",
            'test "$vaws_gpu_count" -gt 0',
            "vaws_gpu_models=$(nvidia-smi --query-gpu=name --format=csv,noheader | LC_ALL=C sort -u | paste -sd, -)",
            "vaws_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
            'docker container inspect "$vaws_container" >/dev/null',
            "vaws_running=$(docker inspect \"$vaws_container\" --format '{{.State.Running}}')",
            'vaws_workspace=$(docker inspect "$vaws_container" --format \'{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}\')',
            'test -n "$vaws_workspace"',
            'vaws_docker_root=$(docker info --format \'{{.DockerRootDir}}\')',
            'vaws_docker_free=$(df -B1 --output=avail "$vaws_docker_root" | tail -1 | tr -d " ")',
            'vaws_workspace_free=$(df -B1 --output=avail "$vaws_workspace" | tail -1 | tr -d " ")',
            'vaws_model_root=$(docker inspect "$vaws_container" --format \'{{range .Mounts}}{{if eq .Destination "/home/weight"}}{{.Source}}{{end}}{{end}}\')',
            'vaws_visible=$(docker exec "$vaws_container" nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)',
            'printf \'gpu_count=%s\\ngpu_models=%s\\ndriver=%s\\ncontainer_running=%s\\nworkspace_host=%s\\ncontainer_visible_gpus=%s\\ndocker_root=%s\\ndocker_free_bytes=%s\\nworkspace_free_bytes=%s\\nmodel_root_host=%s\\n\' "$vaws_gpu_count" "$vaws_gpu_models" "$vaws_driver" "$vaws_running" "$vaws_workspace" "$vaws_visible" "$vaws_docker_root" "$vaws_docker_free" "$vaws_workspace_free" "$vaws_model_root"',
        ]
    )
    process = remote_bash(target, script)
    require_success(process, "GPU workspace probe")
    facts: dict[str, object] = {}
    for line in process.stdout.decode(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            facts[key] = (
                int(value)
                if key
                in {
                    "gpu_count",
                    "container_visible_gpus",
                    "docker_free_bytes",
                    "workspace_free_bytes",
                }
                else value
            )
    return facts


def config_path(args: argparse.Namespace, host: str, container: str) -> Path:
    if args.output:
        return args.output.expanduser().resolve()
    safe_host = "".join(
        character if character.isalnum() or character in ".-" else "_"
        for character in host
    )
    return DEFAULT_GPU_STATE_ROOT / f"{safe_host}-{container}.json"


def cmd_setup(args: argparse.Namespace) -> int:
    target = target_from_args(args)
    progress("gpu_workspace", "probe", host=target.host, container=target.container)
    facts = probe(target)
    repo = args.vllm_repo.expanduser().resolve()
    if not (repo / "vllm" / "__init__.py").is_file():
        raise GpuToolError(f"not a vLLM repository: {repo}")
    path = config_path(args, target.host, target.container)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        **target.public_dict(),
        "local_vllm_repo": str(repo),
        "remote_workspace": "/workspace/vllm",
        "remote_workspace_host": str(facts["workspace_host"]) + "/vllm",
        "tool_root": str(ROOT / ".agents" / "scripts"),
        "tool_prefix": "gpu_",
        "tools": list(GPU_TOOL_NAMES),
        "facts": facts,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    progress("gpu_workspace", "ready", config=str(path))
    print_json({"status": "ready", "workspace_config": str(path), **payload})
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    target = target_from_args(args)
    facts = probe(target)
    print_json({"status": "ready", "target": target.public_dict(), "facts": facts})
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(
            args.workspace_config.expanduser().resolve().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuToolError(str(exc)) from exc
    print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup")
    add_target_arguments(setup)
    setup.add_argument("--vllm-repo", type=Path, default=ROOT / "vllm")
    setup.add_argument("--output", type=Path)
    setup.set_defaults(func=cmd_setup)
    probe_parser = subparsers.add_parser("probe")
    add_target_arguments(probe_parser)
    probe_parser.set_defaults(func=cmd_probe)
    show = subparsers.add_parser("show")
    show.add_argument("--workspace-config", type=Path, required=True)
    show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except GpuToolError as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
