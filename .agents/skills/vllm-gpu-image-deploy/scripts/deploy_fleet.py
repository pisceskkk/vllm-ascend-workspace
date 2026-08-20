#!/usr/bin/env python3
"""Load a pinned vLLM image and replace one managed NVIDIA GPU container per host."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import shlex
import subprocess
import sys
import time

AGENTS_DIR = pathlib.Path(__file__).resolve().parents[3]
if str(AGENTS_DIR / "lib") not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR / "lib"))

from vaws_run_manifest import (  # noqa: E402
    add_artifact,
    new_manifest,
    sha256_file,
    transition_status,
    write_manifest,
)
from vaws_ssh_control import ssh_command_prefix  # noqa: E402

GPU_WORKSPACE_TOOL = AGENTS_DIR / "scripts" / "gpu_workspace.py"


def progress(phase: str, **fields: object) -> None:
    print(
        f"__VAWS_VLLM_GPU_IMAGE_DEPLOY_PROGRESS__={json.dumps({'phase': phase, **fields}, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=pathlib.Path)
    parser.add_argument("--ssh-config", type=pathlib.Path, help="explicit OpenSSH config, e.g. /dev/null")
    parser.add_argument("--remote-image-tar", required=True)
    parser.add_argument("--image-tar-sha256", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--rootfs-sha256", required=True)
    parser.add_argument("--image-build-commit", required=True)
    parser.add_argument("--remote-source-tar", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--model-root", default="/home/weight")
    parser.add_argument(
        "--expected-gpus",
        type=int,
        help="exact visible GPU count; when omitted, require at least one GPU",
    )
    parser.add_argument(
        "--expected-gpu-model",
        help="exact nvidia-smi GPU model name; when omitted, accept any NVIDIA GPU model",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--approved-replace", action="store_true")
    return parser.parse_args()


def validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise SystemExit(f"{name} must contain 64 hexadecimal characters")


def ssh_base(args: argparse.Namespace, host: str) -> list[str]:
    command = [
        *ssh_command_prefix(),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        "-o",
        "LogLevel=ERROR",
    ]
    if args.ssh_config:
        command.extend(["-F", str(args.ssh_config.expanduser().resolve())])
    if args.identity_file:
        command.extend(["-i", str(args.identity_file.expanduser().resolve()), "-o", "IdentitiesOnly=yes"])
    command.extend(["-p", str(args.port), f"{args.user}@{host}"])
    return command


def preflight(args: argparse.Namespace, host: str) -> None:
    prefix = ssh_command_prefix()
    config = subprocess.run(
        prefix
        + (["-F", str(args.ssh_config.expanduser().resolve())] if args.ssh_config else [])
        + ["-G", "-p", str(args.port), f"{args.user}@{host}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    if config.returncode != 0:
        raise RuntimeError(f"ssh -G failed: {config.stderr.strip()}")
    probe = subprocess.run(
        ssh_base(args, host) + ["printf", "ok"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0 or probe.stdout != "ok":
        raise RuntimeError(f"key SSH failed: {probe.stderr.strip()}")


def assignment(name: str, value: object) -> str:
    return f"{name}={shlex.quote(str(value))}"


def render_remote_script(args: argparse.Namespace) -> str:
    image_short = args.image_id.removeprefix("sha256:")[:12]
    lines = [
        "set -euo pipefail",
        assignment("vaws_image_tar", args.remote_image_tar),
        assignment("vaws_image_tar_sha", args.image_tar_sha256.lower()),
        assignment("vaws_image_id", args.image_id),
        assignment("vaws_image_ref", args.image_ref),
        assignment("vaws_rootfs_sha", args.rootfs_sha256.lower()),
        assignment("vaws_expected_build_commit", args.image_build_commit),
        assignment("vaws_source_tar", args.remote_source_tar),
        assignment("vaws_source_sha", args.source_sha256.lower()),
        assignment("vaws_python_sha", args.python_sha256.lower()),
        assignment("vaws_source_head", args.source_head),
        assignment("vaws_stable", args.container),
        assignment("vaws_workspace_root", args.workspace_root),
        assignment("vaws_model_root", args.model_root),
        assignment("vaws_expected_gpus", args.expected_gpus or ""),
        assignment("vaws_expected_gpu_model", args.expected_gpu_model or ""),
        assignment("vaws_image_short", image_short),
        'vaws_workspace="$vaws_workspace_root/${vaws_image_short}-${vaws_python_sha:0:12}-${vaws_source_sha:0:12}"',
        'vaws_candidate="${vaws_stable}-candidate-${vaws_image_short}"',
        'test -f "$vaws_image_tar"',
        'test -f "$vaws_source_tar"',
        'test -d "$vaws_model_root"',
        'test "$(sha256sum "$vaws_image_tar" | cut -d " " -f1)" = "$vaws_image_tar_sha"',
        'test "$(sha256sum "$vaws_source_tar" | cut -d " " -f1)" = "$vaws_source_sha"',
        'vaws_image_ready=false',
        'if docker image inspect "$vaws_image_ref" >/dev/null 2>&1; then',
        '  vaws_runtime_image_id=$(docker image inspect "$vaws_image_ref" --format \'{{.Id}}\')',
        '  vaws_observed_rootfs=$(docker image inspect "$vaws_runtime_image_id" --format \'{{range .RootFS.Layers}}{{println .}}{{end}}\' | awk \'NF\' | sha256sum | cut -d " " -f1)',
        '  vaws_observed_build_commit=$(docker image inspect "$vaws_runtime_image_id" --format \'{{range .Config.Env}}{{println .}}{{end}}\' | sed -n \'s/^VLLM_BUILD_COMMIT=//p\' | head -1)',
        '  if [ "$vaws_observed_rootfs" = "$vaws_rootfs_sha" ] && [ "$vaws_observed_build_commit" = "$vaws_expected_build_commit" ]; then vaws_image_ready=true; fi',
        'fi',
        'if [ "$vaws_image_ready" != true ]; then docker load -i "$vaws_image_tar" >/tmp/vaws-vllm-gpu-image-load.log; fi',
        'vaws_runtime_image_id=$(docker image inspect "$vaws_image_ref" --format \'{{.Id}}\')',
        'vaws_observed_rootfs=$(docker image inspect "$vaws_runtime_image_id" --format \'{{range .RootFS.Layers}}{{println .}}{{end}}\' | awk \'NF\' | sha256sum | cut -d " " -f1)',
        'vaws_observed_build_commit=$(docker image inspect "$vaws_runtime_image_id" --format \'{{range .Config.Env}}{{println .}}{{end}}\' | sed -n \'s/^VLLM_BUILD_COMMIT=//p\' | head -1)',
        'test "$vaws_observed_rootfs" = "$vaws_rootfs_sha"',
        'test "$vaws_observed_build_commit" = "$vaws_expected_build_commit"',
        'if docker container inspect "$vaws_stable" >/dev/null 2>&1; then',
        '  vaws_current_image=$(docker container inspect "$vaws_stable" --format \'{{.Config.Image}}\')',
        '  vaws_current_workspace=$(docker container inspect "$vaws_stable" --format \'{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}\')',
        '  if [ "$vaws_current_image" = "$vaws_runtime_image_id" ] && [ "$vaws_current_workspace" = "$vaws_workspace" ]; then',
        '    vaws_current_python=$(docker exec "$vaws_stable" sha256sum /workspace/vllm/.vaws-python-sha256sums 2>/dev/null | cut -d " " -f1 || true)',
        '    if [ "$vaws_current_python" = "$vaws_python_sha" ] && docker exec "$vaws_stable" bash -lc \'cd /workspace/vllm && sha256sum -c .vaws-source-sha256sums >/dev/null\' && docker exec -w /tmp "$vaws_stable" python3 -c \'import vllm, vllm._C_stable_libtorch\' >/dev/null 2>&1; then',
        '      if docker exec "$vaws_stable" test -f /workspace/vllm/vllm/third_party/deep_gemm/__init__.py; then',
        '        docker exec -w /tmp "$vaws_stable" python3 -c \'import importlib, importlib.util; s=importlib.util.find_spec("vllm.third_party.deep_gemm"); assert s is not None and s.loader.__class__.__name__ == "SourceFileLoader"; m=importlib.import_module("vllm.third_party.deep_gemm"); assert all(callable(getattr(m, n, None)) for n in ("transform_sf_into_required_layout", "fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"))\' >/dev/null',
        '      fi',
        '      printf \'status=already_ready\\ncontainer=%s\\ncontainer_id=%s\\nsource_image_id=%s\\nimage_id=%s\\nimage_build_commit=%s\\nrootfs_sha256=%s\\nworkspace=%s\\npython_sha256=%s\\nshared_libraries=%s\\nruntime_seed_files=%s\\ngpus=%s\\nrollback=\\n\' "$vaws_stable" "$(docker inspect "$vaws_stable" --format \'{{.Id}}\')" "$vaws_image_id" "$vaws_runtime_image_id" "$vaws_observed_build_commit" "$vaws_observed_rootfs" "$(docker inspect "$vaws_stable" --format \'{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}\')" "$vaws_python_sha" "$(docker exec "$vaws_stable" find /workspace/vllm/vllm -type f -name \'*.so\' | wc -l)" "$(docker exec "$vaws_stable" cat /workspace/vllm/.vaws-runtime-seed-count 2>/dev/null || printf 0)" "$(docker exec "$vaws_stable" nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"',
        '      vaws_current_gpu_count=$(docker exec "$vaws_stable" nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)',
        '      vaws_current_gpu_models=$(docker exec "$vaws_stable" nvidia-smi --query-gpu=name --format=csv,noheader | LC_ALL=C sort -u)',
        '      test "$vaws_current_gpu_count" -gt 0',
        '      if [ -n "$vaws_expected_gpus" ]; then test "$vaws_current_gpu_count" -eq "$vaws_expected_gpus"; fi',
        '      if [ -n "$vaws_expected_gpu_model" ]; then test "$vaws_current_gpu_models" = "$vaws_expected_gpu_model"; fi',
        '      vaws_current_gpu_models_csv=$(printf \'%s\\n\' "$vaws_current_gpu_models" | paste -sd, -)',
        '      printf \'gpu_models=%s\\n\' "$vaws_current_gpu_models_csv"',
        "      exit 0",
        "    fi",
        "  fi",
        "fi",
        'mkdir -p "$vaws_workspace_root"',
        'if [ ! -f "$vaws_workspace/vllm/.vaws-source-sha256sums" ]; then',
        '  vaws_workspace_stage="${vaws_workspace}.part-$$"',
        '  mkdir -p "$vaws_workspace_stage/vllm"',
        '  tar -xzf "$vaws_source_tar" -C "$vaws_workspace_stage/vllm"',
        '  test -f "$vaws_workspace_stage/vllm/.vaws-python-sha256sums"',
        '  test -f "$vaws_workspace_stage/vllm/.vaws-source-sha256sums"',
        '  (cd "$vaws_workspace_stage/vllm" && sha256sum -c .vaws-source-sha256sums >/dev/null)',
        '  if [ -e "$vaws_workspace" ]; then mv "$vaws_workspace" "${vaws_workspace}.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"; fi',
        '  mv "$vaws_workspace_stage" "$vaws_workspace"',
        "fi",
        'test -f "$vaws_workspace/vllm/.vaws-python-sha256sums"',
        'test -f "$vaws_workspace/vllm/.vaws-source-sha256sums"',
        'test -f "$vaws_workspace/vllm/.vaws-deleted-paths"',
        'vaws_observed_python=$(sha256sum "$vaws_workspace/vllm/.vaws-python-sha256sums" | cut -d " " -f1)',
        'test "$vaws_observed_python" = "$vaws_python_sha"',
        '(cd "$vaws_workspace/vllm" && sha256sum -c .vaws-source-sha256sums >/dev/null)',
        'if docker container inspect "$vaws_candidate" >/dev/null 2>&1; then',
        '  test "$(docker container inspect "$vaws_candidate" --format \'{{index .Config.Labels "vaws.skill"}}\')" = vllm-gpu-image-deploy',
        '  docker rm -f "$vaws_candidate" >/dev/null',
        "fi",
        'docker run -d --name "$vaws_candidate" --label vaws.managed=true --label vaws.skill=vllm-gpu-image-deploy --label "vaws.source-image-id=$vaws_image_id" --label "vaws.runtime-image-id=$vaws_runtime_image_id" --label "vaws.source-head=$vaws_source_head" --gpus all --network host --ipc host --mount "type=bind,src=$vaws_workspace,dst=/workspace" --mount "type=bind,src=$vaws_model_root,dst=/home/weight,readonly" -w /workspace/vllm --entrypoint /bin/bash "$vaws_runtime_image_id" -lc \'sleep infinity\' >/dev/null',
        'docker exec "$vaws_candidate" bash -lc \'set -euo pipefail',
        r'vaws_image_pkg=$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"] + \"/vllm\")")',
        'vaws_workspace_pkg=/workspace/vllm/vllm',
        'vaws_so_count=$(find "$vaws_image_pkg" -type f -name "*.so" | wc -l)',
        'test "$vaws_so_count" -gt 0',
        'vaws_runtime_seed_count=$(find "$vaws_image_pkg" -type f | while IFS= read -r vaws_runtime_file; do',
        '  vaws_rel=${vaws_runtime_file#"$vaws_image_pkg"/}',
        '  if [ ! -e "$vaws_workspace_pkg/$vaws_rel" ]; then printf ".\\n"; fi',
        'done | wc -l)',
        'printf "%s\\n" "$vaws_runtime_seed_count" > /workspace/vllm/.vaws-runtime-seed-count',
        '# Preserve local source files; add only image-built runtime files that are absent.',
        'cp -a -n "$vaws_image_pkg"/. "$vaws_workspace_pkg"/',
        'while IFS= read -r vaws_deleted; do',
        '  [ -z "$vaws_deleted" ] && continue',
        '  case "$vaws_deleted" in',
        '    vllm/*)',
        '      case "/$vaws_deleted/" in *"/../"*|*"/./"*) printf "unsafe deleted source path: %s\\n" "$vaws_deleted" >&2; exit 66 ;; esac',
        '      rm -f -- "/workspace/vllm/$vaws_deleted"',
        '      ;;',
        '    *) printf "unsafe deleted source path: %s\\n" "$vaws_deleted" >&2; exit 66 ;;',
        '  esac',
        'done < /workspace/vllm/.vaws-deleted-paths',
        '(cd /workspace/vllm && sha256sum -c .vaws-source-sha256sums >/dev/null)',
        'mv "$vaws_image_pkg" "${vaws_image_pkg}.image"',
        'ln -s "$vaws_workspace_pkg" "$vaws_image_pkg"\'',
        'vaws_runtime_seed_count=$(docker exec "$vaws_candidate" cat /workspace/vllm/.vaws-runtime-seed-count)',
        'vaws_so_count=$(docker exec "$vaws_candidate" find /workspace/vllm/vllm -type f -name \'*.so\' | wc -l)',
        'test "$vaws_so_count" -gt 0',
        'vaws_gpu_count=$(docker exec "$vaws_candidate" nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)',
        'vaws_gpu_models=$(docker exec "$vaws_candidate" nvidia-smi --query-gpu=name --format=csv,noheader | LC_ALL=C sort -u)',
        'test "$vaws_gpu_count" -gt 0',
        'if [ -n "$vaws_expected_gpus" ]; then test "$vaws_gpu_count" -eq "$vaws_expected_gpus"; fi',
        'if [ -n "$vaws_expected_gpu_model" ]; then test "$vaws_gpu_models" = "$vaws_expected_gpu_model"; fi',
        'vaws_gpu_models_csv=$(printf \'%s\\n\' "$vaws_gpu_models" | paste -sd, -)',
        "vaws_runtime_target=$(docker exec \"$vaws_candidate\" python3 -c 'import os, sysconfig; print(os.path.realpath(sysconfig.get_paths()[\"purelib\"] + \"/vllm\"))')",
        'test "$vaws_runtime_target" = /workspace/vllm/vllm',
        'docker exec -w /tmp "$vaws_candidate" python3 -c \'import vllm, vllm._C_stable_libtorch\' >/dev/null',
        'if docker exec "$vaws_candidate" test -f /workspace/vllm/vllm/third_party/deep_gemm/__init__.py; then',
        '  docker exec -w /tmp "$vaws_candidate" python3 -c \'import importlib, importlib.util; s=importlib.util.find_spec("vllm.third_party.deep_gemm"); assert s is not None and s.loader.__class__.__name__ == "SourceFileLoader"; m=importlib.import_module("vllm.third_party.deep_gemm"); assert all(callable(getattr(m, n, None)) for n in ("transform_sf_into_required_layout", "fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"))\'',
        'fi',
        'test "$(docker inspect "$vaws_candidate" --format \'{{range .Mounts}}{{if eq .Destination "/home/weight"}}{{.RW}}{{end}}{{end}}\')" = false',
        'vaws_rollback=',
        'if docker container inspect "$vaws_stable" >/dev/null 2>&1; then',
        '  vaws_rollback="${vaws_stable}-rollback-$(date -u +%Y%m%dT%H%M%SZ)"',
        '  docker stop "$vaws_stable" >/dev/null',
        '  docker rename "$vaws_stable" "$vaws_rollback"',
        "fi",
        'if ! docker rename "$vaws_candidate" "$vaws_stable"; then',
        '  if [ -n "$vaws_rollback" ]; then docker rename "$vaws_rollback" "$vaws_stable"; docker start "$vaws_stable" >/dev/null; fi',
        "  exit 75",
        "fi",
        'vaws_container_id=$(docker inspect "$vaws_stable" --format \'{{.Id}}\')',
        'printf \'status=ready\\ncontainer=%s\\ncontainer_id=%s\\nsource_image_id=%s\\nimage_id=%s\\nimage_build_commit=%s\\nrootfs_sha256=%s\\nworkspace=%s\\npython_sha256=%s\\nshared_libraries=%s\\nruntime_seed_files=%s\\ngpus=%s\\nrollback=%s\\n\' "$vaws_stable" "$vaws_container_id" "$vaws_image_id" "$vaws_runtime_image_id" "$vaws_observed_build_commit" "$vaws_observed_rootfs" "$vaws_workspace" "$vaws_python_sha" "$vaws_so_count" "$vaws_runtime_seed_count" "$vaws_gpu_count" "$vaws_rollback"',
        'printf \'gpu_models=%s\\n\' "$vaws_gpu_models_csv"',
    ]
    return "\n".join(lines)


def gpu_workspace_setup_command(args: argparse.Namespace, host: str) -> list[str]:
    command = [
        sys.executable,
        str(GPU_WORKSPACE_TOOL),
        "setup",
        "--host",
        host,
        "--user",
        args.user,
        "--port",
        str(args.port),
        "--container",
        args.container,
    ]
    if args.identity_file:
        command.extend(["--identity-file", str(args.identity_file.expanduser().resolve())])
    if args.ssh_config:
        command.extend(["--ssh-config", str(args.ssh_config.expanduser().resolve())])
    return command


def deploy_one(args: argparse.Namespace, host: str) -> dict[str, object]:
    started = time.monotonic()
    try:
        preflight(args, host)
        progress("deploy", host=host, status="started")
        process = subprocess.run(
            # Feed the script on stdin.  OpenSSH joins remote argv with spaces,
            # so passing a multiline script as an argv element lets the remote
            # login shell split it before ``bash -lc`` can parse it.
            ssh_base(args, host) + ["bash", "-s"],
            input=render_remote_script(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout_seconds,
        )
        fields: dict[str, str] = {}
        expected_fields = {
            "status",
            "container",
            "container_id",
            "image_id",
            "source_image_id",
            "image_build_commit",
            "rootfs_sha256",
            "workspace",
            "python_sha256",
            "shared_libraries",
            "runtime_seed_files",
            "gpus",
            "gpu_models",
            "rollback",
        }
        for line in process.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key in expected_fields:
                fields[key] = value
        ok = process.returncode == 0 and fields.get("status") in {"ready", "already_ready"}
        progress("deploy", host=host, status="ready" if ok else "failed")
        result = {
            "host": host,
            "status": fields.get("status", "failed") if ok else "failed",
            "returncode": process.returncode,
            "fields": fields,
            "stderr_tail": process.stderr[-2000:],
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if ok:
            workspace_command = gpu_workspace_setup_command(args, host)
            result["gpu_workspace_setup_argv"] = workspace_command
            result["gpu_workspace_setup_command"] = shlex.join(workspace_command)
        return result
    except Exception as exc:  # noqa: BLE001
        progress("deploy", host=host, status="failed")
        return {"host": host, "status": "failed", "error": str(exc), "duration_seconds": round(time.monotonic() - started, 3)}


def main() -> int:
    args = parse_args()
    if not args.approved_replace:
        raise SystemExit("refusing container replacement without --approved-replace")
    validate_sha256(args.image_tar_sha256, "--image-tar-sha256")
    validate_sha256(args.source_sha256, "--source-sha256")
    validate_sha256(args.python_sha256, "--python-sha256")
    validate_sha256(args.rootfs_sha256, "--rootfs-sha256")
    if not args.image_id.startswith("sha256:"):
        raise SystemExit("--image-id must be a pinned sha256 image ID")
    if args.expected_gpus is not None and args.expected_gpus < 1:
        raise SystemExit("--expected-gpus must be positive")
    hosts = list(dict.fromkeys(args.host))
    manifest = new_manifest(
        run_type="debug",
        workspace_snapshot={
            "source_head": args.source_head,
            "python_sha256": args.python_sha256.lower(),
        },
        environment={
            "workflow": "vllm-gpu-image-deploy",
            "image_id": args.image_id,
            "image_tar_sha256": args.image_tar_sha256.lower(),
        },
        topology={
            "hosts": hosts,
            "expected_gpus_per_host": args.expected_gpus,
            "expected_gpu_model": args.expected_gpu_model,
        },
        command=sys.argv,
    )
    run_dir = AGENTS_DIR.parent / ".vaws-local" / "vllm-gpu-image-deploy" / manifest["run_id"]
    manifest_path = run_dir / "manifest.json"
    manifest = transition_status(manifest, "running")
    write_manifest(manifest_path, manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        results = list(executor.map(lambda host: deploy_one(args, host), hosts))
    status = "ready" if all(item["status"] in {"ready", "already_ready"} for item in results) else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "image_id": args.image_id,
        "source_head": args.source_head,
        "python_sha256": args.python_sha256.lower(),
        "hosts": results,
        "gpu_tool_root": str(AGENTS_DIR / "scripts"),
        "gpu_tool_prefix": "gpu_",
        "manifest_path": str(manifest_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = add_artifact(
        manifest,
        name="deployment-result",
        kind="json",
        uri=str(result_path),
        sha256=sha256_file(result_path),
    )
    manifest = transition_status(manifest, "passed" if status == "ready" else "failed")
    write_manifest(manifest_path, manifest)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
