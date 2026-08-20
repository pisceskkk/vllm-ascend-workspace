#!/usr/bin/env python3
"""Synchronize only local vLLM Python source into an image-backed GPU workspace."""

from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_gpu_common import (  # noqa: E402
    GpuToolError,
    add_target_arguments,
    print_json,
    progress,
    remote_bash,
    remote_upload_bytes,
    require_success,
    shell_assignment,
    target_from_args,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_files(repo: pathlib.Path) -> list[pathlib.Path]:
    package = repo / "vllm"
    if not (package / "__init__.py").is_file():
        raise GpuToolError(f"not a vLLM repository: {repo}")
    files = []
    for path in package.rglob("*"):
        if path.is_symlink():
            raise GpuToolError(f"source symlinks are not supported: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".so"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(package).as_posix().encode())


def build_archive(repo: pathlib.Path) -> tuple[bytes, str, int]:
    package = repo / "vllm"
    files = source_files(repo)
    manifest_lines = []
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=6) as archive:
        for path in files:
            relative = (
                pathlib.PurePosixPath("vllm") / path.relative_to(package).as_posix()
            )
            payload = path.read_bytes()
            manifest_lines.append(f"{sha256_bytes(payload)}  {relative}\n")
            info = tarfile.TarInfo(str(relative))
            info.size = len(payload)
            info.mode = path.stat().st_mode & 0o777
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        manifest = "".join(manifest_lines).encode()
        info = tarfile.TarInfo(".vaws-source-sha256sums")
        info.size = len(manifest)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest))
    archive_bytes = stream.getvalue()
    return archive_bytes, sha256_bytes(archive_bytes), len(files)


def render_remote_script(target, archive_sha: str, remote_archive: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            shell_assignment("vaws_container", target.container),
            shell_assignment("vaws_expected_archive_sha", archive_sha),
            shell_assignment("vaws_archive", remote_archive),
            'vaws_workspace=$(docker inspect "$vaws_container" --format \'{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}\')',
            'test -n "$vaws_workspace"',
            'vaws_root="$vaws_workspace/vllm"',
            'test -d "$vaws_root/vllm"',
            'vaws_incoming="$vaws_root/.vaws-gpu-sync-incoming-$$"',
            'vaws_stage="$vaws_root/.vaws-gpu-sync-stage-$$"',
            'trap \'rm -rf -- "$vaws_incoming" "$vaws_stage" "$vaws_archive"\' EXIT',
            'mkdir -p "$vaws_incoming"',
            'test "$(sha256sum "$vaws_archive" | cut -d " " -f1)" = "$vaws_expected_archive_sha"',
            'tar -xzf "$vaws_archive" -C "$vaws_incoming"',
            '(cd "$vaws_incoming" && sha256sum -c .vaws-source-sha256sums >/dev/null)',
            'cp -a "$vaws_root/vllm" "$vaws_stage"',
            'if [ -f "$vaws_root/.vaws-source-sha256sums" ]; then',
            "  while read -r vaws_hash vaws_path; do",
            '    case "$vaws_path" in vllm/*) ;; *) printf "unsafe prior source path: %s\\n" "$vaws_path" >&2; exit 66 ;; esac',
            "    vaws_rel=${vaws_path#vllm/}",
            '    case "/$vaws_rel/" in *"/../"*|*"/./"*) printf "unsafe prior source path: %s\\n" "$vaws_path" >&2; exit 66 ;; esac',
            '    rm -f -- "$vaws_stage/$vaws_rel"',
            '  done < "$vaws_root/.vaws-source-sha256sums"',
            "fi",
            'cp -a "$vaws_incoming/vllm"/. "$vaws_stage"/',
            'vaws_rollback="$vaws_root/vllm.rollback-$(date -u +%Y%m%dT%H%M%SZ)"',
            'mv "$vaws_root/vllm" "$vaws_rollback"',
            'mv "$vaws_stage" "$vaws_root/vllm"',
            'if ! docker exec -w /tmp "$vaws_container" python3 -c \'import os,vllm; assert os.path.realpath(vllm.__file__).startswith("/workspace/vllm/vllm/")\'; then',
            '  mv "$vaws_root/vllm" "$vaws_root/vllm.failed-$(date -u +%Y%m%dT%H%M%SZ)"',
            '  mv "$vaws_rollback" "$vaws_root/vllm"',
            "  exit 75",
            "fi",
            'cp "$vaws_incoming/.vaws-source-sha256sums" "$vaws_root/.vaws-source-sha256sums"',
            '(cd "$vaws_root" && sha256sum -c .vaws-source-sha256sums >/dev/null)',
            'printf \'status=ready\\nworkspace=%s\\nrollback=%s\\narchive_sha256=%s\\n\' "$vaws_root" "$vaws_rollback" "$vaws_expected_archive_sha"',
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_target_arguments(parser)
    parser.add_argument("--vllm-repo", type=pathlib.Path, default=ROOT / "vllm")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--archive-only", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        repo = args.vllm_repo.expanduser().resolve()
        archive, archive_sha, file_count = build_archive(repo)
        if args.archive_only:
            output = args.archive_only.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive)
            print_json(
                {
                    "status": "prepared",
                    "archive": str(output),
                    "archive_sha256": archive_sha,
                    "file_count": file_count,
                }
            )
            return 0
        target = target_from_args(args)
        progress(
            "gpu_code_parity",
            "sync",
            host=target.host,
            container=target.container,
            files=file_count,
        )
        remote_archive = f"/tmp/vaws-gpu-code-parity-{archive_sha}.tar.gz"
        remote_upload_bytes(
            target, remote_archive, archive, timeout=args.timeout_seconds
        )
        process = remote_bash(
            target,
            render_remote_script(target, archive_sha, remote_archive),
            timeout=args.timeout_seconds,
        )
        require_success(process, "GPU vLLM source sync")
        fields = dict(
            line.split("=", 1)
            for line in process.stdout.decode(errors="replace").splitlines()
            if "=" in line
        )
        print_json(
            {
                "status": "ready",
                "target": target.public_dict(),
                "file_count": file_count,
                **fields,
            }
        )
        return 0
    except GpuToolError as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
