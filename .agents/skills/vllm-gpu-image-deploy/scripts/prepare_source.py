#!/usr/bin/env python3
"""Package a local vLLM Python tree for an image-runtime overlay."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import tarfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--source-head", help="record the source Git HEAD; discovered when omitted")
    parser.add_argument(
        "--image-build-commit",
        required=True,
        help="exact VLLM_BUILD_COMMIT of the runtime image",
    )
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(package: pathlib.Path) -> list[pathlib.Path]:
    files = []
    for path in package.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"source-tree symlinks are not supported: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".so"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(package).as_posix().encode())


def checksum_lines(package: pathlib.Path, files: list[pathlib.Path], *, python_only: bool) -> list[str]:
    lines = []
    for path in files:
        if python_only and path.suffix != ".py":
            continue
        relative = pathlib.PurePosixPath("vllm") / path.relative_to(package).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}\n")
    return lines


def lines_sha256(lines: list[str]) -> str:
    aggregate = hashlib.sha256()
    for line in lines:
        aggregate.update(line.encode())
    return aggregate.hexdigest()


def discover_head(repo: pathlib.Path) -> str | None:
    head = repo / ".git"
    if not head.exists():
        return None
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def deleted_package_paths(repo: pathlib.Path, image_build_commit: str) -> list[str]:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--diff-filter=D",
            "--name-only",
            image_build_commit,
            "--",
            "vllm",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise SystemExit(
            "--image-build-commit must resolve in the local vLLM repo: "
            f"{process.stderr.strip()}"
        )
    deleted = []
    for raw in process.stdout.splitlines():
        path = pathlib.PurePosixPath(raw)
        if not path.parts or path.parts[0] != "vllm" or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe deleted package path from git diff: {raw}")
        deleted.append(path.as_posix())
    return sorted(set(deleted), key=str.encode)


def main() -> int:
    args = parse_args()
    repo = args.vllm_repo.expanduser().resolve()
    package = repo / "vllm"
    if not (package / "__init__.py").is_file():
        raise SystemExit(f"not a vLLM source repo: {repo}")
    output = args.output.expanduser().resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise SystemExit("--output must end with .tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = included_files(package)
    python_lines = checksum_lines(package, files, python_only=True)
    source_lines = checksum_lines(package, files, python_only=False)
    deleted = deleted_package_paths(repo, args.image_build_commit)
    source_head = args.source_head or discover_head(repo)
    if not source_head:
        raise SystemExit("unable to discover source HEAD; pass --source-head explicitly")
    metadata = {
        "schema_version": 2,
        "source_head": source_head,
        "image_build_commit": args.image_build_commit,
        "deleted_paths": deleted,
    }
    with tarfile.open(output, mode="w:gz", compresslevel=6) as archive:
        for path in files:
            relative = pathlib.PurePosixPath("vllm") / path.relative_to(package).as_posix()
            archive.add(path, arcname=str(relative), recursive=False)
        for name, content in (
            (".vaws-python-sha256sums", "".join(python_lines)),
            (".vaws-source-sha256sums", "".join(source_lines)),
            (".vaws-deleted-paths", "".join(f"{path}\n" for path in deleted)),
            (".vaws-source-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n"),
        ):
            info = tarfile.TarInfo(name)
            payload_bytes = content.encode()
            info.size = len(payload_bytes)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload_bytes))
    payload = {
        "schema_version": 2,
        "status": "ok",
        "vllm_repo": str(repo),
        "source_head": metadata["source_head"],
        "image_build_commit": args.image_build_commit,
        "deleted_path_count": len(deleted),
        "output": str(output),
        "file_count": len(files),
        "size_bytes": output.stat().st_size,
        "source_sha256": sha256_file(output),
        "python_sha256": lines_sha256(python_lines),
        "source_files_sha256": lines_sha256(source_lines),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
