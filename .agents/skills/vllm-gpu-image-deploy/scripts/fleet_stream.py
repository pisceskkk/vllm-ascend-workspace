#!/usr/bin/env python3
"""Hash and stream one local file to multiple SSH hosts without scp/rsync."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

AGENTS_DIR = pathlib.Path(__file__).resolve().parents[3]
if str(AGENTS_DIR / "lib") not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR / "lib"))

from vaws_ssh_control import ssh_command_prefix  # noqa: E402

CHUNK_SIZE = 4 * 1024 * 1024


@dataclass
class Sink:
    host: str
    process: subprocess.Popen[bytes]
    stderr_path: pathlib.Path
    failed_early: bool = False


def progress(phase: str, **fields: object) -> None:
    payload = {"phase": phase, **fields}
    print(
        f"__VAWS_VLLM_GPU_IMAGE_TRANSFER_PROGRESS__={json.dumps(payload, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=pathlib.Path)
    parser.add_argument("--host", action="append", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=pathlib.Path)
    parser.add_argument("--ssh-config", type=pathlib.Path, help="explicit OpenSSH config, e.g. /dev/null")
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--sha256", help="reuse an already computed local SHA256")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


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
        raise RuntimeError(f"ssh -G failed for {host}: {config.stderr.strip()}")
    probe = subprocess.run(
        ssh_base(args, host) + ["printf", "ok"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )
    if probe.returncode != 0 or probe.stdout != "ok":
        raise RuntimeError(f"key SSH failed for {host}: {probe.stderr.strip()}")


def hash_file(path: pathlib.Path) -> str:
    total = path.stat().st_size
    observed = 0
    digest = hashlib.sha256()
    last = time.monotonic()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            observed += len(chunk)
            now = time.monotonic()
            if now - last >= 5:
                progress("hash", bytes=observed, total_bytes=total, percent=round(observed * 100 / total, 2))
                last = now
    progress("hash", bytes=observed, total_bytes=total, percent=100.0)
    return digest.hexdigest()


def remote_prepare(remote_path: str) -> str:
    parent = shlex.quote(str(pathlib.PurePosixPath(remote_path).parent))
    temporary = shlex.quote(remote_path + ".part")
    return "\n".join(
        [
            "set -eu",
            f"mkdir -p {parent}",
            f"rm -f -- {temporary}",
        ]
    )


def remote_transfer_command(args: argparse.Namespace, host: str, remote_path: str) -> list[str]:
    temporary = shlex.quote(remote_path + ".part")
    return ssh_base(args, host) + [f"dd of={temporary} bs=4M status=none"]


def remote_finalize(remote_path: str, expected: str) -> str:
    final = shlex.quote(remote_path)
    temporary = shlex.quote(remote_path + ".part")
    return "\n".join(
        [
            "set -eu",
            f"vaws_observed=$(sha256sum {temporary} | cut -d ' ' -f1)",
            f"test \"$vaws_observed\" = {shlex.quote(expected)}",
            f"mv -f -- {temporary} {final}",
            'printf \'%s\\n\' "$vaws_observed"',
        ]
    )


def main() -> int:
    args = parse_args()
    local_file = args.file.expanduser().resolve()
    if not local_file.is_file():
        raise SystemExit(f"local file does not exist: {local_file}")
    if not pathlib.PurePosixPath(args.remote_path).is_absolute():
        raise SystemExit("--remote-path must be absolute")
    hosts = list(dict.fromkeys(args.host))
    for host in hosts:
        progress("preflight", host=host)
        preflight(args, host)

    expected = args.sha256.lower() if args.sha256 else hash_file(local_file)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise SystemExit("--sha256 must contain 64 lowercase hexadecimal characters")

    total = local_file.stat().st_size
    started = time.monotonic()
    transferred = 0
    last = started
    results = []
    with tempfile.TemporaryDirectory(prefix="vllm-gpu-image-stream-") as temp_dir_text:
        temp_dir = pathlib.Path(temp_dir_text)
        sinks: list[Sink] = []
        handles = []
        try:
            for index, host in enumerate(hosts):
                stderr_path = temp_dir / f"{index}.stderr"
                stderr_handle = stderr_path.open("wb")
                handles.append(stderr_handle)
                prepare = subprocess.run(
                    ssh_base(args, host) + ["sh", "-s"],
                    input=remote_prepare(args.remote_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                if prepare.returncode != 0:
                    raise RuntimeError(
                        f"remote transfer preparation failed for {host}: {prepare.stderr.strip()}"
                    )
                process = subprocess.Popen(
                    remote_transfer_command(args, host, args.remote_path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_handle,
                    bufsize=0,
                )
                sinks.append(Sink(host=host, process=process, stderr_path=stderr_path))

            with local_file.open("rb") as source:
                while chunk := source.read(CHUNK_SIZE):
                    if time.monotonic() - started > args.timeout_seconds:
                        raise TimeoutError(f"fleet transfer exceeded {args.timeout_seconds} seconds")
                    live = 0
                    for sink in sinks:
                        if sink.failed_early:
                            continue
                        if sink.process.poll() is not None:
                            sink.failed_early = True
                            continue
                        try:
                            assert sink.process.stdin is not None
                            sink.process.stdin.write(chunk)
                            live += 1
                        except BrokenPipeError:
                            sink.failed_early = True
                    if live == 0:
                        details = []
                        for handle in handles:
                            handle.flush()
                        for sink in sinks:
                            details.append(
                                {
                                    "host": sink.host,
                                    "returncode": sink.process.poll(),
                                    "stderr_tail": sink.stderr_path.read_text(
                                        encoding="utf-8", errors="replace"
                                    )[-1000:],
                                }
                            )
                        raise RuntimeError(
                            "all SSH transfer sinks failed: "
                            + json.dumps(details, ensure_ascii=False, sort_keys=True)
                        )
                    transferred += len(chunk)
                    now = time.monotonic()
                    if now - last >= 5:
                        progress(
                            "transfer",
                            bytes=transferred,
                            total_bytes=total,
                            percent=round(transferred * 100 / total, 2),
                            live_sinks=live,
                        )
                        last = now

            for sink in sinks:
                if sink.process.stdin is not None:
                    try:
                        sink.process.stdin.close()
                    except BrokenPipeError:
                        sink.failed_early = True
                    sink.process.stdin = None
            for sink in sinks:
                remaining = max(1, int(args.timeout_seconds - (time.monotonic() - started)))
                try:
                    sink.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    sink.process.kill()
                    sink.process.wait()
                transfer_stdout = sink.process.stdout.read().decode("utf-8", errors="replace").strip() if sink.process.stdout else ""
                stderr = sink.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                stdout = transfer_stdout
                returncode = sink.process.returncode
                if returncode == 0:
                    finalize = subprocess.run(
                        ssh_base(args, sink.host) + ["sh", "-s"],
                        input=remote_finalize(args.remote_path, expected),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=max(60, remaining),
                    )
                    returncode = finalize.returncode
                    stdout = finalize.stdout.strip()
                    stderr = "\n".join(item for item in (stderr, finalize.stderr.strip()) if item)
                ok = returncode == 0 and stdout.splitlines()[-1:] == [expected]
                results.append(
                    {
                        "host": sink.host,
                        "status": "ok" if ok else "failed",
                        "returncode": returncode,
                        "remote_sha256": stdout.splitlines()[-1] if stdout else None,
                        "stderr_tail": stderr[-1000:],
                    }
                )
        finally:
            for sink in sinks:
                if sink.process.poll() is None:
                    sink.process.terminate()
            for handle in handles:
                handle.close()

    status = "ok" if all(item["status"] == "ok" for item in results) else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "local_file": str(local_file),
        "size_bytes": total,
        "sha256": expected,
        "remote_path": args.remote_path,
        "duration_seconds": round(time.monotonic() - started, 3),
        "hosts": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
