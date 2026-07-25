from __future__ import annotations

import base64
import json
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .endpoint import Endpoint


REMOTE_PYTHON_RUNNER = (
    "import io,json,sys;"
    "envelope=json.load(sys.stdin);"
    "sys.stdin=io.StringIO(json.dumps(envelope['payload'],ensure_ascii=False));"
    "exec(compile(envelope['code'],'<remote-dev>','exec'),{'__name__':'__main__'})"
)
SAFE_DIRECT_STDIN_BYTES = 768
SAFE_DIRECT_COMMAND_BYTES = 768
STAGED_CHUNK_BYTES = 720


@dataclass
class RemoteCompleted:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def ssh_base_cmd(endpoint: Endpoint) -> list[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-o",
        f"ConnectTimeout={max(1, int(endpoint.connect_timeout_ms / 1000))}",
    ]
    if endpoint.identity_file:
        cmd.extend(["-i", endpoint.identity_file])
    cmd.extend(["-p", str(endpoint.port), endpoint.destination()])
    return cmd


def run_script(endpoint: Endpoint, script: str, *, timeout_ms: int | None = None) -> RemoteCompleted:
    timeout = None if timeout_ms is None else timeout_ms / 1000
    try:
        proc = subprocess.run(
            [*ssh_base_cmd(endpoint), "bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return RemoteCompleted(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_stream(exc.stdout)
        stderr = _decode_stream(exc.stderr)
        return RemoteCompleted(None, stdout, stderr, timed_out=True)


def run_bytes(
    endpoint: Endpoint,
    remote_command: str,
    *,
    stdin: bytes | None = None,
    timeout_ms: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    encoded_command = f"bash -c {shlex.quote(remote_command)}".encode("utf-8")
    stage_stdin = stdin is not None and len(stdin) > SAFE_DIRECT_STDIN_BYTES
    stage_command = len(encoded_command) > SAFE_DIRECT_COMMAND_BYTES
    if not stage_stdin and not stage_command:
        return _run_bytes_once(
            endpoint,
            remote_command,
            stdin=stdin,
            timeout_ms=timeout_ms,
        )

    deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000
    remote_input: str | None = None
    remote_script: str | None = None
    try:
        if stage_stdin:
            assert stdin is not None
            remote_input, failure = _stage_remote_bytes(
                endpoint,
                stdin,
                kind="input",
                deadline=deadline,
            )
            if failure is not None:
                return failure
        if stage_command:
            remote_script, failure = _stage_remote_bytes(
                endpoint,
                remote_command.encode("utf-8"),
                kind="script",
                deadline=deadline,
            )
            if failure is not None:
                return failure
            execution = f"bash {shlex.quote(remote_script)}"
        else:
            execution = remote_command
        if remote_input is None:
            return _run_bytes_once(
                endpoint,
                execution,
                stdin=stdin,
                timeout_ms=_remaining_timeout_ms(deadline),
            )
        return _run_bytes_once(
            endpoint,
            (
                f"bash -c {shlex.quote(execution)} "
                f"< {shlex.quote(remote_input)}"
            ),
            timeout_ms=_remaining_timeout_ms(deadline),
        )
    finally:
        staged_paths = [
            path
            for path in (remote_input, remote_script)
            if path is not None
        ]
        if staged_paths:
            _run_bytes_once(
                endpoint,
                "rm -f " + " ".join(shlex.quote(path) for path in staged_paths),
                timeout_ms=_remaining_timeout_ms(deadline, cleanup=True),
            )


def _stage_remote_bytes(
    endpoint: Endpoint,
    payload: bytes,
    *,
    kind: str,
    deadline: float | None,
) -> tuple[str, subprocess.CompletedProcess[bytes] | None]:
    remote_path = f"/tmp/.remote-dev-{kind}-{uuid.uuid4().hex}.bin"
    init = _run_bytes_once(
        endpoint,
        f"umask 077; : > {shlex.quote(remote_path)}",
        timeout_ms=_remaining_timeout_ms(deadline),
    )
    if init.returncode != 0:
        return remote_path, init
    for offset in range(0, len(payload), STAGED_CHUNK_BYTES):
        encoded = base64.b64encode(
            payload[offset : offset + STAGED_CHUNK_BYTES]
        ).decode("ascii")
        append = _run_bytes_once(
            endpoint,
            (
                f"printf %s {shlex.quote(encoded)} | base64 -d >> "
                f"{shlex.quote(remote_path)}"
            ),
            timeout_ms=_remaining_timeout_ms(deadline),
        )
        if append.returncode != 0:
            return remote_path, append
    return remote_path, None


def _remaining_timeout_ms(
    deadline: float | None,
    *,
    cleanup: bool = False,
) -> int | None:
    if deadline is None:
        return None
    remaining = int(max(0.0, deadline - time.monotonic()) * 1000)
    if cleanup:
        return max(1000, min(remaining, 10000))
    if remaining <= 0:
        raise subprocess.TimeoutExpired("remote staged input", 0)
    return remaining


def _run_bytes_once(
    endpoint: Endpoint,
    remote_command: str,
    *,
    stdin: bytes | None = None,
    timeout_ms: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    timeout = None if timeout_ms is None else timeout_ms / 1000
    return subprocess.run(
        [*ssh_base_cmd(endpoint), f"bash -c {shlex.quote(remote_command)}"],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_remote_python(
    endpoint: Endpoint,
    code: str,
    payload: dict[str, Any],
    *,
    timeout_ms: int | None = None,
    cwd: str | None = None,
    runtime_env: bool = False,
) -> dict[str, Any]:
    envelope = json.dumps(
        {"code": code, "payload": payload},
        ensure_ascii=False,
    ).encode("utf-8")
    if runtime_env or cwd is not None:
        command_parts = ["set -u"]
        if runtime_env:
            command_parts.append(
                "if [ -f /etc/profile.d/vaws-ascend-env.sh ]; then "
                "set +u; . /etc/profile.d/vaws-ascend-env.sh; set -u; fi"
            )
        if cwd is not None:
            command_parts.append(f"cd {shlex.quote(cwd)} || exit 70")
        if runtime_env:
            command_parts.extend(
                [
                    "remote_dev_python=\"$(for candidate in /usr/local/python*/bin/python3; do "
                    "[ -x \"$candidate\" ] && printf '%s\\n' \"$candidate\"; "
                    "done | sort -V | tail -n 1)\"",
                    "if [ -z \"$remote_dev_python\" ]; then "
                    "remote_dev_python=\"$(command -v python3)\"; fi",
                    f"exec \"$remote_dev_python\" -c {shlex.quote(REMOTE_PYTHON_RUNNER)}",
                ]
            )
        else:
            command_parts.append(f"exec python3 -c {shlex.quote(REMOTE_PYTHON_RUNNER)}")
        remote_command = f"bash -c {shlex.quote('; '.join(command_parts))}"
    else:
        remote_command = f"python3 -c {shlex.quote(REMOTE_PYTHON_RUNNER)}"
    try:
        raw_proc = run_bytes(
            endpoint,
            remote_command,
            stdin=envelope,
            timeout_ms=timeout_ms,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "error": f"remote python timed out after {timeout_ms} ms",
            "stdout_tail": _decode_stream(exc.stdout)[-4000:],
            "stderr_tail": _decode_stream(exc.stderr)[-4000:],
        }
    stdout = _decode_stream(raw_proc.stdout)
    stderr = _decode_stream(raw_proc.stderr)
    if raw_proc.returncode != 0:
        return {
            "status": "failed",
            "error": "remote python failed",
            "exit_code": raw_proc.returncode,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    data = _last_json_object(stdout)
    if data is None:
        return {
            "status": "failed",
            "error": "remote python returned no JSON object",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    return data


def _last_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _decode_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
