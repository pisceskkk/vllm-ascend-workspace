"""Invoke the allowlisted HTTP bridge in the Windows host network plane."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

from .errors import JiguangTransportError
from .redaction import redact


class Transport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]: ...


class HostProcessTransport:
    def __init__(self, bridge_path: Path | None = None) -> None:
        self.bridge_path = bridge_path or Path(__file__).resolve().parents[1] / "host" / "jiguang_http.ps1"

    @staticmethod
    def _powershell() -> str:
        override = os.environ.get("JIGUANG_POWERSHELL")
        if override:
            return override
        for name in ("pwsh.exe", "powershell.exe", "pwsh", "powershell"):
            found = shutil.which(name)
            if found:
                return found
        raise JiguangTransportError("Windows PowerShell host executable was not found")

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        source = self.bridge_path.read_text(encoding="utf-8")
        encoded = base64.b64encode(source.encode("utf-16le")).decode("ascii")
        request = {
            "method": method,
            "path": path,
            "query": query or {},
            "body": body,
            "timeout_seconds": timeout_seconds,
            "credential_target": os.environ.get(
                "JIGUANG_CREDENTIAL_TARGET", "Codex:Jiguang:AccessToken"
            ),
        }
        host_environment = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            host_environment.pop(key, None)
        host_environment["NO_PROXY"] = "jiguang.ascend.huawei.com"
        host_environment["no_proxy"] = "jiguang.ascend.huawei.com"
        try:
            completed = subprocess.run(
                [self._powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 15,
                check=False,
                env=host_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JiguangTransportError(f"Windows host bridge failed: {exc}") from exc
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise JiguangTransportError(f"invalid host bridge response: {detail}") from exc
        safe = redact(payload)
        if completed.returncode != 0 or not safe.get("ok"):
            message = safe.get("error") or f"host bridge exited with {completed.returncode}"
            raise JiguangTransportError(str(message))
        return safe
