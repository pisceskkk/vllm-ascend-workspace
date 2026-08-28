#!/usr/bin/env python3
"""Minimal stdio MCP server for Jiguang account-owned resources."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.redaction import redact  # noqa: E402
from mcp.tools import call_tool, list_tools  # noqa: E402


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _send(payload: dict[str, Any], framed: bool) -> None:
    encoded = _encode(payload)
    if framed:
        sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
    else:
        sys.stdout.write(encoded.decode("utf-8") + "\n")
        sys.stdout.flush()


def _handle(message: dict[str, Any], framed: bool) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    if request_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return
    try:
        if method == "initialize":
            value = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "jiguang", "version": "0.1.0"},
            }
        elif method == "tools/list":
            value = {"tools": list_tools()}
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("tools/call requires a string name and object arguments")
            payload = call_tool(name, arguments)
            value = {
                "content": [{"type": "text", "text": payload["text"]}],
                "structuredContent": payload["result"],
                "isError": False,
            }
        else:
            _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method}"}}, framed)
            return
        _send({"jsonrpc": "2.0", "id": request_id, "result": value}, framed)
    except Exception as exc:  # noqa: BLE001
        safe = redact({"type": type(exc).__name__, "message": str(exc)})
        _send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": safe["message"], "data": safe}}, framed)


def _framed() -> int:
    while True:
        headers: dict[str, str] = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return 0
            if line in {b"\r\n", b"\n"}:
                break
            key, _, value = line.decode("ascii", errors="replace").partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length <= 0:
            continue
        message = json.loads(sys.stdin.buffer.read(length).decode("utf-8"))
        if isinstance(message, dict):
            _handle(message, True)


def _lines() -> int:
    for line in sys.stdin:
        if line.strip():
            message = json.loads(line)
            if isinstance(message, dict):
                _handle(message, False)
    return 0


def main() -> int:
    try:
        peeked = sys.stdin.buffer.peek(16)
    except AttributeError:
        peeked = b""
    return _framed() if peeked.startswith(b"Content-Length:") else _lines()


if __name__ == "__main__":
    raise SystemExit(main())
