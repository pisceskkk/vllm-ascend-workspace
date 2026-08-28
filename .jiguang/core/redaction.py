"""Fail-closed redaction for platform responses and errors."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:authorization|cookie|token|secret|password|passwd|credential|private_?key|cftk)(?:_|$)",
    re.IGNORECASE,
)
URL_USERINFO_RE = re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@")
BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*[:=]\s*)?bearer\s+)[^\s,;\"']+")
ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:password|passwd|secret|access_token|refresh_token|cookie)\s*[:=]\s*)[^\s,;\"']+"
)


def _redact_text(value: str) -> str:
    value = URL_USERINFO_RE.sub(r"\1<redacted>@", value)
    value = BEARER_RE.sub(r"\1<redacted>", value)
    return ASSIGNMENT_RE.sub(r"\1<redacted>", value)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if SENSITIVE_KEY_RE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value
