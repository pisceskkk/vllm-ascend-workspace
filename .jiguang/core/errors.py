"""Stable errors returned by the Jiguang MCP adapter."""


class JiguangError(RuntimeError):
    """Base error for the adapter."""


class JiguangPolicyError(JiguangError):
    """Raised when a request violates the account-owned-resource boundary."""


class JiguangTransportError(JiguangError):
    """Raised when the Windows host bridge cannot complete a request."""
