#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vaws_github_auth import classify_github_auth  # noqa: E402


class FakeRunner:
    def __init__(self, auth: subprocess.CompletedProcess[str], api: subprocess.CompletedProcess[str]) -> None:
        self.results = [auth, api]
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return self.results.pop(0)


def completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class GithubAuthTests(unittest.TestCase):
    def test_missing_gh_is_not_an_auth_failure(self) -> None:
        runner = FakeRunner(completed(0), completed(0))
        with mock.patch("vaws_github_auth.shutil.which", return_value=None):
            result = classify_github_auth(network_context="unknown", runner=runner)

        self.assertEqual(result["auth_state"], "not_installed")
        self.assertFalse(result["logged_in"])
        self.assertEqual(runner.commands, [])

    def test_api_success_overrides_failed_auth_status(self) -> None:
        runner = FakeRunner(
            completed(1, stderr="the token in hosts.yml is invalid"),
            completed(0, stdout="octocat\n"),
        )
        result = classify_github_auth(network_context="enabled", gh_path="gh", runner=runner)

        self.assertEqual(result["auth_state"], "authenticated")
        self.assertEqual(result["network_state"], "reachable")
        self.assertEqual(result["user_login"], "octocat")

    def test_restricted_invalid_token_is_unverified(self) -> None:
        token = "secret-token-value"
        runner = FakeRunner(
            completed(1, stderr=f"the token {token} in hosts.yml is invalid"),
            completed(1, stderr="request failed"),
        )
        result = classify_github_auth(network_context="restricted", gh_path="gh", runner=runner)

        self.assertEqual(result["auth_state"], "unverified")
        self.assertEqual(result["retry_required"], "network_enabled")
        self.assertNotIn(token, str(result))

    def test_transport_error_is_not_auth_failure(self) -> None:
        runner = FakeRunner(
            completed(1, stderr="failed to connect to github.com"),
            completed(1, stderr="could not resolve host: api.github.com"),
        )
        result = classify_github_auth(network_context="enabled", gh_path="gh", runner=runner)

        self.assertEqual(result["auth_state"], "unverified")
        self.assertEqual(result["network_state"], "unavailable")

    def test_network_enabled_401_confirms_auth_failure(self) -> None:
        runner = FakeRunner(
            completed(1, stderr="HTTP 401: Bad credentials"),
            completed(1, stderr="HTTP 401: Bad credentials"),
        )
        result = classify_github_auth(network_context="enabled", gh_path="gh", runner=runner)

        self.assertEqual(result["auth_state"], "auth_failed")
        self.assertEqual(result["network_state"], "reachable")
        self.assertFalse(result["logged_in"])

    def test_unknown_context_never_confirms_auth_failure(self) -> None:
        runner = FakeRunner(
            completed(1, stderr="token is invalid"),
            completed(1, stderr="bad credentials"),
        )
        result = classify_github_auth(network_context="unknown", gh_path="gh", runner=runner)

        self.assertEqual(result["auth_state"], "unverified")
        self.assertIsNone(result["logged_in"])


if __name__ == "__main__":
    unittest.main()
