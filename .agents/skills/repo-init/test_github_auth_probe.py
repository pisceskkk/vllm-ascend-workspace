#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents" / "skills" / "repo-init" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_probe():
    spec = importlib.util.spec_from_file_location("_repo_init_auth_test", SCRIPTS / "repo_init_probe.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()


class RepoInitGithubAuthTests(unittest.TestCase):
    def test_probe_preserves_structured_unverified_state(self) -> None:
        classified = {
            "installed": True,
            "path": "/usr/bin/gh",
            "auth_state": "unverified",
            "network_state": "unavailable",
            "network_context": "restricted",
            "logged_in": None,
            "diagnostic_code": "transport_error",
            "retry_required": "network_enabled",
            "knowledge_id": "github-cli-network-enabled-auth-validation",
        }
        with (
            mock.patch.object(probe, "which", return_value="/usr/bin/gh"),
            mock.patch.object(probe, "classify_github_auth", return_value=classified),
            mock.patch.object(probe, "run", return_value=(0, "ssh", "")),
        ):
            result = probe.gh_login("restricted")

        self.assertEqual(result["auth_state"], "unverified")
        self.assertIsNone(result["logged_in"])
        self.assertEqual(result["git_protocol"], "ssh")
        self.assertNotIn("auth_status_stderr", result)

    def test_compact_payload_keeps_two_axis_classification(self) -> None:
        payload = {
            "platform": {},
            "repo_root": None,
            "workspace_profile": {},
            "workspace_identity": {},
            "gh": {
                "installed": True,
                "logged_in": None,
                "auth_state": "unverified",
                "network_state": "unknown",
                "network_context": "unknown",
                "diagnostic_code": "ambiguous",
                "retry_required": "network_enabled",
                "knowledge_id": "github-cli-network-enabled-auth-validation",
            },
            "gh_install_plan": {},
            "submodules": [],
            "repos": {},
            "forks": {},
        }

        compact = probe.compact_payload(payload)

        self.assertEqual(compact["gh"]["auth_state"], "unverified")
        self.assertEqual(compact["gh"]["retry_required"], "network_enabled")
        self.assertIsNone(compact["gh"]["logged_in"])


if __name__ == "__main__":
    unittest.main()
