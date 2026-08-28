#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
LIB_DIR = ROOT / ".agents" / "lib"
SCRIPTS = ROOT / ".agents" / "skills" / "machine-management" / "scripts"
for value in (str(LIB_DIR), str(SCRIPTS)):
    if value not in sys.path:
        sys.path.insert(0, value)

import _workflow_common as workflow  # noqa: E402


class SshPreflightWorkflowTests(unittest.TestCase):
    def test_permission_failure_becomes_public_blocker(self) -> None:
        target = workflow.machine_ops.SshTarget(host="host", port=22, user="root")
        observed = {
            "status": "blocked",
            "category": "local_ssh_config_permissions",
            "knowledge_id": "machine-management-openssh-system-config-ownership",
            "message": "bad owner",
        }
        with mock.patch.object(workflow, "ssh_client_preflight", return_value=observed):
            result = workflow.ssh_client_preflight_blocker(target)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["action"], "ssh-client-preflight")
        self.assertEqual(result["knowledge_id"], observed["knowledge_id"])

    def test_ready_preflight_does_not_block(self) -> None:
        target = workflow.machine_ops.SshTarget(host="host", port=22, user="root")
        with mock.patch.object(workflow, "ssh_client_preflight", return_value={"status": "ready"}):
            self.assertIsNone(workflow.ssh_client_preflight_blocker(target))

    def test_host_execution_requirement_is_preserved_for_agent_rerun(self) -> None:
        target = workflow.machine_ops.SshTarget(host="host", port=22, user="root")
        observed = {
            "status": "blocked",
            "category": "ssh_host_execution_required",
            "host_execution_required": True,
            "message": "rerun outside sandbox",
        }
        with mock.patch.object(workflow, "ssh_client_preflight", return_value=observed):
            result = workflow.ssh_client_preflight_blocker(target)

        assert result is not None
        self.assertTrue(result["host_execution_required"])
        self.assertEqual(result["ssh_preflight"]["category"], "ssh_host_execution_required")


if __name__ == "__main__":
    unittest.main()
