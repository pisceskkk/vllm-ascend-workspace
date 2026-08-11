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

import vaws_ssh_preflight as preflight  # noqa: E402


def completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


class SshPreflightTests(unittest.TestCase):
    def test_success_is_read_only_and_ready(self) -> None:
        runner = mock.Mock(return_value=completed(0))
        with mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/ssh"):
            result = preflight.ssh_client_preflight("host", port=46001, runner=runner)

        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["mutated"])
        self.assertEqual(runner.call_args.args[0], ["ssh", "-G", "-p", "46001", "root@host"])

    def test_bad_owner_is_classified_with_knowledge_id(self) -> None:
        runner = mock.Mock(
            return_value=completed(
                255,
                "Bad owner or permissions on /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf\n",
            )
        )
        with (
            mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/ssh"),
            mock.patch.object(preflight, "inspect_reported_path", return_value=[{"path": "/etc/ssh"}]),
        ):
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["category"], "local_ssh_config_permissions")
        self.assertEqual(result["knowledge_id"], preflight.KNOWLEDGE_ID)
        self.assertTrue(result["repair_required"])
        self.assertFalse(result["auto_repaired"])

    def test_other_config_failure_is_not_misclassified_as_permissions(self) -> None:
        runner = mock.Mock(return_value=completed(255, "/etc/ssh/ssh_config line 3: Bad configuration option\n"))
        with mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/ssh"):
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["category"], "ssh_config_invalid")
        self.assertNotIn("knowledge_id", result)

    def test_missing_client_blocks_before_network_use(self) -> None:
        runner = mock.Mock()
        with mock.patch.object(preflight.shutil, "which", return_value=None):
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["category"], "ssh_client_missing")
        runner.assert_not_called()

    def test_timeout_is_structured(self) -> None:
        runner = mock.Mock(side_effect=subprocess.TimeoutExpired(cmd=["ssh"], timeout=1))
        with mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/ssh"):
            result = preflight.ssh_client_preflight("host", timeout=1, runner=runner)

        self.assertEqual(result["category"], "ssh_config_timeout")


if __name__ == "__main__":
    unittest.main()
