#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import vaws_ssh_preflight as preflight  # noqa: E402
from vaws_ssh_control import SshControlPlane, SshControlPlaneError  # noqa: E402


def completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


class SshPreflightTests(unittest.TestCase):
    @contextmanager
    def native_control_plane(self):
        with (
            mock.patch.object(preflight, "ssh_host_execution_required", return_value=False),
            mock.patch.object(
                preflight,
                "resolve_ssh_control_plane",
                return_value=SshControlPlane(
                    mode="native",
                    command_prefix=("ssh",),
                    source="test",
                ),
            ),
        ):
            yield

    def test_success_is_read_only_and_ready(self) -> None:
        runner = mock.Mock(return_value=completed(0))
        with self.native_control_plane():
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
            self.native_control_plane(),
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
        with self.native_control_plane():
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["category"], "ssh_config_invalid")
        self.assertNotIn("knowledge_id", result)

    def test_invalid_control_plane_blocks_before_network_use(self) -> None:
        runner = mock.Mock()
        with (
            mock.patch.object(preflight, "ssh_host_execution_required", return_value=False),
            mock.patch.object(
                preflight,
                "resolve_ssh_control_plane",
                side_effect=SshControlPlaneError("wsl.exe was not found"),
            ),
        ):
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["category"], "ssh_control_plane_invalid")
        runner.assert_not_called()

    def test_missing_selected_client_is_structured(self) -> None:
        runner = mock.Mock(side_effect=FileNotFoundError)
        with self.native_control_plane():
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["category"], "ssh_client_missing")

    def test_timeout_is_structured(self) -> None:
        runner = mock.Mock(side_effect=subprocess.TimeoutExpired(cmd=["ssh"], timeout=1))
        with self.native_control_plane():
            result = preflight.ssh_client_preflight("host", timeout=1, runner=runner)

        self.assertEqual(result["category"], "ssh_config_timeout")

    def test_host_wsl_control_plane_is_used_for_preflight(self) -> None:
        runner = mock.Mock(return_value=completed(0))
        delegated = SshControlPlane(
            mode="host-wsl",
            command_prefix=("wsl.exe", "-d", "Ubuntu", "-u", "developer", "--", "ssh"),
            source="test",
            distribution="Ubuntu",
            user="developer",
        )
        with (
            mock.patch.object(preflight, "ssh_host_execution_required", return_value=False),
            mock.patch.object(preflight, "resolve_ssh_control_plane", return_value=delegated),
        ):
            result = preflight.ssh_client_preflight("host", port=46001, runner=runner)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["ssh_control_plane"]["mode"], "host-wsl")
        self.assertEqual(
            runner.call_args.args[0],
            [
                "wsl.exe", "-d", "Ubuntu", "-u", "developer", "--", "ssh",
                "-G", "-p", "46001", "root@host",
            ],
        )

    def test_wsl_overflow_view_requires_host_execution_before_ssh(self) -> None:
        runner = mock.Mock()
        with mock.patch.object(preflight, "ssh_host_execution_required", return_value=True):
            result = preflight.ssh_client_preflight("host", runner=runner)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["category"], "ssh_host_execution_required")
        self.assertTrue(result["host_execution_required"])
        self.assertFalse(result["repair_required"])
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
