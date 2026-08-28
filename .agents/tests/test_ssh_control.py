#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import vaws_ssh_control as control  # noqa: E402


class SshControlPlaneTests(unittest.TestCase):
    def config(self, payload: dict) -> Path:
        temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temp.name).unlink, missing_ok=True)
        with temp:
            json.dump(payload, temp)
        return Path(temp.name)

    def test_native_mode_uses_local_ssh(self) -> None:
        path = self.config({"mode": "native"})
        with mock.patch.object(control.shutil, "which", return_value="/usr/bin/ssh"):
            resolved = control.resolve_ssh_control_plane(env={}, config_path=path)

        self.assertEqual(resolved.mode, "native")
        self.assertEqual(resolved.command_prefix, ("ssh",))

    def test_host_wsl_mode_builds_explicit_namespace_command(self) -> None:
        path = self.config(
            {"mode": "host-wsl", "distribution": "Ubuntu", "user": "developer"}
        )
        with mock.patch.object(control, "_find_wsl_executable", return_value="wsl.exe"):
            resolved = control.resolve_ssh_control_plane(env={}, config_path=path)

        self.assertEqual(resolved.mode, "host-wsl")
        self.assertEqual(
            resolved.command_prefix,
            ("wsl.exe", "-d", "Ubuntu", "-u", "developer", "--", "ssh"),
        )

    def test_auto_delegates_when_sandbox_exposes_overflow_uid(self) -> None:
        path = self.config({"mode": "auto", "distribution": "Ubuntu", "user": "developer"})
        with (
            mock.patch.object(control, "_sandbox_root_is_overflow_uid", return_value=True),
            mock.patch.object(control, "_find_wsl_executable", return_value="wsl.exe"),
        ):
            resolved = control.resolve_ssh_control_plane(
                env={"WSL_DISTRO_NAME": "Ubuntu"}, config_path=path
            )

        self.assertEqual(resolved.mode, "host-wsl")

    def test_host_execution_required_for_wsl_overflow_view(self) -> None:
        with mock.patch.object(control, "_sandbox_root_is_overflow_uid", return_value=True):
            self.assertTrue(
                control.ssh_host_execution_required(env={"WSL_DISTRO_NAME": "Ubuntu"})
            )

    def test_host_execution_not_required_outside_wsl(self) -> None:
        with mock.patch.object(control, "_sandbox_root_is_overflow_uid", return_value=True):
            self.assertFalse(control.ssh_host_execution_required(env={}))

    def test_environment_mode_overrides_file(self) -> None:
        path = self.config({"mode": "host-wsl", "distribution": "Ubuntu", "user": "developer"})
        with mock.patch.object(control.shutil, "which", return_value="/usr/bin/ssh"):
            resolved = control.resolve_ssh_control_plane(
                env={control.ENV_MODE: "native"}, config_path=path
            )

        self.assertEqual(resolved.mode, "native")
        self.assertEqual(resolved.source, control.ENV_MODE)

    def test_invalid_mode_is_rejected(self) -> None:
        path = self.config({"mode": "proxy"})
        with self.assertRaises(control.SshControlPlaneError):
            control.resolve_ssh_control_plane(env={}, config_path=path)


if __name__ == "__main__":
    unittest.main()
