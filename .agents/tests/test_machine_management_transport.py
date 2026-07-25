from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / ".agents"
    / "skills"
    / "machine-management"
    / "scripts"
    / "manage_machine.py"
)


def load_manage_machine():
    name = "_machine_management_transport_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manage_machine = load_manage_machine()


class MachineManagementTransportTests(unittest.TestCase):
    def test_bootstrap_persists_session_device_visibility(self) -> None:
        script = manage_machine.render_bootstrap_host_script()

        self.assertIn('visible_devices="${11:-}"', script)
        self.assertIn('ASCEND_RT_VISIBLE_DEVICES=$visible_devices', script)
        self.assertIn('SetEnv ASCEND_RT_VISIBLE_DEVICES=%s', script)
        self.assertIn('"visible_devices": [int(item)', script)

    def test_staged_script_preserves_business_arguments(self) -> None:
        command = manage_machine.remote_script_command(
            "/tmp/bootstrap.sh",
            ["image-config", "container-name"],
        )

        self.assertEqual(
            command,
            "bash /tmp/bootstrap.sh image-config container-name",
        )

    def test_direct_script_uses_option_terminator(self) -> None:
        command = manage_machine.remote_script_command(
            None,
            ["image-config"],
        )

        self.assertEqual(command, "bash -s -- image-config")

    def test_large_script_is_staged_in_bounded_remote_commands(self) -> None:
        target = manage_machine.SshTarget(host="192.0.2.1")
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="",
                stderr="",
            )

        with (
            mock.patch.object(manage_machine.subprocess, "run", fake_run),
            mock.patch.object(
                manage_machine.uuid,
                "uuid4",
                return_value=mock.Mock(hex="abc123"),
            ),
        ):
            remote_path = manage_machine.stage_remote_script(
                target,
                b"x" * 1500,
                batch_mode=True,
            )

        self.assertEqual(remote_path, "/tmp/.vaws-remote-script-abc123.sh")
        self.assertEqual(len(calls), 4)
        commands = [args[-1] for args in calls]
        self.assertIn(": > /tmp/.vaws-remote-script-abc123.sh", commands[0])
        self.assertTrue(all("base64 -d" in command for command in commands[1:]))
        self.assertTrue(
            all(
                len(command) < 1200
                for command in commands
            )
        )

    def test_failed_chunk_removes_partial_remote_script(self) -> None:
        target = manage_machine.SshTarget(host="192.0.2.1")
        results = [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", "link reset"),
            subprocess.CompletedProcess([], 0, "", ""),
        ]

        with (
            mock.patch.object(
                manage_machine,
                "run_remote_command_once",
                side_effect=results,
            ) as run_once,
            mock.patch.object(
                manage_machine.uuid,
                "uuid4",
                return_value=mock.Mock(hex="abc123"),
            ),
        ):
            with self.assertRaisesRegex(
                manage_machine.MachineManagementError,
                "link reset",
            ):
                manage_machine.stage_remote_script(
                    target,
                    b"x" * 1500,
                    batch_mode=True,
                )

        self.assertIn(
            "rm -f /tmp/.vaws-remote-script-abc123.sh",
            run_once.call_args_list[-1].args[1],
        )


if __name__ == "__main__":
    unittest.main()
