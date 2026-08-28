#!/usr/bin/env python3
"""Regression tests for shared SSH command construction."""

from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import vaws_remote_toolbox as toolbox  # noqa: E402


class RemoteToolboxTransportTests(unittest.TestCase):
    def test_remote_bash_command_round_trips_shell_variables(self) -> None:
        script = 'pid=$!\nprintf \'{"pid":%s}\\n\' "$pid"\n'
        command = toolbox._remote_bash_command(script)
        payload = command.split()[2]
        self.assertEqual(base64.b64decode(payload).decode("utf-8"), script)
        self.assertNotIn("$pid", command)
        self.assertNotIn("$!", command)

    def test_command_transport_disables_tty_and_stdin(self) -> None:
        endpoint = toolbox.SshEndpoint(host="host", port=22, user="root")
        command = toolbox._ssh_base_cmd(endpoint, stdin=False)
        self.assertIn("-T", command)
        self.assertIn("-n", command)

    def test_byte_upload_transport_keeps_stdin(self) -> None:
        endpoint = toolbox.SshEndpoint(host="host", port=22, user="root")
        command = toolbox._ssh_base_cmd(endpoint, stdin=True)
        self.assertIn("-T", command)
        self.assertNotIn("-n", command)


if __name__ == "__main__":
    unittest.main()
