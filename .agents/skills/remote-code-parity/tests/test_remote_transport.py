#!/usr/bin/env python3
"""Regression tests for non-interactive parity SSH transport."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents" / "skills" / "remote-code-parity" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("common", SCRIPTS / "common.py")
parity = load_module("_remote_code_parity_test", SCRIPTS / "remote_code_parity.py")


class RemoteParityTransportTests(unittest.TestCase):
    def test_command_ssh_disables_tty_and_stdin(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        command = common._ssh_base_cmd(endpoint, stdin=False)
        self.assertIn("-T", command)
        self.assertIn("-n", command)
        self.assertNotIn("-n", common._ssh_base_cmd(endpoint, stdin=True))

    def test_runtime_env_probe_has_bounded_fallback(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        with mock.patch.object(
            parity,
            "ssh_exec",
            side_effect=RuntimeError("command timed out"),
        ) as execute:
            result = parity.read_runtime_install_env(
                container=endpoint,
                runtime_root="/runtime",
                dry_run=False,
            )
        self.assertEqual(result, {})
        self.assertEqual(execute.call_args.kwargs["timeout"], 15)

    def test_runtime_env_probe_parses_successful_snapshot(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"MAX_JOBS": "8"}\n',
            stderr="",
        )
        with mock.patch.object(parity, "ssh_exec", return_value=completed):
            result = parity.read_runtime_install_env(
                container=endpoint,
                runtime_root="/runtime",
                dry_run=False,
            )
        self.assertEqual(result, {"MAX_JOBS": "8"})


if __name__ == "__main__":
    unittest.main()
