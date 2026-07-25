#!/usr/bin/env python3
"""Regression tests for non-interactive parity SSH transport."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_long_command_script_uses_staged_file(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        script = "true\n" * 1024
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with (
            mock.patch.object(common, "_stage_remote_script", return_value="/tmp/probe.sh") as stage,
            mock.patch.object(common, "run", return_value=completed) as execute,
        ):
            result = common.ssh_exec(endpoint, script)

        self.assertEqual(result.returncode, 0)
        stage.assert_called_once_with(endpoint, script)
        command = execute.call_args.args[0]
        self.assertIn("bash", command)
        self.assertIn("-n", command)
        self.assertTrue(any("/tmp/probe.sh" in part for part in command))

    def test_short_command_keeps_inline_noninteractive_path(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with mock.patch.object(common, "run", return_value=completed) as execute:
            common.ssh_exec(endpoint, "true")

        command = execute.call_args.args[0]
        self.assertIn("-n", command)
        self.assertIn("-c", command)

    def test_remote_script_staging_uses_length_bounded_binary_transfer(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")

        with (
            mock.patch.object(common.uuid, "uuid4", return_value=SimpleNamespace(hex="abc123")),
            mock.patch.object(common, "_ssh_exec_inline") as execute,
        ):
            path = common._stage_remote_script(endpoint, "printf ok")

        self.assertEqual(path, "/tmp/vaws-parity-script-abc123.sh")
        commands = [call.args[1] for call in execute.call_args_list]
        self.assertGreaterEqual(len(commands), 3)
        self.assertTrue(any("base64 -d" in command for command in commands))
        self.assertTrue(all(len(command.encode("utf-8")) <= 2048 for command in commands))

    def test_byte_payload_wrapper_preserves_payload_for_framed_transfer(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        payload = b"payload\x00with-binary"

        def inspect_transfer(actual_endpoint, remote_path, local_path):
            self.assertEqual(actual_endpoint, endpoint)
            self.assertEqual(remote_path, "/remote/payload")
            self.assertEqual(local_path.read_bytes(), payload)

        with mock.patch.object(
            common,
            "ssh_stream_file_to_file",
            side_effect=inspect_transfer,
        ) as transfer:
            common.ssh_stream_bytes_to_file(endpoint, "/remote/payload", payload)

        transfer.assert_called_once()

    def test_vllm_install_bootstraps_declared_rust_build_requirement(self) -> None:
        script = parity.runtime_install_step_script(
            runtime_root="/runtime",
            marker_dirname=".marker",
            container_identity="container@/runtime",
            step="install-vllm-build-requirements",
        )

        self.assertIn("requirements/build/rust.txt", script)
        self.assertIn("runtime-install-vllm-build-requirements", script)
        self.assertNotIn("requirements/build/cuda.txt", script)

    def test_import_smoke_rejects_outer_repository_namespace(self) -> None:
        script = parity.runtime_install_step_script(
            runtime_root="/runtime",
            marker_dirname=".marker",
            container_identity="container@/runtime",
            step="verify-imports",
        )

        self.assertIn("cd /tmp", script)
        self.assertIn("from vllm import LLM, SamplingParams", script)
        self.assertIn("vllm resolved to a namespace package", script)


if __name__ == "__main__":
    unittest.main()
