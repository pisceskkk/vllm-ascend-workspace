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

    def test_runtime_install_env_accepts_proxy_variables(self) -> None:
        expected = {
            "http_proxy",
            "https_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        }

        self.assertTrue(expected.issubset(parity.REMOTE_RUNTIME_ENV_PASSTHROUGH))
        self.assertTrue(expected.issubset(parity.RUNTIME_INSTALL_ENV_KEYS))

    def test_runtime_env_redacts_proxy_userinfo(self) -> None:
        redacted = parity.redact_runtime_env(
            {"https_proxy": "http://real-user:real-password@10.0.0.5:8080"}
        )

        self.assertEqual(
            redacted["https_proxy"],
            "http://***@10.0.0.5:8080",
        )
        self.assertNotIn("real-password", redacted["https_proxy"])

    def test_git_remote_url_supports_ipv4_ipv6_and_escaped_paths(self) -> None:
        ipv4 = common.SshEndpoint(host="10.0.0.2", port=46001, user="root")
        ipv6 = common.SshEndpoint(host="2001:db8::2", port=22, user="build user")

        self.assertEqual(
            parity.git_remote_url(ipv4, "/cache/workspace.git"),
            "ssh://root@10.0.0.2:46001/cache/workspace.git",
        )
        self.assertEqual(
            parity.git_remote_url(ipv6, "/cache/a repo.git"),
            "ssh://build%20user@[2001:db8::2]:22/cache/a%20repo.git",
        )

    def test_git_transport_is_noninteractive_and_uses_endpoint_port(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=46001, user="root")

        env = parity.git_ssh_environment(endpoint)

        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertIn("BatchMode=yes", env["GIT_SSH_COMMAND"])
        self.assertIn("46001", env["GIT_SSH_COMMAND"])

    def test_transport_carrier_ref_is_stable_and_target_scoped(self) -> None:
        first = common.SshEndpoint(host="host-a", port=22, user="root")
        second = common.SshEndpoint(host="host-b", port=22, user="root")
        record = parity.SnapshotRecord(
            relpath="vllm",
            repo_id="vllm",
            source_head="source",
            parent="source",
            commit="snapshot",
            tree="tree",
            ref="refs/parity/test/snapshot/vllm",
            changed_paths=[],
            submodules=[],
        )

        first_ref = parity.transport_carrier_ref(
            first, "/cache/vllm.git", "workspace", record
        )
        repeated_ref = parity.transport_carrier_ref(
            first, "/cache/vllm.git", "workspace", record
        )
        second_ref = parity.transport_carrier_ref(
            second, "/cache/vllm.git", "workspace", record
        )

        self.assertEqual(first_ref, repeated_ref)
        self.assertNotEqual(first_ref, second_ref)
        self.assertTrue(first_ref.startswith("refs/parity-transport/workspace/"))

    def test_git_push_publishes_snapshot_and_transport_carrier(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        record = parity.SnapshotRecord(
            relpath="vllm",
            repo_id="vllm",
            source_head="source",
            parent="source",
            commit="snapshot",
            tree="tree",
            ref="refs/parity/test/snapshot/vllm",
            changed_paths=[],
            submodules=[],
        )
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr=""
        )

        with (
            mock.patch.object(
                parity,
                "build_transport_carrier",
                return_value=("refs/parity-transport/local", "carrier-commit"),
            ),
            mock.patch.object(parity, "git_remote_url", return_value="ssh://remote/mirror.git"),
            mock.patch.object(parity, "git_ssh_environment", return_value={}) as environment,
            mock.patch.object(parity, "git", return_value=completed) as execute,
        ):
            result = parity.push_snapshot_via_git(
                Path("/repo"),
                container=endpoint,
                mirror_path="/cache/vllm.git",
                record=record,
                workspace_id="test",
            )

        command = execute.call_args.args[1]
        self.assertIn(
            "refs/parity-transport/local:refs/parity/test/transport-carrier",
            command,
        )
        self.assertIn(
            "refs/parity/test/snapshot/vllm:refs/parity/test/current",
            command,
        )
        self.assertEqual(result["carrier_commit"], "carrier-commit")
        environment.assert_called_once_with(endpoint)

    def test_auto_transport_prefers_git_without_creating_bundle(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        record = parity.SnapshotRecord(
            relpath="vllm",
            repo_id="vllm",
            source_head="source",
            parent="source",
            commit="snapshot",
            tree="tree",
            ref="refs/parity/test/snapshot/vllm",
            changed_paths=["vllm/file.py"],
            submodules=[],
        )
        expected = {"repo": "vllm", "transport": "git"}

        with (
            mock.patch.object(parity, "push_snapshot_via_git", return_value=expected) as push_git,
            mock.patch.object(parity, "push_snapshot_via_bundle") as push_bundle,
        ):
            result = parity.push_snapshot_to_mirror(
                Path("/repo"),
                container=endpoint,
                mirror_path="/cache/vllm.git",
                container_cache_root="/cache",
                record=record,
                workspace_id="test",
                dry_run=False,
                transport="auto",
            )

        self.assertEqual(result, expected)
        push_git.assert_called_once()
        push_bundle.assert_not_called()

    def test_auto_transport_falls_back_to_bundle(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        record = parity.SnapshotRecord(
            relpath="vllm-ascend",
            repo_id="vllm-ascend",
            source_head="source",
            parent="source",
            commit="snapshot",
            tree="tree",
            ref="refs/parity/test/snapshot/vllm-ascend",
            changed_paths=["vllm_ascend/file.py"],
            submodules=[],
        )

        with (
            mock.patch.object(parity, "push_snapshot_via_git", side_effect=RuntimeError("receive-pack denied")),
            mock.patch.object(
                parity,
                "push_snapshot_via_bundle",
                return_value={"repo": "vllm-ascend", "transport": "bundle"},
            ) as push_bundle,
            mock.patch.object(parity, "emit_progress") as progress,
        ):
            result = parity.push_snapshot_to_mirror(
                Path("/repo"),
                container=endpoint,
                mirror_path="/cache/vllm-ascend.git",
                container_cache_root="/cache",
                record=record,
                workspace_id="test",
                dry_run=False,
                transport="auto",
            )

        self.assertEqual(result["transport"], "bundle")
        self.assertEqual(result["fallback_from"], "git")
        push_bundle.assert_called_once()
        progress.assert_called_once()

    def test_forced_git_transport_does_not_fallback(self) -> None:
        endpoint = common.SshEndpoint(host="host", port=22, user="root")
        record = parity.SnapshotRecord(
            relpath=".",
            repo_id="workspace",
            source_head="source",
            parent="source",
            commit="snapshot",
            tree="tree",
            ref="refs/parity/test/snapshot/workspace",
            changed_paths=[],
            submodules=[],
        )

        with (
            mock.patch.object(parity, "push_snapshot_via_git", side_effect=RuntimeError("denied")),
            mock.patch.object(parity, "push_snapshot_via_bundle") as push_bundle,
        ):
            with self.assertRaisesRegex(RuntimeError, "denied"):
                parity.push_snapshot_to_mirror(
                    Path("/repo"),
                    container=endpoint,
                    mirror_path="/cache/workspace.git",
                    container_cache_root="/cache",
                    record=record,
                    workspace_id="test",
                    dry_run=False,
                    transport="git",
                )

        push_bundle.assert_not_called()

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
