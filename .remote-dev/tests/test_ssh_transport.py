from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.endpoint import Endpoint  # noqa: E402
import core.ssh_transport as ssh_transport  # noqa: E402


class SshTransportTests(unittest.TestCase):
    def test_run_remote_python_quotes_multiline_code_as_one_remote_command(self) -> None:
        endpoint = Endpoint(host="1.2.3.4", port=46000)
        observed: dict[str, object] = {}

        def fake_run(args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return subprocess.CompletedProcess(args=args, returncode=0, stdout='{"status":"ok"}', stderr="")

        with mock.patch.object(ssh_transport.subprocess, "run", fake_run):
            payload = ssh_transport.run_remote_python(endpoint, "import json\nprint(json.dumps({'status':'ok'}))", {})

        args = observed["args"]
        self.assertIsInstance(args, list)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(args[-1].split(" ", 2)[:2], ["bash", "-c"])
        self.assertIn("python3", args[-1])
        self.assertNotIn("print(json.dumps({'status':'ok'}))", args[-1])
        envelope = json.loads(observed["kwargs"]["input"])
        self.assertIn("\n", envelope["code"])
        self.assertEqual(envelope["payload"], {})

    def test_run_remote_python_can_select_runtime_python_and_cwd(self) -> None:
        endpoint = Endpoint(host="1.2.3.4", port=46000)
        observed: dict[str, object] = {}

        def fake_run(args, **kwargs):
            observed["args"] = args
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"status":"ok"}',
                stderr="",
            )

        with mock.patch.object(ssh_transport.subprocess, "run", fake_run):
            payload = ssh_transport.run_remote_python(
                endpoint,
                "print('{}')",
                {},
                cwd="/vllm-workspace",
                runtime_env=True,
            )

        command = observed["args"][-1]
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(command.startswith("bash -c "))
        self.assertIn("/etc/profile.d/vaws-ascend-env.sh", command)
        self.assertIn("/usr/local/python*/bin/python3", command)
        self.assertIn("/vllm-workspace", command)

    def test_run_bytes_quotes_shell_command_as_one_remote_command(self) -> None:
        endpoint = Endpoint(host="1.2.3.4", port=46000)
        observed: dict[str, object] = {}

        def fake_run(args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

        with mock.patch.object(ssh_transport.subprocess, "run", fake_run):
            ssh_transport.run_bytes(endpoint, "cat '/tmp/path with spaces'")

        args = observed["args"]
        self.assertIsInstance(args, list)
        self.assertEqual(args[-1].split(" ", 2)[:2], ["bash", "-c"])
        self.assertIn("path with spaces", args[-1])

    def test_run_bytes_stages_large_stdin_in_bounded_chunks(self) -> None:
        endpoint = Endpoint(host="1.2.3.4", port=46000)
        calls: list[dict[str, object]] = []

        def fake_run(args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=b"done",
                stderr=b"",
            )

        with (
            mock.patch.object(ssh_transport.subprocess, "run", fake_run),
            mock.patch.object(
                ssh_transport.uuid,
                "uuid4",
                return_value=mock.Mock(hex="abc123"),
            ),
        ):
            result = ssh_transport.run_bytes(
                endpoint,
                "wc -c",
                stdin=b"x" * 1500,
                timeout_ms=30000,
            )

        self.assertEqual(result.stdout, b"done")
        self.assertEqual(len(calls), 6)
        commands = [call["args"][-1] for call in calls]
        self.assertIn(".remote-dev-input-abc123.bin", commands[0])
        self.assertIn("base64 -d", commands[1])
        self.assertIn("base64 -d", commands[2])
        self.assertIn("base64 -d", commands[3])
        self.assertIn("wc -c", commands[4])
        self.assertIn("bash -c", commands[4])
        self.assertIn("< /tmp/.remote-dev-input-abc123.bin", commands[4])
        self.assertIn("rm -f", commands[5])
        self.assertTrue(all(call["kwargs"].get("input") is None for call in calls))


if __name__ == "__main__":
    unittest.main()
