from __future__ import annotations

import base64
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

        def fake_run_bytes(_endpoint, remote_command, **kwargs):
            observed["remote_command"] = remote_command
            observed["kwargs"] = kwargs
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=b'{"status":"ok"}',
                stderr=b"",
            )

        with mock.patch.object(ssh_transport, "run_bytes", fake_run_bytes):
            payload = ssh_transport.run_remote_python(
                endpoint,
                "print('{}')",
                {},
                cwd="/vllm-workspace",
                runtime_env=True,
            )

        command = observed["remote_command"]
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(command.startswith("bash -c "))
        self.assertIn("/etc/profile.d/vaws-ascend-env.sh", command)
        self.assertIn("/usr/local/python*/bin/python3", command)
        self.assertIn("/vllm-workspace", command)

    def test_run_remote_python_accepts_runtime_logs_before_json(self) -> None:
        endpoint = Endpoint(host="1.2.3.4", port=46000)

        def fake_run_bytes(_endpoint, _remote_command, **_kwargs):
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    b"INFO plugin activated\n"
                    b'{"status":"ok","summary":{"python":"3.12"}}\n'
                ),
                stderr=b"",
            )

        with mock.patch.object(ssh_transport, "run_bytes", fake_run_bytes):
            payload = ssh_transport.run_remote_python(
                endpoint,
                "print('{}')",
                {},
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["summary"]["python"], "3.12")

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
                ssh_transport,
                "_stage_remote_bytes",
                return_value=("/tmp/.remote-dev-input-abc123.bin", None),
            ) as stage,
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
        stage.assert_called_once()
        self.assertEqual(stage.call_args.args[1], b"x" * 1500)
        self.assertEqual(len(calls), 2)
        commands = [call["args"][-1] for call in calls]
        self.assertIn("wc -c", commands[0])
        self.assertIn("bash -c", commands[0])
        self.assertIn("< /tmp/.remote-dev-input-abc123.bin", commands[0])
        self.assertIn("rm -f", commands[1])
        self.assertTrue(all(call["kwargs"].get("input") is None for call in calls))

    def test_run_bytes_stages_long_remote_command(self) -> None:
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
                ssh_transport,
                "_stage_remote_bytes",
                return_value=("/tmp/.remote-dev-script-script123.bin", None),
            ),
            mock.patch.object(
                ssh_transport.uuid,
                "uuid4",
                return_value=mock.Mock(hex="script123"),
            ),
        ):
            result = ssh_transport.run_bytes(
                endpoint,
                "printf x\n" * 300,
                stdin=b"small",
                timeout_ms=30000,
            )

        self.assertEqual(result.stdout, b"done")
        commands = [call["args"][-1] for call in calls]
        self.assertIn(".remote-dev-script-script123.bin", commands[0])
        self.assertTrue(
            any("bash /tmp/.remote-dev-script-script123.bin" in command for command in commands)
        )
        self.assertIn("rm -f", commands[-1])
        final_call = next(
            call
            for call in calls
            if "bash /tmp/.remote-dev-script-script123.bin" in call["args"][-1]
        )
        self.assertEqual(final_call["kwargs"]["input"], b"small")

    def test_staged_frame_stays_below_constrained_transport_limit(self) -> None:
        frame = base64.b64encode(b"x" * ssh_transport.STAGED_CHUNK_BYTES) + b"\n"
        self.assertLessEqual(len(frame), 700)


if __name__ == "__main__":
    unittest.main()
