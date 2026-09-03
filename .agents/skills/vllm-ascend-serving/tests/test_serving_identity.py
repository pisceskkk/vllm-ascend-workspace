#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents" / "skills" / "vllm-ascend-serving" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import serve_start  # noqa: E402
from serve_start import service_runtime_dir  # noqa: E402


class ServingIdentityTests(unittest.TestCase):
    def test_alias_namespaces_runtime_directory(self) -> None:
        self.assertEqual(
            service_runtime_dir("/vllm-workspace", "20260811_120000", "agent12345"),
            "/vllm-workspace/.vaws-runtime/serving/agent12345/20260811_120000",
        )

    def test_missing_alias_preserves_legacy_layout(self) -> None:
        self.assertEqual(
            service_runtime_dir("/vllm-workspace", "20260811_120000", None),
            "/vllm-workspace/.vaws-runtime/serving/20260811_120000",
        )

    def test_explicit_vllm_commit_is_forwarded_and_blocker_is_preserved(self) -> None:
        process = mock.Mock()
        process.stderr = io.StringIO("")
        process.stdout = io.StringIO(json.dumps({"status": "blocked", "reason": "pair mismatch"}))
        process.wait.return_value = 2

        with mock.patch.object(serve_start.subprocess, "Popen", return_value=process) as popen:
            result = serve_start.run_parity(
                "blue-a",
                None,
                vllm_commit="a" * 40,
            )

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--vllm-commit") + 1], "a" * 40)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "pair mismatch")


if __name__ == "__main__":
    unittest.main()
