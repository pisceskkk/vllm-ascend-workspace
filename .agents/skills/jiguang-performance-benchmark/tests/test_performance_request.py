from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "performance_request.py"


class PerformanceRequestTests(unittest.TestCase):
    def test_builds_repeated_warm_payload(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--name",
                "case",
                "--app-id",
                "app-1",
                "--deployment-id",
                "deployment-1",
                "--device-id",
                "device-1",
                "--model",
                "model",
                "--commit-sha",
                "a" * 40,
                "--submodule-shas-json",
                json.dumps({"vllm": "b" * 40, "vllm-ascend": "c" * 40}),
                "--configuration-json",
                json.dumps({"concurrency": 8}),
                "--repetitions",
                "3",
                "--warmup-runs",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)["payload"]
        self.assertEqual(payload["evaluation_type"], "performance")
        self.assertEqual(payload["repetitions"], 3)
        self.assertEqual(payload["warmup_runs"], 1)
        self.assertEqual(payload["device_ids"], ["device-1"])


if __name__ == "__main__":
    unittest.main()
