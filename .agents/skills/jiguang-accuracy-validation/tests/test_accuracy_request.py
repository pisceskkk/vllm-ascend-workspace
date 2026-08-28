from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accuracy_request.py"


class AccuracyRequestTests(unittest.TestCase):
    def test_builds_pinned_accuracy_payload(self) -> None:
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
                "--dataset",
                "gsm8k",
                "--dataset-version",
                "1",
                "--dataset-split",
                "test",
                "--commit-sha",
                "a" * 40,
                "--submodule-shas-json",
                json.dumps({"vllm": "b" * 40, "vllm-ascend": "c" * 40}),
                "--configuration-json",
                json.dumps({"max_tokens": 512}),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)["payload"]
        self.assertEqual(payload["evaluation_type"], "accuracy")
        self.assertEqual(payload["dataset_split"], "test")
        self.assertEqual(payload["device_ids"], ["device-1"])


if __name__ == "__main__":
    unittest.main()
