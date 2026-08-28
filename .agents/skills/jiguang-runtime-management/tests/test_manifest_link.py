from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_run_manifest import load_manifest, new_manifest, write_manifest  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jiguang_manifest_link.py"


class ManifestLinkTests(unittest.TestCase):
    def test_adds_checksum_bound_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_manifest(path, new_manifest(run_type="correctness", run_id="correctness-test"))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(path),
                    "--archive-url",
                    "https://jiguang.ascend.huawei.com/run/1",
                    "--summary-json",
                    json.dumps({"run_id": "1", "status": "passed"}),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = load_manifest(path)["artifacts"][0]
            self.assertEqual(artifact["name"], "jiguang-summary")
            self.assertEqual(len(artifact["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
