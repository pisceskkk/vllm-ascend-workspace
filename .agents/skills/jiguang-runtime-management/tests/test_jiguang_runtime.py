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

from vaws_jiguang import plan_runtime, record_runtime  # noqa: E402


def make_repo(path: Path, filename: str, content: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / filename).parent.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "initial"], check=True)


class RuntimeTests(unittest.TestCase):
    def test_first_create_then_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root / "vllm", "csrc/op.cpp", "int x = 1;\n")
            make_repo(root / "vllm-ascend", "setup.py", "name = 'x'\n")
            state = root / "state.json"
            first = plan_runtime(
                machine="m1",
                image_digest="image@sha256:" + "a" * 64,
                components={"cann": "1"},
                repo_root=root,
                state_path=state,
            )
            self.assertEqual(first["decision"], "create")
            record_runtime(
                state,
                "m1",
                {
                    "container_name": "vaws-jiguang",
                    "generation": 1,
                    "image_digest": "image@sha256:" + "a" * 64,
                    "runtime_hash": first["runtime_hash"],
                    "native_code_hash": first["native_code_hash"],
                    "health": "ready",
                },
            )
            second = plan_runtime(
                machine="m1",
                image_digest="image@sha256:" + "a" * 64,
                components={"cann": "1"},
                repo_root=root,
                state_path=state,
            )
            self.assertEqual(second["decision"], "reuse")
            self.assertEqual(json.loads(state.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
