from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.endpoint import Endpoint  # noqa: E402
import core.context_snapshot as context_snapshot  # noqa: E402


class ContextSnapshotTests(unittest.TestCase):
    def test_probe_runs_from_workspace_with_runtime_python(self) -> None:
        endpoint = Endpoint(
            host="1.2.3.4",
            port=46000,
            root="/",
            cwd="/vllm-workspace",
            runtime_env=True,
        )
        observed: dict[str, object] = {}

        def fake_run(_endpoint, _code, payload, **kwargs):
            observed["payload"] = payload
            observed["kwargs"] = kwargs
            return {"status": "ok", "summary": {"python": "3.12"}}

        with (
            mock.patch.object(context_snapshot, "run_remote_python", fake_run),
            mock.patch.object(context_snapshot, "write_context_snapshot", return_value={"refs": {}}),
        ):
            result = context_snapshot.remote_probe(endpoint)

        self.assertEqual(result["result"]["status"], "ok")
        self.assertEqual(observed["payload"], {"root": "/vllm-workspace"})
        self.assertEqual(observed["kwargs"]["cwd"], "/vllm-workspace")
        self.assertTrue(observed["kwargs"]["runtime_env"])


if __name__ == "__main__":
    unittest.main()
