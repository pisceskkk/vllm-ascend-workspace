from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.tools import call_tool  # noqa: E402


class McpTests(unittest.TestCase):
    def test_tool_catalog_contains_no_admin_or_container_create(self) -> None:
        request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}) + "\n"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "mcp" / "server.py")],
            input=request,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = json.loads(completed.stdout)
        names = {item["name"] for item in response["result"]["tools"]}
        self.assertFalse(any("admin" in name for name in names))
        self.assertNotIn("jiguang.container_create", names)
        self.assertIn("jiguang.container_connection_apply", names)

    def test_rejects_string_confirmation_before_transport(self) -> None:
        with patch("mcp.tools._client") as client_factory:
            with self.assertRaisesRegex(ValueError, "confirm must be a boolean"):
                call_tool(
                    "jiguang.device_delete_apply",
                    {"device_id": "device-1", "confirm": "false"},
                )
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
