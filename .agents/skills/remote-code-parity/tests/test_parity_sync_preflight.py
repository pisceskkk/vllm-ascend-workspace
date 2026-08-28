#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents" / "skills" / "remote-code-parity" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module():
    spec = importlib.util.spec_from_file_location("_parity_sync_preflight_test", SCRIPTS / "parity_sync.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parity_sync = load_module()


class ParitySyncPreflightTests(unittest.TestCase):
    def test_permission_failure_stops_before_low_level_sync(self) -> None:
        derived = {
            "state_repo_root": str(ROOT),
            "container_host": "host",
            "container_port": 46001,
            "container_user": "root",
            "server_name": "machine",
            "container_identity": "container@/runtime",
        }
        failed = {
            "status": "blocked",
            "category": "local_ssh_config_permissions",
            "knowledge_id": "machine-management-openssh-system-config-ownership",
            "message": "bad owner",
        }
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["parity_sync.py", "--machine", "machine"]),
            mock.patch.object(parity_sync, "repo_root_from", return_value=ROOT),
            mock.patch.object(parity_sync, "build_derived_args", return_value=derived),
            mock.patch.object(parity_sync, "build_low_level_command", return_value=["sync"]),
            mock.patch.object(parity_sync, "load_consent_state", return_value={}),
            mock.patch.object(parity_sync, "resolve_sync_mode", return_value="local"),
            mock.patch.object(parity_sync, "ssh_client_preflight", return_value=failed),
            mock.patch.object(parity_sync.subprocess, "run") as execute,
            contextlib.redirect_stdout(output),
        ):
            returncode = parity_sync.main()

        self.assertEqual(returncode, 2)
        execute.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["action"], "ssh-client-preflight")
        self.assertEqual(payload["knowledge_id"], failed["knowledge_id"])

    def test_host_execution_requirement_stops_before_low_level_sync(self) -> None:
        derived = {
            "state_repo_root": str(ROOT),
            "container_host": "host",
            "container_port": 46001,
            "container_user": "root",
            "server_name": "machine",
            "container_identity": "container@/runtime",
        }
        failed = {
            "status": "blocked",
            "category": "ssh_host_execution_required",
            "host_execution_required": True,
            "message": "rerun outside sandbox",
        }
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["parity_sync.py", "--machine", "machine"]),
            mock.patch.object(parity_sync, "repo_root_from", return_value=ROOT),
            mock.patch.object(parity_sync, "build_derived_args", return_value=derived),
            mock.patch.object(parity_sync, "build_low_level_command", return_value=["sync"]),
            mock.patch.object(parity_sync, "load_consent_state", return_value={}),
            mock.patch.object(parity_sync, "resolve_sync_mode", return_value="local"),
            mock.patch.object(parity_sync, "ssh_client_preflight", return_value=failed),
            mock.patch.object(parity_sync.subprocess, "run") as execute,
            contextlib.redirect_stdout(output),
        ):
            returncode = parity_sync.main()

        self.assertEqual(returncode, 2)
        execute.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["host_execution_required"])
        self.assertEqual(payload["ssh_preflight"]["category"], "ssh_host_execution_required")


if __name__ == "__main__":
    unittest.main()
