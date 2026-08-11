#!/usr/bin/env python3
"""Regression tests for the session-aware parity wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents" / "skills" / "remote-code-parity" / "scripts"
LIB = ROOT / ".agents" / "lib"
for path in (SCRIPTS, LIB):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wrapper = load_module("_parity_sync_test", SCRIPTS / "parity_sync.py")


class SessionParityStateTests(unittest.TestCase):
    def test_session_uses_worktree_for_source_and_base_repo_for_state(self) -> None:
        args = argparse.Namespace(
            session_id="session-a",
            session_file=None,
            runtime_root=None,
            workspace_id=None,
            container_cache_root="/cache",
            container_user="root",
            preserve_path=[],
        )
        session = {
            "session_id": "session-a",
            "base_machine": "server-a",
            "workspace_id": "session-a",
            "local": {
                "worktree_root": "/worktrees/session-a",
                "base_repo_root": "/repos/workspace",
            },
            "remote": {
                "host": "host-a",
                "container": {
                    "name": "container-a",
                    "ssh_port": 46001,
                    "runtime_root": "/vllm-workspace",
                },
            },
        }
        lookup = SimpleNamespace(session=session, session_file=Path("/state/session.json"))

        with (
            mock.patch.object(wrapper, "load_session_lookup", return_value=lookup),
            mock.patch.object(wrapper, "repo_root_from", side_effect=lambda path: path),
        ):
            derived = wrapper.build_derived_args_from_session(Path("/repos/workspace"), args)

        self.assertEqual(derived["workspace_root"], "/worktrees/session-a")
        self.assertEqual(derived["state_repo_root"], "/repos/workspace")

    def test_low_level_sync_receives_explicit_state_root(self) -> None:
        derived = {
            "workspace_root": "/worktrees/session-a",
            "state_repo_root": "/repos/workspace",
            "workspace_id": "session-a",
            "server_name": "server-a",
            "runtime_root": "/vllm-workspace",
            "container_identity": "container-a@/vllm-workspace",
            "container_cache_root": "/cache",
            "container_host": "host-a",
            "container_port": 46001,
            "container_user": "root",
            "preserve_path": [],
        }
        args = argparse.Namespace(
            snapshot_id=None,
            print_manifest=False,
            force_reinstall=False,
            dry_run=False,
            apply_mode="materialize",
        )

        command = wrapper.build_low_level_command(derived, args)

        index = command.index("--state-root")
        self.assertEqual(command[index + 1], "/repos/workspace")
        transport_index = command.index("--transport")
        self.assertEqual(command[transport_index + 1], "auto")


if __name__ == "__main__":
    unittest.main()
