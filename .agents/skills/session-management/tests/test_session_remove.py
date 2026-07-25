#!/usr/bin/env python3
"""Tests for bounded session worktree cleanup."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / ".agents" / "skills" / "session-management"


def load_module():
    name = "_session_remove_test"
    spec = importlib.util.spec_from_file_location(
        name, SKILL / "scripts" / "session_remove.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


session_remove = load_module()


class SessionRemoveTests(unittest.TestCase):
    def test_deinitializes_submodules_before_removing_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp)
            calls: list[tuple[Path, list[str]]] = []

            def runner(args: list[str], *, cwd: Path, check: bool = False):
                del check
                calls.append((cwd, args))
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout="ok",
                    stderr="",
                )

            result = session_remove.remove_session_worktree(
                worktree,
                force=False,
                runner=runner,
            )

            self.assertEqual(
                calls,
                [
                    (worktree, ["submodule", "deinit", "--force", "--all"]),
                    (
                        session_remove.ROOT,
                        ["worktree", "remove", str(worktree)],
                    ),
                ],
            )
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["submodule_deinit"]["returncode"], 0)

    def test_force_is_forwarded_to_worktree_removal(self) -> None:
        missing = Path("/definitely/missing/session-worktree")
        calls: list[tuple[Path, list[str]]] = []

        def runner(args: list[str], *, cwd: Path, check: bool = False):
            del check
            calls.append((cwd, args))
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="",
                stderr="",
            )

        result = session_remove.remove_session_worktree(
            missing,
            force=True,
            runner=runner,
        )

        self.assertEqual(
            calls,
            [
                (
                    session_remove.ROOT,
                    [
                        "worktree",
                        "remove",
                        "--force",
                        "--force",
                        str(missing),
                    ],
                )
            ],
        )
        self.assertIsNone(result["submodule_deinit"])


if __name__ == "__main__":
    unittest.main()
