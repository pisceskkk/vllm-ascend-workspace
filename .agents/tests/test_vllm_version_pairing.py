#!/usr/bin/env python3
"""Regression tests for strict vLLM/vllm-ascend source pairing."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vllm_version_pairing import check_workspace_vllm_pairing  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")


def commit_file(repo: Path, relative: str, content: str, message: str) -> str:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "add", relative)
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


class VllmVersionPairingTests(unittest.TestCase):
    def make_workspace(self, root: Path) -> tuple[Path, Path, str]:
        vllm = root / "vllm"
        ascend = root / "vllm-ascend"
        init_repo(vllm)
        vllm_commit = commit_file(vllm, "source.py", "one\n", "vllm")
        init_repo(ascend)
        commit_file(
            ascend,
            ".github/vllm-main-verified.commit",
            f"{vllm_commit}\n",
            "pin",
        )
        return vllm, ascend, vllm_commit

    def test_default_pair_uses_pin_from_vllm_ascend_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, ascend, vllm_commit = self.make_workspace(root)
            (ascend / ".github/vllm-main-verified.commit").write_text(
                "f" * 40 + "\n",
                encoding="utf-8",
            )

            result = check_workspace_vllm_pairing(root)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["required_vllm_commit"], vllm_commit)
            self.assertEqual(result["precedence"], "vllm-ascend-head-verified-pin")

    def test_mismatch_blocks_unless_exact_commit_was_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vllm, _, pinned_commit = self.make_workspace(root)
            current_commit = commit_file(vllm, "source.py", "two\n", "new vllm")

            automatic = check_workspace_vllm_pairing(root)
            explicit = check_workspace_vllm_pairing(
                root,
                explicit_vllm_commit=current_commit[:12],
            )

            self.assertEqual(automatic["status"], "blocked")
            self.assertEqual(automatic["required_vllm_commit"], pinned_commit)
            self.assertEqual(explicit["status"], "ready")
            self.assertTrue(explicit["explicit_override"])

    def test_missing_or_invalid_verified_pin_fails_closed(self) -> None:
        for content in (None, "main\n", "a" * 39 + "\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                vllm = root / "vllm"
                ascend = root / "vllm-ascend"
                init_repo(vllm)
                commit_file(vllm, "source.py", "one\n", "vllm")
                init_repo(ascend)
                if content is None:
                    commit_file(ascend, "README.md", "no pin\n", "no pin")
                else:
                    commit_file(
                        ascend,
                        ".github/vllm-main-verified.commit",
                        content,
                        "bad pin",
                    )

                result = check_workspace_vllm_pairing(root)

                self.assertEqual(result["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
