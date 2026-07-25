#!/usr/bin/env python3
"""Tests for guarded vLLM upstream compatibility planning."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / ".agents" / "skills" / "vllm-ascend-upstream-sync"


def load_module():
    name = "_upstream_sync_test"
    spec = importlib.util.spec_from_file_location(
        name, SKILL / "scripts" / "upstream_sync.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sync = load_module()
NOW = "2026-07-25T12:00:00Z"


def run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    run(path, "init")
    run(path, "config", "user.email", "tests@example.com")
    run(path, "config", "user.name", "Tests")


def write_and_commit(repo: Path, relative: str, content: str, message: str) -> str:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    run(repo, "add", relative)
    run(repo, "commit", "-m", message)
    return run(repo, "rev-parse", "HEAD")


class UpstreamSyncTests(unittest.TestCase):
    def prepare(self, root: Path) -> tuple[Path, Path, str, str]:
        vllm = root / "vllm"
        ascend = root / "vllm-ascend"
        init_repo(vllm)
        old = write_and_commit(
            vllm,
            "vllm/worker/model_runner.py",
            "def execute_model(x):\n    return x\n",
            "old",
        )
        new = write_and_commit(
            vllm,
            "vllm/worker/model_runner.py",
            "def execute_model(x, metadata):\n    return x\n",
            "new",
        )
        init_repo(ascend)
        write_and_commit(
            ascend,
            "vllm_ascend/worker/model_runner.py",
            "from vllm.worker.model_runner import execute_model\n"
            "result = execute_model(None)\n",
            "consumer",
        )
        return vllm, ascend, old, new

    def test_signature_change_and_consumer_are_high_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, ascend, old, new = self.prepare(Path(tmp))
            output = Path(tmp) / "report"
            result = sync.plan(
                output,
                vllm_repo=vllm,
                ascend_repo=ascend,
                old_ref=old,
                new_ref=new,
                run_id="upstream-sync-1",
                created_at=NOW,
            )
            self.assertEqual(result["risk"], "high")
            plan = json.loads(
                (output / "sync-plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["api_changes"][0]["kind"], "changed")
            self.assertEqual(
                plan["consumers"][0]["path"],
                "vllm_ascend/worker/model_runner.py",
            )
            self.assertIn(
                "correctness:model-runner", plan["recommended_validation"]
            )
            self.assertEqual(plan["apply_preconditions"]["current_head"], new)
            self.assertFalse(plan["apply_preconditions"]["ready"])

    def test_apply_rejects_dirty_submodule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, ascend, old, new = self.prepare(Path(tmp))
            run(vllm, "checkout", "--detach", old)
            output = Path(tmp) / "report"
            sync.plan(
                output,
                vllm_repo=vllm,
                ascend_repo=ascend,
                old_ref=old,
                new_ref=new,
                run_id="upstream-sync-1",
                created_at=NOW,
            )
            (vllm / "dirty.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(sync.UpstreamSyncError, "dirty"):
                sync.apply(output, updated_at=NOW)

    def test_apply_checks_out_exact_planned_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, ascend, old, new = self.prepare(Path(tmp))
            run(vllm, "checkout", "--detach", old)
            output = Path(tmp) / "report"
            sync.plan(
                output,
                vllm_repo=vllm,
                ascend_repo=ascend,
                old_ref=old,
                new_ref=new,
                run_id="upstream-sync-1",
                created_at=NOW,
            )
            result = sync.apply(output, updated_at=NOW)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(run(vllm, "rev-parse", "HEAD"), new)

    def test_apply_rejects_analysis_plan_when_head_was_not_old_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vllm, ascend, old, new = self.prepare(Path(tmp))
            output = Path(tmp) / "report"
            sync.plan(
                output,
                vllm_repo=vllm,
                ascend_repo=ascend,
                old_ref=old,
                new_ref=new,
                run_id="upstream-sync-1",
                created_at=NOW,
            )

            with self.assertRaisesRegex(
                sync.UpstreamSyncError,
                "analysis-only",
            ):
                sync.apply(output, updated_at=NOW)


if __name__ == "__main__":
    unittest.main()
