#!/usr/bin/env python3
"""Regression tests for Benchmark configuration forwarding."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".agents" / "skills" / "vllm-ascend-benchmark" / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("_benchmark_common_test", SCRIPTS / "_common.py")


class BenchmarkConfigTests(unittest.TestCase):
    def test_health_timeout_is_forwarded_to_serving(self) -> None:
        config = common.assemble_config(
            machine="host-a",
            model="/models/example",
            health_timeout=1800,
        )
        args = config.to_serve_start_args()
        self.assertEqual(
            args[args.index("--health-timeout") + 1],
            "1800",
        )
        self.assertEqual(config.summary_dict()["health_timeout"], 1800)

    def test_health_timeout_is_omitted_when_unspecified(self) -> None:
        config = common.assemble_config(
            machine="host-a",
            model="/models/example",
        )
        self.assertNotIn("--health-timeout", config.to_serve_start_args())
        self.assertNotIn("health_timeout", config.summary_dict())

    def test_non_positive_health_timeout_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "positive"):
            common.assemble_config(
                machine="host-a",
                model="/models/example",
                health_timeout=0,
            )


if __name__ == "__main__":
    unittest.main()
