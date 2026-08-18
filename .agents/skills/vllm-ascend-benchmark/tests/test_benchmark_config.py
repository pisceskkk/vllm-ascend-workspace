#!/usr/bin/env python3
"""Regression tests for Benchmark configuration forwarding."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
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
aisbench_perf = load_module(
    "_aisbench_perf_run_test", SCRIPTS / "aisbench_perf_run.py"
)


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


class AisbenchPerfTests(unittest.TestCase):
    def test_generated_config_is_isolated_and_complete(self) -> None:
        text = aisbench_perf.render_auto_tools_config(
            dataset_path="/remote/run/datasets",
            work_path="/vllm-workspace/benchmark",
            model_name="model-a",
            model_path="/weights/model-a",
            host="127.0.0.1",
            port=8000,
            output_dir="./outputs",
            performance_summarizer="default_perf",
            pod_info=["127.0.0.1:8000"],
        )
        self.assertIn("MODEL_NAME = 'model-a'", text)
        self.assertIn("HOST_PORT = '8000'", text)
        self.assertIn("POD_INFO = ['127.0.0.1:8000']", text)
        self.assertNotIn("API_KEY", text)

    def test_result_parser_rejects_auto_tools_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aisbench_result.csv"
            path.write_text(
                "TTFT avg,TPOT avg,output_throughput,qps\n99999,2.0,3.0,4.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sentinel"):
                aisbench_perf.parse_result_csv(path)

    def test_phase_command_validates_result_even_when_tool_exits_zero(self) -> None:
        command = aisbench_perf.build_phase_command(
            source_dir="/workspace/aisbench_auto_tools",
            phase_dir="/remote/run-001",
            config_path="/remote/config.py",
            command=["python3", "aisbench_test.py", "--repeat", "1"],
        )
        self.assertIn("aisbench_result.csv", command)
        self.assertIn("99999.0", command)

    def test_resolved_config_redacts_secret_environment_values(self) -> None:
        redacted = aisbench_perf.redact_extra_env(
            ["OMP_NUM_THREADS=10", "HTTP_PROXY_TOKEN=live-secret"]
        )
        self.assertEqual(
            redacted,
            ["OMP_NUM_THREADS=10", "HTTP_PROXY_TOKEN=<redacted>"],
        )


if __name__ == "__main__":
    unittest.main()
