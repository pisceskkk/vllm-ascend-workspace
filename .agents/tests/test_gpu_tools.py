#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = ROOT / ".agents" / "lib"
SCRIPTS = ROOT / ".agents" / "scripts"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_gpu_common import GpuTarget, target_from_args  # noqa: E402


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GpuToolTests(unittest.TestCase):
    def test_workspace_declares_only_repository_level_gpu_prefix_tools(self) -> None:
        module = load_script("gpu_workspace.py")
        self.assertTrue(module.GPU_TOOL_NAMES)
        for name in module.GPU_TOOL_NAMES:
            self.assertTrue(name.startswith("gpu_"), name)
            self.assertTrue((SCRIPTS / name).is_file(), name)

    def test_workspace_config_resolves_direct_gpu_target(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False
        ) as stream:
            path = pathlib.Path(stream.name)
            json.dump(
                {
                    "schema_version": 1,
                    "host": "192.0.2.8",
                    "user": "gpu",
                    "port": 2202,
                    "container": "vllm-gpu",
                },
                stream,
            )
        self.addCleanup(path.unlink, missing_ok=True)
        args = argparse.Namespace(
            workspace_config=path,
            host=None,
            user=None,
            port=None,
            container=None,
            identity_file=None,
            ssh_config=None,
            _gpu_require_container=True,
        )
        target = target_from_args(args)
        self.assertEqual(
            target,
            GpuTarget(host="192.0.2.8", user="gpu", port=2202, container="vllm-gpu"),
        )

    def test_code_parity_archive_is_vllm_only_and_excludes_shared_libraries(
        self,
    ) -> None:
        module = load_script("gpu_code_parity.py")
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary) / "vllm"
            package = repo / "vllm"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (package / "runtime.json").write_text("{}\n", encoding="utf-8")
            (package / "local.so").write_bytes(b"local build must not move")
            (repo / "vllm-ascend").mkdir()
            archive, _, count = module.build_archive(repo)
            archive_path = pathlib.Path(temporary) / "source.tar.gz"
            archive_path.write_bytes(archive)
            with tarfile.open(archive_path, "r:gz") as stream:
                names = stream.getnames()
        self.assertEqual(count, 2)
        self.assertIn("vllm/__init__.py", names)
        self.assertNotIn("vllm/local.so", names)
        self.assertFalse(any("vllm-ascend" in name for name in names))

    def test_code_parity_remote_script_preserves_runtime_and_has_rollback(self) -> None:
        module = load_script("gpu_code_parity.py")
        script = module.render_remote_script(
            GpuTarget(host="192.0.2.8", container="vllm-gpu"),
            "a" * 64,
            "/tmp/source.tar.gz",
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        self.assertIn('cp -a "$vaws_root/vllm" "$vaws_stage"', script)
        self.assertIn("vllm.rollback-", script)
        self.assertIn("import os,vllm", script)
        self.assertIn("vaws_archive=/tmp/source.tar.gz", script)
        self.assertNotIn('tee "$vaws_archive"', script)
        self.assertNotIn("torch_npu", script)

    def test_serving_parser_and_remote_script_are_shell_valid(self) -> None:
        module = load_script("gpu_serving.py")
        args = module.build_parser().parse_args(
            [
                "--host",
                "192.0.2.8",
                "--container",
                "vllm-gpu",
                "start",
                "--name",
                "smoke",
                "--model",
                "/models/test",
                "--service-port",
                "8100",
                "--cuda-visible-devices",
                "7",
            ]
        )
        script = module.remote_script(
            args, GpuTarget(host="192.0.2.8", container="vllm-gpu")
        )
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        self.assertIn("vllm.entrypoints.openai.api_server", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=7", script)
        self.assertIn('test -e /models/test', script)
        self.assertNotIn("torch_npu", script)

    def test_change_plan_requests_multi_gpu_and_benchmark_for_attention(self) -> None:
        module = load_script("gpu_change_validation.py")
        categories = module.classify(["vllm/v1/attention/backends/flash_attn.py"])
        checks = {item["check"] for item in module.validation_plan(categories)}
        self.assertIn("multi_gpu_smoke", checks)
        self.assertIn("gpu_serving_benchmark", checks)

    def test_performance_directionality(self) -> None:
        module = load_script("gpu_performance_regression.py")
        results = module.compare(
            {"request_throughput": 100.0, "mean_ttft_ms": 10.0},
            {"request_throughput": 94.0, "mean_ttft_ms": 10.4},
            {"*": 5.0},
        )
        by_name = {item["metric"]: item for item in results}
        self.assertFalse(by_name["request_throughput"]["passed"])
        self.assertTrue(by_name["mean_ttft_ms"]["passed"])


if __name__ == "__main__":
    unittest.main()
