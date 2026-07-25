#!/usr/bin/env python3
"""Regression tests for Serving SSH and npu-smi parsing."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = (
    ROOT / ".agents" / "skills" / "vllm-ascend-serving" / "scripts" / "_common.py"
)


def load_module():
    name = "_serving_common_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serving = load_module()


class ServingProbeTests(unittest.TestCase):
    def test_ssh_command_disables_tty_and_stdin(self) -> None:
        endpoint = serving.SshEndpoint(host="host", port=22, user="root")
        command = serving._ssh_base_cmd(endpoint)
        self.assertIn("-T", command)
        self.assertIn("-n", command)

    def test_two_column_chip_format_uses_device_column(self) -> None:
        output = """
| 0  4 | Healthy | 0000:81:00.0 | 0 / 0 | 128 / 65536 |
| 0  5 | Healthy | 0000:82:00.0 | 0 / 0 | 8192 / 65536 |
"""
        result = serving._parse_npu_smi(output)
        self.assertEqual(result["devices"], [4, 5])
        self.assertEqual(result["free"], [4])
        self.assertIn("5", result["busy"])

    def test_header_then_pci_format_uses_header_id(self) -> None:
        output = """
| 2  910B | Healthy |
| PCIe | 0000:81:00.0 | 0 / 0 | 128 / 65536 |
"""
        result = serving._parse_npu_smi(output)
        self.assertEqual(result["devices"], [2])
        self.assertEqual(result["hbm"]["2"]["used_mb"], 128)


if __name__ == "__main__":
    unittest.main()
