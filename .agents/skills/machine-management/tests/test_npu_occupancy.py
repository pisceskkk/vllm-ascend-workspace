from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / ".agents" / "skills" / "machine-management" / "scripts" / "npu_occupancy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("npu_occupancy", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_npu_smi_info_devices_and_processes():
    mod = load_module()
    sample = """
+--------------------------------------------------------------------------------+
| NPU   Name        | Health          Power(W)    Temp(C)           Hugepages-Usage |
| 0     910B4       | OK              91.8        41                0 / 0           |
| 1     910B4       | OK              90.4        39                0 / 0           |
| NPU   Chip        | Bus-Id          AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)   |
| 0     0           | 0000:C1:00.0    87          1024 / 2048       32768 / 65536   |
| 1     0           | 0000:C2:00.0    0           0 / 2048          64 / 65536      |
| NPU   Chip        | Process id      Process name                 Process memory(MB) |
| 0     0           12345            python                       32768              |
+--------------------------------------------------------------------------------+
"""
    parsed = mod.parse_npu_smi_info(sample)
    assert parsed["devices"][0]["npu_id"] == 0
    assert parsed["devices"][0]["name"] == "910B4"
    assert parsed["devices"][0]["aicore_percent"] == 87
    assert parsed["devices"][0]["hbm"] == {"used_mb": 32768, "total_mb": 65536}
    assert parsed["devices"][1]["npu_id"] == 1
    assert parsed["devices"][1]["aicore_percent"] == 0
    assert parsed["process_records"] == [
        {
            "npu_id": 0,
            "chip_id": 0,
            "pid": 12345,
            "npu_process_name": "python",
            "npu_memory_mb": 32768,
        }
    ]


def test_parse_usages_overrides_aicore():
    mod = load_module()
    devices = [
        {
            "npu_id": 0,
            "aicore_percent": 0,
            "hbm_utilization_percent": None,
        }
    ]
    usages = mod.parse_npu_smi_usages(
        """
NPU ID                         : 0
Aicore Usage Rate(%)           : 55
HBM Usage Rate(%)              : 12
"""
    )
    mod.apply_usage_overrides(devices, usages)
    assert devices[0]["aicore_percent"] == 55
    assert devices[0]["hbm_utilization_percent"] == 12


def test_parse_dual_chip_phy_id_layout():
    mod = load_module()
    sample = """
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip  Phy-ID              | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
| 2     Ascend910           | OK            | 159.7       44                0    / 0             |
| 0     4                   | 0000:95:00.0  | 0           0    / 0          62449/ 65536         |
| 2     Ascend910           | OK            | -           42                0    / 0             |
| 1     5                   | 0000:97:00.0  | 12          0    / 0          62498/ 65536         |
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 2       0                 | 824166        | VLLMWorker_TP            | 59316                   |
| 2       1                 | 845053        | VLLMWorker_TP            | 59656                   |
"""
    parsed = mod.parse_npu_smi_info(sample)
    assert len(parsed["devices"]) == 1
    device = parsed["devices"][0]
    assert device["npu_id"] == 2
    assert device["aicore_percent"] == 12
    assert device["hbm"] == {"used_mb": 124947, "total_mb": 131072}
    assert [chip["phy_id"] for chip in device["chips"]] == [4, 5]
    assert [record["pid"] for record in parsed["process_records"]] == [824166, 845053]


def test_extract_container_id_from_cgroup_systemd_scope():
    mod = load_module()
    cid = "4f8ac0f1" * 8
    cgroup = f"0::/system.slice/docker-{cid}.scope"
    assert mod.extract_container_id_from_cgroup(cgroup) == cid


def test_render_table_contains_process_container_and_cwd():
    mod = load_module()
    output = mod.render_table(
        {
            "machine": "dcp14",
            "devices": [
                {
                    "npu_id": 0,
                    "aicore_percent": 70,
                    "hbm": {"used_mb": 32000, "total_mb": 65536},
                    "memory": {"used_mb": 0, "total_mb": 0},
                    "processes": [
                        {
                            "pid": 123,
                            "name": "python",
                            "cwd": "/vllm-workspace",
                            "container": {"name": "vaws-user"},
                        }
                    ],
                }
            ],
        }
    )
    assert "dcp14" in output
    assert "python" in output
    assert "vaws-user" in output
    assert "/vllm-workspace" in output
