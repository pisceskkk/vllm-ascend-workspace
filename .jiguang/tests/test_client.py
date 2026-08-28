from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.client import JiguangClient  # noqa: E402
from core.errors import JiguangPolicyError  # noqa: E402


class FakeTransport:
    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        return {"ok": True, "status": 200, "data": self.routes[(method, path)]}


class ClientTests(unittest.TestCase):
    def test_lists_owned_devices(self) -> None:
        transport = FakeTransport(
            {
                ("GET", "/api/v1/device-management/devices"): {"items": [{"id": "d1", "owner_user_id": "635"}]},
                ("GET", "/api/v1/user-management/users/profile"): {"id": "635", "role": "user"},
            }
        )
        result = JiguangClient(transport).list_devices()
        self.assertEqual(result["items"][0]["id"], "d1")

    def test_rejects_foreign_device(self) -> None:
        transport = FakeTransport(
            {
                ("GET", "/api/v1/device-management/devices"): {"items": [{"id": "d1", "owner_user_id": "999"}]},
                ("GET", "/api/v1/user-management/users/profile"): {"id": "635"},
            }
        )
        with self.assertRaises(JiguangPolicyError):
            JiguangClient(transport).list_devices()

    def test_mutation_requires_confirmation(self) -> None:
        client = JiguangClient(FakeTransport({}))
        with self.assertRaises(JiguangPolicyError):
            client.evaluation_submit({}, confirm=False)

    def test_connection_plan_never_claims_container_creation(self) -> None:
        client = JiguangClient(FakeTransport({}))
        plan = client.deployment_plan(
            {
                "device_ids": ["device-1"],
                "scenario_name": "inference",
                "selected_software": [],
                "python_version": "3.11",
                "pytorch_version": "2.7.1",
                "code_repo": "vllm-ascend",
                "code_branch": "main",
                "commit_sha": "a" * 40,
            }
        )
        self.assertFalse(plan["platform_creates_container"])
        self.assertEqual(plan["api_request"]["deploy_mode"], "container")
        self.assertIn("git checkout --detach", plan["api_request"]["code_update_script"])

    def test_device_registration_uses_credential_reference(self) -> None:
        client = JiguangClient(FakeTransport({}))
        plan = client.device_registration_plan(
            {
                "name": "npu-container-01",
                "device_type": "container",
                "ip": "192.0.2.1",
                "port": 22022,
                "ssh_username": "root",
                "auth_type": "SSH_KEY",
                "credential_target": "Codex:Jiguang:Device:m1",
            }
        )
        self.assertNotIn("secret", plan["api_request"])
        self.assertEqual(plan["secret_transport"], "windows_credential_manager")

    def test_pool_plan_forces_local_managed(self) -> None:
        client = JiguangClient(FakeTransport({}))
        plan = client.pool_create_plan({"name": "owned-pool", "schedule_types": ["server"]})
        self.assertEqual(plan["api_request"]["pool_type"], "local_managed")
        self.assertEqual(plan["api_request"]["provider"], "platform")

    def test_performance_plan_requires_warmup_and_repetitions(self) -> None:
        client = JiguangClient(FakeTransport({}))
        plan = client.evaluation_plan(
            {
                "name": "perf",
                "app_id": "app-1",
                "deployment_id": "deployment-1",
                "device_ids": ["device-1"],
                "evaluation_type": "performance",
                "model": "model",
                "configuration": {"concurrency": 8},
                "repetitions": 3,
                "warmup_runs": 1,
                "commit_sha": "a" * 40,
                "submodule_shas": {"vllm": "b" * 40, "vllm-ascend": "c" * 40},
            }
        )
        self.assertEqual(plan["request"]["repetitions"], 3)

    def test_cancel_rejects_foreign_evaluation_before_mutation(self) -> None:
        transport = FakeTransport(
            {
                ("GET", "/api/v1/task-management/tasks/task-1"): {
                    "id": "task-1",
                    "owner_user_id": "999",
                    "status": "running",
                },
                ("GET", "/api/v1/user-management/users/profile"): {"id": "635", "role": "user"},
            }
        )
        with self.assertRaises(JiguangPolicyError):
            JiguangClient(transport).cancel_evaluation("task-1", confirm=True)
        self.assertFalse(any(method == "POST" for method, _path, _kwargs in transport.calls))

    def test_artifacts_require_owned_terminal_evaluation(self) -> None:
        transport = FakeTransport(
            {
                ("GET", "/api/v1/task-management/tasks/task-1"): {
                    "id": "task-1",
                    "owner_user_id": "635",
                    "status": "running",
                },
                ("GET", "/api/v1/user-management/users/profile"): {"id": "635", "role": "user"},
            }
        )
        with self.assertRaisesRegex(ValueError, "terminal status"):
            JiguangClient(transport).evaluation_artifacts("task-1")
        self.assertFalse(any(path.endswith("/logs") for _method, path, _kwargs in transport.calls))


if __name__ == "__main__":
    unittest.main()
