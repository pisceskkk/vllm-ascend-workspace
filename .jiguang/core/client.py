"""Narrow Jiguang client for resources owned by the authenticated account."""

from __future__ import annotations

from typing import Any

from .host_transport import Transport
from .policy import (
    DEPLOYMENT_INPUT_KEYS,
    assert_non_admin_profile,
    assert_owned,
    assert_owned_collection,
    deployment_api_payload,
    device_registration_payload,
    evaluation_payload,
    pool_create_payload,
    pool_update_payload,
    require_confirmation,
    require_safe_id,
    subject_id,
    validate_payload,
)
from .redaction import redact


class JiguangClient:
    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self._profile: dict[str, Any] | None = None

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.transport.request(method, path, **kwargs)
        return redact(response.get("data"))

    def profile(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._profile is None or refresh:
            data = self._request("GET", "/api/v1/user-management/users/profile")
            if not isinstance(data, dict):
                raise ValueError("profile response must be an object")
            assert_non_admin_profile(data)
            self._profile = data
        return dict(self._profile)

    def _subject(self) -> str:
        return subject_id(self.profile())

    def list_pools(self, query: dict[str, Any] | None = None) -> Any:
        data = self._request("GET", "/api/v1/device-pool/pools", query=query)
        assert_owned_collection(data, self._subject(), "resource pool")
        return data

    def get_pool(self, pool_id: str) -> Any:
        data = self._request("GET", f"/api/v1/device-pool/pools/{require_safe_id(pool_id, 'pool_id')}")
        assert_owned(data, self._subject(), "resource pool")
        return data

    def pool_create_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": "planned", "action": "create_owned_pool", "api_request": pool_create_payload(payload)}

    def pool_create_apply(self, payload: dict[str, Any], *, confirm: bool) -> Any:
        require_confirmation(confirm, "resource pool creation")
        self.profile()
        return self._request("POST", "/api/v1/device-pool/pools", body=pool_create_payload(payload))

    def pool_update_plan(self, pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = pool_update_payload(payload)
        return {"outcome": "planned", "action": "update_owned_pool", "pool_id": require_safe_id(pool_id, "pool_id"), "api_request": normalized}

    def pool_update_apply(self, pool_id: str, payload: dict[str, Any], *, confirm: bool) -> Any:
        require_confirmation(confirm, "resource pool update")
        self.get_pool(pool_id)
        plan = self.pool_update_plan(pool_id, payload)
        return self._request("PATCH", f"/api/v1/device-pool/pools/{plan['pool_id']}", body=plan["api_request"])

    def pool_delete_apply(self, pool_id: str, *, confirm: bool) -> Any:
        require_confirmation(confirm, "resource pool deletion")
        safe_id = require_safe_id(pool_id, "pool_id")
        self.get_pool(safe_id)
        return self._request("DELETE", f"/api/v1/device-pool/pools/{safe_id}")

    def list_devices(self, query: dict[str, Any] | None = None) -> Any:
        data = self._request("GET", "/api/v1/device-management/devices", query=query)
        assert_owned_collection(data, self._subject(), "device")
        return data

    def get_device(self, device_id: str) -> Any:
        data = self._request("GET", f"/api/v1/device-management/devices/{require_safe_id(device_id, 'device_id')}")
        assert_owned(data, self._subject(), "device")
        return data

    def device_registration_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_payload, credential_target, pool_id = device_registration_payload(payload)
        return {
            "outcome": "planned",
            "action": "register_owned_device",
            "api_request": api_payload,
            "credential_target": credential_target,
            "pool_id": pool_id,
            "secret_transport": "windows_credential_manager",
        }

    def device_registration_apply(self, payload: dict[str, Any], *, confirm: bool) -> Any:
        require_confirmation(confirm, "device registration")
        self.profile()
        plan = self.device_registration_plan(payload)
        body = dict(plan["api_request"])
        body["__secret_credential_target"] = plan["credential_target"]
        body["__secret_field"] = "secret"
        device = self._request("POST", "/api/v1/device-management/devices", body=body, timeout_seconds=120)
        assert_owned(device, self._subject(), "device")
        if plan["pool_id"]:
            self.get_pool(plan["pool_id"])
            device_id = require_safe_id(str(device.get("device_id") or device.get("id")), "device_id")
            self._request(
                "POST",
                f"/api/v1/device-pool/pools/{plan['pool_id']}/devices",
                body={"device_id": device_id},
            )
        return device

    def device_delete_apply(self, device_id: str, *, confirm: bool) -> Any:
        require_confirmation(confirm, "device deletion")
        safe_id = require_safe_id(device_id, "device_id")
        self.get_device(safe_id)
        return self._request("DELETE", f"/api/v1/device-management/devices/{safe_id}")

    def list_deployments(self, query: dict[str, Any] | None = None) -> Any:
        data = self._request("GET", "/api/v1/deployment-management/deployments", query=query)
        assert_owned_collection(data, self._subject(), "deployment")
        return data

    def get_deployment(self, deployment_id: str) -> Any:
        data = self._request("GET", f"/api/v1/deployment-management/deployments/{require_safe_id(deployment_id, 'deployment_id')}")
        assert_owned(data, self._subject(), "deployment")
        return data

    def deployment_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_payload(payload, DEPLOYMENT_INPUT_KEYS, "deployment")
        api_payload = deployment_api_payload(normalized)
        return {
            "outcome": "planned",
            "action": "connect_existing_container",
            "request": normalized,
            "api_request": api_payload,
            "platform_creates_container": False,
            "code_update_policy": "fixed_exact_commit_checkout",
        }

    def deployment_apply(self, payload: dict[str, Any], *, confirm: bool) -> Any:
        require_confirmation(confirm, "container connection")
        plan = self.deployment_plan(payload)
        for device_id in plan["api_request"]["device_ids"]:
            device = self.get_device(device_id)
            if not isinstance(device, dict) or device.get("device_type") != "container":
                raise ValueError(f"device {device_id} is not an existing container record")
        return self._request(
            "POST",
            "/api/v1/deployment-management/deployments",
            body=plan["api_request"],
            timeout_seconds=120,
        )

    def evaluation_catalog(self) -> dict[str, Any]:
        return {
            "apps": self._request("GET", "/api/v1/apps/frontend/list"),
            "deployment_templates": self._request("GET", "/api/v1/deployment-management/deployment-templates"),
            "software_catalog": self._request("GET", "/api/v1/deployment-management/config/software-catalog"),
        }

    def evaluation_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = evaluation_payload(payload)
        return {"outcome": "planned", "request": normalized}

    def evaluation_submit(self, payload: dict[str, Any], *, confirm: bool) -> Any:
        require_confirmation(confirm, "evaluation submission")
        plan = self.evaluation_plan(payload)
        self.get_deployment(str(plan["request"]["deployment_id"]))
        for device_id in plan["request"].get("device_ids", []):
            self.get_device(str(device_id))
        return self._request(
            "POST",
            "/api/v1/task-management/tasks",
            body=plan["request"],
            timeout_seconds=120,
        )

    def list_evaluations(self, query: dict[str, Any] | None = None) -> Any:
        data = self._request("GET", "/api/v1/task-management/tasks", query=query)
        assert_owned_collection(data, self._subject(), "evaluation")
        return data

    def get_evaluation(self, task_id: str) -> Any:
        data = self._request("GET", f"/api/v1/task-management/tasks/{require_safe_id(task_id, 'task_id')}")
        assert_owned(data, self._subject(), "evaluation")
        return data

    def cancel_evaluation(self, task_id: str, *, confirm: bool) -> Any:
        require_confirmation(confirm, "evaluation cancellation")
        safe_id = require_safe_id(task_id, "task_id")
        self.get_evaluation(safe_id)
        return self._request("POST", f"/api/v1/task-management/tasks/{safe_id}/interrupt")

    def evaluation_artifacts(self, task_id: str) -> Any:
        safe_id = require_safe_id(task_id, "task_id")
        task = self.get_evaluation(safe_id)
        status = str(task.get("status") or "").lower() if isinstance(task, dict) else ""
        terminal = {"passed", "failed", "inconclusive", "cancelled", "canceled", "timeout", "completed", "success", "error"}
        if status not in terminal:
            raise ValueError("evaluation artifacts are available only after terminal status")
        return self._request("GET", f"/api/v1/task-management/tasks/{safe_id}/logs")
