"""Tool dispatch for the account-owned Jiguang client."""

from __future__ import annotations

import json
from typing import Any

from core import HostProcessTransport, JiguangClient
from core.redaction import redact
from mcp.schemas import ALIASES, TOOL_SCHEMAS, validate_tool_arguments

DESCRIPTIONS = {
    "jiguang.account_get": "Read the authenticated Jiguang user profile without exposing credentials.",
    "jiguang.own_resource_pools_list": "List resource pools visible to and owned by the current account.",
    "jiguang.own_resource_pool_get": "Read one current-account resource pool.",
    "jiguang.resource_pool_create_plan": "Validate creation of a local-managed resource pool owned by the current account.",
    "jiguang.resource_pool_create_apply": "Create a current-account local-managed resource pool after explicit confirmation.",
    "jiguang.resource_pool_update_plan": "Validate an update to a current-account resource pool.",
    "jiguang.resource_pool_update_apply": "Update a current-account resource pool after explicit confirmation.",
    "jiguang.resource_pool_delete_apply": "Delete a current-account resource pool after explicit confirmation and ownership check.",
    "jiguang.own_devices_list": "List current-account physical machines and connected containers.",
    "jiguang.own_device_get": "Read one current-account device record.",
    "jiguang.device_registration_plan": "Validate registration of an owned physical machine or existing container using a Windows credential reference.",
    "jiguang.device_registration_apply": "Register an owned physical machine or existing container after explicit confirmation without returning its SSH secret.",
    "jiguang.device_delete_apply": "Delete one current-account device record after explicit confirmation and ownership check.",
    "jiguang.own_deployments_list": "List current-account existing-container connections.",
    "jiguang.own_deployment_get": "Read one current-account container connection.",
    "jiguang.container_connection_plan": "Validate a plan to connect an already-created container; this never creates a container.",
    "jiguang.container_connection_apply": "Connect an already-created current-account container after explicit confirmation.",
    "jiguang.evaluation_catalog_list": "List Jiguang evaluation apps, templates, and software catalog entries.",
    "jiguang.evaluations_list": "List evaluation tasks owned by the current account.",
    "jiguang.evaluation_plan": "Validate an accuracy or performance evaluation without submitting it.",
    "jiguang.evaluation_submit": "Submit a validated evaluation after explicit confirmation.",
    "jiguang.evaluation_get": "Read one current-account evaluation task.",
    "jiguang.evaluation_cancel": "Interrupt one current-account evaluation after explicit confirmation.",
    "jiguang.evaluation_artifacts": "Read normalized logs/artifact metadata for one evaluation.",
}


def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": DESCRIPTIONS[name], "inputSchema": schema}
        for name, schema in TOOL_SCHEMAS.items()
    ]


def _client() -> JiguangClient:
    return JiguangClient(HostProcessTransport())


def _success(value: Any) -> dict[str, Any]:
    safe = redact(value)
    result = {"outcome": "success", "data": safe}
    return {"text": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), "result": result}


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    name = ALIASES.get(name, name)
    args = validate_tool_arguments(name, arguments or {})
    client = _client()
    if name == "jiguang.account_get":
        return _success(client.profile(refresh=args.get("refresh", False)))
    if name == "jiguang.own_resource_pools_list":
        return _success(client.list_pools(args.get("query")))
    if name == "jiguang.own_resource_pool_get":
        return _success(client.get_pool(str(args["pool_id"])))
    if name == "jiguang.resource_pool_create_plan":
        return _success(client.pool_create_plan(dict(args["payload"])))
    if name == "jiguang.resource_pool_create_apply":
        return _success(client.pool_create_apply(dict(args["payload"]), confirm=args["confirm"]))
    if name == "jiguang.resource_pool_update_plan":
        return _success(client.pool_update_plan(str(args["pool_id"]), dict(args["payload"])))
    if name == "jiguang.resource_pool_update_apply":
        return _success(client.pool_update_apply(str(args["pool_id"]), dict(args["payload"]), confirm=args["confirm"]))
    if name == "jiguang.resource_pool_delete_apply":
        return _success(client.pool_delete_apply(str(args["pool_id"]), confirm=args["confirm"]))
    if name == "jiguang.own_devices_list":
        return _success(client.list_devices(args.get("query")))
    if name == "jiguang.own_device_get":
        return _success(client.get_device(str(args["device_id"])))
    if name == "jiguang.device_registration_plan":
        return _success(client.device_registration_plan(dict(args["payload"])))
    if name == "jiguang.device_registration_apply":
        return _success(client.device_registration_apply(dict(args["payload"]), confirm=args["confirm"]))
    if name == "jiguang.device_delete_apply":
        return _success(client.device_delete_apply(str(args["device_id"]), confirm=args["confirm"]))
    if name == "jiguang.own_deployments_list":
        return _success(client.list_deployments(args.get("query")))
    if name == "jiguang.own_deployment_get":
        return _success(client.get_deployment(str(args["deployment_id"])))
    if name == "jiguang.container_connection_plan":
        return _success(client.deployment_plan(dict(args["payload"])))
    if name == "jiguang.container_connection_apply":
        return _success(client.deployment_apply(dict(args["payload"]), confirm=args["confirm"]))
    if name == "jiguang.evaluation_catalog_list":
        return _success(client.evaluation_catalog())
    if name == "jiguang.evaluations_list":
        return _success(client.list_evaluations(args.get("query")))
    if name == "jiguang.evaluation_plan":
        return _success(client.evaluation_plan(dict(args["payload"])))
    if name == "jiguang.evaluation_submit":
        return _success(client.evaluation_submit(dict(args["payload"]), confirm=args["confirm"]))
    if name == "jiguang.evaluation_get":
        return _success(client.get_evaluation(str(args["task_id"])))
    if name == "jiguang.evaluation_cancel":
        return _success(client.cancel_evaluation(str(args["task_id"]), confirm=args["confirm"]))
    if name == "jiguang.evaluation_artifacts":
        return _success(client.evaluation_artifacts(str(args["task_id"])))
    raise KeyError(f"unknown Jiguang tool: {name}")
