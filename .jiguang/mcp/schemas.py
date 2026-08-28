"""JSON schemas for the narrow Jiguang MCP surface."""

from __future__ import annotations

import re
from typing import Any


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required or [],
    }


QUERY = {"query": {"type": "object", "additionalProperties": True}}
PAYLOAD = {"payload": {"type": "object", "additionalProperties": True}}
CONFIRM = {"confirm": {"type": "boolean", "description": "Must be true for this mutation."}}
ID = {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "jiguang.account_get": schema({"refresh": {"type": "boolean", "default": False}}),
    "jiguang.own_resource_pools_list": schema(QUERY),
    "jiguang.own_resource_pool_get": schema({"pool_id": ID}, ["pool_id"]),
    "jiguang.resource_pool_create_plan": schema(PAYLOAD, ["payload"]),
    "jiguang.resource_pool_create_apply": schema({**PAYLOAD, **CONFIRM}, ["payload", "confirm"]),
    "jiguang.resource_pool_update_plan": schema({"pool_id": ID, **PAYLOAD}, ["pool_id", "payload"]),
    "jiguang.resource_pool_update_apply": schema({"pool_id": ID, **PAYLOAD, **CONFIRM}, ["pool_id", "payload", "confirm"]),
    "jiguang.resource_pool_delete_apply": schema({"pool_id": ID, **CONFIRM}, ["pool_id", "confirm"]),
    "jiguang.own_devices_list": schema(QUERY),
    "jiguang.own_device_get": schema({"device_id": ID}, ["device_id"]),
    "jiguang.device_registration_plan": schema(PAYLOAD, ["payload"]),
    "jiguang.device_registration_apply": schema({**PAYLOAD, **CONFIRM}, ["payload", "confirm"]),
    "jiguang.device_delete_apply": schema({"device_id": ID, **CONFIRM}, ["device_id", "confirm"]),
    "jiguang.own_deployments_list": schema(QUERY),
    "jiguang.own_deployment_get": schema({"deployment_id": ID}, ["deployment_id"]),
    "jiguang.container_connection_plan": schema(PAYLOAD, ["payload"]),
    "jiguang.container_connection_apply": schema({**PAYLOAD, **CONFIRM}, ["payload", "confirm"]),
    "jiguang.evaluation_catalog_list": schema({}),
    "jiguang.evaluations_list": schema(QUERY),
    "jiguang.evaluation_plan": schema(PAYLOAD, ["payload"]),
    "jiguang.evaluation_submit": schema({**PAYLOAD, **CONFIRM}, ["payload", "confirm"]),
    "jiguang.evaluation_get": schema({"task_id": ID}, ["task_id"]),
    "jiguang.evaluation_cancel": schema({"task_id": ID, **CONFIRM}, ["task_id", "confirm"]),
    "jiguang.evaluation_artifacts": schema({"task_id": ID}, ["task_id"]),
}

ALIASES = {name.replace(".", "_"): name for name in TOOL_SCHEMAS}


def validate_tool_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Validate the small MCP schema subset before any transport is created."""
    if name not in TOOL_SCHEMAS:
        raise KeyError(f"unknown Jiguang tool: {name}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    schema_value = TOOL_SCHEMAS[name]
    properties = schema_value["properties"]
    missing = [key for key in schema_value["required"] if key not in arguments]
    if missing:
        raise ValueError(f"missing required arguments: {', '.join(missing)}")
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ValueError(f"unsupported arguments: {', '.join(unknown)}")
    for key, value in arguments.items():
        specification = properties[key]
        expected = specification.get("type")
        if expected == "boolean" and type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")
        if expected == "object" and not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        pattern = specification.get("pattern")
        if pattern and isinstance(value, str) and re.fullmatch(pattern, value) is None:
            raise ValueError(f"{key} has an invalid format")
    return dict(arguments)
