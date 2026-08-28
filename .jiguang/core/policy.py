"""Account-owned-resource policy and payload validation."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any

from .errors import JiguangPolicyError

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
FORBIDDEN_KEY_RE = re.compile(
    r"(?:^|_)(?:script|command|authorization|cookie|access_token|refresh_token|id_token|token|secret|password|passwd|credential|private_?key)(?:_|$)",
    re.IGNORECASE,
)
OWNER_KEYS = (
    "owner_user_id",
    "owner_id",
    "user_id",
    "created_by_user_id",
)

DEPLOYMENT_INPUT_KEYS = frozenset(
    {
        "device_ids",
        "scenario_name",
        "selected_software",
        "python_version",
        "pytorch_version",
        "code_repo",
        "code_branch",
        "commit_sha",
        "metadata",
    }
)
EVALUATION_INPUT_KEYS = frozenset(
    {
        "name",
        "description",
        "app_id",
        "deployment_id",
        "device_ids",
        "evaluation_type",
        "model",
        "dataset",
        "dataset_version",
        "dataset_split",
        "configuration",
        "repetitions",
        "warmup_runs",
        "commit_sha",
        "submodule_shas",
        "metadata",
    }
)
POOL_CREATE_KEYS = frozenset(
    {
        "name",
        "region",
        "schedule_types",
        "coordinator_display_name",
        "coordinator_contact",
        "capabilities",
    }
)
POOL_UPDATE_KEYS = POOL_CREATE_KEYS | {"operational_status", "version"}
DEVICE_REGISTRATION_KEYS = frozenset(
    {
        "name",
        "device_type",
        "ip",
        "port",
        "ssh_username",
        "auth_type",
        "credential_target",
        "tags",
        "remark",
        "device_model",
        "npu_type",
        "arch",
        "card_ids",
        "pool_id",
    }
)
CREDENTIAL_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_.-]{2,127}$")


def require_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise JiguangPolicyError(f"invalid {label}")
    return value


def _reject_forbidden_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            is_credential_reference = key_text in {"credential_target", "secret_credential_target"}
            if FORBIDDEN_KEY_RE.search(key_text) and not is_credential_reference:
                raise JiguangPolicyError(f"forbidden field at {path}.{key_text}")
            _reject_forbidden_keys(item, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}[{index}]")


def validate_payload(payload: Any, allowed_keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JiguangPolicyError(f"{label} must be an object")
    unknown = sorted(set(map(str, payload)) - allowed_keys)
    if unknown:
        raise JiguangPolicyError(f"unsupported {label} fields: {', '.join(unknown)}")
    normalized = dict(payload)
    _reject_forbidden_keys(normalized)
    return normalized


def subject_id(profile: Mapping[str, Any]) -> str:
    for key in ("id", "user_id", "sub"):
        value = profile.get(key)
        if value is not None and str(value):
            return str(value)
    raise JiguangPolicyError("profile did not contain a stable subject id")


def assert_non_admin_profile(profile: Mapping[str, Any]) -> None:
    role = str(profile.get("role") or profile.get("system_role") or "").lower()
    if role in {"admin", "administrator", "system_admin", "super_admin"}:
        raise JiguangPolicyError("administrator accounts are outside this integration boundary")


def assert_owned(resource: Any, expected_subject: str, label: str) -> None:
    """Require explicit ownership evidence for one account-scoped resource."""
    if not isinstance(resource, Mapping):
        raise JiguangPolicyError(f"{label} response did not contain an owned resource")
    for key in OWNER_KEYS:
        value = resource.get(key)
        if value is not None:
            if str(value) != expected_subject:
                raise JiguangPolicyError(f"{label} is not owned by the authenticated account")
            return
    raise JiguangPolicyError(f"{label} response did not contain ownership evidence")


def assert_owned_collection(payload: Any, expected_subject: str, label: str) -> None:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        if any(payload.get(key) is not None for key in OWNER_KEYS):
            items = [payload]
        else:
            items = next(
                (
                    payload[key]
                    for key in ("items", "data", "results", "devices", "pools", "deployments", "tasks")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
            if items is None and isinstance(payload.get("data"), Mapping):
                return assert_owned_collection(payload["data"], expected_subject, label)
            if items is None:
                raise JiguangPolicyError(f"{label} collection did not contain ownership evidence")
    else:
        raise JiguangPolicyError(f"{label} collection response must be an object or array")
    for item in items:
        assert_owned(item, expected_subject, label)


def require_confirmation(confirm: Any, action: str) -> None:
    if confirm is not True:
        raise JiguangPolicyError(f"{action} requires confirm=true")


def deployment_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_payload(payload, DEPLOYMENT_INPUT_KEYS, "deployment")
    device_ids = normalized.get("device_ids")
    if not isinstance(device_ids, list) or not device_ids:
        raise JiguangPolicyError("deployment.device_ids must be a non-empty array")
    normalized["device_ids"] = [require_safe_id(value, "device_id") for value in device_ids]
    commit_sha = normalized.get("commit_sha")
    if not isinstance(commit_sha, str) or not GIT_SHA_RE.fullmatch(commit_sha):
        raise JiguangPolicyError("deployment.commit_sha must be a full lowercase Git SHA")
    required = ("scenario_name", "python_version", "pytorch_version", "code_repo", "code_branch")
    missing = [key for key in required if not isinstance(normalized.get(key), str) or not normalized[key].strip()]
    if missing:
        raise JiguangPolicyError(f"deployment missing required fields: {', '.join(missing)}")
    script = "\n".join(
        (
            "#!/bin/bash",
            "set -euo pipefail",
            "cd /vllm-workspace",
            "test -z \"$(git status --porcelain=v1 --untracked-files=normal)\"",
            "git fetch --all --prune",
            f"git checkout --detach {shlex.quote(commit_sha)}",
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(commit_sha)}",
            "git submodule update --init --recursive",
        )
    )
    return {
        "device_ids": normalized["device_ids"],
        "scenario_name": normalized["scenario_name"],
        "selected_software": normalized.get("selected_software", []),
        "python_version": normalized["python_version"],
        "pytorch_version": normalized["pytorch_version"],
        "code_repo": normalized["code_repo"],
        "code_branch": normalized["code_branch"],
        "deploy_mode": "container",
        "code_update_script": script,
    }


def pool_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_payload(payload, POOL_CREATE_KEYS, "resource pool")
    name = normalized.get("name")
    if not isinstance(name, str) or not (3 <= len(name.strip()) <= 128):
        raise JiguangPolicyError("resource pool name must contain 3-128 characters")
    schedule_types = normalized.get("schedule_types", ["server"])
    if not isinstance(schedule_types, list) or not schedule_types or any(
        value not in {"server"} for value in schedule_types
    ):
        raise JiguangPolicyError("only server scheduling is supported")
    return {
        "pool_type": "local_managed",
        "provider": "platform",
        "visibility": "discoverable",
        "name": name.strip(),
        "region": normalized.get("region"),
        "schedule_types": schedule_types,
        "coordinator_display_name": normalized.get("coordinator_display_name"),
        "coordinator_contact": normalized.get("coordinator_contact"),
        "capabilities": normalized.get("capabilities", {}),
    }


def pool_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_payload(payload, POOL_UPDATE_KEYS, "resource pool update")
    version = normalized.get("version")
    if not isinstance(version, int) or version < 0:
        raise JiguangPolicyError("resource pool update requires a non-negative integer version")
    if "name" in normalized and (
        not isinstance(normalized["name"], str) or not 3 <= len(normalized["name"].strip()) <= 128
    ):
        raise JiguangPolicyError("resource pool name must contain 3-128 characters")
    if "schedule_types" in normalized and (
        not isinstance(normalized["schedule_types"], list)
        or not normalized["schedule_types"]
        or any(value != "server" for value in normalized["schedule_types"])
    ):
        raise JiguangPolicyError("only server scheduling is supported")
    if normalized.get("operational_status") not in {None, "active", "draining", "disabled"}:
        raise JiguangPolicyError("unsupported resource pool operational_status")
    return normalized


def device_registration_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str | None]:
    normalized = validate_payload(payload, DEVICE_REGISTRATION_KEYS, "device registration")
    required = ("name", "device_type", "ip", "port", "ssh_username", "auth_type", "credential_target")
    missing = [key for key in required if normalized.get(key) is None or normalized.get(key) == ""]
    if missing:
        raise JiguangPolicyError(f"device registration missing fields: {', '.join(missing)}")
    if normalized["device_type"] not in {"physical_machine", "container"}:
        raise JiguangPolicyError("only physical_machine and container devices are supported")
    if normalized["auth_type"] not in {"PASSWORD", "SSH_KEY", "KEY"}:
        raise JiguangPolicyError("unsupported device auth_type")
    port = normalized["port"]
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise JiguangPolicyError("device port must be between 1 and 65535")
    credential_target = str(normalized.pop("credential_target"))
    if not CREDENTIAL_TARGET_RE.fullmatch(credential_target):
        raise JiguangPolicyError("invalid Windows credential target")
    pool_id = normalized.pop("pool_id", None)
    if pool_id is not None:
        pool_id = require_safe_id(pool_id, "pool_id")
    normalized["auth_type"] = "KEY" if normalized["auth_type"] in {"SSH_KEY", "KEY"} else "PASSWORD"
    return normalized, credential_target, pool_id


def evaluation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_payload(payload, EVALUATION_INPUT_KEYS, "evaluation")
    evaluation_type = normalized.get("evaluation_type")
    if evaluation_type not in {"accuracy", "performance"}:
        raise JiguangPolicyError("evaluation_type must be accuracy or performance")
    required = (
        "name",
        "app_id",
        "deployment_id",
        "device_ids",
        "model",
        "commit_sha",
        "submodule_shas",
        "configuration",
    )
    missing = [key for key in required if normalized.get(key) is None or normalized.get(key) == ""]
    if missing:
        raise JiguangPolicyError(f"evaluation missing required fields: {', '.join(missing)}")
    if not isinstance(normalized["configuration"], Mapping) or not normalized["configuration"]:
        raise JiguangPolicyError("evaluation.configuration must be a non-empty object")
    if not isinstance(normalized["commit_sha"], str) or not GIT_SHA_RE.fullmatch(normalized["commit_sha"]):
        raise JiguangPolicyError("evaluation.commit_sha must be a full lowercase Git SHA")
    submodules = normalized["submodule_shas"]
    if not isinstance(submodules, Mapping) or not submodules:
        raise JiguangPolicyError("evaluation.submodule_shas must be a non-empty object")
    for name, sha in submodules.items():
        if not isinstance(name, str) or not isinstance(sha, str) or not GIT_SHA_RE.fullmatch(sha):
            raise JiguangPolicyError("evaluation contains an invalid submodule SHA")
    require_safe_id(normalized["app_id"], "app_id")
    require_safe_id(normalized["deployment_id"], "deployment_id")
    if evaluation_type == "accuracy":
        accuracy_required = ("dataset", "dataset_version", "dataset_split")
        missing_accuracy = [key for key in accuracy_required if not normalized.get(key)]
        if missing_accuracy:
            raise JiguangPolicyError(f"accuracy evaluation missing fields: {', '.join(missing_accuracy)}")
    else:
        repetitions = normalized.get("repetitions")
        warmup_runs = normalized.get("warmup_runs")
        if not isinstance(repetitions, int) or repetitions < 1:
            raise JiguangPolicyError("performance evaluation requires positive repetitions")
        if not isinstance(warmup_runs, int) or warmup_runs < 1:
            raise JiguangPolicyError("performance evaluation requires positive warmup_runs")
    device_ids = normalized.get("device_ids")
    if device_ids is not None:
        if not isinstance(device_ids, list) or not device_ids:
            raise JiguangPolicyError("evaluation.device_ids must be a non-empty array when present")
        normalized["device_ids"] = [require_safe_id(value, "device_id") for value in device_ids]
    return normalized
