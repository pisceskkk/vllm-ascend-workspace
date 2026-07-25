#!/usr/bin/env python3
"""Assess and apply guarded vLLM submodule upgrades."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[4]
LIB = ROOT / ".agents" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from vaws_run_manifest import (  # noqa: E402
    RunManifestError,
    add_artifact,
    load_manifest,
    new_manifest,
    transition_status,
    write_manifest,
)

SCHEMA_VERSION = 1


class UpstreamSyncError(ValueError):
    """Raised when refs, repositories, or apply preconditions are invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamSyncError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise UpstreamSyncError(f"{label} root must be an object")
    return payload


def git(repo: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise UpstreamSyncError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"git {' '.join(args)} failed in {repo}"
        )
    return result.stdout.strip()


def resolve_ref(repo: Path, ref: str) -> str:
    sha = git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if len(sha) != 40:
        raise UpstreamSyncError(f"ref did not resolve to a full commit: {ref}")
    return sha


def changed_files(repo: Path, old_sha: str, new_sha: str) -> list[dict[str, str]]:
    output = git(repo, ["diff", "--name-status", "--find-renames", old_sha, new_sha])
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) == 3:
            rows.append({"status": status, "old_path": parts[1], "path": parts[2]})
        elif len(parts) == 2:
            rows.append({"status": status, "path": parts[1]})
    return rows


def source_at(repo: Path, sha: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = node.args
    parts = [arg.arg for arg in arguments.posonlyargs]
    if arguments.posonlyargs:
        parts.append("/")
    parts.extend(arg.arg for arg in arguments.args)
    if arguments.vararg:
        parts.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        parts.append("*")
    parts.extend(arg.arg for arg in arguments.kwonlyargs)
    if arguments.kwarg:
        parts.append(f"**{arguments.kwarg.arg}")
    return f"{node.name}({', '.join(parts)})"


def api_signatures(source: str | None) -> dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures[node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            signatures[node.name] = f"class {node.name}"
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{node.name}.{child.name}"
                    signatures[key] = _signature(child)
    return signatures


def compare_api(
    repo: Path, old_sha: str, new_sha: str, paths: Sequence[str]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        if not path.endswith(".py"):
            continue
        old = api_signatures(source_at(repo, old_sha, path))
        new = api_signatures(source_at(repo, new_sha, path))
        for symbol in sorted(set(old) | set(new)):
            if symbol not in old:
                kind = "added"
            elif symbol not in new:
                kind = "removed"
            elif old[symbol] != new[symbol]:
                kind = "changed"
            else:
                continue
            changes.append(
                {
                    "path": path,
                    "symbol": symbol,
                    "kind": kind,
                    "old": old.get(symbol),
                    "new": new.get(symbol),
                }
            )
    return changes


def module_name(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    module = path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def imported_modules(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def find_consumers(
    ascend_repo: Path,
    changed: Sequence[Mapping[str, str]],
    api_changes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    changed_modules = {
        module
        for row in changed
        if (module := module_name(row["path"])) is not None
    }
    changed_symbols = {
        str(row["symbol"]).split(".")[-1]
        for row in api_changes
        if row["kind"] in {"removed", "changed"}
    }
    consumers: list[dict[str, Any]] = []
    for path in sorted(ascend_repo.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        imports = imported_modules(source)
        matched_modules = sorted(
            module
            for module in changed_modules
            if any(
                imported == module
                or imported.startswith(f"{module}.")
                or module.startswith(f"{imported}.")
                for imported in imports
            )
        )
        matched_symbols = sorted(
            symbol for symbol in changed_symbols if symbol and symbol in source
        )
        if matched_modules or matched_symbols:
            consumers.append(
                {
                    "path": str(path.relative_to(ascend_repo)),
                    "modules": matched_modules,
                    "symbols": matched_symbols,
                }
            )
    return consumers


def validation_recommendations(changed: Sequence[Mapping[str, str]]) -> list[str]:
    joined = "\n".join(row["path"].lower() for row in changed)
    recommendations = {"build:imports", "correctness:single-card-smoke"}
    rules = (
        (("attention", "kv_cache", "kv_connector"), "correctness:attention-kv"),
        (("distributed", "parallel_state", "communicator"), "distributed:rank-smoke"),
        (("worker", "model_runner"), "correctness:model-runner"),
        (("scheduler",), "correctness:scheduler"),
        (("compile", "cudagraph", "graph"), "correctness:eager-graph"),
        (("custom_op", "ops"), "operator:isolated-matrix"),
    )
    for patterns, recommendation in rules:
        if any(pattern in joined for pattern in patterns):
            recommendations.add(recommendation)
    return sorted(recommendations)


def render_report(plan: Mapping[str, Any]) -> str:
    lines = [
        "# vLLM upstream sync report",
        "",
        f"- Old: `{plan['old_sha']}`",
        f"- New: `{plan['new_sha']}`",
        f"- Risk: **{plan['risk']}**",
        f"- Changed paths: `{len(plan['changed_files'])}`",
        f"- API changes: `{len(plan['api_changes'])}`",
        f"- vllm-ascend consumers: `{len(plan['consumers'])}`",
        "",
        "## Recommended validation",
        "",
    ]
    lines.extend(f"- `{item}`" for item in plan["recommended_validation"])
    lines.extend(["", "## API changes", "", "```json"])
    lines.append(json.dumps(plan["api_changes"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Consumers", "", "```json"])
    lines.append(json.dumps(plan["consumers"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def plan(
    output_dir: Path,
    *,
    vllm_repo: Path,
    ascend_repo: Path,
    old_ref: str,
    new_ref: str,
    run_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UpstreamSyncError(f"output directory is not empty: {output_dir}")
    if not (vllm_repo / ".git").exists():
        raise UpstreamSyncError(f"vLLM repository is not initialized: {vllm_repo}")
    if not (ascend_repo / ".git").exists():
        raise UpstreamSyncError(
            f"vllm-ascend repository is not initialized: {ascend_repo}"
        )
    old_sha = resolve_ref(vllm_repo, old_ref)
    new_sha = resolve_ref(vllm_repo, new_ref)
    if old_sha == new_sha:
        raise UpstreamSyncError("old and new refs resolve to the same commit")
    changed = changed_files(vllm_repo, old_sha, new_sha)
    paths = [row["path"] for row in changed]
    api = compare_api(vllm_repo, old_sha, new_sha, paths)
    consumers = find_consumers(ascend_repo, changed, api)
    risky_api = [row for row in api if row["kind"] in {"removed", "changed"}]
    risk = "high" if risky_api and consumers else "medium" if consumers or risky_api else "low"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "planned",
        "vllm_repo": str(vllm_repo.resolve()),
        "ascend_repo": str(ascend_repo.resolve()),
        "old_ref": old_ref,
        "new_ref": new_ref,
        "old_sha": old_sha,
        "new_sha": new_sha,
        "risk": risk,
        "changed_files": changed,
        "api_changes": api,
        "consumers": consumers,
        "recommended_validation": validation_recommendations(changed),
        "apply_preconditions": {
            "vllm_worktree_clean": True,
            "current_head": old_sha,
        },
    }
    timestamp = created_at or utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "sync-plan.json", payload)
    (output_dir / "report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    manifest = new_manifest(
        run_type="change-validation",
        run_id=run_id,
        workspace_snapshot={"vllm_old": old_sha, "vllm_new": new_sha},
        created_at=timestamp,
    )
    for name, kind, uri in (
        ("sync-plan", "upstream-sync-plan", "sync-plan.json"),
        ("report", "report", "report.md"),
    ):
        manifest = add_artifact(
            manifest, name=name, kind=kind, uri=uri, updated_at=timestamp
        )
    write_manifest(output_dir / "manifest.json", manifest)
    return {
        "status": "planned",
        "risk": risk,
        "old_sha": old_sha,
        "new_sha": new_sha,
        "changed_paths": len(changed),
        "api_changes": len(api),
        "consumers": len(consumers),
    }


def apply(
    output_dir: Path, *, updated_at: str | None = None
) -> dict[str, Any]:
    sync_plan = _load_json(output_dir / "sync-plan.json", "sync plan")
    if sync_plan.get("status") != "planned":
        raise UpstreamSyncError(f"sync plan must be planned, got {sync_plan.get('status')}")
    repo = Path(sync_plan["vllm_repo"])
    dirty = git(repo, ["status", "--porcelain=v1", "--untracked-files=normal"])
    if dirty:
        raise UpstreamSyncError("vLLM worktree is dirty; refusing to move submodule")
    current = resolve_ref(repo, "HEAD")
    if current != sync_plan["old_sha"]:
        raise UpstreamSyncError(
            f"vLLM HEAD changed since plan: expected {sync_plan['old_sha']}, got {current}"
        )
    git(repo, ["checkout", "--detach", sync_plan["new_sha"]])
    observed = resolve_ref(repo, "HEAD")
    if observed != sync_plan["new_sha"]:
        raise UpstreamSyncError(
            f"checkout verification failed: expected {sync_plan['new_sha']}, got {observed}"
        )
    timestamp = updated_at or utc_now()
    sync_plan["status"] = "applied"
    sync_plan["applied_at"] = timestamp
    sync_plan["observed_head"] = observed
    _atomic_write(output_dir / "sync-plan.json", sync_plan)
    manifest = load_manifest(output_dir / "manifest.json")
    manifest = transition_status(manifest, "running", updated_at=timestamp)
    manifest = transition_status(manifest, "passed", updated_at=timestamp)
    write_manifest(output_dir / "manifest.json", manifest)
    return {
        "status": "applied",
        "old_sha": sync_plan["old_sha"],
        "new_sha": sync_plan["new_sha"],
        "next_action": "run vllm-ascend-change-validation on the parent workspace diff",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output-dir", required=True, type=Path)
    plan_parser.add_argument("--vllm-repo", type=Path, default=ROOT / "vllm")
    plan_parser.add_argument(
        "--ascend-repo", type=Path, default=ROOT / "vllm-ascend"
    )
    plan_parser.add_argument("--old-ref", required=True)
    plan_parser.add_argument("--new-ref", required=True)
    plan_parser.add_argument("--run-id", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "plan":
            payload = plan(
                args.output_dir,
                vllm_repo=args.vllm_repo,
                ascend_repo=args.ascend_repo,
                old_ref=args.old_ref,
                new_ref=args.new_ref,
                run_id=args.run_id,
            )
        else:
            payload = apply(args.output_dir)
    except (UpstreamSyncError, RunManifestError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
