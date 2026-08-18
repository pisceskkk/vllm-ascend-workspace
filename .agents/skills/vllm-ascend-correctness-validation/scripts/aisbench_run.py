#!/usr/bin/env python3
"""One-click AISBench accuracy workflow: serve, warm up, repeat, pull, freeze."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
LIB_DIR = ROOT / ".agents" / "lib"
BENCH_SCRIPTS = ROOT / ".agents" / "skills" / "vllm-ascend-benchmark" / "scripts"
for path in (SCRIPT_DIR, LIB_DIR, BENCH_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aisbench_adapter  # noqa: E402
from _common import assemble_config, call_serve_start, call_serve_stop  # noqa: E402
from vaws_remote_toolbox import (  # noqa: E402
    artifact_pull,
    artifact_push,
    remote_exec,
    resolve_remote_target,
)
from vaws_run_manifest import (  # noqa: E402
    add_artifact,
    new_manifest,
    sha256_file,
    transition_status,
    write_manifest,
)

TEMPLATES_PATH = SCRIPT_DIR.parent / "references" / "accuracy-datasets.json"
STATE_ROOT = ROOT / ".vaws-local" / "correctness" / "aisbench"
PROGRESS = "__VAWS_AISBENCH_ACCURACY_PROGRESS__="
SECRET_KEY = re.compile(r"(?:key|token|secret|pass|auth|credential)", re.IGNORECASE)


def emit_progress(phase: str, message: str, **extra: Any) -> None:
    payload = {"phase": phase, "message": message, **extra}
    print(PROGRESS + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def load_templates(path: Path = TEMPLATES_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("templates"), dict):
        raise ValueError(f"invalid accuracy template file: {path}")
    return payload


def resolve_case(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_templates(args.templates_file)
    defaults = dict(catalog.get("defaults", {}))
    if args.template:
        try:
            selected = dict(catalog["templates"][args.template])
        except KeyError as exc:
            available = ", ".join(sorted(catalog["templates"]))
            raise ValueError(f"unknown template {args.template!r}; choose one of: {available}") from exc
    else:
        selected = {"aisbench_dataset": args.dataset}
    resolved = {**defaults, **selected}
    for key in (
        "batch_size",
        "temperature",
        "top_p",
        "seed",
        "num_prompts",
        "warmup_prompts",
        "runs",
    ):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value

    requested_profile = getattr(args, "generation_profile", None)
    default_profile = resolved.get("default_output_length_profile")
    profiles = resolved.get("output_length_profiles", {})
    if requested_profile is not None and not profiles:
        raise ValueError(
            "--generation-profile requires a template with output_length_profiles; "
            "use --max-out-len for an arbitrary AISBench dataset"
        )
    selected_profile = requested_profile or default_profile
    if args.max_out_len is not None:
        resolved["max_out_len"] = args.max_out_len
        resolved["selected_output_length_profile"] = "explicit"
        resolved["max_out_len_source"] = "--max-out-len"
    elif selected_profile is not None:
        try:
            resolved["max_out_len"] = int(profiles[selected_profile])
        except KeyError as exc:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"unknown generation profile {selected_profile!r}; "
                f"choose one of: {available}"
            ) from exc
        resolved["selected_output_length_profile"] = selected_profile
        resolved["max_out_len_source"] = f"template profile {selected_profile}"
    elif "max_out_len" in resolved:
        resolved["selected_output_length_profile"] = "template"
        resolved["max_out_len_source"] = "template max_out_len"
    else:
        raise ValueError(
            "max output length is unknown; use a template or pass --max-out-len"
        )
    resolved["metric"] = args.metric or resolved.get("metric", "accuracy")
    if not resolved.get("aisbench_dataset"):
        raise ValueError("use --template or --dataset")
    if int(resolved.get("batch_size", 0)) < 16 and not args.allow_low_concurrency:
        raise ValueError(
            "accuracy concurrency below 16 can hide scheduling/service defects; "
            "raise --batch-size or explicitly use --allow-low-concurrency"
        )
    if float(resolved.get("temperature", 0)) != 0:
        raise ValueError("accuracy orchestration requires temperature=0")
    return resolved


def _git_rev(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def workspace_snapshot() -> dict[str, Any]:
    return {
        "workspace": _git_rev(ROOT),
        "vllm": _git_rev(ROOT / "vllm"),
        "vllm_ascend": _git_rev(ROOT / "vllm-ascend"),
        "benchmark": _git_rev(ROOT / "benchmark"),
        "aisbench_auto_tools": _git_rev(ROOT / "aisbench_auto_tools"),
    }


def redacted_command(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            key = token.split("=", 1)[0]
            redacted.append(f"{key}=<redacted>" if SECRET_KEY.search(key) else token)
            hide_next = False
        elif token == "--extra-env":
            redacted.append(token)
            hide_next = True
        elif token.startswith("--extra-env="):
            prefix, value = token.split("=", 1)
            key = value.split("=", 1)[0]
            redacted.append(f"{prefix}={key}=<redacted>" if SECRET_KEY.search(key) else token)
        else:
            redacted.append(token)
    return redacted


def build_aisbench_command(
    *,
    config_dir: str,
    dataset: str,
    work_dir: str,
    num_prompts: int | None,
    supports_num_warmups: bool,
) -> list[str]:
    command = [
        "ais_bench",
        "--config-dir",
        config_dir,
        "--models",
        aisbench_adapter.MODEL_TASK,
        "--datasets",
        dataset,
        "--work-dir",
        work_dir,
        "--dump-eval-details",
    ]
    if supports_num_warmups:
        command.extend(["--num-warmups", "0"])
    if num_prompts is not None:
        command.extend(["--num-prompts", str(num_prompts)])
    return command


def aggregate_normalized(documents: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = {}
    for document in documents:
        for case in document.get("cases", []):
            if case.get("status") != "ok":
                continue
            for metric, value in case.get("metrics", {}).items():
                values.setdefault((case["id"], metric), []).append(float(value))
    rows = []
    for (case_id, metric), samples in sorted(values.items()):
        rows.append(
            {
                "case_id": case_id,
                "metric": metric,
                "count": len(samples),
                "mean": statistics.fmean(samples),
                "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "values": samples,
            }
        )
    return {"schema_version": 1, "runs": len(documents), "metrics": rows}


def find_and_normalize(local_remote: Path) -> tuple[list[dict[str, Any]], list[str]]:
    documents: list[dict[str, Any]] = []
    sources: list[str] = []
    run_dirs = sorted((local_remote / "runs").glob("run-*"))
    for run_dir in run_dirs:
        summaries = sorted(run_dir.glob("summary/summary_*.csv"))
        if len(summaries) != 1:
            raise ValueError(
                f"expected one AISBench summary in {run_dir}, found {len(summaries)}"
            )
        summary = summaries[0]
        documents.append(
            aisbench_adapter.normalize_summary(
                summary,
                label=run_dir.name,
            )
        )
        sources.append(str(summary))
    return documents, sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--machine")
    target.add_argument("--session-id")
    target.add_argument("--session-file")
    parser.add_argument("--model", required=True, help="remote model path")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--health-timeout", type=int)
    parser.add_argument("--extra-env", action="append")
    parser.add_argument("--serve-arg", action="append", default=[])
    parser.add_argument("--skip-parity", action="store_true")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--template")
    source.add_argument("--dataset", help="AISBench dataset config name")
    parser.add_argument("--templates-file", type=Path, default=TEMPLATES_PATH)
    parser.add_argument("--metric")
    parser.add_argument(
        "--generation-profile",
        help="template output-length profile (standard, reasoning, or long-reasoning)",
    )
    parser.add_argument("--max-out-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--warmup-prompts", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--allow-low-concurrency", action="store_true")
    parser.add_argument("--keep-service", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service_config = None
    service_started = False
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    run_dir: Path | None = None
    cleanup: dict[str, Any] | None = None
    remote_run: str | None = None
    artifacts_collected = False
    try:
        case = resolve_case(args)
        if int(case["runs"]) < 1 or int(case["warmup_prompts"]) < 1:
            raise ValueError("runs and warmup-prompts must be positive")
        target = resolve_remote_target(
            machine=args.machine,
            session_id=args.session_id,
            session_file=args.session_file,
            repo_root=ROOT,
        )
        manifest = new_manifest(
            run_type="correctness",
            workspace_snapshot=workspace_snapshot(),
            environment={"target": target.to_dict()},
            model={"path": args.model, "served_model": Path(args.model).name},
            topology={"tp": args.tp, "dp": args.dp},
            command=redacted_command(
                [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
            ),
        )
        run_dir = (args.output_dir or STATE_ROOT / manifest["run_id"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "manifest.json"
        manifest = transition_status(manifest, "running")
        write_manifest(manifest_path, manifest)
        (run_dir / "resolved-config.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        service_config = assemble_config(
            machine=args.machine,
            session_id=args.session_id,
            session_file=args.session_file,
            model=args.model,
            tp=args.tp,
            dp=args.dp,
            port=args.port,
            health_timeout=args.health_timeout,
            serve_args=args.serve_arg,
            extra_env=args.extra_env,
            skip_parity=args.skip_parity,
        )
        emit_progress("serve", "starting vLLM service")
        serve_result = call_serve_start(service_config)
        if serve_result.get("status") != "ready":
            cleanup = call_serve_stop(service_config, force=True)
            raise RuntimeError(f"service did not become ready: {serve_result}")
        service_started = True
        remote_run = str(PurePosixPath(target.runtime_root) / ".vaws-runtime" / "aisbench-accuracy" / manifest["run_id"])
        remote_adapter = str(PurePosixPath(remote_run) / "adapter")
        local_adapter = run_dir / "adapter"
        aisbench_adapter.prepare(
            local_adapter,
            host="127.0.0.1",
            port=int(serve_result["port"]),
            served_model=serve_result["served_model_name"],
            datasets=[case["aisbench_dataset"]],
            work_dir=Path(remote_run) / "runs" / "run-001",
            metric=case["metric"],
            direction="higher",
            max_absolute_regression=0.0,
            max_relative_regression=0.0,
            max_out_len=int(case["max_out_len"]),
            batch_size=int(case["batch_size"]),
            temperature=float(case["temperature"]),
            top_p=float(case["top_p"]),
            seed=int(case["seed"]),
            num_prompts=case.get("num_prompts"),
        )
        push = artifact_push(target, local_path=local_adapter, remote_path=remote_adapter, timeout=300)
        if push.get("status") != "ok":
            raise RuntimeError(f"failed to publish AISBench config: {push.get('error')}")

        help_result = remote_exec(target, command="ais_bench --help", timeout=60)
        if help_result.get("status") != "ok":
            raise RuntimeError("ais_bench is unavailable in the remote runtime")
        help_text = help_result.get("stdout_tail", "") + help_result.get("stderr_tail", "")
        supports_num_warmups = "--num-warmups" in help_text

        config_dir = str(PurePosixPath(remote_adapter) / "configs")
        warmup_dir = str(PurePosixPath(remote_run) / "warmup")
        warmup_cmd = build_aisbench_command(
            config_dir=config_dir,
            dataset=case["aisbench_dataset"],
            work_dir=warmup_dir,
            num_prompts=int(case["warmup_prompts"]),
            supports_num_warmups=supports_num_warmups,
        )
        emit_progress("warmup", "running AISBench warmup", prompts=case["warmup_prompts"])
        warmup = remote_exec(
            target,
            command=f"mkdir -p {shlex.quote(warmup_dir)} && {shlex.join(warmup_cmd)} > {shlex.quote(warmup_dir + '/aisbench.log')} 2>&1",
            timeout=args.timeout,
        )
        if warmup.get("status") != "ok":
            raise RuntimeError(f"AISBench warmup failed: {warmup.get('stderr_tail') or warmup.get('error')}")

        run_results = []
        for index in range(1, int(case["runs"]) + 1):
            work_dir = str(PurePosixPath(remote_run) / "runs" / f"run-{index:03d}")
            command = build_aisbench_command(
                config_dir=config_dir,
                dataset=case["aisbench_dataset"],
                work_dir=work_dir,
                num_prompts=case.get("num_prompts"),
                supports_num_warmups=supports_num_warmups,
            )
            emit_progress("run", f"running AISBench accuracy round {index}/{case['runs']}")
            result = remote_exec(
                target,
                command=f"mkdir -p {shlex.quote(work_dir)} && {shlex.join(command)} > {shlex.quote(work_dir + '/aisbench.log')} 2>&1",
                timeout=args.timeout,
            )
            run_results.append({"run": index, "status": result.get("status"), "exit_code": result.get("exit_code")})
            if result.get("status") != "ok":
                raise RuntimeError(f"AISBench accuracy round {index} failed")

        emit_progress("collect", "pulling and hash-verifying AISBench artifacts")
        local_remote = run_dir / "artifacts" / "remote"
        pull = artifact_pull(target, remote_path=remote_run, local_dir=local_remote, timeout=args.timeout)
        if pull.get("status") != "ok":
            raise RuntimeError(f"artifact pull failed: {pull.get('error')}")
        artifacts_collected = True
        normalized, summary_sources = find_and_normalize(local_remote)
        if len(normalized) != int(case["runs"]):
            raise RuntimeError(
                f"expected {case['runs']} normalized AISBench summaries, found {len(normalized)}"
            )
        normalization_errors = [
            item
            for document in normalized
            for item in document.get("cases", [])
            if item.get("status") != "ok"
        ]
        if normalization_errors:
            raise RuntimeError(f"AISBench summaries contain {len(normalization_errors)} invalid metric rows")
        aggregate = aggregate_normalized(normalized)
        if not aggregate["metrics"]:
            raise RuntimeError("AISBench summaries contain no numeric metrics")
        summary_payload = {
            "status": "passed",
            "run_id": manifest["run_id"],
            "template": args.template,
            "case": case,
            "runs": run_results,
            "summary_sources": summary_sources,
            "aggregate": aggregate,
            "remote_run_dir": remote_run,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, kind, path in (
            ("resolved-config", "configuration", run_dir / "resolved-config.json"),
            ("artifact-manifest", "artifact-manifest", local_remote / "manifest.json"),
            ("summary", "accuracy-summary", summary_path),
        ):
            manifest = add_artifact(manifest, name=name, kind=kind, uri=str(path), sha256=sha256_file(path))
        if not args.keep_service:
            emit_progress("cleanup", "stopping vLLM service")
            cleanup = call_serve_stop(service_config, force=True)
            if cleanup.get("status") not in {"stopped", "not_found"}:
                raise RuntimeError(f"service cleanup failed: {cleanup}")
            service_started = False
        else:
            cleanup = {"status": "kept"}
        manifest = transition_status(manifest, "passed")
        write_manifest(manifest_path, manifest)
        return_code = 0
        final = {**summary_payload, "run_dir": str(run_dir), "manifest": str(manifest_path)}
    except Exception as exc:  # noqa: BLE001
        return_code = 2
        failure_artifacts = None
        if remote_run and run_dir is not None and not artifacts_collected:
            try:
                failure_artifacts = artifact_pull(
                    target,
                    remote_path=remote_run,
                    local_dir=run_dir / "artifacts" / "remote",
                    timeout=args.timeout,
                )
            except Exception as collect_exc:  # noqa: BLE001
                failure_artifacts = {"status": "failed", "error": str(collect_exc)}
        if manifest is not None and manifest_path is not None and manifest.get("status") == "running":
            existing_names = {item["name"] for item in manifest.get("artifacts", [])}
            candidate_artifacts = [
                ("resolved-config", "configuration", run_dir / "resolved-config.json" if run_dir else None),
                (
                    "failure-artifact-manifest",
                    "artifact-manifest",
                    run_dir / "artifacts" / "remote" / "manifest.json" if run_dir else None,
                ),
            ]
            for name, kind, path in candidate_artifacts:
                if path is not None and path.is_file() and name not in existing_names:
                    manifest = add_artifact(
                        manifest,
                        name=name,
                        kind=kind,
                        uri=str(path),
                        sha256=sha256_file(path),
                    )
            manifest = transition_status(manifest, "failed")
            write_manifest(manifest_path, manifest)
        final = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir) if run_dir else None,
            "manifest": str(manifest_path) if manifest_path else None,
            "failure_artifacts": failure_artifacts,
        }
    finally:
        if service_config is not None and service_started and not args.keep_service:
            emit_progress("cleanup", "stopping vLLM service")
            cleanup = call_serve_stop(service_config, force=True)
        elif args.keep_service and service_started:
            cleanup = {"status": "kept"}
    final["service_cleanup"] = cleanup
    print_json(final)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
