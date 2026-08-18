#!/usr/bin/env python3
"""One-click aisbench_auto_tools performance workflow with durable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
for path in (SCRIPT_DIR, LIB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

STATE_ROOT = ROOT / ".vaws-local" / "benchmark" / "aisbench-auto-tools"
PROGRESS = "__VAWS_AISBENCH_PERF_PROGRESS__="
SENTINELS = {9999.0, 99999.0}
SECRET_KEY = re.compile(r"(?:key|token|secret|pass|auth|credential)", re.IGNORECASE)


def emit_progress(phase: str, message: str, **extra: Any) -> None:
    print(
        PROGRESS + json.dumps({"phase": phase, "message": message, **extra}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


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


def redact_extra_env(items: list[str] | None) -> list[str] | None:
    if items is None:
        return None
    redacted = []
    for item in items:
        key = item.split("=", 1)[0]
        redacted.append(f"{key}=<redacted>" if SECRET_KEY.search(key) else item)
    return redacted


def redacted_command(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.extend(redact_extra_env([token]) or [])
            hide_next = False
        elif token == "--extra-env":
            redacted.append(token)
            hide_next = True
        elif token.startswith("--extra-env="):
            prefix, value = token.split("=", 1)
            redacted_value = (redact_extra_env([value]) or [value])[0]
            redacted.append(f"{prefix}={redacted_value}")
        else:
            redacted.append(token)
    return redacted


def render_auto_tools_config(
    *,
    dataset_path: str,
    work_path: str,
    model_name: str,
    model_path: str,
    host: str,
    port: int,
    output_dir: str,
    performance_summarizer: str,
    pod_info: list[str],
) -> str:
    return "\n".join(
        [
            f"DATASET_PATH = {dataset_path!r}",
            f"WORK_PATH = {work_path!r}",
            f"MODEL_NAME = {model_name!r}",
            f"MODEL_PATH = {model_path!r}",
            f"HOST_IP = {host!r}",
            f"HOST_PORT = {str(port)!r}",
            f"DEFAULT_PERFORMANCE_TEST = {performance_summarizer!r}",
            f"OUTPUT_DIR = {output_dir!r}",
            f"POD_INFO = {pod_info!r}",
            "",
        ]
    )


def build_auto_tools_args(args: argparse.Namespace, *, data_num: int) -> list[str]:
    command = [
        "python3",
        "aisbench_test.py",
        "--input_len",
        str(args.input_len),
        "--output_len",
        str(args.output_len),
        "--data_num",
        str(data_num),
        "--concurrency",
        str(args.concurrency),
        "--request_rate",
        str(args.request_rate),
        "--test_type",
        args.test_type,
        "--repeat",
        "1",
        "--npu_num",
        str(args.npu_num),
        "--dataset_type",
        args.dataset_type,
        "--prefix_num",
        str(args.prefix_num),
        "--repeat_rate",
        args.repeat_rate,
        "--seed",
        str(args.seed),
        "--dp",
        str(args.dp or 1),
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.enable_think:
        command.append("--enable_think")
    if args.prefix_test:
        command.append("--prefix_test")
    for flag, value in (
        ("--length_mean", args.length_mean),
        ("--length_std", args.length_std),
        ("--length_min", args.length_min),
        ("--length_max", args.length_max),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command


def build_phase_command(
    *,
    source_dir: str,
    phase_dir: str,
    config_path: str,
    command: list[str],
) -> str:
    validation = shlex.join(
        [
            "python3",
            "-c",
            (
                "import csv,math,sys;"
                "r=list(csv.DictReader(open('aisbench_result.csv',encoding='utf-8-sig')));"
                "k=('TTFT avg','TPOT avg','output_throughput','qps');"
                "ok=len(r)==1 and all(x in r[0] and math.isfinite(float(r[0][x])) "
                "and float(r[0][x]) not in (9999.0,99999.0) for x in k);"
                "sys.exit(0 if ok else 1)"
            ),
        ]
    )
    return " && ".join(
        [
            f"mkdir -p {shlex.quote(phase_dir)}",
            f"cp -a {shlex.quote(source_dir + '/.')} {shlex.quote(phase_dir + '/')}",
            f"cp {shlex.quote(config_path)} {shlex.quote(phase_dir + '/config.py')}",
            f"cd {shlex.quote(phase_dir)}",
            shlex.join(command),
            "test -s aisbench_result.csv",
            validation,
        ]
    )


def parse_result_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one result row in {path}, got {len(rows)}")
    parsed: dict[str, Any] = {}
    for key, raw in rows[0].items():
        value = (raw or "").strip()
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    required = ("TTFT avg", "TPOT avg", "output_throughput", "qps")
    bad = [
        key
        for key in required
        if not isinstance(parsed.get(key), float)
        or not math.isfinite(parsed[key])
        or parsed[key] in SENTINELS
    ]
    if bad:
        raise ValueError(f"aisbench_auto_tools emitted invalid/sentinel metrics in {path}: {', '.join(bad)}")
    return parsed


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"count": len(rows)}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, float)})
    for key in keys:
        samples = [float(row[key]) for row in rows if isinstance(row.get(key), float)]
        if not samples:
            continue
        metrics[key] = {
            "mean": statistics.fmean(samples),
            "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "values": samples,
        }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--machine")
    target.add_argument("--session-id")
    target.add_argument("--session-file")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--health-timeout", type=int)
    parser.add_argument("--extra-env", action="append")
    parser.add_argument("--serve-arg", action="append", default=[])
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--input-len", type=int, default=3500)
    parser.add_argument("--output-len", type=int, default=1500)
    parser.add_argument("--data-num", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=2048)
    parser.add_argument("--request-rate", default="0")
    parser.add_argument("--test-type", choices=("stream", "text"), default="stream")
    parser.add_argument("--dataset", help="remote GSM8K-format dataset path")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--npu-num", type=int, default=1)
    parser.add_argument("--dataset-type", choices=("normal", "prefix_cache"), default="normal")
    parser.add_argument("--prefix-num", type=int, default=1)
    parser.add_argument("--repeat-rate", default="0")
    parser.add_argument("--prefix-test", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--enable-think", action="store_true")
    parser.add_argument("--length-mean", type=int)
    parser.add_argument("--length-std", type=float)
    parser.add_argument("--length-min", type=int)
    parser.add_argument("--length-max", type=int)
    parser.add_argument("--performance-summarizer", choices=("default_perf", "stable_stage"), default="default_perf")
    parser.add_argument("--pod-info", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--keep-service", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service_config = None
    service_started = False
    cleanup: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    run_dir: Path | None = None
    remote_run: str | None = None
    artifacts_collected = False
    try:
        if min(args.input_len, args.output_len, args.data_num, args.concurrency, args.runs) < 1:
            raise ValueError("lengths, data-num, concurrency, and runs must be positive")
        target = resolve_remote_target(
            machine=args.machine,
            session_id=args.session_id,
            session_file=args.session_file,
            repo_root=ROOT,
        )
        manifest = new_manifest(
            run_type="performance",
            workspace_snapshot=workspace_snapshot(),
            environment={"target": target.to_dict(), "backend": "aisbench_auto_tools"},
            model={"path": args.model, "served_model": Path(args.model).name},
            topology={"tp": args.tp, "dp": args.dp, "npu_num": args.npu_num},
            command=redacted_command(
                [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
            ),
        )
        run_dir = (args.output_dir or STATE_ROOT / manifest["run_id"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "manifest.json"
        manifest = transition_status(manifest, "running")
        write_manifest(manifest_path, manifest)

        resolved = vars(args).copy()
        resolved["output_dir"] = str(args.output_dir) if args.output_dir else None
        resolved["extra_env"] = redact_extra_env(args.extra_env)
        (run_dir / "resolved-config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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

        remote_run = str(PurePosixPath(target.runtime_root) / ".vaws-runtime" / "aisbench-performance" / manifest["run_id"])
        source_dir = str(PurePosixPath(target.runtime_root) / "aisbench_auto_tools")
        benchmark_dir = str(PurePosixPath(target.runtime_root) / "benchmark")
        preflight = remote_exec(
            target,
            command=(
                f"test -f {shlex.quote(source_dir + '/aisbench_test.py')} && "
                f"test -d {shlex.quote(benchmark_dir + '/ais_bench')} && command -v ais_bench"
            ),
            timeout=60,
        )
        if preflight.get("status") != "ok":
            raise RuntimeError("remote aisbench_auto_tools or AISBench runtime is unavailable")

        local_generated = run_dir / "generated"
        local_generated.mkdir(parents=True)
        config_text = render_auto_tools_config(
            dataset_path=str(PurePosixPath(remote_run) / "datasets"),
            work_path=benchmark_dir,
            model_name=serve_result["served_model_name"],
            model_path=args.model,
            host="127.0.0.1",
            port=int(serve_result["port"]),
            output_dir="./outputs",
            performance_summarizer=args.performance_summarizer,
            pod_info=args.pod_info,
        )
        (local_generated / "config.py").write_text(config_text, encoding="utf-8")
        push = artifact_push(
            target,
            local_path=local_generated,
            remote_path=str(PurePosixPath(remote_run) / "generated"),
            timeout=300,
        )
        if push.get("status") != "ok":
            raise RuntimeError(f"failed to publish auto-tools config: {push.get('error')}")
        remote_config = str(PurePosixPath(remote_run) / "generated" / "config.py")

        warmup_requests = args.warmup_requests or min(max(args.concurrency, 64), 256)
        if warmup_requests < 1:
            raise ValueError("warmup-requests must be positive")
        warmup_dir = str(PurePosixPath(remote_run) / "warmup")
        warmup_command = build_phase_command(
            source_dir=source_dir,
            phase_dir=warmup_dir,
            config_path=remote_config,
            command=build_auto_tools_args(args, data_num=warmup_requests),
        )
        emit_progress("warmup", "running aisbench_auto_tools warmup", requests=warmup_requests)
        warmup = remote_exec(target, command=warmup_command, timeout=args.timeout)
        if warmup.get("status") != "ok":
            raise RuntimeError(f"performance warmup failed: {warmup.get('stderr_tail') or warmup.get('error')}")

        run_status = []
        for index in range(1, args.runs + 1):
            phase_dir = str(PurePosixPath(remote_run) / "runs" / f"run-{index:03d}")
            phase_command = build_phase_command(
                source_dir=source_dir,
                phase_dir=phase_dir,
                config_path=remote_config,
                command=build_auto_tools_args(args, data_num=args.data_num),
            )
            emit_progress("run", f"running performance round {index}/{args.runs}")
            result = remote_exec(target, command=phase_command, timeout=args.timeout)
            run_status.append({"run": index, "status": result.get("status"), "exit_code": result.get("exit_code")})
            if result.get("status") != "ok":
                raise RuntimeError(f"performance round {index} failed")

        emit_progress("collect", "pulling and hash-verifying performance artifacts")
        local_remote = run_dir / "artifacts" / "remote"
        pull = artifact_pull(target, remote_path=remote_run, local_dir=local_remote, timeout=args.timeout)
        if pull.get("status") != "ok":
            raise RuntimeError(f"artifact pull failed: {pull.get('error')}")
        artifacts_collected = True
        rows = [parse_result_csv(path) for path in sorted(local_remote.glob("runs/run-*/aisbench_result.csv"))]
        if len(rows) != args.runs:
            raise RuntimeError(f"expected {args.runs} performance result files, found {len(rows)}")
        summary_payload = {
            "status": "passed",
            "run_id": manifest["run_id"],
            "backend": "aisbench_auto_tools",
            "runs": run_status,
            "per_run": rows,
            "aggregate": aggregate(rows),
            "remote_run_dir": remote_run,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name, kind, path in (
            ("resolved-config", "configuration", run_dir / "resolved-config.json"),
            ("generated-auto-tools-config", "configuration", local_generated / "config.py"),
            ("artifact-manifest", "artifact-manifest", local_remote / "manifest.json"),
            ("summary", "performance-summary", summary_path),
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
        final = {**summary_payload, "run_dir": str(run_dir), "manifest": str(manifest_path)}
        return_code = 0
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
                    "generated-auto-tools-config",
                    "configuration",
                    run_dir / "generated" / "config.py" if run_dir else None,
                ),
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
        elif service_started:
            cleanup = {"status": "kept"}
    final["service_cleanup"] = cleanup
    print_json(final)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
