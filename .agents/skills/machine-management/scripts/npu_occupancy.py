#!/usr/bin/env python3
"""Inspect NPU occupancy on a remote Ascend host.

The collector runs on the bare-metal host, not inside the managed container, so
it can see NPU processes from all containers.  It reports device HBM/memory,
device AICore utilization, NPU process PIDs/names, process cwd, and best-effort
Docker container ownership.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
LIB_DIR = ROOT / ".agents" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from vaws_ssh_control import ssh_command_prefix  # noqa: E402

PROGRESS_SENTINEL = "__VAWS_NPU_OCCUPANCY_PROGRESS__="


class LocalEndpoint:
    def __init__(self, *, host: str, port: int, user: str) -> None:
        self.host = host
        self.port = port
        self.user = user

    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def emit_progress(phase: str, message: str | None = None, **extra: Any) -> None:
    payload: dict[str, Any] = {"phase": phase, "at": utc_now()}
    if message:
        payload["message"] = message
    payload.update({key: value for key, value in extra.items() if value is not None})
    sys.stderr.write(PROGRESS_SENTINEL + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    sys.stderr.flush()


def tail_text(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def run_text(cmd: list[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def parse_number(value: str) -> int | float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def normalize_percent(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def ensure_device(devices: dict[int, dict[str, Any]], npu_id: int) -> dict[str, Any]:
    return devices.setdefault(
        npu_id,
        {
            "npu_id": npu_id,
            "chip_id": None,
            "name": None,
            "health": None,
            "aicore_percent": None,
            "memory": {"used_mb": None, "total_mb": None},
            "hbm": {"used_mb": None, "total_mb": None},
            "hbm_utilization_percent": None,
            "chips": [],
            "processes": [],
        },
    )


def add_mb_pair(target: dict[str, Any], key: str, used_mb: int, total_mb: int) -> None:
    pair = target.setdefault(key, {"used_mb": None, "total_mb": None})
    pair["used_mb"] = (pair["used_mb"] or 0) + used_mb
    pair["total_mb"] = (pair["total_mb"] or 0) + total_mb


def update_max_percent(target: dict[str, Any], key: str, value: int | float | None) -> None:
    if value is None:
        return
    current = target.get(key)
    if current is None or value > current:
        target[key] = value


def parse_npu_smi_info(output: str) -> dict[str, Any]:
    """Parse common ``npu-smi info`` layouts.

    Ascend images vary slightly by driver/CANN version.  This parser keeps to
    stable tokens: device/chip ids, PCI bus id, AICore percent, memory pairs,
    HBM pairs, and the process table.
    """
    devices: dict[int, dict[str, Any]] = {}
    process_records: list[dict[str, Any]] = []
    in_process_table = False
    current_npu: int | None = None
    device_layout = "npu-chip"

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        lower = line.lower()
        if "bus-id" in lower and "phy-id" in lower:
            device_layout = "chip-phy"
            continue
        if "bus-id" in lower and "npu" in lower and "chip" in lower:
            device_layout = "npu-chip"
            continue
        if "process id" in lower or "process name" in lower or "process memory" in lower:
            in_process_table = True
            continue
        if "no running processes" in lower:
            continue

        if in_process_table:
            inner = line.strip("|").strip()
            cells = [cell.strip() for cell in inner.split("|")]
            if len(cells) >= 4:
                left = cells[0].split()
                pid_cell = cells[1].strip()
                mem_cell = cells[3].strip()
                if len(left) >= 2 and left[0].isdigit() and left[1].isdigit() and pid_cell.isdigit():
                    record = {
                        "npu_id": int(left[0]),
                        "chip_id": int(left[1]),
                        "pid": int(pid_cell),
                        "npu_process_name": cells[2].strip() or None,
                        "npu_memory_mb": int(mem_cell) if mem_cell.isdigit() else None,
                    }
                    process_records.append(record)
                    ensure_device(devices, record["npu_id"])["processes"].append(record)
                    continue
            tokens = inner.split()
            if len(tokens) >= 5 and tokens[0].isdigit() and tokens[1].isdigit() and tokens[2].isdigit():
                mem_mb: int | None = None
                name_tokens = tokens[3:]
                if tokens[-1].isdigit():
                    mem_mb = int(tokens[-1])
                    name_tokens = tokens[3:-1]
                record = {
                    "npu_id": int(tokens[0]),
                    "chip_id": int(tokens[1]),
                    "pid": int(tokens[2]),
                    "npu_process_name": " ".join(name_tokens) if name_tokens else None,
                    "npu_memory_mb": mem_mb,
                }
                process_records.append(record)
                ensure_device(devices, record["npu_id"])["processes"].append(record)
            continue

        if "0000:" in line:
            # Example:
            # | 0     0      | 0000:C1:00.0     0            0 / 0       8123 / 65536 |
            m = re.search(
                r"\|\s*(?P<npu>\d+)\s+(?P<chip>\d+)\s+\|\s*"
                r"(?P<bus>[0-9a-fA-F:.]+)\s*(?:\|\s*)?"
                r"(?P<aicore>\d+(?:\.\d+)?)\s+"
                r"(?P<mem_used>\d+)\s*/\s*(?P<mem_total>\d+)\s+"
                r"(?P<hbm_used>\d+)\s*/\s*(?P<hbm_total>\d+)",
                line,
            )
            if not m:
                continue
            if device_layout == "chip-phy" and current_npu is not None:
                npu_id = current_npu
                chip_id = int(m.group("npu"))
                phy_id: int | None = int(m.group("chip"))
            else:
                npu_id = int(m.group("npu"))
                chip_id = int(m.group("chip"))
                phy_id = None
            aicore = normalize_percent(parse_number(m.group("aicore")))
            mem_used = int(m.group("mem_used"))
            mem_total = int(m.group("mem_total"))
            hbm_used = int(m.group("hbm_used"))
            hbm_total = int(m.group("hbm_total"))
            device = ensure_device(devices, npu_id)
            if device["chip_id"] is None:
                device["chip_id"] = chip_id
            device["bus_id"] = m.group("bus") if len(device["chips"]) == 0 else device.get("bus_id")
            update_max_percent(device, "aicore_percent", aicore)
            add_mb_pair(device, "memory", mem_used, mem_total)
            add_mb_pair(device, "hbm", hbm_used, hbm_total)
            device["chips"].append({
                "chip_id": chip_id,
                "phy_id": phy_id,
                "bus_id": m.group("bus"),
                "aicore_percent": aicore,
                "memory": {"used_mb": mem_used, "total_mb": mem_total},
                "hbm": {"used_mb": hbm_used, "total_mb": hbm_total},
            })
            continue

        # Example:
        # | 0     910B4      | OK              91.8        41               0 / 0 |
        m = re.search(
            r"\|\s*(?P<npu>\d+)\s+(?P<name>[A-Za-z0-9_.-]+)\s+\|\s*"
            r"(?P<health>[A-Za-z_.-]+)\s+",
            line,
        )
        if m and not m.group("name").isdigit():
            current_npu = int(m.group("npu"))
            device = ensure_device(devices, int(m.group("npu")))
            device["name"] = m.group("name")
            device["health"] = m.group("health")

    return {
        "devices": [devices[key] for key in sorted(devices)],
        "process_records": process_records,
    }


def parse_npu_smi_usages(output: str) -> dict[int, dict[str, Any]]:
    """Parse optional ``npu-smi info -t usages`` output."""
    usages: dict[int, dict[str, Any]] = {}
    current: int | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip().strip("|").strip()
        if not line:
            continue
        m = re.search(r"\bNPU\s*ID\b\s*[:=]?\s*(\d+)", line, re.IGNORECASE)
        if m:
            current = int(m.group(1))
            usages.setdefault(current, {})
            continue
        if current is None:
            continue
        value_match = re.search(r"(-?\d+(?:\.\d+)?)", line)
        if not value_match:
            continue
        value = normalize_percent(parse_number(value_match.group(1)))
        lower = line.lower()
        if "aicore" in lower or "ai core" in lower:
            usages.setdefault(current, {})["aicore_percent"] = value
        elif "hbm" in lower and "usage" in lower and "rate" in lower:
            usages.setdefault(current, {})["hbm_utilization_percent"] = value
    return usages


def apply_usage_overrides(devices: list[dict[str, Any]], usages: dict[int, dict[str, Any]]) -> None:
    for device in devices:
        usage = usages.get(int(device["npu_id"]))
        if not usage:
            continue
        if usage.get("aicore_percent") is not None:
            device["aicore_percent"] = usage["aicore_percent"]
        if usage.get("hbm_utilization_percent") is not None:
            device["hbm_utilization_percent"] = usage["hbm_utilization_percent"]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def read_link(path: Path) -> str | None:
    try:
        return os.readlink(path)
    except OSError:
        return None


def parse_status_nspid(text: str | None) -> list[int]:
    if not text:
        return []
    for line in text.splitlines():
        if line.startswith("NSpid:"):
            values: list[int] = []
            for token in line.split()[1:]:
                if token.isdigit():
                    values.append(int(token))
            return values
    return []


def process_user(pid: int) -> str | None:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:
        return None


def extract_container_id_from_cgroup(text: str | None) -> str | None:
    if not text:
        return None
    # Covers docker, containerd, CRI-O, and systemd scope names such as
    # docker-<id>.scope and cri-containerd-<id>.scope.
    match = re.search(r"\b([0-9a-f]{64})\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def namespace_links(pid: int) -> dict[str, str]:
    links: dict[str, str] = {}
    for ns in ("pid", "mnt"):
        link = read_link(Path(f"/proc/{pid}/ns/{ns}"))
        if link:
            links[ns] = link
    return links


def collect_docker_containers() -> tuple[dict[str, Any], str | None]:
    if shutil.which("docker") is None:
        return {"available": False, "containers": [], "by_id": {}, "by_ns": {}}, "docker command not found"
    ids_result = run_text(["docker", "ps", "-aq", "--no-trunc"], timeout=10)
    if ids_result.returncode != 0:
        return {
            "available": False,
            "containers": [],
            "by_id": {},
            "by_ns": {},
        }, tail_text(ids_result.stderr or ids_result.stdout, 1000)

    containers: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_ns: dict[str, dict[str, Any]] = {}
    ids = [line.strip() for line in ids_result.stdout.splitlines() if line.strip()]
    for cid in ids:
        inspect = run_text(["docker", "inspect", cid], timeout=10)
        if inspect.returncode != 0:
            continue
        try:
            records = json.loads(inspect.stdout)
        except json.JSONDecodeError:
            continue
        if not records:
            continue
        item = records[0]
        state = item.get("State") or {}
        config = item.get("Config") or {}
        info = {
            "id": str(item.get("Id") or cid),
            "short_id": str(item.get("Id") or cid)[:12],
            "name": str(item.get("Name") or "").lstrip("/") or None,
            "image": str(config.get("Image") or item.get("Image") or "") or None,
            "status": str(state.get("Status") or "") or None,
            "init_pid": int(state.get("Pid") or 0),
        }
        containers.append(info)
        by_id[info["id"].lower()] = info
        if info["init_pid"] > 0:
            for ns, link in namespace_links(info["init_pid"]).items():
                by_ns[f"{ns}:{link}"] = info
    return {"available": True, "containers": containers, "by_id": by_id, "by_ns": by_ns}, None


def detect_container_for_pid(pid: int, docker_state: dict[str, Any]) -> dict[str, Any] | None:
    cgroup = read_text(Path(f"/proc/{pid}/cgroup"))
    cgroup_id = extract_container_id_from_cgroup(cgroup)
    by_id = docker_state.get("by_id") or {}
    if cgroup_id:
        for cid, info in by_id.items():
            if cid == cgroup_id or cid.startswith(cgroup_id) or cgroup_id.startswith(cid):
                return {
                    "id": info["id"],
                    "short_id": info["short_id"],
                    "name": info["name"],
                    "image": info["image"],
                    "status": info["status"],
                    "source": "cgroup",
                }
        return {
            "id": cgroup_id,
            "short_id": cgroup_id[:12],
            "name": None,
            "image": None,
            "status": None,
            "source": "cgroup-unmatched",
        }

    by_ns = docker_state.get("by_ns") or {}
    for ns, link in namespace_links(pid).items():
        info = by_ns.get(f"{ns}:{link}")
        if info:
            return {
                "id": info["id"],
                "short_id": info["short_id"],
                "name": info["name"],
                "image": info["image"],
                "status": info["status"],
                "source": f"{ns}-namespace",
            }
    return None


def enrich_process(pid: int, docker_state: dict[str, Any]) -> dict[str, Any]:
    comm = read_text(Path(f"/proc/{pid}/comm"))
    status = read_text(Path(f"/proc/{pid}/status"))
    cwd = read_link(Path(f"/proc/{pid}/cwd"))
    return {
        "pid": pid,
        "exists": Path(f"/proc/{pid}").exists(),
        "name": comm.strip() if comm else None,
        "user": process_user(pid),
        "cwd": cwd,
        "nspid": parse_status_nspid(status),
        "container": detect_container_for_pid(pid, docker_state),
    }


def aggregate_processes(
    process_records: list[dict[str, Any]],
    docker_state: dict[str, Any],
) -> list[dict[str, Any]]:
    by_pid: dict[int, dict[str, Any]] = {}
    for record in process_records:
        pid = int(record["pid"])
        process = by_pid.setdefault(pid, enrich_process(pid, docker_state))
        process.setdefault("npu_usages", [])
        process["npu_usages"].append({
            "npu_id": record["npu_id"],
            "chip_id": record["chip_id"],
            "npu_memory_mb": record.get("npu_memory_mb"),
            "npu_process_name": record.get("npu_process_name"),
        })
        if not process.get("name") and record.get("npu_process_name"):
            process["name"] = record["npu_process_name"]
    return [by_pid[pid] for pid in sorted(by_pid)]


def attach_enriched_processes(
    devices: list[dict[str, Any]],
    processes: list[dict[str, Any]],
) -> None:
    by_pid = {int(proc["pid"]): proc for proc in processes}
    for device in devices:
        enriched: list[dict[str, Any]] = []
        for record in device.get("processes", []):
            process = by_pid.get(int(record["pid"]))
            if not process:
                enriched.append(record)
                continue
            enriched.append({
                "pid": process["pid"],
                "name": process.get("name") or record.get("npu_process_name"),
                "user": process.get("user"),
                "cwd": process.get("cwd"),
                "container": process.get("container"),
                "npu_memory_mb": record.get("npu_memory_mb"),
                "npu_process_name": record.get("npu_process_name"),
            })
        device["processes"] = enriched


def collect_once() -> dict[str, Any]:
    info = run_text(["npu-smi", "info"], timeout=15)
    if info.returncode != 0:
        return {
            "status": "failed",
            "error": "npu-smi info failed",
            "returncode": info.returncode,
            "stderr": tail_text(info.stderr or info.stdout),
        }
    parsed = parse_npu_smi_info(info.stdout)
    usages_result = run_text(["npu-smi", "info", "-t", "usages"], timeout=15)
    if usages_result.returncode == 0:
        apply_usage_overrides(parsed["devices"], parse_npu_smi_usages(usages_result.stdout))

    docker_state, docker_error = collect_docker_containers()
    processes = aggregate_processes(parsed["process_records"], docker_state)
    attach_enriched_processes(parsed["devices"], processes)
    return {
        "status": "ok",
        "collected_at": utc_now(),
        "hostname": socket.gethostname(),
        "devices": parsed["devices"],
        "processes": processes,
        "docker": {
            "available": bool(docker_state.get("available")),
            "container_count": len(docker_state.get("containers") or []),
            "error": docker_error,
        },
        "npu_smi": {
            "info_returncode": info.returncode,
            "usages_returncode": usages_result.returncode,
            "usages_error": None if usages_result.returncode == 0 else tail_text(usages_result.stderr or usages_result.stdout, 1000),
        },
    }


def remote_collector_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="remote NPU occupancy collector", allow_abbrev=False)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.interval < 0:
        parser.error("--interval must be >= 0")

    samples: list[dict[str, Any]] = []
    for index in range(args.samples):
        if index:
            time.sleep(args.interval)
        samples.append(collect_once())
    latest = samples[-1]
    payload = {
        **latest,
        "sample_count": args.samples,
        "sample_interval_seconds": args.interval,
    }
    if args.samples > 1:
        payload["samples"] = samples
    print_json(payload)
    return 0 if latest.get("status") == "ok" else 2


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_local_target(args: argparse.Namespace) -> tuple[str, LocalEndpoint, dict[str, Any]]:
    if args.host:
        alias = args.host
        endpoint = LocalEndpoint(host=args.host, port=args.host_port, user=args.host_user)
        return alias, endpoint, {
            "mode": "direct-host",
            "alias": alias,
            "host": {"host": endpoint.host, "port": endpoint.port, "user": endpoint.user},
        }

    root = workspace_root()
    lib_dir = root / ".agents" / "lib"
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
    from vaws_remote_toolbox import resolve_remote_target  # noqa: WPS433

    target = resolve_remote_target(
        machine=args.machine,
        session_id=args.session_id,
        session_file=args.session_file,
        repo_root=root,
    )
    endpoint = LocalEndpoint(
        host=target.host_endpoint.host,
        port=target.host_endpoint.port,
        user=target.host_endpoint.user,
    )
    return target.alias, endpoint, {
        "mode": target.mode,
        "alias": target.alias,
        "target_id": target.target_id,
        "host": target.host_endpoint.to_dict(plane="host"),
        "container": {
            "name": target.container_name,
            "image_record": target.container_image,
            **target.container_endpoint.to_dict(plane="container"),
        },
    }


def build_remote_command(script_path: Path, *, samples: int, interval: float) -> str:
    source = script_path.read_text(encoding="utf-8")
    args = shlex.join(["--remote-collector", "--samples", str(samples), "--interval", str(interval)])
    delimiter = "__VAWS_NPU_OCCUPANCY_SCRIPT__"
    return (
        "set -euo pipefail\n"
        "if command -v python3 >/dev/null 2>&1; then _py=python3; "
        "elif command -v python >/dev/null 2>&1; then _py=python; "
        "else echo '{\"status\":\"failed\",\"error\":\"python3 not found on remote host\"}'; exit 127; fi\n"
        f"\"$_py\" - {args} <<'{delimiter}'\n"
        f"{source}\n"
        f"{delimiter}\n"
    )


def ssh_collect(endpoint: LocalEndpoint, remote_command: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    cmd = [
        *ssh_command_prefix(),
        "-T",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(endpoint.port),
        endpoint.destination(),
        "bash",
        "-c",
        shlex.quote(remote_command),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def render_table(payload: dict[str, Any]) -> str:
    rows: list[list[str]] = [[
        "machine",
        "npu",
        "aicore%",
        "hbm_mb",
        "memory_mb",
        "pid",
        "process",
        "container",
        "cwd",
    ]]
    machine = str(payload.get("machine") or payload.get("target", {}).get("alias") or "")
    for device in payload.get("devices", []):
        processes = device.get("processes") or [None]
        for process in processes:
            container = None
            if process:
                c = process.get("container") or {}
                container = c.get("name") or c.get("short_id") or c.get("id")
            hbm = device.get("hbm") or {}
            mem = device.get("memory") or {}
            rows.append([
                machine,
                str(device.get("npu_id")),
                "-" if device.get("aicore_percent") is None else str(device.get("aicore_percent")),
                f"{hbm.get('used_mb')}/{hbm.get('total_mb')}",
                f"{mem.get('used_mb')}/{mem.get('total_mb')}",
                "-" if not process else str(process.get("pid")),
                "-" if not process else str(process.get("name") or process.get("npu_process_name") or ""),
                "-" if not container else str(container),
                "-" if not process else str(process.get("cwd") or ""),
            ])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--machine", help="managed machine alias or host IP from inventory")
    target.add_argument("--session-id", help="managed session id")
    target.add_argument("--session-file", help="managed session file")
    target.add_argument("--host", help="direct bare-metal host address")
    parser.add_argument("--host-port", type=int, default=22)
    parser.add_argument("--host-user", default="root")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--format", choices=["json", "table"], default="json")
    return parser


def build_remote_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="remote NPU occupancy collector", allow_abbrev=False)
    parser.add_argument("--remote-collector", action="store_true", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser


def local_main(args: argparse.Namespace) -> int:
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0")
    alias, endpoint, target_info = resolve_local_target(args)
    timeout = args.timeout if args.timeout is not None else max(30.0, args.samples * args.interval + 30.0)
    emit_progress("resolve-target", f"probing NPU occupancy on host {endpoint.destination()}:{endpoint.port}")
    remote_command = build_remote_command(Path(__file__).resolve(), samples=args.samples, interval=args.interval)
    result = ssh_collect(endpoint, remote_command, timeout=timeout)
    if result.returncode != 0:
        print_json({
            "status": "failed",
            "machine": alias,
            "target": target_info,
            "error": "remote collector failed",
            "returncode": result.returncode,
            "stderr": tail_text(result.stderr),
            "stdout": tail_text(result.stdout),
        })
        return 2
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print_json({
            "status": "failed",
            "machine": alias,
            "target": target_info,
            "error": f"remote collector did not return JSON: {exc}",
            "stdout": tail_text(result.stdout),
            "stderr": tail_text(result.stderr),
        })
        return 2
    payload["machine"] = alias
    payload["target"] = target_info
    if args.format == "table":
        print(render_table(payload))
    else:
        print_json(payload)
    return 0 if payload.get("status") == "ok" else 2


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--remote-collector" in raw_args:
            args = build_remote_parser().parse_args(raw_args)
            return remote_collector_main(["--samples", str(args.samples), "--interval", str(args.interval)])
        args = build_parser().parse_args(raw_args)
        return local_main(args)
    except Exception as exc:
        print_json({"status": "failed", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
