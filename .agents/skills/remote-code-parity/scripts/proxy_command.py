#!/usr/bin/env python3
"""Select a remote-reachable local IPv4 address and render proxy commands."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class ProxyCommandError(RuntimeError):
    """User-facing proxy command generation failure."""


@dataclass(frozen=True)
class LocalAddress:
    address: str
    prefix_length: int
    interface: str
    source: str

    @property
    def ip(self) -> ipaddress.IPv4Address:
        return ipaddress.IPv4Address(self.address)

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(
            f"{self.address}/{self.prefix_length}", strict=False
        )


@dataclass(frozen=True)
class RouteMapping:
    remote_network: str
    local_network: str
    name: str

    @property
    def remote(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(self.remote_network, strict=True)

    @property
    def local(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(self.local_network, strict=True)


VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_NO_PROXY = "127.0.0.1,localhost,.local,.huawei.com"
WINDOWS_SOURCE = "windows-powershell"
DEFAULT_ROUTE_MAP = (
    Path(__file__).resolve().parents[1] / "references" / "proxy-routes.json"
)


def _valid_unicast(address: ipaddress.IPv4Address) -> bool:
    return not (
        address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
    )


def _normalize_records(records: Iterable[LocalAddress]) -> list[LocalAddress]:
    result: dict[tuple[str, int, str], LocalAddress] = {}
    for record in records:
        interface_name = record.interface.casefold()
        if interface_name == "lo" or "loopback" in interface_name:
            continue
        try:
            address = record.ip
        except ipaddress.AddressValueError:
            continue
        if not _valid_unicast(address):
            continue
        if not 0 <= record.prefix_length <= 32:
            continue
        key = (record.address, record.prefix_length, record.interface)
        result[key] = record
    return sorted(
        result.values(),
        key=lambda item: (
            item.address,
            -item.prefix_length,
            item.interface,
            item.source,
        ),
    )


def _load_json_command(argv: list[str]) -> Any:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8-sig",
        errors="replace",
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ProxyCommandError(
            f"address discovery command failed ({completed.returncode}): "
            f"{' '.join(shlex.quote(part) for part in argv)}: {detail}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProxyCommandError(
            f"address discovery returned invalid JSON: {argv[0]}"
        ) from exc


def discover_linux_addresses() -> list[LocalAddress]:
    executable = shutil.which("ip")
    if not executable:
        return []
    payload = _load_json_command([executable, "-j", "-4", "address", "show", "up"])
    records: list[LocalAddress] = []
    for interface in payload if isinstance(payload, list) else []:
        if not isinstance(interface, dict):
            continue
        name = str(interface.get("ifname") or "unknown")
        for address in interface.get("addr_info") or []:
            if not isinstance(address, dict) or address.get("family") != "inet":
                continue
            local = address.get("local")
            prefix = address.get("prefixlen")
            if isinstance(local, str) and isinstance(prefix, int):
                records.append(
                    LocalAddress(local, prefix, name, "linux-ip")
                )
    return _normalize_records(records)


def _powershell_executable() -> str | None:
    candidates = (
        "pwsh.exe",
        "powershell.exe",
        "pwsh",
        "powershell",
    )
    return next((path for name in candidates if (path := shutil.which(name))), None)


def discover_windows_addresses() -> list[LocalAddress]:
    executable = _powershell_executable()
    if not executable:
        return []
    command = (
        "$OutputEncoding=[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
        "Get-NetIPAddress -AddressFamily IPv4 | "
        "Where-Object {$_.AddressState -eq 'Preferred'} | "
        "ForEach-Object {[PSCustomObject]@{"
        "interface=$_.InterfaceAlias;address=$_.IPAddress;"
        "prefix_length=$_.PrefixLength}} | ConvertTo-Json -Compress"
    )
    payload = _load_json_command([executable, "-NoProfile", "-Command", command])
    if isinstance(payload, dict):
        payload = [payload]
    records: list[LocalAddress] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        address = item.get("address")
        prefix = item.get("prefix_length")
        interface = item.get("interface")
        if isinstance(address, str) and isinstance(prefix, int):
            records.append(
                LocalAddress(
                    address,
                    prefix,
                    str(interface or "unknown"),
                    WINDOWS_SOURCE,
                )
            )
    return _normalize_records(records)


def discover_local_addresses(source: str = "auto") -> list[LocalAddress]:
    records: list[LocalAddress] = []
    errors: list[str] = []
    if source in {"auto", "linux"}:
        try:
            records.extend(discover_linux_addresses())
        except (OSError, ProxyCommandError, subprocess.SubprocessError) as exc:
            if source == "linux":
                raise
            errors.append(f"linux: {exc}")
    if source in {"auto", "windows"}:
        try:
            records.extend(discover_windows_addresses())
        except (OSError, ProxyCommandError, subprocess.SubprocessError) as exc:
            if source == "windows":
                raise
            errors.append(f"windows: {exc}")
    normalized = _normalize_records(records)
    if not normalized and errors:
        raise ProxyCommandError("; ".join(errors))
    return normalized


def parse_candidate(value: str) -> LocalAddress:
    interface = "manual"
    address_with_prefix = value
    if "=" in value:
        interface, address_with_prefix = value.split("=", 1)
    try:
        parsed = ipaddress.IPv4Interface(address_with_prefix)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be ADDRESS/PREFIX or INTERFACE=ADDRESS/PREFIX"
        ) from exc
    return LocalAddress(
        str(parsed.ip),
        parsed.network.prefixlen,
        interface,
        "manual",
    )


def load_route_mappings(path: Path) -> list[RouteMapping]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProxyCommandError(f"cannot read proxy route map: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProxyCommandError(f"invalid proxy route map JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("routes"), list):
        raise ProxyCommandError("proxy route map must contain a routes array")
    mappings: list[RouteMapping] = []
    for index, item in enumerate(payload["routes"]):
        if not isinstance(item, dict):
            raise ProxyCommandError(f"proxy route map entry {index} must be an object")
        try:
            mapping = RouteMapping(
                remote_network=str(item["remote_network"]),
                local_network=str(item["local_network"]),
                name=str(item.get("name") or f"route-{index}"),
            )
            mapping.remote
            mapping.local
        except (KeyError, ValueError) as exc:
            raise ProxyCommandError(
                f"invalid proxy route map entry {index}: {exc}"
            ) from exc
        mappings.append(mapping)
    return mappings


def select_same_network_address(
    remote_ip: str,
    candidates: Iterable[LocalAddress],
    *,
    route_mappings: Iterable[RouteMapping] = (),
) -> tuple[ipaddress.IPv4Address, LocalAddress]:
    try:
        remote = ipaddress.IPv4Address(remote_ip)
    except ipaddress.AddressValueError as exc:
        raise ProxyCommandError(f"remote server must be an IPv4 address: {remote_ip}") from exc
    if not _valid_unicast(remote):
        raise ProxyCommandError(f"remote server must be a unicast IPv4 address: {remote_ip}")

    normalized = _normalize_records(candidates)
    matches = [candidate for candidate in normalized if remote in candidate.network]
    selected_mapping: RouteMapping | None = None
    if not matches:
        applicable = [
            mapping for mapping in route_mappings if remote in mapping.remote
        ]
        applicable.sort(key=lambda mapping: -mapping.remote.prefixlen)
        for mapping in applicable:
            mapped = [
                candidate for candidate in normalized if candidate.ip in mapping.local
            ]
            if mapped:
                matches = mapped
                selected_mapping = mapping
                break
    if not matches:
        observed = ", ".join(
            f"{candidate.interface}={candidate.address}/{candidate.prefix_length}"
            for candidate in normalized
        )
        suffix = f"; observed: {observed}" if observed else "; no local IPv4 addresses found"
        raise ProxyCommandError(
            f"no local IPv4 interface matches {remote} by declared subnet or "
            f"configured route map{suffix}"
        )

    source_priority = {WINDOWS_SOURCE: 0, "linux-ip": 1, "manual": 2}
    selected = sorted(
        matches,
        key=lambda item: (
            0 if remote in item.network else 1,
            -item.prefix_length if remote in item.network else 0,
            -(
                selected_mapping.local.prefixlen
                if selected_mapping is not None
                else 0
            ),
            source_priority.get(item.source, 9),
            item.interface,
            item.address,
        ),
    )[0]
    return remote, selected


def selection_method(
    remote: ipaddress.IPv4Address,
    selected: LocalAddress,
    *,
    route_mappings: Iterable[RouteMapping],
) -> str:
    if remote in selected.network:
        return "declared-subnet"
    for mapping in sorted(
        route_mappings, key=lambda item: -item.remote.prefixlen
    ):
        if remote in mapping.remote and selected.ip in mapping.local:
            return f"configured-route:{mapping.name}"
    raise ProxyCommandError("selected address is not covered by the route map")


def _validate_variable(name: str) -> str:
    if not VARIABLE_RE.fullmatch(name):
        raise ProxyCommandError(f"invalid shell variable name: {name}")
    return name


def render_remote_env_command(
    selected: LocalAddress,
    *,
    port: int,
    username_variable: str,
    password_variable: str,
    no_proxy: str,
) -> str:
    username_variable = _validate_variable(username_variable)
    password_variable = _validate_variable(password_variable)
    proxy_url = (
        f'http://${{{username_variable}}}:${{{password_variable}}}'
        f"@{selected.address}:{port}"
    )
    return "; ".join(
        [
            f': "${{{username_variable}:?source proxy credentials first}}"',
            f': "${{{password_variable}:?source proxy credentials first}}"',
            f'export http_proxy="{proxy_url}"',
            'export https_proxy="$http_proxy"',
            'export HTTP_PROXY="$http_proxy"',
            'export HTTPS_PROXY="$http_proxy"',
            f"export no_proxy={shlex.quote(no_proxy)}",
            'export NO_PROXY="$no_proxy"',
        ]
    )


def render_local_listener_command(
    selected: LocalAddress,
    *,
    port: int,
    username_variable: str,
    password_variable: str,
    executable: str,
) -> str:
    username_variable = _validate_variable(username_variable)
    password_variable = _validate_variable(password_variable)
    listener_url = (
        f'http://${{{username_variable}}}:${{{password_variable}}}'
        f"@{selected.address}:{port}"
    )
    quoted_executable = shlex.quote(executable)
    return "; ".join(
        [
            f': "${{{username_variable}:?source proxy credentials first}}"',
            f': "${{{password_variable}:?source proxy credentials first}}"',
            f"command -v {quoted_executable} >/dev/null 2>&1 || "
            f'{{ echo "missing proxy executable: {executable}" >&2; exit 127; }}',
            f'{quoted_executable} -L "{listener_url}"',
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("remote_ip", help="remote server IPv4 address")
    parser.add_argument(
        "--source",
        choices=("auto", "linux", "windows", "none"),
        default="auto",
        help="local address discovery backend",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=[],
        metavar="[IFACE=]ADDRESS/PREFIX",
        help="add an explicit local address candidate",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--route-map", type=Path, default=DEFAULT_ROUTE_MAP)
    parser.add_argument(
        "--strict-subnet",
        action="store_true",
        help="disable configured route mappings",
    )
    parser.add_argument("--username-variable", default="VAWS_PROXY_USERNAME")
    parser.add_argument("--password-variable", default="VAWS_PROXY_PASSWORD")
    parser.add_argument("--no-proxy", default=DEFAULT_NO_PROXY)
    parser.add_argument(
        "--command",
        choices=("remote-env", "local-listener"),
        default="remote-env",
    )
    parser.add_argument("--proxy-executable", default="gost")
    parser.add_argument("--json", action="store_true", help="include selection metadata")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ProxyCommandError("proxy port must be between 1 and 65535")
        candidates = list(args.candidate)
        if args.source != "none":
            candidates.extend(discover_local_addresses(args.source))
        route_mappings = (
            [] if args.strict_subnet else load_route_mappings(args.route_map)
        )
        remote, selected = select_same_network_address(
            args.remote_ip,
            candidates,
            route_mappings=route_mappings,
        )
        if args.command == "local-listener":
            command = render_local_listener_command(
                selected,
                port=args.port,
                username_variable=args.username_variable,
                password_variable=args.password_variable,
                executable=args.proxy_executable,
            )
        else:
            command = render_remote_env_command(
                selected,
                port=args.port,
                username_variable=args.username_variable,
                password_variable=args.password_variable,
                no_proxy=args.no_proxy,
            )
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "remote_ip": str(remote),
                        "selected": asdict(selected),
                        "selection_method": selection_method(
                            remote,
                            selected,
                            route_mappings=route_mappings,
                        ),
                        "command_type": args.command,
                        "command": command,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(command)
    except (OSError, ProxyCommandError, subprocess.SubprocessError) as exc:
        print(f"proxy command generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
