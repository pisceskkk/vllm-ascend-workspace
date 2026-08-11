#!/usr/bin/env python3
"""Tests for remote-reachable local proxy command generation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "remote-code-parity"
    / "scripts"
    / "proxy_command.py"
)


def load_module():
    name = "_proxy_command_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proxy_command = load_module()


class ProxyCommandTests(unittest.TestCase):
    def test_selects_address_on_remote_network(self) -> None:
        candidates = [
            proxy_command.LocalAddress("172.28.32.1", 20, "wsl0", "linux-ip"),
            proxy_command.LocalAddress(
                "10.20.30.5", 24, "Ethernet", "windows-powershell"
            ),
        ]

        remote, selected = proxy_command.select_same_network_address(
            "10.20.30.99", candidates
        )

        self.assertEqual(str(remote), "10.20.30.99")
        self.assertEqual(selected.address, "10.20.30.5")

    def test_prefers_longest_matching_prefix(self) -> None:
        candidates = [
            proxy_command.LocalAddress("10.0.0.5", 8, "broad", "linux-ip"),
            proxy_command.LocalAddress("10.20.30.5", 24, "exact", "linux-ip"),
        ]

        _, selected = proxy_command.select_same_network_address(
            "10.20.30.99", candidates
        )

        self.assertEqual(selected.interface, "exact")

    def test_no_matching_network_fails_closed(self) -> None:
        candidates = [
            proxy_command.LocalAddress("192.168.1.10", 24, "lan", "linux-ip")
        ]

        with self.assertRaises(proxy_command.ProxyCommandError):
            proxy_command.select_same_network_address("10.20.30.99", candidates)

    def test_configured_vpn_route_matches_remote_network(self) -> None:
        candidates = [
            proxy_command.LocalAddress(
                "90.254.69.160", 32, "usg1", "windows-powershell"
            ),
            proxy_command.LocalAddress(
                "80.254.11.77", 24, "tap4", "windows-powershell"
            ),
        ]

        remote, selected = proxy_command.select_same_network_address(
            "90.90.97.14",
            candidates,
            route_mappings=[
                proxy_command.RouteMapping(
                    "90.90.97.0/24",
                    "90.254.69.160/32",
                    "managed-90-network",
                )
            ],
        )

        self.assertEqual(selected.address, "90.254.69.160")
        self.assertEqual(
            proxy_command.selection_method(
                remote,
                selected,
                route_mappings=[
                    proxy_command.RouteMapping(
                        "90.90.97.0/24",
                        "90.254.69.160/32",
                        "managed-90-network",
                    )
                ],
            ),
            "configured-route:managed-90-network",
        )

    def test_unconfigured_same_first_octet_fails_closed(self) -> None:
        candidates = [
            proxy_command.LocalAddress(
                "90.254.69.160", 32, "usg1", "windows-powershell"
            )
        ]

        with self.assertRaises(proxy_command.ProxyCommandError):
            proxy_command.select_same_network_address(
                "90.1.2.3",
                candidates,
                route_mappings=[
                    proxy_command.RouteMapping(
                        "90.90.97.0/24",
                        "90.254.69.160/32",
                        "managed-90-network",
                    )
                ],
            )

    def test_default_route_map_is_valid(self) -> None:
        mappings = proxy_command.load_route_mappings(
            proxy_command.DEFAULT_ROUTE_MAP
        )

        self.assertEqual(len(mappings), 3)
        self.assertEqual(mappings[1].name, "managed-90-network")

    def test_remote_command_uses_credential_variables(self) -> None:
        selected = proxy_command.LocalAddress(
            "10.20.30.5", 24, "Ethernet", "windows-powershell"
        )

        command = proxy_command.render_remote_env_command(
            selected,
            port=8080,
            username_variable="VAWS_PROXY_USERNAME",
            password_variable="VAWS_PROXY_PASSWORD",
            no_proxy=proxy_command.DEFAULT_NO_PROXY,
        )

        self.assertIn("${VAWS_PROXY_USERNAME}", command)
        self.assertIn("${VAWS_PROXY_PASSWORD}", command)
        self.assertIn("@10.20.30.5:8080", command)
        self.assertIn("export HTTPS_PROXY", command)
        self.assertNotIn("replace-with", command)

    def test_local_listener_command_uses_selected_bind_address(self) -> None:
        selected = proxy_command.LocalAddress(
            "90.254.69.160", 32, "usg1", "windows-powershell"
        )

        command = proxy_command.render_local_listener_command(
            selected,
            port=8080,
            username_variable="VAWS_PROXY_USERNAME",
            password_variable="VAWS_PROXY_PASSWORD",
            executable="gost",
        )

        self.assertIn("command -v gost", command)
        self.assertIn("gost -L", command)
        self.assertIn("@90.254.69.160:8080", command)
        self.assertNotIn("replace-with", command)

    def test_manual_candidate_parser_accepts_interface_name(self) -> None:
        candidate = proxy_command.parse_candidate("Ethernet=10.20.30.5/24")

        self.assertEqual(candidate.interface, "Ethernet")
        self.assertEqual(candidate.address, "10.20.30.5")
        self.assertEqual(candidate.prefix_length, 24)

    def test_auto_discovery_falls_back_to_windows_from_wsl(self) -> None:
        windows_address = proxy_command.LocalAddress(
            "10.20.30.5", 24, "Ethernet", "windows-powershell"
        )
        with (
            mock.patch.object(
                proxy_command,
                "discover_linux_addresses",
                side_effect=proxy_command.ProxyCommandError("netlink denied"),
            ),
            mock.patch.object(
                proxy_command,
                "discover_windows_addresses",
                return_value=[windows_address],
            ),
        ):
            addresses = proxy_command.discover_local_addresses("auto")

        self.assertEqual(addresses, [windows_address])

    def test_loopback_named_interface_is_not_a_proxy_candidate(self) -> None:
        addresses = proxy_command._normalize_records(
            [
                proxy_command.LocalAddress(
                    "10.255.255.254", 32, "lo", "windows-powershell"
                ),
                proxy_command.LocalAddress(
                    "10.20.30.5", 24, "Ethernet", "windows-powershell"
                ),
            ]
        )

        self.assertEqual([item.interface for item in addresses], ["Ethernet"])


if __name__ == "__main__":
    unittest.main()
