from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "jiguang_device_key.py"
SPEC = importlib.util.spec_from_file_location("jiguang_device_key", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def material() -> object:
    return module.KeyMaterial(
        private_key=Path("/tmp/id_ed25519"),
        public_key=Path("/tmp/id_ed25519.pub"),
        fingerprint="SHA256:test",
    )


def test_plan_requires_selected_key_to_reach_container(monkeypatch) -> None:
    record = {
        "alias": "80.5.9.113",
        "host": {"ip": "80.5.9.113"},
        "container": {"name": "vaws-qcs", "ssh_port": 46001},
    }
    monkeypatch.setattr(module, "container_key_check", lambda *_: {"ok": True})
    result = module.plan(record, material(), "Codex:Jiguang:Device:test")
    assert result["container_key_verified"] is True
    assert result["registration_auth_type"] == "SSH_KEY"
    assert result["key_fingerprint"] == "SHA256:test"


def test_private_key_reference_uses_file_argument_not_key_content(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(module, "windows_path", lambda path: f"WIN:{path.name}")

    def fake_run(command, **kwargs):
        captured["command"] = command
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.store_private_key_reference("Codex:Jiguang:Device:test", material())
    command = " ".join(map(str, captured["command"]))
    assert "-SecretFile" in command
    assert "WIN:id_ed25519" in command
    assert "PRIVATE KEY" not in command
