from __future__ import annotations

import importlib.util
import io
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "jiguang_device_access.py"
SPEC = importlib.util.spec_from_file_location("jiguang_device_access", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_plan_preserves_managed_container_port() -> None:
    record = {
        "alias": "80.5.9.113",
        "container": {"name": "vaws-qcs", "ssh_port": 46001},
    }
    result = module.plan(record)
    assert result["container_ssh_port"] == 46001
    assert result["preserve_container_port"] is True
    assert result["password_source"] == "stdin"


def test_password_read_rejects_short_or_colon(monkeypatch) -> None:
    monkeypatch.setattr(module.sys, "stdin", io.StringIO("short\n"))
    try:
        module.read_password_from_stdin()
    except ValueError as exc:
        assert "16-256" in str(exc)
    else:
        raise AssertionError("short password was accepted")

    monkeypatch.setattr(module.sys, "stdin", io.StringIO("long-enough:password-value\n"))
    try:
        module.read_password_from_stdin()
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("colon password was accepted")


def test_password_is_sent_only_on_stdin(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(module.machine_ops, "find_public_key", lambda _: Path("/tmp/id.pub"))
    monkeypatch.setattr(module.machine_ops, "private_key_for_public_key", lambda _: Path("/tmp/id"))
    monkeypatch.setattr(module.machine_ops, "ssh_command", lambda *args, **kwargs: ["ssh", "host"])
    monkeypatch.setattr(module.machine_ops, "remote_shell_command", lambda argv: "docker exec -i vaws-qcs chpasswd")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    password = "FreshPassword-1234!"
    module.set_root_password(module.machine_ops.SshTarget("host"), "vaws-qcs", password)

    assert password not in " ".join(captured["command"])
    assert captured["input"] == f"root:{password}\n"
