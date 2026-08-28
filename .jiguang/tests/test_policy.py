from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.errors import JiguangPolicyError  # noqa: E402
from core.policy import (  # noqa: E402
    DEPLOYMENT_INPUT_KEYS,
    assert_non_admin_profile,
    assert_owned,
    validate_payload,
)
from core.redaction import redact  # noqa: E402


class PolicyTests(unittest.TestCase):
    def test_rejects_foreign_resource(self) -> None:
        with self.assertRaises(JiguangPolicyError):
            assert_owned({"owner_user_id": "other"}, "635", "device")

    def test_requires_ownership_evidence(self) -> None:
        with self.assertRaises(JiguangPolicyError):
            assert_owned({"id": "device-1"}, "635", "device")

    def test_rejects_administrator_profile(self) -> None:
        with self.assertRaises(JiguangPolicyError):
            assert_non_admin_profile({"id": "1", "role": "system_admin"})

    def test_rejects_script_even_when_nested(self) -> None:
        with self.assertRaises(JiguangPolicyError):
            validate_payload(
                {"metadata": {"launch_command": "rm -rf /"}},
                DEPLOYMENT_INPUT_KEYS,
                "deployment",
            )

    def test_redacts_sensitive_fields_recursively(self) -> None:
        payload = redact({"access_token": "x", "nested": {"ssh_password": "y"}, "ok": 1})
        self.assertEqual(payload["access_token"], "<redacted>")
        self.assertEqual(payload["nested"]["ssh_password"], "<redacted>")
        self.assertEqual(payload["ok"], 1)

    def test_redacts_sensitive_free_text(self) -> None:
        payload = redact(
            {
                "message": "Authorization: Bearer secret-value",
                "log": "password=secret-value",
                "proxy": "http://user:secret-value@example.invalid:8080/",
            }
        )
        self.assertNotIn("secret-value", str(payload))

    def test_allows_workload_token_count_keys(self) -> None:
        payload = validate_payload(
            {"metadata": {"max_tokens": 512, "input_tokens": 128}},
            DEPLOYMENT_INPUT_KEYS,
            "deployment",
        )
        self.assertEqual(payload["metadata"]["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
