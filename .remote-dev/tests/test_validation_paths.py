from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_remote_dev_scaffold.py"


def load_validator():
    name = "_remote_dev_validator_test"
    spec = importlib.util.spec_from_file_location(name, VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = load_validator()


class ValidationPathTests(unittest.TestCase):
    def test_narrow_root_is_the_scratch_directory(self) -> None:
        endpoint = {"root": "/tmp", "cwd": "/tmp"}
        scratch_root, scratch, narrow = validator.validation_paths(
            endpoint,
            "20260725T000000Z",
        )

        self.assertEqual(scratch_root, "/tmp")
        self.assertEqual(
            scratch,
            "/tmp/.remote-dev/validation/20260725T000000Z",
        )
        self.assertEqual(narrow["root"], scratch)
        self.assertEqual(narrow["cwd"], scratch)


if __name__ == "__main__":
    unittest.main()
