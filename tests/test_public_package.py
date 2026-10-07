from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTest(unittest.TestCase):
    def test_verifier(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "verify_results.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: 34,522 rows", result.stdout)

    def test_private_data_is_ignored(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/private/*", gitignore)
        self.assertIn("minhwa_raw_combined.csv", gitignore)


if __name__ == "__main__":
    unittest.main()
