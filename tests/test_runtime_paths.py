from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import runtime_paths


class RuntimePathsTests(unittest.TestCase):
    def test_source_runtime_root_points_to_project(self) -> None:
        root = runtime_paths.runtime_root()
        self.assertTrue((root / "src").exists())
        self.assertTrue((root / "assets").exists())

    def test_frozen_runtime_root_uses_executable_parent(self) -> None:
        fake_exe = str(Path("C:/apps/Sludre/Sludre.exe"))
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", fake_exe):
                self.assertEqual(runtime_paths.runtime_root(), Path("C:/apps/Sludre"))
                self.assertEqual(runtime_paths.assets_dir(), Path("C:/apps/Sludre/assets"))
                self.assertEqual(runtime_paths.models_dir(), Path("C:/apps/Sludre/models"))
                self.assertEqual(runtime_paths.project_env_path(), Path("C:/apps/Sludre/.env"))


if __name__ == "__main__":
    unittest.main()
