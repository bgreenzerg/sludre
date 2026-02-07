from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import shutil
import uuid

from src.core.model_manager import INFERENCE_ALLOW_PATTERNS, ModelManager


class ModelManagerTests(unittest.TestCase):
    def test_uses_manual_model_path_when_present(self) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            manual = tmp / "manual"
            manual.mkdir()
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=tmp / "cache",
                manual_model_path=str(manual),
            )

            resolved = manager.ensure_model_available()

            self.assertEqual(resolved, manual)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("src.core.model_manager._subprocess_run")
    @patch("src.core.model_manager._snapshot_download")
    def test_downloads_model_with_cli_when_available(
        self, snapshot_mock, subprocess_run_mock
    ) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            cache_dir = tmp / "cache"
            target = cache_dir / "syvai--hviske-v2"
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors.index.json").write_text(
                "{}", encoding="utf-8"
            )
            (target / "model.bin").write_bytes(b"ok")
            subprocess_run_mock.return_value = Mock(
                returncode=0, stdout="", stderr=""
            )
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=cache_dir,
                manual_model_path=None,
                hf_token="test-token",
            )

            resolved = manager.ensure_model_available()

            self.assertEqual(resolved, target)
            snapshot_mock.assert_not_called()
            subprocess_run_mock.assert_called_once()
            command = subprocess_run_mock.call_args[0][0]
            self.assertGreaterEqual(len(command), 2)
            self.assertEqual(command[1], "download")
            self.assertIn("download", command)
            self.assertIn("syvai/hviske-v2", command)
            self.assertIn("--local-dir", command)
            self.assertIn(str(target), command)
            self.assertIn("--token", command)
            self.assertIn("test-token", command)
            self.assertIn("--include", command)
            include_index = command.index("--include")
            self.assertEqual(
                command[include_index + 1 : include_index + 1 + len(INFERENCE_ALLOW_PATTERNS)],
                INFERENCE_ALLOW_PATTERNS,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("src.core.model_manager._subprocess_run")
    @patch("src.core.model_manager._snapshot_download")
    def test_falls_back_to_sdk_when_cli_fails(
        self, snapshot_mock, subprocess_run_mock
    ) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            cache_dir = tmp / "cache"
            target = cache_dir / "syvai--hviske-v2"
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.bin").write_bytes(b"ok")
            subprocess_run_mock.return_value = Mock(
                returncode=1, stdout="", stderr="cli error"
            )
            snapshot_mock.return_value = str(target)
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=cache_dir,
                manual_model_path=None,
                hf_token="test-token",
            )

            resolved = manager.ensure_model_available()

            self.assertEqual(resolved, target)
            snapshot_mock.assert_called_once()
            _, kwargs = snapshot_mock.call_args
            self.assertEqual(kwargs["token"], "test-token")
            self.assertEqual(kwargs["local_dir"], str(target))
            self.assertEqual(kwargs["allow_patterns"], INFERENCE_ALLOW_PATTERNS)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("src.core.model_manager._subprocess_run")
    @patch("src.core.model_manager._snapshot_download")
    def test_tries_next_cli_entrypoint_on_missing_module_error(
        self, snapshot_mock, subprocess_run_mock
    ) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            cache_dir = tmp / "cache"
            target = cache_dir / "syvai--hviske-v2"
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors.index.json").write_text(
                "{}", encoding="utf-8"
            )
            (target / "model.bin").write_bytes(b"ok")
            subprocess_run_mock.side_effect = [
                Mock(
                    returncode=1,
                    stdout="",
                    stderr="No module named huggingface_hub.commands",
                ),
                Mock(returncode=0, stdout="", stderr=""),
            ]
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=cache_dir,
                manual_model_path=None,
                hf_token="test-token",
            )

            resolved = manager.ensure_model_available()

            self.assertEqual(resolved, target)
            snapshot_mock.assert_not_called()
            self.assertGreaterEqual(subprocess_run_mock.call_count, 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("src.core.model_manager._module_available", return_value=True)
    @patch("src.core.model_manager._create_transformers_converter")
    def test_converts_transformers_model_to_ctranslate2(
        self, converter_factory_mock, _module_available_mock
    ) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            manual = tmp / "manual"
            manual.mkdir(parents=True, exist_ok=True)
            (manual / "config.json").write_text("{}", encoding="utf-8")
            (manual / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

            class _FakeConverter:
                def convert(self, output_dir: str, quantization: str, force: bool) -> None:
                    del quantization, force
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "model.bin").write_bytes(b"ok")

            converter_factory_mock.return_value = _FakeConverter()
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=tmp / "cache",
                manual_model_path=str(manual),
            )

            resolved = manager.ensure_model_available()

            self.assertEqual(resolved, manual / "ctranslate2")
            converter_factory_mock.assert_called_once()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("src.core.model_manager._module_available", return_value=False)
    def test_conversion_reports_missing_dependencies(self, _module_available_mock) -> None:
        tmp = Path(".test_tmp") / f"model_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            manual = tmp / "manual"
            manual.mkdir(parents=True, exist_ok=True)
            (manual / "config.json").write_text("{}", encoding="utf-8")
            (manual / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            manager = ModelManager(
                repo_id="syvai/hviske-v2",
                cache_dir=tmp / "cache",
                manual_model_path=str(manual),
            )

            with self.assertRaises(RuntimeError) as ctx:
                manager.ensure_model_available()

            self.assertIn("Missing dependencies for conversion", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
