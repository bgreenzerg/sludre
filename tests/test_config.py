from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

from src.core.config import (
    AppConfig,
    ConfigStore,
    _default_model_cache_dir,
    _legacy_default_model_cache_dir,
)


class ConfigStoreTests(unittest.TestCase):
    def test_load_creates_defaults_when_file_missing(self) -> None:
        tmp = Path(".test_tmp") / f"config_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            config_path = tmp / "config.json"
            store = ConfigStore(config_path)

            cfg = store.load()

            self.assertEqual(cfg.hotkey, "ctrl+space")
            self.assertEqual(cfg.model_cache_dir, str(_default_model_cache_dir()))
            self.assertGreaterEqual(len(cfg.llm_prompt_presets), 1)
            self.assertTrue(config_path.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_save_and_reload_round_trip(self) -> None:
        tmp = Path(".test_tmp") / f"config_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            config_path = tmp / "config.json"
            store = ConfigStore(config_path)
            cfg = AppConfig.defaults()
            cfg.language = "da"
            cfg.manual_model_path = r"C:\models\hviske-v2"
            cfg.hf_token = "secret-token"
            cfg.llm_prompt_presets = [
                {"name": "Standard", "prompt": "default"},
                {"name": "Formel", "prompt": "rewrite formally"},
            ]
            cfg.llm_selected_prompt_name = "Formel"
            store.save(cfg)

            loaded = store.load()

            self.assertEqual(loaded.language, "da")
            self.assertEqual(loaded.manual_model_path, r"C:\models\hviske-v2")
            self.assertEqual(loaded.hf_token, "secret-token")
            self.assertEqual(loaded.llm_selected_prompt_name, "Formel")
            self.assertEqual(loaded.llm_system_prompt, "rewrite formally")
            self.assertEqual(len(loaded.llm_prompt_presets), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_migrates_legacy_model_cache_path(self) -> None:
        tmp = Path(".test_tmp") / f"config_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            config_path = tmp / "config.json"
            store = ConfigStore(config_path)
            cfg = AppConfig.defaults()
            cfg.model_cache_dir = str(_legacy_default_model_cache_dir())
            store.save(cfg)

            loaded = store.load()

            self.assertEqual(loaded.model_cache_dir, str(_default_model_cache_dir()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
