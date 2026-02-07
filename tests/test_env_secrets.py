from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from src.core.env_secrets import EnvSecretsStore


class EnvSecretsStoreTests(unittest.TestCase):
    def test_ensure_exists_creates_env_file(self) -> None:
        tmp = Path(".test_tmp") / f"env_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp / ".env"
            store = EnvSecretsStore(path)
            store.ensure_exists()
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_set_get_and_remove_secret(self) -> None:
        tmp = Path(".test_tmp") / f"env_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp / ".env"
            store = EnvSecretsStore(path)
            store.set_secret("LLM_API_KEY", "abc 123")
            store.set_secret("HF_TOKEN", "hf_testtoken")
            self.assertEqual(store.get_secret("LLM_API_KEY"), "abc 123")
            self.assertEqual(store.get_secret("HF_TOKEN"), "hf_testtoken")

            store.set_secret("HF_TOKEN", "")
            self.assertEqual(store.get_secret("HF_TOKEN"), "")
            self.assertIn("LLM_API_KEY", path.read_text(encoding="utf-8"))
            self.assertNotIn("HF_TOKEN", path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
