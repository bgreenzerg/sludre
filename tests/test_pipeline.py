from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock

from src.core.config import AppConfig
from src.core.pipeline import TranscriptionPostProcessor
from src.core.wordlist_store import ReplacementRule, WordlistData, WordlistStore


class PipelineTests(unittest.TestCase):
    def test_wordlist_replacements_apply_without_llm(self) -> None:
        tmp = Path(".test_tmp") / f"pipeline_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            store = WordlistStore(tmp / "wordlist.json")
            store.save(
                WordlistData(
                    replacements=[
                        ReplacementRule(source="gpu", target="GPU", whole_word=True)
                    ],
                    preferred_terms=[],
                )
            )
            llm_refiner = Mock()
            cfg = AppConfig.defaults()
            cfg.wordlist_path = str(tmp / "wordlist.json")
            cfg.wordlist_enabled = True
            cfg.wordlist_apply_replacements = True
            cfg.llm_enabled = False
            processor = TranscriptionPostProcessor(store, llm_refiner)

            result = processor.process("gpu test", cfg)

            self.assertEqual(result.text, "GPU test")
            self.assertEqual(result.replacement_hits, 1)
            llm_refiner.refine.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_llm_failure_sets_error_and_keeps_pre_llm_text(self) -> None:
        tmp = Path(".test_tmp") / f"pipeline_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            store = WordlistStore(tmp / "wordlist.json")
            store.save(WordlistData())
            llm_refiner = Mock()
            llm_refiner.refine.side_effect = RuntimeError("timeout")
            cfg = AppConfig.defaults()
            cfg.wordlist_path = str(tmp / "wordlist.json")
            cfg.llm_enabled = True
            cfg.llm_provider = "openai_compatible"
            cfg.llm_base_url = "https://example.com/v1"
            cfg.llm_api_key = "secret"
            cfg.llm_model = "model"
            processor = TranscriptionPostProcessor(store, llm_refiner)

            result = processor.process("raw text", cfg)

            self.assertEqual(result.raw_text, "raw text")
            self.assertEqual(result.text, "raw text")
            self.assertIsNotNone(result.llm_error)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
