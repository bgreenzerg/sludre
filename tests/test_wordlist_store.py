from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from src.core.wordlist_store import ReplacementRule, WordlistData, WordlistStore


class WordlistStoreTests(unittest.TestCase):
    def test_load_creates_default_file(self) -> None:
        tmp = Path(".test_tmp") / f"wordlist_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp / "wordlist.json"
            store = WordlistStore(path)

            data = store.load()

            self.assertEqual(data.replacements, [])
            self.assertTrue(path.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_save_and_load_roundtrip(self) -> None:
        tmp = Path(".test_tmp") / f"wordlist_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp / "wordlist.json"
            store = WordlistStore(path)
            expected = WordlistData(
                replacements=[
                    ReplacementRule(
                        source="wrong",
                        target="right",
                        match_case=False,
                        whole_word=True,
                    )
                ],
                preferred_terms=["Hviske", "CUDA"],
            )

            store.save(expected)
            loaded = store.load()

            self.assertEqual(len(loaded.replacements), 1)
            self.assertEqual(loaded.replacements[0].source, "wrong")
            self.assertEqual(loaded.preferred_terms, ["Hviske", "CUDA"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
