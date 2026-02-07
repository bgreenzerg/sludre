from __future__ import annotations

import unittest

from src.core.text_cleaner import apply_wordlist_replacements
from src.core.wordlist_store import ReplacementRule


class TextCleanerTests(unittest.TestCase):
    def test_applies_case_insensitive_whole_word_replacement(self) -> None:
        rules = [
            ReplacementRule(
                source="hviske",
                target="Hviske",
                match_case=False,
                whole_word=True,
            )
        ]

        result = apply_wordlist_replacements("HVISKE appen er god", rules)

        self.assertEqual(result.text, "Hviske appen er god")
        self.assertEqual(result.replacement_hits, 1)

    def test_does_not_replace_partial_words_when_whole_word_enabled(self) -> None:
        rules = [ReplacementRule(source="cat", target="dog", whole_word=True)]

        result = apply_wordlist_replacements("concatenate cat category", rules)

        self.assertEqual(result.text, "concatenate dog category")
        self.assertEqual(result.replacement_hits, 1)


if __name__ == "__main__":
    unittest.main()
