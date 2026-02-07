from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.core.text_inserter import TextInserter


class TextInserterTests(unittest.TestCase):
    @patch("src.core.text_inserter.time.sleep", return_value=None)
    @patch("src.core.text_inserter._keyboard")
    @patch("src.core.text_inserter._pyperclip")
    def test_insert_text_uses_clipboard_and_restores(
        self,
        pyperclip_factory,
        keyboard_factory,
        _sleep_mock,
    ) -> None:
        pyperclip_mock = Mock()
        pyperclip_mock.paste.return_value = "old-value"
        keyboard_mock = Mock()
        pyperclip_factory.return_value = pyperclip_mock
        keyboard_factory.return_value = keyboard_mock
        inserter = TextInserter(restore_clipboard=True)

        inserter.insert_text_at_cursor("hej verden")

        keyboard_mock.send.assert_called_once_with("ctrl+v")
        self.assertEqual(pyperclip_mock.copy.call_count, 2)
        pyperclip_mock.copy.assert_any_call("hej verden")
        pyperclip_mock.copy.assert_any_call("old-value")

    @patch("src.core.text_inserter._keyboard")
    @patch("src.core.text_inserter._pyperclip")
    def test_insert_skips_empty_text(self, pyperclip_factory, keyboard_factory) -> None:
        pyperclip_mock = Mock()
        keyboard_mock = Mock()
        pyperclip_factory.return_value = pyperclip_mock
        keyboard_factory.return_value = keyboard_mock
        inserter = TextInserter()

        inserter.insert_text_at_cursor("   ")

        pyperclip_mock.copy.assert_not_called()
        keyboard_mock.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
