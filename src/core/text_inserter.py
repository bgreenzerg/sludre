from __future__ import annotations

import time


def _keyboard():
    import keyboard

    return keyboard


def _pyperclip():
    import pyperclip

    return pyperclip


class TextInserter:
    def __init__(self, restore_clipboard: bool = True) -> None:
        self.restore_clipboard = restore_clipboard

    def insert_text_at_cursor(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        keyboard = _keyboard()
        pyperclip = _pyperclip()

        previous = None
        if self.restore_clipboard:
            try:
                previous = pyperclip.paste()
            except Exception:
                previous = None

        pyperclip.copy(text)
        time.sleep(0.02)
        keyboard.send("ctrl+v")
        time.sleep(0.02)

        if self.restore_clipboard and previous is not None:
            pyperclip.copy(previous)
