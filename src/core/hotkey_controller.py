from __future__ import annotations

from collections.abc import Callable

import keyboard


class HotkeyController:
    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        self.on_start = on_start
        self.on_stop = on_stop
        self._space_press_hook = None
        self._space_release_hook = None
        self._active = False

    def register(self) -> None:
        if self._space_press_hook is not None:
            return
        self._space_press_hook = keyboard.on_press_key("space", self._on_space_down)
        self._space_release_hook = keyboard.on_release_key(
            "space", self._on_space_up
        )

    def unregister(self) -> None:
        if self._space_press_hook is not None:
            keyboard.unhook(self._space_press_hook)
            self._space_press_hook = None
        if self._space_release_hook is not None:
            keyboard.unhook(self._space_release_hook)
            self._space_release_hook = None
        self._active = False

    def _on_space_down(self, _event) -> None:
        if self._active:
            return
        if keyboard.is_pressed("ctrl"):
            self._active = True
            self.on_start()

    def _on_space_up(self, _event) -> None:
        if not self._active:
            return
        self._active = False
        self.on_stop()
