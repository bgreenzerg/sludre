from __future__ import annotations

import threading
import time
from typing import cast

import numpy as np
import sounddevice as sd


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        max_record_seconds: int = 60,
        silence_trim: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.silence_trim = silence_trim
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._is_recording = False
        self._started_at = 0.0

    def is_recording(self) -> bool:
        return self._is_recording

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            # Stream status is informational and not fatal by itself.
            pass
        mono = cast(np.ndarray, indata[:, 0] if indata.ndim > 1 else indata)
        with self._lock:
            self._frames.append(mono.copy())

    def start(self) -> None:
        if self._is_recording:
            return
        with self._lock:
            self._frames.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._started_at = time.monotonic()
        self._is_recording = True

    def stop(self) -> np.ndarray:
        if not self._is_recording:
            return np.array([], dtype=np.float32)

        self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames).astype(np.float32, copy=False)
            self._frames.clear()

        max_samples = int(self.sample_rate * self.max_record_seconds)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]

        if self.silence_trim and audio.shape[0] > 0:
            audio = self._trim_silence(audio)
        return audio

    @staticmethod
    def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        idx = np.where(np.abs(audio) > threshold)[0]
        if idx.size == 0:
            return np.array([], dtype=np.float32)
        return audio[idx[0] : idx[-1] + 1]
