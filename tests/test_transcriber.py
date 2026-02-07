from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.transcriber import Transcriber


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    language = "da"


class _FakeModel:
    def __init__(self, model_path, device, compute_type) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio, language, beam_size, vad_filter):
        del audio, language, beam_size, vad_filter
        return [_FakeSegment(" hej"), _FakeSegment(" verden ")], _FakeInfo()


class _FakeAudio:
    def __init__(self, sample_count: int) -> None:
        self.ndim = 1
        self.shape = (sample_count,)


class TranscriberTests(unittest.TestCase):
    @patch("src.core.transcriber._import_whisper_model", return_value=_FakeModel)
    def test_transcribe_returns_joined_text(self, _import_mock) -> None:
        transcriber = Transcriber(Path("model"), device="cuda")
        audio = _FakeAudio(16000)

        result = transcriber.transcribe(audio=audio, sample_rate=16000, language="da")

        self.assertEqual(result.text, "hej verden")
        self.assertEqual(result.language, "da")
        self.assertGreaterEqual(result.duration_sec, 1.0)

    @patch("src.core.transcriber._import_whisper_model", return_value=_FakeModel)
    def test_transcribe_rejects_non_16khz(self, _import_mock) -> None:
        transcriber = Transcriber(Path("model"), device="cuda")
        audio = _FakeAudio(8000)

        with self.assertRaises(ValueError):
            transcriber.transcribe(audio=audio, sample_rate=8000, language="da")


if __name__ == "__main__":
    unittest.main()
