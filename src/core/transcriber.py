from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_sec: float
    latency_ms: int


class Transcriber:
    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        compute_type: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _compute_type_candidates(self) -> list[str]:
        if self.compute_type:
            return [self.compute_type]
        if self.device == "cuda":
            return ["float16", "int8_float16", "int8"]
        return ["int8", "float32"]

    def load(self) -> None:
        whisper_model_cls = _import_whisper_model()
        errors: list[str] = []
        for candidate in self._compute_type_candidates():
            try:
                self._model = whisper_model_cls(
                    str(self.model_path),
                    device=self.device,
                    compute_type=candidate,
                )
                self.compute_type = candidate
                return
            except Exception as exc:  # pragma: no cover - device specific path
                errors.append(f"{candidate}: {exc}")
        joined = "\n".join(errors) if errors else "No compute types tried."
        raise RuntimeError(f"Failed to load model.\n{joined}")

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def transcribe(
        self,
        audio: Any,
        sample_rate: int,
        language: str = "da",
    ) -> TranscriptionResult:
        self._ensure_loaded()
        if getattr(audio, "ndim", 1) != 1:
            raise ValueError("Expected mono audio as 1D array.")
        if sample_rate != 16000:
            raise ValueError("Expected sample rate of 16000 Hz.")

        start = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        detected_language = getattr(info, "language", language)
        sample_count = int(audio.shape[0]) if hasattr(audio, "shape") else len(audio)
        return TranscriptionResult(
            text=text,
            language=detected_language,
            duration_sec=float(sample_count) / float(sample_rate),
            latency_ms=elapsed_ms,
        )
