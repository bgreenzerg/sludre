from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.core.config import AppConfig, DEFAULT_MISTRAL_BASE_URL
from src.core.env_secrets import EnvSecretsStore


def _urlopen(request: Any, timeout: int):
    from urllib.request import urlopen

    return urlopen(request, timeout=timeout)


def _request(url: str, data: bytes, headers: dict[str, str]) -> Any:
    from urllib.request import Request

    return Request(url=url, data=data, headers=headers, method="POST")


class LlmRefineError(RuntimeError):
    pass


@dataclass
class LlmRefineResult:
    text: str
    latency_ms: int
    provider: str
    model: str


class LlmRefiner:
    def __init__(self, log_callback: Callable[[str], None] | None = None) -> None:
        self.logger = logging.getLogger("sludre.llm_refiner")
        self.log_callback = log_callback

    def _emit_ui_log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _preview_text(text: str, max_chars: int = 220) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3] + "..."

    def refine(
        self,
        text: str,
        config: AppConfig,
        preferred_terms: list[str],
    ) -> LlmRefineResult:
        if not config.llm_enabled:
            raise LlmRefineError("LLM refine requested while disabled.")
        model = self._resolve_model(config)
        api_key = self._resolve_api_key(config)
        endpoint = self._resolve_endpoint(config)
        messages = self._build_messages(
            text=text,
            system_prompt=config.llm_system_prompt,
            preferred_terms=preferred_terms,
        )
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(config.llm_temperature),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        self.logger.info(
            "Sending LLM refine request. provider=%s model=%s endpoint=%s timeout=%ss",
            config.llm_provider,
            model,
            endpoint,
            config.llm_timeout_seconds,
        )
        self._emit_ui_log(
            f"LLM request -> provider={config.llm_provider} model={model} endpoint={endpoint} "
            f"timeout={config.llm_timeout_seconds}s input_chars={len(text)} preferred_terms={len(preferred_terms)}"
        )
        self._emit_ui_log(
            f"LLM input preview: {self._preview_text(text)}"
        )
        started = time.perf_counter()
        try:
            response = _urlopen(
                _request(endpoint, body, headers),
                timeout=config.llm_timeout_seconds,
            )
            status = getattr(response, "status", None) or getattr(response, "code", None)
            raw = response.read().decode("utf-8")
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.logger.info(
                "LLM refine response received. provider=%s model=%s status=%s latency_ms=%s",
                config.llm_provider,
                model,
                status,
                latency_ms,
            )
            text_out = self._extract_text_from_response(raw)
            if not text_out.strip():
                raise LlmRefineError("LLM returned empty content.")
            self._emit_ui_log(
                f"LLM response <- provider={config.llm_provider} model={model} status={status} latency={latency_ms}ms output_chars={len(text_out.strip())}"
            )
            self._emit_ui_log(
                f"LLM output preview: {self._preview_text(text_out)}"
            )
            return LlmRefineResult(
                text=text_out.strip(),
                latency_ms=latency_ms,
                provider=config.llm_provider,
                model=model,
            )
        except LlmRefineError:
            self._emit_ui_log(
                f"LLM request failed (handled): provider={config.llm_provider} model={model} endpoint={endpoint}"
            )
            raise
        except Exception as exc:
            self._emit_ui_log(
                f"LLM request failed: provider={config.llm_provider} model={model} endpoint={endpoint} error={exc}"
            )
            raise LlmRefineError(str(exc)) from exc

    @staticmethod
    def _resolve_api_key(config: AppConfig) -> str:
        api_key = config.llm_api_key.strip()
        if api_key:
            return api_key
        env_key = os.getenv("LLM_API_KEY", "").strip()
        if env_key:
            return env_key
        dot_env_key = EnvSecretsStore.default().get_secret("LLM_API_KEY").strip()
        if dot_env_key:
            return dot_env_key
        raise LlmRefineError("Missing LLM API key.")

    @staticmethod
    def _resolve_model(config: AppConfig) -> str:
        custom = config.llm_model.strip()
        if custom:
            return custom
        if config.llm_provider == "mistral_api":
            preset = config.mistral_model_preset.strip()
            if preset:
                return preset
        raise LlmRefineError("Missing LLM model name.")

    @staticmethod
    def _resolve_endpoint(config: AppConfig) -> str:
        if config.llm_provider == "mistral_api":
            base = config.mistral_base_url.strip() or DEFAULT_MISTRAL_BASE_URL
            return base.rstrip("/") + "/chat/completions"
        base_url = config.llm_base_url.strip()
        if not base_url:
            raise LlmRefineError("Missing OpenAI-compatible base URL.")
        return base_url.rstrip("/") + "/chat/completions"

    @staticmethod
    def _build_messages(
        text: str,
        system_prompt: str,
        preferred_terms: list[str],
    ) -> list[dict[str, str]]:
        system_content = system_prompt.strip()
        if preferred_terms:
            joined = ", ".join(preferred_terms)
            system_content += (
                "\n\nPrefer these terms when appropriate: "
                f"{joined}"
            )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

    @staticmethod
    def _extract_text_from_response(raw: str) -> str:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LlmRefineError(f"Invalid JSON response: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LlmRefineError("Unexpected LLM response schema.") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    pieces.append(str(item["text"]))
                elif isinstance(item, str):
                    pieces.append(item)
            return "".join(pieces)
        return str(content)
