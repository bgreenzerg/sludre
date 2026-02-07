from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.core.config import AppConfig
from src.core.llm_refiner import LlmRefineError, LlmRefiner


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class LlmRefinerTests(unittest.TestCase):
    @patch("src.core.llm_refiner._urlopen")
    def test_openai_compatible_request(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "renset tekst"}},
                ]
            }
        )
        cfg = AppConfig.defaults()
        cfg.llm_enabled = True
        cfg.llm_provider = "openai_compatible"
        cfg.llm_base_url = "https://example.com/v1"
        cfg.llm_api_key = "secret"
        cfg.llm_model = "gpt-test"
        cfg.llm_timeout_seconds = 5
        refiner = LlmRefiner()

        result = refiner.refine("raw text", cfg, preferred_terms=["Hviske"])

        self.assertEqual(result.text, "renset tekst")
        request = urlopen_mock.call_args[0][0]
        self.assertTrue(str(request.full_url).endswith("/chat/completions"))
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["messages"][1]["content"], "raw text")

    @patch("src.core.llm_refiner._urlopen")
    def test_mistral_provider_uses_preset_model(self, urlopen_mock) -> None:
        urlopen_mock.return_value = _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "mistral output"}},
                ]
            }
        )
        cfg = AppConfig.defaults()
        cfg.llm_enabled = True
        cfg.llm_provider = "mistral_api"
        cfg.mistral_base_url = "https://api.mistral.ai/v1"
        cfg.llm_api_key = "secret"
        cfg.mistral_model_preset = "mistral-small-latest"
        refiner = LlmRefiner()

        result = refiner.refine("hej", cfg, preferred_terms=[])

        self.assertEqual(result.model, "mistral-small-latest")
        request = urlopen_mock.call_args[0][0]
        self.assertEqual(
            str(request.full_url),
            "https://api.mistral.ai/v1/chat/completions",
        )

    def test_missing_api_key_raises(self) -> None:
        cfg = AppConfig.defaults()
        cfg.llm_enabled = True
        cfg.llm_provider = "openai_compatible"
        cfg.llm_base_url = "https://example.com/v1"
        cfg.llm_model = "gpt-test"
        refiner = LlmRefiner()

        with patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            with patch("src.core.llm_refiner.EnvSecretsStore.default") as secrets_default:
                secrets_default.return_value.get_secret.return_value = ""
                with self.assertRaises(LlmRefineError):
                    refiner.refine("hej", cfg, preferred_terms=[])


if __name__ == "__main__":
    unittest.main()
