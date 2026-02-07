from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.config import AppConfig
from src.core.llm_refiner import LlmRefineError, LlmRefiner
from src.core.text_cleaner import apply_wordlist_replacements
from src.core.wordlist_store import WordlistStore


@dataclass
class PostProcessResult:
    raw_text: str
    text: str
    llm_used: bool
    replacement_hits: int
    llm_error: str | None = None


class TranscriptionPostProcessor:
    def __init__(
        self,
        wordlist_store: WordlistStore,
        llm_refiner: LlmRefiner,
    ) -> None:
        self.wordlist_store = wordlist_store
        self.llm_refiner = llm_refiner
        self.logger = logging.getLogger("sludre.pipeline")

    def process(self, raw_text: str, config: AppConfig) -> PostProcessResult:
        base_text = raw_text.strip()
        if not base_text:
            return PostProcessResult(
                raw_text="",
                text="",
                llm_used=False,
                replacement_hits=0,
            )

        working = base_text
        replacement_hits = 0
        preferred_terms: list[str] = []

        if config.wordlist_enabled:
            wordlist = self.wordlist_store.load()
            if config.wordlist_apply_replacements:
                replacement_result = apply_wordlist_replacements(
                    text=working,
                    rules=wordlist.replacements,
                )
                working = replacement_result.text
                replacement_hits = replacement_result.replacement_hits
                self.logger.info(
                    "Wordlist replacements applied. hits=%s",
                    replacement_hits,
                )
            if config.wordlist_include_in_prompt:
                preferred_terms = wordlist.preferred_terms
                self.logger.info(
                    "Wordlist preferred terms added to prompt. count=%s",
                    len(preferred_terms),
                )

        if not config.llm_enabled:
            return PostProcessResult(
                raw_text=base_text,
                text=working,
                llm_used=False,
                replacement_hits=replacement_hits,
            )

        try:
            llm_result = self.llm_refiner.refine(
                text=working,
                config=config,
                preferred_terms=preferred_terms,
            )
            return PostProcessResult(
                raw_text=base_text,
                text=llm_result.text,
                llm_used=True,
                replacement_hits=replacement_hits,
            )
        except (LlmRefineError, Exception) as exc:
            self.logger.warning("LLM refine failed: %s", exc)
            return PostProcessResult(
                raw_text=base_text,
                text=working,
                llm_used=False,
                replacement_hits=replacement_hits,
                llm_error=str(exc),
            )
