from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.wordlist_store import ReplacementRule


@dataclass
class ReplacementResult:
    text: str
    replacement_hits: int


def apply_wordlist_replacements(
    text: str,
    rules: list[ReplacementRule],
) -> ReplacementResult:
    updated = text
    total_hits = 0
    for rule in rules:
        source = rule.source.strip()
        if not source:
            continue
        target = rule.target
        escaped = re.escape(source)
        pattern = rf"\b{escaped}\b" if rule.whole_word else escaped
        flags = 0 if rule.match_case else re.IGNORECASE
        updated, hits = re.subn(pattern, target, updated, flags=flags)
        total_hits += hits
    return ReplacementResult(text=updated, replacement_hits=total_hits)
