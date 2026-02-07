from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ReplacementRule:
    source: str
    target: str
    match_case: bool = False
    whole_word: bool = True


@dataclass
class WordlistData:
    replacements: list[ReplacementRule] = field(default_factory=list)
    preferred_terms: list[str] = field(default_factory=list)


class WordlistStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> WordlistData:
        if not self.path.exists():
            data = WordlistData()
            self.save(data)
            return data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return WordlistData()

        replacements: list[ReplacementRule] = []
        for item in raw.get("replacements", []):
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source:
                continue
            replacements.append(
                ReplacementRule(
                    source=source,
                    target=target,
                    match_case=bool(item.get("match_case", False)),
                    whole_word=bool(item.get("whole_word", True)),
                )
            )

        preferred_terms: list[str] = []
        for value in raw.get("preferred_terms", []):
            term = str(value).strip()
            if term:
                preferred_terms.append(term)
        return WordlistData(
            replacements=replacements,
            preferred_terms=preferred_terms,
        )

    def save(self, data: WordlistData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "replacements": [asdict(rule) for rule in data.replacements],
            "preferred_terms": data.preferred_terms,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
