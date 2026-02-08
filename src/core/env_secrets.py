from __future__ import annotations

import re
from pathlib import Path

from src.core.runtime_paths import project_env_path


_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class EnvSecretsStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def default(cls) -> "EnvSecretsStore":
        return cls(project_env_path())

    def ensure_exists(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            return
        self.path.write_text(
            "# Sludre local secrets\n",
            encoding="utf-8",
        )

    def get_secret(self, key: str) -> str:
        values, _ = self._read_values()
        return values.get(key, "")

    def set_secret(self, key: str, value: str) -> None:
        values, order = self._read_values()
        clean = value.strip()
        if clean:
            if key not in values:
                order.append(key)
            values[key] = clean
        else:
            values.pop(key, None)
            order = [existing for existing in order if existing != key]
        self._write_values(values, order)

    def _read_values(self) -> tuple[dict[str, str], list[str]]:
        if not self.path.exists():
            return {}, []
        values: dict[str, str] = {}
        order: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            match = _KEY_PATTERN.match(line)
            if not match:
                continue
            key = match.group(1)
            raw_value = match.group(2)
            if key not in values:
                order.append(key)
            values[key] = self._decode_value(raw_value)
        return values, order

    def _write_values(self, values: dict[str, str], order: list[str]) -> None:
        self.ensure_exists()
        lines = ["# Sludre local secrets\n"]
        for key in order:
            if key in values:
                lines.append(f"{key}={self._encode_value(values[key])}\n")
        for key in values:
            if key not in order:
                lines.append(f"{key}={self._encode_value(values[key])}\n")
        self.path.write_text("".join(lines), encoding="utf-8")

    @staticmethod
    def _decode_value(raw: str) -> str:
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            inner = raw[1:-1]
            if raw[0] == '"':
                inner = inner.replace('\\"', '"').replace("\\\\", "\\")
            return inner
        return raw

    @staticmethod
    def _encode_value(value: str) -> str:
        if not value:
            return '""'
        if any(ch.isspace() for ch in value) or any(ch in value for ch in {'"', "#", "="}):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
