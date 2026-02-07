from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


APP_DIR_NAME = "HviskeSTT"
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You clean up Danish speech-to-text output. "
    "Return only the cleaned final text without explanations."
)
DEFAULT_PROMPT_PRESET_NAME = "Standard"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MODEL_PRESETS = [
    "mistral-small-latest",
    "mistral-large-latest",
    "codestral-latest",
]


def _default_app_dir() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / APP_DIR_NAME
    return Path.home() / f".{APP_DIR_NAME.lower()}"


def _project_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_default_model_cache_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIR_NAME / "models"
    return _default_app_dir() / "models"


def _default_model_cache_dir() -> Path:
    return _project_root_dir() / "models"


def _default_wordlist_path() -> Path:
    return _project_root_dir() / "wordlist.json"


def _same_path(a: str | Path, b: str | Path) -> bool:
    left = os.path.normcase(os.path.abspath(str(a)))
    right = os.path.normcase(os.path.abspath(str(b)))
    return left == right


@dataclass
class AppConfig:
    hotkey: str = "ctrl+space"
    language: str = "da"
    sample_rate: int = 16000
    channels: int = 1
    max_record_seconds: int = 60
    silence_trim: bool = True
    model_repo_id: str = "syvai/hviske-v2"
    model_cache_dir: str = ""
    manual_model_path: str = ""
    hf_token: str = ""
    llm_enabled: bool = False
    llm_provider: str = "openai_compatible"
    llm_base_url: str = ""
    mistral_base_url: str = DEFAULT_MISTRAL_BASE_URL
    llm_api_key: str = ""
    llm_model: str = ""
    mistral_model_preset: str = "mistral-small-latest"
    llm_timeout_seconds: int = 5
    llm_system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT
    llm_prompt_presets: list[dict[str, str]] = field(default_factory=list)
    llm_selected_prompt_name: str = DEFAULT_PROMPT_PRESET_NAME
    llm_temperature: float = 0.0
    wordlist_enabled: bool = True
    wordlist_apply_replacements: bool = True
    wordlist_include_in_prompt: bool = True
    wordlist_path: str = ""
    insert_mode: str = "clipboard_paste"
    restore_clipboard: bool = True

    @classmethod
    def defaults(cls) -> "AppConfig":
        cfg = cls()
        cfg.model_cache_dir = str(_default_model_cache_dir())
        cfg.wordlist_path = str(_default_wordlist_path())
        cfg.llm_prompt_presets = [
            {
                "name": DEFAULT_PROMPT_PRESET_NAME,
                "prompt": DEFAULT_LLM_SYSTEM_PROMPT,
            }
        ]
        cfg.llm_selected_prompt_name = DEFAULT_PROMPT_PRESET_NAME
        return cfg


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def default(cls) -> "ConfigStore":
        return cls(_default_app_dir() / "config.json")

    def load(self) -> AppConfig:
        if not self.path.exists():
            cfg = AppConfig.defaults()
            self.save(cfg, persist_secrets=True)
            return cfg

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        cfg = AppConfig.defaults()
        should_save = False
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        if not cfg.model_cache_dir:
            cfg.model_cache_dir = str(_default_model_cache_dir())
            should_save = True
        elif _same_path(cfg.model_cache_dir, _legacy_default_model_cache_dir()):
            cfg.model_cache_dir = str(_default_model_cache_dir())
            should_save = True
        if not cfg.wordlist_path:
            cfg.wordlist_path = str(_default_wordlist_path())
            should_save = True
        if cfg.llm_timeout_seconds < 1 or cfg.llm_timeout_seconds > 60:
            cfg.llm_timeout_seconds = 5
            should_save = True
        if not isinstance(cfg.llm_prompt_presets, list):
            cfg.llm_prompt_presets = AppConfig.defaults().llm_prompt_presets
            should_save = True
        sanitized_presets: list[dict[str, str]] = []
        for preset in cfg.llm_prompt_presets:
            if not isinstance(preset, dict):
                continue
            name = str(preset.get("name", "")).strip()
            prompt = str(preset.get("prompt", "")).strip()
            if not name or not prompt:
                continue
            sanitized_presets.append({"name": name, "prompt": prompt})
        if not sanitized_presets:
            sanitized_presets = AppConfig.defaults().llm_prompt_presets
            should_save = True
        if sanitized_presets != cfg.llm_prompt_presets:
            should_save = True
        cfg.llm_prompt_presets = sanitized_presets
        selected_names = {preset["name"] for preset in cfg.llm_prompt_presets}
        if cfg.llm_selected_prompt_name not in selected_names:
            cfg.llm_selected_prompt_name = cfg.llm_prompt_presets[0]["name"]
            should_save = True
        selected_prompt_text = ""
        for preset in cfg.llm_prompt_presets:
            if preset["name"] == cfg.llm_selected_prompt_name:
                selected_prompt_text = preset["prompt"]
                break
        if selected_prompt_text and cfg.llm_system_prompt != selected_prompt_text:
            cfg.llm_system_prompt = selected_prompt_text
            should_save = True
        if should_save:
            self.save(cfg, persist_secrets=True)
        return cfg

    def save(self, config: AppConfig, persist_secrets: bool = False) -> None:
        payload = asdict(config)
        if not persist_secrets:
            payload["hf_token"] = ""
            payload["llm_api_key"] = ""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
