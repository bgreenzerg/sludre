from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    return runtime_root() / "assets"


def models_dir() -> Path:
    return runtime_root() / "models"


def logs_dir() -> Path:
    return runtime_root() / "logs"


def project_env_path() -> Path:
    return runtime_root() / ".env"


def wordlist_path_default() -> Path:
    return runtime_root() / "wordlist.json"


def config_path_default() -> Path:
    return runtime_root() / "config.json"
