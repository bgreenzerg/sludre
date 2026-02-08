from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.core.runtime_paths import logs_dir


def default_log_file() -> Path:
    return logs_dir() / "sludre.log"


def configure_logging(log_file: Path | None = None) -> Path:
    target = log_file or default_log_file()
    target.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == target
        for handler in root_logger.handlers
    ):
        return target

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        filename=target,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return target
