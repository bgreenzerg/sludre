from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from src.core.app_logging import configure_logging
from src.ui.main_window import MainWindow


def main() -> int:
    log_file = configure_logging()
    logger = logging.getLogger("sludre.app")
    logger.info("Application startup initiated. Log file: %s", log_file)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    logger.info("Application exiting with code %s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
