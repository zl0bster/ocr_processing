from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import Settings

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="OCR processing pipeline CLI. Future iterations will extend functionality.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the image file for processing (pipeline coming in later iterations).",
    )
    return parser


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure logger with console and rotating file handlers."""
    logger = logging.getLogger("image_ocr")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = log_dir / log_filename

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.log_max_size_mb * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logger.level)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug("Logging initialized. Log file: %s", log_path)
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    logger = setup_logging(settings)

    if args.file:
        logger.info(
            "Received file '%s' for processing. The processing pipeline will be implemented in upcoming iterations.",
            args.file,
        )
    else:
        logger.info("No input file provided. Use --file to specify an image for processing.")

    logger.info("Initialization complete. Awaiting further implementation in next iterations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

