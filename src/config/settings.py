from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    input_dir: Path = Path("images")
    output_dir: Path = Path("results")
    log_dir: Path = Path("logs")
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
    log_max_size_mb: int = 10
    ocr_language: str = "ru"
    ocr_confidence_threshold: float = 0.5
    ocr_use_gpu: bool = False
    image_scale_factor: float = 2.0
    enable_deskew: bool = True
    enable_denoising: bool = True
    processed_suffix: str = "-cor"
    template_name: str = "otk_v1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("input_dir", "output_dir", "log_dir", mode="before")
    def _ensure_path(cls, value: str | Path) -> Path:
        return value if isinstance(value, Path) else Path(value)

