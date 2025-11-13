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
    log_ocr_verbose: bool = False  # Enable verbose OCR debugging
    save_ocr_debug_files: bool = False  # Save intermediate OCR files for debugging
    ocr_language: str = "ru"
    ocr_confidence_threshold: float = 0.5
    ocr_use_gpu: bool = False
    enable_adaptive_scaling: bool = True
    adaptive_scaling_min_mp: float = 4.0
    adaptive_scaling_target_mp: float = 5.0  # Снижено с 6.0 для стабилизации OCR
    adaptive_scaling_max_width: int = 3000   # Снижено с 3500 (рекомендация PaddleOCR)
    adaptive_scaling_max_height: int = 2000  # Снижено с 2500 для соотношения сторон
    # OCR safety limits
    ocr_max_image_dimension: int = 3000  # Max dimension (width or height) for OCR processing
    ocr_det_limit_side_len: int = 2000   # PaddleOCR detection limit
    ocr_memory_reload_threshold_mb: int = 700  # Reload OCR engine if memory exceeds this (MB)
    enable_deskew: bool = True
    enable_denoising: bool = True
    processed_suffix: str = "-cor"
    template_name: str = "otk_v1"
    binarization_mode: Literal["otsu", "adaptive"] = "otsu"
    gaussian_blur_kernel: int = 3
    enable_illumination_correction: bool = True
    illumination_kernel: int = 31
    enable_region_detection: bool = True
    region_detection_strategy: Literal["auto", "template", "adaptive", "text_based"] = "auto"
    region_min_confidence: float = 0.7
    region_max_width: int = 3000
    region_max_height: int = 2000
    region_horizontal_line_min_width_ratio: float = 0.7
    region_template_file: Path = Path("config/templates/regions.json")
    # Region detection constraints (all values are ratios 0.0-1.0)
    region_min_header_ratio: float = 0.20  # Header: min 20% of image height
    region_max_header_ratio: float = 0.25  # Header: max 25% of image height
    region_min_defects_ratio: float = 0.25  # Defects: min 25% of image height
    region_max_defects_ratio: float = 0.60  # Defects: max 60% of image height
    region_adaptive_template_tolerance: float = 0.10  # Max 10% deviation from template

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("input_dir", "output_dir", "log_dir", mode="before")
    def _ensure_path(cls, value: str | Path) -> Path:
        return value if isinstance(value, Path) else Path(value)

    @field_validator("region_template_file", mode="before")
    def _ensure_region_template_path(cls, value: str | Path) -> Path:
        return value if isinstance(value, Path) else Path(value)

    @field_validator("gaussian_blur_kernel")
    def _ensure_kernel_is_positive_odd(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("gaussian_blur_kernel must be a positive integer")
        if value % 2 == 0:
            raise ValueError("gaussian_blur_kernel must be odd")
        return value

    @field_validator("illumination_kernel")
    def _ensure_illumination_kernel(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("illumination_kernel must be a positive integer")
        if value % 2 == 0:
            raise ValueError("illumination_kernel must be odd")
        if value < 15:
            raise ValueError("illumination_kernel should be sufficiently large (>= 15)")
        return value

