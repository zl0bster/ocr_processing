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
    # OCR Engine Pool Settings
    ocr_pool_enabled: bool = True  # Enable engine pooling for batch processing
    ocr_pool_size: int = 2  # Number of engines in pool (1-4 recommended)
    ocr_pool_timeout: float = 30.0  # Max seconds to wait for available engine
    ocr_engine_memory_check_interval: int = 1  # Check memory every N files
    ocr_engine_auto_restart_threshold: float = 0.8  # Restart at 80% of reload threshold
    enable_deskew: bool = True
    # Perspective correction parameters
    enable_perspective_correction: bool = True
    perspective_min_area_ratio: float = 0.2  # Min area of contour relative to image
    perspective_corner_epsilon: float = 0.02  # Polygon approximation epsilon
    perspective_target_width_limit: int = 3000  # Max width after correction
    perspective_target_height_limit: int = 2000  # Max height after correction
    perspective_min_corner_distance: int = 50  # Min pixels between corners
    enable_denoising: bool = True
    processed_suffix: str = "-cor"
    template_name: str = "otk_v1"
    binarization_mode: Literal["otsu", "adaptive"] = "otsu"
    gaussian_blur_kernel: int = 3
    enable_illumination_correction: bool = True
    illumination_kernel: int = 31
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    use_morphological_enhancement: bool = True
    gamma_correction: float = 0.7
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
    # Table detection parameters
    enable_table_detection: bool = True
    table_detection_strategy: Literal["morphology", "template", "auto"] = "auto"
    table_h_kernel_ratio: float = 0.5  # horizontal kernel width as ratio of image width
    table_v_kernel_ratio: float = 0.5  # vertical kernel height as ratio of image height
    table_line_min_length_ratio: float = 0.7  # min line length relative to dimension
    table_line_merge_threshold: int = 10  # merge lines within this pixel distance
    # Cell-level processing
    table_cell_margin: int = 2  # pixel margin around each cell
    table_cell_preprocess: bool = True  # apply cell-level preprocessing
    table_cell_min_width: int = 20  # skip cells smaller than this
    table_cell_min_height: int = 15
    # Empty cell detection (skip OCR for blank cells)
    table_enable_empty_cell_detection: bool = True  # Enable/disable empty cell detection
    table_empty_edge_density_threshold: float = 0.01  # Max edge density for empty cells (ratio)
    table_empty_white_pixel_threshold: float = 0.95  # Min white pixel ratio for empty cells
    table_empty_variance_threshold: float = 100.0  # Max pixel variance for empty cells
    table_empty_component_threshold: int = 2  # Max connected components for empty cells
    # Template-based detection (for known forms)
    table_template_file: Path = Path("config/templates/table_templates.json")
    table_use_template_fallback: bool = True
    # Parallel processing configuration
    enable_parallel_processing: bool = True
    parallel_regions_enabled: bool = False  # Disable region parallelization (memory issues)
    parallel_regions_workers: int = 2  # Number of workers for region processing
    parallel_cells_workers: int = 4    # Number of workers for table cell processing
    parallel_use_separate_engines: bool = True  # Use det+rec engine split
    parallel_min_regions_for_parallelization: int = 2  # Min regions to trigger parallel mode
    parallel_min_cells_for_parallelization: int = 4   # Min cells to trigger parallel mode

    # === Header field extraction settings ===
    header_field_enable_specialized_extraction: bool = True  # Enable specialized extractor for blank number/date
    header_blank_area_x_min: float = 70.0  # Top-right region: 70-100% width
    header_blank_area_y_max: float = 8.0   # Top region: 0-8% height
    header_blank_min_confidence: float = 0.4  # Lower threshold for small text in cells
    header_enable_slash_correction: bool = True  # Correct I, l, | to /
    header_enable_digit_correction: bool = True  # Correct O→0, S→5, Z→2, etc.

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("input_dir", "output_dir", "log_dir", mode="before")
    def _ensure_path(cls, value: str | Path) -> Path:
        return value if isinstance(value, Path) else Path(value)

    @field_validator("region_template_file", mode="before")
    def _ensure_region_template_path(cls, value: str | Path) -> Path:
        return value if isinstance(value, Path) else Path(value)

    @field_validator("table_template_file", mode="before")
    def _ensure_table_template_path(cls, value: str | Path) -> Path:
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

    @field_validator("adaptive_block_size")
    def _ensure_adaptive_block_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("adaptive_block_size must be a positive integer")
        if value % 2 == 0:
            raise ValueError("adaptive_block_size must be odd")
        return value

    @field_validator("perspective_min_area_ratio")
    def _ensure_perspective_min_area_ratio(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("perspective_min_area_ratio must be between 0.0 and 1.0")
        return value

    @field_validator("perspective_corner_epsilon")
    def _ensure_perspective_corner_epsilon(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("perspective_corner_epsilon must be between 0.0 and 1.0")
        return value

    @field_validator("perspective_target_width_limit", "perspective_target_height_limit")
    def _ensure_perspective_target_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("perspective target size limits must be positive integers")
        return value

    @field_validator("perspective_min_corner_distance")
    def _ensure_perspective_min_corner_distance(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("perspective_min_corner_distance must be a positive integer")
        return value

