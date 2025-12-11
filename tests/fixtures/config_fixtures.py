"""Configuration fixtures for testing."""
import pytest
from src.config.settings import Settings


@pytest.fixture
def fast_test_settings():
    """Settings optimized for fastest test execution.
    
    Disables expensive operations like denoising, GPU, parallel processing.
    """
    return Settings(
        enable_parallel_processing=False,
        ocr_use_gpu=False,
        enable_denoising=False,
        enable_perspective_correction=False,
        enable_deskew=False,
        log_level="ERROR",  # Minimal logging
        ocr_confidence_threshold=0.5,
    )


@pytest.fixture
def minimal_settings():
    """Minimal settings with only essential features enabled."""
    return Settings(
        enable_region_detection=False,
        enable_table_detection=False,
        enable_perspective_correction=False,
        enable_deskew=False,
        enable_denoising=False,
        enable_parallel_processing=False,
        ocr_use_gpu=False,
        log_level="ERROR",
    )



