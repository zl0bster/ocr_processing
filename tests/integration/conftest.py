"""Integration test fixtures for real PaddleOCR and component interactions."""
import sys
from pathlib import Path
import logging
import tempfile
import shutil
import pytest

# Add src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.config.settings import Settings
from src.ocr_engine import OCREngine


@pytest.fixture(scope="session")
def integration_settings():
    """Settings optimized for integration testing (CPU mode, no GPU)."""
    return Settings(
        enable_parallel_processing=True,  # Test parallel processing
        ocr_use_gpu=False,  # CPU mode for CI compatibility
        enable_denoising=True,
        ocr_confidence_threshold=0.5,
        log_level="WARNING",  # Reduce log noise
        enable_perspective_correction=True,
        enable_deskew=True,
        enable_region_detection=True,
        enable_table_detection=True,
        output_dir=Path(tempfile.mkdtemp(prefix="integration_test_output_")),
    )


@pytest.fixture(scope="session")
def integration_logger(integration_settings):
    """Logger configured for integration tests."""
    logger = logging.getLogger("integration_tests")
    logger.setLevel(getattr(logging, integration_settings.log_level.upper(), logging.WARNING))
    logger.propagate = False
    
    # Add console handler if not present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)
    
    return logger


@pytest.fixture(scope="session")
def shared_ocr_engine(integration_settings, integration_logger):
    """Shared OCR engine fixture to avoid multiple expensive initializations.
    
    This fixture creates a single OCR engine instance that is reused across
    all integration tests in the session. This significantly speeds up test
    execution since PaddleOCR initialization takes 2-3 seconds.
    """
    engine = OCREngine(integration_settings, integration_logger, engine_mode="full")
    yield engine
    # Cleanup
    engine.close()


@pytest.fixture
def test_image_paths():
    """Fixture providing paths to test images."""
    project_root = Path(__file__).parent.parent.parent
    test_images_dir = project_root / "images" / "test_images"
    
    compressed_path = test_images_dir / "034_compr.jpg"
    full_resolution_path = test_images_dir / "034.jpg"
    
    paths = {}
    if compressed_path.exists():
        paths["compressed"] = compressed_path
    if full_resolution_path.exists():
        paths["full"] = full_resolution_path
    
    if not paths:
        pytest.skip("No test images found in images/test_images/")
    
    return paths


@pytest.fixture
def cleanup_integration_outputs(integration_settings):
    """Auto-cleanup fixture for output files."""
    output_dir = integration_settings.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yield output_dir
    
    # Cleanup after test
    if output_dir.exists() and output_dir.is_dir():
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass  # Ignore cleanup errors in tests


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_output_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

