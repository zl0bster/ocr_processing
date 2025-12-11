"""E2E test fixtures for full pipeline testing with real components."""
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
def e2e_settings():
    """Settings optimized for E2E testing (CPU mode, no GPU)."""
    temp_output = Path(tempfile.mkdtemp(prefix="e2e_test_output_"))
    return Settings(
        enable_parallel_processing=True,
        ocr_use_gpu=False,  # CPU mode for CI compatibility
        enable_denoising=True,
        ocr_confidence_threshold=0.5,
        log_level="WARNING",  # Reduce log noise
        enable_perspective_correction=True,
        enable_deskew=True,
        enable_region_detection=True,
        enable_table_detection=True,
        output_dir=temp_output,
    )


@pytest.fixture(scope="session")
def e2e_logger(e2e_settings):
    """Logger configured for E2E tests."""
    logger = logging.getLogger("e2e_tests")
    logger.setLevel(getattr(logging, e2e_settings.log_level.upper(), logging.WARNING))
    logger.propagate = False
    
    # Add console handler if not present
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)
    
    return logger


@pytest.fixture(scope="session")
def shared_ocr_engine_e2e(e2e_settings, e2e_logger):
    """Shared OCR engine fixture for E2E tests.
    
    This fixture creates a single OCR engine instance that is reused across
    all E2E tests in the session. This significantly speeds up test
    execution since PaddleOCR initialization takes 2-3 seconds.
    """
    engine = OCREngine(e2e_settings, e2e_logger, engine_mode="full")
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
def batch_images_dir():
    """Fixture providing path to batch images directory."""
    project_root = Path(__file__).parent.parent.parent
    batch_dir = project_root / "images" / "batch1"
    
    if not batch_dir.exists() or not batch_dir.is_dir():
        pytest.skip(f"Batch images directory not found: {batch_dir}")
    
    # Check if directory has image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in batch_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        pytest.skip(f"No image files found in batch directory: {batch_dir}")
    
    return batch_dir


@pytest.fixture
def cleanup_e2e_outputs(e2e_settings):
    """Auto-cleanup fixture for E2E output files."""
    output_dir = e2e_settings.output_dir
    
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
def temp_output_dir_e2e():
    """Create temporary directory for E2E test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="e2e_test_output_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)







