"""Global fixtures for all tests."""
import sys
from pathlib import Path

# Add src directory to Python path for source file imports
# Source files use relative imports like "from config.settings"
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock

from src.config.settings import Settings


@pytest.fixture
def test_settings():
    """Settings optimized for fast testing."""
    return Settings(
        enable_parallel_processing=False,
        ocr_use_gpu=False,
        enable_denoising=False,
        ocr_confidence_threshold=0.5,
        log_level="WARNING",  # Reduce log noise
        enable_perspective_correction=True,
        enable_deskew=True,
        enable_region_detection=True,
        enable_table_detection=True,
    )


@pytest.fixture
def test_image_034():
    """Load actual test image 034_compr.jpg."""
    path = Path("images/test_images/034_compr.jpg")
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        pytest.skip(f"Failed to load test image: {path}")
    return image


@pytest.fixture
def test_image_034_full():
    """Load full resolution test image 034.jpg."""
    path = Path("images/test_images/034.jpg")
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        pytest.skip(f"Failed to load test image: {path}")
    return image


@pytest.fixture
def synthetic_skewed_image():
    """Generate synthetic skewed document for testing."""
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    # Add horizontal lines (simulating text)
    cv2.line(image, (50, 200), (750, 250), (0, 0, 0), 2)
    cv2.line(image, (50, 400), (750, 450), (0, 0, 0), 2)
    cv2.line(image, (50, 600), (750, 650), (0, 0, 0), 2)
    # Rotate to create skew (5 degrees)
    center = (400, 500)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(image, rotation_matrix, (800, 1000))
    return skewed


@pytest.fixture
def mock_ocr_response():
    """Mock PaddleOCR response structure."""
    return [
        [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('ГЕОМЕТРИЯ', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('ОТВЕРСТИЯ', 0.92)),
            ([[0, 120], [100, 120], [100, 170], [0, 170]], ('ПОВЕРХНОСТЬ', 0.88)),
        ]
    ]


@pytest.fixture
def mock_ocr_engine(mock_ocr_response):
    """Mock OCR engine for unit tests."""
    mock_engine = MagicMock()
    mock_engine.ocr.return_value = mock_ocr_response
    return mock_engine

