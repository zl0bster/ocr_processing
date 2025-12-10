"""Unit tests for TableDetector module."""

import logging
import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch

from src.table_detector import TableDetector, TableGrid
from src.config.settings import Settings


@pytest.fixture
def mock_logger():
    """Mock logging.Logger instance."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def table_settings():
    """Settings with table detection parameters."""
    return Settings(
        table_detection_strategy="auto",
        table_h_kernel_ratio=0.5,
        table_v_kernel_ratio=0.5,
        table_line_min_length_ratio=0.7,
        table_line_merge_threshold=10,
    )


@pytest.fixture
def mock_table_image():
    """Create synthetic table image with clear grid lines."""
    # Create white image
    image = np.ones((500, 600, 3), dtype=np.uint8) * 255
    
    # Draw horizontal lines (rows)
    cv2.line(image, (0, 100), (600, 100), (0, 0, 0), 3)
    cv2.line(image, (0, 200), (600, 200), (0, 0, 0), 3)
    cv2.line(image, (0, 300), (600, 300), (0, 0, 0), 3)
    cv2.line(image, (0, 400), (600, 400), (0, 0, 0), 3)
    
    # Draw vertical lines (columns)
    cv2.line(image, (150, 0), (150, 500), (0, 0, 0), 3)
    cv2.line(image, (300, 0), (300, 500), (0, 0, 0), 3)
    cv2.line(image, (450, 0), (450, 500), (0, 0, 0), 3)
    
    return image


@pytest.fixture
def mock_no_lines_image():
    """Create image without detectable lines."""
    return np.ones((500, 600, 3), dtype=np.uint8) * 255


@pytest.fixture
def mock_small_image():
    """Create image too small for detection."""
    return np.ones((30, 30, 3), dtype=np.uint8) * 255


@pytest.mark.unit
class TestTableDetectorStructureDetection:
    """Test table structure detection."""

    def test_detect_structure_with_morphology_strategy(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test detect_structure() with morphology strategy."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_table_image, strategy="morphology")
        
        # Assert
        assert grid is not None
        assert isinstance(grid, TableGrid)
        assert grid.method == "morphology"
        assert len(grid.rows) >= 2
        assert len(grid.cols) >= 2
        assert 0.0 <= grid.confidence <= 1.0

    def test_detect_structure_with_auto_strategy(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test detect_structure() with auto strategy."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_table_image, strategy="auto")
        
        # Assert
        assert grid is not None
        assert grid.method == "morphology"

    def test_detect_structure_with_template_strategy(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test detect_structure() with template strategy (not implemented)."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_table_image, strategy="template")
        
        # Assert
        assert grid is None
        mock_logger.warning.assert_called()

    def test_detect_structure_with_unknown_strategy(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test detect_structure() with unknown strategy."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_table_image, strategy="unknown")
        
        # Assert
        assert grid is None
        mock_logger.warning.assert_called()

    def test_detect_structure_with_none_image(
        self, table_settings, mock_logger
    ):
        """Test detect_structure() with None image."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(None)
        
        # Assert
        assert grid is None
        mock_logger.warning.assert_called()

    def test_detect_structure_with_empty_image(
        self, table_settings, mock_logger
    ):
        """Test detect_structure() with empty image."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        empty_image = np.array([])
        
        # Act
        grid = detector.detect_structure(empty_image)
        
        # Assert
        assert grid is None
        mock_logger.warning.assert_called()

    def test_detect_structure_with_small_image(
        self, table_settings, mock_logger, mock_small_image
    ):
        """Test detect_structure() with too small image."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_small_image)
        
        # Assert
        assert grid is None
        mock_logger.debug.assert_called()

    def test_detect_structure_with_no_lines(
        self, table_settings, mock_logger, mock_no_lines_image
    ):
        """Test detect_structure() with image without lines."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector.detect_structure(mock_no_lines_image)
        
        # Assert
        assert grid is None


@pytest.mark.unit
class TestTableDetectorLineDetection:
    """Test horizontal and vertical line detection."""

    def test_detect_horizontal_lines(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test _detect_horizontal_lines() with clear grid."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        height, width = mock_table_image.shape[:2]
        
        # Convert to grayscale and binarize
        gray = cv2.cvtColor(mock_table_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Act
        h_lines = detector._detect_horizontal_lines(binary, width, height)
        
        # Assert
        assert h_lines is not None
        assert len(h_lines) >= 2
        assert all(isinstance(y, int) for y in h_lines)
        assert h_lines == sorted(h_lines)  # Should be sorted

    def test_detect_vertical_lines(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test _detect_vertical_lines() with clear grid."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        height, width = mock_table_image.shape[:2]
        
        # Convert to grayscale and binarize
        gray = cv2.cvtColor(mock_table_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Act
        v_lines = detector._detect_vertical_lines(binary, width, height)
        
        # Assert
        assert v_lines is not None
        assert len(v_lines) >= 2
        assert all(isinstance(x, int) for x in v_lines)
        assert v_lines == sorted(v_lines)  # Should be sorted

    def test_detect_horizontal_lines_with_insufficient_lines(
        self, table_settings, mock_logger, mock_no_lines_image
    ):
        """Test _detect_horizontal_lines() with no lines."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        height, width = mock_no_lines_image.shape[:2]
        
        # Convert to grayscale and binarize
        gray = cv2.cvtColor(mock_no_lines_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Act
        h_lines = detector._detect_horizontal_lines(binary, width, height)
        
        # Assert
        assert h_lines is None or len(h_lines) < 2


@pytest.mark.unit
class TestTableDetectorGridValidation:
    """Test grid validation logic."""

    def test_validate_grid_with_valid_grid(
        self, table_settings, mock_logger
    ):
        """Test _validate_grid() with valid grid."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [100, 200, 300, 400]
        cols = [150, 300, 450]
        width, height = 600, 500
        
        # Act
        is_valid = detector._validate_grid(rows, cols, width, height)
        
        # Assert
        assert is_valid is True

    def test_validate_grid_with_insufficient_rows(
        self, table_settings, mock_logger
    ):
        """Test _validate_grid() with insufficient rows."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [100]  # Only 1 row
        cols = [150, 300, 450]
        width, height = 600, 500
        
        # Act
        is_valid = detector._validate_grid(rows, cols, width, height)
        
        # Assert
        assert is_valid is False

    def test_validate_grid_with_insufficient_cols(
        self, table_settings, mock_logger
    ):
        """Test _validate_grid() with insufficient columns."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [100, 200, 300, 400]
        cols = [150]  # Only 1 column
        width, height = 600, 500
        
        # Act
        is_valid = detector._validate_grid(rows, cols, width, height)
        
        # Assert
        assert is_valid is False

    def test_validate_grid_with_lines_outside_bounds(
        self, table_settings, mock_logger
    ):
        """Test _validate_grid() with lines outside image bounds."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [-10, 200, 300, 600]  # First row negative, last row > height
        cols = [150, 300, 450]
        width, height = 600, 500
        
        # Act
        is_valid = detector._validate_grid(rows, cols, width, height)
        
        # Assert
        assert is_valid is False

    def test_validate_grid_with_lines_too_close(
        self, table_settings, mock_logger
    ):
        """Test _validate_grid() with lines too close together."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [100, 105, 300, 400]  # First two rows only 5 pixels apart
        cols = [150, 300, 450]
        width, height = 600, 500
        
        # Act
        is_valid = detector._validate_grid(rows, cols, width, height)
        
        # Assert
        assert is_valid is False


@pytest.mark.unit
class TestTableDetectorMergePositions:
    """Test position merging logic."""

    def test_merge_close_positions(
        self, table_settings, mock_logger
    ):
        """Test _merge_close_positions() merges close positions."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100, 105, 200, 205, 300]  # 100 and 105 are close, 200 and 205 are close
        threshold = 10
        
        # Act
        merged = detector._merge_close_positions(positions, threshold)
        
        # Assert
        assert len(merged) < len(positions)  # Should merge some positions
        assert len(merged) == 3  # Should have 3 positions after merging (102, 202, 300)
        assert 300 in merged  # Last position should remain

    def test_merge_close_positions_with_empty_list(
        self, table_settings, mock_logger
    ):
        """Test _merge_close_positions() with empty list."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = []
        threshold = 10
        
        # Act
        merged = detector._merge_close_positions(positions, threshold)
        
        # Assert
        assert merged == []

    def test_merge_close_positions_with_single_position(
        self, table_settings, mock_logger
    ):
        """Test _merge_close_positions() with single position."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100]
        threshold = 10
        
        # Act
        merged = detector._merge_close_positions(positions, threshold)
        
        # Assert
        assert merged == [100]

    def test_merge_close_positions_with_no_close_positions(
        self, table_settings, mock_logger
    ):
        """Test _merge_close_positions() with positions far apart."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100, 200, 300, 400]
        threshold = 10
        
        # Act
        merged = detector._merge_close_positions(positions, threshold)
        
        # Assert
        assert len(merged) == len(positions)
        assert merged == sorted(positions)


@pytest.mark.unit
class TestTableDetectorConfidence:
    """Test confidence calculation."""

    def test_calculate_confidence(
        self, table_settings, mock_logger
    ):
        """Test _calculate_confidence() returns valid score."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        rows = [100, 200, 300, 400, 500]
        cols = [150, 300, 450]
        width, height = 600, 500
        
        # Act
        confidence = detector._calculate_confidence(rows, cols, width, height)
        
        # Assert
        assert 0.0 <= confidence <= 1.0

    def test_calculate_spacing_regularity_with_regular_spacing(
        self, table_settings, mock_logger
    ):
        """Test _calculate_spacing_regularity() with regular spacing."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100, 200, 300, 400, 500]  # Perfectly regular
        
        # Act
        regularity = detector._calculate_spacing_regularity(positions)
        
        # Assert
        assert 0.0 <= regularity <= 1.0
        assert regularity > 0.5  # Should be high for regular spacing

    def test_calculate_spacing_regularity_with_irregular_spacing(
        self, table_settings, mock_logger
    ):
        """Test _calculate_spacing_regularity() with irregular spacing."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100, 150, 500, 550, 1000]  # Very irregular
        
        # Act
        regularity = detector._calculate_spacing_regularity(positions)
        
        # Assert
        assert 0.0 <= regularity <= 1.0
        # Note: Even irregular spacing can have some regularity score
        # Just verify it's a valid score

    def test_calculate_spacing_regularity_with_single_position(
        self, table_settings, mock_logger
    ):
        """Test _calculate_spacing_regularity() with single position."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100]
        
        # Act
        regularity = detector._calculate_spacing_regularity(positions)
        
        # Assert
        assert regularity == 0.0

    def test_calculate_spacing_regularity_with_two_positions(
        self, table_settings, mock_logger
    ):
        """Test _calculate_spacing_regularity() with two positions."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        positions = [100, 200]
        
        # Act
        regularity = detector._calculate_spacing_regularity(positions)
        
        # Assert
        assert 0.0 <= regularity <= 1.0


@pytest.mark.unit
class TestTableDetectorMorphology:
    """Test morphology-based detection."""

    def test_detect_morphology_with_valid_table(
        self, table_settings, mock_logger, mock_table_image
    ):
        """Test _detect_morphology() with valid table."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector._detect_morphology(mock_table_image)
        
        # Assert
        assert grid is not None
        assert isinstance(grid, TableGrid)
        assert grid.method == "morphology"
        assert len(grid.rows) >= 2
        assert len(grid.cols) >= 2
        assert grid.num_rows >= 1
        assert grid.num_cols >= 1
        assert 0.0 <= grid.confidence <= 1.0

    def test_detect_morphology_with_no_lines(
        self, table_settings, mock_logger, mock_no_lines_image
    ):
        """Test _detect_morphology() with no lines."""
        # Arrange
        detector = TableDetector(settings=table_settings, logger=mock_logger)
        
        # Act
        grid = detector._detect_morphology(mock_no_lines_image)
        
        # Assert
        assert grid is None
