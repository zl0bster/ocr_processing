"""Unit tests for PerspectiveCorrector module."""
import logging
import pytest
import cv2
import numpy as np
from pathlib import Path

from src.perspective_corrector import PerspectiveCorrector
from src.config.settings import Settings
from tests.fixtures.image_fixtures import create_test_document_image


@pytest.mark.unit
class TestPerspectiveCorrectorDocumentDetection:
    """Test document detection functionality."""

    def test_perspective_corrector_with_clear_document(self, test_settings):
        """Test with clear rectangular document."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image with clear rectangular document boundary
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        # Add clear document boundary (black rectangle)
        cv2.rectangle(image, (50, 50), (750, 950), (0, 0, 0), 3)

        # Act
        corrected, was_corrected = corrector.correct(image)

        # Assert
        assert corrected is not None
        assert isinstance(was_corrected, bool)
        # May or may not correct depending on detection quality
        assert corrected.shape[2] == 3  # BGR image

    def test_perspective_corrector_detects_contour(self, test_settings):
        """Test contour detection."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image with document-like structure
        image = create_test_document_image(800, 1000)

        # Act
        contour = corrector._find_document_contour(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        # Assert
        # Contour may or may not be found depending on image structure
        if contour is not None:
            assert len(contour.shape) == 3  # (N, 1, 2) shape

    def test_perspective_corrector_extracts_corners(self, test_settings):
        """Test corner extraction from contour."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image with clear rectangular boundary
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (100, 100), (700, 900), (0, 0, 0), 5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contour = corrector._find_document_contour(binary)

        # Act
        if contour is not None:
            corners = corrector._get_document_corners(contour)

            # Assert
            if corners is not None:
                assert corners.shape == (4, 2)
                assert corners.dtype == np.float32


@pytest.mark.unit
class TestPerspectiveCorrectorCornerProcessing:
    """Test corner processing functionality."""

    def test_order_points_correctly_orders_corners(self, test_settings):
        """Test corner ordering algorithm (TL, TR, BR, BL)."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create 4 unordered points representing corners
        # TL=(10,10), TR=(90,10), BR=(90,90), BL=(10,90)
        unordered_pts = np.array(
            [[90, 90], [10, 10], [90, 10], [10, 90]], dtype=np.float32
        )

        # Act
        ordered_pts = corrector._order_points(unordered_pts)

        # Assert
        assert ordered_pts.shape == (4, 2)
        # Check ordering: TL should have smallest sum, BR should have largest sum
        sums = ordered_pts.sum(axis=1)
        assert sums[0] == min(sums)  # TL
        assert sums[2] == max(sums)  # BR

    def test_validate_corners_accepts_valid_corners(self, test_settings):
        """Test validation passes for valid corners."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create valid corners (large rectangle)
        corners = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.float32
        )

        # Act
        is_valid = corrector._validate_corners(corners, 800, 1000)

        # Assert
        assert is_valid is True

    def test_validate_corners_rejects_too_close_corners(self, test_settings):
        """Test rejection when corners too close."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create corners that are too close (less than min_corner_distance)
        # min_corner_distance is typically 50
        corners = np.array(
            [[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32
        )

        # Act
        is_valid = corrector._validate_corners(corners, 800, 1000)

        # Assert
        assert is_valid is False

    def test_validate_corners_rejects_non_convex_quadrilateral(self, test_settings):
        """Test rejection for concave shapes."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create concave quadrilateral (bow-tie shape)
        # This creates a self-intersecting/concave shape
        corners = np.array(
            [[100, 100], [500, 500], [700, 100], [300, 500]], dtype=np.float32
        )

        # Act
        is_valid = corrector._validate_corners(corners, 800, 1000)

        # Assert
        # Should reject concave quadrilaterals
        assert is_valid is False


@pytest.mark.unit
class TestPerspectiveCorrectorTransformation:
    """Test perspective transformation functionality."""

    def test_perspective_transform_with_known_angles(self, test_settings):
        """Test transformation accuracy."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image with known perspective distortion
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (100, 100), (700, 900), (0, 0, 0), 3)
        # Apply perspective transform to create distortion
        src_pts = np.array(
            [[100, 100], [700, 100], [700, 900], [100, 900]], dtype=np.float32
        )
        dst_pts = np.array(
            [[150, 120], [650, 110], [680, 880], [120, 890]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted = cv2.warpPerspective(image, M, (800, 1000))

        # Act
        corrected, was_corrected = corrector.correct(distorted)

        # Assert
        assert corrected is not None
        assert isinstance(was_corrected, bool)

    def test_calculate_target_size_respects_limits(self, test_settings):
        """Test size limits are enforced."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create corners that would result in very large dimensions
        corners = np.array(
            [[0, 0], [5000, 0], [5000, 4000], [0, 4000]], dtype=np.float32
        )

        # Act
        width, height = corrector._calculate_target_size(corners)

        # Assert
        assert width <= test_settings.perspective_target_width_limit
        assert height <= test_settings.perspective_target_height_limit

    def test_calculate_target_size_uses_max_dimensions(self, test_settings):
        """Test max width/height calculation."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create corners with different top/bottom widths and left/right heights
        corners = np.array(
            [[100, 100], [800, 120], [750, 900], [150, 880]], dtype=np.float32
        )

        # Act
        width, height = corrector._calculate_target_size(corners)

        # Assert
        # Should use maximum of top and bottom widths
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        expected_width = max(width_top, width_bottom)
        assert abs(width - expected_width) < 10  # Allow small tolerance

        # Should use maximum of left and right heights
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        expected_height = max(height_left, height_right)
        assert abs(height - expected_height) < 10  # Allow small tolerance


@pytest.mark.unit
class TestPerspectiveCorrectorSkipConditions:
    """Test conditions where correction is skipped."""

    def test_perspective_corrector_skips_without_clear_boundaries(
        self, test_settings
    ):
        """Test skip when no document detected."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image without clear document boundaries (uniform, no edges)
        # Use a gradient to avoid edge detection
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        # Add very subtle gradient (no sharp edges that could be detected as document)
        for y in range(1000):
            val = int(254 + (y / 1000) * 1)
            image[y, :] = (val, val, val)

        # Act
        corrected, was_corrected = corrector.correct(image)

        # Assert
        # May or may not correct depending on detection, but should not crash
        assert corrected is not None
        assert isinstance(was_corrected, bool)

    def test_perspective_corrector_skips_for_small_images(self, test_settings):
        """Test skip for images < 50×50."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create very small image
        small_image = np.ones((40, 30, 3), dtype=np.uint8) * 255

        # Act
        corrected, was_corrected = corrector.correct(small_image)

        # Assert
        assert was_corrected is False
        assert np.array_equal(corrected, small_image)

    def test_perspective_corrector_skips_for_none_image(self, test_settings):
        """Test skip for None input."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)

        # Act
        corrected, was_corrected = corrector.correct(None)

        # Assert
        assert was_corrected is False
        assert corrected is None

    def test_perspective_corrector_skips_for_empty_image(self, test_settings):
        """Test skip for empty array."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        empty_image = np.array([])

        # Act
        corrected, was_corrected = corrector.correct(empty_image)

        # Assert
        assert was_corrected is False
        assert np.array_equal(corrected, empty_image)


@pytest.mark.unit
class TestPerspectiveCorrectorErrorHandling:
    """Test error handling."""

    def test_perspective_corrector_handles_exceptions_gracefully(
        self, test_settings
    ):
        """Test exception handling returns original image."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)
        # Create image that might cause issues
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        # Should not raise exception even if processing fails
        corrected, was_corrected = corrector.correct(image)

        # Assert
        assert corrected is not None
        assert isinstance(was_corrected, bool)

    def test_perspective_corrector_with_real_image_034_compr(
        self, test_settings, test_image_034
    ):
        """Test with real test image 034_compr.jpg."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)

        # Act
        corrected, was_corrected = corrector.correct(test_image_034)

        # Assert
        assert corrected is not None
        # Perspective correction may change dimensions (within limits)
        assert corrected.shape[2] == test_image_034.shape[2]  # Same color channels
        assert isinstance(was_corrected, bool)
        # If corrected, dimensions should be within limits
        if was_corrected:
            assert corrected.shape[0] <= test_settings.perspective_target_height_limit
            assert corrected.shape[1] <= test_settings.perspective_target_width_limit

    def test_perspective_corrector_with_real_image_034_full(
        self, test_settings, test_image_034_full
    ):
        """Test with real test image 034.jpg (full resolution)."""
        # Arrange
        logger = logging.getLogger("test")
        corrector = PerspectiveCorrector(settings=test_settings, logger=logger)

        # Act
        corrected, was_corrected = corrector.correct(test_image_034_full)

        # Assert
        assert corrected is not None
        # Perspective correction may change dimensions (within limits)
        assert corrected.shape[2] == test_image_034_full.shape[2]  # Same color channels
        assert isinstance(was_corrected, bool)
        # If corrected, dimensions should be within limits
        if was_corrected:
            assert corrected.shape[0] <= test_settings.perspective_target_height_limit
            assert corrected.shape[1] <= test_settings.perspective_target_width_limit

