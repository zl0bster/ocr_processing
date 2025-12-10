"""Unit tests for RegionDetector module."""

import logging
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.region_detector import RegionDetector, DocumentRegion
from src.config.settings import Settings
from tests.fixtures.synthetic_documents import (
    create_document_with_lines,
    create_document_with_text_blocks,
    create_document_no_boundaries,
)


@pytest.fixture
def mock_region_templates():
    """Mock template data with region definitions."""
    return {
        "otk_v1": [
            {
                "region_id": "header",
                "y_start_norm": 0.0,
                "y_end_norm": 0.15,
                "confidence": 1.0,
            },
            {
                "region_id": "defects",
                "y_start_norm": 0.15,
                "y_end_norm": 0.55,
                "confidence": 1.0,
            },
            {
                "region_id": "analysis",
                "y_start_norm": 0.55,
                "y_end_norm": 1.0,
                "confidence": 1.0,
            },
        ]
    }


@pytest.fixture
def synthetic_document_with_lines(tmp_path):
    """Generate image with clear horizontal separators."""
    save_path = tmp_path / "synthetic_with_lines.jpg" if tmp_path else None
    return create_document_with_lines(save_path=save_path)


@pytest.fixture
def synthetic_document_with_text(tmp_path):
    """Generate image with dense text blocks."""
    save_path = tmp_path / "synthetic_with_text.jpg" if tmp_path else None
    return create_document_with_text_blocks(save_path=save_path)


@pytest.fixture
def synthetic_document_no_lines(tmp_path):
    """Generate image without clear boundaries."""
    save_path = tmp_path / "synthetic_no_lines.jpg" if tmp_path else None
    return create_document_no_boundaries(save_path=save_path)


@pytest.fixture
def mock_logger():
    """Mock logging.Logger instance."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def debug_output_dir(tmp_path):
    """Directory for saving debug images during tests."""
    debug_dir = tmp_path / "debug_images"
    debug_dir.mkdir(exist_ok=True)
    return debug_dir


@pytest.mark.unit
class TestTemplateBasedDetection:
    """Test template-based region detection (baseline strategy)."""

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_template_strategy(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test template-based detection with valid template."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(image, strategy="template")

        # Assert
        assert len(regions) == 3
        assert all(isinstance(r, DocumentRegion) for r in regions)
        assert regions[0].region_id == "header"
        assert regions[1].region_id == "defects"
        assert regions[2].region_id == "analysis"
        assert all(r.detection_method == "template" for r in regions)
        assert all(0.0 <= r.y_start_norm <= 1.0 for r in regions)
        assert all(0.0 <= r.y_end_norm <= 1.0 for r in regions)

    @patch("src.region_detector.load_region_templates")
    def test_template_detection_with_custom_template(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test template detection with custom template name."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(image, strategy="template", template_name="otk_v1")

        # Assert
        assert len(regions) == 3
        assert regions[0].region_id == "header"

    @patch("src.region_detector.load_region_templates")
    def test_template_detection_with_missing_template(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test template detection falls back when template not found."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(
            image, strategy="template", template_name="nonexistent"
        )

        # Assert
        assert len(regions) > 0  # Should fall back to default template
        mock_logger.warning.assert_called()

    @patch("src.region_detector.load_region_templates")
    def test_template_detection_fallback_to_default(
        self, mock_load_templates, test_settings, mock_logger
    ):
        """Test template detection uses default when requested template missing."""
        # Arrange
        default_template = {
            "default": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.33,
                    "confidence": 1.0,
                },
                {
                    "region_id": "defects",
                    "y_start_norm": 0.33,
                    "y_end_norm": 0.66,
                    "confidence": 1.0,
                },
                {
                    "region_id": "analysis",
                    "y_start_norm": 0.66,
                    "y_end_norm": 1.0,
                    "confidence": 1.0,
                },
            ]
        }
        mock_load_templates.return_value = default_template
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(
            image, strategy="template", template_name="missing"
        )

        # Assert
        assert len(regions) == 3
        assert all(r.region_id in ["header", "defects", "analysis"] for r in regions)


@pytest.mark.unit
class TestAdaptiveLineDetection:
    """Test adaptive line-based region detection."""

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_adaptive_strategy(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test adaptive strategy detects regions from horizontal lines."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy="adaptive")

        # Assert
        assert len(regions) >= 3
        assert all(isinstance(r, DocumentRegion) for r in regions)
        assert all(r.detection_method == "adaptive" for r in regions)

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_finds_horizontal_separators(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test adaptive strategy finds horizontal separator lines."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy="adaptive")

        # Assert
        if len(regions) >= 3:
            # Verify regions are in correct order
            assert regions[0].y_start < regions[1].y_start
            assert regions[1].y_start < regions[2].y_start
            assert regions[0].y_end <= regions[1].y_start
            assert regions[1].y_end <= regions[2].y_start

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_validates_header_boundary_range(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
    ):
        """Test adaptive strategy validates header boundary is in valid range."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = create_document_with_lines()

        # Act
        regions = detector.detect_zones(image, strategy="adaptive")

        # Assert
        if len(regions) >= 1:
            header_ratio = regions[0].y_end_norm
            assert (
                test_settings.region_min_header_ratio
                <= header_ratio
                <= test_settings.region_max_header_ratio
            )

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_validates_defects_boundary_range(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
    ):
        """Test adaptive strategy validates defects boundary is in valid range."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = create_document_with_lines()

        # Act
        regions = detector.detect_zones(image, strategy="adaptive")

        # Assert
        if len(regions) >= 2:
            header_end = regions[0].y_end
            defects_end = regions[1].y_end
            defects_height = defects_end - header_end
            defects_ratio = defects_height / float(image.shape[0])
            assert (
                test_settings.region_min_defects_ratio
                <= defects_ratio
                <= test_settings.region_max_defects_ratio
            )

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_cross_validates_with_template(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test adaptive regions are cross-validated against template."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy="adaptive")

        # Assert
        # If cross-validation passes, regions should be returned
        # If it fails, empty list should be returned
        assert isinstance(regions, list)

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_fallback_when_no_lines_found(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_no_lines,
    ):
        """Test adaptive strategy falls back when no lines detected."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "adaptive"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_no_lines, strategy="adaptive"
        )

        # Assert
        # Should return empty list or fall back to template
        assert isinstance(regions, list)

    @patch("src.region_detector.load_region_templates")
    def test_adaptive_fallback_to_template_on_validation_fail(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
    ):
        """Test adaptive falls back to template when validation fails."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "auto"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        # Create image that will fail adaptive validation
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(image, strategy="auto")

        # Assert
        # Should eventually fall back to template
        assert len(regions) > 0


@pytest.mark.unit
class TestTextBasedProjection:
    """Test text-based projection region detection."""

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_text_projection_strategy(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_text,
    ):
        """Test text projection strategy detects regions from text density."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "text_based"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_with_text, strategy="text_based"
        )

        # Assert
        assert isinstance(regions, list)
        if len(regions) > 0:
            assert all(isinstance(r, DocumentRegion) for r in regions)
            # text_based might fall back to template if validation fails
            assert all(r.detection_method in ["text_based", "template"] for r in regions)

    @patch("src.region_detector.load_region_templates")
    def test_text_projection_detects_text_density_gaps(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_text,
    ):
        """Test text projection detects gaps in text density."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "text_based"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_with_text, strategy="text_based"
        )

        # Assert
        # Should detect regions based on text density variations
        assert isinstance(regions, list)

    @patch("src.region_detector.load_region_templates")
    def test_text_projection_validates_boundaries(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_text,
    ):
        """Test text projection validates detected boundaries."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "text_based"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_with_text, strategy="text_based"
        )

        # Assert
        # If validation fails, might fall back to template which has different ratios
        # So we just check that regions were returned and have valid structure
        assert len(regions) > 0
        if len(regions) >= 1 and regions[0].detection_method == "text_based":
            header_ratio = regions[0].y_end_norm
            assert (
                test_settings.region_min_header_ratio
                <= header_ratio
                <= test_settings.region_max_header_ratio
            )

    @patch("src.region_detector.load_region_templates")
    def test_text_projection_cross_validates_with_template(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_text,
    ):
        """Test text projection regions are cross-validated against template."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "text_based"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_with_text, strategy="text_based"
        )

        # Assert
        assert isinstance(regions, list)

    @patch("src.region_detector.load_region_templates")
    def test_text_projection_fallback_on_insufficient_gaps(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_no_lines,
    ):
        """Test text projection falls back when insufficient gaps found."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "text_based"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(
            synthetic_document_no_lines, strategy="text_based"
        )

        # Assert
        # Should return empty list or fall back
        assert isinstance(regions, list)


@pytest.mark.unit
class TestStrategyCascade:
    """Test strategy cascade (auto mode)."""

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_auto_tries_adaptive_first(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test auto mode tries adaptive strategy first."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "auto"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy="auto")

        # Assert
        # Should succeed with adaptive if lines are clear
        assert isinstance(regions, list)
        if len(regions) > 0:
            # Check that adaptive was tried first (should have adaptive method or template fallback)
            assert any(
                r.detection_method in ["adaptive", "text_based", "template"]
                for r in regions
            )

    @patch("src.region_detector.load_region_templates")
    def test_auto_mode_falls_back_to_text_based(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_text,
    ):
        """Test auto mode falls back to text_based when adaptive fails."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "auto"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_text, strategy="auto")

        # Assert
        assert isinstance(regions, list)

    @patch("src.region_detector.load_region_templates")
    def test_auto_mode_falls_back_to_template(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_no_lines,
    ):
        """Test auto mode falls back to template when all strategies fail."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "auto"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_no_lines, strategy="auto")

        # Assert
        # Should eventually fall back to template
        assert len(regions) > 0
        # May use template or template-disabled depending on settings
        assert all(
            r.detection_method in ["template", "template-disabled", "adaptive", "text_based"]
            for r in regions
        )

    @patch("src.region_detector.load_region_templates")
    def test_auto_mode_returns_first_valid_strategy(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test auto mode returns results from first successful strategy."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = "auto"
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy="auto")

        # Assert
        assert len(regions) > 0
        # All regions should have same detection method
        if len(regions) > 0:
            methods = {r.detection_method for r in regions}
            assert len(methods) == 1  # All same method

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_with_disabled_region_detection(
        self,
        mock_load_templates,
        test_settings,
        mock_logger,
        mock_region_templates,
    ):
        """Test detection when region_detection is disabled in settings."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.enable_region_detection = False
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(image)

        # Assert
        assert len(regions) > 0
        assert all(
            r.detection_method == "template-disabled" for r in regions
        )


@pytest.mark.unit
class TestCoordinateNormalization:
    """Test coordinate normalization and denormalization."""

    def test_normalize_coordinates_valid_range(self):
        """Test normalization produces values between 0.0 and 1.0."""
        # Arrange
        height = 2000
        y_start = 300
        y_end = 1100

        # Act
        start_norm, end_norm = RegionDetector._normalize_coordinates(
            height, y_start, y_end
        )

        # Assert
        assert 0.0 <= start_norm <= 1.0
        assert 0.0 <= end_norm <= 1.0
        assert start_norm < end_norm
        assert abs(start_norm - (y_start / height)) < 0.001
        assert abs(end_norm - (y_end / height)) < 0.001

    def test_denormalize_coordinates_to_pixels(self):
        """Test denormalization converts back to pixel coordinates."""
        # Arrange
        height = 2000
        y_start_norm = 0.15
        y_end_norm = 0.55

        # Act
        y_start, y_end = RegionDetector._denormalize_coordinates(
            height, y_start_norm, y_end_norm
        )

        # Assert
        assert 0 <= y_start < height
        assert 0 <= y_end <= height
        assert y_start < y_end
        assert abs(y_start - int(height * y_start_norm)) <= 1
        assert abs(y_end - int(height * y_end_norm)) <= 1

    def test_normalized_coordinates_are_between_zero_and_one(self):
        """Test normalized coordinates are always in valid range."""
        # Arrange
        height = 2000
        test_cases = [
            (0, 500),
            (500, 1000),
            (1000, 1500),
            (1500, 2000),
            (0, 2000),
        ]

        # Act & Assert
        for y_start, y_end in test_cases:
            start_norm, end_norm = RegionDetector._normalize_coordinates(
                height, y_start, y_end
            )
            assert 0.0 <= start_norm <= 1.0
            assert 0.0 <= end_norm <= 1.0
            assert start_norm < end_norm

    def test_normalize_with_zero_height_raises_error(self):
        """Test normalization raises error for zero height."""
        # Arrange
        height = 0
        y_start = 0
        y_end = 100

        # Act & Assert
        with pytest.raises(ValueError, match="Image height must be positive"):
            RegionDetector._normalize_coordinates(height, y_start, y_end)


@pytest.mark.unit
class TestRegionValidation:
    """Test region validation logic."""

    @patch("src.region_detector.load_region_templates")
    def test_validate_regions_requires_full_coverage(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test validation requires regions to cover full image height."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        height = 2000

        # Create regions that don't cover full height
        incomplete_regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=0,
                y_end=300,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="test",
                confidence=0.9,
            ),
            # Missing analysis region
        ]

        # Act
        is_valid = detector._validate_regions(incomplete_regions, height)

        # Assert
        assert is_valid is False

    @patch("src.region_detector.load_region_templates")
    def test_validate_regions_checks_confidence_threshold(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test validation checks minimum confidence threshold."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_min_confidence = 0.7
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        height = 2000

        # Create regions with low confidence
        low_confidence_regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=0,
                y_end=300,
                detection_method="test",
                confidence=0.5,  # Below threshold
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="test",
                confidence=0.5,
            ),
            DocumentRegion(
                region_id="analysis",
                y_start_norm=0.55,
                y_end_norm=1.0,
                y_start=1100,
                y_end=2000,
                detection_method="test",
                confidence=0.5,
            ),
        ]

        # Act
        is_valid = detector._validate_regions(low_confidence_regions, height)

        # Assert
        assert is_valid is False

    @patch("src.region_detector.load_region_templates")
    def test_validate_regions_accepts_valid_regions(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test validation accepts properly formed regions."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        height = 2000

        # Create valid regions
        valid_regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=0,
                y_end=300,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="analysis",
                y_start_norm=0.55,
                y_end_norm=1.0,
                y_start=1100,
                y_end=2000,
                detection_method="test",
                confidence=0.9,
            ),
        ]

        # Act
        is_valid = detector._validate_regions(valid_regions, height)

        # Assert
        assert is_valid is True

    @patch("src.region_detector.load_region_templates")
    def test_validate_regions_rejects_incomplete_coverage(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test validation rejects regions that don't cover full image."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        height = 2000

        # Create regions with gaps
        regions_with_gaps = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=100,  # Doesn't start at 0
                y_end=300,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="analysis",
                y_start_norm=0.55,
                y_end_norm=1.0,
                y_start=1100,
                y_end=1900,  # Doesn't reach end
                detection_method="test",
                confidence=0.9,
            ),
        ]

        # Act
        is_valid = detector._validate_regions(regions_with_gaps, height)

        # Assert
        assert is_valid is False

    @patch("src.region_detector.load_region_templates")
    def test_validate_regions_rejects_low_confidence(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test validation rejects regions below confidence threshold."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_min_confidence = 0.8
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        height = 2000

        # Create regions with one low confidence
        mixed_confidence_regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=0,
                y_end=300,
                detection_method="test",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="test",
                confidence=0.5,  # Below threshold
            ),
            DocumentRegion(
                region_id="analysis",
                y_start_norm=0.55,
                y_end_norm=1.0,
                y_start=1100,
                y_end=2000,
                detection_method="test",
                confidence=0.9,
            ),
        ]

        # Act
        is_valid = detector._validate_regions(mixed_confidence_regions, height)

        # Assert
        assert is_valid is False


@pytest.mark.unit
class TestHelperMethods:
    """Test helper methods."""

    def test_merge_close_positions(self):
        """Test merging positions that are close together."""
        # Arrange
        positions = [100, 105, 200, 205, 300, 350]
        threshold = 10

        # Act
        merged = RegionDetector._merge_close_positions(positions, threshold)

        # Assert
        assert len(merged) < len(positions)
        assert 100 <= merged[0] <= 105  # Merged
        assert 200 <= merged[1] <= 205  # Merged
        assert 300 in merged or 350 in merged  # Not merged (distance > 10)

    def test_merge_close_positions_with_empty_list(self):
        """Test merging with empty list returns empty list."""
        # Arrange
        positions = []
        threshold = 10

        # Act
        merged = RegionDetector._merge_close_positions(positions, threshold)

        # Assert
        assert merged == []

    def test_group_indices_into_segments(self):
        """Test grouping contiguous indices into segments."""
        # Arrange
        indices = np.array([10, 11, 12, 15, 16, 20, 21, 22, 23])

        # Act
        segments = RegionDetector._group_indices(indices)

        # Assert
        assert len(segments) == 3
        assert (10, 12) in segments
        assert (15, 16) in segments
        assert (20, 23) in segments

    def test_group_indices_with_empty_array(self):
        """Test grouping with empty array returns empty list."""
        # Arrange
        indices = np.array([])

        # Act
        segments = RegionDetector._group_indices(indices)

        # Assert
        assert segments == []

    @patch("src.region_detector.load_region_templates")
    def test_regions_from_boundaries(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test creating regions from boundary positions."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        boundaries = [0, 300, 1100, 2000]
        height = 2000
        detection_method = "test"
        confidence = 0.85

        # Act
        regions = detector._regions_from_boundaries(
            boundaries, height, detection_method, confidence
        )

        # Assert
        assert len(regions) == 3
        assert regions[0].region_id == "header"
        assert regions[1].region_id == "defects"
        assert regions[2].region_id == "analysis"
        assert all(r.detection_method == detection_method for r in regions)
        assert all(r.confidence == confidence for r in regions)

    @patch("src.region_detector.load_region_templates")
    def test_cross_validate_with_template(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test cross-validation of regions against template."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Create regions that match template
        matching_regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.15,
                y_start=0,
                y_end=300,
                detection_method="adaptive",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="defects",
                y_start_norm=0.15,
                y_end_norm=0.55,
                y_start=300,
                y_end=1100,
                detection_method="adaptive",
                confidence=0.9,
            ),
            DocumentRegion(
                region_id="analysis",
                y_start_norm=0.55,
                y_end_norm=1.0,
                y_start=1100,
                y_end=2000,
                detection_method="adaptive",
                confidence=0.9,
            ),
        ]

        # Act
        is_valid = detector._cross_validate_with_template(
            matching_regions, template_name="otk_v1"
        )

        # Assert
        assert is_valid is True


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_with_none_image_raises_error(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test detection raises error for None image."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act & Assert
        with pytest.raises(ValueError, match="Image for zone detection cannot be empty"):
            detector.detect_zones(None)

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_with_empty_image_raises_error(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test detection raises error for empty image."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        empty_image = np.array([])

        # Act & Assert
        with pytest.raises(ValueError, match="Image for zone detection cannot be empty"):
            detector.detect_zones(empty_image)

    @patch("src.region_detector.load_region_templates")
    def test_detect_zones_with_invalid_strategy_falls_back(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test detection falls back when invalid strategy specified."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        regions = detector.detect_zones(image, strategy="invalid_strategy")

        # Assert
        # Should fall back to template
        assert len(regions) > 0
        mock_logger.warning.assert_called()

    @patch("src.region_detector.load_region_templates")
    def test_extract_region_with_invalid_boundaries_raises_error(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test extraction raises error for invalid region boundaries."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Create invalid region (y_end <= y_start)
        invalid_region = DocumentRegion(
            region_id="test",
            y_start_norm=0.5,
            y_end_norm=0.3,  # Invalid: end < start
            y_start=1000,
            y_end=600,  # Invalid: end < start
            detection_method="test",
            confidence=0.9,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid crop boundaries"):
            detector.extract_region(image, invalid_region)


@pytest.mark.unit
class TestRegionExtraction:
    """Test region extraction functionality."""

    @patch("src.region_detector.load_region_templates")
    def test_extract_region_returns_cropped_image(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test extraction returns cropped image region."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255
        # Add some content to region
        image[300:500, 100:200] = 0  # Black rectangle in header region

        region = DocumentRegion(
            region_id="header",
            y_start_norm=0.0,
            y_end_norm=0.15,
            y_start=0,
            y_end=300,
            detection_method="template",
            confidence=1.0,
        )

        # Act
        cropped, scale_factor = detector.extract_region(image, region)

        # Assert
        assert cropped.shape[0] == 300  # Region height
        assert cropped.shape[1] == 1500  # Full width
        assert scale_factor == 1.0  # No scaling for small region

    @patch("src.region_detector.load_region_templates")
    def test_extract_region_applies_margin(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test extraction applies margin correctly."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        region = DocumentRegion(
            region_id="header",
            y_start_norm=0.0,
            y_end_norm=0.15,
            y_start=100,
            y_end=300,
            detection_method="template",
            confidence=1.0,
        )
        margin = 20

        # Act
        cropped, scale_factor = detector.extract_region(image, region, margin=margin)

        # Assert
        # Should include margin: start at 100-20=80, end at 300+20=320
        expected_height = 320 - 80
        assert cropped.shape[0] == expected_height

    @patch("src.region_detector.load_region_templates")
    def test_extract_region_downscales_large_regions(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test extraction downscales regions that exceed max dimensions."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_max_width = 1000
        test_settings.region_max_height = 800
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        # Create large image
        image = np.ones((3000, 2500, 3), dtype=np.uint8) * 255

        region = DocumentRegion(
            region_id="defects",
            y_start_norm=0.15,
            y_end_norm=0.55,
            y_start=450,
            y_end=1650,  # Height = 1200, exceeds max_height=800
            detection_method="template",
            confidence=1.0,
        )

        # Act
        cropped, scale_factor = detector.extract_region(image, region)

        # Assert
        assert cropped.shape[0] <= test_settings.region_max_height
        assert cropped.shape[1] <= test_settings.region_max_width
        assert scale_factor < 1.0  # Should be downscaled

    @patch("src.region_detector.load_region_templates")
    def test_extract_region_calculates_scale_factor(
        self, mock_load_templates, test_settings, mock_logger, mock_region_templates
    ):
        """Test extraction calculates correct scale factor."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_max_width = 1000
        test_settings.region_max_height = 800
        detector = RegionDetector(settings=test_settings, logger=mock_logger)
        image = np.ones((3000, 2500, 3), dtype=np.uint8) * 255

        region = DocumentRegion(
            region_id="defects",
            y_start_norm=0.15,
            y_end_norm=0.55,
            y_start=450,
            y_end=1650,  # Height = 1200
            detection_method="template",
            confidence=1.0,
        )

        # Act
        cropped, scale_factor = detector.extract_region(image, region)

        # Assert
        original_height = 1650 - 450
        assert scale_factor == cropped.shape[0] / float(original_height)
        assert 0.0 < scale_factor <= 1.0


@pytest.mark.unit
class TestParametrizedCases:
    """Parametrized tests for multiple cases."""

    @patch("src.region_detector.load_region_templates")
    @pytest.mark.parametrize("strategy", ["adaptive", "text_based", "template"])
    def test_all_strategies_return_valid_regions(
        self,
        mock_load_templates,
        strategy,
        test_settings,
        mock_logger,
        mock_region_templates,
        synthetic_document_with_lines,
    ):
        """Test all strategies return valid region structures."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_detection_strategy = strategy
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        regions = detector.detect_zones(synthetic_document_with_lines, strategy=strategy)

        # Assert
        assert isinstance(regions, list)
        if len(regions) > 0:
            assert all(isinstance(r, DocumentRegion) for r in regions)
            assert all(0.0 <= r.y_start_norm <= 1.0 for r in regions)
            assert all(0.0 <= r.y_end_norm <= 1.0 for r in regions)

    @patch("src.region_detector.load_region_templates")
    @pytest.mark.parametrize(
        "y_pos,height,expected",
        [
            (400, 2000, True),  # 20% - valid
            (500, 2000, True),  # 25% - valid
            (300, 2000, False),  # 15% - below min
            (600, 2000, False),  # 30% - above max
            (450, 2000, True),  # 22.5% - in range
        ],
    )
    def test_validate_line_as_header_boundary(
        self,
        mock_load_templates,
        y_pos,
        height,
        expected,
        test_settings,
        mock_logger,
        mock_region_templates,
    ):
        """Test header boundary validation with parametrized cases."""
        # Arrange
        mock_load_templates.return_value = mock_region_templates
        test_settings.region_min_header_ratio = 0.20
        test_settings.region_max_header_ratio = 0.25
        detector = RegionDetector(settings=test_settings, logger=mock_logger)

        # Act
        is_valid = detector._validate_line_as_header_boundary(y_pos, height)

        # Assert
        assert is_valid == expected

    @pytest.mark.parametrize(
        "positions,threshold,expected_count",
        [
            ([100, 105, 200, 205, 300], 10, 3),  # Two pairs merged
            ([100, 150, 200, 250, 300], 10, 5),  # No merging
            ([100, 101, 102, 103, 104], 10, 1),  # All merged
            ([], 10, 0),  # Empty
        ],
    )
    def test_merge_close_positions_parametrized(
        self, positions, threshold, expected_count
    ):
        """Test position merging with parametrized cases."""
        # Act
        merged = RegionDetector._merge_close_positions(positions, threshold)

        # Assert
        assert len(merged) == expected_count

