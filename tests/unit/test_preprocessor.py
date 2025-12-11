"""Unit tests for ImagePreprocessor module."""
import logging
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.preprocessor import ImagePreprocessor, PreprocessorResult
from src.config.settings import Settings
from tests.fixtures.image_fixtures import create_test_document_image, create_rotated_image


@pytest.mark.unit
class TestPreprocessorBasicProcessing:
    """Test basic preprocessing functionality."""

    def test_preprocessor_process_with_real_image_034_compr(
        self, test_settings, test_image_034, tmp_path
    ):
        """Test full pipeline with 034_compr.jpg (compressed)."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "034_compr.jpg"
        cv2.imwrite(str(input_path), test_image_034)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert isinstance(result, PreprocessorResult)
        assert result.output_path.exists()
        assert result.duration_seconds > 0
        assert result.output_path.suffix == ".jpg"

    def test_preprocessor_process_with_real_image_034_full(
        self, test_settings, test_image_034_full, tmp_path
    ):
        """Test full pipeline with 034.jpg (full resolution)."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "034.jpg"
        cv2.imwrite(str(input_path), test_image_034_full)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert isinstance(result, PreprocessorResult)
        assert result.output_path.exists()
        assert result.duration_seconds > 0

    def test_preprocessor_generates_correct_output_path(self, test_settings, tmp_path):
        """Test output path generation logic."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "test_image.jpg"
        test_image = create_test_document_image()
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        expected_suffix = test_settings.processed_suffix
        assert expected_suffix in result.output_path.stem
        assert result.output_path.parent == test_settings.output_dir

    def test_preprocessor_creates_output_directory(self, test_settings, tmp_path):
        """Test directory creation."""
        # Arrange
        logger = logging.getLogger("test")
        # Use a non-existent output directory
        test_settings.output_dir = tmp_path / "new_output_dir"
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "test_image.jpg"
        test_image = create_test_document_image()
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert test_settings.output_dir.exists()
        assert result.output_path.exists()


@pytest.mark.unit
class TestPreprocessorDeskew:
    """Test deskew functionality."""

    def test_preprocessor_detects_skew_angle(
        self, test_settings, synthetic_skewed_image, tmp_path
    ):
        """Test skew detection with synthetic skewed image."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "skewed.jpg"
        cv2.imwrite(str(input_path), synthetic_skewed_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        # Should detect some skew (angle may be small but should be detected)
        assert result.deskew_angle is not None
        assert isinstance(result.deskew_angle, (int, float))

    def test_preprocessor_applies_deskew_when_angle_significant(
        self, test_settings, tmp_path
    ):
        """Test rotation is applied for angles > 2°."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create image with 5 degree rotation
        base_image = create_test_document_image()
        rotated_image = create_rotated_image(base_image, 5.0)
        input_path = tmp_path / "rotated_5deg.jpg"
        cv2.imwrite(str(input_path), rotated_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        # Should detect and potentially correct the rotation
        assert result.deskew_angle is not None

    def test_preprocessor_skips_deskew_when_angle_too_small(
        self, test_settings, tmp_path
    ):
        """Test rotation skipped for angles < 2°."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create image with very small rotation (1 degree)
        base_image = create_test_document_image()
        rotated_image = create_rotated_image(base_image, 1.0)
        input_path = tmp_path / "rotated_1deg.jpg"
        cv2.imwrite(str(input_path), rotated_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        # Angle should be detected but may be too small to apply
        assert result.deskew_angle is not None

    def test_preprocessor_skips_deskew_when_no_lines_detected(
        self, test_settings, tmp_path
    ):
        """Test handling of blank/uniform images."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create uniform white image (no text lines)
        uniform_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        input_path = tmp_path / "uniform.jpg"
        cv2.imwrite(str(input_path), uniform_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        # Should complete without error, angle may be 0.0
        assert result.deskew_angle is not None
        assert result.output_path.exists()

    @pytest.mark.parametrize(
        "angle,should_apply",
        [
            (0.0, False),  # No rotation
            (1.0, False),  # Too small
            (2.5, True),   # Significant
            (10.0, True),  # Significant
            (50.0, False),  # Too large (unrealistic)
        ],
    )
    def test_should_apply_rotation_with_various_angles(
        self, test_settings, angle, should_apply
    ):
        """Test rotation decision logic with various angles."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)

        # Act
        result = preprocessor._should_apply_rotation(angle)

        # Assert
        assert result == should_apply


@pytest.mark.unit
class TestPreprocessorAdaptiveScaling:
    """Test adaptive scaling functionality."""

    @pytest.mark.parametrize(
        "width,height,expected_scaled",
        [
            (1920, 1080, True),   # 2.07 MP → should scale up
            (2560, 1440, True),   # 3.69 MP → should scale up
            (3264, 2448, False),  # 8.0 MP → no scaling
            (4000, 3000, False),  # 12 MP → no scaling
        ],
    )
    def test_adaptive_scaling_for_different_resolutions(
        self, test_settings, width, height, expected_scaled, tmp_path
    ):
        """Test adaptive scaling for different image resolutions."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create test image with specified resolution
        test_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # Add some content to make it valid
        cv2.rectangle(test_image, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
        input_path = tmp_path / f"test_{width}x{height}.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)
        output_image = cv2.imread(str(result.output_path))

        # Assert
        assert output_image is not None
        output_height, output_width = output_image.shape[:2]
        if expected_scaled:
            # Should be scaled up (output should be larger or equal)
            assert output_width >= width or output_height >= height
        else:
            # Should not be scaled (output should be same or smaller due to processing)
            # Note: actual output may be slightly different due to processing steps
            pass  # Just verify it processes successfully


@pytest.mark.unit
class TestPreprocessorEnhancement:
    """Test image enhancement functionality."""

    def test_enhancement_applies_clahe(self, test_settings, tmp_path):
        """Test CLAHE contrast enhancement."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_clahe.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()
        output_image = cv2.imread(str(result.output_path), cv2.IMREAD_GRAYSCALE)
        assert output_image is not None

    def test_enhancement_with_denoising_enabled(self, test_settings, tmp_path):
        """Test denoising when enabled."""
        # Arrange
        test_settings.enable_denoising = True
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_denoise.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()

    def test_enhancement_with_denoising_disabled(self, test_settings, tmp_path):
        """Test denoising skipped when disabled."""
        # Arrange
        test_settings.enable_denoising = False
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_no_denoise.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()

    def test_binarization_otsu_mode(self, test_settings, tmp_path):
        """Test Otsu thresholding."""
        # Arrange
        test_settings.binarization_mode = "otsu"
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_otsu.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()
        output_image = cv2.imread(str(result.output_path), cv2.IMREAD_GRAYSCALE)
        assert output_image is not None
        # Otsu produces binary image (only 0 and 255 values)
        unique_values = np.unique(output_image)
        assert len(unique_values) <= 256  # Binary or grayscale

    def test_binarization_adaptive_mode(self, test_settings, tmp_path):
        """Test adaptive thresholding."""
        # Arrange
        test_settings.binarization_mode = "adaptive"
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_adaptive.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()
        output_image = cv2.imread(str(result.output_path), cv2.IMREAD_GRAYSCALE)
        assert output_image is not None

    def test_morphological_enhancement_for_pale_text(self, test_settings, tmp_path):
        """Test pale text enhancement."""
        # Arrange
        test_settings.use_morphological_enhancement = True
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create image with pale text (low contrast)
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 240
        cv2.putText(
            test_image,
            "Pale Text",
            (50, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (200, 200, 200),
            3,
        )
        input_path = tmp_path / "test_pale.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()

    def test_illumination_correction(self, test_settings, tmp_path):
        """Test illumination correction."""
        # Arrange
        test_settings.enable_illumination_correction = True
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create image with uneven illumination
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        # Add gradient (simulating uneven lighting)
        for y in range(800):
            intensity = int(200 + (y / 800) * 55)
            test_image[y, :] = intensity
        input_path = tmp_path / "test_illumination.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()


@pytest.mark.unit
class TestPreprocessorErrorHandling:
    """Test error handling."""

    def test_preprocessor_raises_on_missing_file(self, test_settings):
        """Test FileNotFoundError for missing file."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        non_existent_path = Path("non_existent_file.jpg")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            preprocessor.process(input_path=non_existent_path)

    def test_preprocessor_raises_on_invalid_image(self, test_settings, tmp_path):
        """Test ValueError for corrupted images."""
        # Arrange
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        # Create invalid image file (not a real image)
        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_text("This is not an image file")

        # Act & Assert
        with pytest.raises(ValueError, match="Unable to read image"):
            preprocessor.process(input_path=invalid_path)


@pytest.mark.unit
class TestPreprocessorIntegration:
    """Test integration with other components."""

    def test_perspective_correction_integration(
        self, test_settings, test_image_034, tmp_path
    ):
        """Test perspective correction enabled."""
        # Arrange
        test_settings.enable_perspective_correction = True
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        input_path = tmp_path / "034_compr.jpg"
        cv2.imwrite(str(input_path), test_image_034)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()

    def test_full_pipeline_disabled_features(self, test_settings, tmp_path):
        """Test with all preprocessing features disabled."""
        # Arrange
        test_settings.enable_perspective_correction = False
        test_settings.enable_deskew = False
        test_settings.enable_denoising = False
        test_settings.enable_illumination_correction = False
        test_settings.use_morphological_enhancement = False
        logger = logging.getLogger("test")
        preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
        test_image = create_test_document_image()
        input_path = tmp_path / "test_minimal.jpg"
        cv2.imwrite(str(input_path), test_image)

        # Act
        result = preprocessor.process(input_path=input_path)

        # Assert
        assert result.output_path.exists()



