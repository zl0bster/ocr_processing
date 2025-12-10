"""Unit tests for Settings configuration module."""
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config.settings import Settings


@pytest.mark.unit
class TestSettingsDefaults:
    """Test default values for Settings."""

    def test_default_values(self):
        """Test that Settings has correct default values."""
        # Arrange & Act
        settings = Settings()

        # Assert
        assert settings.input_dir == Path("images")
        assert settings.output_dir == Path("results")
        assert settings.log_dir == Path("logs")
        assert settings.log_level == "INFO"
        assert settings.ocr_language == "ru"
        assert settings.ocr_confidence_threshold == 0.5
        assert settings.ocr_use_gpu is False
        assert settings.enable_deskew is True
        assert settings.enable_perspective_correction is True
        assert settings.gaussian_blur_kernel == 3
        assert settings.illumination_kernel == 31
        assert settings.adaptive_block_size == 31
        assert settings.template_name == "otk_v1"
        assert settings.binarization_mode == "otsu"


@pytest.mark.unit
class TestSettingsPathConversion:
    """Test path field conversion."""

    def test_path_conversion_from_string(self):
        """Test that string paths are converted to Path objects."""
        # Arrange & Act
        settings = Settings(
            input_dir="test_input",
            output_dir="test_output",
            log_dir="test_logs",
        )

        # Assert
        assert isinstance(settings.input_dir, Path)
        assert isinstance(settings.output_dir, Path)
        assert isinstance(settings.log_dir, Path)
        assert settings.input_dir == Path("test_input")
        assert settings.output_dir == Path("test_output")
        assert settings.log_dir == Path("test_logs")

    def test_path_conversion_from_path(self):
        """Test that Path objects remain as Path objects."""
        # Arrange
        input_path = Path("test_input")
        output_path = Path("test_output")

        # Act
        settings = Settings(
            input_dir=input_path,
            output_dir=output_path,
        )

        # Assert
        assert isinstance(settings.input_dir, Path)
        assert isinstance(settings.output_dir, Path)
        assert settings.input_dir == input_path
        assert settings.output_dir == output_path

    def test_region_template_file_path_conversion(self):
        """Test region_template_file path conversion."""
        # Arrange & Act
        settings = Settings(region_template_file="config/custom/regions.json")

        # Assert
        assert isinstance(settings.region_template_file, Path)
        assert settings.region_template_file == Path("config/custom/regions.json")

    def test_table_template_file_path_conversion(self):
        """Test table_template_file path conversion."""
        # Arrange & Act
        settings = Settings(table_template_file="config/custom/tables.json")

        # Assert
        assert isinstance(settings.table_template_file, Path)
        assert settings.table_template_file == Path("config/custom/tables.json")


@pytest.mark.unit
class TestGaussianBlurKernelValidator:
    """Test gaussian_blur_kernel field validator."""

    @pytest.mark.parametrize("value", [1, 3, 5, 7, 9, 15, 31])
    def test_valid_odd_values(self, value):
        """Test that odd positive values are accepted."""
        # Arrange & Act
        settings = Settings(gaussian_blur_kernel=value)

        # Assert
        assert settings.gaussian_blur_kernel == value

    @pytest.mark.parametrize("value", [2, 4, 6, 8, 10, 16, 32])
    def test_invalid_even_values(self, value):
        """Test that even values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(gaussian_blur_kernel=value)

        assert "gaussian_blur_kernel must be odd" in str(exc_info.value)

    @pytest.mark.parametrize("value", [0, -1, -3, -5])
    def test_invalid_non_positive_values(self, value):
        """Test that non-positive values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(gaussian_blur_kernel=value)

        assert "gaussian_blur_kernel must be a positive integer" in str(exc_info.value)


@pytest.mark.unit
class TestIlluminationKernelValidator:
    """Test illumination_kernel field validator."""

    @pytest.mark.parametrize("value", [15, 17, 19, 31, 51, 101])
    def test_valid_odd_values_above_threshold(self, value):
        """Test that odd values >= 15 are accepted."""
        # Arrange & Act
        settings = Settings(illumination_kernel=value)

        # Assert
        assert settings.illumination_kernel == value

    @pytest.mark.parametrize("value", [1, 3, 5, 7, 9, 11, 13])
    def test_invalid_values_below_threshold(self, value):
        """Test that values < 15 raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(illumination_kernel=value)

        assert "illumination_kernel should be sufficiently large (>= 15)" in str(exc_info.value)

    @pytest.mark.parametrize("value", [2, 4, 16, 18, 32])
    def test_invalid_even_values(self, value):
        """Test that even values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(illumination_kernel=value)

        assert "illumination_kernel must be odd" in str(exc_info.value)

    @pytest.mark.parametrize("value", [0, -1, -15])
    def test_invalid_non_positive_values(self, value):
        """Test that non-positive values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(illumination_kernel=value)

        assert "illumination_kernel must be a positive integer" in str(exc_info.value)


@pytest.mark.unit
class TestAdaptiveBlockSizeValidator:
    """Test adaptive_block_size field validator."""

    @pytest.mark.parametrize("value", [1, 3, 5, 7, 31, 51])
    def test_valid_odd_values(self, value):
        """Test that odd positive values are accepted."""
        # Arrange & Act
        settings = Settings(adaptive_block_size=value)

        # Assert
        assert settings.adaptive_block_size == value

    @pytest.mark.parametrize("value", [2, 4, 6, 8, 16, 32])
    def test_invalid_even_values(self, value):
        """Test that even values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(adaptive_block_size=value)

        assert "adaptive_block_size must be odd" in str(exc_info.value)

    @pytest.mark.parametrize("value", [0, -1, -3])
    def test_invalid_non_positive_values(self, value):
        """Test that non-positive values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(adaptive_block_size=value)

        assert "adaptive_block_size must be a positive integer" in str(exc_info.value)


@pytest.mark.unit
class TestPerspectiveMinAreaRatioValidator:
    """Test perspective_min_area_ratio field validator."""

    @pytest.mark.parametrize("value", [0.1, 0.2, 0.5, 0.9, 1.0])
    def test_valid_values_in_range(self, value):
        """Test that values between 0.0 and 1.0 are accepted."""
        # Arrange & Act
        settings = Settings(perspective_min_area_ratio=value)

        # Assert
        assert settings.perspective_min_area_ratio == value

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.1, 2.0])
    def test_invalid_values_out_of_range(self, value):
        """Test that values outside 0.0-1.0 raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(perspective_min_area_ratio=value)

        assert "perspective_min_area_ratio must be between 0.0 and 1.0" in str(exc_info.value)


@pytest.mark.unit
class TestPerspectiveCornerEpsilonValidator:
    """Test perspective_corner_epsilon field validator."""

    @pytest.mark.parametrize("value", [0.01, 0.02, 0.05, 0.1, 0.5, 1.0])
    def test_valid_values_in_range(self, value):
        """Test that values between 0.0 and 1.0 are accepted."""
        # Arrange & Act
        settings = Settings(perspective_corner_epsilon=value)

        # Assert
        assert settings.perspective_corner_epsilon == value

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.1, 2.0])
    def test_invalid_values_out_of_range(self, value):
        """Test that values outside 0.0-1.0 raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(perspective_corner_epsilon=value)

        assert "perspective_corner_epsilon must be between 0.0 and 1.0" in str(exc_info.value)


@pytest.mark.unit
class TestPerspectiveTargetSizeValidators:
    """Test perspective target size limit validators."""

    @pytest.mark.parametrize("value", [1, 100, 1000, 3000, 5000])
    def test_valid_positive_values_width(self, value):
        """Test that positive values are accepted for width limit."""
        # Arrange & Act
        settings = Settings(perspective_target_width_limit=value)

        # Assert
        assert settings.perspective_target_width_limit == value

    @pytest.mark.parametrize("value", [1, 100, 1000, 2000, 5000])
    def test_valid_positive_values_height(self, value):
        """Test that positive values are accepted for height limit."""
        # Arrange & Act
        settings = Settings(perspective_target_height_limit=value)

        # Assert
        assert settings.perspective_target_height_limit == value

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_invalid_non_positive_values_width(self, value):
        """Test that non-positive values raise ValidationError for width."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(perspective_target_width_limit=value)

        assert "perspective target size limits must be positive integers" in str(exc_info.value)

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_invalid_non_positive_values_height(self, value):
        """Test that non-positive values raise ValidationError for height."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(perspective_target_height_limit=value)

        assert "perspective target size limits must be positive integers" in str(exc_info.value)


@pytest.mark.unit
class TestPerspectiveMinCornerDistanceValidator:
    """Test perspective_min_corner_distance field validator."""

    @pytest.mark.parametrize("value", [1, 10, 50, 100, 500])
    def test_valid_positive_values(self, value):
        """Test that positive values are accepted."""
        # Arrange & Act
        settings = Settings(perspective_min_corner_distance=value)

        # Assert
        assert settings.perspective_min_corner_distance == value

    @pytest.mark.parametrize("value", [0, -1, -10])
    def test_invalid_non_positive_values(self, value):
        """Test that non-positive values raise ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Settings(perspective_min_corner_distance=value)

        assert "perspective_min_corner_distance must be a positive integer" in str(exc_info.value)


@pytest.mark.unit
class TestSettingsFromEnvironment:
    """Test loading settings from environment variables."""

    def test_load_from_env_variables(self, monkeypatch):
        """Test that settings can be loaded from environment variables."""
        # Arrange
        monkeypatch.setenv("INPUT_DIR", "env_images")
        monkeypatch.setenv("OUTPUT_DIR", "env_results")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("OCR_LANGUAGE", "en")
        monkeypatch.setenv("GAUSSIAN_BLUR_KERNEL", "5")

        # Act
        settings = Settings()

        # Assert
        assert settings.input_dir == Path("env_images")
        assert settings.output_dir == Path("env_results")
        assert settings.log_level == "DEBUG"
        assert settings.ocr_language == "en"
        assert settings.gaussian_blur_kernel == 5

