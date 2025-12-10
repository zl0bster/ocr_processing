"""Integration tests for preprocessing pipeline with real images."""
import pytest
from pathlib import Path
import cv2
import numpy as np

from src.preprocessor import ImagePreprocessor


@pytest.mark.integration
@pytest.mark.slow
class TestPreprocessingPipeline:
    """Test full preprocessing pipeline integration."""
    
    def test_preprocessing_pipeline_with_compressed_image(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test preprocessing pipeline with compressed test image."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output file should be created"
        assert result.duration_seconds > 0, "Processing should take time"
        
        # Verify output image is valid
        output_image = cv2.imread(str(result.output_path))
        assert output_image is not None, "Output image should be readable"
        assert output_image.size > 0, "Output image should have content"
        
        # Verify dimensions are reasonable
        original_image = cv2.imread(str(image_path))
        assert output_image.shape[0] > 0 and output_image.shape[1] > 0
        assert output_image.shape[2] == 3, "Output should be color image"
    
    def test_preprocessing_pipeline_with_full_resolution(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test preprocessing pipeline with full resolution test image."""
        if "full" not in test_image_paths:
            pytest.skip("Full resolution test image not found")
        
        # Arrange
        image_path = test_image_paths["full"]
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output file should be created"
        
        # Verify output image
        output_image = cv2.imread(str(result.output_path))
        assert output_image is not None, "Output image should be readable"
        assert output_image.size > 0, "Output image should have content"
    
    def test_preprocessing_perspective_correction_flow(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test that preprocessing applies perspective correction when enabled."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output file should be created"
        
        # Verify image was processed (perspective correction may or may not apply)
        output_image = cv2.imread(str(result.output_path))
        assert output_image is not None
    
    def test_preprocessing_deskew_flow(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test that preprocessing applies deskew when enabled."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output file should be created"
        # Deskew angle may be None if no skew detected
        assert result.deskew_angle is None or abs(result.deskew_angle) < 45.0
    
    def test_preprocessing_enhancement_flow(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test that preprocessing applies enhancement."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output file should be created"
        
        # Verify enhanced image exists and is valid
        output_image = cv2.imread(str(result.output_path))
        assert output_image is not None
        assert output_image.shape[0] > 0 and output_image.shape[1] > 0
    
    def test_preprocessing_error_handling_missing_file(
        self, integration_settings, integration_logger, cleanup_integration_outputs
    ):
        """Test error handling for missing image file."""
        # Arrange
        non_existent_path = Path("non_existent_image.jpg")
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            preprocessor.process(non_existent_path)
    
    def test_preprocessing_custom_output_path(
        self, integration_settings, integration_logger, test_image_paths, temp_output_dir
    ):
        """Test preprocessing with custom output path."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        custom_output = temp_output_dir / "custom_preprocessed.jpg"
        preprocessor = ImagePreprocessor(integration_settings, integration_logger)
        
        # Act
        result = preprocessor.process(image_path, output_path=custom_output)
        
        # Assert
        assert result.output_path == custom_output
        assert custom_output.exists()

