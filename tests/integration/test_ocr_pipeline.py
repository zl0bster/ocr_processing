"""Integration tests for OCR pipeline with real PaddleOCR."""
import pytest
import json
from pathlib import Path

from src.ocr_engine import OCREngine


@pytest.mark.integration
@pytest.mark.requires_ocr
@pytest.mark.slow
class TestOCRPipeline:
    """Test OCR engine with actual PaddleOCR."""
    
    def test_ocr_with_compressed_image(
        self, shared_ocr_engine, integration_settings, integration_logger, 
        test_image_paths, cleanup_integration_outputs
    ):
        """Test OCR processing with compressed test image."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        result = shared_ocr_engine.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output JSON should be created"
        assert result.total_texts_found > 0, "Should detect at least some text"
        assert result.average_confidence > 0.0, "Average confidence should be positive"
        assert result.duration_seconds > 0, "Processing should take time"
        assert result.duration_seconds < 60, "Should complete in reasonable time (< 60s)"
        
        # Verify JSON structure
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        assert "text_regions" in ocr_data, "Should contain text_regions"
        assert "metadata" in ocr_data, "Should contain metadata"
        assert len(ocr_data["text_regions"]) == result.total_texts_found
        assert ocr_data["metadata"]["total_texts_found"] == result.total_texts_found
        assert ocr_data["metadata"]["average_confidence"] == result.average_confidence
    
    def test_ocr_with_full_resolution(
        self, shared_ocr_engine, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test OCR processing with full resolution test image."""
        if "full" not in test_image_paths:
            pytest.skip("Full resolution test image not found")
        
        # Arrange
        image_path = test_image_paths["full"]
        
        # Act
        result = shared_ocr_engine.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output JSON should be created"
        assert result.total_texts_found > 0, "Should detect text in full resolution image"
        assert result.average_confidence > 0.0, "Average confidence should be positive"
    
    def test_ocr_output_json_structure(
        self, shared_ocr_engine, test_image_paths, cleanup_integration_outputs
    ):
        """Test that OCR output JSON has correct structure."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        result = shared_ocr_engine.process(image_path)
        
        # Assert
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Verify top-level structure
        assert "text_regions" in ocr_data
        assert "metadata" in ocr_data
        assert isinstance(ocr_data["text_regions"], list)
        
        # Verify metadata structure
        metadata = ocr_data["metadata"]
        assert "input_file" in metadata
        assert "output_file" in metadata
        assert "total_texts_found" in metadata
        assert "average_confidence" in metadata
        assert "processing_time_seconds" in metadata
        assert "image_width" in metadata
        assert "image_height" in metadata
        
        # Verify text region structure (if any found)
        if ocr_data["text_regions"]:
            region = ocr_data["text_regions"][0]
            assert "text" in region
            assert "confidence" in region
            assert "bbox" in region
            assert "center_x" in region
            assert "center_y" in region
            assert isinstance(region["confidence"], (int, float))
            assert 0.0 <= region["confidence"] <= 1.0
    
    def test_ocr_confidence_filtering(
        self, shared_ocr_engine, test_image_paths, cleanup_integration_outputs
    ):
        """Test that OCR respects confidence threshold."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        result = shared_ocr_engine.process(image_path)
        
        # Assert
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # All text regions should meet confidence threshold
        threshold = shared_ocr_engine._settings.ocr_confidence_threshold
        for region in ocr_data["text_regions"]:
            assert region["confidence"] >= threshold, \
                f"Text region confidence {region['confidence']} below threshold {threshold}"
    
    def test_ocr_context_manager(
        self, integration_settings, integration_logger, test_image_paths, cleanup_integration_outputs
    ):
        """Test OCR engine as context manager for resource cleanup."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        with OCREngine(integration_settings, integration_logger) as engine:
            result = engine.process(image_path)
        
        # Assert
        assert result.output_path.exists(), "Output should be created"
        assert result.total_texts_found > 0, "Should detect text"
        # Context manager should handle cleanup automatically
    
    def test_ocr_performance_metrics(
        self, shared_ocr_engine, test_image_paths, cleanup_integration_outputs
    ):
        """Test that OCR performance metrics are logged."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        result = shared_ocr_engine.process(image_path)
        
        # Assert
        assert result.duration_seconds > 0, "Duration should be measured"
        assert result.total_texts_found >= 0, "Text count should be non-negative"
        assert result.average_confidence >= 0.0, "Average confidence should be non-negative"
        assert result.low_confidence_count >= 0, "Low confidence count should be non-negative"
    
    def test_ocr_memory_cleanup(
        self, shared_ocr_engine, test_image_paths, cleanup_integration_outputs
    ):
        """Test that OCR cleans up memory after processing."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        
        # Act
        result1 = shared_ocr_engine.process(image_path)
        result2 = shared_ocr_engine.process(image_path)  # Process again
        
        # Assert
        # Both should succeed without memory issues
        assert result1.output_path.exists()
        assert result2.output_path.exists()
        assert result1.total_texts_found == result2.total_texts_found, \
            "Results should be consistent"
    
    def test_ocr_custom_output_path(
        self, shared_ocr_engine, test_image_paths, temp_output_dir
    ):
        """Test OCR with custom output path."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        custom_output = temp_output_dir / "custom_ocr_output.json"
        
        # Act
        result = shared_ocr_engine.process(image_path, output_path=custom_output)
        
        # Assert
        assert result.output_path == custom_output
        assert custom_output.exists()

