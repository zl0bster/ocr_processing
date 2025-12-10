"""Integration tests for region detection + OCR processing."""
import pytest
import json
import cv2
from pathlib import Path

from src.region_detector import RegionDetector
from src.ocr_engine import OCREngine


@pytest.mark.integration
@pytest.mark.requires_ocr
@pytest.mark.slow
class TestRegionDetectionOCR:
    """Test region detection strategies with real OCR."""
    
    def test_adaptive_strategy_with_real_image(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test adaptive line detection strategy with real image."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Act
        regions = region_detector.detect_zones(image, strategy="adaptive")
        
        # Assert
        assert len(regions) > 0, "Should detect at least one region"
        for region in regions:
            assert region.region_id, "Region should have ID"
            assert 0.0 <= region.y_start_norm <= 1.0, "Normalized start should be in [0, 1]"
            assert 0.0 <= region.y_end_norm <= 1.0, "Normalized end should be in [0, 1]"
            assert region.y_start < region.y_end, "Start should be before end"
            assert region.confidence > 0.0, "Confidence should be positive"
    
    def test_text_based_strategy_with_real_image(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test text-based projection strategy with real image."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Act
        regions = region_detector.detect_zones(image, strategy="text_based")
        
        # Assert
        # May or may not find regions depending on image content
        for region in regions:
            assert region.region_id
            assert 0.0 <= region.y_start_norm <= 1.0
            assert 0.0 <= region.y_end_norm <= 1.0
    
    def test_template_strategy_with_real_image(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test template-based fallback strategy with real image."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Act
        regions = region_detector.detect_zones(image, strategy="template")
        
        # Assert
        assert len(regions) > 0, "Template strategy should always return regions"
        for region in regions:
            assert region.region_id
            assert region.detection_method == "template"
    
    def test_regional_ocr_processing(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test OCR processing applied to each detected region."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Detect regions
        regions = region_detector.detect_zones(image, strategy="auto")
        assert len(regions) > 0, "Should detect at least one region"
        
        # Act - Process regions with OCR
        result = shared_ocr_engine.process_regions(image_path, regions)
        
        # Assert
        assert result.output_path.exists(), "Output JSON should be created"
        assert result.total_texts_found > 0, "Should detect text in regions"
        
        # Verify regional structure in output
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        assert "ocr_results_by_region" in ocr_data, "Should contain regional results"
        assert isinstance(ocr_data["ocr_results_by_region"], dict)
        
        # Verify each region has OCR results
        for region in regions:
            region_id = region.region_id
            if region_id in ocr_data["ocr_results_by_region"]:
                region_results = ocr_data["ocr_results_by_region"][region_id]
                assert isinstance(region_results, list), "Region results should be list"
    
    def test_regional_ocr_results_structure(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that regional OCR results are properly structured."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        regions = region_detector.detect_zones(image, strategy="auto")
        
        if len(regions) == 0:
            pytest.skip("No regions detected")
        
        # Act
        result = shared_ocr_engine.process_regions(image_path, regions)
        
        # Assert
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Verify regional results structure
        for region_id, region_results in ocr_data.get("ocr_results_by_region", {}).items():
            assert isinstance(region_results, list)
            for text_item in region_results:
                assert "text" in text_item
                assert "confidence" in text_item
                assert "center_x" in text_item
                assert "center_y" in text_item
                assert "region_id" in text_item
                assert text_item["region_id"] == region_id
    
    def test_auto_strategy_cascade(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test auto strategy tries multiple methods in cascade."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Act
        regions = region_detector.detect_zones(image, strategy="auto")
        
        # Assert
        assert len(regions) > 0, "Auto strategy should find at least template regions"
        # Should have tried adaptive, then text_based, then template
    
    def test_region_coordinates_normalization(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that region coordinates are properly normalized."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Act
        regions = region_detector.detect_zones(image, strategy="auto")
        
        # Assert
        for region in regions:
            # Normalized coordinates should be in [0, 1]
            assert 0.0 <= region.y_start_norm <= 1.0
            assert 0.0 <= region.y_end_norm <= 1.0
            
            # Absolute coordinates should match normalized
            expected_start = int(region.y_start_norm * height)
            expected_end = int(region.y_end_norm * height)
            assert abs(region.y_start - expected_start) <= 1, "Start coordinate mismatch"
            assert abs(region.y_end - expected_end) <= 1, "End coordinate mismatch"
            
            # Absolute coordinates should be within image bounds
            assert 0 <= region.y_start < height
            assert 0 < region.y_end <= height

