"""Integration tests for error correction + field validation flow."""
import pytest
import json
import tempfile
from pathlib import Path

from src.error_corrector import ErrorCorrector
from src.field_validator import FieldValidator
from src.ocr_engine import OCREngine


@pytest.mark.integration
@pytest.mark.slow
class TestCorrectionValidationFlow:
    """Test correction and validation pipeline integration."""
    
    def test_ocr_to_correction_flow(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test OCR → ErrorCorrector integration."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        
        # Act - Run OCR first
        ocr_result = shared_ocr_engine.process(image_path)
        
        # Then apply corrections
        correction_result = corrector.process(ocr_result.output_path)
        
        # Assert
        assert correction_result.output_path.exists(), "Corrected output should exist"
        assert correction_result.total_texts > 0, "Should process texts"
        assert correction_result.corrections_applied >= 0, "Corrections count should be non-negative"
        assert 0.0 <= correction_result.correction_rate <= 1.0, "Correction rate should be in [0, 1]"
        
        # Verify corrected JSON structure
        with open(correction_result.output_path, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        assert "corrected_text_regions" in corrected_data, "Should contain corrected regions"
        assert "corrections_log" in corrected_data, "Should contain corrections log"
        assert "metadata" in corrected_data, "Should contain metadata"
    
    def test_correction_to_validation_flow(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test ErrorCorrector → FieldValidator integration."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        
        # Act - Run OCR → Correction → Validation
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        
        # Assert
        assert validation_result.output_path.exists(), "Validated output should exist"
        assert validation_result.total_fields >= 0, "Should process fields"
        assert validation_result.validated_fields >= 0, "Validated fields count should be non-negative"
        assert validation_result.failed_validations >= 0, "Failed validations count should be non-negative"
        assert 0.0 <= validation_result.validation_rate <= 1.0, "Validation rate should be in [0, 1]"
        
        # Verify validated JSON structure
        with open(validation_result.output_path, 'r', encoding='utf-8') as f:
            validated_data = json.load(f)
        
        assert "validated_text_regions" in validated_data, "Should contain validated regions"
        assert "validation_results" in validated_data, "Should contain validation results"
        assert "metadata" in validated_data, "Should contain metadata"
    
    def test_full_correction_validation_flow(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test full flow: OCR → correction → validation."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        
        # Assert - All steps should succeed
        assert ocr_result.output_path.exists()
        assert correction_result.output_path.exists()
        assert validation_result.output_path.exists()
        
        # Verify data flows through pipeline
        with open(ocr_result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        with open(correction_result.output_path, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        with open(validation_result.output_path, 'r', encoding='utf-8') as f:
            validated_data = json.load(f)
        
        # Text count should be consistent
        ocr_text_count = len(ocr_data.get("text_regions", []))
        corrected_text_count = len(corrected_data.get("corrected_text_regions", []))
        validated_text_count = len(validated_data.get("validated_text_regions", []))
        
        assert ocr_text_count == corrected_text_count, "Text count should be preserved in correction"
        assert corrected_text_count == validated_text_count, "Text count should be preserved in validation"
    
    def test_corrections_applied_from_dictionary(
        self, integration_settings, integration_logger, temp_output_dir
    ):
        """Test that corrections are applied from dictionary."""
        # Arrange - Create mock OCR data with known errors
        mock_ocr_data = {
            "text_regions": [
                {
                    "text": "Homep",  # Should be corrected to "Номер"
                    "confidence": 0.9,
                    "bbox": [[0, 0], [100, 0], [100, 50], [0, 50]],
                    "center_x": 50,
                    "center_y": 25,
                },
                {
                    "text": "PeB",  # Should be corrected to "Рев"
                    "confidence": 0.85,
                    "bbox": [[0, 60], [100, 60], [100, 110], [0, 110]],
                    "center_x": 50,
                    "center_y": 85,
                },
                {
                    "text": "ValidText",  # Should not be corrected
                    "confidence": 0.95,
                    "bbox": [[0, 120], [100, 120], [100, 170], [0, 170]],
                    "center_x": 50,
                    "center_y": 145,
                },
            ],
            "metadata": {
                "input_file": "test.jpg",
                "total_texts_found": 3,
            }
        }
        
        # Create temporary OCR JSON file
        ocr_json = temp_output_dir / "test_ocr.json"
        with open(ocr_json, 'w', encoding='utf-8') as f:
            json.dump(mock_ocr_data, f, ensure_ascii=False, indent=2)
        
        corrector = ErrorCorrector(integration_settings, integration_logger)
        
        # Act
        result = corrector.process(ocr_json)
        
        # Assert
        with open(result.output_path, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        assert result.corrections_applied > 0, "Should apply at least one correction"
        assert len(corrected_data.get("corrections_log", [])) > 0, "Should log corrections"
        
        # Verify corrections in output
        corrected_regions = corrected_data.get("corrected_text_regions", [])
        assert len(corrected_regions) == 3, "Should have all regions"
    
    def test_validation_rules_detect_invalid_formats(
        self, integration_settings, integration_logger, temp_output_dir
    ):
        """Test that validation rules detect invalid formats."""
        # Arrange - Create corrected OCR data with invalid formats
        mock_corrected_data = {
            "corrected_text_regions": [
                {
                    "text": "abc/2025",  # Invalid act_number format
                    "confidence": 0.9,
                    "bbox": [[0, 0], [100, 0], [100, 50], [0, 50]],
                    "center_x": 50,
                    "center_y": 25,
                    "field_type": "act_number",
                },
                {
                    "text": "15/10/2025",  # Invalid date format (should be DD.MM.YYYY)
                    "confidence": 0.85,
                    "bbox": [[0, 60], [100, 60], [100, 110], [0, 110]],
                    "center_x": 50,
                    "center_y": 85,
                    "field_type": "date",
                },
                {
                    "text": "001/2025",  # Valid act_number
                    "confidence": 0.95,
                    "bbox": [[0, 120], [100, 120], [100, 170], [0, 170]],
                    "center_x": 50,
                    "center_y": 145,
                    "field_type": "act_number",
                },
            ],
            "corrections_log": [],
            "metadata": {
                "input_file": "test.jpg",
                "total_texts": 3,
            }
        }
        
        # Create temporary corrected JSON file
        corrected_json = temp_output_dir / "test_corrected.json"
        with open(corrected_json, 'w', encoding='utf-8') as f:
            json.dump(mock_corrected_data, f, ensure_ascii=False, indent=2)
        
        validator = FieldValidator(integration_settings, integration_logger)
        
        # Act
        result = validator.process(corrected_json)
        
        # Assert
        assert result.output_path.exists(), "Validated output should exist"
        assert result.failed_validations > 0, "Should detect invalid formats"
        
        # Verify validation results
        with open(result.output_path, 'r', encoding='utf-8') as f:
            validated_data = json.load(f)
        
        validation_results = validated_data.get("validation_results", {})
        assert len(validation_results) > 0, "Should have validation results"
    
    def test_output_json_contains_metadata(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that output JSON contains correction and validation metadata."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        
        # Assert - Check correction metadata
        with open(correction_result.output_path, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        assert "metadata" in corrected_data
        assert "corrections_applied" in corrected_data["metadata"]
        assert "correction_rate" in corrected_data["metadata"]
        
        # Assert - Check validation metadata
        with open(validation_result.output_path, 'r', encoding='utf-8') as f:
            validated_data = json.load(f)
        
        assert "metadata" in validated_data
        assert "validated_fields" in validated_data["metadata"]
        assert "failed_validations" in validated_data["metadata"]
        assert "validation_rate" in validated_data["metadata"]

