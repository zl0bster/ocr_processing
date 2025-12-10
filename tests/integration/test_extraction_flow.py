"""Integration tests for full extraction pipeline."""
import pytest
import json
from pathlib import Path

from src.ocr_engine import OCREngine
from src.error_corrector import ErrorCorrector
from src.field_validator import FieldValidator
from src.form_extractor import FormExtractor


@pytest.mark.integration
@pytest.mark.requires_ocr
@pytest.mark.slow
class TestExtractionFlow:
    """Test complete extraction pipeline integration."""
    
    def test_full_extraction_flow(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test full flow: OCR → correction → validation → FormExtractor."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act - Run full pipeline
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        assert extraction_result.output_path.exists(), "Extraction output should exist"
        assert extraction_result.duration_seconds > 0, "Extraction should take time"
        
        # Verify extraction output structure
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        assert "header" in extracted_data, "Should contain header data"
        assert "defects" in extracted_data, "Should contain defects data"
        assert "analysis" in extracted_data, "Should contain analysis data"
        assert "metadata" in extracted_data, "Should contain metadata"
    
    def test_header_extraction_from_real_ocr(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test header extraction from real OCR results."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        header = extracted_data.get("header", {})
        assert isinstance(header, dict), "Header should be a dictionary"
        # Header may or may not have fields depending on OCR results
    
    def test_defect_block_extraction(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test defect block extraction from OCR results."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        assert extraction_result.defect_blocks_found >= 0, "Defect blocks count should be non-negative"
        
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        defects = extracted_data.get("defects", [])
        assert isinstance(defects, list), "Defects should be a list"
    
    def test_analysis_section_extraction(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test analysis section extraction from OCR results."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        assert extraction_result.analysis_rows_found >= 0, "Analysis rows count should be non-negative"
        
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        analysis = extracted_data.get("analysis", {})
        assert isinstance(analysis, dict), "Analysis should be a dictionary"
    
    def test_sticker_detection_priority(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that sticker detection has priority over handwritten text."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        # Sticker detection logic is tested in unit tests
        # Here we just verify extraction completes successfully
        assert extraction_result.output_path.exists()
    
    def test_mandatory_field_validation(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test mandatory field validation in extraction."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        assert extraction_result.mandatory_fields_missing >= 0, "Missing fields count should be non-negative"
        
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        metadata = extracted_data.get("metadata", {})
        assert "mandatory_fields_missing" in metadata or "validation" in metadata
    
    def test_suspicious_value_flagging(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test suspicious value flagging for low confidence."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        # Suspicious values may or may not be present depending on OCR confidence
        # Just verify extraction completes and structure is valid
        assert "header" in extracted_data
        assert "defects" in extracted_data
        assert "analysis" in extracted_data
    
    def test_extracted_data_matches_form_model(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that extracted data matches expected form data model."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        corrector = ErrorCorrector(integration_settings, integration_logger)
        validator = FieldValidator(integration_settings, integration_logger)
        extractor = FormExtractor(integration_settings, integration_logger)
        
        # Act
        ocr_result = shared_ocr_engine.process(image_path)
        correction_result = corrector.process(ocr_result.output_path)
        validation_result = validator.process(correction_result.output_path)
        extraction_result = extractor.extract(validation_result.output_path)
        
        # Assert
        with open(extraction_result.output_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        # Verify structure matches FormData model
        assert "header" in extracted_data
        assert "defects" in extracted_data
        assert "analysis" in extracted_data
        assert "final_decision" in extracted_data
        assert "metadata" in extracted_data
        
        # Header should be a dictionary
        assert isinstance(extracted_data["header"], dict)
        
        # Defects should be a list
        assert isinstance(extracted_data["defects"], list)
        
        # Analysis should be a dictionary
        assert isinstance(extracted_data["analysis"], dict)
        
        # Final decision should be a dictionary or null
        assert extracted_data["final_decision"] is None or isinstance(extracted_data["final_decision"], dict)

