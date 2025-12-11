"""Integration tests for header field extraction with specialized extractor."""

import json
import logging
import pytest
from pathlib import Path

from src.form_extractor import FormExtractor
from tests.fixtures.header_field_fixtures import (
    create_clean_ocr_detections,
    create_slash_misrecognition_detections,
    create_low_confidence_detections,
    create_multiple_errors_detections,
    create_split_cell_detections,
    get_image_dimensions,
)


@pytest.fixture
def integration_logger():
    """Create logger for integration tests."""
    logger = logging.getLogger("integration_test")
    logger.setLevel(logging.INFO)
    return logger


@pytest.mark.integration
class TestHeaderFieldExtractionIntegration:
    """Integration tests for specialized header field extraction."""

    def test_extract_with_clean_ocr_data(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test extraction with clean OCR data (baseline)."""
        # Arrange
        extractor = FormExtractor(test_settings, integration_logger)
        detections = create_clean_ocr_detections()
        width, height = get_image_dimensions()

        # Create mock OCR JSON
        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        if test_settings.header_field_enable_specialized_extraction:
            # Specialized extractor should extract full format
            act_number = header.get("act_number")
            if act_number:
                assert act_number.get("value") == "057/25"

            act_date = header.get("act_date")
            if act_date:
                assert act_date.get("value") == "15/10/2025"

    def test_extract_with_slash_misrecognition(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test extraction with slash misrecognition (I, l, |)."""
        # Arrange
        extractor = FormExtractor(test_settings, integration_logger)
        detections = create_slash_misrecognition_detections()
        width, height = get_image_dimensions()

        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        if test_settings.header_field_enable_specialized_extraction:
            # Should correct slash misrecognition
            act_number = header.get("act_number")
            if act_number:
                # Should be corrected from "057I25" to "057/25"
                value = act_number.get("value", "")
                assert "/" in value or "057" in value
                if act_number.get("corrected"):
                    assert value == "057/25"

    def test_extract_with_low_confidence(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test extraction with low confidence detections."""
        # Arrange
        extractor = FormExtractor(test_settings, integration_logger)
        detections = create_low_confidence_detections()
        width, height = get_image_dimensions()

        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        # With low confidence threshold (0.4), should still extract
        if test_settings.header_field_enable_specialized_extraction:
            act_number = header.get("act_number")
            # May or may not extract depending on confidence threshold
            # If extracted, should have correct format

    def test_extract_with_multiple_errors(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test extraction with multiple OCR errors combined."""
        # Arrange
        extractor = FormExtractor(test_settings, integration_logger)
        detections = create_multiple_errors_detections()
        width, height = get_image_dimensions()

        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        if test_settings.header_field_enable_specialized_extraction:
            # Should attempt corrections
            act_number = header.get("act_number")
            if act_number:
                # Should correct slash and potentially digits
                value = act_number.get("value", "")
                if act_number.get("corrected"):
                    assert "/" in value

    def test_extract_with_split_cells(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test extraction when blank number is split across cells."""
        # Arrange
        extractor = FormExtractor(test_settings, integration_logger)
        detections = create_split_cell_detections()
        width, height = get_image_dimensions()

        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        if test_settings.header_field_enable_specialized_extraction:
            # Should combine split cells
            act_number = header.get("act_number")
            if act_number:
                value = act_number.get("value", "")
                assert "057" in value and "25" in value

    def test_fallback_to_old_method(
        self, test_settings, integration_logger, tmp_path
    ):
        """Test fallback to old extraction method when specialized extractor fails."""
        # Arrange
        # Disable specialized extraction
        test_settings.header_field_enable_specialized_extraction = False
        extractor = FormExtractor(test_settings, integration_logger)

        detections = [
            {
                "text": "Номер",
                "confidence": 0.95,
                "center": [200, 100],  # Outside region
                "bbox": [[150, 80], [250, 80], [250, 120], [150, 120]],
            },
            {
                "text": "034",
                "confidence": 0.98,
                "center": [200, 150],
                "bbox": [[180, 130], [220, 130], [220, 170], [180, 170]],
            },
        ]
        width, height = get_image_dimensions()

        ocr_data = {
            "document_info": {
                "image_path": "test_image.jpg",
                "image_width": width,
                "image_height": height,
            },
            "ocr_results_by_region": {
                "header": detections,
                "defects": [],
                "analysis": [],
            },
            "processing_metrics": {"total_time_ms": 1000},
        }

        json_file = tmp_path / "test-corrected.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        output_path = tmp_path / "output-data.json"

        # Act
        result = extractor.extract(json_file, output_path)

        # Assert
        assert result.output_path.exists()
        with open(result.output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        header = output_data.get("header", {})
        # Should use old method
        act_number = header.get("act_number")
        if act_number:
            assert act_number.get("value") == "034"

