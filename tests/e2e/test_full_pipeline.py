"""E2E tests for full processing pipeline."""
import json
import time
from pathlib import Path

import cv2
import pytest

from src.batch_processor import BatchProcessor
from src.error_corrector import ErrorCorrector
from src.field_validator import FieldValidator
from src.form_extractor import FormExtractor
from src.ocr_engine import OCREngine
from src.preprocessor import ImagePreprocessor
from src.region_detector import RegionDetector


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_ocr
class TestFullPipeline:
    """End-to-end tests for complete processing pipeline."""

    def test_full_pipeline_compressed_image(
        self, test_image_paths, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Test full pipeline with compressed test image (034_compr.jpg)."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_path = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        start_time = time.perf_counter()

        # Step 1: Preprocessing
        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)
        preprocess_result = preprocessor.process(input_path=input_path)

        assert preprocess_result.output_path.exists(), "Preprocessed image should exist"
        assert preprocess_result.duration_seconds > 0, "Processing should take time"

        # Verify preprocessed image is valid
        preprocessed_image = cv2.imread(str(preprocess_result.output_path))
        assert preprocessed_image is not None, "Preprocessed image should be readable"

        # Step 2: Region Detection + OCR
        region_detector = RegionDetector(settings=e2e_settings, logger=e2e_logger)
        regions = region_detector.detect_zones(preprocessed_image)
        assert len(regions) > 0, "Should detect at least one region"

        ocr_engine = OCREngine(settings=e2e_settings, logger=e2e_logger)
        ocr_result = ocr_engine.process_regions(
            image_path=preprocess_result.output_path,
            regions=regions,
        )

        assert ocr_result.output_path.exists(), "OCR results file should exist"
        assert ocr_result.total_texts_found > 0, "Should find some text regions"
        assert ocr_result.average_confidence > 0, "Should have confidence scores"

        # Verify OCR results structure
        with open(ocr_result.output_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        assert "ocr_results_by_region" in ocr_data, "Should have regional OCR results"
        assert "regions_detected" in ocr_data, "Should have detected regions info"

        # Step 3: Error Correction
        error_corrector = ErrorCorrector(settings=e2e_settings, logger=e2e_logger)
        correction_result = error_corrector.process(input_path=ocr_result.output_path)

        assert correction_result.output_path.exists(), "Corrected results file should exist"
        assert correction_result.total_texts > 0, "Should process some texts"

        # Step 4: Field Validation
        field_validator = FieldValidator(settings=e2e_settings, logger=e2e_logger)
        validation_result = field_validator.process(input_path=correction_result.output_path)

        assert validation_result.output_path.exists(), "Validation results file should exist"
        assert validation_result.total_fields > 0, "Should validate some fields"

        # Step 5: Data Extraction
        form_extractor = FormExtractor(settings=e2e_settings, logger=e2e_logger)
        extraction_result = form_extractor.extract(ocr_json_path=validation_result.output_path)

        assert extraction_result.output_path.exists(), "Extraction results file should exist"

        # Verify extracted data structure
        with open(extraction_result.output_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)
        assert "header" in extracted_data, "Should have header data"
        assert "defects" in extracted_data, "Should have defects data"
        assert "analysis" in extracted_data, "Should have analysis data"

        total_time = time.perf_counter() - start_time

        # Verify all intermediate files were created
        assert preprocess_result.output_path.exists()
        assert ocr_result.output_path.exists()
        assert correction_result.output_path.exists()
        assert validation_result.output_path.exists()
        assert extraction_result.output_path.exists()

        # Log performance metrics
        e2e_logger.info("Full pipeline completed in %.3f seconds", total_time)
        e2e_logger.info("Preprocessing: %.3f seconds", preprocess_result.duration_seconds)
        e2e_logger.info("OCR: %.3f seconds (%d texts, %.3f avg confidence)",
                       ocr_result.duration_seconds,
                       ocr_result.total_texts_found,
                       ocr_result.average_confidence)
        e2e_logger.info("Correction: %.3f seconds (%d corrections)",
                       correction_result.duration_seconds,
                       correction_result.corrections_applied)
        e2e_logger.info("Validation: %.3f seconds (%d/%d fields validated)",
                       validation_result.duration_seconds,
                       validation_result.validated_fields,
                       validation_result.total_fields)
        e2e_logger.info("Extraction: %.3f seconds (%d header fields, %d defects, %d analysis)",
                       extraction_result.duration_seconds,
                       extraction_result.header_fields_extracted,
                       extraction_result.defect_blocks_found,
                       extraction_result.analysis_rows_found)

    def test_full_pipeline_full_resolution(
        self, test_image_paths, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Test full pipeline with full resolution test image (034.jpg)."""
        if "full" not in test_image_paths:
            pytest.skip("Full resolution test image not found")

        input_path = test_image_paths["full"]
        output_dir = cleanup_e2e_outputs

        start_time = time.perf_counter()

        # Step 1: Preprocessing
        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)
        preprocess_result = preprocessor.process(input_path=input_path)

        assert preprocess_result.output_path.exists()

        # Step 2: Region Detection + OCR
        preprocessed_image = cv2.imread(str(preprocess_result.output_path))
        assert preprocessed_image is not None

        region_detector = RegionDetector(settings=e2e_settings, logger=e2e_logger)
        regions = region_detector.detect_zones(preprocessed_image)
        assert len(regions) > 0

        ocr_engine = OCREngine(settings=e2e_settings, logger=e2e_logger)
        ocr_result = ocr_engine.process_regions(
            image_path=preprocess_result.output_path,
            regions=regions,
        )

        assert ocr_result.output_path.exists()
        assert ocr_result.total_texts_found > 0

        # Step 3: Error Correction
        error_corrector = ErrorCorrector(settings=e2e_settings, logger=e2e_logger)
        correction_result = error_corrector.process(input_path=ocr_result.output_path)

        assert correction_result.output_path.exists()

        # Step 4: Field Validation
        field_validator = FieldValidator(settings=e2e_settings, logger=e2e_logger)
        validation_result = field_validator.process(input_path=correction_result.output_path)

        assert validation_result.output_path.exists()

        # Step 5: Data Extraction
        form_extractor = FormExtractor(settings=e2e_settings, logger=e2e_logger)
        extraction_result = form_extractor.extract(ocr_json_path=validation_result.output_path)

        assert extraction_result.output_path.exists()

        total_time = time.perf_counter() - start_time

        # Verify extracted data has meaningful content
        with open(extraction_result.output_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        # Verify structure
        assert "header" in extracted_data
        assert "defects" in extracted_data
        assert "analysis" in extracted_data

        e2e_logger.info("Full resolution pipeline completed in %.3f seconds", total_time)

    def test_full_pipeline_output_files_structure(
        self, test_image_paths, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Verify all output files are created with correct naming and structure."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_path = test_image_paths["compressed"]

        # Run full pipeline
        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)
        preprocess_result = preprocessor.process(input_path=input_path)

        preprocessed_image = cv2.imread(str(preprocess_result.output_path))
        region_detector = RegionDetector(settings=e2e_settings, logger=e2e_logger)
        regions = region_detector.detect_zones(preprocessed_image)

        ocr_engine = OCREngine(settings=e2e_settings, logger=e2e_logger)
        ocr_result = ocr_engine.process_regions(
            image_path=preprocess_result.output_path,
            regions=regions,
        )

        error_corrector = ErrorCorrector(settings=e2e_settings, logger=e2e_logger)
        correction_result = error_corrector.process(input_path=ocr_result.output_path)

        field_validator = FieldValidator(settings=e2e_settings, logger=e2e_logger)
        validation_result = field_validator.process(input_path=correction_result.output_path)

        form_extractor = FormExtractor(settings=e2e_settings, logger=e2e_logger)
        extraction_result = form_extractor.extract(ocr_json_path=validation_result.output_path)

        # Verify file naming patterns
        assert "-cor" in preprocess_result.output_path.name or preprocess_result.output_path.suffix in {".jpg", ".png"}
        assert ocr_result.output_path.suffix == ".json"
        assert "-corrected" in correction_result.output_path.name
        assert "-validated" in validation_result.output_path.name
        assert "-extracted" in extraction_result.output_path.name

        # Verify JSON files are valid
        for json_path in [ocr_result.output_path, correction_result.output_path,
                         validation_result.output_path, extraction_result.output_path]:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{json_path.name} should contain a dictionary"

        # Verify extracted data has required fields
        with open(extraction_result.output_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        assert isinstance(extracted_data.get("header"), dict)
        assert isinstance(extracted_data.get("defects"), list)
        assert isinstance(extracted_data.get("analysis"), dict)

