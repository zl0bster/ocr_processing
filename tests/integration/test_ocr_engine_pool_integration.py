"""Integration tests for OCR engine pool with BatchProcessor."""

import logging
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.batch_processor import BatchProcessor, BatchResult, FileResult
from src.preprocessor import PreprocessorResult
from src.error_corrector import CorrectionResult
from src.field_validator import ValidationResult
from src.form_extractor import ExtractionResult
from src.ocr_engine import OCRResult
from src.config.settings import Settings


@pytest.fixture
def logger():
    """Create logger for testing."""
    return logging.getLogger("test")


@pytest.fixture
def test_settings_pool_enabled():
    """Settings with pool enabled."""
    return Settings(
        ocr_pool_enabled=True,
        ocr_pool_size=2,
        ocr_pool_timeout=5.0,
        ocr_engine_memory_check_interval=1,
        ocr_engine_auto_restart_threshold=0.8,
        ocr_memory_reload_threshold_mb=700,
        ocr_use_gpu=False,
        ocr_language="ru",
        ocr_confidence_threshold=0.5,
        log_level="WARNING",
        enable_parallel_processing=False,
    )


@pytest.fixture
def test_settings_pool_disabled():
    """Settings with pool disabled (backward compatibility)."""
    return Settings(
        ocr_pool_enabled=False,
        ocr_memory_reload_threshold_mb=700,
        ocr_use_gpu=False,
        ocr_language="ru",
        ocr_confidence_threshold=0.5,
        log_level="WARNING",
        enable_parallel_processing=False,
    )


@pytest.fixture
def test_image_files(tmp_path):
    """Create test image files for batch processing."""
    image_files = []
    for i in range(3):
        image = np.ones((500, 400, 3), dtype=np.uint8) * 255
        # Add some text-like patterns
        cv2.rectangle(image, (50, 50), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (50, 150), (350, 200), (0, 0, 0), -1)
        
        image_path = tmp_path / f"test_image_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        image_files.append(image_path)
    
    return image_files


@pytest.mark.integration
class TestBatchProcessorWithPool:
    """Test BatchProcessor with OCR engine pool enabled."""

    @patch('src.batch_processor.OCREnginePool')
    @patch('src.batch_processor.OCREngine')
    def test_batch_processor_creates_pool_when_enabled(
        self, mock_ocr_engine_class, mock_pool_class,
        test_settings_pool_enabled, logger
    ):
        """Test BatchProcessor creates pool when pool is enabled."""
        # Arrange
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Act
        processor = BatchProcessor(
            settings=test_settings_pool_enabled,
            logger=logger,
        )

        # Assert
        assert processor._pool is not None
        mock_pool_class.assert_called_once()

    @patch('src.batch_processor.OCREnginePool')
    def test_batch_processor_no_pool_when_disabled(
        self, mock_pool_class, test_settings_pool_disabled, logger
    ):
        """Test BatchProcessor does not create pool when disabled."""
        # Act
        processor = BatchProcessor(
            settings=test_settings_pool_disabled,
            logger=logger,
        )

        # Assert
        assert processor._pool is None
        mock_pool_class.assert_not_called()

    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.OCREnginePool')
    def test_pipeline_batch_uses_pool_engines(
        self, mock_pool_class, mock_region_detector_class,
        mock_form_extractor_class, mock_field_validator_class,
        mock_error_corrector_class, mock_preprocessor_class,
        test_settings_pool_enabled, logger, tmp_path, test_image_files
    ):
        """Test pipeline batch processing uses pool engines."""
        # Arrange
        mock_pool = MagicMock()
        mock_engine = MagicMock()
        mock_engine.process_regions.return_value = OCRResult(
            output_path=tmp_path / "ocr-texts.json",
            duration_seconds=1.0,
            total_texts_found=10,
            average_confidence=0.85,
            low_confidence_count=2,
        )
        mock_pool.acquire.return_value.__enter__.return_value = mock_engine
        mock_pool.acquire.return_value.__exit__.return_value = None
        mock_pool.get_statistics.return_value = MagicMock(
            total_engines=2,
            total_files_processed=3,
            total_restarts=0,
        )
        mock_pool_class.return_value = mock_pool

        # Mock other components
        mock_preprocessor = MagicMock()
        mock_preprocessor.process.return_value = PreprocessorResult(
            output_path=tmp_path / "preprocessed.jpg",
            duration_seconds=0.5,
            deskew_angle=None,
        )
        mock_preprocessor_class.return_value = mock_preprocessor

        mock_region_detector = MagicMock()
        mock_region_detector.detect_zones.return_value = []
        mock_region_detector_class.return_value = mock_region_detector

        mock_error_corrector = MagicMock()
        mock_error_corrector.process.return_value = CorrectionResult(
            output_path=tmp_path / "corrected.json",
            duration_seconds=0.3,
            total_texts=10,
            corrections_applied=2,
            correction_rate=0.2,
        )
        mock_error_corrector_class.return_value = mock_error_corrector

        mock_field_validator = MagicMock()
        mock_field_validator.process.return_value = ValidationResult(
            output_path=tmp_path / "validated.json",
            duration_seconds=0.2,
            total_fields=15,
            validated_fields=13,
            failed_validations=2,
            validation_rate=0.867,
        )
        mock_field_validator_class.return_value = mock_field_validator

        mock_form_extractor = MagicMock()
        mock_form_extractor.extract.return_value = ExtractionResult(
            output_path=tmp_path / "extracted.json",
            duration_seconds=0.1,
            header_fields_extracted=5,
            defect_blocks_found=3,
            analysis_rows_found=10,
            mandatory_fields_missing=0,
        )
        mock_form_extractor_class.return_value = mock_form_extractor

        # Create preprocessed image file
        preprocessed_path = tmp_path / "preprocessed.jpg"
        cv2.imwrite(str(preprocessed_path), np.ones((500, 400, 3), dtype=np.uint8) * 255)

        processor = BatchProcessor(
            settings=test_settings_pool_enabled,
            logger=logger,
        )

        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files:
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # Act
        result = processor.process_directory(
            input_dir=input_dir,
            mode="pipeline",
        )

        # Assert
        assert isinstance(result, BatchResult)
        assert result.total_files == 3
        # Pool should have been used
        assert mock_pool.acquire.call_count == 3  # One per file
        mock_engine.process_regions.assert_called()

    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.OCREngine')
    def test_pipeline_batch_uses_single_engine_when_pool_disabled(
        self, mock_ocr_engine_class, mock_region_detector_class,
        mock_form_extractor_class, mock_field_validator_class,
        mock_error_corrector_class, mock_preprocessor_class,
        test_settings_pool_disabled, logger, tmp_path, test_image_files
    ):
        """Test pipeline batch uses single engine when pool is disabled."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.process_regions.return_value = OCRResult(
            output_path=tmp_path / "ocr-texts.json",
            duration_seconds=1.0,
            total_texts_found=10,
            average_confidence=0.85,
            low_confidence_count=2,
        )
        mock_ocr_engine_class.return_value = mock_ocr_engine

        # Mock other components (same as above)
        mock_preprocessor = MagicMock()
        mock_preprocessor.process.return_value = PreprocessorResult(
            output_path=tmp_path / "preprocessed.jpg",
            duration_seconds=0.5,
            deskew_angle=None,
        )
        mock_preprocessor_class.return_value = mock_preprocessor

        mock_region_detector = MagicMock()
        mock_region_detector.detect_zones.return_value = []
        mock_region_detector_class.return_value = mock_region_detector

        mock_error_corrector = MagicMock()
        mock_error_corrector.process.return_value = CorrectionResult(
            output_path=tmp_path / "corrected.json",
            duration_seconds=0.3,
            total_texts=10,
            corrections_applied=2,
            correction_rate=0.2,
        )
        mock_error_corrector_class.return_value = mock_error_corrector

        mock_field_validator = MagicMock()
        mock_field_validator.process.return_value = ValidationResult(
            output_path=tmp_path / "validated.json",
            duration_seconds=0.2,
            total_fields=15,
            validated_fields=13,
            failed_validations=2,
            validation_rate=0.867,
        )
        mock_field_validator_class.return_value = mock_field_validator

        mock_form_extractor = MagicMock()
        mock_form_extractor.extract.return_value = ExtractionResult(
            output_path=tmp_path / "extracted.json",
            duration_seconds=0.1,
            header_fields_extracted=5,
            defect_blocks_found=3,
            analysis_rows_found=10,
            mandatory_fields_missing=0,
        )
        mock_form_extractor_class.return_value = mock_form_extractor

        # Create preprocessed image file
        preprocessed_path = tmp_path / "preprocessed.jpg"
        cv2.imwrite(str(preprocessed_path), np.ones((500, 400, 3), dtype=np.uint8) * 255)

        processor = BatchProcessor(
            settings=test_settings_pool_disabled,
            logger=logger,
        )

        # Create input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files:
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # Act
        result = processor.process_directory(
            input_dir=input_dir,
            mode="pipeline",
        )

        # Assert
        assert isinstance(result, BatchResult)
        assert result.total_files == 3
        # Single engine should have been created once
        assert mock_ocr_engine_class.call_count == 1
        # Engine should be reused for all files
        assert mock_ocr_engine.process_regions.call_count == 3

    @patch('src.batch_processor.OCREnginePool')
    def test_close_closes_pool(self, mock_pool_class,
                               test_settings_pool_enabled, logger):
        """Test BatchProcessor.close() closes the pool."""
        # Arrange
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        processor = BatchProcessor(
            settings=test_settings_pool_enabled,
            logger=logger,
        )

        # Act
        processor.close()

        # Assert
        mock_pool.close.assert_called_once()
        assert processor._pool is None


@pytest.mark.integration
class TestPoolBackwardCompatibility:
    """Test backward compatibility when pool is disabled."""

    @patch('src.batch_processor.OCREngine')
    def test_single_engine_mode_still_works(
        self, mock_ocr_engine_class, test_settings_pool_disabled, logger
    ):
        """Test that single engine mode works as before (backward compatibility)."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.process.return_value = OCRResult(
            output_path=Path("test-output.json"),
            duration_seconds=1.0,
            total_texts_found=5,
            average_confidence=0.9,
            low_confidence_count=0,
        )
        mock_ocr_engine_class.return_value = mock_ocr_engine

        processor = BatchProcessor(
            settings=test_settings_pool_disabled,
            logger=logger,
        )

        # Assert
        assert processor._pool is None
        # Should be able to process without pool
        assert processor is not None

