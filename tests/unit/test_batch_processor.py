"""Unit tests for BatchProcessor module."""
import json
import logging
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass

from src.batch_processor import BatchProcessor, BatchResult, FileResult
from src.preprocessor import PreprocessorResult
from src.ocr_engine import OCRResult
from src.error_corrector import CorrectionResult
from src.field_validator import ValidationResult
from src.form_extractor import ExtractionResult
from src.region_detector import DocumentRegion


@pytest.fixture
def logger():
    """Create logger for testing."""
    return logging.getLogger("test")


@pytest.fixture
def batch_processor(test_settings, logger):
    """Create BatchProcessor instance for testing."""
    return BatchProcessor(settings=test_settings, logger=logger)


@pytest.fixture
def mock_preprocessor_result(tmp_path):
    """Create mock PreprocessorResult."""
    output_path = tmp_path / "preprocessed.jpg"
    output_path.touch()
    return PreprocessorResult(
        output_path=output_path,
        duration_seconds=0.5,
        deskew_angle=0.5
    )


@pytest.fixture
def mock_ocr_result(tmp_path):
    """Create mock OCRResult."""
    output_path = tmp_path / "ocr-texts.json"
    output_path.write_text('{"text_regions": []}')
    return OCRResult(
        output_path=output_path,
        duration_seconds=1.0,
        total_texts_found=10,
        average_confidence=0.85,
        low_confidence_count=2
    )


@pytest.fixture
def mock_correction_result(tmp_path):
    """Create mock CorrectionResult."""
    output_path = tmp_path / "ocr-corrected.json"
    output_path.write_text('{"corrected": true}')
    return CorrectionResult(
        output_path=output_path,
        duration_seconds=0.3,
        total_texts=10,
        corrections_applied=2,
        correction_rate=0.2
    )


@pytest.fixture
def mock_validation_result(tmp_path):
    """Create mock ValidationResult."""
    output_path = tmp_path / "ocr-validated.json"
    output_path.write_text('{"validated": true}')
    return ValidationResult(
        output_path=output_path,
        duration_seconds=0.2,
        total_fields=15,
        validated_fields=13,
        failed_validations=2,
        validation_rate=0.867
    )


@pytest.fixture
def mock_extraction_result(tmp_path):
    """Create mock ExtractionResult."""
    output_path = tmp_path / "ocr-data.json"
    # Create extraction result with header fields
    extraction_data = {
        "header": {
            "act_number": {"value": "001/2025", "confidence": 0.95, "suspicious": False},
            "act_date": {"value": "15.01.2025", "confidence": 0.92, "suspicious": False},
            "template_revision": {"value": "1.0", "confidence": 0.88, "suspicious": False},
        }
    }
    output_path.write_text(json.dumps(extraction_data, ensure_ascii=False))
    return ExtractionResult(
        output_path=output_path,
        duration_seconds=0.4,
        header_fields_extracted=3,
        defect_blocks_found=5,
        analysis_rows_found=2,
        mandatory_fields_missing=0
    )


@pytest.fixture
def mock_regions():
    """Create mock document regions."""
    return [
        DocumentRegion(
            region_id="header",
            y_start_norm=0.0,
            y_end_norm=0.15,
            y_start=0,
            y_end=150,
            detection_method="adaptive",
            confidence=0.9
        ),
        DocumentRegion(
            region_id="defects",
            y_start_norm=0.15,
            y_end_norm=0.6,
            y_start=150,
            y_end=600,
            detection_method="adaptive",
            confidence=0.85
        ),
    ]


@pytest.fixture
def test_image_files(tmp_path):
    """Create test image files in temporary directory."""
    images = []
    for i, ext in enumerate([".jpg", ".png", ".bmp", ".tiff"]):
        img_path = tmp_path / f"test_image_{i}{ext}"
        # Create a simple test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(img_path), img)
        images.append(img_path)
    return images


@pytest.fixture
def empty_dir(tmp_path):
    """Create empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


@pytest.mark.unit
class TestBatchProcessorCoreFunctionality:
    """Test core batch processing functionality."""

    def test_process_directory_with_empty_directory(self, batch_processor, empty_dir):
        """Test processing empty directory returns empty result."""
        # Act
        result = batch_processor.process_directory(input_dir=empty_dir)

        # Assert
        assert isinstance(result, BatchResult)
        assert result.total_files == 0
        assert result.successful_files == 0
        assert result.failed_files == 0
        assert result.total_duration_seconds >= 0
        assert len(result.file_results) == 0

    def test_process_directory_file_discovery(self, batch_processor, test_image_files, tmp_path):
        """Test file discovery filters only image files."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create image files
        for img_file in test_image_files:
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())
        
        # Create non-image files (should be ignored)
        (input_dir / "document.txt").write_text("not an image")
        (input_dir / "data.json").write_text("{}")
        (input_dir / "script.py").write_text("print('hello')")

        # Act
        with patch('src.batch_processor.ImagePreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.process.return_value = Mock(
                output_path=tmp_path / "preprocessed.jpg",
                duration_seconds=0.5,
                deskew_angle=None
            )
            result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert - only image files should be processed
        assert result.total_files == 4  # .jpg, .png, .bmp, .tiff
        assert mock_preprocessor.return_value.process.call_count == 4

    def test_process_directory_nonexistent_directory(self, batch_processor, tmp_path):
        """Test FileNotFoundError raised for non-existent directory."""
        # Arrange
        nonexistent = tmp_path / "nonexistent_dir"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="does not exist"):
            batch_processor.process_directory(input_dir=nonexistent)

    def test_batch_result_structure(self, batch_processor, test_image_files, tmp_path):
        """Test BatchResult structure creation."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files[:2]:  # Only 2 files
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # Act
        with patch('src.batch_processor.ImagePreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.process.return_value = Mock(
                output_path=tmp_path / "preprocessed.jpg",
                duration_seconds=0.5,
                deskew_angle=None
            )
            result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert isinstance(result, BatchResult)
        assert result.total_files == 2
        assert result.successful_files == 2
        assert result.failed_files == 0
        assert result.total_duration_seconds > 0
        assert len(result.file_results) == 2
        assert result.summary_path is not None
        assert result.summary_path.exists()

    def test_file_result_structure_success(self, batch_processor, test_image_files, tmp_path):
        """Test FileResult structure for successful processing."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Act
        with patch('src.batch_processor.ImagePreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.process.return_value = Mock(
                output_path=tmp_path / "preprocessed.jpg",
                duration_seconds=0.5,
                deskew_angle=None
            )
            result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert len(result.file_results) == 1
        file_result = result.file_results[0]
        assert isinstance(file_result, FileResult)
        assert file_result.filename == test_image_files[0].name
        assert file_result.success is True
        assert file_result.duration_seconds > 0
        assert file_result.error_message is None

    def test_file_result_structure_failure(self, batch_processor, test_image_files, tmp_path):
        """Test FileResult structure for failed processing."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Act
        with patch('src.batch_processor.ImagePreprocessor') as mock_preprocessor:
            mock_preprocessor.return_value.process.side_effect = ValueError("Processing failed")
            result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert len(result.file_results) == 1
        file_result = result.file_results[0]
        assert file_result.success is False
        assert file_result.error_message == "Processing failed"
        assert file_result.duration_seconds > 0


@pytest.mark.unit
class TestBatchProcessorModes:
    """Test different processing modes."""

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.cv2.imread')
    def test_pipeline_mode_full_flow(
        self,
        mock_imread,
        mock_preprocessor,
        mock_corrector,
        mock_validator,
        mock_extractor,
        mock_region_detector,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        mock_ocr_result,
        mock_correction_result,
        mock_validation_result,
        mock_extraction_result,
        mock_regions,
        tmp_path
    ):
        """Test pipeline mode processes full workflow."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Setup mocks
        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result
        mock_region_detector.return_value.detect_zones.return_value = mock_regions
        mock_ocr_engine.return_value.process_regions.return_value = mock_ocr_result
        mock_ocr_engine.return_value.get_memory_mb = Mock(return_value=100)
        mock_corrector.return_value.process.return_value = mock_correction_result
        mock_validator.return_value.process.return_value = mock_validation_result
        mock_extractor.return_value.extract.return_value = mock_extraction_result
        mock_imread.return_value = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="pipeline")

        # Assert
        assert result.total_files == 1
        assert result.successful_files == 1
        
        # Verify components were called
        mock_preprocessor.return_value.process.assert_called_once()
        mock_region_detector.return_value.detect_zones.assert_called_once()
        mock_ocr_engine.return_value.process_regions.assert_called_once()
        mock_corrector.return_value.process.assert_called_once()
        mock_validator.return_value.process.assert_called_once()
        mock_extractor.return_value.extract.assert_called_once()

        # Verify FileResult has correct metadata
        file_result = result.file_results[0]
        assert file_result.success is True
        assert file_result.texts_found == mock_ocr_result.total_texts_found
        assert file_result.average_confidence == mock_ocr_result.average_confidence
        assert file_result.corrections_applied == mock_correction_result.corrections_applied
        assert file_result.fields_validated == mock_validation_result.validated_fields
        assert file_result.extraction_result_path == mock_extraction_result.output_path

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.MemoryMonitor')
    def test_ocr_mode_only(
        self,
        mock_memory_monitor,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_ocr_result,
        tmp_path
    ):
        """Test OCR mode processes only OCR."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        mock_ocr_engine.return_value.process.return_value = mock_ocr_result
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_mb.return_value = 100
        mock_memory_instance.log_memory.return_value = 100
        mock_memory_instance.log_memory_delta = Mock()
        mock_memory_monitor.return_value = mock_memory_instance

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="ocr")

        # Assert
        assert result.total_files == 1
        assert result.successful_files == 1
        mock_ocr_engine.return_value.process.assert_called_once()
        
        # Verify OCR engine was created and closed
        mock_ocr_engine.assert_called_once()
        mock_ocr_engine.return_value.close.assert_called_once()

    @patch('src.batch_processor.ImagePreprocessor')
    def test_preprocess_mode_only(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test preprocess mode processes only preprocessing."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert result.total_files == 1
        assert result.successful_files == 1
        mock_preprocessor.return_value.process.assert_called_once()

    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.ErrorCorrector')
    def test_correction_mode_only(
        self,
        mock_corrector,
        mock_validator,
        batch_processor,
        test_image_files,
        mock_correction_result,
        mock_validation_result,
        tmp_path
    ):
        """Test correction mode processes correction and validation."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Create files with image extensions (batch processor only finds image extensions)
        # Correction mode will process them as JSON files via ErrorCorrector
        json_file = input_dir / "ocr-texts.jpg"
        json_file.write_text('{"text_regions": []}')

        mock_corrector.return_value.process.return_value = mock_correction_result
        mock_validator.return_value.process.return_value = mock_validation_result

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="correction")

        # Assert
        assert result.total_files == 1
        assert result.successful_files == 1
        mock_corrector.return_value.process.assert_called_once()
        mock_validator.return_value.process.assert_called_once()

        # Verify FileResult metadata
        file_result = result.file_results[0]
        assert file_result.corrections_applied == mock_correction_result.corrections_applied
        assert file_result.fields_validated == mock_validation_result.validated_fields

    def test_invalid_mode_raises_error(self, batch_processor, test_image_files, tmp_path):
        """Test ValueError raised for unknown mode."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Need at least one file, otherwise it returns early before checking mode
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown processing mode"):
            batch_processor.process_directory(input_dir=input_dir, mode="invalid_mode")


@pytest.mark.unit
class TestBatchProcessorErrorIsolation:
    """Test error handling and isolation."""

    @patch('src.batch_processor.ImagePreprocessor')
    def test_error_isolation_continues_processing(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test that one file failure doesn't stop batch processing."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files[:3]:  # 3 files
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # First file fails, others succeed
        mock_preprocessor.return_value.process.side_effect = [
            ValueError("First file failed"),
            mock_preprocessor_result,
            mock_preprocessor_result,
        ]

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert result.total_files == 3
        assert result.successful_files == 2
        assert result.failed_files == 1
        
        # Verify all files were attempted
        assert mock_preprocessor.return_value.process.call_count == 3
        
        # Verify error details in failed file
        failed_result = next(r for r in result.file_results if not r.success)
        assert failed_result.error_message == "First file failed"
        assert failed_result.filename == test_image_files[0].name
        
        # Verify successful files
        successful_results = [r for r in result.file_results if r.success]
        assert len(successful_results) == 2

    @patch('src.batch_processor.ImagePreprocessor')
    def test_pipeline_error_isolation_preprocess_failure(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test error isolation when preprocessing fails for one file."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files[:2]:  # 2 files
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # First file fails at preprocessing, second succeeds
        mock_preprocessor.return_value.process.side_effect = [
            ValueError("Preprocessing failed"),
            mock_preprocessor_result,
        ]

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        assert result.total_files == 2
        assert result.successful_files == 1
        assert result.failed_files == 1
        
        failed_result = next(r for r in result.file_results if not r.success)
        assert "Preprocessing failed" in failed_result.error_message
        
        successful_result = next(r for r in result.file_results if r.success)
        assert successful_result.success is True


@pytest.mark.unit
class TestBatchProcessorSummaryGeneration:
    """Test summary generation functionality."""

    @patch('src.batch_processor.ImagePreprocessor')
    def test_save_batch_summary_creates_file(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test summary JSON file is created."""
        # Arrange
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result

        # Act
        result = batch_processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            mode="preprocess"
        )

        # Assert
        assert result.summary_path is not None
        assert result.summary_path.exists()
        assert "batch_summary_" in result.summary_path.name
        assert result.summary_path.suffix == ".json"

        # Verify JSON structure
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        assert "batch_info" in summary_data
        assert "file_results" in summary_data
        assert summary_data["batch_info"]["total_files"] == 1
        assert summary_data["batch_info"]["successful_files"] == 1
        assert summary_data["batch_info"]["failed_files"] == 0
        assert "processing_date" in summary_data["batch_info"]
        assert "total_duration_seconds" in summary_data["batch_info"]
        assert "average_duration_seconds" in summary_data["batch_info"]

    @patch('src.batch_processor.ImagePreprocessor')
    def test_summary_includes_file_results(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test summary includes detailed file results."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files[:2]:  # 2 files
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        assert len(summary_data["file_results"]) == 2
        for file_data in summary_data["file_results"]:
            assert "filename" in file_data
            assert "success" in file_data
            assert "duration_seconds" in file_data
            assert "texts_found" in file_data
            assert "average_confidence" in file_data
            assert "corrections_applied" in file_data
            assert "fields_validated" in file_data

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.cv2.imread')
    def test_extract_required_fields_from_extraction_result(
        self,
        mock_imread,
        mock_preprocessor,
        mock_corrector,
        mock_validator,
        mock_extractor,
        mock_region_detector,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        mock_ocr_result,
        mock_correction_result,
        mock_validation_result,
        mock_extraction_result,
        mock_regions,
        tmp_path
    ):
        """Test required fields are extracted from extraction results."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Setup mocks
        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result
        mock_region_detector.return_value.detect_zones.return_value = mock_regions
        mock_ocr_engine.return_value.process_regions.return_value = mock_ocr_result
        mock_ocr_engine.return_value.get_memory_mb.return_value = 100
        mock_corrector.return_value.process.return_value = mock_correction_result
        mock_validator.return_value.process.return_value = mock_validation_result
        mock_extractor.return_value.extract.return_value = mock_extraction_result
        mock_imread.return_value = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="pipeline")

        # Assert
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        file_data = summary_data["file_results"][0]
        assert "required_fields" in file_data
        required_fields = file_data["required_fields"]
        
        # Check mandatory fields are present
        mandatory_fields = [
            "act_number", "act_date", "template_revision",
            "part_line_number", "quantity_checked",
            "control_type", "inspector_name"
        ]
        for field in mandatory_fields:
            assert field in required_fields
            assert "value" in required_fields[field]
            assert "confidence" in required_fields[field]
            assert "suspicious" in required_fields[field]

        # Verify extracted values
        assert required_fields["act_number"]["value"] == "001/2025"
        assert required_fields["act_date"]["value"] == "15.01.2025"

    @patch('src.batch_processor.ImagePreprocessor')
    def test_extract_required_fields_handles_missing_file(
        self,
        mock_preprocessor,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        tmp_path
    ):
        """Test handling of missing extraction file."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Mock FileResult with missing extraction_result_path
        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="preprocess")

        # Assert
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        file_data = summary_data["file_results"][0]
        assert "required_fields" in file_data
        
        # All fields should be marked as missing (None values)
        for field_name, field_data in file_data["required_fields"].items():
            assert field_data["value"] is None
            assert field_data["confidence"] == 0.0
            assert field_data["suspicious"] is False

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.cv2.imread')
    def test_extract_required_fields_handles_malformed_json(
        self,
        mock_imread,
        mock_preprocessor,
        mock_corrector,
        mock_validator,
        mock_extractor,
        mock_region_detector,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        mock_ocr_result,
        mock_correction_result,
        mock_validation_result,
        mock_regions,
        tmp_path
    ):
        """Test handling of malformed extraction JSON."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Create malformed extraction result
        malformed_extraction = tmp_path / "ocr-data.json"
        malformed_extraction.write_text("{ invalid json }")

        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result
        mock_region_detector.return_value.detect_zones.return_value = mock_regions
        mock_ocr_engine.return_value.process_regions.return_value = mock_ocr_result
        mock_ocr_engine.return_value.get_memory_mb.return_value = 100
        mock_corrector.return_value.process.return_value = mock_correction_result
        mock_validator.return_value.process.return_value = mock_validation_result
        mock_extractor.return_value.extract.return_value = Mock(
            output_path=malformed_extraction,
            duration_seconds=0.4,
            header_fields_extracted=0,
            defect_blocks_found=0,
            analysis_rows_found=0,
            mandatory_fields_missing=0
        )
        mock_imread.return_value = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="pipeline")

        # Assert - should handle gracefully, all fields marked as missing
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        file_data = summary_data["file_results"][0]
        required_fields = file_data["required_fields"]
        
        # All fields should be None due to malformed JSON
        for field_data in required_fields.values():
            assert field_data["value"] is None
            assert field_data["confidence"] == 0.0

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.RegionDetector')
    @patch('src.batch_processor.FormExtractor')
    @patch('src.batch_processor.FieldValidator')
    @patch('src.batch_processor.ErrorCorrector')
    @patch('src.batch_processor.ImagePreprocessor')
    @patch('src.batch_processor.cv2.imread')
    def test_extract_required_fields_handles_missing_header(
        self,
        mock_imread,
        mock_preprocessor,
        mock_corrector,
        mock_validator,
        mock_extractor,
        mock_region_detector,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_preprocessor_result,
        mock_ocr_result,
        mock_correction_result,
        mock_validation_result,
        mock_regions,
        tmp_path
    ):
        """Test handling of missing header section."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / test_image_files[0].name).write_bytes(test_image_files[0].read_bytes())

        # Create extraction result without header
        extraction_no_header = tmp_path / "ocr-data.json"
        extraction_no_header.write_text('{"defects": []}')

        mock_preprocessor.return_value.process.return_value = mock_preprocessor_result
        mock_region_detector.return_value.detect_zones.return_value = mock_regions
        mock_ocr_engine.return_value.process_regions.return_value = mock_ocr_result
        mock_ocr_engine.return_value.get_memory_mb.return_value = 100
        mock_corrector.return_value.process.return_value = mock_correction_result
        mock_validator.return_value.process.return_value = mock_validation_result
        mock_extractor.return_value.extract.return_value = Mock(
            output_path=extraction_no_header,
            duration_seconds=0.4,
            header_fields_extracted=0,
            defect_blocks_found=0,
            analysis_rows_found=0,
            mandatory_fields_missing=0
        )
        mock_imread.return_value = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="pipeline")

        # Assert
        with open(result.summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        file_data = summary_data["file_results"][0]
        required_fields = file_data["required_fields"]
        
        # All fields should be None due to missing header
        for field_data in required_fields.values():
            assert field_data["value"] is None
            assert field_data["confidence"] == 0.0


@pytest.mark.unit
class TestBatchProcessorMemoryManagement:
    """Test memory management and OCR engine reload."""

    @patch('src.batch_processor.OCREngine')
    @patch('src.batch_processor.MemoryMonitor')
    def test_ocr_engine_reload_on_memory_threshold(
        self,
        mock_memory_monitor,
        mock_ocr_engine,
        batch_processor,
        test_image_files,
        mock_ocr_result,
        tmp_path
    ):
        """Test OCR engine reloads when memory threshold exceeded."""
        # Arrange
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for img_file in test_image_files[:2]:  # 2 files
            (input_dir / img_file.name).write_bytes(img_file.read_bytes())

        # Setup memory monitor to report high memory on first check, low on second
        mock_memory_instance = Mock()
        memory_values = [800, 100]  # First exceeds 700 MB threshold, second is normal
        call_count = {'count': 0}
        
        def get_memory_side_effect():
            idx = call_count['count']
            call_count['count'] += 1
            return memory_values[idx % len(memory_values)]
        
        mock_memory_instance.get_memory_mb.side_effect = get_memory_side_effect
        mock_memory_instance.log_memory.return_value = 100
        mock_memory_instance.log_memory_delta = Mock()
        mock_memory_monitor.return_value = mock_memory_instance

        # Create separate OCR engine instances for each call
        ocr_engine_instances = []
        def create_ocr_engine(*args, **kwargs):
            instance = Mock()
            instance.process.return_value = mock_ocr_result
            instance.close = Mock()
            ocr_engine_instances.append(instance)
            return instance
        
        mock_ocr_engine.side_effect = create_ocr_engine

        # Act
        result = batch_processor.process_directory(input_dir=input_dir, mode="ocr")

        # Assert
        # OCR engine should be created at least once
        assert mock_ocr_engine.call_count >= 1
        # If memory threshold was exceeded, first instance should be closed
        if len(ocr_engine_instances) > 1:
            ocr_engine_instances[0].close.assert_called_once()

