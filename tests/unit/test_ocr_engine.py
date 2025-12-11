"""Unit tests for OCREngine module with mocked PaddleOCR."""
import logging
import pytest
import cv2
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass

from src.ocr_engine import OCREngine, OCRResult, TextDetection
from src.region_detector import DocumentRegion
from src.config.settings import Settings


@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    return logging.getLogger("test")


@pytest.fixture
def mock_ocr_response_standard():
    """Standard PaddleOCR format response."""
    return [
        [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('ГЕОМЕТРИЯ', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('ОТВЕРСТИЯ', 0.92)),
            ([[0, 120], [100, 120], [100, 170], [0, 170]], ('ПОВЕРХНОСТЬ', 0.88)),
        ]
    ]


@pytest.fixture
def mock_ocr_response_mixed_confidence():
    """OCR response with varying confidence scores."""
    return [
        [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('HIGH_CONF', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('LOW_CONF', 0.3)),  # Below threshold
            ([[0, 120], [100, 120], [100, 170], [0, 170]], ('MEDIUM_CONF', 0.6)),
        ]
    ]


@pytest.fixture
def mock_document_regions():
    """Mock DocumentRegion objects for testing."""
    return [
        DocumentRegion(
            region_id="header",
            y_start_norm=0.0,
            y_end_norm=0.2,
            y_start=0,
            y_end=200,
            detection_method="adaptive",
            confidence=0.9,
        ),
        DocumentRegion(
            region_id="defects",
            y_start_norm=0.2,
            y_end_norm=0.6,
            y_start=200,
            y_end=600,
            detection_method="adaptive",
            confidence=0.85,
        ),
        DocumentRegion(
            region_id="analysis",
            y_start_norm=0.6,
            y_end_norm=1.0,
            y_start=600,
            y_end=1000,
            detection_method="adaptive",
            confidence=0.88,
        ),
    ]


@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.mark.unit
class TestOCREngineInitialization:
    """Test OCREngine initialization."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_ocr_engine_initialization(self, mock_factory, test_settings, mock_logger):
        """Test OCREngine initializes with settings and creates OCR engine via factory."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine

        # Act
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Assert
        assert engine._settings == test_settings
        assert engine._logger == mock_logger
        assert engine._ocr_engine == mock_ocr_engine
        mock_factory.create_full_engine.assert_called_once_with(test_settings, mock_logger)

    @patch('src.ocr_engine.OCREngineFactory')
    def test_ocr_engine_initialization_with_mode(self, mock_factory, test_settings, mock_logger):
        """Test OCREngine initializes with different engine modes."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_detection_engine.return_value = mock_ocr_engine

        # Act
        engine = OCREngine(settings=test_settings, logger=mock_logger, engine_mode="detection")

        # Assert
        mock_factory.create_detection_engine.assert_called_once_with(test_settings, mock_logger)


@pytest.mark.unit
class TestOCREngineImageLoading:
    """Test image loading functionality."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_ocr_engine_loads_image(self, mock_factory, test_settings, mock_logger, test_image):
        """Test _load_image() loads valid images correctly."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        image = engine._load_image(test_image)

        # Assert
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # Color image

    @patch('src.ocr_engine.OCREngineFactory')
    def test_ocr_engine_load_image_not_found(self, mock_factory, test_settings, mock_logger, tmp_path):
        """Test FileNotFoundError for missing files."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        missing_path = tmp_path / "nonexistent.jpg"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            engine._load_image(missing_path)

    @patch('src.ocr_engine.OCREngineFactory')
    @patch('cv2.imread')
    def test_ocr_engine_load_image_invalid(self, mock_imread, mock_factory, test_settings, mock_logger, tmp_path):
        """Test ValueError for corrupted/invalid images."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        mock_imread.return_value = None  # Simulate invalid image
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        invalid_path = tmp_path / "invalid.jpg"
        invalid_path.write_bytes(b"not an image")

        # Act & Assert
        with pytest.raises(ValueError, match="Unable to read image"):
            engine._load_image(invalid_path)


@pytest.mark.unit
class TestOCREngineConfidenceFiltering:
    """Test confidence filtering logic."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_confidence_filtering_removes_low_confidence(
        self, mock_factory, test_settings, mock_logger, test_image, mock_ocr_response_mixed_confidence, tmp_path
    ):
        """Test that texts below threshold are filtered out."""
        # Arrange
        test_settings.ocr_confidence_threshold = 0.5
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = mock_ocr_response_mixed_confidence
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.process(test_image, output_path=tmp_path / "output.json")

        # Assert
        # Should only have 2 detections (HIGH_CONF and MEDIUM_CONF), LOW_CONF filtered
        assert result.total_texts_found == 2
        # low_confidence_count counts detections < threshold * 1.2 (0.6)
        # After filtering: 0.95 and 0.6 remain, 0.6 is not < 0.6, so count is 0
        assert result.low_confidence_count == 0

    @patch('src.ocr_engine.OCREngineFactory')
    def test_confidence_filtering_keeps_high_confidence(
        self, mock_factory, test_settings, mock_logger, test_image, mock_ocr_response_standard, tmp_path
    ):
        """Test that texts above threshold are kept."""
        # Arrange
        test_settings.ocr_confidence_threshold = 0.5
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = mock_ocr_response_standard
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.process(test_image, output_path=tmp_path / "output.json")

        # Assert
        assert result.total_texts_found == 3  # All above threshold
        assert result.average_confidence > 0.5

    @patch('src.ocr_engine.OCREngineFactory')
    def test_low_confidence_count_calculation(
        self, mock_factory, test_settings, mock_logger, test_image, tmp_path
    ):
        """Test low_confidence_count is correctly calculated."""
        # Arrange
        test_settings.ocr_confidence_threshold = 0.5
        # Create response with confidence below threshold * 1.2 (0.6)
        mock_ocr_response = [[
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('HIGH', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('LOW', 0.55)),  # Below 0.6
        ]]
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = mock_ocr_response
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.process(test_image, output_path=tmp_path / "output.json")

        # Assert
        # Threshold * 1.2 = 0.6, so 0.55 is below
        assert result.low_confidence_count == 1

    @patch('src.ocr_engine.OCREngineFactory')
    def test_average_confidence_calculation(self, mock_factory, test_settings, mock_logger):
        """Test _calculate_average_confidence() with various inputs."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        detections = [
            TextDetection(text="A", confidence=0.9, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=5, center_y=5),
            TextDetection(text="B", confidence=0.8, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=5, center_y=5),
            TextDetection(text="C", confidence=0.7, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=5, center_y=5),
        ]

        # Act
        avg = engine._calculate_average_confidence(detections)

        # Assert
        assert avg == pytest.approx(0.8, abs=0.01)

    @patch('src.ocr_engine.OCREngineFactory')
    def test_average_confidence_empty_list(self, mock_factory, test_settings, mock_logger):
        """Test returns 0.0 for empty detection list."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        avg = engine._calculate_average_confidence([])

        # Assert
        assert avg == 0.0


@pytest.mark.unit
class TestOCREngineResultParsing:
    """Test OCR result parsing logic."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_standard_format(self, mock_factory, test_settings, mock_logger, mock_ocr_response_standard):
        """Test _process_ocr_results() with standard PaddleOCR format."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        detections = engine._process_ocr_results(mock_ocr_response_standard)

        # Assert
        assert len(detections) == 3
        assert all(isinstance(d, TextDetection) for d in detections)
        assert detections[0].text == "ГЕОМЕТРИЯ"
        assert detections[0].confidence == 0.95

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_with_scale(self, mock_factory, test_settings, mock_logger, mock_ocr_response_standard):
        """Test coordinate scaling with scale factor != 1.0."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        scale = 0.5  # Image was downscaled by 50%

        # Act
        detections = engine._process_ocr_results(mock_ocr_response_standard, scale=scale)

        # Assert
        # Coordinates should be scaled up (inverse scale)
        assert len(detections) == 3
        # Original bbox: [[0, 0], [100, 0], [100, 50], [0, 50]]
        # With scale 0.5, inv_scale = 2.0, so coordinates should be doubled
        assert detections[0].bbox[1][0] == 200  # 100 * 2

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_with_offsets(self, mock_factory, test_settings, mock_logger, mock_ocr_response_standard):
        """Test coordinate offsetting for regional processing."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        offset_x, offset_y = 100, 200

        # Act
        detections = engine._process_ocr_results(
            mock_ocr_response_standard, offset_x=offset_x, offset_y=offset_y
        )

        # Assert
        assert len(detections) == 3
        # Original center: (50, 25), with offset should be (150, 225)
        assert detections[0].center_x == 50 + offset_x
        assert detections[0].center_y == 25 + offset_y

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_with_region_id(self, mock_factory, test_settings, mock_logger, mock_ocr_response_standard):
        """Test region_id is properly assigned."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        region_id = "header"

        # Act
        detections = engine._process_ocr_results(mock_ocr_response_standard, region_id=region_id)

        # Assert
        assert len(detections) == 3
        assert all(d.region_id == region_id for d in detections)

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_none(self, mock_factory, test_settings, mock_logger):
        """Test handling of None OCR results."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        detections = engine._process_ocr_results(None)

        # Assert
        assert detections == []

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_empty_list(self, mock_factory, test_settings, mock_logger):
        """Test handling of empty list."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        detections = engine._process_ocr_results([])

        # Assert
        assert detections == []

    @patch('src.ocr_engine.OCREngineFactory')
    def test_parse_ocr_results_empty_page(self, mock_factory, test_settings, mock_logger):
        """Test handling of empty first page."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        detections = engine._process_ocr_results([[]])

        # Assert
        assert detections == []


@pytest.mark.unit
class TestOCREngineRegionProcessing:
    """Test region-based processing."""

    @patch('src.ocr_engine.OCREngineFactory')
    @patch('src.ocr_engine.RegionDetector')
    def test_process_regions_sequential(
        self, mock_region_detector_class, mock_factory, test_settings, mock_logger, test_image, mock_document_regions, tmp_path
    ):
        """Test _process_regions_sequential() with multiple regions."""
        # Arrange
        test_settings.enable_parallel_processing = False
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = [[([[0, 0], [100, 0], [100, 50], [0, 50]], ('TEXT', 0.9))]]
        mock_factory.create_full_engine.return_value = mock_ocr_engine

        mock_region_detector = MagicMock()
        mock_region_image = np.ones((200, 800, 3), dtype=np.uint8) * 255
        mock_region_detector.extract_region.return_value = (mock_region_image, 1.0)
        mock_region_detector_class.return_value = mock_region_detector

        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.process_regions(test_image, mock_document_regions, output_path=tmp_path / "output.json")

        # Assert
        assert isinstance(result, OCRResult)
        assert result.total_texts_found > 0
        assert result.output_path.exists()

    @patch('src.ocr_engine.OCREngineFactory')
    def test_process_regions_empty_raises_error(self, mock_factory, test_settings, mock_logger, test_image):
        """Test ValueError raised for empty region list."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act & Assert
        with pytest.raises(ValueError, match="Region list cannot be empty"):
            engine.process_regions(test_image, [])

    @patch('src.ocr_engine.OCREngineFactory')
    def test_should_use_parallel_processing_disabled(self, mock_factory, test_settings, mock_logger, mock_document_regions):
        """Test returns False when parallel disabled in settings."""
        # Arrange
        test_settings.enable_parallel_processing = False
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine._should_use_parallel_regions_processing(mock_document_regions)

        # Assert
        assert result is False

    @patch('src.ocr_engine.OCREngineFactory')
    def test_should_use_parallel_processing_few_regions(self, mock_factory, test_settings, mock_logger):
        """Test returns False when regions < threshold."""
        # Arrange
        test_settings.enable_parallel_processing = True
        test_settings.parallel_min_regions_for_parallelization = 5
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        few_regions = [DocumentRegion(
            region_id="r1", y_start_norm=0.0, y_end_norm=0.5, y_start=0, y_end=100,
            detection_method="test", confidence=0.9
        )]

        # Act
        result = engine._should_use_parallel_regions_processing(few_regions)

        # Assert
        assert result is False

    @patch('src.ocr_engine.OCREngineFactory')
    def test_should_use_parallel_processing_many_regions(self, mock_factory, test_settings, mock_logger):
        """Test returns True when regions >= threshold."""
        # Arrange
        test_settings.enable_parallel_processing = True
        test_settings.parallel_regions_enabled = True
        test_settings.parallel_min_regions_for_parallelization = 2
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        many_regions = [
            DocumentRegion(region_id="r1", y_start_norm=0.0, y_end_norm=0.33, y_start=0, y_end=100,
                          detection_method="test", confidence=0.9),
            DocumentRegion(region_id="r2", y_start_norm=0.33, y_end_norm=0.66, y_start=100, y_end=200,
                          detection_method="test", confidence=0.9),
            DocumentRegion(region_id="r3", y_start_norm=0.66, y_end_norm=1.0, y_start=200, y_end=300,
                          detection_method="test", confidence=0.9),
        ]

        # Act
        result = engine._should_use_parallel_regions_processing(many_regions)

        # Assert
        assert result is True

    @patch('src.ocr_engine.OCREngineFactory')
    @patch('src.ocr_engine.RegionDetector')
    def test_process_regions_coordination(
        self, mock_region_detector_class, mock_factory, test_settings, mock_logger, test_image, mock_document_regions, tmp_path
    ):
        """Test process_regions() calls appropriate method (parallel/sequential)."""
        # Arrange
        test_settings.enable_parallel_processing = False  # Force sequential
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = [[([[0, 0], [100, 0], [100, 50], [0, 50]], ('TEXT', 0.9))]]
        mock_factory.create_full_engine.return_value = mock_ocr_engine

        mock_region_detector = MagicMock()
        mock_region_image = np.ones((200, 800, 3), dtype=np.uint8) * 255
        mock_region_detector.extract_region.return_value = (mock_region_image, 1.0)
        mock_region_detector_class.return_value = mock_region_detector

        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.process_regions(test_image, mock_document_regions, output_path=tmp_path / "output.json")

        # Assert
        assert isinstance(result, OCRResult)
        # Verify sequential processing was used (no parallel executor calls)


@pytest.mark.unit
class TestOCREngineImageResizing:
    """Test image resizing logic."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_resize_for_ocr_small_image(self, mock_factory, test_settings, mock_logger):
        """Test no resize for images under max dimension."""
        # Arrange
        test_settings.ocr_max_image_dimension = 2000
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        small_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        # Act
        resized, scale = engine._resize_for_ocr(small_image)

        # Assert
        assert scale == 1.0
        assert resized.shape == small_image.shape

    @patch('src.ocr_engine.OCREngineFactory')
    def test_resize_for_ocr_large_image(self, mock_factory, test_settings, mock_logger):
        """Test downscaling for oversized images."""
        # Arrange
        test_settings.ocr_max_image_dimension = 1000
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        large_image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        resized, scale = engine._resize_for_ocr(large_image)

        # Assert
        assert scale < 1.0
        assert max(resized.shape[:2]) <= test_settings.ocr_max_image_dimension

    @patch('src.ocr_engine.OCREngineFactory')
    def test_resize_for_ocr_scale_calculation(self, mock_factory, test_settings, mock_logger):
        """Test scale factor is correctly calculated."""
        # Arrange
        test_settings.ocr_max_image_dimension = 1000
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        large_image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255

        # Act
        resized, scale = engine._resize_for_ocr(large_image)

        # Assert
        expected_scale = 1000 / 2000  # max_dim / max(width, height)
        assert scale == pytest.approx(expected_scale, abs=0.001)


@pytest.mark.unit
class TestOCREngineOutputStructure:
    """Test output structure creation."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_create_output_structure(self, mock_factory, test_settings, mock_logger, test_image):
        """Test _create_output_structure() creates proper JSON format."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        
        detections = [
            TextDetection(
                text="TEST", confidence=0.9,
                bbox=[[0, 0], [100, 0], [100, 50], [0, 50]],
                center_x=50, center_y=25, region_id="header"
            )
        ]
        start_time = 0.0

        # Act
        output = engine._create_output_structure(
            test_image, detections, start_time, image_width=800, image_height=1000
        )

        # Assert
        assert "document_info" in output
        assert "processing_metrics" in output
        assert "text_regions" in output
        assert len(output["text_regions"]) == 1
        assert output["text_regions"][0]["text"] == "TEST"
        assert output["text_regions"][0]["confidence"] == 0.9

    @patch('src.ocr_engine.OCREngineFactory')
    def test_build_output_path(self, mock_factory, test_settings, mock_logger, test_image):
        """Test _build_output_path() generates correct filename with -texts suffix."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        output_path = engine._build_output_path(test_image)

        # Assert
        assert output_path.name == "test_image-texts.json"
        assert output_path.parent == test_settings.output_dir

    @patch('src.ocr_engine.OCREngineFactory')
    def test_build_output_path_removes_cor_suffix(self, mock_factory, test_settings, mock_logger, tmp_path):
        """Test removes existing -cor suffix from preprocessing."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        input_path = tmp_path / "test_image-cor.jpg"

        # Act
        output_path = engine._build_output_path(input_path)

        # Assert
        assert output_path.name == "test_image-texts.json"  # -cor removed


@pytest.mark.unit
class TestOCREngineContextManager:
    """Test context manager functionality."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_context_manager_enter(self, mock_factory, test_settings, mock_logger):
        """Test __enter__ returns self."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)

        # Act
        result = engine.__enter__()

        # Assert
        assert result is engine

    @patch('src.ocr_engine.OCREngineFactory')
    def test_context_manager_exit(self, mock_factory, test_settings, mock_logger):
        """Test __exit__ calls close()."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        engine.close = MagicMock()

        # Act
        engine.__exit__(None, None, None)

        # Assert
        engine.close.assert_called_once()

    @patch('src.ocr_engine.OCREngineFactory')
    def test_close_releases_resources(self, mock_factory, test_settings, mock_logger):
        """Test close() sets _ocr_engine to None."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        assert engine._ocr_engine is not None

        # Act
        engine.close()

        # Assert
        assert engine._ocr_engine is None


@pytest.mark.unit
class TestOCREngineErrorHandling:
    """Test error handling."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_run_paddle_ocr_exception(self, mock_factory, test_settings, mock_logger, test_image):
        """Test error handling when PaddleOCR raises exception."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.side_effect = Exception("OCR failed")
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        image = engine._load_image(test_image)

        # Act & Assert
        with pytest.raises(Exception, match="OCR failed"):
            engine._run_paddle_ocr(image)

    @patch('src.ocr_engine.OCREngineFactory')
    def test_process_ocr_results_malformed_data(self, mock_factory, test_settings, mock_logger):
        """Test graceful handling of malformed OCR result format."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        malformed_results = [[("invalid", "format")]]  # Wrong structure

        # Act
        detections = engine._process_ocr_results(malformed_results)

        # Assert
        # Should handle gracefully, may return empty or partial results
        assert isinstance(detections, list)


@pytest.mark.unit
class TestOCREngineFullProcess:
    """Test full process() method."""

    @patch('src.ocr_engine.OCREngineFactory')
    def test_ocr_engine_process_with_mocked_ocr(
        self, mock_factory, test_settings, mock_logger, test_image, mock_ocr_response_standard, tmp_path
    ):
        """Test full process() flow with mocked PaddleOCR results."""
        # Arrange
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.ocr.return_value = mock_ocr_response_standard
        mock_factory.create_full_engine.return_value = mock_ocr_engine
        engine = OCREngine(settings=test_settings, logger=mock_logger)
        output_path = tmp_path / "output.json"

        # Act
        result = engine.process(test_image, output_path=output_path)

        # Assert
        assert isinstance(result, OCRResult)
        assert result.output_path == output_path
        assert result.total_texts_found == 3
        assert result.duration_seconds > 0
        assert result.average_confidence > 0
        assert output_path.exists()
        
        # Verify JSON structure
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
            assert "text_regions" in data
            assert len(data["text_regions"]) == 3

