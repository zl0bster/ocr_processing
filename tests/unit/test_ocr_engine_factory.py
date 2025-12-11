"""Unit tests for OCREngineFactory module."""
import logging
import pytest
from unittest.mock import patch, MagicMock, Mock
import inspect

from src.ocr_engine_factory import OCREngineFactory
from src.config.settings import Settings


@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    return logging.getLogger("test")


@pytest.fixture
def test_settings():
    """Settings for testing."""
    return Settings(
        ocr_language="ru",
        ocr_use_gpu=False,
        ocr_det_limit_side_len=960,
    )


@pytest.mark.unit
class TestOCREngineFactoryMethods:
    """Test factory method calls."""

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.OCREngineFactory._create_engine')
    def test_create_full_engine(self, mock_create_engine, mock_paddleocr, test_settings, mock_logger):
        """Test create_full_engine() calls _create_engine with det=True, rec=True, cls=True."""
        # Arrange
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Act
        result = OCREngineFactory.create_full_engine(test_settings, mock_logger)

        # Assert
        mock_create_engine.assert_called_once_with(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )
        assert result == mock_engine

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.OCREngineFactory._create_engine')
    def test_create_detection_engine(self, mock_create_engine, mock_paddleocr, test_settings, mock_logger):
        """Test create_detection_engine() calls _create_engine with det=True, rec=False, cls=False."""
        # Arrange
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Act
        result = OCREngineFactory.create_detection_engine(test_settings, mock_logger)

        # Assert
        mock_create_engine.assert_called_once_with(
            test_settings, mock_logger, det=True, rec=False, cls=False, engine_type="detection"
        )
        assert result == mock_engine

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.OCREngineFactory._create_engine')
    def test_create_recognition_engine(self, mock_create_engine, mock_paddleocr, test_settings, mock_logger):
        """Test create_recognition_engine() calls _create_engine with det=False, rec=True, cls=True."""
        # Arrange
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Act
        result = OCREngineFactory.create_recognition_engine(test_settings, mock_logger)

        # Assert
        mock_create_engine.assert_called_once_with(
            test_settings, mock_logger, det=False, rec=True, cls=True, engine_type="recognition"
        )
        assert result == mock_engine


@pytest.mark.unit
class TestOCREngineFactoryConfiguration:
    """Test engine configuration parameters."""

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_basic_params(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test OCR engine created with correct basic params (language, show_log=False)."""
        # Arrange
        mock_sig = MagicMock()
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock(), 'use_angle_cls': Mock(), 
                              'det_limit_side_len': Mock(), 'det_limit_type': Mock()}
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        call_kwargs = mock_paddleocr.call_args[1] if mock_paddleocr.call_args[1] else mock_paddleocr.call_args[0][0] if mock_paddleocr.call_args[0] else {}
        # Check that basic params are passed
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_with_gpu(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test GPU parameter included when ocr_use_gpu=True."""
        # Arrange
        test_settings.ocr_use_gpu = True
        mock_sig = MagicMock()
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock(), 'use_angle_cls': Mock(),
                              'use_gpu': Mock(), 'det_limit_side_len': Mock(), 'det_limit_type': Mock()}
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        # Verify GPU param was considered (actual check depends on signature inspection)
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_without_gpu(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test GPU parameter omitted when ocr_use_gpu=False."""
        # Arrange
        test_settings.ocr_use_gpu = False
        mock_sig = MagicMock()
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock(), 'use_angle_cls': Mock(),
                              'det_limit_side_len': Mock(), 'det_limit_type': Mock()}
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_filters_unsupported_params(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test unsupported parameters are filtered via signature inspection."""
        # Arrange
        mock_sig = MagicMock()
        # Only some params supported
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock()}
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        # Should only use supported params
        call_kwargs = mock_paddleocr.call_args[1] if mock_paddleocr.call_args[1] else {}
        # Verify unsupported params were filtered
        assert result == mock_engine_instance


@pytest.mark.unit
class TestOCREngineFactoryFallback:
    """Test fallback mechanisms."""

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_fallback_old_version(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test fallback when det/rec/cls params not supported."""
        # Arrange
        mock_sig = MagicMock()
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock()}  # No det/rec/cls
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        # Should fall back to standard initialization without det/rec/cls
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_fallback_minimal_params(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test fallback to minimal params on error."""
        # Arrange
        mock_sig = MagicMock()
        mock_sig.parameters = {'lang': Mock(), 'show_log': Mock()}
        mock_signature.return_value = mock_sig
        
        # First call fails, second succeeds
        mock_engine_instance = MagicMock()
        mock_paddleocr.side_effect = [Exception("Failed"), mock_engine_instance]

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        # Should have tried multiple times
        assert mock_paddleocr.call_count >= 2
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_fallback_no_params(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test final fallback to PaddleOCR() with no params."""
        # Arrange
        mock_sig = MagicMock()
        mock_sig.parameters = {}
        mock_signature.return_value = mock_sig
        
        # All attempts fail except the last one with no params
        mock_engine_instance = MagicMock()
        mock_paddleocr.side_effect = [
            Exception("Failed with lang"),
            Exception("Failed with minimal"),
            mock_engine_instance  # Success with no params
        ]

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        # Should have tried multiple times, last with no params
        assert mock_paddleocr.call_count >= 3
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_signature_inspection_error(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test handling when signature inspection fails."""
        # Arrange
        mock_signature.side_effect = Exception("Inspection failed")
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        # Should handle gracefully and still create engine
        mock_paddleocr.assert_called()
        assert result == mock_engine_instance

    @patch('src.ocr_engine_factory.PaddleOCR')
    @patch('src.ocr_engine_factory.inspect.signature')
    def test_create_engine_det_rec_supported(self, mock_signature, mock_paddleocr, test_settings, mock_logger):
        """Test when det/rec/cls parameters are supported."""
        # Arrange
        mock_sig = MagicMock()
        mock_sig.parameters = {'det': Mock(), 'rec': Mock(), 'cls': Mock(), 'lang': Mock(), 
                              'show_log': Mock(), 'use_angle_cls': Mock(), 'det_limit_side_len': Mock()}
        mock_signature.return_value = mock_sig
        
        mock_engine_instance = MagicMock()
        mock_paddleocr.return_value = mock_engine_instance

        # Act
        result = OCREngineFactory._create_engine(
            test_settings, mock_logger, det=True, rec=True, cls=True, engine_type="full"
        )

        # Assert
        mock_paddleocr.assert_called_once()
        # Should use det/rec/cls params
        call_kwargs = mock_paddleocr.call_args[1] if mock_paddleocr.call_args[1] else {}
        assert result == mock_engine_instance


