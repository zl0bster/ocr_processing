"""Unit tests for SelfMonitoringOCREngine module."""

import logging
import pytest
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, PropertyMock
from dataclasses import dataclass

from src.self_monitoring_ocr_engine import SelfMonitoringOCREngine, EngineMetrics
from src.ocr_engine import OCRResult
from src.region_detector import DocumentRegion
from src.config.settings import Settings


@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    return logging.getLogger("test")


@pytest.fixture
def test_settings_with_pool():
    """Settings with pool configuration for testing."""
    return Settings(
        ocr_pool_enabled=True,
        ocr_pool_size=2,
        ocr_engine_memory_check_interval=1,
        ocr_engine_auto_restart_threshold=0.8,
        ocr_memory_reload_threshold_mb=700,
        ocr_use_gpu=False,
        ocr_language="ru",
        ocr_confidence_threshold=0.5,
        log_level="WARNING",
    )


@pytest.fixture
def mock_ocr_engine():
    """Mock OCREngine instance."""
    mock_engine = MagicMock()
    mock_engine.process.return_value = OCRResult(
        output_path=Path("test-output.json"),
        duration_seconds=1.0,
        total_texts_found=10,
        average_confidence=0.85,
        low_confidence_count=2,
    )
    mock_engine.process_regions.return_value = OCRResult(
        output_path=Path("test-output.json"),
        duration_seconds=1.0,
        total_texts_found=10,
        average_confidence=0.85,
        low_confidence_count=2,
    )
    return mock_engine


@pytest.mark.unit
class TestSelfMonitoringOCREngineInitialization:
    """Test SelfMonitoringOCREngine initialization."""

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_initialization(self, mock_memory_monitor_class, mock_ocr_engine_class,
                           test_settings_with_pool, mock_logger):
        """Test SelfMonitoringOCREngine initializes correctly."""
        # Arrange
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 100.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        # Act
        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
            engine_id="test-engine",
        )

        # Assert
        assert engine._engine_id == "test-engine"
        assert engine._restart_threshold_mb == 560.0  # 700 * 0.8
        assert engine._metrics.files_processed == 0
        assert engine._metrics.restarts_performed == 0
        mock_ocr_engine_class.assert_called_once()

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_initialization_calculates_restart_threshold(self, mock_memory_monitor_class,
                                                        mock_ocr_engine_class,
                                                        test_settings_with_pool, mock_logger):
        """Test restart threshold is calculated correctly."""
        # Arrange
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 100.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        test_settings_with_pool.ocr_memory_reload_threshold_mb = 1000
        test_settings_with_pool.ocr_engine_auto_restart_threshold = 0.75

        # Act
        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Assert
        assert engine._restart_threshold_mb == 750.0  # 1000 * 0.75


@pytest.mark.unit
class TestSelfMonitoringOCREngineMemoryMonitoring:
    """Test memory monitoring and self-restart functionality."""

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_memory_check_skips_when_not_interval(self, mock_memory_monitor_class,
                                                   mock_ocr_engine_class,
                                                   test_settings_with_pool, mock_logger):
        """Test memory check is skipped when not at check interval."""
        # Arrange
        test_settings_with_pool.ocr_engine_memory_check_interval = 5
        
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 1000.0  # High memory
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        engine._metrics.files_processed = 3  # Not a multiple of 5

        # Act
        engine._check_and_restart_if_needed()

        # Assert - should not check memory or restart
        assert engine._metrics.restarts_performed == 0
        mock_ocr_engine.close.assert_not_called()

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    @patch('gc.collect')
    def test_memory_check_triggers_restart_when_threshold_exceeded(
        self, mock_gc, mock_memory_monitor_class, mock_ocr_engine_class,
        test_settings_with_pool, mock_logger
    ):
        """Test engine restarts when memory exceeds threshold."""
        # Arrange
        test_settings_with_pool.ocr_engine_memory_check_interval = 1
        test_settings_with_pool.ocr_memory_reload_threshold_mb = 700
        test_settings_with_pool.ocr_engine_auto_restart_threshold = 0.8
        
        mock_memory_monitor = MagicMock()
        # First call: baseline, second: high memory (exceeds 560 MB threshold)
        mock_memory_monitor.get_memory_mb.side_effect = [100.0, 600.0, 150.0]
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        engine._metrics.files_processed = 1  # At check interval

        # Act
        engine._check_and_restart_if_needed()

        # Assert
        assert engine._metrics.restarts_performed == 1
        mock_ocr_engine.close.assert_called_once()
        mock_gc.assert_called_once()
        mock_ocr_engine_class.assert_called()  # Should create new engine

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_memory_check_no_restart_when_below_threshold(
        self, mock_memory_monitor_class, mock_ocr_engine_class,
        test_settings_with_pool, mock_logger
    ):
        """Test engine does not restart when memory is below threshold."""
        # Arrange
        test_settings_with_pool.ocr_engine_memory_check_interval = 1
        
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 400.0  # Below 560 MB threshold
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        engine._metrics.files_processed = 1

        # Act
        engine._check_and_restart_if_needed()

        # Assert
        assert engine._metrics.restarts_performed == 0
        mock_ocr_engine.close.assert_not_called()


@pytest.mark.unit
class TestSelfMonitoringOCREngineProcessing:
    """Test processing methods with memory monitoring."""

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_process_calls_memory_check(self, mock_memory_monitor_class,
                                       mock_ocr_engine_class,
                                       test_settings_with_pool, mock_logger, tmp_path):
        """Test process() method calls memory check before processing."""
        # Arrange
        test_settings_with_pool.ocr_engine_memory_check_interval = 1
        
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 100.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.process.return_value = OCRResult(
            output_path=tmp_path / "output.json",
            duration_seconds=1.0,
            total_texts_found=5,
            average_confidence=0.9,
            low_confidence_count=0,
        )
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        
        test_image = tmp_path / "test.jpg"
        test_image.touch()

        # Act
        result = engine.process(input_path=test_image)

        # Assert
        assert result.total_texts_found == 5
        assert engine._metrics.files_processed == 1
        mock_ocr_engine.process.assert_called_once()

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_process_regions_calls_memory_check(self, mock_memory_monitor_class,
                                                mock_ocr_engine_class,
                                                test_settings_with_pool, mock_logger, tmp_path):
        """Test process_regions() method calls memory check before processing."""
        # Arrange
        test_settings_with_pool.ocr_engine_memory_check_interval = 1
        
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 100.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.process_regions.return_value = OCRResult(
            output_path=tmp_path / "output.json",
            duration_seconds=1.0,
            total_texts_found=5,
            average_confidence=0.9,
            low_confidence_count=0,
        )
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        
        test_image = tmp_path / "test.jpg"
        test_image.touch()
        regions = [
            DocumentRegion(
                region_id="header",
                y_start_norm=0.0,
                y_end_norm=0.2,
                y_start=0,
                y_end=200,
                detection_method="adaptive",
                confidence=0.9,
            )
        ]

        # Act
        result = engine.process_regions(
            image_path=test_image,
            regions=regions,
        )

        # Assert
        assert result.total_texts_found == 5
        assert engine._metrics.files_processed == 1
        mock_ocr_engine.process_regions.assert_called_once()


@pytest.mark.unit
class TestSelfMonitoringOCREngineMetrics:
    """Test metrics tracking."""

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_get_metrics_returns_current_stats(self, mock_memory_monitor_class,
                                               mock_ocr_engine_class,
                                               test_settings_with_pool, mock_logger):
        """Test get_metrics() returns current engine statistics."""
        # Arrange
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 200.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        engine = SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )
        engine._metrics.files_processed = 5
        engine._metrics.restarts_performed = 2
        engine._metrics.memory_baseline_mb = 100.0
        engine._metrics.last_memory_check_mb = 150.0

        # Act
        metrics = engine.get_metrics()

        # Assert
        assert isinstance(metrics, EngineMetrics)
        assert metrics.files_processed == 5
        assert metrics.restarts_performed == 2
        assert metrics.memory_baseline_mb == 100.0
        assert metrics.last_memory_check_mb == 150.0


@pytest.mark.unit
class TestSelfMonitoringOCREngineContextManager:
    """Test context manager support."""

    @patch('src.self_monitoring_ocr_engine.OCREngine')
    @patch('src.self_monitoring_ocr_engine.MemoryMonitor')
    def test_context_manager_closes_engine(self, mock_memory_monitor_class,
                                          mock_ocr_engine_class,
                                          test_settings_with_pool, mock_logger):
        """Test context manager automatically closes engine."""
        # Arrange
        mock_memory_monitor = MagicMock()
        mock_memory_monitor.get_memory_mb.return_value = 100.0
        mock_memory_monitor_class.return_value = mock_memory_monitor
        
        mock_ocr_engine = MagicMock()
        mock_ocr_engine_class.return_value = mock_ocr_engine

        # Act
        with SelfMonitoringOCREngine(
            settings=test_settings_with_pool,
            logger=mock_logger,
        ) as engine:
            assert engine is not None

        # Assert
        mock_ocr_engine.close.assert_called_once()


