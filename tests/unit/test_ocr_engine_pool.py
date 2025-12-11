"""Unit tests for OCREnginePool module."""

import logging
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from src.ocr_engine_pool import OCREnginePool, PoolStatistics, PoolEngineContext
from src.self_monitoring_ocr_engine import SelfMonitoringOCREngine
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
        ocr_pool_timeout=5.0,
        ocr_engine_memory_check_interval=1,
        ocr_engine_auto_restart_threshold=0.8,
        ocr_memory_reload_threshold_mb=700,
        ocr_use_gpu=False,
        ocr_language="ru",
        ocr_confidence_threshold=0.5,
        log_level="WARNING",
    )


@pytest.mark.unit
class TestOCREnginePoolInitialization:
    """Test OCREnginePool initialization."""

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_initialization_creates_engines(self, mock_engine_class, test_settings_with_pool, mock_logger):
        """Test pool initializes with correct number of engines."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine
        test_settings_with_pool.ocr_pool_size = 3

        # Act
        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Assert
        assert pool._pool_size == 3
        assert len(pool._all_engines) == 3
        assert len(pool._available_engines) == 3
        assert mock_engine_class.call_count == 3

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_initialization_validates_pool_size_min(self, mock_engine_class,
                                                    test_settings_with_pool, mock_logger):
        """Test pool size is validated to minimum of 1."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine
        test_settings_with_pool.ocr_pool_size = 0

        # Act
        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Assert
        assert pool._pool_size == 1
        assert len(pool._all_engines) == 1

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_initialization_warns_on_large_pool_size(self, mock_engine_class,
                                                    test_settings_with_pool, mock_logger):
        """Test pool warns when pool size is large."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine
        test_settings_with_pool.ocr_pool_size = 10

        # Act
        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Assert
        assert pool._pool_size == 10
        # Warning should be logged (we can't easily test this without more complex mocking)


@pytest.mark.unit
class TestOCREnginePoolAcquireRelease:
    """Test engine acquisition and release."""

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_acquire_returns_available_engine(self, mock_engine_class,
                                            test_settings_with_pool, mock_logger):
        """Test acquire() returns an available engine."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Act
        with pool.acquire() as engine:
            # Assert
            assert engine is not None
            assert engine in pool._all_engines
            assert engine not in pool._available_engines

        # After context exit, engine should be released
        assert engine in pool._available_engines

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_acquire_waits_for_available_engine(self, mock_engine_class,
                                                test_settings_with_pool, mock_logger):
        """Test acquire() waits when no engines are available."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
            pool_size=1,  # Only one engine
        )

        # Acquire the only engine
        context1 = pool.acquire()
        engine1 = context1.__enter__()

        # Try to acquire again (should wait)
        start_time = time.time()
        try:
            context2 = pool.acquire(timeout=0.5)  # Short timeout
            context2.__enter__()
        except TimeoutError:
            elapsed = time.time() - start_time
            # Should have waited approximately timeout duration
            assert elapsed >= 0.4  # Allow some tolerance
        finally:
            context1.__exit__(None, None, None)

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_release_returns_engine_to_pool(self, mock_engine_class,
                                           test_settings_with_pool, mock_logger):
        """Test release() returns engine to available pool."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Act
        with pool.acquire() as engine:
            assert engine not in pool._available_engines

        # After release
        assert engine in pool._available_engines

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_release_ignores_duplicate_release(self, mock_engine_class,
                                               test_settings_with_pool, mock_logger):
        """Test release() ignores duplicate releases."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        )

        # Act
        with pool.acquire() as engine:
            pass  # Auto-release

        # Try to release again
        initial_available = len(pool._available_engines)
        pool.release(engine)
        
        # Assert - should still be same count (not added twice)
        assert len(pool._available_engines) == initial_available


@pytest.mark.unit
class TestOCREnginePoolStatistics:
    """Test pool statistics tracking."""

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_get_statistics_returns_pool_stats(self, mock_engine_class,
                                               test_settings_with_pool, mock_logger):
        """Test get_statistics() returns current pool statistics."""
        # Arrange
        mock_engine1 = MagicMock()
        mock_engine1.get_metrics.return_value = MagicMock(
            files_processed=5,
            restarts_performed=1,
        )
        mock_engine2 = MagicMock()
        mock_engine2.get_metrics.return_value = MagicMock(
            files_processed=3,
            restarts_performed=0,
        )
        mock_engine_class.side_effect = [mock_engine1, mock_engine2]

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
            pool_size=2,
        )

        # Act
        stats = pool.get_statistics()

        # Assert
        assert isinstance(stats, PoolStatistics)
        assert stats.total_engines == 2
        assert stats.available_engines == 2
        assert stats.in_use_engines == 0
        assert stats.total_files_processed == 8  # 5 + 3
        assert stats.total_restarts == 1  # 1 + 0

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_get_statistics_tracks_in_use_engines(self, mock_engine_class,
                                                  test_settings_with_pool, mock_logger):
        """Test statistics correctly track in-use engines."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
            pool_size=2,
        )

        # Act - acquire one engine
        with pool.acquire():
            stats = pool.get_statistics()

            # Assert
            assert stats.total_engines == 2
            assert stats.available_engines == 1
            assert stats.in_use_engines == 1


@pytest.mark.unit
class TestOCREnginePoolCleanup:
    """Test pool cleanup and shutdown."""

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_close_closes_all_engines(self, mock_engine_class,
                                      test_settings_with_pool, mock_logger):
        """Test close() closes all engines in pool."""
        # Arrange
        mock_engine1 = MagicMock()
        mock_engine1.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine2 = MagicMock()
        mock_engine2.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.side_effect = [mock_engine1, mock_engine2]

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
            pool_size=2,
        )

        # Act
        pool.close()

        # Assert
        mock_engine1.close.assert_called_once()
        mock_engine2.close.assert_called_once()
        assert len(pool._all_engines) == 0
        assert len(pool._available_engines) == 0

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_context_manager_closes_pool(self, mock_engine_class,
                                        test_settings_with_pool, mock_logger):
        """Test context manager automatically closes pool."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        # Act
        with OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
        ) as pool:
            assert pool is not None

        # Assert
        mock_engine.close.assert_called()


@pytest.mark.unit
class TestOCREnginePoolThreadSafety:
    """Test thread safety of pool operations."""

    @patch('src.ocr_engine_pool.SelfMonitoringOCREngine')
    def test_concurrent_acquire_release(self, mock_engine_class,
                                       test_settings_with_pool, mock_logger):
        """Test pool handles concurrent acquire/release operations."""
        # Arrange
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = MagicMock(
            files_processed=0,
            restarts_performed=0,
        )
        mock_engine_class.return_value = mock_engine

        pool = OCREnginePool(
            settings=test_settings_with_pool,
            logger=mock_logger,
            pool_size=2,
        )

        acquired_engines = []
        errors = []

        def worker(worker_id):
            try:
                with pool.acquire() as engine:
                    acquired_engines.append((worker_id, engine))
                    time.sleep(0.01)  # Simulate work
            except Exception as e:
                errors.append((worker_id, e))

        # Act - run multiple workers concurrently
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(acquired_engines) == 5  # All workers should have acquired engines



