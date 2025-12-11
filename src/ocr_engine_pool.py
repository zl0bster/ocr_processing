"""OCR engine pool for managing reusable self-monitoring OCR engines."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from self_monitoring_ocr_engine import SelfMonitoringOCREngine, EngineMetrics


@dataclass
class PoolStatistics:
    """Statistics for the engine pool."""
    
    total_engines: int
    available_engines: int
    in_use_engines: int
    total_restarts: int
    total_files_processed: int


class PoolEngineContext:
    """Context manager for pool engine acquisition."""
    
    def __init__(
        self,
        pool: "OCREnginePool",
        engine: SelfMonitoringOCREngine,
    ) -> None:
        """Initialize context manager.
        
        Args:
            pool: Reference to the pool for release
            engine: The acquired engine
        """
        self._pool = pool
        self._engine = engine
    
    def __enter__(self) -> SelfMonitoringOCREngine:
        """Return the engine for use."""
        return self._engine
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the engine back to the pool."""
        self._pool.release(self._engine)


class OCREnginePool:
    """Pool of self-monitoring OCR engines for batch processing.
    
    Manages a pool of SelfMonitoringOCREngine instances that can be acquired
    and released. Engines are reused across multiple files, reducing initialization
    overhead. Each engine independently monitors its memory and performs self-restarts.
    
    Thread-safe for concurrent access (though batch processing is currently sequential).
    """
    
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        pool_size: Optional[int] = None,
    ) -> None:
        """Initialize OCR engine pool.
        
        Args:
            settings: Application settings
            logger: Logger instance
            pool_size: Number of engines in pool (defaults to settings.ocr_pool_size)
        """
        self._settings = settings
        self._logger = logger
        self._pool_size = pool_size or settings.ocr_pool_size
        
        # Validate pool size
        if self._pool_size < 1:
            self._logger.warning(
                "Pool size %d is invalid, using minimum size 1",
                self._pool_size,
            )
            self._pool_size = 1
        elif self._pool_size > 4:
            self._logger.warning(
                "Pool size %d is large, may consume significant memory. "
                "Consider using 1-4 engines.",
                self._pool_size,
            )
        
        # Thread-safe queue for available engines
        self._available_engines: list[SelfMonitoringOCREngine] = []
        self._all_engines: list[SelfMonitoringOCREngine] = []
        self._lock = threading.Lock()
        
        # Initialize engines
        self._logger.info(
            "Initializing OCR engine pool with %d engines...",
            self._pool_size,
        )
        
        for i in range(self._pool_size):
            engine = SelfMonitoringOCREngine(
                settings=self._settings,
                logger=self._logger,
                engine_id=f"pool-{i+1}",
                engine_pool=self,
            )
            self._available_engines.append(engine)
            self._all_engines.append(engine)
        
        self._logger.info(
            "OCR engine pool initialized with %d engines",
            len(self._all_engines),
        )
    
    def acquire(self, timeout: Optional[float] = None) -> PoolEngineContext:
        """Acquire an available engine from the pool.
        
        Args:
            timeout: Maximum seconds to wait for available engine
                    (defaults to settings.ocr_pool_timeout)
        
        Returns:
            PoolEngineContext that can be used as context manager
            
        Raises:
            TimeoutError: If no engine becomes available within timeout
        """
        timeout = timeout or self._settings.ocr_pool_timeout
        start_time = time.time()
        
        while True:
            with self._lock:
                if self._available_engines:
                    engine = self._available_engines.pop(0)
                    self._logger.debug(
                        "Engine acquired from pool. "
                        "Available: %d, In use: %d",
                        len(self._available_engines),
                        len(self._all_engines) - len(self._available_engines),
                    )
                    return PoolEngineContext(self, engine)
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"No engine available in pool after {timeout:.1f} seconds. "
                    f"Pool size: {len(self._all_engines)}, "
                    f"Available: {len(self._available_engines)}"
                )
            
            # Wait a bit before retrying
            time.sleep(0.1)
    
    def release(self, engine: SelfMonitoringOCREngine) -> None:
        """Release an engine back to the pool.
        
        Args:
            engine: Engine to release
        """
        with self._lock:
            if engine not in self._all_engines:
                self._logger.warning(
                    "Attempted to release engine not in pool. Ignoring.",
                )
                return
            
            if engine in self._available_engines:
                self._logger.warning(
                    "Engine already in available pool. Ignoring duplicate release.",
                )
                return
            
            self._available_engines.append(engine)
            self._logger.debug(
                "Engine released to pool. "
                "Available: %d, In use: %d",
                len(self._available_engines),
                len(self._all_engines) - len(self._available_engines),
            )
    
    def get_statistics(self) -> PoolStatistics:
        """Get current pool statistics.
        
        Returns:
            PoolStatistics with current pool state
        """
        with self._lock:
            available = len(self._available_engines)
            in_use = len(self._all_engines) - available
            
            # Aggregate metrics from all engines
            total_restarts = 0
            total_files_processed = 0
            
            for engine in self._all_engines:
                metrics = engine.get_metrics()
                total_restarts += metrics.restarts_performed
                total_files_processed += metrics.files_processed
            
            return PoolStatistics(
                total_engines=len(self._all_engines),
                available_engines=available,
                in_use_engines=in_use,
                total_restarts=total_restarts,
                total_files_processed=total_files_processed,
            )
    
    def close(self) -> None:
        """Close all engines in the pool and release resources."""
        import gc
        import time
        
        self._logger.info("Closing OCR engine pool...")
        start_time = time.perf_counter()
        
        with self._lock:
            stats = self.get_statistics()
            self._logger.info(
                "Pool statistics before shutdown: "
                "Total engines: %d, Files processed: %d, Total restarts: %d",
                stats.total_engines,
                stats.total_files_processed,
                stats.total_restarts,
            )
            
            # Close all engines with progress logging
            for i, engine in enumerate(self._all_engines, 1):
                engine_start = time.perf_counter()
                try:
                    self._logger.debug("Closing engine %d/%d...", i, len(self._all_engines))
                    engine.close()
                    engine_duration = time.perf_counter() - engine_start
                    self._logger.debug("Engine %d closed in %.3fs", i, engine_duration)
                except Exception as e:
                    self._logger.warning(
                        "Error closing engine %d: %s",
                        i,
                        e,
                    )
            
            self._available_engines.clear()
            self._all_engines.clear()
        
        # Force garbage collection to release PaddleOCR resources
        self._logger.debug("Running garbage collection...")
        gc_start = time.perf_counter()
        gc.collect()
        gc_duration = time.perf_counter() - gc_start
        self._logger.debug("Garbage collection completed in %.3fs", gc_duration)
        
        total_duration = time.perf_counter() - start_time
        self._logger.info("OCR engine pool closed in %.3f seconds", total_duration)
    
    def __enter__(self) -> "OCREnginePool":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

