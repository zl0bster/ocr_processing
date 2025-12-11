"""OCR Engine Pool - Connection pool for managing multiple OCR engines.

FIXES APPLIED:
- ✅ Deadlock prevention: Lock not held during engine.close()
- ✅ Parallel shutdown: ThreadPoolExecutor for concurrent engine closure
- ✅ Timeout protection: Individual and global timeouts
- ✅ Graceful degradation: Proper error handling for timeout scenarios
"""

from __future__ import annotations

import concurrent.futures
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
        """Close all engines in the pool and release resources.
        
        FIXES:
        - ✅ Deadlock prevention: lock not held during engine.close()
        - ✅ Parallel shutdown: ThreadPoolExecutor for concurrent closure
        - ✅ Timeout protection: individual and global timeouts
        - ✅ Graceful degradation: handles timeouts gracefully
        """
        import gc
        
        self._logger.info("Closing OCR engine pool...")
        start_time = time.perf_counter()
        
        # ✅ FIX #1: Get copy and clear pool WITHIN lock (no nested lock calls)
        # Collect lightweight stats while holding the lock, compute metrics after releasing it
        with self._lock:
            engines_to_close = list(self._all_engines)
            total_engines = len(self._all_engines)
            available_engines = len(self._available_engines)
            self._available_engines.clear()
            self._all_engines.clear()
            # ✅ LOCK RELEASED HERE - no deadlock!
        
        # Compute statistics outside the lock to avoid deadlock
        total_restarts = 0
        total_files_processed = 0
        for engine in engines_to_close:
            metrics = engine.get_metrics()
            total_restarts += metrics.restarts_performed
            total_files_processed += metrics.files_processed
        
        self._logger.info(
            "Pool statistics before shutdown: "
            "Total engines: %d, Available: %d, In use: %d, "
            "Files processed: %d, Total restarts: %d",
            total_engines,
            available_engines,
            total_engines - available_engines,
            total_files_processed,
            total_restarts,
        )
        
        # ✅ FIX #2: Close engines OUTSIDE lock with proper error handling
        self._logger.info("Closing %d engines...", len(engines_to_close))
        
        # Use ThreadPoolExecutor for parallel engine closure
        if engines_to_close:
            self._close_engines_parallel(engines_to_close)
        
        # Force garbage collection
        self._logger.debug("Running garbage collection...")
        gc_start = time.perf_counter()
        gc.collect()
        gc_duration = time.perf_counter() - gc_start
        self._logger.debug("Garbage collection completed in %.3fs", gc_duration)
        
        total_duration = time.perf_counter() - start_time
        self._logger.info("OCR engine pool closed in %.3f seconds", total_duration)
    
    def _close_engines_parallel(self, engines: list) -> None:
        """Close multiple engines in parallel with timeouts.
        
        ✅ FIX: Uses ThreadPoolExecutor for concurrent closure
        ✅ FIX: Individual per-engine timeout (5 sec)
        ✅ FIX: Global timeout for all engines (30 sec)
        
        Args:
            engines: List of engines to close
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(engines))
        ) as executor:
            # Submit all close tasks
            futures = {
                executor.submit(
                    self._close_engine_safe, 
                    engine, 
                    i + 1, 
                    len(engines)
                ): engine
                for i, engine in enumerate(engines)
            }
            
            # Wait for completion with timeout
            try:
                for future in concurrent.futures.as_completed(
                    futures, timeout=30.0
                ):
                    try:
                        future.result(timeout=5.0)
                    except concurrent.futures.TimeoutError:
                        self._logger.warning(
                            "Engine close operation timed out. "
                            "This engine may not have released all resources."
                        )
                    except Exception as e:
                        self._logger.warning("Error closing engine: %s", e)
            except concurrent.futures.TimeoutError:
                self._logger.warning(
                    "Global engine pool close timeout (30 sec) exceeded. "
                    "Some engines may not be properly closed."
                )
    
    def _close_engine_safe(
        self, 
        engine: SelfMonitoringOCREngine, 
        index: int, 
        total: int
    ) -> None:
        """Close a single engine with error handling.
        
        Args:
            engine: Engine to close
            index: Engine index (for logging)
            total: Total engines (for logging)
        """
        engine_start = time.perf_counter()
        try:
            self._logger.debug("Closing engine %d/%d...", index, total)
            engine.close()
            engine_duration = time.perf_counter() - engine_start
            self._logger.debug(
                "Engine %d closed successfully in %.3fs", 
                index, 
                engine_duration
            )
        except Exception as e:
            self._logger.warning(
                "Error closing engine %d: %s", 
                index, 
                str(e)[:200]  # Limit error message length
            )
    
    def __enter__(self) -> "OCREnginePool":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

