"""Self-monitoring OCR engine wrapper with automatic memory management and restart."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from config.settings import Settings
from ocr_engine import OCREngine, OCRResult
from region_detector import DocumentRegion
from utils.memory_monitor import MemoryMonitor

if TYPE_CHECKING:
    from ocr_engine_pool import OCREnginePool


@dataclass
class EngineMetrics:
    """Metrics tracked for a single engine instance."""
    
    files_processed: int = 0
    restarts_performed: int = 0
    memory_baseline_mb: float = 0.0
    last_memory_check_mb: float = 0.0


class SelfMonitoringOCREngine:
    """OCR engine wrapper that monitors its own memory and performs self-restarts.
    
    Wraps an OCREngine instance and tracks memory usage. Before each processing
    operation, checks if memory exceeds the auto-restart threshold. If so, performs
    a self-restart by closing the current engine, running GC, and creating a new one.
    
    This enables proactive memory management instead of reactive reloads after OOM.
    """
    
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        engine_id: Optional[str] = None,
        engine_pool: Optional["OCREnginePool"] = None,
    ) -> None:
        """Initialize self-monitoring OCR engine.
        
        Args:
            settings: Application settings
            logger: Logger instance
            engine_id: Optional identifier for this engine instance (for logging)
            engine_pool: Optional reference to parent engine pool for table processing
        """
        self._settings = settings
        self._logger = logger
        self._engine_id = engine_id or "default"
        self._engine_pool = engine_pool
        self._memory_monitor = MemoryMonitor(logger)
        
        # Calculate restart threshold (percentage of reload threshold)
        self._restart_threshold_mb = (
            self._settings.ocr_memory_reload_threshold_mb
            * self._settings.ocr_engine_auto_restart_threshold
        )
        
        # Initialize metrics
        self._metrics = EngineMetrics()
        
        # Create initial engine
        self._engine: Optional[OCREngine] = None
        self._initialize_engine()
        
        # Track baseline memory after initialization
        self._metrics.memory_baseline_mb = self._memory_monitor.get_memory_mb()
        self._metrics.last_memory_check_mb = self._metrics.memory_baseline_mb
        
        self._logger.info(
            "SelfMonitoringOCREngine[%s] initialized. "
            "Restart threshold: %.1f MB (%.0f%% of reload threshold %d MB)",
            self._engine_id,
            self._restart_threshold_mb,
            self._settings.ocr_engine_auto_restart_threshold * 100,
            self._settings.ocr_memory_reload_threshold_mb,
        )
    
    def _initialize_engine(self) -> None:
        """Create a new OCREngine instance."""
        self._logger.debug(
            "SelfMonitoringOCREngine[%s] initializing new engine instance...",
            self._engine_id,
        )
        self._engine = OCREngine(
            settings=self._settings,
            logger=self._logger,
            engine_pool=self._engine_pool,
        )
    
    def _check_and_restart_if_needed(self) -> None:
        """Check memory usage and restart engine if threshold exceeded.
        
        Checks current memory against restart threshold. If exceeded, performs
        self-restart: closes current engine, runs GC, creates new engine, resets counters.
        """
        # Check memory every N files (as configured)
        if (
            self._metrics.files_processed
            % self._settings.ocr_engine_memory_check_interval
            != 0
        ):
            return
        
        current_memory_mb = self._memory_monitor.get_memory_mb()
        self._metrics.last_memory_check_mb = current_memory_mb
        
        if current_memory_mb > self._restart_threshold_mb:
            self._logger.warning(
                "SelfMonitoringOCREngine[%s] memory usage (%.1f MB) exceeds "
                "restart threshold (%.1f MB). Performing self-restart...",
                self._engine_id,
                current_memory_mb,
                self._restart_threshold_mb,
            )
            
            # Perform self-restart
            self._restart_engine()
            
            # Log memory after restart
            mem_after_restart = self._memory_monitor.get_memory_mb()
            memory_freed = current_memory_mb - mem_after_restart
            
            self._logger.info(
                "SelfMonitoringOCREngine[%s] self-restart completed. "
                "Memory freed: %.1f MB (from %.1f MB to %.1f MB). "
                "Total restarts: %d",
                self._engine_id,
                memory_freed,
                current_memory_mb,
                mem_after_restart,
                self._metrics.restarts_performed,
            )
        else:
            self._logger.debug(
                "SelfMonitoringOCREngine[%s] memory check: %.1f MB "
                "(threshold: %.1f MB) - OK",
                self._engine_id,
                current_memory_mb,
                self._restart_threshold_mb,
            )
    
    def _restart_engine(self) -> None:
        """Perform engine restart: close current, GC, create new."""
        # Close current engine
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        
        # Force garbage collection
        gc.collect()
        
        # Create new engine
        self._initialize_engine()
        
        # Update metrics
        self._metrics.restarts_performed += 1
        self._metrics.memory_baseline_mb = self._memory_monitor.get_memory_mb()
    
    def process(self, input_path: Path, output_path: Optional[Path] = None) -> OCRResult:
        """Process image with OCR (with automatic memory check and restart if needed).
        
        Args:
            input_path: Path to input image
            output_path: Optional output path
            
        Returns:
            OCRResult with processing results
        """
        # Check memory and restart if needed before processing
        self._check_and_restart_if_needed()
        
        # Process with current engine
        result = self._engine.process(input_path=input_path, output_path=output_path)
        
        # Update metrics
        self._metrics.files_processed += 1
        
        return result
    
    def process_regions(
        self,
        image_path: Path,
        regions: list[DocumentRegion],
        output_path: Optional[Path] = None,
    ) -> OCRResult:
        """Process regions with OCR (with automatic memory check and restart if needed).
        
        Args:
            image_path: Path to input image
            regions: List of document regions to process
            output_path: Optional output path
            
        Returns:
            OCRResult with processing results
        """
        # Check memory and restart if needed before processing
        self._check_and_restart_if_needed()
        
        # Process with current engine
        result = self._engine.process_regions(
            image_path=image_path,
            regions=regions,
            output_path=output_path,
        )
        
        # Update metrics
        self._metrics.files_processed += 1
        
        return result
    
    def close(self) -> None:
        """Close the underlying OCR engine and release resources."""
        if self._engine is not None:
            self._logger.debug(
                "SelfMonitoringOCREngine[%s] closing engine...",
                self._engine_id,
            )
            try:
                self._engine.close()
            except Exception as e:
                self._logger.warning(
                    "SelfMonitoringOCREngine[%s] error during close: %s",
                    self._engine_id,
                    e,
                )
            finally:
                # Force deallocation
                del self._engine
                self._engine = None
                
            self._logger.debug(
                "SelfMonitoringOCREngine[%s] closed",
                self._engine_id,
            )
    
    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics.
        
        Returns:
            EngineMetrics with current statistics
        """
        return EngineMetrics(
            files_processed=self._metrics.files_processed,
            restarts_performed=self._metrics.restarts_performed,
            memory_baseline_mb=self._metrics.memory_baseline_mb,
            last_memory_check_mb=self._metrics.last_memory_check_mb,
        )
    
    def __enter__(self) -> "SelfMonitoringOCREngine":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

