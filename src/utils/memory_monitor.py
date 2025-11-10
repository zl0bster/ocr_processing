"""Memory monitoring utility for tracking RAM usage during OCR processing."""
from __future__ import annotations

import logging

import psutil


class MemoryMonitor:
    """Monitor process memory usage with logging integration."""
    
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._process = psutil.Process()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._process.memory_info().rss / 1024 / 1024
    
    def log_memory(self, label: str = "", level: str = "DEBUG") -> float:
        """Log current memory usage with optional label."""
        mem_mb = self.get_memory_mb()
        log_method = getattr(self._logger, level.lower())
        log_method(f"Memory usage{' ' + label if label else ''}: {mem_mb:.1f} MB")
        return mem_mb
    
    def log_memory_delta(self, start_mb: float, label: str = "") -> float:
        """Log memory delta since start measurement."""
        current_mb = self.get_memory_mb()
        delta_mb = current_mb - start_mb
        self._logger.info(
            f"Memory delta{' ' + label if label else ''}: {delta_mb:+.1f} MB "
            f"(from {start_mb:.1f} to {current_mb:.1f} MB)"
        )
        return current_mb

