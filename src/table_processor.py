"""Table cell extraction and OCR processing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from parallel_ocr_worker import (
    CellOCRResult,
    CellOCRTask,
    create_engine_params,
)
from table_detector import TableGrid


@dataclass(frozen=True)
class TableCell:
    """Extracted table cell with OCR results."""

    row_idx: int  # 0-based row index
    col_idx: int  # 0-based column index
    x_start: int  # Pixel coordinates
    y_start: int
    x_end: int
    y_end: int
    text: str  # OCR text
    confidence: float  # OCR confidence
    field_type: Optional[str] = None  # Inferred field type
    field_name: Optional[str] = None  # Column name/label


class TableProcessor:
    """Extract and process individual table cells with OCR."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        ocr_engine: Any,  # OCREngine instance
        engine_pool: Optional[Any] = None,  # OCREnginePool instance
    ):
        """Initialize table processor.

        Args:
            settings: Application settings
            logger: Logger instance
            ocr_engine: OCREngine instance for cell-level OCR
            engine_pool: Optional OCR engine pool for parallel processing
        """
        self._settings = settings
        self._logger = logger
        self._ocr_engine = ocr_engine
        self._engine_pool = engine_pool

    def extract_cells(
        self,
        image: np.ndarray,
        grid: TableGrid,
        column_mapping: Optional[Dict[int, str]] = None,
    ) -> List[TableCell]:
        """Extract cells from table region using detected grid.

        Args:
            image: Original (not binarized) table region
            grid: Detected table structure
            column_mapping: Dict mapping col_idx → field_name
                           E.g. {0: 'row_number', 1: 'parameter', ...}

        Returns:
            List of extracted TableCell objects
        """
        if image is None or image.size == 0:
            self._logger.warning("Table processing: input image is None or empty")
            return []

        # Choose parallel or sequential processing
        if self._should_use_parallel_processing(grid):
            return self._extract_cells_parallel(image, grid, column_mapping)
        else:
            return self._extract_cells_sequential(image, grid, column_mapping)

    def _should_use_parallel_processing(self, grid: TableGrid) -> bool:
        """Check if parallel processing should be used.

        Args:
            grid: Table grid structure

        Returns:
            True if parallel processing should be used
        """
        if not self._settings.enable_parallel_processing:
            return False
        # Parallel processing requires engine pool
        if self._engine_pool is None:
            return False
        cell_count = grid.num_rows * grid.num_cols
        return cell_count >= self._settings.parallel_min_cells_for_parallelization

    def _extract_cells_parallel(
        self,
        image: np.ndarray,
        grid: TableGrid,
        column_mapping: Optional[Dict[int, str]] = None,
    ) -> List[TableCell]:
        """Extract cells using parallel processing with engine pool.

        Uses ThreadPoolExecutor to process cells in parallel, acquiring engines
        from the pool as needed. This avoids creating new OCR engines in workers.

        Args:
            image: Original (not binarized) table region
            grid: Detected table structure
            column_mapping: Dict mapping col_idx → field_name

        Returns:
            List of extracted TableCell objects
        """
        from concurrent.futures import ThreadPoolExecutor
        
        start_time = time.perf_counter()
        height, width = image.shape[:2]

        # Build row and column boundaries
        row_boundaries = self._build_row_boundaries(grid.rows, height)
        col_boundaries = self._build_col_boundaries(grid.cols, width)

        total_cells = (len(row_boundaries) - 1) * (len(col_boundaries) - 1)
        self._logger.info(
            "Extracting %d cells using engine pool (%d rows × %d columns)",
            total_cells,
            len(row_boundaries) - 1,
            len(col_boundaries) - 1,
        )

        # Prepare all cell data (no need for pickling with threads)
        cell_tasks = []
        empty_cells: List[TableCell] = []  # Store empty cells immediately
        
        for row_idx in range(len(row_boundaries) - 1):
            for col_idx in range(len(col_boundaries) - 1):
                # Apply margin
                margin = self._settings.table_cell_margin
                y_start = row_boundaries[row_idx]
                y_end = row_boundaries[row_idx + 1]
                x_start = col_boundaries[col_idx]
                x_end = col_boundaries[col_idx + 1]

                y1 = max(0, y_start + margin)
                y2 = min(image.shape[0], y_end - margin)
                x1 = max(0, x_start + margin)
                x2 = min(image.shape[1], x_end - margin)

                # Check minimum size
                cell_width = x2 - x1
                cell_height = y2 - y1
                if (
                    cell_width < self._settings.table_cell_min_width
                    or cell_height < self._settings.table_cell_min_height
                ):
                    continue

                # Extract cell ROI
                cell_image = image[y1:y2, x1:x2].copy()

                # Preprocess cell if enabled
                if self._settings.table_cell_preprocess:
                    cell_image = self._preprocess_cell(cell_image)

                # Get field name from mapping
                field_name = column_mapping.get(col_idx) if column_mapping else None

                # Check if cell is empty before adding to OCR tasks
                if self._is_cell_empty(cell_image):
                    # Create empty cell result immediately, skip OCR
                    empty_cells.append(
                        TableCell(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            x_start=x1,
                            y_start=y1,
                            x_end=x2,
                            y_end=y2,
                            text="",
                            confidence=1.0,  # High confidence for empty detection
                            field_name=field_name,
                        )
                    )
                    continue

                cell_tasks.append({
                    'cell_image': cell_image,
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'x_start': x1,
                    'y_start': y1,
                    'x_end': x2,
                    'y_end': y2,
                    'field_name': field_name,
                })

        if not cell_tasks:
            self._logger.warning("No valid cells to process")
            return []

        # Process cells in parallel using thread pool + engine pool
        # Start with empty cells that were detected
        cells: List[TableCell] = empty_cells.copy()
        
        try:
            # Use min of pool size and parallel_cells_workers
            num_workers = min(
                self._engine_pool._pool_size,
                self._settings.parallel_cells_workers
            )
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._process_cell_with_pool, task): task
                    for task in cell_tasks
                }

                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            cells.append(result)
                    except Exception as e:
                        self._logger.warning(
                            "Cell OCR failed for cell [%d,%d]: %s",
                            task['row_idx'],
                            task['col_idx'],
                            e,
                        )

        except Exception as e:
            self._logger.error(
                "Parallel cell processing failed, falling back to sequential: %s", e
            )
            # Fallback to sequential
            return self._extract_cells_sequential(image, grid, column_mapping)

        duration = time.perf_counter() - start_time
        avg_confidence = (
            sum(c.confidence for c in cells) / len(cells) if cells else 0.0
        )
        
        # Calculate empty cell statistics
        empty_count = len(empty_cells)
        total_processed = len(cells)
        empty_percentage = (empty_count / total_processed * 100) if total_processed > 0 else 0.0
        
        # Estimate time saved (assuming ~100ms per empty cell OCR)
        estimated_ocr_time_per_cell = 0.1  # seconds
        estimated_time_saved = empty_count * estimated_ocr_time_per_cell

        if empty_count > 0:
            self._logger.info(
                "Extracted %d cells using pool (%.3fs, %d workers, avg confidence: %.3f, "
                "%d empty skipped %.1f%%, saved ~%.1fs)",
                total_processed,
                duration,
                num_workers,
                avg_confidence,
                empty_count,
                empty_percentage,
                estimated_time_saved,
            )
        else:
            self._logger.info(
                "Extracted %d cells using pool (%.3fs, %d workers, avg confidence: %.3f)",
                total_processed,
                duration,
                num_workers,
                avg_confidence,
            )

        return cells
    
    def _process_cell_with_pool(self, task: Dict[str, Any]) -> Optional[TableCell]:
        """Process single cell using engine from pool.
        
        Args:
            task: Dictionary with cell data and metadata
            
        Returns:
            TableCell or None if processing failed
        """
        try:
            # Double-check if cell is empty (defensive check)
            # This should rarely happen since we check before adding to tasks
            if self._is_cell_empty(task['cell_image']):
                return TableCell(
                    row_idx=task['row_idx'],
                    col_idx=task['col_idx'],
                    x_start=task['x_start'],
                    y_start=task['y_start'],
                    x_end=task['x_end'],
                    y_end=task['y_end'],
                    text="",
                    confidence=1.0,
                    field_name=task['field_name'],
                )
            
            # Acquire engine from pool (blocks if none available)
            with self._engine_pool.acquire() as ocr_engine:
                # Use the engine's internal OCR method
                text, confidence = self._ocr_cell_with_engine(
                    task['cell_image'],
                    ocr_engine
                )
                
                return TableCell(
                    row_idx=task['row_idx'],
                    col_idx=task['col_idx'],
                    x_start=task['x_start'],
                    y_start=task['y_start'],
                    x_end=task['x_end'],
                    y_end=task['y_end'],
                    text=text,
                    confidence=confidence,
                    field_name=task['field_name'],
                )
        except Exception as e:
            self._logger.warning(
                "Failed to process cell [%d,%d] with pool: %s",
                task['row_idx'],
                task['col_idx'],
                e,
            )
            return None
    
    def _ocr_cell_with_engine(
        self, 
        cell_image: np.ndarray, 
        ocr_engine: Any
    ) -> Tuple[str, float]:
        """Run OCR on cell using provided engine.
        
        Args:
            cell_image: Preprocessed cell image
            ocr_engine: SelfMonitoringOCREngine instance from pool
            
        Returns:
            (text, confidence) tuple
        """
        try:
            # Access the underlying OCREngine
            engine = ocr_engine._engine if hasattr(ocr_engine, '_engine') else ocr_engine
            
            # Resize if needed
            prepared_image, scale = engine._resize_for_ocr(cell_image)
            
            # Run OCR
            ocr_results = engine._run_paddle_ocr(prepared_image)
            
            # Process results
            detections = engine._process_ocr_results(
                ocr_results, scale=scale if scale > 0 else 1.0
            )
            
            if not detections:
                return "", 0.0
            
            # Combine all detections
            texts = [d.text for d in detections if d.text.strip()]
            confidences = [d.confidence for d in detections if d.text.strip()]
            
            if not texts:
                return "", 0.0
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return combined_text.strip(), avg_confidence
            
        except Exception as e:
            self._logger.debug("OCR failed for cell: %s", e)
            return "", 0.0

    def _extract_cells_sequential(
        self,
        image: np.ndarray,
        grid: TableGrid,
        column_mapping: Optional[Dict[int, str]] = None,
    ) -> List[TableCell]:
        """Extract cells sequentially (original implementation).

        Args:
            image: Original (not binarized) table region
            grid: Detected table structure
            column_mapping: Dict mapping col_idx → field_name

        Returns:
            List of extracted TableCell objects
        """
        start_time = time.perf_counter()
        height, width = image.shape[:2]
        cells: List[TableCell] = []

        # Build row and column boundaries
        row_boundaries = self._build_row_boundaries(grid.rows, height)
        col_boundaries = self._build_col_boundaries(grid.cols, width)

        self._logger.info(
            "Extracting cells sequentially from %d rows × %d columns",
            len(row_boundaries) - 1,
            len(col_boundaries) - 1,
        )

        # Extract each cell
        empty_count = 0
        for row_idx in range(len(row_boundaries) - 1):
            for col_idx in range(len(col_boundaries) - 1):
                cell = self._extract_single_cell(
                    image,
                    row_idx,
                    col_idx,
                    row_boundaries[row_idx],
                    row_boundaries[row_idx + 1],
                    col_boundaries[col_idx],
                    col_boundaries[col_idx + 1],
                    column_mapping,
                )
                if cell:
                    cells.append(cell)
                    # Count empty cells (text is empty and confidence is 1.0)
                    if cell.text == "" and cell.confidence == 1.0:
                        empty_count += 1

        duration = time.perf_counter() - start_time
        avg_confidence = (
            sum(c.confidence for c in cells) / len(cells) if cells else 0.0
        )
        
        # Calculate empty cell statistics
        total_processed = len(cells)
        empty_percentage = (empty_count / total_processed * 100) if total_processed > 0 else 0.0
        
        # Estimate time saved (assuming ~100ms per empty cell OCR)
        estimated_ocr_time_per_cell = 0.1  # seconds
        estimated_time_saved = empty_count * estimated_ocr_time_per_cell

        if empty_count > 0:
            self._logger.info(
                "Extracted %d cells sequentially (%.3fs, avg confidence: %.3f, "
                "%d empty skipped %.1f%%, saved ~%.1fs)",
                total_processed,
                duration,
                avg_confidence,
                empty_count,
                empty_percentage,
                estimated_time_saved,
            )
        else:
            self._logger.info(
                "Extracted %d cells sequentially (%.3fs, avg confidence: %.3f)",
                total_processed,
                duration,
                avg_confidence,
            )

        return cells

    def _build_row_boundaries(self, row_lines: List[int], image_height: int) -> List[int]:
        """Build row boundaries from detected lines.

        Args:
            row_lines: Y-coordinates of horizontal lines
            image_height: Total image height

        Returns:
            List of row boundaries [0, y1, y2, ..., height]
        """
        boundaries = [0]
        for i in range(len(row_lines) - 1):
            # Boundary is midpoint between adjacent lines
            boundary = (row_lines[i] + row_lines[i + 1]) // 2
            boundaries.append(boundary)
        boundaries.append(image_height)
        return boundaries

    def _build_col_boundaries(self, col_lines: List[int], image_width: int) -> List[int]:
        """Build column boundaries from detected lines.

        Args:
            col_lines: X-coordinates of vertical lines
            image_width: Total image width

        Returns:
            List of column boundaries [0, x1, x2, ..., width]
        """
        boundaries = [0]
        for i in range(len(col_lines) - 1):
            # Boundary is midpoint between adjacent lines
            boundary = (col_lines[i] + col_lines[i + 1]) // 2
            boundaries.append(boundary)
        boundaries.append(image_width)
        return boundaries

    def _extract_single_cell(
        self,
        image: np.ndarray,
        row_idx: int,
        col_idx: int,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        column_mapping: Optional[Dict[int, str]],
    ) -> Optional[TableCell]:
        """Extract and OCR a single cell.

        Args:
            image: Full table region image
            row_idx: Row index (0-based)
            col_idx: Column index (0-based)
            y_start, y_end: Row boundaries
            x_start, x_end: Column boundaries
            column_mapping: Column index to field name mapping

        Returns:
            TableCell or None if extraction failed
        """
        # Apply margin
        margin = self._settings.table_cell_margin
        y1 = max(0, y_start + margin)
        y2 = min(image.shape[0], y_end - margin)
        x1 = max(0, x_start + margin)
        x2 = min(image.shape[1], x_end - margin)

        # Check minimum size
        cell_width = x2 - x1
        cell_height = y2 - y1
        if (
            cell_width < self._settings.table_cell_min_width
            or cell_height < self._settings.table_cell_min_height
        ):
            self._logger.debug(
                "Skipping cell [%d,%d]: too small (%dx%d)",
                row_idx,
                col_idx,
                cell_width,
                cell_height,
            )
            return None

        # Extract cell ROI
        cell_image = image[y1:y2, x1:x2].copy()

        # Preprocess cell if enabled
        if self._settings.table_cell_preprocess:
            cell_image = self._preprocess_cell(cell_image)

        # Check if cell is empty before OCR
        if self._is_cell_empty(cell_image):
            # Return empty cell result without OCR processing
            field_name = column_mapping.get(col_idx) if column_mapping else None
            return TableCell(
                row_idx=row_idx,
                col_idx=col_idx,
                x_start=x1,
                y_start=y1,
                x_end=x2,
                y_end=y2,
                text="",
                confidence=1.0,  # High confidence for empty detection
                field_name=field_name,
            )

        # Run OCR on cell
        text, confidence = self._ocr_cell(cell_image)

        # Get field name from mapping
        field_name = column_mapping.get(col_idx) if column_mapping else None

        return TableCell(
            row_idx=row_idx,
            col_idx=col_idx,
            x_start=x1,
            y_start=y1,
            x_end=x2,
            y_end=y2,
            text=text,
            confidence=confidence,
            field_name=field_name,
        )

    def _is_cell_empty(self, cell_image: np.ndarray) -> bool:
        """Fast check if cell image is empty using CV techniques.
        
        Uses multiple criteria to determine if cell is empty:
        1. Pixel intensity variance (low variance = uniform/empty)
        2. White pixel ratio (high white ratio = empty)
        3. Edge density (few edges = empty)
        4. Connected components count (few components = empty)
        
        Args:
            cell_image: Cell ROI image (can be BGR or grayscale)
            
        Returns:
            True if cell appears empty, False otherwise
        """
        if not self._settings.table_enable_empty_cell_detection:
            return False
        
        try:
            # Convert to grayscale if needed
            if cell_image.ndim == 3:
                gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_image.copy()
            
            height, width = gray.shape
            total_pixels = height * width
            
            if total_pixels == 0:
                return True
            
            # Check 1: Pixel intensity variance
            # Empty cells have low variance (uniform color)
            mean, std_dev = cv2.meanStdDev(gray)
            variance = std_dev[0][0] ** 2
            if variance > self._settings.table_empty_variance_threshold:
                return False
            
            # Check 2: White pixel ratio
            # Empty cells are mostly white/blank
            # Consider pixels > 240 as white (out of 255)
            white_pixels = np.sum(gray > 240)
            white_ratio = white_pixels / total_pixels
            if white_ratio < self._settings.table_empty_white_pixel_threshold:
                return False
            
            # Check 3: Edge density
            # Empty cells have very few edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels
            if edge_density > self._settings.table_empty_edge_density_threshold:
                return False
            
            # Check 4: Connected components count
            # Empty cells have minimal distinct regions
            # Use binary threshold to find components
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, _ = cv2.connectedComponents(binary)
            # Subtract 1 because background is counted as a component
            num_components = num_labels - 1
            if num_components > self._settings.table_empty_component_threshold:
                return False
            
            # All checks passed - cell is empty
            return True
            
        except Exception as e:
            # If detection fails, assume cell is not empty (process it)
            self._logger.debug("Empty cell detection failed, processing cell: %s", e)
            return False

    def _preprocess_cell(self, cell_image: np.ndarray) -> np.ndarray:
        """Preprocess single cell before OCR.

        Options:
        - Slight contrast boost
        - Denoise if too noisy
        - Small morphological cleanup

        Args:
            cell_image: Cell ROI

        Returns:
            Preprocessed cell image
        """
        # Convert to grayscale if needed
        if cell_image.ndim == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Apply slight contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(gray)

        # Optional: denoise if very noisy
        # (Skip for now to keep it simple)

        # Convert back to BGR for consistency with OCR engine
        if cell_image.ndim == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced

    def _ocr_cell(self, cell_image: np.ndarray) -> Tuple[str, float]:
        """Run OCR on single cell.

        Args:
            cell_image: Preprocessed cell ROI

        Returns:
            (text, confidence) tuple
        """
        try:
            # Use OCR engine's internal method to process single image
            # We need to resize if too large
            prepared_image, scale = self._ocr_engine._resize_for_ocr(cell_image)

            # Run OCR
            ocr_results = self._ocr_engine._run_paddle_ocr(prepared_image)

            # Process results
            detections = self._ocr_engine._process_ocr_results(
                ocr_results, scale=scale if scale > 0 else 1.0
            )

            if not detections:
                return "", 0.0

            # Combine all detections in cell (usually just one)
            texts = [d.text for d in detections if d.text.strip()]
            confidences = [d.confidence for d in detections if d.text.strip()]

            if not texts:
                return "", 0.0

            # Combine texts (space-separated)
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return combined_text.strip(), avg_confidence

        except Exception as e:
            self._logger.warning(
                "OCR failed for cell: %s", e, exc_info=self._logger.level == logging.DEBUG
            )
            return "", 0.0

    def build_table_rows(
        self,
        cells: List[TableCell],
        column_mapping: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """Convert flat list of cells into row-structured format.

        Each row becomes a dictionary:
        {
            'row_index': 0,
            'field_name': {'value': '...', 'confidence': ...},
            ...
        }

        Args:
            cells: Extracted cells
            column_mapping: Col index → field name

        Returns:
            List of row dictionaries
        """
        # Group cells by row
        rows_dict: Dict[int, List[TableCell]] = {}
        for cell in cells:
            if cell.row_idx not in rows_dict:
                rows_dict[cell.row_idx] = []
            rows_dict[cell.row_idx].append(cell)

        # Build row structures
        rows = []
        for row_idx in sorted(rows_dict.keys()):
            row_cells = rows_dict[row_idx]
            row_data: Dict[str, Any] = {"row_index": row_idx}

            for cell in row_cells:
                field_name = cell.field_name or f"col_{cell.col_idx}"
                row_data[field_name] = {
                    "value": cell.text,
                    "confidence": cell.confidence,
                    "x_start": cell.x_start,
                    "y_start": cell.y_start,
                    "x_end": cell.x_end,
                    "y_end": cell.y_end,
                }

            rows.append(row_data)

        return rows


